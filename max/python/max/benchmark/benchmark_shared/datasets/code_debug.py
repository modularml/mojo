# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import random
from collections.abc import Sequence

import msgspec
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ._hf_download import hf_hub_download_with_retry
from .distribution import BaseDistribution, DistributionParameter
from .huggingface import HuggingFaceBenchmarkDataset
from .types import (
    ChatSamples,
    ChatSession,
    RequestSamples,
    SampledRequest,
    build_chat_message,
)

CODE_DEBUG_TEMPLATE = """\
There is ONLY ONE function in the large project that is deliberately made to \
include an obvious error. Please find the function that contains the most \
obvious errors. I will give you four options to narrow your scope. You can \
inspect the options and think. Eventually, tell me the answer using one \
single letter (A, B, C, or D).

{context}

Which function has deliberate error?
A. {OPTION_A}
B. {OPTION_B}
C. {OPTION_C}
D. {OPTION_D}

You should first find the functions in the options. Repeat their content, \
inspect through code, and at last give me your answer for the function that \
has the deliberate and obvious error in A, B, C, or D.\
"""


class CodeDebugLine(msgspec.Struct):
    id: int
    context: str
    input: str
    answer: Sequence[str]
    options: Sequence[str]


class CodeDebugBenchmarkDataset(HuggingFaceBenchmarkDataset):
    def fetch(self) -> None:
        self.dataset_path = hf_hub_download_with_retry(
            repo_id="xinrongzhang2022/InfiniteBench",
            filename="code_debug.jsonl",
            repo_type="dataset",
        )

    def gen_twoturn_longcontext_requests(
        self,
        num_chat_sessions: int,
        delay_between_chat_turns: DistributionParameter | None,
        tokenizer: PreTrainedTokenizerBase,
    ) -> ChatSamples:
        # Expand code_debug dataset to 2-turn chats with a pre-defined followup question
        delay_dist = BaseDistribution.from_distribution_parameter(
            delay_between_chat_turns
        )
        DUMMY_OUTPUT = "A"
        CODE_DEBUG_FOLLOWUP_QUESTION = "Explain your reasoning?"
        request_samples = self.sample_requests(
            num_requests=num_chat_sessions,
            tokenizer=tokenizer,
        )

        sessions: list[ChatSession] = []
        for session_id, input_request in enumerate(request_samples.requests):
            assert isinstance(input_request.prompt_formatted, str)
            messages = [
                build_chat_message(
                    "user", input_request.prompt_formatted, tokenizer
                ),
                # TODO, put correct answers for verification
                # NOTE: Specific single letter answer (2-token)
                build_chat_message(
                    "assistant",
                    DUMMY_OUTPUT,
                    tokenizer,
                    2,
                    delay_until_next_message=max(delay_dist.sample_value(), 0)
                    if delay_dist
                    else None,
                ),
                build_chat_message(
                    "user", CODE_DEBUG_FOLLOWUP_QUESTION, tokenizer
                ),
                build_chat_message(
                    "assistant",
                    DUMMY_OUTPUT,
                    tokenizer,
                    delay_until_next_message=max(delay_dist.sample_value(), 0)
                    if delay_dist
                    else None,
                ),
            ]
            sessions.append(ChatSession(session_id, messages))

        return ChatSamples(chat_sessions=sessions)

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs,
    ) -> RequestSamples:
        """
        The Long-Context dataset workload is based on InfiniteBench Code.debug
        """
        assert self.dataset_path is not None, (
            "dataset_path must be provided for CodeDebugBenchmarkDataset"
        )
        with open(self.dataset_path) as jsonl_file:
            decoded_lines = [
                msgspec.json.decode(json_line, type=CodeDebugLine)
                for json_line in jsonl_file
            ]

        # format context/options/answer -> template of (prompt, completion)
        dataset = [
            (
                self.format_code_debug_context(data),
                self.get_code_debug_answer(data),
            )
            for data in decoded_lines
        ]
        # Filter out the task with LICENSE
        dataset = [data for data in dataset if "LICENSE" not in data[0]]

        # Shuffle the dataset.
        if shuffle:
            if output_lengths is not None:
                raise NotImplementedError(
                    "TODO: Add support for shuffling + pinned output lengths"
                )
            random.shuffle(dataset)

        # Filter out sequences that are too long or too short
        filtered_dataset: list[SampledRequest] = []
        model_max_length = tokenizer.model_max_length
        for i in range(len(dataset)):
            if len(filtered_dataset) == num_requests:
                break

            # Tokenize the prompts and completions.
            prompt = dataset[i][0]
            prompt_token_ids = tokenizer.encode(prompt)
            completion = dataset[i][1]
            completion_token_ids = tokenizer.encode(completion)
            prompt_len = len(prompt_token_ids)
            output_len = (
                len(completion_token_ids)
                if output_lengths is None
                else output_lengths[i]
            )
            assert output_len is not None, "Unexpected null output length"
            if (
                prompt_len > model_max_length
                or prompt_len + output_len > model_max_length
            ):
                # Prune too long sequences.
                print(
                    f"Skip too long sequences ({prompt_len} >"
                    f" {model_max_length})..."
                )
                continue
            filtered_dataset.append(
                SampledRequest(
                    prompt_formatted=prompt,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    encoded_images=[],
                    ignore_eos=(output_len is not None),
                )
            )

        if __debug__:
            from statistics import mean

            list_prompt_len = [data.prompt_len for data in filtered_dataset]
            print(
                f"INFO: Sampled {len(filtered_dataset)} Long-Context Requests: "
                f"Input Tokens(Average: {mean(list_prompt_len)}, "
                f"Min: {min(list_prompt_len)}, Max: {max(list_prompt_len)})"
            )

        return RequestSamples(requests=filtered_dataset)

    @staticmethod
    def format_code_debug_context(request_features: CodeDebugLine) -> str:
        prompt = CODE_DEBUG_TEMPLATE.format(
            context=request_features.context,
            OPTION_A=request_features.options[0],
            OPTION_B=request_features.options[1],
            OPTION_C=request_features.options[2],
            OPTION_D=request_features.options[3],
        )
        return prompt

    @staticmethod
    def get_code_debug_answer(request_features: CodeDebugLine) -> str:
        if len(request_features.answer) != 1:
            raise ValueError("More than 1 answers")
        OPTIONS = "ABCD"
        return OPTIONS[
            request_features.options.index(request_features.answer[0])
        ]
