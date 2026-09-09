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

import json
import random
from collections.abc import Sequence

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ._hf_download import hf_hub_download_with_retry
from .huggingface import HuggingFaceBenchmarkDataset
from .types import RequestSamples, SampledRequest


class ShareGPTBenchmarkDataset(HuggingFaceBenchmarkDataset):
    def fetch(self) -> None:
        self.dataset_path = hf_hub_download_with_retry(
            repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
            filename="ShareGPT_V3_unfiltered_cleaned_split.json",
            repo_type="dataset",
        )

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs,
    ) -> RequestSamples:
        """Sample requests from ShareGPT dataset."""
        assert self.dataset_path is not None, (
            "dataset_path must be provided for ShareGPTBenchmarkDataset"
        )
        # Load the dataset.
        with open(self.dataset_path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [
            (
                data["conversations"][0]["value"],
                data["conversations"][1]["value"],
            )
            for data in dataset
        ]

        # Shuffle the dataset.
        if shuffle:
            if output_lengths is not None:
                raise NotImplementedError(
                    "TODO: Add support for shuffling + pinned output lengths"
                )
            random.shuffle(dataset)

        # Filter out sequences that are too long or too short
        filtered_dataset: list[SampledRequest] = []
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
                else output_lengths[len(filtered_dataset)]
            )
            assert output_len is not None, "Unexpected null output length"
            if prompt_len < 4:
                # Prune too short sequences.
                continue
            if prompt_len > 1024 or prompt_len + output_len > 2048:
                # Prune too long sequences.
                continue
            # If we're given explicit output lengths, then run with whatever
            # we're given. Otherwise, filter requests with super short responses.
            if output_lengths is None and output_len < 4:
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

        return RequestSamples(requests=filtered_dataset)
