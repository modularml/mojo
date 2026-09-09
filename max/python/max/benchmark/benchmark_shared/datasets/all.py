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

import logging
import os

import numpy as np
import yaml
from max.benchmark.benchmark_shared.config import (
    PIXEL_GENERATION_TASKS,
    BenchmarkTask,
    ServingBenchmarkConfig,
)
from max.benchmark.benchmark_shared.datasets._tokenizer_pool import (
    TokenizerPool,
)
from max.benchmark.benchmark_shared.datasets.agentic_code import (
    AgenticCodeBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets.artificial_analysis import (
    ArtificialAnalysisBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets.arxiv_summarization import (
    ArxivSummarizationBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets.axolotl import (
    AxolotlBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets.batch_job import (
    BatchJobBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets.chat_judge import (
    ChatJudgeBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets.code_debug import (
    CodeDebugBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets.instruct_coder import (
    InstructCoderBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets.interface import BenchmarkDataset
from max.benchmark.benchmark_shared.datasets.multiturn_distribution_fit import (
    resolve_constant_delay_ms,
)
from max.benchmark.benchmark_shared.datasets.nemotron_opencode import (
    NemotronOpenCodeBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets.obfuscated_conversations import (
    ObfuscatedConversationsBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets.pixel_image_edit import (
    LocalImageBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets.pixel_synthetic import (
    SyntheticPixelBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets.random import (
    RandomBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets.sharegpt import (
    ShareGPTBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets.sonnet import (
    SonnetBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets.types import Samples
from max.benchmark.benchmark_shared.datasets.vision_arena import (
    VisionArenaBenchmarkDataset,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _inflated_chat_session_count(
    args: ServingBenchmarkConfig, base_session_count: int
) -> int:
    """Inflate the dataset request to ``base + factor * max_warmup_count``
    so the length-biased warmup pick has cap headroom (sweeps inflate by
    the largest requested concurrency)."""
    if not args.warmup_to_steady_state or args.warmup_oversample_factor <= 0:
        return base_session_count
    max_warmup = 0
    if args.max_concurrent_conversations is not None:
        max_warmup = max(max_warmup, args.max_concurrent_conversations)
    if args.max_concurrency:
        try:
            mcs_ints = [m for m in args.max_concurrency if m is not None]
            if mcs_ints:
                max_warmup = max(max_warmup, max(mcs_ints))
        except Exception:
            pass
    if max_warmup <= 0:
        return base_session_count
    return base_session_count + args.warmup_oversample_factor * max_warmup


def sample_requests(
    *,
    args: ServingBenchmarkConfig,
    benchmark_task: BenchmarkTask,
    tokenizer: PreTrainedTokenizerBase | None,
    chat: bool,
) -> Samples:
    """Sample requests from the dataset for use in a benchmark.

    Args:
        args: Command-line arguments passed to benchmark_serving.py
        benchmark_task: What type of model is being benchmarked.
        tokenizer: Tokenizer for measuring token lengths of samples.
        chat: Whether the model is being benchmarked with a chat endpoint.

    Returns:
        The sampled requests.
    """

    if benchmark_task == "text-generation":
        if tokenizer is None:
            raise ValueError(
                "Tokenizer is required for text-generation benchmarks"
            )

        benchmark_dataset = BenchmarkDataset.from_flags(
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
        )

        if (
            args.num_chat_sessions
            and not benchmark_dataset.has_multiturn_chat_support
        ):
            raise ValueError(
                f"Multiturn chat is not supported for dataset {benchmark_dataset}"
            )

        logger.info("sampling requests")

        # Build output_lengths array
        if args.num_prompts is not None:
            num_requests = args.num_prompts
        elif args.num_chat_sessions is not None:
            num_requests = args.num_chat_sessions
        else:
            raise ValueError(
                "Please specify either '--num-prompts' or '--num-chat-sessions'."
            )

        # NOTE: args.output_lengths is a path to a YAML file, while output_lengths
        # is a list of ints.
        if args.output_lengths is None:
            output_lengths = None
        elif isinstance(args.output_lengths, str) and os.path.exists(
            args.output_lengths
        ):
            with open(args.output_lengths) as f:
                output_lengths = yaml.safe_load(f)["output_lengths"]
        else:
            output_lengths = [int(args.output_lengths)] * num_requests

        # We should not be using / accessing args.output_lengths from here on out.

        # This entire isinstance tree is tech debt.  BenchmarkDataset has been
        # designed to have methods that can be overridden by subclasses.  We do
        # this, and then completely throw out the benefits of that design by
        # specializing on each dataset type here.  This is badly in need of
        # refactoring.  TODO(MXTOOLS-154)

        if isinstance(benchmark_dataset, CodeDebugBenchmarkDataset):
            # code_debug is a long-context dataset based on InfiniteBench
            if args.num_chat_sessions:
                if args.fit_distributions:
                    raise ValueError(
                        "--fit-distributions is not supported for --dataset-name "
                        "code-debug with --num-chat-sessions. Use random, "
                        "instruct-coder, or agentic-code for distribution-shaped "
                        "multiturn workloads, or omit --fit-distributions to keep "
                        "code-debug's fixed two-turn template."
                    )
                if output_lengths is not None:
                    raise NotImplementedError(
                        "TODO: Add support for fixed output lengths with multi-turn"
                        " code-debug"
                    )
                return benchmark_dataset.gen_twoturn_longcontext_requests(
                    num_chat_sessions=args.num_chat_sessions,
                    delay_between_chat_turns=args.delay_between_chat_turns,
                    tokenizer=tokenizer,
                )
            else:
                assert args.num_prompts is not None
                return benchmark_dataset.sample_requests(
                    num_requests=args.num_prompts,
                    tokenizer=tokenizer,
                    output_lengths=output_lengths,
                    shuffle=(
                        output_lengths is None
                        and not args.record_output_lengths
                    ),
                )

        elif isinstance(benchmark_dataset, ShareGPTBenchmarkDataset):
            assert args.num_prompts is not None
            return benchmark_dataset.sample_requests(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                output_lengths=output_lengths,
                shuffle=(
                    output_lengths is None and not args.record_output_lengths
                ),
            )

        elif isinstance(benchmark_dataset, SonnetBenchmarkDataset):
            # For sonnet, formatting depends on the endpoint
            apply_chat_template = chat
            # Sample sonnet requests with common parameters
            assert args.num_prompts is not None
            return benchmark_dataset.sample_requests(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                output_lengths=output_lengths,
                input_len=args.sonnet_input_len,
                prefix_len=args.sonnet_prefix_len,
                apply_chat_template=apply_chat_template,
            )

        elif isinstance(benchmark_dataset, VisionArenaBenchmarkDataset):
            assert args.num_prompts is not None
            return benchmark_dataset.sample_requests(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                output_lengths=output_lengths,
            )
        elif isinstance(benchmark_dataset, ArtificialAnalysisBenchmarkDataset):
            assert args.num_prompts is not None
            return benchmark_dataset.sample_requests(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                output_lengths=output_lengths,
                shuffle=(
                    output_lengths is None and not args.record_output_lengths
                ),
                input_len=args.random_input_len,
                output_len=args.random_output_len,
                seed=args.seed,
            )
        elif isinstance(benchmark_dataset, ArxivSummarizationBenchmarkDataset):
            if output_lengths:
                raise ValueError(
                    "Arxiv summarization dataset does not support --output-lengths."
                    " Please use --max-output-len"
                )
            assert args.num_prompts is not None
            return benchmark_dataset.sample_requests(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                shuffle=not args.record_output_lengths,
                input_len=args.arxiv_summarization_input_len,
                max_output_len=args.max_output_len,
            )
        elif isinstance(benchmark_dataset, RandomBenchmarkDataset):
            # `with` ensures the spawn pool joins cleanly when the random
            # dataset's encode/decode fan-out is done; workers spawn lazily
            # on first use, so this is free when none of the methods below
            # actually exercise the pool.
            with TokenizerPool(tokenizer) as pool:
                if args.num_chat_sessions:
                    return benchmark_dataset.gen_multiturn_random_requests(
                        input_len=args.random_input_len,
                        output_len=args.random_output_len,
                        num_chat_sessions=_inflated_chat_session_count(
                            args, args.num_chat_sessions
                        ),
                        num_turns=args.random_num_turns,
                        delay_between_chat_turns=args.delay_between_chat_turns,
                        pool=pool,
                        sys_prompt_ratio=args.random_sys_prompt_ratio,
                        max_num_unique_sys_prompt=args.random_max_num_unique_sys_prompt,
                    )
                else:
                    assert args.num_prompts is not None
                    return benchmark_dataset.sample_requests(
                        num_requests=args.num_prompts,
                        tokenizer=tokenizer,
                        pool=pool,
                        input_len=args.random_input_len,
                        output_len=args.random_output_len,
                        sys_prompt_ratio=args.random_sys_prompt_ratio,
                        max_num_unique_sys_prompt=args.random_max_num_unique_sys_prompt,
                        image_size=args.random_image_size,
                        image_count=args.random_image_count,
                    )
        elif isinstance(benchmark_dataset, AxolotlBenchmarkDataset):
            assert args.num_prompts is not None
            return benchmark_dataset.sample_requests(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                output_lengths=output_lengths,
                shuffle=(
                    output_lengths is None and not args.record_output_lengths
                ),
            )
        elif isinstance(benchmark_dataset, InstructCoderBenchmarkDataset):
            if args.num_chat_sessions:
                inflated_n = _inflated_chat_session_count(
                    args, args.num_chat_sessions
                )
                if args.fit_distributions:
                    with TokenizerPool(tokenizer) as pool:
                        return benchmark_dataset.gen_multiturn_sessions(
                            num_sessions=inflated_n,
                            tokenizer=tokenizer,
                            pool=pool,
                            shuffle=(not args.record_output_lengths),
                            fit_length_distributions=True,
                            num_turns=args.random_num_turns,
                            input_len=args.random_input_len,
                            output_len=args.random_output_len,
                            delay_between_turns_dist=args.delay_between_chat_turns,
                            sys_prompt_ratio=args.random_sys_prompt_ratio,
                            max_num_unique_sys_prompt=args.random_max_num_unique_sys_prompt,
                        )
                else:
                    return benchmark_dataset.gen_multiturn_sessions(
                        num_sessions=inflated_n,
                        tokenizer=tokenizer,
                        shuffle=(not args.record_output_lengths),
                        delay_between_chat_turns=resolve_constant_delay_ms(
                            args.delay_between_chat_turns
                        ),
                    )
            else:
                assert args.num_prompts is not None
                return benchmark_dataset.sample_requests(
                    num_requests=args.num_prompts,
                    tokenizer=tokenizer,
                    output_lengths=output_lengths,
                    shuffle=(
                        output_lengths is None
                        and not args.record_output_lengths
                    ),
                )
        elif isinstance(
            benchmark_dataset, ObfuscatedConversationsBenchmarkDataset
        ):
            if output_lengths is None:
                output_scale = (
                    args.obfuscated_conversations_average_output_len
                    * args.obfuscated_conversations_coefficient_of_variation
                )
                output_lengths = np.random.normal(
                    loc=args.obfuscated_conversations_average_output_len,
                    scale=output_scale,
                    size=num_requests,
                ).tolist()
                output_lengths = np.round(output_lengths).astype(int).tolist()
                output_lengths = [
                    max(output_len, 1) for output_len in output_lengths
                ]
            assert args.num_prompts is not None
            return benchmark_dataset.sample_requests(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                output_lengths=output_lengths,
                shuffle=args.obfuscated_conversations_shuffle,
                seed=args.seed,
            )
        elif isinstance(benchmark_dataset, BatchJobBenchmarkDataset):
            assert args.num_prompts is not None
            return benchmark_dataset.sample_requests(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                output_lengths=output_lengths,
                shuffle=(
                    output_lengths is None and not args.record_output_lengths
                ),
                image_dir=args.batch_job_image_dir,
            )
        elif isinstance(benchmark_dataset, ChatJudgeBenchmarkDataset):
            if not args.num_chat_sessions:
                raise ValueError(
                    "chat-judge dataset requires --num-chat-sessions; "
                    "single-turn mode is not supported."
                )
            return benchmark_dataset.gen_chat_sessions(
                num_sessions=_inflated_chat_session_count(
                    args, args.num_chat_sessions
                ),
                tokenizer=tokenizer,
                shuffle=(not args.record_output_lengths),
                seed=args.seed,
                delay_between_turns_dist=args.delay_between_chat_turns,
            )
        elif isinstance(benchmark_dataset, AgenticCodeBenchmarkDataset):
            if args.num_chat_sessions:
                inflated_n = _inflated_chat_session_count(
                    args, args.num_chat_sessions
                )
                if args.fit_distributions:
                    with TokenizerPool(tokenizer) as pool:
                        return benchmark_dataset.gen_multiturn_sessions(
                            num_sessions=inflated_n,
                            tokenizer=tokenizer,
                            pool=pool,
                            shuffle=(not args.record_output_lengths),
                            fit_length_distributions=True,
                            num_turns=args.random_num_turns,
                            input_len=args.random_input_len,
                            output_len=args.random_output_len,
                            delay_between_turns_dist=args.delay_between_chat_turns,
                            sys_prompt_ratio=args.random_sys_prompt_ratio,
                            max_num_unique_sys_prompt=args.random_max_num_unique_sys_prompt,
                            enable_tool_calls=args.tool_calls,
                        )
                else:
                    return benchmark_dataset.gen_multiturn_sessions(
                        num_sessions=inflated_n,
                        shuffle=(not args.record_output_lengths),
                        enable_tool_calls=args.tool_calls,
                    )
            else:
                assert args.num_prompts is not None
                return benchmark_dataset.sample_requests(
                    num_requests=args.num_prompts,
                    tokenizer=tokenizer,
                    output_lengths=output_lengths,
                    shuffle=(
                        output_lengths is None
                        and not args.record_output_lengths
                    ),
                    enable_tool_calls=args.tool_calls,
                )
        elif isinstance(benchmark_dataset, NemotronOpenCodeBenchmarkDataset):
            if args.num_chat_sessions:
                inflated_n = _inflated_chat_session_count(
                    args, args.num_chat_sessions
                )
                if args.fit_distributions:
                    with TokenizerPool(tokenizer) as pool:
                        return benchmark_dataset.gen_multiturn_sessions(
                            num_sessions=inflated_n,
                            tokenizer=tokenizer,
                            pool=pool,
                            shuffle=(not args.record_output_lengths),
                            fit_length_distributions=True,
                            num_turns=args.random_num_turns,
                            input_len=args.random_input_len,
                            output_len=args.random_output_len,
                            delay_between_turns_dist=args.delay_between_chat_turns,
                            sys_prompt_ratio=args.random_sys_prompt_ratio,
                            max_num_unique_sys_prompt=args.random_max_num_unique_sys_prompt,
                            enable_tool_calls=args.tool_calls,
                        )
                else:
                    return benchmark_dataset.gen_multiturn_sessions(
                        num_sessions=inflated_n,
                        tokenizer=tokenizer,
                        shuffle=(not args.record_output_lengths),
                        enable_tool_calls=args.tool_calls,
                    )
            else:
                assert args.num_prompts is not None
                return benchmark_dataset.sample_requests(
                    num_requests=args.num_prompts,
                    tokenizer=tokenizer,
                    output_lengths=output_lengths,
                    shuffle=(
                        output_lengths is None
                        and not args.record_output_lengths
                    ),
                    enable_tool_calls=args.tool_calls,
                )
        else:
            raise ValueError(
                f"Unknown / unsupported dataset: {benchmark_dataset}"
            )
    elif benchmark_task in PIXEL_GENERATION_TASKS:
        if args.num_prompts is None:
            raise ValueError(
                "Please specify '--num-prompts' for "
                f"{benchmark_task} benchmarks."
            )
        if args.dataset_name == "local-image" and args.dataset_path is None:
            raise ValueError(
                "--benchmark-task image-to-image with "
                f"--dataset-name {args.dataset_name} requires --dataset-path"
            )
        benchmark_dataset = BenchmarkDataset.from_flags(
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
        )
        if benchmark_task == "text-to-image":
            if not isinstance(
                benchmark_dataset, SyntheticPixelBenchmarkDataset
            ):
                raise ValueError(
                    "text-to-image currently supports only "
                    "--dataset-name synthetic-pixel"
                )
        elif benchmark_task == "text-to-video":
            if not isinstance(
                benchmark_dataset, SyntheticPixelBenchmarkDataset
            ):
                raise ValueError(
                    "text-to-video currently supports only "
                    "--dataset-name synthetic-pixel"
                )
            if args.num_frames is None:
                raise ValueError(
                    "--num-frames is required for --benchmark-task text-to-video"
                )
        elif benchmark_task == "image-to-video":
            if not isinstance(
                benchmark_dataset,
                (LocalImageBenchmarkDataset, SyntheticPixelBenchmarkDataset),
            ):
                raise ValueError(
                    "image-to-video currently supports only "
                    "--dataset-name local-image or synthetic-pixel"
                )
            if args.num_frames is None:
                raise ValueError(
                    "--num-frames is required for --benchmark-task image-to-video"
                )
        elif not isinstance(
            benchmark_dataset,
            (LocalImageBenchmarkDataset, SyntheticPixelBenchmarkDataset),
        ):
            raise ValueError(
                "image-to-image currently supports only "
                "--dataset-name local-image or synthetic-pixel"
            )
        logger.info("sampling requests")
        return benchmark_dataset.sample_requests(
            num_requests=args.num_prompts,
            tokenizer=None,
            benchmark_task=benchmark_task,
            image_width=args.image_width,
            image_height=args.image_height,
            image_steps=args.image_steps,
            image_guidance_scale=args.image_guidance_scale,
            image_negative_prompt=args.image_negative_prompt,
            image_seed=args.image_seed,
            num_frames=args.num_frames,
        )
    else:
        raise ValueError(f"Unsupported benchmark task: {benchmark_task}")
