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

"""Artificial Analysis (AA) representative-workload prompt corpus.

Reproduces the Artificial Analysis performance-benchmarking methodology
(https://artificialanalysis.ai/methodology/performance-benchmarking): each
request pairs long-form input content (real arxiv articles) with one of a
range of tasks (summarization, Q&A generation, comparative analysis,
translation), with the content trimmed to fill a target *input* token budget.

Token budgets ("1k / 10k / 100k") are sized with the served model's own
tokenizer: content is trimmed so each prompt encodes to ~``input_len`` tokens
under that tokenizer, and each :class:`SampledRequest` records the resulting
``prompt_len`` (the framework convention used for metrics).

Prompt diversity matters: under speculative decoding, output speed varies with
output type, so a representative benchmark must span tasks and content rather
than emit uniform synthetic text.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator, Sequence

from datasets import Dataset, load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .distribution import BaseDistribution, DistributionParameter
from .interface import BenchmarkDataset
from .types import RequestSamples, SampledRequest

logger = logging.getLogger(__name__)

# Content source: the same arxiv papers the arxiv-summarization dataset uses,
# here as long-form content. Papers are long and coherent, so a budget (esp.
# 10k/100k) fills from a few concatenated papers rather than many short
# fragments.
DEFAULT_HF_REPO = "ccdv/arxiv-summarization"
DEFAULT_HF_SPLIT = "train"
DEFAULT_HF_TEXT_FIELD = "article"

# Joiner inserted between accumulated articles when a single article is shorter
# than the target budget (needed for the 10k/100k workloads).
_ARTICLE_JOINER = "\n\n---\n\n"

# Floors for per-request lengths sampled from the input/output distributions,
# guarding against pathological distribution tails.
_MIN_INPUT_LEN = 4
_MIN_OUTPUT_LEN = 1

# AA's task mix. Each entry is ``(prefix, suffix)`` wrapped around the content.
# Visual-artifact generation is handled separately by the vision workload.
_TASK_TEMPLATES: dict[str, tuple[str, str]] = {
    "summarize": (
        "Read the following text carefully and write a thorough, "
        "well-structured summary that captures its main arguments, "
        "supporting details, and conclusions.\n\n=== TEXT START ===\n",
        "\n=== TEXT END ===\n\nSummary:",
    ),
    "qa_generation": (
        "Read the following text and generate a comprehensive set of "
        "question-and-answer pairs that test understanding of its key facts, "
        "concepts, and implications. Provide a detailed answer for each "
        "question.\n\n=== TEXT START ===\n",
        "\n=== TEXT END ===\n\nQuestion-and-answer pairs:",
    ),
    "comparative_analysis": (
        "Read the following text and write a detailed comparative analysis of "
        "the different ideas, perspectives, or approaches it presents, "
        "explaining their similarities, differences, and relative "
        "merits.\n\n=== TEXT START ===\n",
        "\n=== TEXT END ===\n\nComparative analysis:",
    ),
    "translation": (
        "Translate the following text into French. Preserve the meaning, "
        "tone, and structure as closely as possible.\n\n=== TEXT START ===\n",
        "\n=== TEXT END ===\n\nFrench translation:",
    ),
}


class ArtificialAnalysisBenchmarkDataset(BenchmarkDataset):
    """Artificial Analysis representative-workload corpus generator.

    Pairs long-form arxiv articles with AA's task mix, sizing each prompt to
    a target input-token budget under the served model's tokenizer.
    """

    def fetch(self) -> None:
        """No-op: content is loaded lazily in :meth:`sample_requests`."""

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        input_len: DistributionParameter | None = None,
        output_len: DistributionParameter | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> RequestSamples:
        """Generate ``num_requests`` AA-style prompts sized to an input budget.

        Args:
            num_requests: Number of prompts to generate.
            tokenizer: Served model's tokenizer; used both to size prompts to
                the input budget and to record ``prompt_len`` for metrics.
            output_lengths: Optional explicit per-request output-token targets.
                When provided, they take precedence over ``output_len`` and
                ``ignore_eos`` is enabled so the model generates the full
                budget (AA requires "at least N" output tokens).
            shuffle: Whether to shuffle the content order. Default ``True``.
            input_len: Input-token budget per prompt as a distribution (e.g.
                ``"N(10000, 1500)"`` or a constant like ``10000``); sampled
                per request and measured with the served model's tokenizer
                (required).
            output_len: Output-token target per prompt as a distribution,
                sampled per request when ``output_lengths`` is not given. The
                sample is used as a max-tokens cap with EOS left enabled (the
                model may stop early but cannot run away).
            seed: Seed for reproducible content/length sampling. Default 0.

        Returns:
            The generated requests.
        """
        if input_len is None:
            raise ValueError(
                "input_len is required for ArtificialAnalysisBenchmarkDataset"
            )

        if output_lengths is not None and len(output_lengths) != num_requests:
            raise ValueError(
                "output_lengths must have length num_requests "
                f"({len(output_lengths)} != {num_requests})"
            )

        input_dist = BaseDistribution.from_distribution_parameter(input_len)
        assert input_dist is not None
        output_dist = (
            BaseDistribution.from_distribution_parameter(output_len)
            if output_len is not None
            else None
        )

        # Balanced task mix, shuffled so task doesn't track request order.
        rng = random.Random(seed)
        task_pool = tuple(_TASK_TEMPLATES)
        tasks = [task_pool[i % len(task_pool)] for i in range(num_requests)]
        rng.shuffle(tasks)

        content_iter = self._iter_articles(shuffle, seed)

        def count(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

        # Cache per-task template overhead instead of recomputing each request.
        task_overhead = {
            name: count(prefix) + count(suffix)
            for name, (prefix, suffix) in _TASK_TEMPLATES.items()
        }

        sampled: list[SampledRequest] = []
        prompt_lens: list[int] = []
        for i in range(num_requests):
            task = tasks[i]
            prefix, suffix = _TASK_TEMPLATES[task]

            # Per-request input budget sampled from the distribution; reserve
            # the template's own tokens so the whole prompt lands near it.
            req_input_len = max(
                round(input_dist.sample_value()), _MIN_INPUT_LEN
            )
            overhead = task_overhead[task]
            content_budget = req_input_len - overhead
            if content_budget <= 0:
                raise ValueError(
                    f"sampled input_len={req_input_len} is too small for task "
                    f"'{task}' (template overhead is {overhead} model tokens)"
                )

            content = self._fill_content(
                content_iter, content_budget, tokenizer
            )
            prompt = f"{prefix}{content}{suffix}"

            prompt_len = count(prompt)
            prompt_lens.append(prompt_len)

            # Explicit output_lengths are forced targets (ignore_eos). A
            # sampled length is a max-tokens cap with EOS left enabled, so the
            # model may stop early but can't run away.
            if output_lengths is not None:
                req_output_len: int | None = int(output_lengths[i])
                req_ignore_eos = True
            elif output_dist is not None:
                req_output_len = max(
                    round(output_dist.sample_value()), _MIN_OUTPUT_LEN
                )
                req_ignore_eos = False
            else:
                req_output_len = None
                req_ignore_eos = False
            sampled.append(
                SampledRequest(
                    prompt_formatted=prompt,
                    prompt_len=prompt_len,
                    output_len=req_output_len,
                    encoded_images=[],
                    ignore_eos=req_ignore_eos,
                )
            )

        if prompt_lens:
            logger.info(
                "AA corpus: generated %d prompts (input_len=%s; actual "
                "model-token min/mean/max = %d/%d/%d); tasks=%s",
                len(sampled),
                input_len,
                min(prompt_lens),
                sum(prompt_lens) // len(prompt_lens),
                max(prompt_lens),
                ",".join(tasks),
            )

        return RequestSamples(requests=sampled)

    def _iter_articles(self, shuffle: bool, seed: int | None) -> Iterator[str]:
        """Yield non-empty article texts from the HF dataset.

        Loads the full split (memory-mapped Arrow, not materialized) and walks
        index order, shuffled for content diversity. Articles are read lazily,
        so only as many as the prompts need are fetched.
        """
        dataset = load_dataset(DEFAULT_HF_REPO, split=DEFAULT_HF_SPLIT)
        assert isinstance(dataset, Dataset)
        indices = list(range(len(dataset)))
        if shuffle:
            random.Random(seed).shuffle(indices)
        for idx in indices:
            text = (dataset[idx][DEFAULT_HF_TEXT_FIELD] or "").strip()
            if text:
                yield text

    def _fill_content(
        self,
        content_iter: Iterator[str],
        budget: int,
        tokenizer: PreTrainedTokenizerBase,
    ) -> str:
        """Accumulate articles up to ``budget`` model tokens.

        A single article is often shorter than the 10k/100k budgets, so
        articles are concatenated (joined by a separator) until the budget is
        reached, then trimmed to ``budget`` tokens under the model tokenizer.
        Concatenating distinct sources also gives the comparative-analysis task
        multiple things to compare.

        Each article is encoded once and assembled at the token level (token
        counting dominates dataset generation, so the joined text is not
        re-encoded).
        """
        joiner_ids = tokenizer.encode(_ARTICLE_JOINER, add_special_tokens=False)
        token_ids: list[int] = []
        for article in content_iter:
            if token_ids:
                token_ids.extend(joiner_ids)
            token_ids.extend(
                tokenizer.encode(article, add_special_tokens=False)
            )
            if len(token_ids) >= budget:
                break
        else:
            logger.warning(
                "AA content stream exhausted at %d/%d model tokens; prompt "
                "will be shorter than the target budget.",
                len(token_ids),
                budget,
            )

        return tokenizer.decode(token_ids[:budget], skip_special_tokens=True)
