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

from datasets import load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .local import LocalBenchmarkDataset
from .types import RequestSamples, SampledRequest


class ArxivSummarizationBenchmarkDataset(LocalBenchmarkDataset):
    def fetch(self) -> None:
        """Fetch ArxivSummarization dataset based on the current dataset_mode.

        ArxivSummarization datasets are loaded directly in sample_requests, not as a separate fetch step.
        """

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs,
    ) -> RequestSamples:
        # Extract required parameters from kwargs
        input_len = kwargs.get("input_len")
        max_output_len = kwargs.get("max_output_len")

        # Validate required parameters
        if input_len is None:
            raise ValueError(
                "input_len is required for ArxivSummarizationBenchmarkDataset"
            )

        """Sample requests from the arxiv-summarization dataset.

        Args:
            num_requests: Number of requests to sample
            input_len: Maximum input length in tokens
            output_len: Target output length in tokens
            tokenizer: Tokenizer for processing text

        Returns:
            Sampled requests
        """
        # Load the dataset with train split
        dataset = load_dataset("ccdv/arxiv-summarization", split="train")

        # Shuffle the dataset indices
        indices = list(range(len(dataset)))  # type: ignore[arg-type]
        if shuffle:
            random.shuffle(indices)

        # Create a summarization prompt
        prompt_prefix = "Summarize the following research paper:\n\n"
        prompt_suffix = "\n\nSummary:"

        # Calculate tokens for prefix and suffix
        prefix_tokens = tokenizer.encode(
            prompt_prefix, add_special_tokens=False
        )
        suffix_tokens = tokenizer.encode(
            prompt_suffix, add_special_tokens=False
        )

        # Reserve space for prefix and suffix
        max_article_len = input_len - len(prefix_tokens) - len(suffix_tokens)

        sampled_requests: list[SampledRequest] = []
        for idx in indices:
            if len(sampled_requests) >= num_requests:
                break

            # TODO: Figure out what type to 'assert isinstance' on dataset s.t.
            # MyPy is OK with this (ignored error: Value of type
            # "Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]"
            # is not indexable)
            item = dataset[idx]  # type: ignore[index]
            article = item["article"]

            # Tokenize the article to check length
            article_tokens = tokenizer.encode(article, add_special_tokens=False)

            # Truncate article if necessary
            if len(article_tokens) > max_article_len:
                article_tokens = article_tokens[:max_article_len]
                article = tokenizer.decode(
                    article_tokens, skip_special_tokens=True
                )

            # Create the full prompt
            prompt_formatted = f"{prompt_prefix}{article}{prompt_suffix}"

            # Re-tokenize and get the actual prompt length.
            # Note that the final prompt size usually does not match
            # len(prefix)+len(suffix)+len(article_tokens) exactly because most
            # tokenizers are not entirely stateless; i.e. adding the prefix
            # changes the behavior. This means the result may be slightly larger
            # than the given input_len (by up to ~10 tokens) despite the
            # truncation logic above. The prompt could of course also be shorter
            # than the given input_len, if the downloaded paper happens to be a
            # small one.
            prompt_len = len(
                tokenizer.encode(prompt_formatted, add_special_tokens=False)
            )

            # Tokenize the abtsract to get output length.
            abstract = item["abstract"]
            abstract_tokens = tokenizer.encode(
                abstract, add_special_tokens=False
            )
            output_len = len(abstract_tokens)

            # Skip outputs that are too large.
            if max_output_len and output_len > max_output_len:
                continue

            sampled_requests.append(
                SampledRequest(
                    prompt_formatted=prompt_formatted,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    encoded_images=[],
                    ignore_eos=(output_len is not None),
                )
            )

        return RequestSamples(requests=sampled_requests)
