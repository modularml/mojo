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

from collections.abc import Sequence

from datasets import load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .local import LocalBenchmarkDataset
from .types import RequestSamples, SampledRequest, encode_image


class VisionArenaBenchmarkDataset(LocalBenchmarkDataset):
    def fetch(self) -> None:
        """Fetch VisionArena dataset based on the current dataset_mode.

        VisionArena datasets are loaded directly in sample_requests, not as a separate fetch step.
        """

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs,
    ) -> RequestSamples:
        dataset = load_dataset(
            "lmarena-ai/vision-arena-bench-v0.1", split="train"
        )
        sampled_requests: list[SampledRequest] = []
        for i in range(num_requests):
            # TODO: Figure out what type to 'assert isinstance' on dataset s.t.
            # MyPy is OK with this (ignored error: Value of type
            # "Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]"
            # is not indexable)
            item = dataset[len(sampled_requests)]  # type: ignore[index]
            prompt = item["turns"][0][0]["content"]
            encoded_images = [encode_image(img) for img in item["images"]]
            prompt_len = len(tokenizer.encode(prompt))
            output_len = None if output_lengths is None else output_lengths[i]
            sampled_requests.append(
                SampledRequest(
                    prompt_formatted=prompt,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    encoded_images=encoded_images,
                    ignore_eos=(output_len is not None),
                )
            )
        return RequestSamples(requests=sampled_requests)
