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

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing_extensions import override

from .pixel import PixelBenchmarkDataset
from .types import RequestSamples


class SyntheticPixelBenchmarkDataset(PixelBenchmarkDataset):
    @override
    def fetch(self) -> None:
        """Fetch Synthetic Pixel dataset.

        Synthetic pixel prompts are generated in-memory and do not require a
        local file.
        """

    def _get_placeholder_image_path(
        self,
        *,
        width: int,
        height: int,
    ) -> str:
        image_path = (
            Path(tempfile.gettempdir())
            / f"max_benchmark_random_image_{width}x{height}.png"
        )
        if not image_path.exists():
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (width, height), color="white").save(image_path)
        return str(image_path)

    @override
    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase | None,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs: Any,
    ) -> RequestSamples:
        image_options = self._build_image_options(
            image_width=kwargs.get("image_width"),
            image_height=kwargs.get("image_height"),
            image_steps=kwargs.get("image_steps"),
            image_guidance_scale=kwargs.get("image_guidance_scale"),
            image_negative_prompt=kwargs.get("image_negative_prompt"),
            image_seed=kwargs.get("image_seed"),
            num_frames=kwargs.get("num_frames"),
        )
        benchmark_task = kwargs.get("benchmark_task")
        input_image_paths: list[str] = []
        # Both image-to-image and image-to-video consume an input image; only
        # the payload routing (num-frames) differs downstream.
        if benchmark_task in ("image-to-image", "image-to-video"):
            input_image_paths = [
                self._get_placeholder_image_path(
                    width=kwargs.get("image_width") or 1024,
                    height=kwargs.get("image_height") or 1024,
                )
            ]

        requests = [
            self._build_request(
                prompt=f"Random prompt {idx} for benchmarking pixel generation pipelines",
                image_options=image_options,
                input_image_paths=input_image_paths,
            )
            for idx in range(num_requests)
        ]
        return RequestSamples(requests=requests)
