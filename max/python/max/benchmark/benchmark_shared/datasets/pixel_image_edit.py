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
from pathlib import Path
from typing import Any

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing_extensions import override

from .pixel import PixelBenchmarkDataset
from .types import PixelGenerationSampledRequest, RequestSamples


class LocalImageBenchmarkDataset(PixelBenchmarkDataset):
    @override
    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase | None,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs: Any,
    ) -> RequestSamples:
        assert self.dataset_path is not None, (
            "dataset_path must be provided for LocalImageBenchmarkDataset"
        )

        dataset_file = Path(self.dataset_path)
        rows: list[dict[str, Any]] = []
        with open(dataset_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} in {dataset_file}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"Expected JSON object on line {line_num} in {dataset_file}"
                    )
                rows.append(row)

        # Keep request count exact by trimming when large and repeating when small.
        if num_requests <= 0:
            return RequestSamples(requests=[])
        if not rows:
            raise ValueError("Dataset is empty.")
        if shuffle:
            random.shuffle(rows)
        if len(rows) >= num_requests:
            rows = rows[:num_requests]
        else:
            factor = (num_requests // len(rows)) + 1
            rows = (rows * factor)[:num_requests]

        # TODO(MAX-PR): Support per-row image options in local-image datasets.
        cli_options = self._build_image_options(
            image_width=kwargs.get("image_width"),
            image_height=kwargs.get("image_height"),
            image_steps=kwargs.get("image_steps"),
            image_guidance_scale=kwargs.get("image_guidance_scale"),
            image_negative_prompt=kwargs.get("image_negative_prompt"),
            image_seed=kwargs.get("image_seed"),
            num_frames=kwargs.get("num_frames"),
        )
        requests: list[PixelGenerationSampledRequest] = []

        for row in rows:
            prompt = row.get("prompt")
            image_path_value = row.get("image_path")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    "local-image dataset rows must include a non-empty 'prompt'."
                )
            if not isinstance(image_path_value, str) or not image_path_value:
                raise ValueError(
                    "local-image dataset rows must include 'image_path'."
                )

            image_path = Path(image_path_value)
            if not image_path.is_absolute():
                image_path = dataset_file.parent / image_path
            image_path = image_path.resolve()
            if not image_path.exists():
                raise ValueError(f"Image path {image_path} does not exist.")

            requests.append(
                self._build_request(
                    prompt=prompt,
                    image_options=cli_options,
                    input_image_paths=[str(image_path)],
                )
            )

        return RequestSamples(requests=requests)
