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
"""Shared helpers for VLM input batching (image stacking, empty placeholders)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device
from max.dtype import DType


class VisionStacker:
    """Parallel image stacker using a thread pool."""

    def __init__(self, max_workers: int = 24) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def stack(
        self, images: list[npt.NDArray[np.floating[Any]]]
    ) -> npt.NDArray[np.floating[Any]]:
        """Stack images using parallel bulk copy operations."""
        n = len(images)
        if n == 0:
            return np.empty((0,), dtype=np.float32)
        out = np.empty((n, *images[0].shape), dtype=images[0].dtype)
        workers = self._pool._max_workers
        step = math.ceil(n / workers)
        slices = [slice(i, min(i + step, n)) for i in range(0, n, step)]
        futures = [
            self._pool.submit(self._copy_block, out, images, sl)
            for sl in slices
        ]
        for f in as_completed(futures):
            f.result()
        return out

    @staticmethod
    def _copy_block(
        out: npt.NDArray[np.floating[Any]],
        images: list[npt.NDArray[np.floating[Any]]],
        sl: slice,
    ) -> None:
        np.copyto(out[sl], np.asarray(images[sl], dtype=images[0].dtype))


def create_empty_image_embeddings(
    devices: Sequence[Device],
    hidden_size: int,
    dtype: DType = DType.bfloat16,
) -> list[Buffer]:
    """Create per-device zero-row image embedding buffers for text-only decode."""
    return [
        Buffer.zeros(shape=[0, hidden_size], dtype=dtype).to(dev)
        for dev in devices
    ]


def create_empty_image_embeddings_single(
    device: Device,
    hidden_size: int,
    dtype: DType = DType.bfloat16,
) -> Buffer:
    """Create a single-device zero-row image embedding buffer."""
    return Buffer.zeros(shape=[0, hidden_size], dtype=dtype).to(device)


def create_empty_image_token_indices(
    devices: Sequence[Device],
) -> list[Buffer]:
    """Create per-device zero-length scatter-index buffers for text-only decode."""
    return [
        Buffer.zeros(shape=[0], dtype=DType.int32).to(dev) for dev in devices
    ]


def create_empty_image_token_indices_single(device: Device) -> Buffer:
    """Create a single-device zero-length scatter-index buffer."""
    return Buffer.zeros(shape=[0], dtype=DType.int32).to(device)
