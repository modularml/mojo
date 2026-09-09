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
"""Packing of Qwen3.5 vision encoder inputs for a batch's cache-miss images."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device
from max.pipelines.architectures.qwen3vl_moe.context import (
    Qwen3VLTextAndVisionContext,
)
from max.pipelines.architectures.qwen3vl_moe.nn.data_processing import (
    get_seqlens,
)
from max.pipelines.context import ImageMetadata
from max.profiler import traced


@dataclass
class Qwen3_5VisionInputs:
    """One batch's vision encoder graph inputs, in graph argument order."""

    pixel_values: Buffer
    weights: Buffer
    indices: Buffer
    vision_position_ids: Buffer
    max_grid_size: Buffer
    grid_thw: Buffer
    cu_seqlens: Buffer
    max_seqlen: Buffer


@dataclass
class _ImagePiece:
    """One image's share of its context's concatenated vision arrays."""

    grid: npt.NDArray[np.integer[Any]]
    pixel_values: npt.NDArray[np.floating[Any]]
    weights: npt.NDArray[np.floating[Any]]
    indices: npt.NDArray[np.integer[Any]]
    vision_position_ids: npt.NDArray[np.integer[Any]]


def _split_vision_data(
    ctx: Qwen3VLTextAndVisionContext,
) -> list[_ImagePiece]:
    """Cut a context's per-image vision arrays back out of their concatenation.

    The tokenizer builds one ``VisionEncodingData`` per context by
    concatenating each image's arrays in ``ctx.images`` order, so the split is
    a cumulative-sum walk. Two strides are involved: pixels and rotary
    positions carry ``t * h * w`` rows per image, while the bilinear
    interpolation weights and indices carry ``h * w`` columns — the position
    grid is interpolated once per frame, not once per temporal patch.
    """
    vision_data = ctx.vision_data
    assert vision_data is not None
    grids = np.asarray(vision_data.image_grid_thw)
    patch_counts = grids[:, 0] * grids[:, 1] * grids[:, 2]
    plane_counts = grids[:, 1] * grids[:, 2]
    patch_offsets = np.concatenate([[0], np.cumsum(patch_counts)])
    plane_offsets = np.concatenate([[0], np.cumsum(plane_counts)])

    pieces: list[_ImagePiece] = []
    for i in range(len(grids)):
        patch_lo, patch_hi = int(patch_offsets[i]), int(patch_offsets[i + 1])
        plane_lo, plane_hi = int(plane_offsets[i]), int(plane_offsets[i + 1])
        pieces.append(
            _ImagePiece(
                grid=grids[i],
                pixel_values=vision_data.concatenated_pixel_values[
                    patch_lo:patch_hi
                ],
                weights=vision_data.weights[:, plane_lo:plane_hi, :],
                indices=vision_data.indices[:, plane_lo:plane_hi],
                vision_position_ids=vision_data.vision_position_ids[
                    patch_lo:patch_hi
                ],
            )
        )
    return pieces


@traced
def pack_uncached_images(
    selection: Sequence[
        tuple[Qwen3VLTextAndVisionContext, Sequence[ImageMetadata]]
    ],
    devices: list[Device],
) -> Qwen3_5VisionInputs | None:
    """Pack a batch's pipeline-selected cache-miss image inputs to device.

    Takes the ``(context, miss-images)`` pairs the pipeline's ``select``
    returned and rebuilds the encoder's batch-level arrays over just those
    images, so a partial cache hit re-encodes only what it missed. Returns
    ``None`` when nothing needs encoding.

    Safety invariant: ``select`` and this packer must see the SAME image
    objects within one ``run_vision_encode`` call. The miss set is matched by
    object identity while walking ``ctx.images``, so a context whose images
    were rebuilt in between raises here instead of silently encoding the wrong
    pixels.

    Raises:
        ValueError: If a selected context carries no vision data, or a
            selected image is not one of its context's own.
    """
    grids: list[npt.NDArray[np.integer[Any]]] = []
    pixel_values: list[npt.NDArray[np.floating[Any]]] = []
    weights: list[npt.NDArray[np.floating[Any]]] = []
    indices: list[npt.NDArray[np.integer[Any]]] = []
    position_ids: list[npt.NDArray[np.integer[Any]]] = []

    for ctx, miss_images in selection:
        if ctx.vision_data is None:
            raise ValueError(
                f"Request {ctx.request_id} was selected for vision encoding "
                "but carries no vision_data."
            )
        wanted = {id(img) for img in miss_images}
        consumed = 0
        for img, piece in zip(ctx.images, _split_vision_data(ctx), strict=True):
            if id(img) not in wanted:
                continue
            consumed += 1
            grids.append(piece.grid)
            pixel_values.append(piece.pixel_values)
            weights.append(piece.weights)
            indices.append(piece.indices)
            position_ids.append(piece.vision_position_ids)
        if consumed != len(miss_images):
            raise ValueError(
                f"{len(miss_images) - consumed} of {len(miss_images)} selected "
                f"image(s) for request {ctx.request_id} are not present in "
                "ctx.images. The selection must hold the same ImageMetadata "
                "objects as the context."
            )

    if not grids:
        return None

    grid_thw = np.stack(grids).astype(np.int32)
    cu_seqlens, max_seqlen = get_seqlens(grid_thw)
    device0 = devices[0]
    return Qwen3_5VisionInputs(
        pixel_values=Buffer.from_numpy(
            np.concatenate(pixel_values).astype(np.float32)
        ).to(device0),
        weights=Buffer.from_numpy(
            np.concatenate(weights, axis=1).astype(np.float32)
        ).to(device0),
        indices=Buffer.from_numpy(
            np.concatenate(indices, axis=1).astype(np.int64)
        ).to(device0),
        vision_position_ids=Buffer.from_numpy(
            np.concatenate(position_ids).astype(np.int32)
        ).to(device0),
        max_grid_size=Buffer.from_numpy(
            np.array(int(np.max(grid_thw[:, 1:])), dtype=np.int32)
        ),
        grid_thw=Buffer.from_numpy(grid_thw.astype(np.int64)).to(device0),
        cu_seqlens=Buffer.from_numpy(cu_seqlens.astype(np.uint32)).to(device0),
        max_seqlen=Buffer.from_numpy(np.array([max_seqlen], dtype=np.uint32)),
    )
