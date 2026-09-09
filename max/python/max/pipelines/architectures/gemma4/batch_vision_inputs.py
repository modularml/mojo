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

"""Dataclasses and builders for batched vision / video model inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device, DevicePinnedBuffer
from max.dtype import DType
from max.graph.buffer_utils import cast_tensor_to
from max.pipelines.context import ImageMetadata
from max.profiler import traced

from .context import Gemma4Context
from .vision_model.pooling import compute_pool_gather_index


@dataclass
class VisionRawInputs:
    """Raw vision-encoder inputs for a batch of uncached images or video frames.

    All buffer lists are per-device replicas (length = ``n_devices``),
    except ``max_seq_len`` which is a single CPU scalar.
    """

    patches_flat: list[Buffer]
    pixel_position_ids: list[Buffer]
    cu_seqlens: list[Buffer]
    pool_gather_index: list[Buffer]
    max_seq_len: Buffer


def create_empty_embeddings(
    devices: list[Device], hidden_size: int, dtype: DType = DType.bfloat16
) -> list[Buffer]:
    """Create empty (zero-row) embedding buffers, one per device."""
    return [
        Buffer.zeros(shape=[0, hidden_size], dtype=dtype).to(dev)
        for dev in devices
    ]


def create_empty_indices(devices: list[Device]) -> list[Buffer]:
    """Create empty (zero-length) scatter-index buffers, one per device."""
    return [
        Buffer.zeros(shape=[0], dtype=DType.int32).to(dev) for dev in devices
    ]


def merge_per_device_buffers(
    a_bufs: list[Buffer],
    b_bufs: list[Buffer],
) -> list[Buffer]:
    """Concatenate two per-device buffer lists element-wise along axis 0.

    When either side is empty the other is returned directly. Otherwise the
    concat stays on-device: allocate the combined buffer and copy each half
    into a contiguous leading-axis slice, avoiding a GPU->host->GPU round-trip.
    """
    merged: list[Buffer] = []
    for a, b in zip(a_bufs, b_bufs, strict=True):
        a_rows = a.shape[0]
        b_rows = b.shape[0]
        if a_rows == 0 and b_rows == 0:
            merged.append(a)
        elif a_rows == 0:
            merged.append(b)
        elif b_rows == 0:
            merged.append(a)
        else:
            combined = Buffer(
                shape=(a_rows + b_rows, *a.shape[1:]),
                dtype=a.dtype,
                device=a.device,
            )
            # Slice the leading axis at the buffer's rank: this merges both
            # rank-2 embeddings and rank-1 indices, and MAX requires
            # index-count == rank.
            n_extra = len(a.shape) - 1
            front = (slice(None, a_rows), *(slice(None),) * n_extra)
            back = (slice(a_rows, None), *(slice(None),) * n_extra)
            combined[front].inplace_copy_from(a)
            combined[back].inplace_copy_from(b)
            merged.append(combined)
    return merged


def _pinned_to_devices(
    np_array: npt.NDArray[Any], dtype: DType, devices: list[Device]
) -> list[Buffer]:
    """Copy a numpy array to each device via a pinned host buffer."""
    dev0 = devices[0]
    host: Buffer
    if not dev0.is_host:
        host = DevicePinnedBuffer(
            dtype=dtype, shape=np_array.shape, device=dev0
        )
    else:
        host = Buffer(shape=np_array.shape, dtype=dtype, device=dev0)
    host.to_numpy()[:] = np_array
    device_bufs = [host.to(d) for d in devices]
    for d in device_bufs:
        d.inplace_copy_from(host)
    return device_bufs


@traced
def pack_vision_buffers(
    devices: list[Device],
    pooling_kernel_size: int,
    all_patches: list[npt.NDArray[np.floating[Any]]],
    all_pos_ids: list[npt.NDArray[np.integer[Any]]],
    patch_counts: list[int],
    soft_token_counts: list[int],
    dtype: DType,
) -> VisionRawInputs:
    """Build device-replicated ``VisionRawInputs`` from numpy arrays."""
    patches_flat_np = np.concatenate(all_patches, axis=0).astype(np.float32)
    pos_ids_np = np.concatenate(all_pos_ids, axis=0)

    n_items = len(all_patches)
    cu_seqlens_np = np.empty(n_items + 1, dtype=np.uint32)
    cu_seqlens_np[0] = 0
    np.cumsum(patch_counts, out=cu_seqlens_np[1:])

    max_seq_len_np = np.array(max(patch_counts), dtype=np.uint32)

    # Pooling gather index: per output token, the patch indices that pool into
    # it (shape [num_pooled, max_per_bin]).
    pool_gather_index_np = compute_pool_gather_index(
        all_pos_ids, soft_token_counts, pooling_kernel_size
    )

    # Use pinned host buffers for h2d copies.
    patches_flat_bufs = _pinned_to_devices(
        patches_flat_np, DType.float32, devices
    )
    patches_flat = [cast_tensor_to(buf, dtype) for buf in patches_flat_bufs]

    return VisionRawInputs(
        patches_flat=patches_flat,
        pixel_position_ids=_pinned_to_devices(
            pos_ids_np.astype(np.int32), DType.int32, devices
        ),
        cu_seqlens=_pinned_to_devices(cu_seqlens_np, DType.uint32, devices),
        pool_gather_index=_pinned_to_devices(
            pool_gather_index_np, DType.int32, devices
        ),
        max_seq_len=Buffer.from_numpy(max_seq_len_np),
    )


@traced
def pack_uncached_images(
    selection: Sequence[tuple[Gemma4Context, Sequence[ImageMetadata]]],
    devices: list[Device],
    pooling_kernel_size: int,
    dtype: DType,
) -> VisionRawInputs | None:
    """Pack a batch's pipeline-selected cache-miss image pixels to device.

    Takes the ``(context, miss-images)`` pairs the pipeline's ``select``
    returned and does the pinned host-to-device copy via
    :func:`pack_vision_buffers`. Slices ``pixel_position_ids[ctx.image_idx:]``
    so the full per-image position list realigns with ``next_images`` under
    chunked prefill, and validates each image's patch count. Returns ``None``
    when nothing needs encoding.

    Safety invariant: ``select`` and this packer must see the SAME image objects
    within one ``run_vision_encode`` call. The miss set is matched by object
    identity (``id(img)``) while iterating ``next_images`` (so each image keeps
    its ``pixel_position_ids`` slot). Rebuilding or copying ``ctx.images``
    between select and pack raises here instead of silently dropping images.
    """
    k = pooling_kernel_size
    all_patches: list[npt.NDArray[np.floating[Any]]] = []
    all_pos_ids: list[npt.NDArray[np.integer[Any]]] = []
    patch_counts: list[int] = []
    soft_token_counts: list[int] = []
    for ctx, miss_images in selection:
        ctx_pos_ids = ctx.pixel_position_ids[ctx.image_idx :]
        uncached_ids = {id(img) for img in miss_images}
        consumed = 0
        for j, img in enumerate(ctx.next_images):
            if id(img) not in uncached_ids:
                continue
            consumed += 1
            num_soft = img.end_idx - img.start_idx
            num_patches = num_soft * k * k
            if num_patches != len(img.pixel_values):
                raise ValueError(
                    f"Expected {num_patches} patches, "
                    f"got {len(img.pixel_values)}"
                )
            all_patches.append(img.pixel_values)
            all_pos_ids.append(ctx_pos_ids[j])
            patch_counts.append(num_patches)
            soft_token_counts.append(num_soft)
        if consumed != len(miss_images):
            raise ValueError(
                f"{len(miss_images) - consumed} of {len(miss_images)} selected "
                f"image(s) for request {ctx.request_id} are not present in "
                "ctx.next_images. The selection must hold the same "
                "ImageMetadata objects as the context."
            )
    if not all_patches:
        return None
    return pack_vision_buffers(
        devices,
        k,
        all_patches,
        all_pos_ids,
        patch_counts,
        soft_token_counts,
        dtype,
    )
