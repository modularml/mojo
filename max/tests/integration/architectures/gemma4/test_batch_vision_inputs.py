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
"""Tests for Gemma4 shared vision packing (``pack_vision_buffers``).

Selection, miss-set threading, and chunked-prefill pos-id alignment moved onto
the pipeline-driven encoder cache and each model's ``pack_vision_inputs`` (see
``test_vision_encoder_cache.py`` and the per-arch ``pack_vision_inputs`` tests).
What remains shared here is the device-side packing itself.
"""

from __future__ import annotations

import numpy as np
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.pipelines.architectures.gemma4.batch_vision_inputs import (
    create_empty_embeddings,
    merge_per_device_buffers,
    pack_vision_buffers,
)
from max.pipelines.architectures.gemma4.context import Gemma4Context
from max.pipelines.context import ImageMetadata, TokenBuffer
from max.pipelines.lib.vision_encoder_cache import (
    VisionCachePlan,
    VisionEncoderCache,
)

_HIDDEN = 4


def _buf(rows: int) -> Buffer:
    """A rank-2 [rows, _HIDDEN] CPU buffer with distinct, recoverable values."""
    data = np.arange(rows * _HIDDEN, dtype=np.float32).reshape(rows, _HIDDEN)
    return Buffer.from_numpy(data).to(CPU())


def test_merge_concatenates_rank2_buffers() -> None:
    # Regression test: the on-device concat indexes ``combined`` with a leading
    # slice (``combined[:a_rows, :]``). Indexing a rank-2 buffer with a single
    # index raised "the provided number of indices (1) is not equal to the
    # tensor rank (2)" and crashed the model worker on any batch with 2+ images.
    a, b = _buf(2), _buf(3)
    a_np, b_np = a.to_numpy().copy(), b.to_numpy().copy()

    [merged] = merge_per_device_buffers([a], [b])

    out = merged.to_numpy()
    assert out.shape == (5, _HIDDEN)
    np.testing.assert_array_equal(out[:2], a_np)
    np.testing.assert_array_equal(out[2:], b_np)


def _buf1d(rows: int) -> Buffer:
    """A rank-1 [rows] CPU buffer (the scatter-index shape)."""
    data = np.arange(rows, dtype=np.int32)
    return Buffer.from_numpy(data).to(CPU())


def test_merge_concatenates_rank1_buffers() -> None:
    # Scatter indices are rank-1; regression for the rank-2-only slice that
    # crashed this caller with "indices (2) != tensor rank (1)".
    a, b = _buf1d(2), _buf1d(3)
    a_np, b_np = a.to_numpy().copy(), b.to_numpy().copy()

    [merged] = merge_per_device_buffers([a], [b])

    out = merged.to_numpy()
    assert out.shape == (5,)
    np.testing.assert_array_equal(out[:2], a_np)
    np.testing.assert_array_equal(out[2:], b_np)


def test_merge_is_elementwise_across_devices() -> None:
    # Two per-device replicas must be concatenated pairwise.
    a0, a1 = _buf(1), _buf(2)
    b0, b1 = _buf(2), _buf(1)

    merged = merge_per_device_buffers([a0, a1], [b0, b1])

    assert len(merged) == 2
    assert merged[0].to_numpy().shape == (3, _HIDDEN)
    assert merged[1].to_numpy().shape == (3, _HIDDEN)


def test_merge_returns_other_side_when_one_is_empty() -> None:
    a, empty = _buf(2), _buf(0)

    # Empty right side -> left returned untouched.
    assert merge_per_device_buffers([a], [empty]) == [a]
    # Empty left side -> right returned untouched.
    assert merge_per_device_buffers([empty], [a]) == [a]
    # Both empty -> first returned.
    assert merge_per_device_buffers([empty], [empty]) == [empty]


def _pixels() -> np.ndarray:
    return np.arange(4 * 3, dtype=np.float32).reshape(4, 3)


def test_pack_vision_buffers_concatenates_multi_image_pos_ids() -> None:
    devices: list[Device] = [CPU()]
    pos0 = np.stack([np.arange(4), np.zeros(4)], axis=1).astype(np.int32)
    pos1 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int32)

    raw = pack_vision_buffers(
        devices,
        1,  # pooling_kernel_size
        [_pixels(), _pixels()],
        [pos0, pos1],
        [4, 4],  # patch_counts
        [4, 4],  # soft_token_counts
        DType.float32,
    )

    np.testing.assert_array_equal(
        raw.pixel_position_ids[0].to_numpy(), np.concatenate([pos0, pos1])
    )
    np.testing.assert_array_equal(
        raw.cu_seqlens[0].to_numpy(), np.array([0, 4, 8], dtype=np.uint32)
    )
    assert raw.patches_flat[0].shape[0] == 8


def test_pack_vision_buffers_packs_multiple_frames_as_images() -> None:
    devices: list[Device] = [CPU()]
    pos = np.stack([np.arange(4), np.zeros(4)], axis=1).astype(np.int32)

    raw = pack_vision_buffers(
        devices,
        1,
        [_pixels() for _ in range(3)],
        [pos, pos, pos],
        [4, 4, 4],
        [4, 4, 4],
        DType.float32,
    )

    assert raw.patches_flat[0].shape[0] == 12
    np.testing.assert_array_equal(
        raw.cu_seqlens[0].to_numpy(), np.array([0, 4, 8, 12], dtype=np.uint32)
    )


_VISION_TOKEN_ID = 98


def _two_image_windowed_context() -> Gemma4Context:
    """Two unencoded images with the active window chunked to cover only img0.

    Nothing is processed yet and the window is chunked to [0, 8), so img1
    (12..16) is ahead of the window (deferred to a later CE iteration).
    """
    tokens = np.array(
        [
            51,
            52,
            53,
            54,
            98,
            98,
            98,
            98,
            55,
            56,
            57,
            58,
            98,
            98,
            98,
            98,
            59,
            60,
        ],
        dtype=np.int64,
    )
    pos0 = np.stack([np.arange(4), np.full(4, 0)], axis=1).astype(np.int32)
    pos1 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int32)
    pixels = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    ctx = Gemma4Context(
        max_length=64,
        tokens=TokenBuffer(tokens),
        images=[
            ImageMetadata(
                start_idx=4,
                end_idx=8,
                pixel_values=pixels,
                image_hash=0xA,
            ),
            ImageMetadata(
                start_idx=12,
                end_idx=16,
                pixel_values=pixels,
                image_hash=0xB,
            ),
        ],
        vision_token_ids=[_VISION_TOKEN_ID],
        mm_token_type_ids=np.zeros(len(tokens), dtype=np.int64),
        pixel_position_ids=[pos0, pos1],
    )
    assert ctx.image_idx == 0
    ctx.tokens.chunk(8)
    return ctx


def test_select_narrows_to_active_window() -> None:
    # The pipeline-owned encode path (run_vision_encode -> select) must
    # respect the scheduler-chunked window: an image ahead of the window is
    # not selected for encoding this iteration.
    ctx = _two_image_windowed_context()
    assert [img.start_idx for img in ctx.next_images_in_window] == [4]

    ve_cache: VisionEncoderCache[Gemma4Context] = VisionEncoderCache(
        plan=VisionCachePlan(
            bytes_per_device=1024 * 1024,
            hidden_size=_HIDDEN,
            dtype=DType.float32,
        ),
        devices=[CPU()],
        block_tokens=4,
    )
    selection = ve_cache.select([ctx])

    assert len(selection) == 1
    sel_ctx, miss_images = selection[0]
    assert sel_ctx is ctx
    assert [img.start_idx for img in miss_images] == [4]


def test_prepare_vision_outputs_skips_deferred_image() -> None:
    # A window-deferred image (ahead of the active window) is neither cached
    # nor in-window. Window-bounded assembly emits no rows for it, and the
    # dense scatter indices carry no positions for it, so the not-in-cache
    # assert is never reached.
    ctx = _two_image_windowed_context()

    ve_cache: VisionEncoderCache[Gemma4Context] = VisionEncoderCache(
        plan=VisionCachePlan(
            bytes_per_device=1024 * 1024,
            hidden_size=_HIDDEN,
            dtype=DType.float32,
        ),
        devices=[CPU()],
        block_tokens=4,
    )
    encoder_out = [
        Buffer.from_numpy(np.ones((4, _HIDDEN), dtype=np.float32)).to(CPU())
    ]
    embeddings, indices = ve_cache.prepare_vision_outputs(
        context_batch=[ctx],
        uncached_contexts=[ctx],
        uncached_images=[[ctx.images[0]]],
        vision_embeds=encoder_out,
        per_image_token_counts=[4],
        n_devices=1,
        empty_embeddings=create_empty_embeddings(
            [CPU()], _HIDDEN, DType.float32
        ),
    )

    arr = embeddings[0].to_numpy()
    # Rows for the in-window image only; the deferred image contributes
    # neither rows nor indices.
    assert arr.shape == (4, _HIDDEN)
    np.testing.assert_array_equal(arr, 1.0)
    np.testing.assert_array_equal(indices, [4, 5, 6, 7])
