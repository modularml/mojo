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

"""Vision encoder cache wiring for Qwen3.5.

Two halves: the arch config's vision-cache facts, which decide whether memory
planning reserves a cache at all, and the packer that rebuilds the encoder's
batch arrays over just the images the cache missed.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import CPU
from max.dtype import DType
from max.pipelines.architectures.qwen2_5vl.nn.data_processing import (
    mrope_pos_ids_3d,
)
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.qwen3_5.vision_packing import (
    pack_uncached_images,
)
from max.pipelines.architectures.qwen3vl_moe.context import (
    Qwen3VLTextAndVisionContext,
    VisionEncodingData,
)
from max.pipelines.architectures.qwen3vl_moe.nn.data_processing import (
    QWEN3VL_MAX_PIXELS,
    get_bilinear_interpolation_weights_and_indices,
    get_seqlens,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.arch import (
    UnifiedMTPQwen3_5Config,
)
from max.pipelines.context import ImageMetadata, TokenBuffer
from max.pipelines.lib.interfaces import arch_has_vision_tower

IMAGE_TOKEN_ID = 151655
PATCH_SIZE = 16
MERGE_SIZE = 2
NUM_GRID_PER_SIDE = 48
PATCH_DIM = 4
GRIDS = [(1, 4, 6), (1, 2, 4)]


def _hf_config(*, with_vision: bool = True) -> SimpleNamespace:
    text_config = SimpleNamespace(hidden_size=2048, dtype="bfloat16")
    if not with_vision:
        return SimpleNamespace(text_config=text_config)
    return SimpleNamespace(
        text_config=text_config,
        vision_config=SimpleNamespace(
            patch_size=PATCH_SIZE,
            spatial_merge_size=MERGE_SIZE,
            out_hidden_size=2048,
        ),
    )


def test_row_spec_is_the_lm_hidden_at_the_declared_dtype() -> None:
    assert Qwen3_5Config.get_vision_cache_row_spec(_hf_config()) == (
        2048,
        DType.bfloat16,
    )


def test_entry_bytes_bound_by_the_processor_resolution_ceiling() -> None:
    max_tokens = QWEN3VL_MAX_PIXELS // (PATCH_SIZE**2 * MERGE_SIZE**2)
    assert Qwen3_5Config.estimate_vision_cache_entry_bytes(_hf_config()) == (
        max_tokens * 2048 * DType.bfloat16.size_in_bytes
    )
    assert arch_has_vision_tower(Qwen3_5Config, _hf_config())


def test_text_only_checkpoint_publishes_no_vision_cache() -> None:
    text_only = _hf_config(with_vision=False)
    assert Qwen3_5Config.estimate_vision_cache_entry_bytes(text_only) == 0
    assert Qwen3_5Config.get_vision_cache_row_spec(text_only) is None
    assert not arch_has_vision_tower(Qwen3_5Config, text_only)


def test_unified_mtp_withdraws_the_vision_cache() -> None:
    """The fused MTP graph drops the tower, so it must not reserve a cache."""
    assert (
        UnifiedMTPQwen3_5Config.estimate_vision_cache_entry_bytes(_hf_config())
        == 0
    )
    assert (
        UnifiedMTPQwen3_5Config.get_vision_cache_row_spec(_hf_config()) is None
    )
    assert not arch_has_vision_tower(UnifiedMTPQwen3_5Config, _hf_config())


def _tokens_for(grids: list[tuple[int, int, int]]) -> npt.NDArray[np.int64]:
    tokens: list[int] = [1, 1]
    for t, h, w in grids:
        tokens.extend([IMAGE_TOKEN_ID] * (t * h * w // MERGE_SIZE**2))
        tokens.append(2)
    return np.array(tokens, dtype=np.int64)


def _context() -> Qwen3VLTextAndVisionContext:
    """A two-image context built the way the tokenizer builds one."""
    grid_thw = np.array(GRIDS, dtype=np.int32)
    patch_counts = [t * h * w for t, h, w in GRIDS]
    rng = np.random.default_rng(0)
    pixel_values_list = [
        rng.standard_normal((n, PATCH_DIM), dtype=np.float32)
        for n in patch_counts
    ]
    indices, weights = get_bilinear_interpolation_weights_and_indices(
        grid_thw=grid_thw, num_grid_per_side=NUM_GRID_PER_SIDE
    )
    cu_seqlens, max_seqlen = get_seqlens(grid_thw=grid_thw)

    tokens = _tokens_for(GRIDS)
    image_token_indices = (tokens == IMAGE_TOKEN_ID).nonzero()[0]
    spans: list[tuple[int, int]] = []
    cursor = 0
    for t, h, w in GRIDS:
        rows = t * h * w // MERGE_SIZE**2
        start = int(image_token_indices[cursor])
        spans.append((start, start + rows))
        cursor += rows

    return Qwen3VLTextAndVisionContext(
        max_length=len(tokens) + 1,
        tokens=TokenBuffer(tokens),
        images=[
            ImageMetadata(start_idx=start, end_idx=end, pixel_values=pixels)
            for (start, end), pixels in zip(
                spans, pixel_values_list, strict=True
            )
        ],
        vision_token_ids=[IMAGE_TOKEN_ID],
        spatial_merge_size=MERGE_SIZE,
        rope_delta=0,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        image_token_indices=image_token_indices.astype(np.int32),
        decoder_position_ids=np.zeros((3, len(tokens)), dtype=np.int64),
        vision_data=VisionEncodingData(
            image_grid_thw=grid_thw,
            video_grid_thw=None,
            vision_position_ids=mrope_pos_ids_3d(
                grid_thw=grid_thw, spatial_merge_size=MERGE_SIZE
            ),
            max_grid_size=np.array(
                int(np.max(grid_thw[:, 1:])), dtype=np.int32
            ),
            weights=weights,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=np.array(max_seqlen, dtype=np.uint32),
            concatenated_pixel_values=np.vstack(pixel_values_list),
        ),
    )


def test_packing_every_image_reproduces_the_whole_context() -> None:
    """Selecting both images must rebuild the tokenizer's own arrays."""
    ctx = _context()
    vision_data = ctx.vision_data
    assert vision_data is not None

    packed = pack_uncached_images([(ctx, ctx.images)], [CPU()])
    assert packed is not None

    np.testing.assert_array_equal(
        packed.pixel_values.to_numpy(),
        vision_data.concatenated_pixel_values,
    )
    # The bilinear weights are computed in float64 and narrowed on the way
    # to the graph's float32 input, exactly as the batch-level pack did.
    np.testing.assert_array_equal(
        packed.weights.to_numpy(), vision_data.weights.astype(np.float32)
    )
    np.testing.assert_array_equal(
        packed.indices.to_numpy(), vision_data.indices
    )
    np.testing.assert_array_equal(
        packed.vision_position_ids.to_numpy(),
        vision_data.vision_position_ids,
    )
    np.testing.assert_array_equal(
        packed.grid_thw.to_numpy(), vision_data.image_grid_thw
    )
    np.testing.assert_array_equal(
        packed.cu_seqlens.to_numpy(), vision_data.cu_seqlens
    )
    assert packed.max_seqlen.to_numpy()[0] == vision_data.max_seqlen
    assert packed.max_grid_size.to_numpy() == vision_data.max_grid_size


def test_packing_one_image_takes_only_its_slice() -> None:
    """A partial cache hit must re-encode only the image it missed."""
    ctx = _context()
    vision_data = ctx.vision_data
    assert vision_data is not None
    _, h0, w0 = GRIDS[0]
    _, h1, w1 = GRIDS[1]

    packed = pack_uncached_images([(ctx, [ctx.images[1]])], [CPU()])
    assert packed is not None

    np.testing.assert_array_equal(
        packed.pixel_values.to_numpy(),
        vision_data.concatenated_pixel_values[h0 * w0 :],
    )
    np.testing.assert_array_equal(
        packed.weights.to_numpy(),
        vision_data.weights[:, h0 * w0 :, :].astype(np.float32),
    )
    np.testing.assert_array_equal(
        packed.indices.to_numpy(), vision_data.indices[:, h0 * w0 :]
    )
    np.testing.assert_array_equal(
        packed.vision_position_ids.to_numpy(),
        vision_data.vision_position_ids[h0 * w0 :],
    )
    np.testing.assert_array_equal(
        packed.cu_seqlens.to_numpy(), np.array([0, h1 * w1], dtype=np.uint32)
    )
    assert packed.max_grid_size.to_numpy() == max(h1, w1)


def test_packing_nothing_selected_is_none() -> None:
    assert pack_uncached_images([], [CPU()]) is None


def test_packing_rejects_a_foreign_image() -> None:
    ctx = _context()
    stranger = ImageMetadata(
        start_idx=0,
        end_idx=1,
        pixel_values=np.zeros((1, PATCH_DIM), np.float32),
    )
    with pytest.raises(
        ValueError, match=re.escape("not present in ctx.images")
    ):
        pack_uncached_images([(ctx, [stranger])], [CPU()])
