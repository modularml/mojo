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

from typing import Any

import numpy as np
import numpy.typing as npt
from max.profiler import traced

QWEN3VL_MAX_PIXELS = 16777216
"""Qwen3VL's default ``longest_edge``: the post-resize pixel ceiling per image.

The process-wide resolution bound, so it is also the image hash's size tier and
the largest image an arch config can be asked to size a cache entry for.
"""


@traced
def get_rope_index(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    input_ids: npt.NDArray[np.integer[Any]],
    image_grid_thw: npt.NDArray[np.integer[Any]] | None = None,
    video_grid_thw: npt.NDArray[np.integer[Any]] | None = None,
    second_per_grid_ts: npt.NDArray[np.floating[Any]] | None = None,
    attention_mask: npt.NDArray[np.floating[Any]] | None = None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """NumPy implementation of Qwen3-VL MoE get_rope_index.

    Matches Transformers' Qwen3-VL behavior:
    - Uses timestamp semantics for videos: collapse temporal grid to 1 per frame.
    - Produces position_ids of shape (3, batch, seq_len) and mrope deltas (batch, 1).

    For images, temporal extent is 1. For videos, each original (t,h,w) is expanded to
    t frames of (1,h,w) conceptually (as in repeat_interleave + set t=1), making the
    temporal RoPE index effectively 0 for all vision tokens.
    """

    # Prepare attention mask
    total_input_ids = input_ids
    if attention_mask is None:
        attention_mask = np.ones_like(total_input_ids)

    # Adjust video_grid_thw to per-frame rows with t=1 (timestamp semantics)
    adjusted_video_grid: npt.NDArray[np.integer[Any]] | None = None
    if video_grid_thw is not None:
        # Repeat each row by its temporal dimension, then set t=1
        repeats = video_grid_thw[:, 0].astype(np.int64)
        expanded = np.repeat(video_grid_thw, repeats, axis=0).copy()
        expanded[:, 0] = 1  # force temporal to 1 per frame
        adjusted_video_grid = expanded.astype(np.int64)

    # Initialize outputs
    position_ids = np.ones(
        (3, total_input_ids.shape[0], total_input_ids.shape[1]), dtype=np.int64
    )
    mrope_position_deltas: list[int] = []

    if image_grid_thw is not None or adjusted_video_grid is not None:
        image_index, video_index = 0, 0
        for i, input_ids_row in enumerate(total_input_ids):
            valid_mask = attention_mask[i] == 1
            row_tokens = input_ids_row[valid_mask]

            # Count vision blocks after each vision_start token
            vision_start_indices = np.where(
                row_tokens == vision_start_token_id
            )[0]
            if vision_start_indices.size > 0:
                vision_tokens = row_tokens[vision_start_indices + 1]
                image_nums = int(np.sum(vision_tokens == image_token_id))
                video_nums = int(np.sum(vision_tokens == video_token_id))
            else:
                image_nums = 0
                video_nums = 0

            input_list = row_tokens.tolist()
            llm_pos_ids_list: list[npt.NDArray[np.int64]] = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            # Interleave text chunks and vision chunks
            for _ in range(image_nums + video_nums):
                if image_token_id in input_list and remain_images > 0:
                    ed_image = input_list.index(image_token_id, st)
                else:
                    ed_image = len(input_list) + 1

                if video_token_id in input_list and remain_videos > 0:
                    ed_video = input_list.index(video_token_id, st)
                else:
                    ed_video = len(input_list) + 1

                if ed_image < ed_video:
                    # Image block: t assumed 1 in practice
                    assert image_grid_thw is not None
                    _, h, w = image_grid_thw[image_index]
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    # Video block: use adjusted per-frame grid with t=1
                    assert adjusted_video_grid is not None
                    _, h, w = adjusted_video_grid[video_index]
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                # Accumulate text positions up to the start of vision block
                llm_grid_t = 1
                llm_grid_h = int(h) // spatial_merge_size
                llm_grid_w = int(w) // spatial_merge_size
                text_len = ed - st
                st_idx = int(
                    (llm_pos_ids_list[-1].max() + 1) if llm_pos_ids_list else 0
                )

                # Text portion positions (3 x text_len), same values for all 3 dims
                if text_len > 0:
                    text_pos = np.arange(text_len, dtype=np.int64).reshape(
                        1, -1
                    )
                    text_pos_expanded = np.repeat(text_pos, 3, axis=0) + st_idx
                    llm_pos_ids_list.append(text_pos_expanded)

                # Vision grid positions: temporal zeros, spatial h/w indices
                # t_index is zeros since llm_grid_t == 1
                num_vision_tokens = llm_grid_t * llm_grid_h * llm_grid_w
                # Pre-allocate array and assign directly (more efficient than vstack)
                vision_pos = np.zeros((3, num_vision_tokens), dtype=np.int64)
                vision_pos[1] = np.tile(
                    np.arange(llm_grid_h, dtype=np.int64).reshape(1, -1, 1),
                    (llm_grid_t, 1, llm_grid_w),
                ).flatten()
                vision_pos[2] = np.tile(
                    np.arange(llm_grid_w, dtype=np.int64).reshape(1, 1, -1),
                    (llm_grid_t, llm_grid_h, 1),
                ).flatten()
                vision_pos += text_len + st_idx
                llm_pos_ids_list.append(vision_pos)

                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            # Tail text after last vision block
            if st < len(input_list):
                st_idx = int(
                    (llm_pos_ids_list[-1].max() + 1) if llm_pos_ids_list else 0
                )
                text_len = len(input_list) - st
                text_pos = np.arange(text_len, dtype=np.int64).reshape(1, -1)
                text_pos_expanded = np.repeat(text_pos, 3, axis=0) + st_idx
                llm_pos_ids_list.append(text_pos_expanded)

            # Assemble and write positions for valid tokens
            llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(
                3, -1
            )
            position_ids[:, i, valid_mask] = llm_positions
            mrope_position_deltas.append(
                int(llm_positions.max() + 1 - len(input_ids_row))
            )

        mrope_position_deltas_array = np.array(
            mrope_position_deltas, dtype=np.int64
        ).reshape(-1, 1)
        return position_ids, mrope_position_deltas_array

    # Fallback: no vision grids provided
    if attention_mask is not None:
        base = (np.cumsum(attention_mask, axis=-1) - 1).astype(np.int64)
        base[attention_mask == 0] = 1
        position_ids = np.tile(base[np.newaxis, ...], (3, 1, 1)).astype(
            np.int64
        )
        max_position_ids = position_ids.max(axis=(0, -1))
        mrope_position_deltas_array = (
            (max_position_ids + 1 - attention_mask.shape[-1])
            .reshape(-1, 1)
            .astype(np.int64)
        )
    else:
        position_ids = np.tile(
            np.arange(input_ids.shape[1], dtype=np.int64)[
                np.newaxis, np.newaxis, :
            ],
            (3, input_ids.shape[0], 1),
        )
        mrope_position_deltas_array = np.zeros(
            (input_ids.shape[0], 1), dtype=np.int64
        )

    return position_ids, mrope_position_deltas_array


@traced
def get_bilinear_interpolation_weights_and_indices(
    grid_thw: npt.NDArray[np.integer[Any]],
    num_grid_per_side: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """Calculate the bilinear interpolation weights and indices from the offsets
    of patches in the original pixel values.
    Converted the original implementation from torch to numpy. Original implementation from:
    https://github.com/vllm-project/vllm/blob/9fce7bee745230d61c60ad467966790553b0ba48/vllm/model_executor/models/qwen3_vl.py#L444

    Returns a tuple (indices, weights).

    indices : np.ndarray of shape (4, N), dtype=int64
        Indices of the four bilinear neighbors for each patch position, where N is the total number of patch positions.
        The first axis (size 4) corresponds to the four neighbors: (top-left, top-right, bottom-left, bottom-right).
    weights : np.ndarray of shape (4, N, 1), dtype=float32
        Bilinear interpolation weights for each neighbor and patch position.
        The first axis (size 4) matches the order of indices.
    """
    indices_list = []
    weights_list = []
    for _, h, w in grid_thw:
        h_idxs = np.linspace(0, num_grid_per_side - 1, h, dtype=np.float32)
        w_idxs = np.linspace(0, num_grid_per_side - 1, w, dtype=np.float32)

        h_floor = h_idxs.astype(np.int64)
        w_floor = w_idxs.astype(np.int64)
        h_ceil = np.clip(h_floor + 1, 0, num_grid_per_side - 1)
        w_ceil = np.clip(w_floor + 1, 0, num_grid_per_side - 1)

        dh = h_idxs - h_floor
        dw = w_idxs - w_floor

        # Create meshgrid view for all h, w vars
        dh_grid, dw_grid = np.meshgrid(dh, dw, indexing="ij")
        h_floor_grid, w_floor_grid = np.meshgrid(
            h_floor, w_floor, indexing="ij"
        )
        h_ceil_grid, w_ceil_grid = np.meshgrid(h_ceil, w_ceil, indexing="ij")
        h_floor_grid_idx = h_floor_grid * num_grid_per_side
        h_ceil_grid_idx = h_ceil_grid * num_grid_per_side

        # original computation of weights
        # w00 = (1 - dh_grid) * (1 - dw_grid)
        # w01 = (1 - dh_grid) * dw_grid
        # w10 = dh_grid * (1 - dw_grid)
        # w11 = dh_grid * dw_grid
        # we reuse w11 here to avoid duplicate
        # dh_grid * dw_grid computation
        w11 = dh_grid * dw_grid
        w10 = dh_grid - w11
        w01 = dw_grid - w11
        w00 = 1 - dh_grid - dw_grid + w11

        idx00 = h_floor_grid_idx + w_floor_grid
        idx01 = h_floor_grid_idx + w_ceil_grid
        idx10 = h_ceil_grid_idx + w_floor_grid
        idx11 = h_ceil_grid_idx + w_ceil_grid

        indices = np.stack([idx00, idx01, idx10, idx11], axis=0).reshape(4, -1)
        weights = np.stack([w00, w01, w10, w11], axis=0).reshape(4, -1, 1)
        indices_list.append(indices)
        weights_list.append(weights)

    return np.concatenate(indices_list, axis=1), np.concatenate(
        weights_list, axis=1
    )


@traced
def get_seqlens(
    grid_thw: npt.NDArray[np.int32],
) -> tuple[
    npt.NDArray[np.uint32],
    np.uint32,
]:
    """Generate attention masks for visual tokens using seq_length and cu_seqlens.
    cu_seqlens is used for all blocks in Qwen3VL to implement full attention.

    Args:
        grid_thw: number of patches in spatial and temporal dims in images. Shape = [n_images, 3], dtype=int32

    Returns:
        Tuple of (cu_seqlens, max_seqlen) where:
        - cu_seqlens: Array of cumulative sequence lengths, dtype=uint32
        - max_seqlen: Maximum sequence length as a scalar, dtype=uint32
    """
    repeated_sizes = np.repeat(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    cu_seqlens = np.cumsum(repeated_sizes, dtype=np.uint32)

    cu_seqlens = np.pad(cu_seqlens, (1, 0), constant_values=0)
    max_seqlen = np.uint32(np.max(np.diff(cu_seqlens)))

    return cu_seqlens, max_seqlen
