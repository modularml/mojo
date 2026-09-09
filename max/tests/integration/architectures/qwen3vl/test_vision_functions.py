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


import numpy as np
import pytest
import torch
import torch.nn.functional as F
from max.pipelines.architectures.qwen2_5vl.nn.data_processing import (
    mrope_pos_ids_3d,
    mrope_pos_ids_3d_inner,
)
from max.pipelines.architectures.qwen3vl_moe.nn.data_processing import (
    get_bilinear_interpolation_weights_and_indices,
    get_seqlens,
)
from max.pipelines.architectures.qwen3vl_moe.nn.data_processing import (
    get_rope_index as get_rope_index_qwen3vl_np,
)
from utils.config_loader import ConfigNames, get_config_loader


def get_rope_index_torch(
    input_ids: torch.Tensor,
    video_grid_thw: torch.Tensor,
    image_grid_thw: torch.Tensor,
    attention_mask: torch.Tensor,
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """copied from the original implementation of Qwen3VLMoeForConditionalGeneration in transformers.

    Original implementation from:
    https://github.com/huggingface/transformers/blob/71db0d49e99884566026c140f8b12b61056fa8dc/src/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py#L1080
    """

    # Since we use timestamps to separate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw, video_grid_thw[:, 0], dim=0
        )
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):  # noqa: PLR1704 (FIXME)
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []  # type: ignore[type-arg]
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1
                    if len(llm_pos_ids_list) > 0
                    else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                t_index = (
                    torch.arange(llm_grid_t)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .flatten()
                )
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1
                    if len(llm_pos_ids_list) > 0
                    else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0)
                .expand(3, -1, -1)
                .to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = (
                max_position_ids + 1 - attention_mask.shape[-1]
            )
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def get_mrope_pos_ids_3d_torch(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    """copied from the original implementation of get_mrope_pos_ids_3d in vllm.

    Original implementation from:
    https://github.com/vllm-project/vllm/blob/9fce7bee745230d61c60ad467966790553b0ba48/vllm/model_executor/models/qwen3_vl.py#L409
    """
    merge_size = spatial_merge_size

    total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
    pos_ids = torch.empty(
        (total_tokens, 2), dtype=torch.long, device=grid_thw.device
    )

    offset = 0
    for num_frames, height, width in grid_thw:
        merged_h, merged_w = height // merge_size, width // merge_size

        block_rows = torch.arange(
            merged_h, device=grid_thw.device
        )  # block row indices
        block_cols = torch.arange(
            merged_w, device=grid_thw.device
        )  # block col indices
        intra_row = torch.arange(
            merge_size, device=grid_thw.device
        )  # intra-block row offsets
        intra_col = torch.arange(
            merge_size, device=grid_thw.device
        )  # intra-block col offsets

        # Compute full-resolution positions
        row_idx = (
            block_rows[:, None, None, None] * merge_size
            + intra_row[None, None, :, None]
        )
        col_idx = (
            block_cols[None, :, None, None] * merge_size
            + intra_col[None, None, None, :]
        )

        row_idx = row_idx.expand(
            merged_h, merged_w, merge_size, merge_size
        ).reshape(-1)
        col_idx = col_idx.expand(
            merged_h, merged_w, merge_size, merge_size
        ).reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)

        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)

        num_tokens = coords.shape[0]
        pos_ids[offset : offset + num_tokens] = coords
        offset += num_tokens

    return pos_ids


def get_mrope_pos_ids_3d_vllm(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    """VLLM implementation of mrope_pos_ids_3d for comparison."""
    pos_ids = []
    # Support both Tensor and list inputs for DP path
    for t, h, w in grid_thw:
        hpos_ids = torch.arange(h, dtype=torch.long).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = torch.arange(w, dtype=torch.long).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    pos_ids = torch.cat(pos_ids, dim=0)
    return pos_ids


def get_bilinear_interpolation_weights_and_indices_torch(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """copied from the original implementation of get_bilinear_interpolation_weights_and_indices in vllm.

    Original implementation from:
    https://github.com/vllm-project/vllm/blob/9fce7bee745230d61c60ad467966790553b0ba48/vllm/model_executor/models/qwen3_vl.py#L444
    """
    weights_list = []
    indices_list = []
    for _, h, w in grid_thw:
        h_idxs = torch.linspace(
            0, num_grid_per_side - 1, h, dtype=torch.float32
        )
        w_idxs = torch.linspace(
            0, num_grid_per_side - 1, w, dtype=torch.float32
        )

        h_floor = h_idxs.to(torch.long)
        w_floor = w_idxs.to(torch.long)
        h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
        w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

        dh = h_idxs - h_floor
        dw = w_idxs - w_floor

        # Create meshgrid view for all h, w vars
        dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
        h_floor_grid, w_floor_grid = torch.meshgrid(
            h_floor, w_floor, indexing="ij"
        )
        h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")
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

        indices = torch.stack([idx00, idx01, idx10, idx11], dim=0).reshape(
            4, -1
        )
        weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
        weights = weights.to(dtype=dtype)
        indices_list.append(indices)
        weights_list.append(weights)

    return torch.cat(indices_list, dim=1), torch.cat(weights_list, dim=1)


def get_seqlens_torch(
    grid_thw: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Torch implementation of get_seqlens for testing purposes.
    Args:
        grid_thw: Buffer of shape [n_images, 3] with (t, h, w) for each image/video

    Returns:
        Tuple of (cu_seqlens, max_seqlen)
    """
    # Calculate repeated sizes: repeat h*w for each frame (t times)
    repeated_sizes = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    )

    # Calculate cumulative sum with proper dtype handling
    cu_seqlens = repeated_sizes.cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )

    # Pad with zero at the beginning
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # Calculate max sequence length
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

    return cu_seqlens, max_seqlen


def test_mrope_pos_ids_3d() -> None:
    """Test mrope_pos_ids_3d implementation comparing numpy, torch, and vllm versions."""
    # Load configuration
    loader = get_config_loader()
    qwen3vl_config = loader.load_config(ConfigNames.QWEN3VL_30B)
    vision_config = qwen3vl_config["vision_config"]

    # Test cases with different grid configurations
    test_cases = [
        # Single frame, small grid
        np.array([[1, 4, 4]], dtype=np.int64),
        # Single frame, larger grid
        np.array([[1, 8, 8]], dtype=np.int64),
        # Multiple frames
        np.array([[2, 4, 4], [1, 6, 6], [3, 4, 4], [3, 6, 6]], dtype=np.int64),
        # Real-world example
        np.array([[1, 98, 146], [1, 76, 114], [1, 76, 114]], dtype=np.int64),
    ]

    spatial_merge_size = vision_config["spatial_merge_size"]

    expected_cache_hits = [0, 0, 3, 4]
    for i, grid_thw in enumerate(test_cases):
        # Convert to torch tensor for torch implementations
        grid_thw_torch = torch.from_numpy(grid_thw)

        pos_ids_numpy = mrope_pos_ids_3d(
            grid_thw=grid_thw, spatial_merge_size=spatial_merge_size
        )
        assert (
            mrope_pos_ids_3d_inner.cache_info().hits == expected_cache_hits[i]
        )

        pos_ids_torch = get_mrope_pos_ids_3d_torch(
            grid_thw_torch, spatial_merge_size
        )
        pos_ids_vllm = get_mrope_pos_ids_3d_vllm(
            grid_thw_torch, spatial_merge_size
        )

        # Convert torch results to numpy for comparison
        pos_ids_torch_np = pos_ids_torch.numpy()
        pos_ids_vllm_np = pos_ids_vllm.numpy()

        # Verify all implementations produce the same shape
        assert pos_ids_numpy.shape == pos_ids_torch_np.shape, (
            f"Shape mismatch: numpy {pos_ids_numpy.shape} vs torch {pos_ids_torch_np.shape}"
        )
        assert pos_ids_numpy.shape == pos_ids_vllm_np.shape, (
            f"Shape mismatch: numpy {pos_ids_numpy.shape} vs vllm {pos_ids_vllm_np.shape}"
        )

        # Verify all implementations produce identical results
        assert np.array_equal(pos_ids_torch_np, pos_ids_vllm_np), (
            "Torch and vllm implementations differ"
        )
        assert np.array_equal(pos_ids_numpy, pos_ids_torch_np), (
            "Numpy and torch implementations differ"
        )
        assert np.array_equal(pos_ids_numpy, pos_ids_vllm_np), (
            "Numpy and vllm implementations differ"
        )

        # Verify the structure of the results
        # Each position should have 2 coordinates (height, width)
        assert pos_ids_numpy.shape[1] == 2, (
            f"Expected 2 coordinates per position, got {pos_ids_numpy.shape[1]}"
        )


def test_get_rope_index() -> None:
    """Test get_rope_index comparing torch (HF behavior) and Qwen3VL NumPy version."""
    # Load configuration
    loader = get_config_loader()
    qwen3vl_config = loader.load_config(ConfigNames.QWEN3VL_30B)
    vision_config = qwen3vl_config["vision_config"]

    # Test cases with different configurations using config values
    test_cases = [
        {
            "name": "Simple image case",
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
            "image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long),
            "video_grid_thw": None,
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "spatial_merge_size": vision_config["spatial_merge_size"],
            "image_token_id": qwen3vl_config["image_token_id"],
            "video_token_id": qwen3vl_config["video_token_id"],
            "vision_start_token_id": qwen3vl_config["vision_start_token_id"],
            "second_per_grid_ts": None,
        },
        {
            "name": "Image and video case",
            "input_ids": torch.tensor(
                [[1, 2, 3, 2, 4, 5, 6]], dtype=torch.long
            ),
            "image_grid_thw": torch.tensor([[1, 6, 6]], dtype=torch.long),
            "video_grid_thw": torch.tensor([[2, 4, 4]], dtype=torch.long),
            "attention_mask": torch.ones(1, 7, dtype=torch.long),
            "spatial_merge_size": vision_config["spatial_merge_size"],
            "image_token_id": qwen3vl_config["image_token_id"],
            "video_token_id": qwen3vl_config["video_token_id"],
            "vision_start_token_id": qwen3vl_config["vision_start_token_id"],
            "second_per_grid_ts": torch.tensor([0.5], dtype=torch.float32),
        },
        {
            "name": "Multiple images case",
            "input_ids": torch.tensor(
                [[1, 2, 3, 2, 3, 5, 6]], dtype=torch.long
            ),
            "image_grid_thw": torch.tensor(
                [[1, 4, 4], [1, 6, 6]], dtype=torch.long
            ),
            "video_grid_thw": None,
            "attention_mask": torch.ones(1, 7, dtype=torch.long),
            "spatial_merge_size": vision_config["spatial_merge_size"],
            "image_token_id": qwen3vl_config["image_token_id"],
            "video_token_id": qwen3vl_config["video_token_id"],
            "vision_start_token_id": qwen3vl_config["vision_start_token_id"],
            "second_per_grid_ts": None,
        },
    ]

    for test_case in test_cases:
        # Convert torch inputs to numpy for Qwen2.5VL implementation
        input_ids_numpy = test_case["input_ids"].numpy()
        attention_mask_numpy = test_case["attention_mask"].numpy()
        image_grid_thw_numpy = (
            test_case["image_grid_thw"].numpy()
            if test_case["image_grid_thw"] is not None
            else None
        )
        video_grid_thw_numpy = (
            test_case["video_grid_thw"].numpy()
            if test_case["video_grid_thw"] is not None
            else None
        )
        second_per_grid_ts_numpy = (
            test_case["second_per_grid_ts"].numpy()
            if test_case["second_per_grid_ts"] is not None
            else None
        )

        # Get results from both implementations
        # Qwen3VL NumPy implementation (timestamp semantics)
        pos_ids_np, mrope_deltas_np = get_rope_index_qwen3vl_np(
            spatial_merge_size=test_case["spatial_merge_size"],
            image_token_id=test_case["image_token_id"],
            video_token_id=test_case["video_token_id"],
            vision_start_token_id=test_case["vision_start_token_id"],
            input_ids=input_ids_numpy,
            image_grid_thw=image_grid_thw_numpy,
            video_grid_thw=video_grid_thw_numpy,
            second_per_grid_ts=second_per_grid_ts_numpy,
            attention_mask=attention_mask_numpy,
        )

        # Torch implementation
        pos_ids_torch, mrope_deltas_torch = get_rope_index_torch(
            input_ids=test_case["input_ids"],
            video_grid_thw=test_case["video_grid_thw"],
            image_grid_thw=test_case["image_grid_thw"],
            attention_mask=test_case["attention_mask"],
            spatial_merge_size=test_case["spatial_merge_size"],
            image_token_id=test_case["image_token_id"],
            video_token_id=test_case["video_token_id"],
            vision_start_token_id=test_case["vision_start_token_id"],
        )

        # Convert torch results to numpy for comparison
        pos_ids_torch_np = pos_ids_torch.numpy()
        mrope_deltas_torch_np = mrope_deltas_torch.numpy()

        # Verify shapes match
        assert pos_ids_np.shape == pos_ids_torch_np.shape, (
            f"Position IDs shape mismatch: NumPy {pos_ids_np.shape} vs Torch {pos_ids_torch_np.shape}"
        )
        assert mrope_deltas_np.shape == mrope_deltas_torch_np.shape, (
            f"MRoPE deltas shape mismatch: NumPy {mrope_deltas_np.shape} vs Torch {mrope_deltas_torch_np.shape}"
        )

        # Verify values match (allowing for small numerical differences)
        assert np.allclose(
            pos_ids_np, pos_ids_torch_np, rtol=1e-5, atol=1e-5
        ), "Position IDs values differ between NumPy and Torch implementations"
        assert np.allclose(
            mrope_deltas_np, mrope_deltas_torch_np, rtol=1e-5, atol=1e-5
        ), "MRoPE deltas values differ between NumPy and Torch implementations"
        # Verify the structure of the results
        # Position IDs should have shape (3, batch_size, seq_len) for 3D RoPE
        assert pos_ids_np.shape[0] == 3, (
            f"Expected 3 dimensions for RoPE, got {pos_ids_np.shape[0]}"
        )
        assert pos_ids_np.shape[1] == test_case["input_ids"].shape[0], (
            "Batch size mismatch"
        )
        assert pos_ids_np.shape[2] == test_case["input_ids"].shape[1], (
            "Sequence length mismatch"
        )

        # MRoPE deltas should have shape (batch_size, 1)
        assert mrope_deltas_np.shape[1] == 1, (
            f"Expected 1 column for MRoPE deltas, got {mrope_deltas_np.shape[1]}"
        )
        assert mrope_deltas_np.shape[0] == test_case["input_ids"].shape[0], (
            "Batch size mismatch in MRoPE deltas"
        )


def test_get_bilinear_interpolation_weights_and_indices() -> None:
    # Load configuration
    loader = get_config_loader()
    qwen3vl_config = loader.load_config(ConfigNames.QWEN3VL_30B)
    vision_config = qwen3vl_config["vision_config"]

    grid_thw = np.array([[1, 98, 146], [1, 76, 114]], dtype=np.int64)
    num_position_embeddings = vision_config["num_position_embeddings"]
    num_grid_per_side = int(num_position_embeddings**0.5)
    dtype = torch.float64

    indices_torch, weights_torch = (
        get_bilinear_interpolation_weights_and_indices_torch(
            grid_thw, num_grid_per_side, dtype
        )
    )

    indices_numpy, weights_numpy = (
        get_bilinear_interpolation_weights_and_indices(
            grid_thw, num_grid_per_side
        )
    )

    # Convert numpy window_index to torch tensor for comparison
    weights_numpy_torch = torch.from_numpy(weights_numpy)

    # Compare window indices (convert to same dtype first)
    assert np.array_equal(
        indices_torch,
        indices_numpy,
    )
    assert torch.allclose(
        weights_torch.float(),
        weights_numpy_torch.float(),
        rtol=1e-4,
        atol=1e-4,
    )


def test_get_seqlens() -> None:
    """Test get_seqlens implementation comparing numpy and torch versions."""
    # Test cases with different grid configurations
    test_cases = [
        # Single frame, small grid
        np.array([[1, 4, 4]], dtype=np.int32),
        # Single frame, larger grid
        np.array([[1, 8, 8]], dtype=np.int32),
        # Multiple frames
        np.array([[2, 4, 4], [1, 6, 6]], dtype=np.int32),
        # Real-world example
        np.array([[1, 98, 146], [1, 76, 114]], dtype=np.int32),
        # Multiple videos with different frame counts
        np.array([[3, 4, 4], [2, 6, 6], [1, 8, 8]], dtype=np.int32),
        # Edge case: single pixel
        np.array([[1, 1, 1]], dtype=np.int32),
        # Edge case: very large grid
        np.array([[1, 256, 256]], dtype=np.int32),
    ]

    for grid_thw in test_cases:
        # Convert to torch tensor for torch implementation
        grid_thw_torch = torch.from_numpy(grid_thw)

        # Get results from both implementations
        cu_seqlens_numpy, max_seqlen_numpy = get_seqlens(grid_thw)
        cu_seqlens_torch, max_seqlen_torch = get_seqlens_torch(grid_thw_torch)

        # Convert torch results to numpy for comparison
        cu_seqlens_torch_np = cu_seqlens_torch.numpy()

        # Verify shapes match
        assert cu_seqlens_numpy.shape == cu_seqlens_torch_np.shape, (
            f"Shape mismatch: numpy {cu_seqlens_numpy.shape} vs torch {cu_seqlens_torch_np.shape}"
        )

        # Verify values match exactly
        assert np.array_equal(cu_seqlens_numpy, cu_seqlens_torch_np), (
            "cu_seqlens values differ between numpy and torch implementations"
        )

        # Verify max_seqlen matches
        assert max_seqlen_numpy == max_seqlen_torch, (
            f"max_seqlen mismatch: numpy {max_seqlen_numpy} vs torch {max_seqlen_torch}"
        )


if __name__ == "__main__":
    pytest.main([__file__])
