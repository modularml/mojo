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
from max.pipelines.architectures.qwen2_5vl.nn.data_processing import (
    get_rope_index,
    mrope_pos_ids_3d,
    mrope_pos_ids_3d_inner,
)
from utils.config_loader import ConfigNames, get_config_loader


def test_mrope_pos_ids_3d() -> None:
    grid_thw = np.array([[1, 20, 20], [1, 20, 20], [1, 20, 20]], dtype=np.int64)
    spatial_merge_size: int = 10

    pos_ids = mrope_pos_ids_3d(
        grid_thw=grid_thw, spatial_merge_size=spatial_merge_size
    )
    assert len(pos_ids) == 3 * 20 * 20
    # First call should result in cache miss
    assert mrope_pos_ids_3d_inner.cache_info().hits == 2
    assert mrope_pos_ids_3d_inner.cache_info().misses == 1
    assert mrope_pos_ids_3d_inner.cache_info().currsize == 1

    _ = mrope_pos_ids_3d(
        grid_thw=grid_thw, spatial_merge_size=spatial_merge_size + 10
    )
    assert mrope_pos_ids_3d_inner.cache_info().hits == 4
    assert mrope_pos_ids_3d_inner.cache_info().misses == 2
    assert mrope_pos_ids_3d_inner.cache_info().currsize == 2


def get_rope_index_torch(
    input_ids: torch.LongTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    config: dict | None = None,  # type: ignore[type-arg]
) -> tuple[torch.Tensor, torch.Tensor]:
    assert config is not None
    vision_config = config["vision_config"]

    spatial_merge_size = vision_config["spatial_merge_size"]
    image_token_id = config["image_token_id"]
    video_token_id = config["video_token_id"]
    vision_start_token_id = config["vision_start_token_id"]
    tokens_per_second = vision_config["tokens_per_second"]

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
            second_per_grid_t: float = 0.0
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
                    assert image_grid_thw is not None
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0.0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    assert video_grid_thw is not None
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
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

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(
                    -1, llm_grid_h * llm_grid_w
                )

                ## normalize type, send to device.
                second_per_grid_t = torch.as_tensor(
                    second_per_grid_t,
                    dtype=range_tensor.dtype,
                    device=range_tensor.device,
                )

                time_tensor = (
                    expanded_range * second_per_grid_t * tokens_per_second
                )

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

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
            assert input_ids is not None
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


@pytest.mark.parametrize(
    "config_name",
    [
        pytest.param(ConfigNames.QWEN2_5VL_3B),
    ],
)
@pytest.mark.parametrize(
    "image_grid_thw,video_grid_thw,second_per_grid_ts,test_name",
    [
        (
            torch.tensor([[1, 2, 2]]),
            torch.tensor([[1, 1, 1]]),
            torch.tensor([1.0]),
            "both_image_and_video_with_time",
        ),
        (torch.tensor([[1, 2, 2]]), None, None, "image_only"),
        (
            None,
            torch.tensor([[1, 1, 1]]),
            torch.tensor([1.0]),
            "video_only_with_time",
        ),
        (None, torch.tensor([[1, 1, 1]]), None, "video_only_without_time"),
        (None, None, None, "text_only"),
    ],
)
def test_get_rope_index(
    config_name: ConfigNames,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    second_per_grid_ts: torch.Tensor | None,
    test_name: str,
) -> None:
    """Test get_rope_index function with basic text-only input."""

    inputs = {
        "input_ids": torch.randint(0, 1000, (1, 10)),
        "attention_mask": torch.ones(1, 10),
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": video_grid_thw,
        "second_per_grid_ts": second_per_grid_ts,
    }

    input_ids_torch = inputs["input_ids"]
    attention_mask_torch = inputs["attention_mask"]
    image_grid_thw_torch = inputs.get("image_grid_thw")
    video_grid_thw_torch = inputs.get("video_grid_thw")
    second_per_grid_ts_torch = inputs.get("second_per_grid_ts")

    input_ids_numpy = input_ids_torch.numpy()
    attention_mask_numpy = attention_mask_torch.numpy()
    image_grid_thw_numpy = (
        image_grid_thw_torch.numpy()
        if image_grid_thw_torch is not None
        else None
    )
    video_grid_thw_numpy = (
        video_grid_thw_torch.numpy()
        if video_grid_thw_torch is not None
        else None
    )
    second_per_grid_ts_numpy = (
        second_per_grid_ts_torch.numpy()
        if second_per_grid_ts_torch is not None
        else None
    )

    loader = get_config_loader()
    qwen2_5vl_config = loader.create_qwen2_5vl_config(config_name)
    vision_config = qwen2_5vl_config["vision_config"]

    # Call NumPy implementation
    position_ids_numpy, mrope_deltas_numpy = get_rope_index(
        spatial_merge_size=vision_config["spatial_merge_size"],
        image_token_id=qwen2_5vl_config["image_token_id"],
        video_token_id=qwen2_5vl_config["video_token_id"],
        vision_start_token_id=qwen2_5vl_config["vision_start_token_id"],
        tokens_per_second=vision_config["tokens_per_second"],
        input_ids=input_ids_numpy,
        attention_mask=attention_mask_numpy,
        image_grid_thw=image_grid_thw_numpy,
        video_grid_thw=video_grid_thw_numpy,
        second_per_grid_ts=second_per_grid_ts_numpy,
    )

    # Call PyTorch implementation
    position_ids_torch, mrope_deltas_torch = get_rope_index_torch(
        input_ids=input_ids_torch,
        attention_mask=attention_mask_torch,
        image_grid_thw=image_grid_thw_torch,
        video_grid_thw=video_grid_thw_torch,
        second_per_grid_ts=second_per_grid_ts_torch,
        config=qwen2_5vl_config,
    )

    # Convert torch results to numpy for comparison
    position_ids_torch_numpy = position_ids_torch.numpy()
    mrope_deltas_torch_numpy = mrope_deltas_torch.numpy()

    # Compare results
    assert np.allclose(
        position_ids_numpy, position_ids_torch_numpy, rtol=1e-6, atol=1e-8
    ), "Position IDs don't match between NumPy and PyTorch implementations"

    assert np.allclose(
        mrope_deltas_numpy, mrope_deltas_torch_numpy, rtol=1e-6, atol=1e-8
    ), (
        "MRoPE position deltas don't match between NumPy and PyTorch implementations"
    )


if __name__ == "__main__":
    pytest.main([__file__])
