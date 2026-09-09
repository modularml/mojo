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

"""Tests for Kimi K2.5 PatchEmbedding (MoonVision3dPatchEmbed).

Reference: nvidia/Kimi-K2.5-NVFP4 modeling_kimi_k25.py
- MoonVision3dPatchEmbed: proj (Conv2d) + pos_emb (Learnable2DInterpPosEmbDivided_fixed).
- Forward: x = proj(x).view(x.size(0), -1); x = pos_emb(x, grid_thws).

MAX PatchEmbedding implements the same proj + pos_emb via a custom Mojo
GPU kernel for learnable 2D interpolated positional embeddings.
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import TorchPatchEmbed, TorchPosEmb
from max.driver import CPU, Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Dim, Graph, TensorType
from max.pipelines.architectures.kimik2_5.layers.vision.patch_embedding import (
    Learnable2DInterpPosEmbDividedFixed,
    PatchEmbedding,
)
from torch import nn
from torch.utils.dlpack import from_dlpack

# Kimi K2.5 vision config values (from checkpoint / reference)
PATCH_SIZE = 14
IN_CHANNELS = 3
HIDDEN_SIZE = 1152
INIT_POS_EMB_HEIGHT = 64
INIT_POS_EMB_WIDTH = 64
INIT_POS_EMB_TIME = 4

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16
RTOL = 2e-2
ATOL = 2 * torch.finfo(TORCH_DTYPE).eps


def _generate_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.randn(shape) * (1.0 / math.sqrt(shape[-1]))).to(TORCH_DTYPE)


def _create_state_dict(has_bias: bool) -> dict[str, torch.Tensor]:
    """State dict keys match Module.raw_state_dict."""
    state: dict[str, torch.Tensor] = {
        "proj.weight": _generate_tensor(
            (HIDDEN_SIZE, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)
        ),
        "pos_emb.weight": _generate_tensor(
            (INIT_POS_EMB_HEIGHT, INIT_POS_EMB_WIDTH, HIDDEN_SIZE)
        ),
    }
    if has_bias:
        state["proj.bias"] = _generate_tensor((HIDDEN_SIZE,))
    return state


def _create_max_patch_embedding_layer(
    device: DeviceRef, has_bias: bool = True
) -> PatchEmbedding:
    """Build PatchEmbedding with config values (no JSON)."""
    return PatchEmbedding(
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        hidden_size=HIDDEN_SIZE,
        init_pos_emb_height=INIT_POS_EMB_HEIGHT,
        init_pos_emb_width=INIT_POS_EMB_WIDTH,
        init_pos_emb_time=INIT_POS_EMB_TIME,
        dtype=MAX_DTYPE,
        device=device,
        has_bias=has_bias,
    )


def _build_and_run_max_patch_embedding_layer(
    pixel_values: torch.Tensor,
    grid_thws: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    has_bias: bool,
    n_gpus: int = 0,
) -> torch.Tensor:
    """Build MAX graph, run PatchEmbedding, return output as torch tensor."""
    devices: list[Device] = (
        [Accelerator(i) for i in range(n_gpus)] if n_gpus > 0 else [CPU(0)]
    )
    device_ref = DeviceRef.GPU() if n_gpus > 0 else DeviceRef.CPU()

    layer = _create_max_patch_embedding_layer(device_ref, has_bias)
    layer.load_state_dict(state_dict)

    session = InferenceSession(devices=devices)

    pixel_type = TensorType(
        MAX_DTYPE,
        [Dim("n_patches"), *pixel_values.shape[1:]],
        device_ref,
    )
    grid_type = TensorType(DType.int64, [grid_thws.shape[0], 3], device_ref)

    with Graph(
        "kimik2_5_patch_embed_test",
        input_types=(pixel_type, grid_type),
    ) as graph:
        pv, grid = graph.inputs
        out = layer(pv.tensor, grid.tensor)
        graph.output(out)

    compiled = session.load(graph, weights_registry=layer.state_dict())
    device = devices[0]
    result = compiled.execute(
        Buffer.from_dlpack(pixel_values).to(device),
        Buffer.from_dlpack(grid_thws).to(device),
    )
    return from_dlpack(result[0])


def _assert_close(expected: torch.Tensor, actual: torch.Tensor) -> None:
    torch.testing.assert_close(
        expected.cpu().float(),
        actual.cpu().float(),
        rtol=RTOL,
        atol=ATOL,
    )


def _torch_full_patch_embed(
    pixel_values: torch.Tensor,
    grid_thws: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    has_bias: bool,
) -> torch.Tensor:
    """Full reference: MoonVision3dPatchEmbed (proj + pos_emb)."""
    model = TorchPatchEmbed(
        out_dim=HIDDEN_SIZE,
        in_dim=IN_CHANNELS,
        patch_size=PATCH_SIZE,
        pos_emb_height=INIT_POS_EMB_HEIGHT,
        pos_emb_width=INIT_POS_EMB_WIDTH,
        pos_emb_time=INIT_POS_EMB_TIME,
        has_bias=has_bias,
    ).to(dtype=TORCH_DTYPE)
    model.proj.weight = nn.Parameter(state_dict["proj.weight"])
    if has_bias:
        model.proj.bias = nn.Parameter(state_dict["proj.bias"])
    model.pos_emb.weight = nn.Parameter(state_dict["pos_emb.weight"])
    model.eval()
    with torch.no_grad():
        return model(pixel_values, grid_thws)


def _build_and_run_pos_emb(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    pos_emb_weight: torch.Tensor,
    height: int,
    width: int,
    dim: int,
    num_frames: int,
    dtype: DType,
) -> torch.Tensor:
    """Build a MAX graph with just Learnable2DInterpPosEmbDividedFixed and run it."""
    devices: list[Device] = [Accelerator(0)]
    device_ref = DeviceRef.GPU()

    layer = Learnable2DInterpPosEmbDividedFixed(
        height=height,
        width=width,
        dim=dim,
        num_frames=num_frames,
        dtype=dtype,
        device=device_ref,
    )
    layer.load_state_dict({"weight": pos_emb_weight})

    session = InferenceSession(devices=devices)

    x_type = TensorType(dtype, [Dim("total_patches"), x.shape[1]], device_ref)
    grid_type = TensorType(
        DType.int64, [Dim("n_grids"), grid_thws.shape[1]], device_ref
    )

    with Graph(
        "kimi2_5_pos_emb_test",
        input_types=(x_type, grid_type),
    ) as graph:
        x_in, grid_in = graph.inputs
        out = layer(x_in.tensor, grid_in.tensor)
        graph.output(out)

    compiled = session.load(graph, weights_registry=layer.state_dict())
    result = compiled.execute(
        Buffer.from_dlpack(x.cuda()),
        Buffer.from_dlpack(grid_thws.cuda()),
    )
    return from_dlpack(result[0])


def _run_torch_pos_emb(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    pos_emb_weight: torch.Tensor,
    height: int,
    width: int,
    dim: int,
    num_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """Run the torch reference Learnable2DInterpPosEmbDivided_fixed."""
    model = TorchPosEmb(
        height=height,
        width=width,
        num_frames=num_frames,
        dim=dim,
    ).to(device=device, dtype=x.dtype)
    model.weight = nn.Parameter(pos_emb_weight.to(device))
    model.eval()
    with torch.no_grad():
        return model(x.to(device), grid_thws.to(device))


@pytest.mark.parametrize(
    "grid_thws_list, description",
    [
        ([[1, 4, 4]], "no interp, single image, t=1"),
        ([[3, 4, 4]], "no interp, single video, t=3"),
        ([[1, 8, 6]], "bicubic interp, single image"),
        ([[1, 98, 148]], "realistic non-square single image"),
        ([[1, 4, 4], [2, 8, 6]], "mixed: no-interp image + interp video"),
    ],
    ids=[
        "no_interp_t1",
        "no_interp_t3",
        "bicubic",
        "realistic_non_square",
        "multi_mixed",
    ],
)
def test_pos_emb_matches_torch(
    grid_thws_list: list[list[int]], description: str
) -> None:
    """Learnable2DInterpPosEmbDividedFixed matches torch reference.

    Tests no-interpolation, bicubic interpolation, temporal embedding,
    and multi-image/video batches.
    """
    torch.manual_seed(42)
    height, width, dim, num_frames = 4, 4, 32, 4

    grid_thws = torch.tensor(grid_thws_list, dtype=torch.int64)
    total_patches = sum(t * h * w for t, h, w in grid_thws_list)

    pos_emb_weight = _generate_tensor((height, width, dim))
    x = _generate_tensor((total_patches, dim))

    device = torch.device("cuda")
    torch_out = _run_torch_pos_emb(
        x, grid_thws, pos_emb_weight, height, width, dim, num_frames, device
    )

    max_out = _build_and_run_pos_emb(
        x, grid_thws, pos_emb_weight, height, width, dim, num_frames, MAX_DTYPE
    )

    _assert_close(torch_out, max_out)
    assert max_out.shape == (total_patches, dim), (
        f"{description}: expected shape ({total_patches}, {dim}), "
        f"got {max_out.shape}"
    )


@pytest.mark.parametrize(
    "grid_t, grid_h, grid_w",
    [
        (1, 2, 2),
        (1, 98, 148),
    ],
    ids=["single_image_2x2", "single_image_98x148"],
)
def test_patch_embedding_full_matches_torch(
    grid_t: int, grid_h: int, grid_w: int
) -> None:
    """Compare full PatchEmbedding (proj + pos_emb) to torch MoonVision3dPatchEmbed."""
    n_patches = grid_t * grid_h * grid_w
    has_bias = True
    pixel_values = _generate_tensor(
        (n_patches, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    )
    grid_thws = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int64)
    state_dict = _create_state_dict(has_bias)

    torch_out = _torch_full_patch_embed(
        pixel_values, grid_thws, state_dict, has_bias
    )
    max_out = _build_and_run_max_patch_embedding_layer(
        pixel_values, grid_thws, state_dict, has_bias, n_gpus=1
    )

    _assert_close(torch_out, max_out)
    assert max_out.shape == (n_patches, HIDDEN_SIZE)
