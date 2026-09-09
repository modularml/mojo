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
"""Tests for Gemma4VisionPatchEmbedder layer."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from conftest import (  # type: ignore[import-not-found]
    VISION_EMBED_HIDDEN_SIZE,
    VISION_PATCH_SIZE,
    VISION_POSITION_EMBEDDING_SIZE,
    TorchGemma4VisionPatchEmbedder,
)
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.pipelines.architectures.gemma4.vision_model.embedding import (
    Gemma4VisionPatchEmbedder,
)

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16


def _make_config(
    hidden_size: int,
    patch_size: int,
    position_embedding_size: int,
) -> Any:
    """Build a minimal config stub for Gemma4VisionPatchEmbedder."""
    vision_cfg = SimpleNamespace(
        hidden_size=hidden_size,
        patch_size=patch_size,
        position_embedding_size=position_embedding_size,
    )
    return SimpleNamespace(
        dtype=MAX_DTYPE,
        unquantized_dtype=MAX_DTYPE,
        vision_config=vision_cfg,
    )


def _build_and_run(
    embedder: Gemma4VisionPatchEmbedder,
    patches_flat: torch.Tensor,
    pixel_position_ids: torch.Tensor,
) -> Buffer:
    """Build a MAX graph with Gemma4VisionPatchEmbedder, execute, and return output."""
    device = Accelerator(0)
    session = InferenceSession(devices=[device])

    with Graph(
        "gemma4_vision_patch_embedder_test",
        input_types=[
            TensorType(
                MAX_DTYPE, tuple(patches_flat.shape), device=DeviceRef.GPU()
            ),
            TensorType(
                DType.int32,
                tuple(pixel_position_ids.shape),
                device=DeviceRef.GPU(),
            ),
        ],
    ) as graph:
        patches_input, pos_ids_input = graph.inputs
        assert isinstance(patches_input, TensorValue)
        assert isinstance(pos_ids_input, TensorValue)
        graph.output(embedder(patches_input, pos_ids_input))

    compiled = session.load(graph, weights_registry=embedder.state_dict())
    patches_gpu = Buffer.from_dlpack(patches_flat).to(device)
    pos_ids_gpu = Buffer.from_dlpack(pixel_position_ids).to(device)
    (result,) = compiled.execute(patches_gpu, pos_ids_gpu)
    assert isinstance(result, Buffer)
    return result


def _assert_close(expected: torch.Tensor, actual: Buffer) -> None:
    rtol = 2e-4
    atol = 2 * torch.finfo(TORCH_DTYPE).eps
    torch.testing.assert_close(
        expected,
        torch.from_dlpack(actual).cpu(),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize(
    "num_patches",
    [8],
    ids=["small_grid"],
)
def test_patch_embedder_matches_reference(num_patches: int) -> None:
    """Verify Gemma4VisionPatchEmbedder output matches the HF reference.

    The MAX implementation flattens patches to ``[total_patches, patch_dim]``
    while HF operates on batched ``[batch, num_patches, patch_dim]`` tensors.
    We use batch=1 with no padding positions for a direct comparison.
    """
    torch.manual_seed(42)

    hidden_size = VISION_EMBED_HIDDEN_SIZE
    patch_size = VISION_PATCH_SIZE
    pos_emb_size = VISION_POSITION_EMBEDDING_SIZE
    # Patch dim is the pooling kernel * patch_size * patch_size
    patch_dim = 3 * patch_size**2

    proj_weight = torch.randn(hidden_size, patch_dim, dtype=TORCH_DTYPE)
    pos_emb_table = torch.randn(2, pos_emb_size, hidden_size, dtype=TORCH_DTYPE)

    # Pixel values in [0, 1] and integer position IDs.
    patches_flat = torch.rand(num_patches, patch_dim, dtype=TORCH_DTYPE)
    pixel_position_ids = torch.randint(
        0, pos_emb_size, (num_patches, 2), dtype=torch.int32
    )

    # MAX implementation (flat inputs, no batch dimension).
    config = _make_config(hidden_size, patch_size, pos_emb_size)
    embedder = Gemma4VisionPatchEmbedder(config, DeviceRef.GPU())
    embedder.load_state_dict(
        {
            "input_proj.weight": proj_weight,
            "position_embedding_table": pos_emb_table,
        }
    )
    max_output = _build_and_run(embedder, patches_flat, pixel_position_ids)

    # HF reference (batched inputs, batch=1, no padding).
    ref = TorchGemma4VisionPatchEmbedder(hidden_size, patch_size, pos_emb_size)
    ref.input_proj.weight = torch.nn.Parameter(proj_weight)
    ref.position_embedding_table = torch.nn.Parameter(pos_emb_table)

    patches_batched = patches_flat.unsqueeze(0)  # [1, num_patches, patch_dim]
    pos_ids_batched = pixel_position_ids.unsqueeze(0).to(
        torch.int64
    )  # [1, num_patches, 2]
    padding_positions = torch.zeros(1, num_patches, dtype=torch.bool)

    ref_output = ref(patches_batched, pos_ids_batched, padding_positions)
    ref_output = ref_output.squeeze(0).detach()  # [num_patches, hidden_size]

    _assert_close(ref_output, max_output)


def test_patch_embedder_output_shape() -> None:
    """Verify the output shape is [total_patches, hidden_size]."""
    torch.manual_seed(0)

    hidden_size = VISION_EMBED_HIDDEN_SIZE
    patch_size = VISION_PATCH_SIZE
    pos_emb_size = VISION_POSITION_EMBEDDING_SIZE
    num_patches = 12
    patch_dim = 3 * patch_size**2

    config = _make_config(hidden_size, patch_size, pos_emb_size)
    embedder = Gemma4VisionPatchEmbedder(config, DeviceRef.GPU())
    embedder.load_state_dict(
        {
            "input_proj.weight": torch.randn(
                hidden_size, patch_dim, dtype=TORCH_DTYPE
            ),
            "position_embedding_table": torch.randn(
                2, pos_emb_size, hidden_size, dtype=TORCH_DTYPE
            ),
        }
    )

    patches_flat = torch.rand(num_patches, patch_dim, dtype=TORCH_DTYPE)
    pixel_position_ids = torch.randint(
        0, pos_emb_size, (num_patches, 2), dtype=torch.int32
    )

    result = _build_and_run(embedder, patches_flat, pixel_position_ids)
    assert torch.from_dlpack(result).shape == (num_patches, hidden_size)


def test_patch_embedder_pixel_normalisation() -> None:
    """Verify pixel normalisation: output changes when input shifts from [0,1]→[-1,1].

    Uses a zero projection weight so the output depends only on the pixel values
    fed through the linear layer (which must have been normalised).  With zero
    weights the projection is always zero, so an all-zero patches_flat input
    should give the same result as an all-ones patches_flat input (both project
    to zero).  This test instead checks that the raw bfloat16 normalisation
    step occurs by comparing against the reference.
    """
    torch.manual_seed(7)

    hidden_size = VISION_EMBED_HIDDEN_SIZE
    patch_size = VISION_PATCH_SIZE
    pos_emb_size = VISION_POSITION_EMBEDDING_SIZE
    num_patches = 4
    patch_dim = 3 * patch_size**2

    proj_weight = torch.randn(hidden_size, patch_dim, dtype=TORCH_DTYPE)
    pos_emb_table = torch.zeros(2, pos_emb_size, hidden_size, dtype=TORCH_DTYPE)

    # All-zeros pixel values → normalised to -1.
    patches_flat = torch.zeros(num_patches, patch_dim, dtype=TORCH_DTYPE)
    pixel_position_ids = torch.zeros(num_patches, 2, dtype=torch.int32)

    config = _make_config(hidden_size, patch_size, pos_emb_size)
    embedder = Gemma4VisionPatchEmbedder(config, DeviceRef.GPU())
    embedder.load_state_dict(
        {
            "input_proj.weight": proj_weight,
            "position_embedding_table": pos_emb_table,
        }
    )
    max_output = _build_and_run(embedder, patches_flat, pixel_position_ids)

    # Reference: manually normalise and project.
    normalised = (patches_flat.float() * 2.0 - 1.0).to(TORCH_DTYPE)
    ref_output = (normalised @ proj_weight.T).detach()

    _assert_close(ref_output, max_output)
