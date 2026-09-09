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
"""Test script comparing MAX YarnRotaryEmbedding vs PyTorch GptOssRotaryEmbedding implementation."""

import pytest
import torch
from max.driver import Accelerator, Buffer
from max.engine import InferenceSession
from test_common.mef_precompile import init_from_mef, mefs_from_env
from transformers.models.gpt_oss.configuration_gpt_oss import (
    GptOssConfig as PytorchGptOssConfig,
)
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssRotaryEmbedding,
)


@torch.no_grad()
def generate_torch_gpt_oss_rope_outputs(
    config: PytorchGptOssConfig,
    input_tensor: torch.Tensor,
    position_ids: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate outputs using PyTorch GPT-OSS rotary embedding."""
    config.torch_dtype = dtype
    rotary_emb = GptOssRotaryEmbedding(config=config, device="cuda")
    cos, sin = rotary_emb(input_tensor, position_ids)

    torch_cos = cos.to(dtype).squeeze(0).to("cpu")
    torch_sin = sin.to(dtype).squeeze(0).to("cpu")
    print("Torch position embeddings shape: ", torch_cos.shape, torch_sin.shape)
    return torch_cos, torch_sin


def generate_max_yarn_rope_outputs(
    position_ids: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate outputs using MAX YarnRotaryEmbedding."""
    session = InferenceSession(devices=[Accelerator()])

    mefs = mefs_from_env("MEF_RLOCATION")
    compiled = init_from_mef(session, mefs["spec.mef"])

    frequencies_tensor = compiled.execute()[0]
    assert isinstance(frequencies_tensor, Buffer)
    frequencies_np = frequencies_tensor.to_numpy()

    # Convert position_ids to CPU and extract the sequence positions (remove batch dimension)
    positions = position_ids.squeeze(0).cpu().numpy()  # Shape: [seq_len]

    # Extract cos and sin from the frequency tensor (shape: [max_seq_len, head_dim, 2])
    # cos is at index 0, sin is at index 1 along the last dimension
    cos_sliced = torch.from_numpy(frequencies_np[positions, :, 0]).to(dtype)
    sin_sliced = torch.from_numpy(frequencies_np[positions, :, 1]).to(dtype)

    print("MAX position embeddings shape: ", cos_sliced.shape, sin_sliced.shape)

    return cos_sliced, sin_sliced


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_gpt_oss_vs_yarn_rotary_embeddings(
    config: PytorchGptOssConfig,
    input_tensor: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    """Test GPT-OSS rotary embedding against MAX YarnRotaryEmbedding."""
    # Convert input tensor to the appropriate dtype
    input_tensor = input_tensor.to(dtype).to("cuda")

    # Create position ids
    seq_len = input_tensor.shape[1]
    position_ids = torch.arange(
        seq_len, dtype=torch.long, device="cuda"
    ).unsqueeze(0)

    # Generate outputs from both implementations
    torch_cos, torch_sin = generate_torch_gpt_oss_rope_outputs(
        config, input_tensor, position_ids, dtype
    )

    max_cos, max_sin = generate_max_yarn_rope_outputs(position_ids, dtype)

    # Compare outputs with dtype-appropriate tolerances
    eps = torch.finfo(dtype).eps
    torch.testing.assert_close(
        torch_cos,
        max_cos,
        rtol=1 * eps,
        atol=4 * eps,
    )

    torch.testing.assert_close(
        torch_sin,
        max_sin,
        rtol=1 * eps,
        atol=4 * eps,
    )
