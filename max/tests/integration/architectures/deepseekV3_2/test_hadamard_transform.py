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
"""Test comparing MAX Hadamard transform implementation to PyTorch.

The graphs are compiled to MEFs by a CPU-only build action
(``:hadamard_mefs`` via ``mef_precompile.bzl``); this test does NOT compile. It
initializes each :data:`HADAMARD_SPECS` MEF and compares its output against the
PyTorch reference transform.
"""

from __future__ import annotations

import math

import pytest
import scipy.linalg
import torch
import torch.nn.functional as F
from _hadamard_graphs import HADAMARD_SPECS, HadamardSpec
from max.driver import Accelerator, Buffer
from max.engine import InferenceSession
from test_common.mef_precompile import init_from_mef, mefs_from_env
from torch.utils.dlpack import from_dlpack

# Check if outputs are close with appropriate tolerance
RTOL = 2 * torch.finfo(torch.bfloat16).eps
ATOL = 8 * torch.finfo(torch.bfloat16).eps

TORCH_DTYPE = torch.bfloat16


def hadamard_transform_ref(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Reference implementation with padding logic.
    [Source](https://github.com/Dao-AILab/fast-hadamard-transform/blob/master/fast_hadamard_transform/fast_hadamard_transform_interface.py#L156)
    Arguments:
        x: (..., dim)
        scale: float. Multiply the output by this number.
    Returns:
        out: (..., dim)

    Multiply each row of x by the Hadamard transform matrix.
    Equivalent to F.linear(x, torch.tensor(scipy.linalg.hadamard(dim))) * scale.
    If dim is not a power of 2, we implicitly pad x with zero so that dim is the next power of 2.
    """
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))

    hadamard_weight = torch.tensor(
        scipy.linalg.hadamard(dim_padded, dtype=float),
        dtype=x.dtype,
        device=x.device,
    )
    out = F.linear(x, hadamard_weight)
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


@pytest.mark.parametrize("spec", HADAMARD_SPECS, ids=lambda s: s.name)
@torch.no_grad()
def test_hadamard_transform(spec: HadamardSpec) -> None:
    """Test Hadamard transform comparing MAX vs PyTorch implementation.

    Args:
        spec: The precompiled parametrization (input shape and scale) to run.
    """
    mef_path = mefs_from_env("HADAMARD_MEF_RLOCATIONS")[f"{spec.name}.mef"]
    assert mef_path.is_file(), f"precompiled MEF missing: {mef_path}"

    # Create random input tensor
    torch.manual_seed(42)
    input_tensor = torch.randn(*spec.shape, dtype=TORCH_DTYPE, device="cuda")

    # Generate PyTorch output
    torch_output = hadamard_transform_ref(input_tensor, spec.scale)

    # Generate MAX output. The graph bakes the Hadamard matrix in as a
    # constant, so there are no weights to bind.
    device = Accelerator()
    session = InferenceSession(devices=[device])
    model = init_from_mef(session, mef_path)
    max_output = model.execute(Buffer.from_dlpack(input_tensor).to(device))[0]

    # Convert MAX output to torch for comparison
    max_output_torch = from_dlpack(max_output)

    # Verify shapes match
    assert torch_output.shape == max_output_torch.shape, (
        f"Shape mismatch: {torch_output.shape} vs {max_output_torch.shape}"
    )

    torch.testing.assert_close(
        torch_output,
        max_output_torch,
        rtol=RTOL,
        atol=ATOL,
    )
