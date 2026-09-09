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
"""Orientation regression test for ``max.experimental.nn.Conv2d``.

Loads identical weights into V3 ``Conv2d(permute=True)`` and
``torch.nn.Conv2d``, feeds an explicitly asymmetric input (linear
L-to-R gradient on top of noise), and compares outputs element-wise.
The failure message also reports whether the V3 output matches the
torch output mirrored along H or W — a positive mirror match localizes
the bug to the Conv2d permute path.

Single compile + single shape keeps wall time ~30s on a GPU runner,
which is acceptable for CI regression coverage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from max.driver import CPU, Accelerator, Device, accelerator_count
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Conv2d
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from torch import nn


@pytest.fixture
def device() -> Device:
    """GPU when available, CPU otherwise."""
    return Accelerator() if accelerator_count() else CPU()


def _to_numpy(tensor_or_buf: Any) -> np.ndarray:
    """Best-effort conversion of a Tensor/Buffer to a numpy array."""
    if hasattr(tensor_or_buf, "to_numpy"):
        return tensor_or_buf.to_numpy()
    if hasattr(tensor_or_buf, "to"):
        return np.from_dlpack(tensor_or_buf.to(CPU()))
    return np.from_dlpack(tensor_or_buf)


def test_v3_conv2d_permute_preserves_orientation(device: Device) -> None:
    """V3 Conv2d(permute=True) must equal torch.nn.Conv2d, and must not mirror.

    One representative shape (3-in, 4-out, 3x3 kernel, padding=1, 6x6
    spatial) covers the most common FLUX.2 VAE conv path.  A larger or
    parameterized sweep would multiply compile cost without adding
    coverage: any axis-mirror or transpose in Conv2d shows up at this
    shape just as cleanly.
    """
    torch.manual_seed(0)

    in_c, out_c, k, padding = 3, 4, 3, 1
    h, w = 6, 6

    # Weights: source of truth is torch's random init so V3 and torch
    # see bit-identical filter values.
    torch_device = torch.device("cuda" if not device.is_host else "cpu")
    torch_conv = nn.Conv2d(
        in_c, out_c, kernel_size=k, padding=padding, bias=False
    ).to(torch_device)
    weight_np = torch_conv.weight.data.detach().cpu().numpy().copy()

    # Input: noise + strong L-to-R column gradient so mirroring is
    # immediately visible in the output.
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((1, in_c, h, w)).astype(np.float32)
    x_np = x_np + (np.arange(w, dtype=np.float32) * 5.0)[None, None, None, :]
    x_np = np.ascontiguousarray(x_np)

    # V3 Conv2d on the same device.
    with F.lazy():
        conv = Conv2d(
            kernel_size=k,
            in_channels=in_c,
            out_channels=out_c,
            dtype=DType.float32,
            stride=1,
            padding=padding,
            has_bias=False,
            permute=True,
        ).to(device)
    conv.load_state_dict({"weight": Tensor.from_dlpack(weight_np).to(device)})
    compiled = conv.compile(
        TensorType(
            DType.float32,
            [1, in_c, h, w],
            device=DeviceRef.from_device(device),
        )
    )
    v3_out = _to_numpy(compiled(Tensor.from_dlpack(x_np).to(device)))

    # Torch reference.
    with torch.no_grad():
        torch_out = (
            torch_conv(torch.from_numpy(x_np).to(torch_device)).cpu().numpy()
        )

    assert v3_out.shape == torch_out.shape

    # Diagnostic: surface the specific failure mode in the message
    # (horizontal mirror -> the FLUX flip hypothesis; vertical or
    # transpose mirrors are listed too in case the bug is elsewhere).
    mirrored_w = np.allclose(v3_out, torch_out[..., ::-1], rtol=5e-3, atol=1e-4)
    mirrored_h = np.allclose(
        v3_out, torch_out[..., ::-1, :], rtol=5e-3, atol=1e-4
    )

    np.testing.assert_allclose(
        v3_out,
        torch_out,
        rtol=5e-3,
        atol=1e-4,
        err_msg=(
            "V3 Conv2d(permute=True) diverges from torch.nn.Conv2d.\n"
            f"  max abs diff: {np.abs(v3_out - torch_out).max()}\n"
            f"  matches torch[..., ::-1]    (W mirror): {mirrored_w}\n"
            f"  matches torch[..., ::-1, :] (H mirror): {mirrored_h}\n"
            "If W mirror is True, V3 Conv2d is the source of the FLUX.2 "
            "ModuleV3 horizontal flip."
        ),
    )
