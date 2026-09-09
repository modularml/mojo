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
"""Python wrapper for the fused gated group-RMSNorm kernel."""

from __future__ import annotations

from typing import cast

from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, ops


def gated_group_rmsnorm(
    y: TensorValue,
    gate: TensorValue,
    norm_weight: TensorValue,
    eps: float,
    group_size: int,
) -> TensorValue:
    """Fused gated group-RMSNorm (HF ``Zamba2RMSNormGated``, ``norm_before_gate=False``).

    Collapses ``cast(y->f32) -> silu(gate)*y -> group rms_norm -> *norm_weight ->
    cast`` into one dispatch. ``y`` and ``gate`` are ``[N, intermediate]``;
    ``norm_weight`` is fp32 ``[intermediate]``. Each contiguous ``group_size``
    slice of the intermediate axis is normalized independently. Returns the model
    dtype (``y.dtype``), so the downstream ``out_proj`` cast is a no-op.

    Args:
        y: ``[N, intermediate]`` SSD scan output (model dtype).
        gate: ``[N, intermediate]`` gate projection (any float dtype; may be a
            strided split view of the fused in-proj).
        norm_weight: ``[intermediate]`` fp32 RMSNorm weight.
        eps: Epsilon inside ``rsqrt(mean_sq + eps)``.
        group_size: Width of each normalized group (``intermediate // n_groups``).

    Returns:
        ``[N, intermediate]`` in ``y.dtype``.
    """
    device = y.device
    out_type = TensorType(y.dtype, [y.shape[0], y.shape[1]], device)
    # eps is a rank-0 fp32 CPU constant; the framework marshals it into the
    # kernel's ``Float32`` execute arg (the ``ops.rms_norm`` / fused-qk-rmsnorm
    # idiom).
    eps_c = ops.constant(eps, DType.float32, device=DeviceRef.CPU())
    results = ops.custom(
        "gated_group_rmsnorm",
        device,
        [y, gate, norm_weight, eps_c],
        [out_type],
        parameters={"group_size": group_size},
    )
    return cast(TensorValue, results[0])
