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
"""General dispatch for grouped block-scaled matmul.

Routes to format-specific grouped matmul implementations based on the
input dtype and target GPU architecture. Currently supports NVFP4, MXFP4,
and MXFP8 on SM100.
"""

from max.gpu.host import DeviceContext
from max.gpu.host.info import _is_sm10x_gpu
from max.gpu.primitives.grid_controls import PDLLevel
from layout import TileTensor

from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_block_scaled_sm100_dispatch,
)


def grouped_matmul_block_scaled_dispatch[
    transpose_b: Bool = True,
    target: StaticString = "cpu",
    pdl_level: PDLLevel = PDLLevel.ON,
](
    c: TileTensor,
    a: TileTensor,
    b: TileTensor,
    a_scales: TileTensor,
    b_scales: TileTensor,
    a_offsets: TileTensor,
    a_scale_offsets: TileTensor,
    expert_ids: TileTensor,
    expert_scales: TileTensor,
    num_active_experts: Int,
    estimated_total_m: Int,
    ctx: DeviceContext,
) raises:
    """Dispatch grouped block-scaled matmul to format-specific implementation.

    Currently NVFP4, MXFP4, and MXFP8 on SM100 are supported.

    Parameters:
        transpose_b: Whether B is transposed (must be True).
        target: Target device (unused, for MOGG interface compatibility).
        pdl_level: Programmatic dependent launch level.

    Args:
        c: Output tensor (total_tokens, N).
        a: Input A tensor (total_tokens, K//2 packed).
        b: Weight tensor B (num_experts, N, K//2 packed).
        a_scales: Scale factors for A (5D).
        b_scales: Scale factors for B (6D).
        a_offsets: Per-expert token offsets (num_active_experts + 1).
        a_scale_offsets: Per-expert scale offsets (num_active_experts).
        expert_ids: Active expert IDs (num_active_experts).
        expert_scales: Per-expert output scaling (num_experts).
        num_active_experts: Number of active experts.
        estimated_total_m: Estimated number of total non-padded tokens.
        ctx: Device context.
    """
    comptime assert _is_sm10x_gpu(
        ctx.default_device_info
    ), "Only support SM100 for grouped block-scaled matmul"

    grouped_matmul_block_scaled_sm100_dispatch[
        transpose_b, target, pdl_level=pdl_level
    ](
        c,
        a,
        b,
        a_scales,
        b_scales,
        a_offsets,
        a_scale_offsets,
        expert_ids,
        expert_scales,
        num_active_experts,
        estimated_total_m,
        ctx,
    )
