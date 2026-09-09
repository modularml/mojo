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
"""Graph-op binding for Kimi K3's attention-residual softmax mixture.

The kernel lives in `//Kernels/lib/attn_res` (`attn_res.mix`); only the
`@extensibility.register` wrapper lives here, mirroring the `msa.mojo`
binding -- registration has to be declared inside the built-in kernel
library itself, not the standalone lib, for a served graph to resolve
the op.
"""

import extensibility

from extensibility import InputTensor, OutputTensor
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
from max.gpu.host.info import is_gpu

from attn_res.mix import attn_res_mix_gpu

comptime _PDL_LEVEL = PDLLevel.ON


@extensibility.register("attn_res_mix")
struct AttnResMix:
    """Kimi K3 attention-residual softmax mixture, fused into one kernel.

    Replaces the reference's `ops.stack` + RMS-normalize + score-reduce +
    softmax + weighted-reduce chain (6-7 separate kernel launches; see
    `Kernels/lib/attn_res/mix.mojo`'s module docstring for the profile and
    the reassociation this fuses on) with one, starting AFTER the stack
    (the caller still builds `candidates` with its own `ops.stack`).

    Tensor shapes:
        - output      : [tokens, hidden]              (OUT)
        - candidates  : [tokens, num_candidates, hidden]
        - proj_weight : [1, hidden]
        - norm_weight : [hidden]
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
        eps: StaticString = "1e-6",
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        candidates: InputTensor[dtype=dtype, rank=3, ...],
        proj_weight: InputTensor[dtype=dtype, rank=2, ...],
        norm_weight: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert is_gpu[
            target
        ](), "attn_res_mix is only supported on GPU."

        # `ops.custom`'s extensibility bridge only accepts bool/int/str/DType
        # parameters (no float), so `eps` -- a host-known constant at every
        # call site -- arrives string-encoded; `atof` is prelude, no import.
        var eps_f32 = Float32(atof(eps))

        var tokens = candidates.dim_size(0)
        var num_candidates = candidates.dim_size(1)
        var hidden = candidates.dim_size(2)

        debug_assert(
            proj_weight.dim_size(1) == hidden,
            "attn_res_mix: proj_weight width must match hidden",
        )
        debug_assert(
            norm_weight.dim_size(0) == hidden,
            "attn_res_mix: norm_weight width must match hidden",
        )

        var output_tt = output.to_tile_tensor[.int64]()
        var candidates_tt = candidates.to_tile_tensor[.int64]()
        var proj_tt = proj_weight.to_tile_tensor[.int64]()
        var norm_tt = norm_weight.to_tile_tensor[.int64]()

        comptime BLOCK_SIZE = 256

        # Candidate count is bounded by the model's residual block size
        # (2-8 wide) and known at every Python call site, so it is a
        # comptime kernel parameter -- dispatch on the runtime value.
        comptime MAX_CANDIDATES = 8
        var dispatched = False

        comptime for c in range(1, MAX_CANDIDATES + 1):
            if num_candidates == c:
                dispatched = True
                ctx.enqueue_function[
                    attn_res_mix_gpu[
                        dtype,
                        output_tt.LayoutType,
                        output_tt.Engine,
                        candidates_tt.LayoutType,
                        candidates_tt.Engine,
                        proj_tt.LayoutType,
                        proj_tt.Engine,
                        norm_tt.LayoutType,
                        norm_tt.Engine,
                        c,
                        BLOCK_SIZE,
                    ]
                ](
                    output_tt,
                    candidates_tt,
                    proj_tt,
                    norm_tt,
                    eps_f32,
                    Int32(hidden),
                    grid_dim=(tokens,),
                    block_dim=(BLOCK_SIZE,),
                    attributes=pdl_launch_attributes(_PDL_LEVEL),
                )

        if not dispatched:
            raise Error(
                "attn_res_mix: unsupported candidate count "
                + String(num_candidates)
                + " (compiled: 1.."
                + String(MAX_CANDIDATES)
                + ")"
            )
