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
"""Fused gated group-RMSNorm for the Mamba-2 mixer (`norm_before_gate=False`).

Portable GPU/CPU kernel; the immediate target is Apple M5 Metal
(`compute_capability == 5`) batch-1 decode. Collapses the four-op decode-graph
chain that HF `Zamba2RMSNormGated` lowers to -- `cast(y -> f32)`,
`silu(gate) * y`, group `rms_norm` (a reduction, a hard dispatch boundary), and
`* norm_weight` + cast -- into ONE kernel dispatch per Mamba layer. On the
Nemotron-H decode graph that reduction cannot fold into the surrounding
elementwise ops, so the chain is ~3-4 serial dispatches x 21 layers; the win is
removing those launches, not FLOPs (decode processes 1 x 7680 elements).

Math is the `norm_before_gate=False` branch of the dense combined scan
(`selective_scan.mojo:2643-2648`), ported to a standalone varlen op. For a row
`n` and group `g` over `group_size`-wide contiguous groups of the intermediate
axis, with `col = g * group_size + j`:

    gated = f32(y[n, col]) * silu(f32(gate[n, col]))          # f32
    m2    = sum_j gated^2                                     # f32, over group
    nf    = rsqrt(m2 / group_size + eps)                      # matches ops.rms_norm
    out[n, col] = cast(f32(norm_weight[col]) * f32(cast(gated * nf, y.dtype)),
                       out.dtype)

The intermediate `cast(gated * nf, y.dtype)` before the fp32 `norm_weight`
multiply reproduces the reference exactly: `w * ops.cast(yf, input_dtype)` then a
final cast to the model dtype (`nemotron_h.py::_gated_group_rmsnorm` + the
`out_proj` cast). `norm_weight` is fp32; the product is fp32; this op returns the
model dtype so the downstream `out_proj` cast is a no-op.

Parallelization (Apple-M5 idiom -- mirrors `apple/fa_prefill`, `apple/fp4_gemv`):
one warp (simdgroup) owns one `(row, group)` reduction; its lanes stride the
group by `WARP_SIZE`, accumulate a partial sum-of-squares in fp32, and one
`warp.sum` reduces to the group `m2`. No shared memory, no `barrier()` -- the two
levers Apple silicon is most sensitive to (KB `kernels/apple-m5-fa-prefill`,
`patterns/apple-m5-gpu-performance-considerations`). A `group_size` that is not a
multiple of `WARP_SIZE` is handled by the strided while-loop (idle lanes add 0),
so no divisibility constraint is needed. All memory access is through TileTensor
indexing (`.load`/`.store` with `Coord`) -- no raw pointer arithmetic -- so a
strided `y`/`gate`/`output` view (a split of the fused in-proj) is read correctly
with no host-side stride plumbing.
"""

from max.gpu import WARP_SIZE, global_idx, lane_id
from max.gpu.host import DeviceContext
from std.math import ceildiv, rsqrt
import max.gpu.primitives.warp as warp

from layout import Coord, TensorLayout, TensorEngine, TileTensor

from nn.activations import silu


# ===----------------------------------------------------------------------=== #
# GPU kernel: one warp per (row, group)
# ===----------------------------------------------------------------------=== #


@__name(t"gated_group_rmsnorm_{dtype}_{gate_dtype}")
def gated_group_rmsnorm_kernel[
    dtype: DType,
    gate_dtype: DType,
    OutLayout: TensorLayout,
    YLayout: TensorLayout,
    GateLayout: TensorLayout,
    WeightLayout: TensorLayout,
    OutEngine: TensorEngine,
    YEngine: TensorEngine,
    GateEngine: TensorEngine,
    WeightEngine: TensorEngine,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin, Engine=OutEngine],
    y: TileTensor[dtype, YLayout, ImmutAnyOrigin, Engine=YEngine],
    gate: TileTensor[gate_dtype, GateLayout, ImmutAnyOrigin, Engine=GateEngine],
    weight: TileTensor[
        .float32, WeightLayout, ImmutAnyOrigin, Engine=WeightEngine
    ],
    n_rows_dev: Int32,
    num_groups_dev: Int32,
    group_size_dev: Int32,
    eps: Float32,
):
    var n_rows = Int(n_rows_dev)
    var num_groups = Int(num_groups_dev)
    var group_size = Int(group_size_dev)
    comptime assert output.flat_rank == 2 and y.flat_rank == 2
    comptime assert gate.flat_rank == 2 and weight.flat_rank == 1

    # One warp owns one (row, group). All 32 lanes share `group_flat` (it is
    # `global_idx.x // WARP_SIZE`, constant within a simdgroup), so the early
    # return is warp-uniform -- safe before the `warp.sum` collective below.
    var group_flat = Int(global_idx.x) // WARP_SIZE
    if group_flat >= n_rows * num_groups:
        return
    var n = group_flat // num_groups
    var base = (group_flat % num_groups) * group_size
    var lane = Int(lane_id())

    # Pass 1: group sum-of-squares of the silu-gated value (fp32 accumulate).
    var acc = Float32(0)
    var j = lane
    while j < group_size:
        var col = base + j
        var yv = y.load[width=1](Coord(n, col)).cast[.float32]()
        var gv = gate.load[width=1](Coord(n, col)).cast[.float32]()
        var gated = yv * silu(gv)
        acc += gated * gated
        j += WARP_SIZE
    var m2 = warp.sum(acc)
    var nf = rsqrt(m2 / Float32(group_size) + eps)

    # Pass 2: normalize, cast to input dtype, scale by fp32 weight, cast out.
    # Re-loading y/gate (instead of caching pass-1 `gated` in registers) is
    # acceptable: at decode this op is launch-bound (a single 1 x 7680 row),
    # so the extra loads are free. A prefill-scale reuse would want to cache
    # pass-1 `gated` in registers, but a static register array is blocked today
    # by the runtime `group_size` (not a comptime bound).
    var jj = lane
    while jj < group_size:
        var col = base + jj
        var yv = y.load[width=1](Coord(n, col)).cast[.float32]()
        var gv = gate.load[width=1](Coord(n, col)).cast[.float32]()
        var gated = yv * silu(gv)
        var t_in = (gated * nf).cast[dtype]()
        var w = weight.load[width=1](Coord(col)).cast[.float32]()
        var prod = w * t_in.cast[.float32]()
        output.store(Coord(n, col), prod.cast[dtype]())
        jj += WARP_SIZE


@always_inline
def gated_group_rmsnorm_gpu[
    dtype: DType,
    gate_dtype: DType,
](
    output: TileTensor[mut=True, dtype, ...],
    y: TileTensor[dtype, ...],
    gate: TileTensor[gate_dtype, ...],
    weight: TileTensor[.float32, ...],
    n_rows: Int,
    num_groups: Int,
    group_size: Int,
    eps: Float32,
    ctx: DeviceContext,
) raises:
    """Enqueues the fused gated group-RMSNorm; one warp per `(row, group)`."""
    comptime BLK = 256  # 8 warps / threadgroup
    var total_warps = n_rows * num_groups
    var grid = ceildiv(total_warps * WARP_SIZE, BLK)

    comptime kernel = gated_group_rmsnorm_kernel[
        dtype,
        gate_dtype,
        type_of(output).LayoutType,
        type_of(y).LayoutType,
        type_of(gate).LayoutType,
        type_of(weight).LayoutType,
        type_of(output).Engine,
        type_of(y).Engine,
        type_of(gate).Engine,
        type_of(weight).Engine,
    ]
    ctx.enqueue_function[kernel](
        output,
        y,
        gate,
        weight,
        Int32(n_rows),
        Int32(num_groups),
        Int32(group_size),
        eps,
        grid_dim=grid,
        block_dim=BLK,
    )


# ===----------------------------------------------------------------------=== #
# CPU reference (device-complete op; matches the GPU math element-for-element)
# ===----------------------------------------------------------------------=== #


def gated_group_rmsnorm_cpu[
    dtype: DType,
    gate_dtype: DType,
](
    output: TileTensor[mut=True, dtype, ...],
    y: TileTensor[dtype, ...],
    gate: TileTensor[gate_dtype, ...],
    weight: TileTensor[.float32, ...],
    n_rows: Int,
    num_groups: Int,
    group_size: Int,
    eps: Float32,
):
    comptime assert output.flat_rank == 2 and y.flat_rank == 2
    comptime assert gate.flat_rank == 2 and weight.flat_rank == 1

    for n in range(n_rows):
        for g in range(num_groups):
            var base = g * group_size
            var m2 = Float32(0)
            for j in range(group_size):
                var col = base + j
                var yv = y.load[width=1](Coord(n, col)).cast[.float32]()
                var gv = gate.load[width=1](Coord(n, col)).cast[.float32]()
                var gated = yv * silu(gv)
                m2 += gated * gated
            var nf = rsqrt(m2 / Float32(group_size) + eps)
            for j in range(group_size):
                var col = base + j
                var yv = y.load[width=1](Coord(n, col)).cast[.float32]()
                var gv = gate.load[width=1](Coord(n, col)).cast[.float32]()
                var gated = yv * silu(gv)
                var t_in = (gated * nf).cast[dtype]()
                var w = weight.load[width=1](Coord(col)).cast[.float32]()
                var prod = w * t_in.cast[.float32]()
                output.store(Coord(n, col), prod.cast[dtype]())
