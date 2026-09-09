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
"""GPU tests for the fused gated group-RMSNorm (Mamba-2 `norm_before_gate=False`).

Compares the fused kernel against a host reference that mirrors the unfused
op chain the model builds today (`cast -> silu(gate) * y -> ops.rms_norm(group)
-> * norm_weight -> cast`). The reference uses the same `rsqrt(mean(x^2) + eps)`
formula as `ops.rms_norm` and the same `nn.activations.silu`, so a bf16-output
match within a small tolerance (reduction order differs) is the correctness
gate. Runs on any GPU, including Apple Metal (the kernel is TileTensor-only, no
pointer arithmetic, no shared memory).
"""

from std.math import rsqrt
from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from std.testing import TestSuite, assert_almost_equal

from nn.activations import silu

from state_space.gated_group_rmsnorm import gated_group_rmsnorm_gpu


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def run_gated_group_rmsnorm[
    y_dtype: DType,
    gate_dtype: DType,
](
    ctx: DeviceContext,
    n_rows: Int,
    num_groups: Int,
    group_size: Int,
    gate_stride: Int = 0,
    eps: Float32 = 1e-5,
    rtol: Float64 = 2e-2,
    atol: Float64 = 1e-2,
) raises:
    """Runs the fused kernel and checks it against a host reference chain.

    `gate_stride` sets the gate tile's row stride: 0 means contiguous (stride
    == intermediate); a value > intermediate models the production layout where
    the gate is a strided split view of the fused in-proj (row stride wider than
    the accessed column count). `y`/`output` stay contiguous.
    """
    var intermediate = num_groups * group_size
    var total = n_rows * intermediate
    var gstride = gate_stride if gate_stride != 0 else intermediate
    var gate_total = n_rows * gstride

    # Host inputs (deterministic, mixed signs so silu is exercised on both).
    var y_h = ctx.enqueue_create_host_buffer[y_dtype](total)
    var gate_h = ctx.enqueue_create_host_buffer[gate_dtype](gate_total)
    var weight_h = ctx.enqueue_create_host_buffer[.float32](intermediate)
    var out_h = ctx.enqueue_create_host_buffer[y_dtype](total)

    for i in range(total):
        y_h[i] = Scalar[y_dtype](Float32(((i * 7) % 101) - 50) * 0.05)
    # Gate laid out with row stride `gstride`: the value at logical (n, col)
    # uses the same formula as the contiguous case (so a strided run exercises a
    # pure layout change), stored at physical offset `n * gstride + col`. Pad
    # columns [intermediate, gstride) are zeroed (never read by the kernel).
    for i in range(gate_total):
        gate_h[i] = Scalar[gate_dtype](0)
    for n in range(n_rows):
        for col in range(intermediate):
            var lin = n * intermediate + col
            gate_h[n * gstride + col] = Scalar[gate_dtype](
                Float32(((lin * 13) % 97) - 48) * 0.06
            )
    for c in range(intermediate):
        weight_h[c] = Float32(((c * 5) % 41) + 10) * 0.05

    # Device buffers + tiles.
    var y_d = ctx.enqueue_create_buffer[y_dtype](total)
    var gate_d = ctx.enqueue_create_buffer[gate_dtype](gate_total)
    var weight_d = ctx.enqueue_create_buffer[.float32](intermediate)
    var out_d = ctx.enqueue_create_buffer[y_dtype](total)

    ctx.enqueue_copy(y_d, y_h)
    ctx.enqueue_copy(gate_d, gate_h)
    ctx.enqueue_copy(weight_d, weight_h)

    var y_t = TileTensor(y_d, row_major(Coord(n_rows, intermediate)))
    # Gate tile: shape (n_rows, gstride) with row stride `gstride`. The kernel
    # only indexes columns [0, intermediate), so `gstride > intermediate`
    # exercises the wide-stride access exactly (a split view of the fused
    # in-proj) with no pointer arithmetic.
    var gate_t = TileTensor(gate_d, row_major(Coord(n_rows, gstride)))
    var weight_t = TileTensor(weight_d, row_major(Coord(intermediate)))
    var out_t = TileTensor(out_d, row_major(Coord(n_rows, intermediate)))

    gated_group_rmsnorm_gpu[y_dtype, gate_dtype](
        out_t, y_t, gate_t, weight_t, n_rows, num_groups, group_size, eps, ctx
    )

    ctx.enqueue_copy(out_h, out_d)
    ctx.synchronize()

    # Host reference: the exact math the fused kernel folds. `y`/`output` are
    # contiguous (row stride == intermediate); the gate is read at row stride
    # `gstride`.
    for n in range(n_rows):
        for g in range(num_groups):
            var base = g * group_size
            var m2 = Float32(0)
            for j in range(group_size):
                var col = base + j
                var yv = y_h[n * intermediate + col].cast[.float32]()
                var gv = gate_h[n * gstride + col].cast[.float32]()
                var gated = yv * silu(gv)
                m2 += gated * gated
            var nf = rsqrt(m2 / Float32(group_size) + eps)
            for j in range(group_size):
                var col = base + j
                var idx = n * intermediate + col
                var yv = y_h[idx].cast[.float32]()
                var gv = gate_h[n * gstride + col].cast[.float32]()
                var gated = yv * silu(gv)
                var t_in = (gated * nf).cast[y_dtype]()
                var prod = weight_h[col] * t_in.cast[.float32]()
                var expected = prod.cast[y_dtype]()
                assert_almost_equal(expected, out_h[idx], rtol=rtol, atol=atol)


# =============================================================================
# Test functions
# =============================================================================


def test_gated_group_rmsnorm_decode_production() raises:
    """Nemotron-H decode shape: n_rows=1, intermediate=7680, group_size=960."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_gated_group_rmsnorm[.bfloat16, DType.float32](
        ctx, n_rows=1, num_groups=8, group_size=960
    )


def test_gated_group_rmsnorm_prefill_rows() raises:
    """Prefill-like: multiple rows, production group geometry."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_gated_group_rmsnorm[.bfloat16, DType.float32](
        ctx, n_rows=5, num_groups=8, group_size=960
    )


def test_gated_group_rmsnorm_small() raises:
    """Small shape: group_size not a multiple of WARP_SIZE (strided-loop tail).
    """
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_gated_group_rmsnorm[.bfloat16, DType.float32](
        ctx, n_rows=3, num_groups=4, group_size=100
    )


def test_gated_group_rmsnorm_bf16_gate() raises:
    """BF16 gate (the fused-in-proj split path, no upstream fp32 cast)."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_gated_group_rmsnorm[.bfloat16, DType.bfloat16](
        ctx, n_rows=1, num_groups=8, group_size=960
    )


def test_gated_group_rmsnorm_strided_gate() raises:
    """Strided gate (production layout): the gate tile's row stride is STRICTLY
    GREATER than intermediate, mirroring the fused-in-proj split view (bf16
    gate). `y`/`output` stay contiguous. A strided-access bug would pass the
    contiguous cases above but fail here.
    """
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    # intermediate = 8 * 960 = 7680; pad the gate row stride to 8192 (> 7680).
    run_gated_group_rmsnorm[.bfloat16, DType.bfloat16](
        ctx, n_rows=3, num_groups=8, group_size=960, gate_stride=8192
    )


def test_gated_group_rmsnorm_f32() raises:
    """All-fp32 path (cross-platform reference-equivalence, tight tol)."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_gated_group_rmsnorm[.float32, DType.float32](
        ctx, n_rows=2, num_groups=8, group_size=960, rtol=1e-4, atol=1e-5
    )
