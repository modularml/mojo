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
"""KERN-3352: CI coverage for the `router_gate_mixed_gemv` launcher.

`router_gate_mixed_gemv` is a host-side launcher generic over `static_N` plus
six layout/storage parameters, so its body is only type-checked once something
specializes it. Its sole production caller is the `mo.router.gate.mixed.gemv`
custom op, reachable only from a `manual`-tagged target -- which is how #91194
("Treat Int and UInt as non-device passable") swept the `Int32(...)` casts
through `gemv.mojo` while silently skipping this launch site and breaking
MiniMax-M3 on main. These tests exist to instantiate the launcher body itself
from a cheap, always-run target; calling the inner `gemv_split_k` instead would
not have caught that break (`test_gemv.mojo` already does exactly that).

The instantiation mirrors the production call site
(`graph_compiler/builtin_kernels/linalg.mojo`): runtime `M`, compile-time
`N`/`K`. A static `M` would specialize a shape production never compiles.
"""

from std.random import randn, random_float64, seed
from std.sys import has_amd_gpu_accelerator
from std.testing import assert_equal, assert_false, assert_true

from internal_utils import assert_almost_equal
from layout import Coord, Idx, TileTensor, row_major
from linalg.gemv import router_gate_mixed_gemv, router_gate_use_mixed_gemv
from max.gpu.host import DeviceContext

# MiniMax-M3's router-gate contraction depth. `_k_iter_body` has no K-tail
# guard and the launcher tiles K by `simd_width * num_threads == 4 * 128`, so
# any K here must stay a multiple of 512 (6144 == 12 * 512).
comptime ROUTER_K = 6144

# Pre-written into the output. Elements the kernel must not touch -- the
# trailing guard, and every element when `m == 0` -- still hold it afterwards.
comptime SENTINEL = Float32(-12345.0)
comptime GUARD_ELEMS = 8


def _run_case[static_N: Int, K: Int](ctx: DeviceContext, m: Int) raises:
    """Runs the launcher at `[m, K] x [static_N, K]^T` and checks the output.

    Parameters:
        static_N: Output width. Drives the launcher's `check_bounds_n` guard
            (`static_N % 2 != 0`), so an even and an odd value compile the two
            halves of its compile-time branching.
        K: Contraction depth, a multiple of 512.

    Args:
        ctx: The device context.
        m: Runtime row count. `m == 0` exercises the empty-launch guard.
    """
    print("== router_gate_mixed_gemv m=", m, "N=", static_N, "K=", K)

    # Buffers stay non-empty at m == 0 so the launcher gets valid pointers,
    # which is exactly what the graph-capture warmup hands it.
    var rows = m if m > 0 else 1
    var c_elems = rows * static_N + GUARD_ELEMS

    var a_host = ctx.enqueue_create_host_buffer[.bfloat16](rows * K)
    var b_host = ctx.enqueue_create_host_buffer[.float32](static_N * K)
    var c_host = ctx.enqueue_create_host_buffer[.float32](c_elems)
    var c_expected = ctx.enqueue_create_host_buffer[.float32](c_elems)

    for i in range(rows * K):
        a_host[i] = random_float64(min=-1.0, max=1.0).cast[.bfloat16]()
    randn(b_host.unsafe_ptr(), static_N * K)
    for i in range(c_elems):
        c_host[i] = SENTINEL

    var a_dev = ctx.enqueue_create_buffer[.bfloat16](rows * K)
    var b_dev = ctx.enqueue_create_buffer[.float32](static_N * K)
    var c_dev = ctx.enqueue_create_buffer[.float32](c_elems)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(c_dev, c_host)

    var c_tt = TileTensor(
        c_dev.unsafe_ptr(), row_major(Coord(m, Idx[static_N]))
    )
    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(Coord(m, Idx[K])))
    var b_tt = TileTensor(b_dev.unsafe_ptr(), row_major[static_N, K]())

    router_gate_mixed_gemv[static_N](
        c_tt, a_tt.as_immut(), b_tt.as_immut(), m, static_N, K, ctx
    )

    ctx.enqueue_copy(c_host, c_dev)
    ctx.synchronize()

    # DRIV-199: keep device buffers alive past synchronize.
    _ = a_dev^
    _ = b_dev^
    _ = c_dev^

    # bf16 -> fp32 widening is lossless and the reduction structure matches
    # `gemv_split_k`, so an fp32 host reference is legitimate at a tight
    # tolerance. Untouched elements must still read back as the sentinel.
    for i in range(c_elems):
        c_expected[i] = SENTINEL
    for row in range(m):
        for col in range(static_N):
            var acc = Float32(0)
            for kk in range(K):
                acc += (
                    a_host[row * K + kk].cast[.float32]() * b_host[col * K + kk]
                )
            c_expected[row * static_N + col] = acc

    assert_almost_equal(
        c_host.unsafe_ptr(),
        c_expected.unsafe_ptr(),
        num_elements=c_elems,
        atol=1e-4,
        rtol=1e-2,
    )

    # Top-1 is what the MoE gate actually consumes: a tolerance check alone
    # would pass through a routing flip.
    for row in range(m):
        var got_arg = 0
        var expected_arg = 0
        for col in range(1, static_N):
            if c_host[row * static_N + col] > c_host[row * static_N + got_arg]:
                got_arg = col
            if (
                c_expected[row * static_N + col]
                > c_expected[row * static_N + expected_arg]
            ):
                expected_arg = col
        assert_equal(
            got_arg,
            expected_arg,
            "top-1 routing mismatch on row " + String(row),
        )

    print("PASS")


def test_router_gate_use_mixed_gemv() raises:
    """Pins the M window the fused path claims (`0 < m <= 64`)."""
    assert_false(router_gate_use_mixed_gemv(0))
    assert_true(router_gate_use_mixed_gemv(1))
    assert_true(router_gate_use_mixed_gemv(16))
    assert_true(router_gate_use_mixed_gemv(17))
    assert_true(router_gate_use_mixed_gemv(64))
    assert_false(router_gate_use_mixed_gemv(65))


def main() raises:
    comptime if not has_amd_gpu_accelerator():
        print("SKIP: AMD GPU not available")
        return

    seed(0)
    test_router_gate_use_mixed_gemv()

    with DeviceContext() as ctx:
        _run_case[128, ROUTER_K](ctx, 1)
        _run_case[128, ROUTER_K](ctx, 2)
        _run_case[128, ROUTER_K](ctx, 16)
        _run_case[128, ROUTER_K](ctx, 17)
        _run_case[128, ROUTER_K](ctx, 24)
        _run_case[128, ROUTER_K](ctx, 32)
        _run_case[128, ROUTER_K](ctx, 64)
        # Odd N exercises check_bounds_n in both tile_n bands.
        _run_case[127, ROUTER_K](ctx, 2)
        _run_case[127, ROUTER_K](ctx, 24)
        # Empty-launch guard: no launch, no fault, output untouched.
        _run_case[128, ROUTER_K](ctx, 0)

    print("ALL TESTS PASSED")
