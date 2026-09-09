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
"""Tests for MmaOpApple: fragment load layout, MMA correctness, store, tiling.

Validates:
- _load_fragment produces the correct per-thread element mapping
- 1x1 MMA (NN) against host reference matmul
- 2x2 tiled MMA with K-loop against host reference
- All 4 transpose combos (NN, TN, NT, TT)
- Parent stride (subtile of a larger matrix)
- zero_accum resets accumulators correctly
- Bounded MMA with zero-fill for OOB elements
- Bounded store with partial output region
"""

from std.memory import AddressSpace, unsafe_stack_allocation
from std.random import random_si64
from std.sys.info import _accelerator_arch

from max.gpu import WARP_SIZE, lane_id
from max.gpu.sync import barrier
from max.gpu.compute.arch.mma_apple import _apple_frag_layout
from max.gpu.host import DeviceContext

from layout import TileTensor
from layout.tile_layout import row_major, col_major
from linalg.arch.apple.mma import MmaOpApple

comptime _N = 16
comptime _NUM_ELEMENTS = _N * _N
comptime _FRAG_SIZE = 8  # 8 elements per thread


# ---------------------------------------------------------------------------
# Host-side reference matmul
# ---------------------------------------------------------------------------


def _host_matmul_ref[
    ta: Bool, tb: Bool
](
    a: ImmPointer[Float16, _],
    b: ImmPointer[Float16, _],
    M: Int,
    N: Int,
    K: Int,
    i: Int,
    j: Int,
) -> Float32:
    """Compute one element D[i,j] = (A @ B)[i,j] on the host.

    When ta=False: A[i,k] is at a[i*K + k]. When ta=True: a[k*M + i].
    When tb=False: B[k,j] is at b[k*N + j]. When tb=True: b[j*K + k].
    """
    var acc = Float32(0)
    for k in range(K):
        comptime if ta and tb:
            acc += Float32(a[k * M + i]) * Float32(b[j * K + k])
        elif ta and not tb:
            acc += Float32(a[k * M + i]) * Float32(b[k * N + j])
        elif not ta and tb:
            acc += Float32(a[i * K + k]) * Float32(b[j * K + k])
        else:
            acc += Float32(a[i * K + k]) * Float32(b[k * N + j])
    return acc


# ---------------------------------------------------------------------------
# Host-side verification against canonical _apple_frag_layout
# ---------------------------------------------------------------------------


def _verify_fragments(
    out_ptr: MutPointer[Float32, _],
) -> Bool:
    """Verify 32 threads' fragment outputs against the canonical layout.

    Each thread wrote its 8-element fragment to out_ptr[tid * 8 .. tid * 8 + 7].
    Compare against _apple_frag_layout(tid) applied to a 16x16 tile with
    sequential values [r * 16 + c].
    """
    var pass_ = True
    for tid in range(WARP_SIZE):
        var layout = _apple_frag_layout(tid)
        var row_lo = layout[0]
        var col_base = layout[1]
        var base = tid * _FRAG_SIZE

        for j in range(4):
            var r = row_lo
            var c = col_base + j
            var expected = Float32(r * _N + c)
            var got = out_ptr[base + j]
            if got != expected:
                print(
                    "FAIL: tid",
                    tid,
                    "lo[" + String(j) + "]",
                    got,
                    "!=",
                    expected,
                )
                pass_ = False

        for j in range(4):
            var r = row_lo + 8
            var c = col_base + j
            var expected = Float32(r * _N + c)
            var got = out_ptr[base + 4 + j]
            if got != expected:
                print(
                    "FAIL: tid",
                    tid,
                    "hi[" + String(j) + "]",
                    got,
                    "!=",
                    expected,
                )
                pass_ = False
    return pass_


# ---------------------------------------------------------------------------
# GPU kernel: raw fragment load (same pointer math as _load_fragment)
# ---------------------------------------------------------------------------


def fragment_load_kernel(
    input_ptr: MutPointer[Float32, MutAnyOrigin],
    output_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Each thread loads its 8-element fragment and writes to output.

    Uses the same pointer arithmetic as MmaOpApple._load_fragment:
    rb/cb from lane_id, then two vectorized 4-wide loads separated
    by 8 rows.
    """
    var tile = TileTensor(input_ptr, row_major[16, 16]())
    var lid = lane_id()
    var rb = UInt16(((Int(lid) & 7) >> 1) + ((Int(lid) & 16) >> 2))
    var cb = UInt16(((Int(lid) & 1) << 2) + (Int(lid) & 8))

    var row_stride = Int(tile.layout.stride[0]().value())
    var offset_lo = Int(rb) * row_stride + Int(cb)
    var offset_hi = offset_lo + 8 * row_stride

    var lo = (tile._storage + offset_lo).load[width=4]()
    var hi = (tile._storage + offset_hi).load[width=4]()
    var frag = lo.join(hi)

    # Write 8 elements to output at offset tid * 8
    var base = Int(lid) * _FRAG_SIZE
    for i in range(_FRAG_SIZE):
        output_ptr[base + i] = frag[i]


# ---------------------------------------------------------------------------
# GPU kernels for MMA tests
# ---------------------------------------------------------------------------


def mma_1x1_kernel(
    a_ptr: MutPointer[Float16, MutAnyOrigin],
    b_ptr: MutPointer[Float16, MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """1x1 MMA: 16x16 @ 16x16 -> 16x16 (NN, F16 -> F32)."""
    var a_tile = TileTensor(a_ptr, row_major[16, 16]())
    var b_tile = TileTensor(b_ptr, row_major[16, 16]())
    var d_tile = TileTensor(d_ptr, row_major[16, 16]())

    var mma_op = MmaOpApple[.float32, .float16, 1, 1]()
    var accum = type_of(mma_op).zero_accum()
    mma_op.mma(accum, a_tile, b_tile)
    mma_op.store(accum, d_tile)


comptime _K_2x2 = 64


def mma_2x2_kernel(
    a_ptr: MutPointer[Float16, MutAnyOrigin],
    b_ptr: MutPointer[Float16, MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """2x2 MMA: A[32,64] @ B[64,32] -> D[32,32], K-loop with 4 iters."""
    var a_mat = TileTensor(a_ptr, row_major[32, _K_2x2]())
    var b_mat = TileTensor(b_ptr, row_major[_K_2x2, 32]())
    var d_mat = TileTensor(d_ptr, row_major[32, 32]())

    var mma_op = MmaOpApple[.float32, .float16, 2, 2]()
    var accum = type_of(mma_op).zero_accum()

    for k16 in range(_K_2x2 // 16):
        var a_slice = a_mat.tile[32, 16](0, k16)
        var b_slice = b_mat.tile[16, 32](k16, 0)
        mma_op.mma(accum, a_slice, b_slice)

    mma_op.store(accum, d_mat)


# Transpose kernels: each needs its own comptime transpose_a/transpose_b.
def mma_tn_kernel(
    a_ptr: MutPointer[Float16, MutAnyOrigin],
    b_ptr: MutPointer[Float16, MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """TN: transpose_a=True, transpose_b=False."""
    var a_tile = TileTensor(a_ptr, row_major[16, 16]())
    var b_tile = TileTensor(b_ptr, row_major[16, 16]())
    var d_tile = TileTensor(d_ptr, row_major[16, 16]())

    var mma_op = MmaOpApple[
        DType.float32, DType.float16, 1, 1, transpose_a=True
    ]()
    var accum = type_of(mma_op).zero_accum()
    mma_op.mma(accum, a_tile, b_tile)
    mma_op.store(accum, d_tile)


def mma_nt_kernel(
    a_ptr: MutPointer[Float16, MutAnyOrigin],
    b_ptr: MutPointer[Float16, MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """NT: transpose_a=False, transpose_b=True."""
    var a_tile = TileTensor(a_ptr, row_major[16, 16]())
    # B stored as (N, K) = (16, 16) when transpose_b
    var b_tile = TileTensor(b_ptr, row_major[16, 16]())
    var d_tile = TileTensor(d_ptr, row_major[16, 16]())

    var mma_op = MmaOpApple[
        DType.float32, DType.float16, 1, 1, transpose_b=True
    ]()
    var accum = type_of(mma_op).zero_accum()
    mma_op.mma(accum, a_tile, b_tile)
    mma_op.store(accum, d_tile)


def mma_tt_kernel(
    a_ptr: MutPointer[Float16, MutAnyOrigin],
    b_ptr: MutPointer[Float16, MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """TT: transpose_a=True, transpose_b=True."""
    var a_tile = TileTensor(a_ptr, row_major[16, 16]())
    var b_tile = TileTensor(b_ptr, row_major[16, 16]())
    var d_tile = TileTensor(d_ptr, row_major[16, 16]())

    var mma_op = MmaOpApple[
        DType.float32, DType.float16, 1, 1, transpose_a=True, transpose_b=True
    ]()
    var accum = type_of(mma_op).zero_accum()
    mma_op.mma(accum, a_tile, b_tile)
    mma_op.store(accum, d_tile)


# Parent stride kernel: operates on a 16x16 subtile of a 256x256 matrix.
def mma_parent_stride_kernel(
    a_ptr: MutPointer[Float16, MutAnyOrigin],
    b_ptr: MutPointer[Float16, MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """MMA on subtile at (1,2) of a 256x256 parent matrix.

    A subtile at row=1, col=2 means rows [16..32), cols [32..48).
    The TileTensor stride[0] is 256, not 16.
    """
    var a_mat = TileTensor(a_ptr, row_major[256, 256]())
    var b_mat = TileTensor(b_ptr, row_major[256, 256]())
    var d_mat = TileTensor(d_ptr, row_major[16, 16]())

    # Extract 16x16 subtile at tile position (1, 2) in both A and B
    var a_sub = a_mat.tile[16, 16](1, 2)
    var b_sub = b_mat.tile[16, 16](2, 1)

    var mma_op = MmaOpApple[.float32, .float16, 1, 1]()
    var accum = type_of(mma_op).zero_accum()
    mma_op.mma(accum, a_sub, b_sub)
    mma_op.store(accum, d_mat)


# zero_accum kernel: two matmuls with reset between them.
def zero_accum_kernel(
    a_ptr: MutPointer[Float16, MutAnyOrigin],
    b_ptr: MutPointer[Float16, MutAnyOrigin],
    d1_ptr: MutPointer[Float32, MutAnyOrigin],
    d2_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Two successive matmuls with zero_accum between them.

    d1 = A @ B, then zero_accum, then d2 = A @ B.
    Should give d1 == d2 (not d2 == 2 * d1).
    """
    var a_tile = TileTensor(a_ptr, row_major[16, 16]())
    var b_tile = TileTensor(b_ptr, row_major[16, 16]())
    var d1_tile = TileTensor(d1_ptr, row_major[16, 16]())
    var d2_tile = TileTensor(d2_ptr, row_major[16, 16]())

    var mma_op = MmaOpApple[.float32, .float16, 1, 1]()
    var accum = type_of(mma_op).zero_accum()
    mma_op.mma(accum, a_tile, b_tile)
    mma_op.store(accum, d1_tile)
    accum = type_of(mma_op).zero_accum()
    mma_op.mma(accum, a_tile, b_tile)
    mma_op.store(accum, d2_tile)


# ---------------------------------------------------------------------------
# Test: fragment load layout correctness
# ---------------------------------------------------------------------------


def test_fragment_load_layout(ctx: DeviceContext) raises:
    print("== test_fragment_load_layout")

    # Fill 16x16 tile with tile[r][c] = r * 16 + c
    var input_host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    for r in range(_N):
        for c in range(_N):
            input_host[r * _N + c] = Float32(r * _N + c)

    var input_dev = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    var output_dev = ctx.enqueue_create_buffer[.float32](WARP_SIZE * _FRAG_SIZE)
    ctx.enqueue_copy(input_dev, input_host)

    ctx.enqueue_function[fragment_load_kernel](
        input_dev, output_dev, grid_dim=(1), block_dim=(WARP_SIZE)
    )

    var output_host = ctx.enqueue_create_host_buffer[.float32](
        WARP_SIZE * _FRAG_SIZE
    )
    ctx.enqueue_copy(output_host, output_dev)
    ctx.synchronize()

    if _verify_fragments(output_host.unsafe_ptr()):
        print("PASS")


# ---------------------------------------------------------------------------
# Test: 1x1 MMA (16x16, NN, F16 -> F32)
# ---------------------------------------------------------------------------


def test_mma_1x1(ctx: DeviceContext) raises:
    _run_16x16_mma_test[mma_1x1_kernel]("test_mma_1x1", ctx)


# ---------------------------------------------------------------------------
# Test: 2x2 tiled MMA (32x32 output, K=64)
# ---------------------------------------------------------------------------


def test_mma_2x2(ctx: DeviceContext) raises:
    print("== test_mma_2x2")

    comptime M = 32
    comptime N = 32
    comptime K = _K_2x2  # 64

    # Fill with small integers [-2, 2]
    var a_host = ctx.enqueue_create_host_buffer[.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float16](K * N)
    for i in range(M * K):
        a_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
    for i in range(K * N):
        b_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())

    var a_dev = ctx.enqueue_create_buffer[.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[.float32](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    ctx.enqueue_function[mma_2x2_kernel](
        a_dev, b_dev, d_dev, grid_dim=(1), block_dim=(WARP_SIZE)
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    if _check_matmul_result[False, False](
        "2x2",
        a_host.unsafe_ptr(),
        b_host.unsafe_ptr(),
        d_host.unsafe_ptr(),
        M,
        N,
        K,
        0.5,
    ):
        print("PASS")


# ---------------------------------------------------------------------------
# Test: transpose combos (TN, NT, TT)
# ---------------------------------------------------------------------------


def _check_matmul_result[
    ta: Bool, tb: Bool
](
    name: String,
    a_ptr: ImmPointer[Float16, _],
    b_ptr: ImmPointer[Float16, _],
    d_ptr: ImmPointer[Float32, _],
    M: Int,
    N: Int,
    K: Int,
    tol: Float32,
) -> Bool:
    """Compare GPU result against host reference matmul."""
    var pass_ = True
    for i in range(M):
        for j in range(N):
            var expected = _host_matmul_ref[ta, tb](
                a_ptr,
                b_ptr,
                M,
                N,
                K,
                i,
                j,
            )
            var got = d_ptr[i * N + j]
            if abs(got - expected) > tol:
                print(
                    "FAIL:",
                    name,
                    "index",
                    i * N + j,
                    "got",
                    got,
                    "expected",
                    expected,
                )
                pass_ = False
    return pass_


# ---------------------------------------------------------------------------
# Test: K > 16 (internal K iteration)
# ---------------------------------------------------------------------------


def mma_k32_kernel(
    a_ptr: MutPointer[Float16, MutAnyOrigin],
    b_ptr: MutPointer[Float16, MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """16x16 matmul with K=32: single mma() call handles two K steps."""
    var a_tile = TileTensor(a_ptr, row_major[16, 32]())
    var b_tile = TileTensor(b_ptr, row_major[32, 16]())
    var d_tile = TileTensor(d_ptr, row_major[16, 16]())

    var mma_op = MmaOpApple[.float32, .float16, 1, 1]()
    var accum = type_of(mma_op).zero_accum()
    mma_op.mma(accum, a_tile, b_tile)
    mma_op.store(accum, d_tile)


def test_mma_k32(ctx: DeviceContext) raises:
    """Test that mma() handles K=32 (two 16-deep steps) in a single call."""
    print("== test_mma_k32")

    comptime M = 16
    comptime N = 16
    comptime K = 32

    var a_host = ctx.enqueue_create_host_buffer[.float16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float16](K * N)
    for i in range(M * K):
        a_host[i] = Float16(Int(random_si64(-2, 2)))
    for i in range(K * N):
        b_host[i] = Float16(Int(random_si64(-2, 2)))

    var a_dev = ctx.enqueue_create_buffer[.float16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float16](K * N)
    var d_dev = ctx.enqueue_create_buffer[.float32](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    ctx.enqueue_function[mma_k32_kernel](
        a_dev, b_dev, d_dev, grid_dim=(1), block_dim=(WARP_SIZE)
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    if _check_matmul_result[False, False](
        "(K=32)",
        a_host.unsafe_ptr(),
        b_host.unsafe_ptr(),
        d_host.unsafe_ptr(),
        M,
        N,
        K,
        Float32(0.5),
    ):
        print("PASS")


def mma_shared_kernel(
    a_ptr: MutPointer[Float16, MutAnyOrigin],
    b_ptr: MutPointer[Float16, MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """1x1 MMA with operands staged through threadgroup shared memory.

    Correctness-only — the simdgroup can load directly from global with
    better perf. This confirms the MMA path works when fragments come
    from an `AddressSpace.SHARED` TileTensor.
    """
    var a_shared = unsafe_stack_allocation[
        _NUM_ELEMENTS, DType.float16, address_space=.SHARED
    ]()
    var b_shared = unsafe_stack_allocation[
        _NUM_ELEMENTS, DType.float16, address_space=.SHARED
    ]()

    # 32 threads cooperatively copy 256 elements (8 per lane, strided).
    var lid = Int(lane_id())
    for i in range(8):
        a_shared[i * WARP_SIZE + lid] = a_ptr[i * WARP_SIZE + lid]
        b_shared[i * WARP_SIZE + lid] = b_ptr[i * WARP_SIZE + lid]
    barrier()

    var a_tile = TileTensor(a_shared, row_major[16, 16]())
    var b_tile = TileTensor(b_shared, row_major[16, 16]())
    var d_tile = TileTensor(d_ptr, row_major[16, 16]())

    var mma_op = MmaOpApple[.float32, .float16, 1, 1]()
    var accum = type_of(mma_op).zero_accum()
    mma_op.mma(accum, a_tile, b_tile)
    mma_op.store(accum, d_tile)


def test_mma_shared_mem(ctx: DeviceContext) raises:
    """Staging inputs through shared memory must produce the same result
    as loading directly from global memory."""
    _run_16x16_mma_test[mma_shared_kernel]("test_mma_shared_mem", ctx)


def mma_i8_k32_kernel(
    a_ptr: MutPointer[Int8, MutAnyOrigin],
    b_ptr: MutPointer[Int8, MutAnyOrigin],
    d_ptr: MutPointer[Int32, MutAnyOrigin],
):
    """16x16 matmul with K=32 and i8 inputs, i32 accumulator.

    Exercises the int-path of `_mma_apple_transposable` (widening i8 MMA)
    and the K=32 internal two-step path of `mma()`.
    """
    var a_tile = TileTensor(a_ptr, row_major[16, 32]())
    var b_tile = TileTensor(b_ptr, row_major[32, 16]())
    var d_tile = TileTensor(d_ptr, row_major[16, 16]())

    var mma_op = MmaOpApple[.int32, .int8, 1, 1]()
    var accum = type_of(mma_op).zero_accum()
    mma_op.mma(accum, a_tile, b_tile)
    mma_op.store(accum, d_tile)


def test_mma_i8_k32(ctx: DeviceContext) raises:
    """Test i8 inputs, i32 accumulator, K=32 (two 16-deep steps per mma())."""
    print("== test_mma_i8_k32")

    comptime M = 16
    comptime N = 16
    comptime K = 32

    var a_host = ctx.enqueue_create_host_buffer[.int8](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.int8](K * N)
    for i in range(M * K):
        a_host[i] = Int8(Int(random_si64(-5, 5)))
    for i in range(K * N):
        b_host[i] = Int8(Int(random_si64(-5, 5)))

    var a_dev = ctx.enqueue_create_buffer[.int8](M * K)
    var b_dev = ctx.enqueue_create_buffer[.int8](K * N)
    var d_dev = ctx.enqueue_create_buffer[.int32](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    ctx.enqueue_function[mma_i8_k32_kernel](
        a_dev, b_dev, d_dev, grid_dim=(1), block_dim=(WARP_SIZE)
    )

    var d_host = ctx.enqueue_create_host_buffer[.int32](M * N)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var acc = Int32(0)
            for k in range(K):
                acc += Int32(Int(a_host[i * K + k])) * Int32(
                    Int(b_host[k * N + j])
                )
            var got = d_host[i * N + j]
            if got != acc:
                print(
                    "FAIL: i8 K=32 index",
                    i * N + j,
                    "got",
                    got,
                    "expected",
                    acc,
                )
                pass_ = False

    if pass_:
        print("PASS")


def test_transpose_tn(ctx: DeviceContext) raises:
    _run_16x16_mma_test[mma_tn_kernel, ta=True]("test_transpose_TN", ctx)


def test_transpose_nt(ctx: DeviceContext) raises:
    _run_16x16_mma_test[mma_nt_kernel, tb=True]("test_transpose_NT", ctx)


def test_transpose_tt(ctx: DeviceContext) raises:
    _run_16x16_mma_test[mma_tt_kernel, ta=True, tb=True](
        "test_transpose_TT", ctx
    )


# ---------------------------------------------------------------------------
# Test: parent stride (subtile of a larger matrix)
# ---------------------------------------------------------------------------


def test_parent_stride(ctx: DeviceContext) raises:
    print("== test_parent_stride")

    comptime _P = 256  # parent matrix is 256x256

    # Fill two 256x256 F16 matrices with small integers
    var a_host = ctx.enqueue_create_host_buffer[.float16](_P * _P)
    var b_host = ctx.enqueue_create_host_buffer[.float16](_P * _P)
    for i in range(_P * _P):
        a_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())
        b_host[i] = Float16(random_si64(Int64(-2), Int64(2)).cast[.float16]())

    var a_dev = ctx.enqueue_create_buffer[.float16](_P * _P)
    var b_dev = ctx.enqueue_create_buffer[.float16](_P * _P)
    var d_dev = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    ctx.enqueue_function[mma_parent_stride_kernel](
        a_dev, b_dev, d_dev, grid_dim=(1), block_dim=(WARP_SIZE)
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # Host reference: compute the same 16x16 subtile matmul.
    # A subtile at (1,2): rows [16..32), cols [32..48) of A.
    # B subtile at (2,1): rows [32..48), cols [16..32) of B.
    # D[i,j] = sum_k A[16+i, 32+k] * B[32+k, 16+j] for i,j in [0,16), k in [0,16)
    var a_ptr = a_host.unsafe_ptr()
    var b_ptr = b_host.unsafe_ptr()
    var d_ptr = d_host.unsafe_ptr()

    var pass_ = True
    for i in range(_N):
        for j in range(_N):
            var acc = Float32(0)
            for k in range(_N):
                var av = Float32(a_ptr[(16 + i) * _P + (32 + k)])
                var bv = Float32(b_ptr[(32 + k) * _P + (16 + j)])
                acc += av * bv
            var got = d_ptr[i * _N + j]
            if abs(got - acc) > 0.01:
                print(
                    "FAIL: index",
                    i * _N + j,
                    "got",
                    got,
                    "expected",
                    acc,
                )
                pass_ = False

    if pass_:
        print("PASS")


# ---------------------------------------------------------------------------
# Test: zero_accum
# ---------------------------------------------------------------------------


def test_zero_accum(ctx: DeviceContext) raises:
    print("== test_zero_accum")

    # Fill A and B with small integers
    var a_host = ctx.enqueue_create_host_buffer[.float16](_NUM_ELEMENTS)
    var b_host = ctx.enqueue_create_host_buffer[.float16](_NUM_ELEMENTS)
    for i in range(_N):
        for j in range(_N):
            var idx = i * _N + j
            a_host[idx] = Float16((i * 3 + j * 7) % 5 - 2)
            b_host[idx] = Float16((i * 11 + j * 5) % 5 - 2)

    var a_dev = ctx.enqueue_create_buffer[.float16](_NUM_ELEMENTS)
    var b_dev = ctx.enqueue_create_buffer[.float16](_NUM_ELEMENTS)
    var d1_dev = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    var d2_dev = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    ctx.enqueue_function[zero_accum_kernel](
        a_dev, b_dev, d1_dev, d2_dev, grid_dim=(1), block_dim=(WARP_SIZE)
    )

    var d1_host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    var d2_host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(d1_host, d1_dev)
    ctx.enqueue_copy(d2_host, d2_dev)
    ctx.synchronize()

    # d1 and d2 should be equal (not d2 = 2 * d1)
    var d1_ptr = d1_host.unsafe_ptr()
    var d2_ptr = d2_host.unsafe_ptr()
    var pass_ = True
    for i in range(_NUM_ELEMENTS):
        if abs(d1_ptr[i] - d2_ptr[i]) > 0.001:
            print(
                "FAIL: index",
                i,
                "d1",
                d1_ptr[i],
                "d2",
                d2_ptr[i],
            )
            pass_ = False

    if pass_:
        print("PASS")


# ---------------------------------------------------------------------------
# GPU kernels for bounded tests
# ---------------------------------------------------------------------------


def bounded_mma_kernel(
    a_ptr: MutPointer[Float16, MutAnyOrigin],
    b_ptr: MutPointer[Float16, MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
    m_valid_dev: Int32,
    k_valid_dev: Int32,
):
    """Bounded MMA: 16x16 @ 16x16 with partial valid region."""
    var m_valid = Int(m_valid_dev)
    var k_valid = Int(k_valid_dev)
    var a_tile = TileTensor(a_ptr, row_major[16, 16]())
    var b_tile = TileTensor(b_ptr, row_major[16, 16]())
    var d_tile = TileTensor(d_ptr, row_major[16, 16]())

    var mma_op = MmaOpApple[.float32, .float16, 1, 1]()
    var accum = type_of(mma_op).zero_accum()
    mma_op.mma[bounded=True](
        accum,
        a_tile,
        b_tile,
        a_valid_rows=m_valid,
        k_valid=k_valid,
    )
    mma_op.store(accum, d_tile)


def bounded_store_kernel(
    a_ptr: MutPointer[Float16, MutAnyOrigin],
    b_ptr: MutPointer[Float16, MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
    m_valid_dev: Int32,
    n_valid_dev: Int32,
):
    """Bounded MMA + bounded store: partial valid output region."""
    var m_valid = Int(m_valid_dev)
    var n_valid = Int(n_valid_dev)
    var a_tile = TileTensor(a_ptr, row_major[16, 16]())
    var b_tile = TileTensor(b_ptr, row_major[16, 16]())
    var d_tile = TileTensor(d_ptr, row_major[16, 16]())

    var mma_op = MmaOpApple[.float32, .float16, 1, 1]()
    var accum = type_of(mma_op).zero_accum()
    mma_op.mma[bounded=True](
        accum,
        a_tile,
        b_tile,
        a_valid_rows=m_valid,
        b_valid_cols=n_valid,
    )
    mma_op.store_bounded(accum, d_tile, valid_rows=m_valid, valid_cols=n_valid)


# ---------------------------------------------------------------------------
# Test: bounded MMA (zero-fill semantics)
# ---------------------------------------------------------------------------


def test_bounded_mma(ctx: DeviceContext) raises:
    print("== test_bounded_mma")

    comptime M_VALID = 12
    comptime K_VALID = 10

    # Fill full 16x16 A and B with small integers in [-2, 2]
    var a_host = ctx.enqueue_create_host_buffer[.float16](_NUM_ELEMENTS)
    var b_host = ctx.enqueue_create_host_buffer[.float16](_NUM_ELEMENTS)
    for i in range(_N):
        for j in range(_N):
            var idx = i * _N + j
            a_host[idx] = Float16(
                random_si64(Int64(-2), Int64(2)).cast[.float16]()
            )
            b_host[idx] = Float16(
                random_si64(Int64(-2), Int64(2)).cast[.float16]()
            )

    var a_dev = ctx.enqueue_create_buffer[.float16](_NUM_ELEMENTS)
    var b_dev = ctx.enqueue_create_buffer[.float16](_NUM_ELEMENTS)
    var d_dev = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    ctx.enqueue_function[bounded_mma_kernel](
        a_dev,
        b_dev,
        d_dev,
        Int32(M_VALID),
        Int32(K_VALID),
        grid_dim=(1),
        block_dim=(WARP_SIZE),
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # Host reference: zero-pad A (rows >= M_VALID or cols >= K_VALID)
    # and B (rows >= K_VALID), then multiply.
    var a_ptr = a_host.unsafe_ptr()
    var b_ptr = b_host.unsafe_ptr()
    var d_ptr = d_host.unsafe_ptr()

    var pass_ = True
    for i in range(_N):
        for j in range(_N):
            var acc = Float32(0)
            for k in range(_N):
                # A is zero-padded: rows >= M_VALID or cols >= K_VALID
                var av = Float32(0)
                if i < M_VALID and k < K_VALID:
                    av = Float32(a_ptr[i * _N + k])
                # B is zero-padded: rows >= K_VALID (NN layout)
                var bv = Float32(0)
                if k < K_VALID:
                    bv = Float32(b_ptr[k * _N + j])
                acc += av * bv
            var got = d_ptr[i * _N + j]
            if abs(got - acc) > 0.01:
                print(
                    "FAIL: index",
                    i * _N + j,
                    "got",
                    got,
                    "expected",
                    acc,
                )
                pass_ = False

    if pass_:
        print("PASS")


# ---------------------------------------------------------------------------
# Test: bounded store (partial output store)
# ---------------------------------------------------------------------------


def test_store_bounded(ctx: DeviceContext) raises:
    print("== test_store_bounded")

    comptime M_VALID = 12
    comptime N_VALID = 14

    # Fill full 16x16 A and B with small integers in [-2, 2]
    var a_host = ctx.enqueue_create_host_buffer[.float16](_NUM_ELEMENTS)
    var b_host = ctx.enqueue_create_host_buffer[.float16](_NUM_ELEMENTS)
    for i in range(_N):
        for j in range(_N):
            var idx = i * _N + j
            a_host[idx] = Float16(
                random_si64(Int64(-2), Int64(2)).cast[.float16]()
            )
            b_host[idx] = Float16(
                random_si64(Int64(-2), Int64(2)).cast[.float16]()
            )

    # Initialize output to -1.0 sentinel
    var d_host_init = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    for i in range(_NUM_ELEMENTS):
        d_host_init[i] = Float32(-1.0)

    var a_dev = ctx.enqueue_create_buffer[.float16](_NUM_ELEMENTS)
    var b_dev = ctx.enqueue_create_buffer[.float16](_NUM_ELEMENTS)
    var d_dev = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(d_dev, d_host_init)

    ctx.enqueue_function[bounded_store_kernel](
        a_dev,
        b_dev,
        d_dev,
        Int32(M_VALID),
        Int32(N_VALID),
        grid_dim=(1),
        block_dim=(WARP_SIZE),
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # Host reference: bounded MMA (a_valid_rows=M_VALID, b_valid_cols=N_VALID,
    # k_valid defaults to 16 since we don't restrict K here).
    var a_ptr = a_host.unsafe_ptr()
    var b_ptr = b_host.unsafe_ptr()
    var d_ptr = d_host.unsafe_ptr()

    var pass_ = True
    for i in range(_N):
        for j in range(_N):
            if i < M_VALID and j < N_VALID:
                # Compute bounded reference: A rows < M_VALID, B cols < N_VALID,
                # k_valid=16 (default, all K elements valid).
                # In the bounded MMA, A is zero-padded for rows >= M_VALID,
                # B is zero-padded for cols >= N_VALID. Since i < M_VALID
                # and j < N_VALID here, all A rows and B cols are valid.
                var acc = Float32(0)
                for k in range(_N):
                    acc += Float32(a_ptr[i * _N + k]) * Float32(
                        b_ptr[k * _N + j]
                    )
                var got = d_ptr[i * _N + j]
                if abs(got - acc) > 0.01:
                    print(
                        "FAIL: in-bounds index",
                        i * _N + j,
                        "got",
                        got,
                        "expected",
                        acc,
                    )
                    pass_ = False
            else:
                # Outside valid region: should still be -1.0 sentinel
                var got = d_ptr[i * _N + j]
                if got != Float32(-1.0):
                    print(
                        "FAIL: OOB index",
                        i * _N + j,
                        "got",
                        got,
                        "expected -1.0",
                    )
                    pass_ = False

    if pass_:
        print("PASS")


# ---------------------------------------------------------------------------
# Tests: col-major tile support (stride-aware load + XOR hardware flag)
# ---------------------------------------------------------------------------


def mma_col_major_a_kernel(
    a_ptr: MutPointer[Float16, MutAnyOrigin],
    b_ptr: MutPointer[Float16, MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """A stored col-major, B stored row-major. D = A @ B."""
    var a_tile = TileTensor(a_ptr, col_major[16, 16]())
    var b_tile = TileTensor(b_ptr, row_major[16, 16]())
    var d_tile = TileTensor(d_ptr, row_major[16, 16]())

    var mma_op = MmaOpApple[.float32, .float16, 1, 1]()
    var accum = type_of(mma_op).zero_accum()
    mma_op.mma(accum, a_tile, b_tile)
    mma_op.store(accum, d_tile)


def mma_col_major_b_kernel(
    a_ptr: MutPointer[Float16, MutAnyOrigin],
    b_ptr: MutPointer[Float16, MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """A stored row-major, B stored col-major. D = A @ B."""
    var a_tile = TileTensor(a_ptr, row_major[16, 16]())
    var b_tile = TileTensor(b_ptr, col_major[16, 16]())
    var d_tile = TileTensor(d_ptr, row_major[16, 16]())

    var mma_op = MmaOpApple[.float32, .float16, 1, 1]()
    var accum = type_of(mma_op).zero_accum()
    mma_op.mma(accum, a_tile, b_tile)
    mma_op.store(accum, d_tile)


def mma_col_major_ab_kernel(
    a_ptr: MutPointer[Float16, MutAnyOrigin],
    b_ptr: MutPointer[Float16, MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Both A and B stored col-major. D = A @ B."""
    var a_tile = TileTensor(a_ptr, col_major[16, 16]())
    var b_tile = TileTensor(b_ptr, col_major[16, 16]())
    var d_tile = TileTensor(d_ptr, row_major[16, 16]())

    var mma_op = MmaOpApple[.float32, .float16, 1, 1]()
    var accum = type_of(mma_op).zero_accum()
    mma_op.mma(accum, a_tile, b_tile)
    mma_op.store(accum, d_tile)


def mma_col_major_a_transpose_a_kernel(
    a_ptr: MutPointer[Float16, MutAnyOrigin],
    b_ptr: MutPointer[Float16, MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """A col-major + transpose_a=True. XOR cancels: hw_flag=False.

    D = A^T @ B. A stored col-major as (M, K) with stride (1, M).
    transpose_a=True means compute A^T @ B. Col-major XOR True = False,
    so hardware sees no transpose — the col-major load already
    "transposes" the fragment relative to row-major.
    """
    var a_tile = TileTensor(a_ptr, col_major[16, 16]())
    var b_tile = TileTensor(b_ptr, row_major[16, 16]())
    var d_tile = TileTensor(d_ptr, row_major[16, 16]())

    var mma_op = MmaOpApple[
        DType.float32, DType.float16, 1, 1, transpose_a=True
    ]()
    var accum = type_of(mma_op).zero_accum()
    mma_op.mma(accum, a_tile, b_tile)
    mma_op.store(accum, d_tile)


def mma_col_major_b_transpose_b_kernel(
    a_ptr: MutPointer[Float16, MutAnyOrigin],
    b_ptr: MutPointer[Float16, MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """B col-major + transpose_b=True. XOR cancels: hw_flag=False.

    D = A @ B^T. B stored col-major as (K, N) with stride (1, K).
    transpose_b=True means compute A @ B^T. Col-major XOR True = False.
    """
    var a_tile = TileTensor(a_ptr, row_major[16, 16]())
    var b_tile = TileTensor(b_ptr, col_major[16, 16]())
    var d_tile = TileTensor(d_ptr, row_major[16, 16]())

    var mma_op = MmaOpApple[
        DType.float32, DType.float16, 1, 1, transpose_b=True
    ]()
    var accum = type_of(mma_op).zero_accum()
    mma_op.mma(accum, a_tile, b_tile)
    mma_op.store(accum, d_tile)


def _run_16x16_mma_test[
    kernel_fn: def(
        MutPointer[Float16, MutAnyOrigin],
        MutPointer[Float16, MutAnyOrigin],
        MutPointer[Float32, MutAnyOrigin],
    ) thin -> None,
    a_col_major: Bool = False,
    b_col_major: Bool = False,
    ta: Bool = False,
    tb: Bool = False,
](name: String, ctx: DeviceContext) raises:
    """Shared driver for 16x16 f16->f32 MMA tests.

    Generates logical A/B, stores them in the requested physical layout
    (row- or col-major), runs `kernel_fn`, and compares against the host
    reference for matmul with the given `ta`/`tb` semantics.
    """
    print("==", name)

    var a_logical = ctx.enqueue_create_host_buffer[.float16](_NUM_ELEMENTS)
    var b_logical = ctx.enqueue_create_host_buffer[.float16](_NUM_ELEMENTS)
    for i in range(_N):
        for j in range(_N):
            a_logical[i * _N + j] = Float16(
                random_si64(Int64(-2), Int64(2)).cast[.float16]()
            )
            b_logical[i * _N + j] = Float16(
                random_si64(Int64(-2), Int64(2)).cast[.float16]()
            )

    # Store each input in the requested physical layout.
    var a_phys = ctx.enqueue_create_host_buffer[.float16](_NUM_ELEMENTS)
    var b_phys = ctx.enqueue_create_host_buffer[.float16](_NUM_ELEMENTS)
    for r in range(_N):
        for c in range(_N):
            var val_a = a_logical[r * _N + c]
            var val_b = b_logical[r * _N + c]

            comptime if a_col_major:
                a_phys[c * _N + r] = val_a
            else:
                a_phys[r * _N + c] = val_a

            comptime if b_col_major:
                b_phys[c * _N + r] = val_b
            else:
                b_phys[r * _N + c] = val_b

    var a_dev = ctx.enqueue_create_buffer[.float16](_NUM_ELEMENTS)
    var b_dev = ctx.enqueue_create_buffer[.float16](_NUM_ELEMENTS)
    var d_dev = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(a_dev, a_phys)
    ctx.enqueue_copy(b_dev, b_phys)

    ctx.enqueue_function[kernel_fn](
        a_dev, b_dev, d_dev, grid_dim=(1), block_dim=(WARP_SIZE)
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    if _check_matmul_result[ta, tb](
        name,
        a_logical.unsafe_ptr(),
        b_logical.unsafe_ptr(),
        d_host.unsafe_ptr(),
        _N,
        _N,
        _N,
        0.01,
    ):
        print("PASS")


def test_col_major_a(ctx: DeviceContext) raises:
    _run_16x16_mma_test[mma_col_major_a_kernel, True, False](
        "test_col_major_a", ctx
    )


def test_col_major_b(ctx: DeviceContext) raises:
    _run_16x16_mma_test[mma_col_major_b_kernel, False, True](
        "test_col_major_b", ctx
    )


def test_col_major_ab(ctx: DeviceContext) raises:
    _run_16x16_mma_test[mma_col_major_ab_kernel, True, True](
        "test_col_major_ab", ctx
    )


def test_col_major_a_transpose_a(ctx: DeviceContext) raises:
    """Col-major A + transpose_a=True: XOR cancels. D = A^T @ B."""
    _run_16x16_mma_test[
        mma_col_major_a_transpose_a_kernel,
        a_col_major=True,
        ta=True,
    ]("test_col_major_a_transpose_a", ctx)


def test_col_major_b_transpose_b(ctx: DeviceContext) raises:
    """Col-major B + transpose_b=True: XOR cancels. D = A @ B^T."""
    _run_16x16_mma_test[
        mma_col_major_b_transpose_b_kernel,
        b_col_major=True,
        tb=True,
    ]("test_col_major_b_transpose_b", ctx)


# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------


def _skip(name: String):
    print("==", name)
    print("SKIP: requires Apple M5 + Metal 4")


def _skip_all():
    """Print SKIP for all tests -- used on non-M5 Apple GPUs."""
    _skip("test_fragment_load_layout")
    _skip("test_mma_1x1")
    _skip("test_mma_2x2")
    _skip("test_mma_k32")
    _skip("test_mma_shared_mem")
    _skip("test_mma_i8_k32")
    _skip("test_transpose_TN")
    _skip("test_transpose_NT")
    _skip("test_transpose_TT")
    _skip("test_parent_stride")
    _skip("test_zero_accum")
    _skip("test_bounded_mma")
    _skip("test_store_bounded")
    _skip("test_col_major_a")
    _skip("test_col_major_b")
    _skip("test_col_major_ab")
    _skip("test_col_major_a_transpose_a")
    _skip("test_col_major_b_transpose_b")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() raises:
    comptime if "metal" not in _accelerator_arch():
        _skip_all()
        return

    var ctx = DeviceContext()
    if ctx.compute_capability() < 5:
        _skip_all()
        return

    test_fragment_load_layout(ctx)
    test_mma_1x1(ctx)
    test_mma_2x2(ctx)
    test_mma_k32(ctx)
    test_mma_shared_mem(ctx)
    test_mma_i8_k32(ctx)
    test_transpose_tn(ctx)
    test_transpose_nt(ctx)
    test_transpose_tt(ctx)
    test_parent_stride(ctx)
    test_zero_accum(ctx)
    test_bounded_mma(ctx)
    test_store_bounded(ctx)
    test_col_major_a(ctx)
    test_col_major_b(ctx)
    test_col_major_ab(ctx)
    test_col_major_a_transpose_a(ctx)
    test_col_major_b_transpose_b(ctx)
