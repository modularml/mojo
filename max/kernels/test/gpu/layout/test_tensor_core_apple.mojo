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
"""Functional tests for Apple M5 simdgroup_matrix MMA operations.

Tests all supported type combinations:
- Float: {F16, BF16, F32} x {F16, BF16, F32} -> F32 (9 combos)
- FP8: {E4M3, E5M2} inputs -> F32 (e4m3xe4m3, e5m2xe5m2, mixed)
- Integer: {I8, U8} x {I8, U8} -> I32 (4 combos)
- Strided loads (col_stride > 1)
- Runtime transpose

Pattern: A = sequential values, B = identity. D = A @ I = A widened to
accumulator type. Verify D[i] is close to the original A[i].
"""

from std.sys.info import _accelerator_arch

from max.gpu import WARP_SIZE
from max.gpu.compute.arch.mma_apple import (
    _mma_apple,
    _mma_apple_transposable,
    apple_mma_load,
    apple_mma_store,
)
from max.gpu.host import DeviceContext

comptime _N = 16
comptime _NUM_ELEMENTS = _N * _N


# ---------------------------------------------------------------------------
# Parametric MMA kernel: A(a_dtype) x B(b_dtype) -> D(d_dtype)
# ---------------------------------------------------------------------------


def mma_kernel[
    a_dtype: DType, b_dtype: DType, d_dtype: DType
](
    a_ptr: MutPointer[Scalar[a_dtype], MutAnyOrigin],
    b_ptr: MutPointer[Scalar[b_dtype], MutAnyOrigin],
    d_ptr: MutPointer[Scalar[d_dtype], MutAnyOrigin],
):
    var a_frag = apple_mma_load[a_dtype](a_ptr, _N)
    var b_frag = apple_mma_load[b_dtype](b_ptr, _N)
    var c_frag = SIMD[d_dtype, 8](0)
    var d_frag = SIMD[d_dtype, 8](0)
    _mma_apple(d_frag, a_frag, b_frag, c_frag)
    apple_mma_store[d_dtype](d_ptr, _N, d_frag)


# ---------------------------------------------------------------------------
# Generic test runner
# ---------------------------------------------------------------------------


def run_mma_test[
    a_dtype: DType, b_dtype: DType, d_dtype: DType
](name: String, ctx: DeviceContext, tol: Float32 = 0.0) raises:
    print("==", name)

    # A = sequential values cast to a_dtype
    var a_host = ctx.enqueue_create_host_buffer[a_dtype](_NUM_ELEMENTS)
    for i in range(_N):
        for j in range(_N):
            a_host[i * _N + j] = Scalar[a_dtype](i * _N + j)

    # B = identity cast to b_dtype
    var b_host = ctx.enqueue_create_host_buffer[b_dtype](_NUM_ELEMENTS)
    for i in range(_NUM_ELEMENTS):
        b_host[i] = Scalar[b_dtype](0)
    for i in range(_N):
        b_host[i * _N + i] = Scalar[b_dtype](1)

    var a_dev = ctx.enqueue_create_buffer[a_dtype](_NUM_ELEMENTS)
    var b_dev = ctx.enqueue_create_buffer[b_dtype](_NUM_ELEMENTS)
    var d_dev = ctx.enqueue_create_buffer[d_dtype](_NUM_ELEMENTS)

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    ctx.enqueue_function[mma_kernel[a_dtype, b_dtype, d_dtype]](
        a_dev, b_dev, d_dev, grid_dim=(1), block_dim=(WARP_SIZE)
    )

    var d_host = ctx.enqueue_create_host_buffer[d_dtype](_NUM_ELEMENTS)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    # D = A @ I = A, widened to d_dtype.
    var pass_ = True
    for i in range(_NUM_ELEMENTS):
        var got = Float32(d_host[i])
        var expected = Float32(a_host[i])
        if abs(got - expected) > tol:
            print("FAIL", name + ": index", i, "got", got, "expected", expected)
            pass_ = False

    if pass_:
        print("PASS")


# ---------------------------------------------------------------------------
# Strided kernel and test
# ---------------------------------------------------------------------------


def mma_strided_kernel(
    a_ptr: MutPointer[Float32, MutAnyOrigin],
    b_ptr: MutPointer[Float32, MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
):
    var a_frag = apple_mma_load[.float32](a_ptr, _N * 2, col_stride=2)
    var b_frag = apple_mma_load[.float32](b_ptr, _N * 2, col_stride=2)
    var c_frag = SIMD[.float32, 8](0)
    var d_frag = SIMD[.float32, 8](0)
    _mma_apple(d_frag, a_frag, b_frag, c_frag)
    apple_mma_store[.float32](d_ptr, _N, d_frag)


def run_mma_test_strided(ctx: DeviceContext) raises:
    print("== test_strided")

    comptime _STRIDE = 2
    comptime _BUF_COLS = _N * _STRIDE
    comptime _BUF_SIZE = _N * _BUF_COLS

    var a_host = ctx.enqueue_create_host_buffer[.float32](_BUF_SIZE)
    var b_host = ctx.enqueue_create_host_buffer[.float32](_BUF_SIZE)

    for i in range(_BUF_SIZE):
        a_host[i] = Float32(0)
        b_host[i] = Float32(0)

    # A = sequential at even columns
    for i in range(_N):
        for j in range(_N):
            a_host[i * _BUF_COLS + j * _STRIDE] = Float32(i * _N + j)

    # B = identity at even columns
    for i in range(_N):
        b_host[i * _BUF_COLS + i * _STRIDE] = Float32(1)

    var a_dev = ctx.enqueue_create_buffer[.float32](_BUF_SIZE)
    var b_dev = ctx.enqueue_create_buffer[.float32](_BUF_SIZE)
    var d_dev = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    ctx.enqueue_function[mma_strided_kernel](
        a_dev, b_dev, d_dev, grid_dim=(1), block_dim=(WARP_SIZE)
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    var pass_ = True
    for i in range(_NUM_ELEMENTS):
        var expected = Float32(i)
        if d_host[i] != expected:
            print(
                "FAIL strided: index", i, "got", d_host[i], "expected", expected
            )
            pass_ = False

    if pass_:
        print("PASS")


# ---------------------------------------------------------------------------
# Runtime transpose kernel and test
# ---------------------------------------------------------------------------


def mma_rt_transpose_kernel(
    a_ptr: MutPointer[Float32, MutAnyOrigin],
    b_ptr: MutPointer[Float32, MutAnyOrigin],
    d_ff_ptr: MutPointer[Float32, MutAnyOrigin],
    d_tf_ptr: MutPointer[Float32, MutAnyOrigin],
    d_ft_ptr: MutPointer[Float32, MutAnyOrigin],
    d_tt_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Runs MMA for all 4 transpose combos: (F,F), (T,F), (F,T), (T,T)."""
    var a_frag = apple_mma_load[.float32](a_ptr, _N)
    var b_frag = apple_mma_load[.float32](b_ptr, _N)
    var zero = SIMD[.float32, 8](0)

    var d_ff = SIMD[.float32, 8](0)
    _mma_apple_transposable(d_ff, a_frag, b_frag, zero, False, False)
    apple_mma_store[.float32](d_ff_ptr, _N, d_ff)

    var d_tf = SIMD[.float32, 8](0)
    _mma_apple_transposable(d_tf, a_frag, b_frag, zero, True, False)
    apple_mma_store[.float32](d_tf_ptr, _N, d_tf)

    var d_ft = SIMD[.float32, 8](0)
    _mma_apple_transposable(d_ft, a_frag, b_frag, zero, False, True)
    apple_mma_store[.float32](d_ft_ptr, _N, d_ft)

    var d_tt = SIMD[.float32, 8](0)
    _mma_apple_transposable(d_tt, a_frag, b_frag, zero, True, True)
    apple_mma_store[.float32](d_tt_ptr, _N, d_tt)


def _check_transpose(
    name: String,
    a: ImmPointer[Float32, _],
    b: ImmPointer[Float32, _],
    d: ImmPointer[Float32, _],
    ta: Bool,
    tb: Bool,
) -> Bool:
    """Compare GPU result against host reference matmul with transpose."""
    var pass_ = True
    for i in range(_N):
        for j in range(_N):
            var acc = Float32(0)
            for k in range(_N):
                var av = a[k * _N + i] if ta else a[i * _N + k]
                var bv = b[j * _N + k] if tb else b[k * _N + j]
                acc += av * bv
            var idx = i * _N + j
            var got = d[idx]
            if got != acc:
                print(
                    "FAIL", name + ": index", idx, "got", got, "expected", acc
                )
                pass_ = False
    return pass_


def run_mma_test_runtime_transpose(ctx: DeviceContext) raises:
    print("== test_runtime_transpose")

    # Semi-random values in {-2, -1, 0, 1, 2} — no special structure.
    var a_host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    var b_host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    for i in range(_N):
        for j in range(_N):
            a_host[i * _N + j] = Float32((i * 7 + j * 3) % 5 - 2)
            b_host[i * _N + j] = Float32((i * 5 + j * 11) % 5 - 2)

    var a_dev = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    var b_dev = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    var d_ff_dev = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    var d_tf_dev = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    var d_ft_dev = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    var d_tt_dev = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    ctx.enqueue_function[mma_rt_transpose_kernel](
        a_dev,
        b_dev,
        d_ff_dev,
        d_tf_dev,
        d_ft_dev,
        d_tt_dev,
        grid_dim=(1),
        block_dim=(WARP_SIZE),
    )

    var d_ff = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    var d_tf = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    var d_ft = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    var d_tt = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(d_ff, d_ff_dev)
    ctx.enqueue_copy(d_tf, d_tf_dev)
    ctx.enqueue_copy(d_ft, d_ft_dev)
    ctx.enqueue_copy(d_tt, d_tt_dev)
    ctx.synchronize()

    var a_ptr = a_host.unsafe_ptr()
    var b_ptr = b_host.unsafe_ptr()
    var pass_ = _check_transpose(
        "(F,F)", a_ptr, b_ptr, d_ff.unsafe_ptr(), False, False
    )
    pass_ = (
        _check_transpose("(T,F)", a_ptr, b_ptr, d_tf.unsafe_ptr(), True, False)
        and pass_
    )
    pass_ = (
        _check_transpose("(F,T)", a_ptr, b_ptr, d_ft.unsafe_ptr(), False, True)
        and pass_
    )
    pass_ = (
        _check_transpose("(T,T)", a_ptr, b_ptr, d_tt.unsafe_ptr(), True, True)
        and pass_
    )

    if pass_:
        print("PASS")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Float combos: {F16, BF16, F32} x {F16, BF16, F32} -> F32
# CHECK-LABEL: test_f16_f16
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_f16_bf16
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_f16_f32
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_bf16_f16
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_bf16_bf16
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_bf16_f32
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_f32_f16
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_f32_bf16
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_f32_f32
# CHECK: {{PASS|SKIP}}
# FP8 combos: {E4M3, E5M2} inputs -> F32
# CHECK-LABEL: test_e4m3_e4m3
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_e5m2_e5m2
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_e4m3_e5m2
# CHECK: {{PASS|SKIP}}
# Integer 8-bit combos: {I8, U8} x {I8, U8} -> I32
# CHECK-LABEL: test_i8_i8
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_i8_u8
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_u8_i8
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_u8_u8
# CHECK: {{PASS|SKIP}}
# Feature tests
# CHECK-LABEL: test_strided
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_runtime_transpose
# CHECK: {{PASS|SKIP}}


def _skip(name: String):
    print("==", name)
    print("SKIP: requires Apple M5 + Metal 4")


def _skip_all():
    """Print SKIP for all tests — used on non-M5 Apple GPUs."""
    _skip("test_f16_f16")
    _skip("test_f16_bf16")
    _skip("test_f16_f32")
    _skip("test_bf16_f16")
    _skip("test_bf16_bf16")
    _skip("test_bf16_f32")
    _skip("test_f32_f16")
    _skip("test_f32_bf16")
    _skip("test_f32_f32")
    _skip("test_e4m3_e4m3")
    _skip("test_e5m2_e5m2")
    _skip("test_e4m3_e5m2")
    _skip("test_i8_i8")
    _skip("test_i8_u8")
    _skip("test_u8_i8")
    _skip("test_u8_u8")
    _skip("test_strided")
    _skip("test_runtime_transpose")


def main() raises:
    # Compile-time: verify the compiler is targeting Metal 4+.
    # _accelerator_arch() is "metal:4" (bazel) or "metal:5-metal4" (mojo).
    comptime if "metal" not in _accelerator_arch():
        _skip_all()
        return

    # Runtime: verify M5 hardware (compute_capability 5 == Apple M5).
    var ctx = DeviceContext()
    if ctx.compute_capability() < 5:
        _skip_all()
        return

    # Float combos (tight tolerance for F16/BF16 hw rounding)
    run_mma_test[.float16, DType.float16, DType.float32](
        "test_f16_f16", ctx, tol=0.00001
    )
    run_mma_test[.float16, DType.bfloat16, DType.float32](
        "test_f16_bf16", ctx, tol=0.00001
    )
    run_mma_test[.float16, DType.float32, DType.float32](
        "test_f16_f32", ctx, tol=0.00001
    )
    run_mma_test[.bfloat16, DType.float16, DType.float32](
        "test_bf16_f16", ctx, tol=0.00001
    )
    run_mma_test[.bfloat16, DType.bfloat16, DType.float32](
        "test_bf16_bf16", ctx, tol=0.00001
    )
    run_mma_test[.bfloat16, DType.float32, DType.float32](
        "test_bf16_f32", ctx, tol=0.00001
    )
    run_mma_test[.float32, DType.float16, DType.float32](
        "test_f32_f16", ctx, tol=0.00001
    )
    run_mma_test[.float32, DType.bfloat16, DType.float32](
        "test_f32_bf16", ctx, tol=0.00001
    )
    run_mma_test[.float32, DType.float32, DType.float32](
        "test_f32_f32", ctx, tol=0.00001
    )

    # tol=0: A @ I is exact (1.0 is representable in fp8), so D reproduces the
    # fp8-rounded A bit-for-bit.
    run_mma_test[.float8_e4m3fn, DType.float8_e4m3fn, DType.float32](
        "test_e4m3_e4m3", ctx
    )
    run_mma_test[.float8_e5m2, DType.float8_e5m2, DType.float32](
        "test_e5m2_e5m2", ctx
    )
    run_mma_test[.float8_e4m3fn, DType.float8_e5m2, DType.float32](
        "test_e4m3_e5m2", ctx
    )

    # Integer combos
    run_mma_test[.int8, DType.int8, DType.int32]("test_i8_i8", ctx)
    run_mma_test[.int8, DType.uint8, DType.int32]("test_i8_u8", ctx)
    run_mma_test[.uint8, DType.int8, DType.int32]("test_u8_i8", ctx)
    run_mma_test[.uint8, DType.uint8, DType.int32]("test_u8_u8", ctx)

    # Feature tests
    run_mma_test_strided(ctx)
    run_mma_test_runtime_transpose(ctx)
