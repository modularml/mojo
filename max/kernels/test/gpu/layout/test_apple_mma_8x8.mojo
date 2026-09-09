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
"""Functional tests for Apple 8x8 simdgroup_matrix MMA operations (M1-M5).

Unlike the 16x16 path (M5-only), the 8x8 simdgroup_matrix intrinsic is
available on every Apple GPU generation, so this test runs on any Apple GPU.

Covers all 9 float input combinations {F16, BF16, F32} x {F16, BF16, F32} ->
F32 under two patterns -- A @ I = A and A @ B against an integer host
reference -- plus a strided load.
"""

from std.sys.info import _accelerator_arch

from max.gpu import WARP_SIZE
from max.gpu.compute.arch.mma_apple import (
    _mma_apple_8x8,
    apple_mma_load_8x8,
    apple_mma_store_8x8,
)
from max.gpu.host import DeviceContext

comptime _N = 8
comptime _NUM_ELEMENTS = _N * _N


# Non-symmetric, non-commuting fills: a transposed or row-col-confused fragment
# layout passes A @ I = A (I is transpose-symmetric) but fails A @ B here.
def _a_val(i: Int, j: Int) -> Int:
    return (i * 2 + j * 3) % 7 - 3


def _b_val(i: Int, j: Int) -> Int:
    return (i * 5 + j + 2) % 7 - 3


def mma_kernel[
    a_dtype: DType, b_dtype: DType
](
    a_ptr: MutPointer[Scalar[a_dtype], MutAnyOrigin],
    b_ptr: MutPointer[Scalar[b_dtype], MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
):
    var a_frag = apple_mma_load_8x8[a_dtype](a_ptr, _N)
    var b_frag = apple_mma_load_8x8[b_dtype](b_ptr, _N)
    var c_frag = SIMD[.float32, 2](0)
    var d_frag = SIMD[.float32, 2](0)
    _mma_apple_8x8(d_frag, a_frag, b_frag, c_frag)
    apple_mma_store_8x8[.float32](d_ptr, _N, d_frag)


def run_mma_test[
    a_dtype: DType, b_dtype: DType
](name: String, ctx: DeviceContext, tol: Float32 = 0.00001) raises:
    print("==", name)

    var a_host = ctx.enqueue_create_host_buffer[a_dtype](_NUM_ELEMENTS)
    for i in range(_N):
        for j in range(_N):
            a_host[i * _N + j] = Scalar[a_dtype](i * _N + j)

    var b_host = ctx.enqueue_create_host_buffer[b_dtype](_NUM_ELEMENTS)
    for i in range(_NUM_ELEMENTS):
        b_host[i] = Scalar[b_dtype](0)
    for i in range(_N):
        b_host[i * _N + i] = Scalar[b_dtype](1)

    var a_dev = ctx.enqueue_create_buffer[a_dtype](_NUM_ELEMENTS)
    var b_dev = ctx.enqueue_create_buffer[b_dtype](_NUM_ELEMENTS)
    var d_dev = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    ctx.enqueue_function[mma_kernel[a_dtype, b_dtype]](
        a_dev, b_dev, d_dev, grid_dim=(1), block_dim=(WARP_SIZE)
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    var pass_ = True
    for i in range(_NUM_ELEMENTS):
        var got = Float32(d_host[i])
        var expected = Float32(a_host[i])
        if abs(got - expected) > tol:
            print("FAIL", name + ": index", i, "got", got, "expected", expected)
            pass_ = False

    if pass_:
        print("PASS")


def mma_strided_kernel(
    a_ptr: MutPointer[Float32, MutAnyOrigin],
    b_ptr: MutPointer[Float32, MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
):
    var a_frag = apple_mma_load_8x8[.float32](a_ptr, _N * 2, col_stride=2)
    var b_frag = apple_mma_load_8x8[.float32](b_ptr, _N * 2, col_stride=2)
    var c_frag = SIMD[.float32, 2](0)
    var d_frag = SIMD[.float32, 2](0)
    _mma_apple_8x8(d_frag, a_frag, b_frag, c_frag)
    apple_mma_store_8x8[.float32](d_ptr, _N, d_frag)


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

    for i in range(_N):
        for j in range(_N):
            a_host[i * _BUF_COLS + j * _STRIDE] = Float32(i * _N + j)
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
    for i in range(_N):
        for j in range(_N):
            var got = Float32(d_host[i * _N + j])
            var expected = Float32(i * _N + j)
            if abs(got - expected) > 0.00001:
                print(
                    "FAIL test_strided: index",
                    i * _N + j,
                    "got",
                    got,
                    "expected",
                    expected,
                )
                pass_ = False
    if pass_:
        print("PASS")


def run_mma_matmul_test[
    a_dtype: DType, b_dtype: DType
](name: String, ctx: DeviceContext, tol: Float32 = 0.001) raises:
    print("==", name)

    var a_host = ctx.enqueue_create_host_buffer[a_dtype](_NUM_ELEMENTS)
    var b_host = ctx.enqueue_create_host_buffer[b_dtype](_NUM_ELEMENTS)
    for i in range(_N):
        for j in range(_N):
            a_host[i * _N + j] = Scalar[a_dtype](_a_val(i, j))
            b_host[i * _N + j] = Scalar[b_dtype](_b_val(i, j))

    var a_dev = ctx.enqueue_create_buffer[a_dtype](_NUM_ELEMENTS)
    var b_dev = ctx.enqueue_create_buffer[b_dtype](_NUM_ELEMENTS)
    var d_dev = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    ctx.enqueue_function[mma_kernel[a_dtype, b_dtype]](
        a_dev, b_dev, d_dev, grid_dim=(1), block_dim=(WARP_SIZE)
    )

    var d_host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(d_host, d_dev)
    ctx.synchronize()

    var pass_ = True
    for i in range(_N):
        for j in range(_N):
            var ref_val = 0
            for k in range(_N):
                ref_val += _a_val(i, k) * _b_val(k, j)
            var got = Float32(d_host[i * _N + j])
            if abs(got - Float32(ref_val)) > tol:
                print(
                    "FAIL",
                    name + ": [" + String(i) + "," + String(j) + "] got",
                    got,
                    "expected",
                    ref_val,
                )
                pass_ = False

    if pass_:
        print("PASS")


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
# CHECK-LABEL: test_strided
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_matmul_f16_f16
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_matmul_f16_bf16
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_matmul_f16_f32
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_matmul_bf16_f16
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_matmul_bf16_bf16
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_matmul_bf16_f32
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_matmul_f32_f16
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_matmul_f32_bf16
# CHECK: {{PASS|SKIP}}
# CHECK-LABEL: test_matmul_f32_f32
# CHECK: {{PASS|SKIP}}


def _skip(name: String):
    print("==", name)
    print("SKIP: requires Apple GPU + Metal")


def _skip_all():
    _skip("test_f16_f16")
    _skip("test_f16_bf16")
    _skip("test_f16_f32")
    _skip("test_bf16_f16")
    _skip("test_bf16_bf16")
    _skip("test_bf16_f32")
    _skip("test_f32_f16")
    _skip("test_f32_bf16")
    _skip("test_f32_f32")
    _skip("test_strided")
    _skip("test_matmul_f16_f16")
    _skip("test_matmul_f16_bf16")
    _skip("test_matmul_f16_f32")
    _skip("test_matmul_bf16_f16")
    _skip("test_matmul_bf16_bf16")
    _skip("test_matmul_bf16_f32")
    _skip("test_matmul_f32_f16")
    _skip("test_matmul_f32_bf16")
    _skip("test_matmul_f32_f32")


def main() raises:
    # No M5 runtime gate (unlike the 16x16 path): the 8x8 intrinsic exists on
    # all Apple GPU generations (M1-M5).
    comptime if "metal" not in _accelerator_arch():
        _skip_all()
        return

    var ctx = DeviceContext()

    run_mma_test[.float16, DType.float16]("test_f16_f16", ctx)
    run_mma_test[.float16, DType.bfloat16]("test_f16_bf16", ctx)
    run_mma_test[.float16, DType.float32]("test_f16_f32", ctx)
    run_mma_test[.bfloat16, DType.float16]("test_bf16_f16", ctx)
    run_mma_test[.bfloat16, DType.bfloat16]("test_bf16_bf16", ctx)
    run_mma_test[.bfloat16, DType.float32]("test_bf16_f32", ctx)
    run_mma_test[.float32, DType.float16]("test_f32_f16", ctx)
    run_mma_test[.float32, DType.bfloat16]("test_f32_bf16", ctx)
    run_mma_test[.float32, DType.float32]("test_f32_f32", ctx)

    run_mma_test_strided(ctx)

    run_mma_matmul_test[.float16, DType.float16]("test_matmul_f16_f16", ctx)
    run_mma_matmul_test[.float16, DType.bfloat16]("test_matmul_f16_bf16", ctx)
    run_mma_matmul_test[.float16, DType.float32]("test_matmul_f16_f32", ctx)
    run_mma_matmul_test[.bfloat16, DType.float16]("test_matmul_bf16_f16", ctx)
    run_mma_matmul_test[.bfloat16, DType.bfloat16]("test_matmul_bf16_bf16", ctx)
    run_mma_matmul_test[.bfloat16, DType.float32]("test_matmul_bf16_f32", ctx)
    run_mma_matmul_test[.float32, DType.float16]("test_matmul_f32_f16", ctx)
    run_mma_matmul_test[.float32, DType.bfloat16]("test_matmul_f32_bf16", ctx)
    run_mma_matmul_test[.float32, DType.float32]("test_matmul_f32_f32", ctx)
