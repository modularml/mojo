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
"""Tests for the GPU buffer verification kernel used in bench_matmul."""

from std.math import ceildiv
from max.gpu import global_idx, grid_dim, block_dim, thread_idx, block_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceBuffer, DeviceContext
from std.testing import assert_equal, assert_true


# ---------------------------------------------------------------------------
# Kernel under test — copied from bench_matmul.mojo (cannot import from a
# mojo_binary target, so duplication is the standard pattern).
# ---------------------------------------------------------------------------
def _verify_buffers_gpu[
    c_type: DType, BLOCK_SIZE: Int
](
    output: ImmPointer[Scalar[c_type], ImmutAnyOrigin],
    reference: ImmPointer[Scalar[c_type], ImmutAnyOrigin],
    length_dev: Int32,
    atol: Float32,
    rtol: Float32,
    result: MutPointer[Float32, MutAnyOrigin],
):
    """GPU kernel that computes verification metrics in one pass.

    Each block computes partial reductions and writes 5 Float32 values:
      [0] abs_diff_sum — for relative difference metric
      [1] abs_ref_sum  — for relative difference metric
      [2] max_violation — max(|x-y| - (atol + rtol*|y|)), <=0 means pass
      [3] out_nz — 1.0 if any output element is nonzero
      [4] ref_nz — 1.0 if any reference element is nonzero
    """
    var length = Int(length_dev)
    var abs_diff_sum: Float32 = 0
    var abs_ref_sum: Float32 = 0
    var max_violation = Float32.MIN_FINITE
    var out_nz: Float32 = 0
    var ref_nz: Float32 = 0

    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < length:
        var x = output[i].cast[.float32]()
        var y = reference[i].cast[.float32]()
        abs_diff_sum += abs(x - y)
        abs_ref_sum += abs(y)
        max_violation = max(max_violation, abs(x - y) - (atol + rtol * abs(y)))
        if x != 0:
            out_nz = 1.0
        if y != 0:
            ref_nz = 1.0
        i += stride

    abs_diff_sum = block.sum[block_size=BLOCK_SIZE](abs_diff_sum)
    abs_ref_sum = block.sum[block_size=BLOCK_SIZE](abs_ref_sum)
    max_violation = block.max[block_size=BLOCK_SIZE](max_violation)
    out_nz = block.max[block_size=BLOCK_SIZE](out_nz)
    ref_nz = block.max[block_size=BLOCK_SIZE](ref_nz)

    if thread_idx.x == 0:
        var base = block_idx.x * 5
        result[base + 0] = abs_diff_sum
        result[base + 1] = abs_ref_sum
        result[base + 2] = max_violation
        result[base + 3] = out_nz
        result[base + 4] = ref_nz


# ---------------------------------------------------------------------------
# Helper kernel to fill a device buffer with a constant value.
# ---------------------------------------------------------------------------
def _fill_buffer[
    dtype: DType,
](
    ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    length_dev: Int32,
    val: Scalar[dtype],
):
    var length = Int(length_dev)
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < length:
        ptr[i] = val
        i += stride


# ---------------------------------------------------------------------------
# Metrics struct for readability.
# ---------------------------------------------------------------------------
@fieldwise_init
struct VerifyMetrics:
    var abs_diff_sum: Float32
    var abs_ref_sum: Float32
    var max_violation: Float32
    var out_nz: Float32
    var ref_nz: Float32


# ---------------------------------------------------------------------------
# Host-side helper: launch the verification kernel, copy back partial
# results, reduce them, and return the 5 final metrics.
# ---------------------------------------------------------------------------
def run_verify_kernel[
    dtype: DType,
    NUM_BLOCKS: Int,
    BLOCK_SIZE: Int,
](
    ctx: DeviceContext,
    output_buf: DeviceBuffer[dtype],
    reference_buf: DeviceBuffer[dtype],
    length: Int,
    atol: Float32,
    rtol: Float32,
) raises -> VerifyMetrics:
    var result_device = ctx.enqueue_create_buffer[.float32](NUM_BLOCKS * 5)

    comptime kernel = _verify_buffers_gpu[dtype, BLOCK_SIZE]
    ctx.enqueue_function[kernel](
        output_buf,
        reference_buf,
        Int32(length),
        atol,
        rtol,
        result_device,
        grid_dim=NUM_BLOCKS,
        block_dim=BLOCK_SIZE,
    )

    var result_host = ctx.enqueue_create_host_buffer[.float32](NUM_BLOCKS * 5)
    ctx.enqueue_copy(result_host, result_device)
    ctx.synchronize()

    var total_abs_diff: Float32 = 0
    var total_abs_ref: Float32 = 0
    var worst_violation = Float32.MIN_FINITE
    var any_out_nz: Float32 = 0
    var any_ref_nz: Float32 = 0

    for b_idx in range(NUM_BLOCKS):
        var base = b_idx * 5
        total_abs_diff += result_host[base + 0]
        total_abs_ref += result_host[base + 1]
        worst_violation = max(worst_violation, result_host[base + 2])
        any_out_nz = max(any_out_nz, result_host[base + 3])
        any_ref_nz = max(any_ref_nz, result_host[base + 4])

    return VerifyMetrics(
        total_abs_diff,
        total_abs_ref,
        worst_violation,
        any_out_nz,
        any_ref_nz,
    )


# ---------------------------------------------------------------------------
# Helper to fill a device buffer using the GPU fill kernel.
# ---------------------------------------------------------------------------
def fill_on_device[
    dtype: DType,
](
    ctx: DeviceContext,
    buf: DeviceBuffer[dtype],
    length: Int,
    val: Scalar[dtype],
) raises:
    comptime FILL_BLOCK = 256
    var fill_grid = ceildiv(length, FILL_BLOCK)
    comptime kernel = _fill_buffer[dtype]
    ctx.enqueue_function[kernel](
        buf,
        Int32(length),
        val,
        grid_dim=fill_grid,
        block_dim=FILL_BLOCK,
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
comptime test_dtype = DType.bfloat16


def test_identical_nonzero(ctx: DeviceContext) raises:
    """Both buffers contain 1.0 — expect zero diff, no violation."""
    comptime N = 128
    comptime NUM_BLOCKS = 1
    comptime BLOCK_SIZE = 128

    var out_buf = ctx.enqueue_create_buffer[test_dtype](N)
    var ref_buf = ctx.enqueue_create_buffer[test_dtype](N)
    fill_on_device(ctx, out_buf, N, Scalar[test_dtype](1.0))
    fill_on_device(ctx, ref_buf, N, Scalar[test_dtype](1.0))

    var m = run_verify_kernel[test_dtype, NUM_BLOCKS, BLOCK_SIZE](
        ctx, out_buf, ref_buf, N, 1e-5, 1.6e-2
    )

    assert_equal(m.abs_diff_sum, 0.0, msg="abs_diff_sum should be 0")
    assert_equal(
        m.abs_ref_sum, Float32(N), msg=String(t"abs_ref_sum should be {N}")
    )
    assert_true(m.max_violation <= 0, "max_violation should be <= 0")
    assert_equal(m.out_nz, 1.0, msg="out_nz should be 1")
    assert_equal(m.ref_nz, 1.0, msg="ref_nz should be 1")
    print("PASS: test_identical_nonzero")


def test_both_zeros(ctx: DeviceContext) raises:
    """Both buffers all zeros — diff is zero, both nz flags are 0."""
    comptime N = 128
    comptime NUM_BLOCKS = 1
    comptime BLOCK_SIZE = 128

    var out_buf = ctx.enqueue_create_buffer[test_dtype](N)
    var ref_buf = ctx.enqueue_create_buffer[test_dtype](N)
    fill_on_device(ctx, out_buf, N, Scalar[test_dtype](0.0))
    fill_on_device(ctx, ref_buf, N, Scalar[test_dtype](0.0))

    var m = run_verify_kernel[test_dtype, NUM_BLOCKS, BLOCK_SIZE](
        ctx, out_buf, ref_buf, N, 1e-5, 1.6e-2
    )

    assert_equal(m.abs_diff_sum, 0.0, msg="abs_diff_sum should be 0")
    assert_equal(m.abs_ref_sum, 0.0, msg="abs_ref_sum should be 0")
    assert_equal(m.out_nz, 0.0, msg="out_nz should be 0")
    assert_equal(m.ref_nz, 0.0, msg="ref_nz should be 0")
    print("PASS: test_both_zeros")


def test_known_constant_diff(ctx: DeviceContext) raises:
    """Output=1.5, reference=1.0 — abs_diff should be N*0.5, violation > 0."""
    comptime N = 128
    comptime NUM_BLOCKS = 1
    comptime BLOCK_SIZE = 128

    var out_buf = ctx.enqueue_create_buffer[test_dtype](N)
    var ref_buf = ctx.enqueue_create_buffer[test_dtype](N)
    # 1.5 and 1.0 are both exactly representable in bfloat16.
    fill_on_device(ctx, out_buf, N, Scalar[test_dtype](1.5))
    fill_on_device(ctx, ref_buf, N, Scalar[test_dtype](1.0))

    var m = run_verify_kernel[test_dtype, NUM_BLOCKS, BLOCK_SIZE](
        ctx, out_buf, ref_buf, N, 1e-5, 1.6e-2
    )

    # Each element: |1.5 - 1.0| = 0.5
    var expected_abs_diff = Float32(N) * 0.5
    assert_equal(
        m.abs_diff_sum,
        expected_abs_diff,
        msg=String(t"abs_diff_sum should be N*0.5"),
    )
    assert_equal(
        m.abs_ref_sum, Float32(N), msg=String(t"abs_ref_sum should be {N}")
    )
    # violation = 0.5 - (1e-5 + 1.6e-2 * 1.0) > 0
    assert_true(m.max_violation > 0, "max_violation should be > 0")
    assert_equal(m.out_nz, 1.0, msg="out_nz should be 1")
    assert_equal(m.ref_nz, 1.0, msg="ref_nz should be 1")
    print("PASS: test_known_constant_diff")


def test_within_tolerance(ctx: DeviceContext) raises:
    """Output=1.0, reference=1.0 with generous tolerances — no violation."""
    comptime N = 128
    comptime NUM_BLOCKS = 1
    comptime BLOCK_SIZE = 128

    var out_buf = ctx.enqueue_create_buffer[test_dtype](N)
    var ref_buf = ctx.enqueue_create_buffer[test_dtype](N)
    fill_on_device(ctx, out_buf, N, Scalar[test_dtype](1.0))
    fill_on_device(ctx, ref_buf, N, Scalar[test_dtype](1.0))

    # With atol=0.1, rtol=0.1, identical buffers should have violation << 0.
    var m = run_verify_kernel[test_dtype, NUM_BLOCKS, BLOCK_SIZE](
        ctx, out_buf, ref_buf, N, 0.1, 0.1
    )

    # violation = |1.0-1.0| - (0.1 + 0.1*1.0) = -0.2
    assert_true(m.max_violation <= 0, "max_violation should be <= 0")
    print("PASS: test_within_tolerance")


def test_output_zero_ref_nonzero(ctx: DeviceContext) raises:
    """Output all zeros, reference all 1.0 — out_nz=0, ref_nz=1."""
    comptime N = 128
    comptime NUM_BLOCKS = 1
    comptime BLOCK_SIZE = 128

    var out_buf = ctx.enqueue_create_buffer[test_dtype](N)
    var ref_buf = ctx.enqueue_create_buffer[test_dtype](N)
    fill_on_device(ctx, out_buf, N, Scalar[test_dtype](0.0))
    fill_on_device(ctx, ref_buf, N, Scalar[test_dtype](1.0))

    var m = run_verify_kernel[test_dtype, NUM_BLOCKS, BLOCK_SIZE](
        ctx, out_buf, ref_buf, N, 1e-5, 1.6e-2
    )

    assert_equal(m.out_nz, 0.0, msg="out_nz should be 0")
    assert_equal(m.ref_nz, 1.0, msg="ref_nz should be 1")
    # abs_diff = N * 1.0
    assert_equal(
        m.abs_diff_sum, Float32(N), msg=String(t"abs_diff_sum should be {N}")
    )
    print("PASS: test_output_zero_ref_nonzero")


def test_large_buffer_multi_block(ctx: DeviceContext) raises:
    """Buffer larger than one block — exercises grid-stride loop and
    multi-block partial reduction."""
    comptime N = 32768
    comptime NUM_BLOCKS = 4
    comptime BLOCK_SIZE = 256

    var out_buf = ctx.enqueue_create_buffer[test_dtype](N)
    var ref_buf = ctx.enqueue_create_buffer[test_dtype](N)
    # 2.0 and 1.0 are exactly representable in bfloat16.
    fill_on_device(ctx, out_buf, N, Scalar[test_dtype](2.0))
    fill_on_device(ctx, ref_buf, N, Scalar[test_dtype](1.0))

    var m = run_verify_kernel[test_dtype, NUM_BLOCKS, BLOCK_SIZE](
        ctx, out_buf, ref_buf, N, 1e-5, 1.6e-2
    )

    # Each element: |2.0 - 1.0| = 1.0
    var expected_abs_diff = Float32(N)
    assert_equal(
        m.abs_diff_sum,
        expected_abs_diff,
        msg=String(t"abs_diff_sum should be N for large buffer"),
    )
    assert_equal(
        m.abs_ref_sum,
        Float32(N),
        msg=String(t"abs_ref_sum should be N for large buffer"),
    )
    # violation = 1.0 - (1e-5 + 1.6e-2 * 1.0) ≈ 0.984 > 0
    assert_true(
        m.max_violation > 0,
        String(t"max_violation should be > 0 for large buffer"),
    )
    assert_equal(m.out_nz, 1.0, msg="out_nz should be 1")
    assert_equal(m.ref_nz, 1.0, msg="ref_nz should be 1")
    print("PASS: test_large_buffer_multi_block")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() raises:
    with DeviceContext() as ctx:
        test_identical_nonzero(ctx)
        test_both_zeros(ctx)
        test_known_constant_diff(ctx)
        test_within_tolerance(ctx)
        test_output_zero_ref_nonzero(ctx)
        test_large_buffer_multi_block(ctx)
