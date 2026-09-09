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
"""GPU tests for RMSNorm with fused residual connection."""

from std.math import sqrt
from max.gpu.host import DeviceContext
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    row_major,
)
from std.random import rand, Random
from state_space.rms_norm_fused_residual import rms_norm_fused_residual_gpu
from std.testing import TestSuite, assert_almost_equal

from std.utils.index import Index, IndexList


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def compute_rms_ref[
    dtype: DType
](data_ptr: ImmPointer[Scalar[dtype], _], size: Int, eps: Float32) -> Scalar[
    DType.float32
]:
    """Compute reference RMS value."""
    var sum_of_squares = Float32()
    for i in range(size):
        var d = data_ptr[i].cast[.float32]()
        sum_of_squares += d * d
    return sqrt((sum_of_squares / Float32(size)) + eps)


def run_rms_norm_fused_residual_gpu[
    dtype: DType, rank: Int
](
    ctx: DeviceContext,
    shape: IndexList[rank],
    rtol: Float64 = 0.01,
    dropout_p: Float64 = 0.0,
    seed: UInt64 = 0,
) raises:
    """Test rms_norm_fused_residual GPU implementation."""
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    # Allocate host memory
    var input_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var residual_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var output_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var residual_output_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)

    # Initialize input data
    rand[dtype](input_h.unsafe_ptr(), rows * cols)
    rand[dtype](residual_h.unsafe_ptr(), rows * cols)

    # Scale inputs to reasonable range
    for i in range(rows * cols):
        input_h[i] = input_h[i] * Scalar[dtype](0.5)
        residual_h[i] = residual_h[i] * Scalar[dtype](0.5)

    # Initialize gamma (weight)
    for i in range(cols):
        gamma_h[i] = Scalar[dtype](Float64(i + cols) / Float64(cols))

    # Allocate device memory
    var input_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var residual_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var output_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var residual_output_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)

    # Copy to device
    ctx.enqueue_copy(input_d, input_h)
    ctx.enqueue_copy(residual_d, residual_h)
    ctx.enqueue_copy(output_d, output_h)
    ctx.enqueue_copy(residual_output_d, residual_output_h)
    ctx.enqueue_copy(gamma_d, gamma_h)

    # Create device LayoutTensors
    comptime layout_nd = Layout.row_major[rank]()

    var input_tensor = LayoutTensor[dtype, layout_nd](
        input_d,
        RuntimeLayout[layout_nd].row_major(shape),
    )
    var residual_tensor = LayoutTensor[dtype, layout_nd](
        residual_d,
        RuntimeLayout[layout_nd].row_major(shape),
    )
    var output_tensor = LayoutTensor[dtype, layout_nd](
        output_d,
        RuntimeLayout[layout_nd].row_major(shape),
    )
    var residual_output_tensor = LayoutTensor[dtype, layout_nd](
        residual_output_d,
        RuntimeLayout[layout_nd].row_major(shape),
    )
    var gamma_tensor = TileTensor(gamma_d, row_major(cols))

    var epsilon = Float32(1e-5)
    var weight_offset = Scalar[dtype](0.0)

    # Define input functions
    @__copy_capture(input_tensor)
    @always_inline
    @__parameter
    def input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        return input_tensor.load[width=width](rebind[IndexList[rank]](coords))

    @__copy_capture(residual_tensor)
    @always_inline
    @__parameter
    def residual_input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        return residual_tensor.load[width=width](
            rebind[IndexList[rank]](coords)
        )

    # Define output functions
    @__copy_capture(output_tensor)
    @always_inline
    @__parameter
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        output_tensor.store[width=width](coords, val)

    @__copy_capture(residual_output_tensor)
    @always_inline
    @__parameter
    def residual_output_fn[
        width: SIMDLength, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        residual_output_tensor.store[width=width](coords, val)

    var dropout_p_scalar = Scalar[dtype](dropout_p)

    # Run the GPU kernel
    rms_norm_fused_residual_gpu[
        input_fn,
        residual_input_fn,
        residual_output_fn,
        output_fn,
        multiply_before_cast=True,
    ](
        shape,
        gamma_tensor,
        epsilon,
        weight_offset,
        ctx,
        dropout_p=dropout_p_scalar,
        seed=seed,
    )

    # Copy results back
    ctx.enqueue_copy(output_h, output_d)
    ctx.enqueue_copy(residual_output_h, residual_output_d)
    ctx.synchronize()

    # Compute dropout scale
    var dropout_scale = Scalar[dtype](1.0)
    var zero_scalar = Scalar[dtype](0.0)
    if dropout_p_scalar > zero_scalar:
        var one_scalar = Scalar[dtype](1.0)
        dropout_scale = one_scalar / (one_scalar - dropout_p_scalar)

    # Verify results
    for r in range(rows):
        # Compute expected residual output: dropout(input) + residual
        var sum_ptr = ctx.enqueue_create_host_buffer[dtype](cols)
        for c in range(cols):
            var idx = r * cols + c
            var input_val = input_h[idx]

            # Apply the same deterministic dropout as the kernel
            if dropout_p_scalar > zero_scalar:
                var element_offset = UInt64(r) * UInt64(cols) + UInt64(c)
                var generator = Random(seed=seed, offset=element_offset)
                var rng = generator.step_uniform()
                var rng_val = rng[0].cast[dtype]()
                if rng_val >= dropout_p_scalar:
                    input_val = input_val * dropout_scale
                else:
                    input_val = zero_scalar

            sum_ptr[c] = input_val + residual_h[idx]

        # Verify residual output
        for c in range(cols):
            var idx = r * cols + c
            assert_almost_equal(sum_ptr[c], residual_output_h[idx], rtol=rtol)

        # Compute RMS of the sum
        var rms_val = compute_rms_ref(sum_ptr.unsafe_ptr(), cols, epsilon)

        # Verify normalized output
        for c in range(cols):
            var idx = r * cols + c
            var sum_val = sum_ptr[c].cast[.float32]()
            var expected_norm = (sum_val / rms_val).cast[dtype]() * (
                gamma_h[c] + weight_offset
            )
            assert_almost_equal(expected_norm, output_h[idx], rtol=rtol)


# =============================================================================
# Test functions
# =============================================================================


def test_rms_norm_fused_residual_gpu_float32_2d() raises:
    """Test rms_norm_fused_residual GPU with float32 and 2D shape."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_rms_norm_fused_residual_gpu[.float32](ctx, Index(4, 16), rtol=1e-3)


def test_rms_norm_fused_residual_gpu_float32_small() raises:
    """Test rms_norm_fused_residual GPU with small dimensions."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_rms_norm_fused_residual_gpu[.float32](ctx, Index(2, 8), rtol=1e-3)


def test_rms_norm_fused_residual_gpu_float32_large_cols() raises:
    """Test rms_norm_fused_residual GPU with larger column count."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_rms_norm_fused_residual_gpu[.float32](ctx, Index(2, 128), rtol=1e-3)


def test_rms_norm_fused_residual_gpu_float32_3d() raises:
    """Test rms_norm_fused_residual GPU with 3D shape."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_rms_norm_fused_residual_gpu[.float32](ctx, Index(2, 3, 16), rtol=1e-3)


def test_rms_norm_fused_residual_gpu_float32_larger() raises:
    """Test rms_norm_fused_residual GPU with larger dimensions."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_rms_norm_fused_residual_gpu[.float32](ctx, Index(4, 8, 64), rtol=1e-3)


def test_rms_norm_fused_residual_gpu_bfloat16() raises:
    """Test rms_norm_fused_residual GPU with bfloat16."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_rms_norm_fused_residual_gpu[.bfloat16](ctx, Index(4, 16), rtol=1e-2)


def test_rms_norm_fused_residual_gpu_float16() raises:
    """Test rms_norm_fused_residual GPU with float16."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_rms_norm_fused_residual_gpu[.float16](ctx, Index(4, 16), rtol=1e-2)


def test_rms_norm_fused_residual_gpu_many_rows() raises:
    """Test rms_norm_fused_residual GPU with many rows."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_rms_norm_fused_residual_gpu[.float32](ctx, Index(32, 64), rtol=1e-3)


def test_rms_norm_fused_residual_gpu_large_cols_loop() raises:
    """Test rms_norm_fused_residual GPU when num_cols > block_dim * simd_width.

    This exercises the loop in the first stage of the GPU kernel that populates
    shared memory, ensuring all columns are written before the normalization
    subkernel reads them.
    """
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    # 4096 columns is large enough to require multiple loop iterations per
    # thread on most GPU configurations.
    run_rms_norm_fused_residual_gpu[.float32](ctx, Index(2, 4096), rtol=1e-3)


def test_rms_norm_fused_residual_gpu_bfloat16_large_cols_loop() raises:
    """Test fused residual GPU with bfloat16 and large cols requiring loop."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_rms_norm_fused_residual_gpu[.bfloat16](ctx, Index(2, 4096), rtol=1e-2)


# =============================================================================
# Dropout tests (dropout_p > 0)
# =============================================================================


def test_rms_norm_fused_residual_gpu_dropout_float32_2d() raises:
    """Test rms_norm_fused_residual GPU with dropout enabled (float32, 2D)."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_rms_norm_fused_residual_gpu[.float32](
        ctx, Index(4, 16), rtol=1e-3, dropout_p=0.3, seed=42
    )


def test_rms_norm_fused_residual_gpu_dropout_float32_3d() raises:
    """Test rms_norm_fused_residual GPU with dropout enabled (float32, 3D)."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_rms_norm_fused_residual_gpu[.float32](
        ctx, Index(2, 3, 16), rtol=1e-3, dropout_p=0.5, seed=123
    )


def test_rms_norm_fused_residual_gpu_dropout_float32_large_cols() raises:
    """Test dropout path with large columns requiring loop iterations."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_rms_norm_fused_residual_gpu[.float32](
        ctx, Index(2, 4096), rtol=1e-3, dropout_p=0.1, seed=7
    )


def test_rms_norm_fused_residual_gpu_dropout_bfloat16() raises:
    """Test rms_norm_fused_residual GPU with dropout enabled (bfloat16)."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_rms_norm_fused_residual_gpu[.bfloat16](
        ctx, Index(4, 64), rtol=1e-2, dropout_p=0.2, seed=99
    )


def test_rms_norm_fused_residual_gpu_dropout_float16() raises:
    """Test rms_norm_fused_residual GPU with dropout enabled (float16)."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_rms_norm_fused_residual_gpu[.float16](
        ctx, Index(4, 64), rtol=1e-2, dropout_p=0.4, seed=55
    )
