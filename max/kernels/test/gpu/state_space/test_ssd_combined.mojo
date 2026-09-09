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

from max.gpu.host import DeviceContext
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from std.math import exp, exp2, log
from std.random import rand
from state_space.selective_scan import (
    ssd_combined_cpu,
    ssd_combined_gpu,
)
from std.testing import TestSuite, assert_almost_equal

from std.utils.index import Index

comptime MAX_DSTATE = 16
comptime LOG2E = 1.4426950408889634


@always_inline
def softplus_ref(val: Float32) -> Float32:
    """Reference softplus: log(1 + exp(x)) with numerical stability."""
    if val > 20.0:
        return val
    return log(Float32(1.0) + exp(val))


@always_inline
def silu_ref(val: Float32) -> Float32:
    """Reference SiLU: x * sigmoid(x) = x / (1 + exp(-x))."""
    if val < -20.0:
        return 0.0
    return val / (Float32(1.0) + exp(-val))


def run_ssd_combined_gpu[
    dtype: DType,
    DSTATE: Int,
    has_D: Bool = True,
    has_z: Bool = True,
    has_delta_bias: Bool = True,
    delta_softplus: Bool = False,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    n_groups: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Test SSD combined GPU kernel against CPU reference."""
    comptime assert DSTATE <= 16, "DSTATE exceeds kernel limit"
    comptime dstate = DSTATE

    var group_size = dim // n_groups
    var chunk_size = 2048
    var n_chunks = (seqlen + chunk_size - 1) // chunk_size

    # Allocate host memory
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)

    var output_cpu_h = alloc[Scalar[dtype]](batch * dim * seqlen)
    var output_gpu_h = alloc[Scalar[dtype]](batch * dim * seqlen)
    var x_cpu_h = alloc[Scalar[dtype]](batch * dim * n_chunks * 2 * dstate)
    var x_gpu_h = alloc[Scalar[dtype]](batch * dim * n_chunks * 2 * dstate)
    var out_z_cpu_h = alloc[Scalar[dtype]](batch * dim * seqlen)
    var out_z_gpu_h = alloc[Scalar[dtype]](batch * dim * seqlen)
    var residual_h = alloc[Scalar[dtype]](batch * dim * seqlen)
    var u_h = alloc[Scalar[dtype]](batch * dim * seqlen)
    var delta_h = alloc[Scalar[dtype]](batch * dim * seqlen)
    var A_h = alloc[Scalar[dtype]](dim * dstate)
    var B_h = alloc[Scalar[dtype]](batch * n_groups * dstate * seqlen)
    var C_h = alloc[Scalar[dtype]](batch * n_groups * dstate * seqlen)
    var D_size = dim if has_D else 0
    var D_h = alloc[Scalar[dtype]](max(D_size, 1))
    var z_size = batch * dim * seqlen if has_z else 0
    var z_h = alloc[Scalar[dtype]](max(z_size, 1))
    var delta_bias_size = dim if has_delta_bias else 0
    var delta_bias_h = alloc[Scalar[dtype]](max(delta_bias_size, 1))
    var gamma_h = alloc[Scalar[dtype]](dim)

    # Create LayoutTensors for initialization
    var u_init = LayoutTensor[dtype, layout_3d](
        u_h, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var delta_init = LayoutTensor[dtype, layout_3d](
        delta_h, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var residual_init = LayoutTensor[dtype, layout_3d](
        residual_h,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )
    var A_init = LayoutTensor[dtype, layout_2d](
        A_h, RuntimeLayout[layout_2d].row_major(Index(dim, dstate))
    )
    var B_init = LayoutTensor[dtype, layout_4d](
        B_h,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_groups, dstate, seqlen)
        ),
    )
    var C_init = LayoutTensor[dtype, layout_4d](
        C_h,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_groups, dstate, seqlen)
        ),
    )
    var D_init = LayoutTensor[dtype, layout_1d](
        D_h, RuntimeLayout[layout_1d].row_major(Index(D_size))
    )
    var z_init = LayoutTensor[dtype, layout_3d](
        z_h,
        RuntimeLayout[layout_3d].row_major(
            Index(
                batch if has_z else 0,
                dim if has_z else 0,
                seqlen if has_z else 0,
            )
        ),
    )
    var delta_bias_init = LayoutTensor[dtype, layout_1d](
        delta_bias_h, RuntimeLayout[layout_1d].row_major(Index(delta_bias_size))
    )
    var gamma_init = LayoutTensor[dtype, layout_1d](
        gamma_h, RuntimeLayout[layout_1d].row_major(Index(dim))
    )

    # Initialize with random data
    rand[dtype](u_init.ptr, u_init.size())
    rand[dtype](delta_init.ptr, delta_init.size())
    rand[dtype](residual_init.ptr, residual_init.size())
    rand[dtype](A_init.ptr, A_init.size())
    rand[dtype](B_init.ptr, B_init.size())
    rand[dtype](C_init.ptr, C_init.size())
    if has_D:
        rand[dtype](D_init.ptr, D_init.size())
    if has_z:
        rand[dtype](z_init.ptr, z_init.size())
    if has_delta_bias:
        rand[dtype](delta_bias_init.ptr, delta_bias_init.size())
    rand[dtype](gamma_init.ptr, gamma_init.size())

    # Initialize gamma to positive values
    for i in range(dim):
        gamma_h[i] = abs(gamma_h[i]) + Scalar[dtype](0.1)

    # Allocate GPU memory (only for GPU kernel)
    var output_gpu_gpu = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var x_gpu_gpu = ctx.enqueue_create_buffer[dtype](
        batch * dim * n_chunks * 2 * dstate
    )
    var out_z_gpu_gpu = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var residual_gpu = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var u_gpu = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var delta_gpu = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var A_gpu = ctx.enqueue_create_buffer[dtype](dim * dstate)
    var B_gpu = ctx.enqueue_create_buffer[dtype](
        batch * n_groups * dstate * seqlen
    )
    var C_gpu = ctx.enqueue_create_buffer[dtype](
        batch * n_groups * dstate * seqlen
    )
    var D_gpu = ctx.enqueue_create_buffer[dtype](max(D_size, 1))
    var z_gpu = ctx.enqueue_create_buffer[dtype](max(z_size, 1))
    var delta_bias_gpu = ctx.enqueue_create_buffer[dtype](
        max(delta_bias_size, 1)
    )
    var gamma_gpu = ctx.enqueue_create_buffer[dtype](dim)

    # Copy input data to GPU
    with ctx.push_context():
        ctx.enqueue_copy[dtype](residual_gpu, residual_h)
        ctx.enqueue_copy[dtype](u_gpu, u_h)
        ctx.enqueue_copy[dtype](delta_gpu, delta_h)
        ctx.enqueue_copy[dtype](A_gpu, A_h)
        ctx.enqueue_copy[dtype](B_gpu, B_h)
        ctx.enqueue_copy[dtype](C_gpu, C_h)
        if has_D:
            ctx.enqueue_copy[dtype](D_gpu, D_h)
        if has_z:
            ctx.enqueue_copy[dtype](z_gpu, z_h)
        if has_delta_bias:
            ctx.enqueue_copy[dtype](delta_bias_gpu, delta_bias_h)
        ctx.enqueue_copy[dtype](gamma_gpu, gamma_h)

    # Create CPU TileTensors for CPU kernel (using host memory)
    var output_cpu_lt = TileTensor(output_cpu_h, row_major(batch, dim, seqlen))
    var x_cpu_lt = TileTensor(
        x_cpu_h, row_major(batch, dim, n_chunks, 2 * dstate)
    )
    var out_z_cpu_lt = TileTensor(out_z_cpu_h, row_major(batch, dim, seqlen))
    var residual_cpu_lt = TileTensor(residual_h, row_major(batch, dim, seqlen))
    var u_cpu_lt = TileTensor(u_h, row_major(batch, dim, seqlen))
    var delta_cpu_lt = TileTensor(delta_h, row_major(batch, dim, seqlen))
    var A_cpu_lt = TileTensor(A_h, row_major(dim, dstate))
    var B_cpu_lt = TileTensor(B_h, row_major(batch, n_groups, dstate, seqlen))
    var C_cpu_lt = TileTensor(C_h, row_major(batch, n_groups, dstate, seqlen))
    var D_cpu_lt = TileTensor(D_h, row_major(D_size))
    var z_cpu_lt = TileTensor(
        z_h,
        row_major(
            (
                batch if has_z else 0,
                dim if has_z else 0,
                seqlen if has_z else 0,
            )
        ),
    )
    var delta_bias_cpu_lt = TileTensor(delta_bias_h, row_major(delta_bias_size))
    var gamma_cpu_lt = TileTensor(gamma_h, row_major(dim))

    # Create GPU TileTensors for GPU kernel
    var output_gpu_lt = TileTensor(
        output_gpu_gpu, row_major(batch, dim, seqlen)
    )
    var x_gpu_lt = TileTensor(
        x_gpu_gpu, row_major(batch, dim, n_chunks, 2 * dstate)
    )
    var out_z_gpu_lt = TileTensor(out_z_gpu_gpu, row_major(batch, dim, seqlen))
    var residual_gpu_lt = TileTensor(
        residual_gpu, row_major(batch, dim, seqlen)
    )
    var u_gpu_lt = TileTensor(u_gpu, row_major(batch, dim, seqlen))
    var delta_gpu_lt = TileTensor(delta_gpu, row_major(batch, dim, seqlen))
    var A_gpu_lt = TileTensor(A_gpu, row_major(dim, dstate))
    var B_gpu_lt = TileTensor(B_gpu, row_major(batch, n_groups, dstate, seqlen))
    var C_gpu_lt = TileTensor(C_gpu, row_major(batch, n_groups, dstate, seqlen))
    var D_gpu_lt = TileTensor(D_gpu, row_major(D_size))
    var z_gpu_lt = TileTensor(
        z_gpu,
        row_major(
            (
                batch if has_z else 0,
                dim if has_z else 0,
                seqlen if has_z else 0,
            )
        ),
    )
    var delta_bias_gpu_lt = TileTensor(
        delta_bias_gpu, row_major(delta_bias_size)
    )
    var gamma_gpu_lt = TileTensor(gamma_gpu, row_major(dim))

    var epsilon = Float32(0.001)
    var weight_offset = Scalar[dtype](0.0)

    # Run CPU kernel with host tensors
    ssd_combined_cpu[
        dtype,
        DSTATE,
        output_cpu_lt.LayoutType,
        x_cpu_lt.LayoutType,
        out_z_cpu_lt.LayoutType,
        residual_cpu_lt.LayoutType,
        u_cpu_lt.LayoutType,
        delta_cpu_lt.LayoutType,
        A_cpu_lt.LayoutType,
        B_cpu_lt.LayoutType,
        C_cpu_lt.LayoutType,
        D_cpu_lt.LayoutType,
        z_cpu_lt.LayoutType,
        delta_bias_cpu_lt.LayoutType,
        gamma_cpu_lt.LayoutType,
    ](
        batch,
        dim,
        seqlen,
        group_size,
        Int8(1) if delta_softplus else Int8(0),
        output_cpu_lt,
        x_cpu_lt,
        out_z_cpu_lt,
        residual_cpu_lt,
        u_cpu_lt,
        delta_cpu_lt,
        A_cpu_lt,
        B_cpu_lt,
        C_cpu_lt,
        D_cpu_lt,
        z_cpu_lt,
        delta_bias_cpu_lt,
        gamma_cpu_lt,
        epsilon.cast[dtype](),
        weight_offset,
    )

    # Run GPU kernel
    var total_batch_dim = batch * dim
    comptime BLOCK_SIZE = 128
    from std.math import ceildiv

    var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)

    var compiled_kernel = ctx.compile_function[
        ssd_combined_gpu[
            dtype,
            DSTATE,
            output_gpu_lt.LayoutType,
            x_gpu_lt.LayoutType,
            out_z_gpu_lt.LayoutType,
            residual_gpu_lt.LayoutType,
            u_gpu_lt.LayoutType,
            delta_gpu_lt.LayoutType,
            A_gpu_lt.LayoutType,
            B_gpu_lt.LayoutType,
            C_gpu_lt.LayoutType,
            D_gpu_lt.LayoutType,
            z_gpu_lt.LayoutType,
            delta_bias_gpu_lt.LayoutType,
            gamma_gpu_lt.LayoutType,
        ]
    ]()

    ctx.enqueue_function(
        compiled_kernel,
        Int32(total_batch_dim),
        Int32(batch),
        Int32(dim),
        Int32(seqlen),
        Int32(group_size),
        Int8(1) if delta_softplus else Int8(0),
        output_gpu_lt,
        x_gpu_lt,
        out_z_gpu_lt,
        residual_gpu_lt,
        u_gpu_lt,
        delta_gpu_lt,
        A_gpu_lt,
        B_gpu_lt,
        C_gpu_lt,
        D_gpu_lt,
        z_gpu_lt,
        delta_bias_gpu_lt,
        gamma_gpu_lt,
        epsilon.cast[dtype](),
        weight_offset,
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,),
    )

    ctx.synchronize()

    # Copy GPU results back (CPU results are already in host memory)
    with ctx.push_context():
        ctx.enqueue_copy[dtype](output_gpu_h, output_gpu_gpu)
        ctx.enqueue_copy[dtype](out_z_gpu_h, out_z_gpu_gpu)
    ctx.synchronize()

    # Compare GPU output vs CPU output
    var flattened_size = batch * dim * seqlen
    for i in range(flattened_size):
        assert_almost_equal(
            output_cpu_h[i],
            output_gpu_h[i],
            rtol=rtol,
        )

    # Compare out_z GPU vs CPU when z gating is enabled
    if has_z:
        for i in range(flattened_size):
            assert_almost_equal(
                out_z_cpu_h[i],
                out_z_gpu_h[i],
                rtol=rtol,
            )

    # Reference implementation for numerical verification
    var output_ref_h = alloc[Scalar[dtype]](flattened_size)
    var out_z_ref_h = alloc[Scalar[dtype]](flattened_size)
    for i in range(flattened_size):
        output_ref_h[i] = Scalar[dtype](0)
        out_z_ref_h[i] = Scalar[dtype](0)

    for b_idx in range(batch):
        for d_idx in range(dim):
            var group_id = d_idx // group_size

            # Pre-load A values with LOG2E scaling (matches kernel)
            var A_ref = SIMD[.float32, MAX_DSTATE](0.0)
            for n in range(dstate):
                A_ref[n] = Float32(A_h[d_idx * dstate + n]) * LOG2E

            # Load per-dim scalars
            var gamma_val = Float32(gamma_h[d_idx])
            var D_val = Float32(0)
            if has_D:
                D_val = Float32(D_h[d_idx])
            var delta_bias_val = Float32(0)
            if has_delta_bias:
                delta_bias_val = Float32(delta_bias_h[d_idx])
            var weight_offset_val = Float32(weight_offset)

            # Initialize state to zero
            var state_ref = SIMD[.float32, MAX_DSTATE](0.0)

            for t in range(seqlen):
                var off_3d = b_idx * dim * seqlen + d_idx * seqlen + t
                var u_val = Float32(u_h[off_3d])
                var delta_val = Float32(delta_h[off_3d])
                var residual_val = Float32(residual_h[off_3d])

                if has_delta_bias:
                    delta_val += delta_bias_val
                if delta_softplus:
                    delta_val = softplus_ref(delta_val)

                var delta_u = delta_val * u_val

                # Load B, C values
                var B_vals = SIMD[.float32, MAX_DSTATE](0.0)
                var C_vals = SIMD[.float32, MAX_DSTATE](0.0)
                for n in range(dstate):
                    var bc_offset = (
                        b_idx * n_groups * dstate * seqlen
                        + group_id * dstate * seqlen
                        + n * seqlen
                        + t
                    )
                    B_vals[n] = Float32(B_h[bc_offset])
                    C_vals[n] = Float32(C_h[bc_offset])

                # State update: state = state * exp2(A * delta) + B * delta * u
                var a_t = exp2(A_ref * delta_val)
                var b_t = B_vals * delta_u
                state_ref = state_ref * a_t + b_t

                # Compute selective scan output
                var ss_output = (state_ref * C_vals).reduce_add()

                if has_D:
                    ss_output += D_val * u_val

                # Combine with residual and apply gamma scaling
                var combined = residual_val + ss_output
                var normalized = combined * (gamma_val + weight_offset_val)

                if has_z:
                    var z_val = Float32(z_h[off_3d])
                    var out_z_val = normalized * silu_ref(z_val)
                    out_z_ref_h[off_3d] = Scalar[dtype](out_z_val)
                    normalized = out_z_val

                output_ref_h[off_3d] = Scalar[dtype](normalized)

    # Compare CPU output vs reference
    for i in range(flattened_size):
        assert_almost_equal(
            output_cpu_h[i],
            output_ref_h[i],
            rtol=rtol,
        )

    # Verify out_z against reference when z gating is enabled
    if has_z:
        for i in range(flattened_size):
            assert_almost_equal(
                out_z_cpu_h[i],
                out_z_ref_h[i],
                rtol=rtol,
            )

    output_ref_h.free()
    out_z_ref_h.free()

    # Cleanup
    output_cpu_h.free()
    output_gpu_h.free()
    x_cpu_h.free()
    x_gpu_h.free()
    out_z_cpu_h.free()
    out_z_gpu_h.free()
    residual_h.free()
    u_h.free()
    delta_h.free()
    A_h.free()
    B_h.free()
    C_h.free()
    D_h.free()
    z_h.free()
    delta_bias_h.free()
    gamma_h.free()
    # Device buffers are automatically freed when they go out of scope
    _ = output_gpu_gpu^
    _ = x_gpu_gpu^
    _ = out_z_gpu_gpu^
    _ = residual_gpu^
    _ = u_gpu^
    _ = delta_gpu^
    _ = A_gpu^
    _ = B_gpu^
    _ = C_gpu^
    _ = D_gpu^
    _ = z_gpu^
    _ = delta_bias_gpu^
    _ = gamma_gpu^


def test_ssd_combined_gpu_basic() raises:
    """Test basic ssd_combined on GPU."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_ssd_combined_gpu[
        DType.float32,
        4,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=2, dim=4, seqlen=8, n_groups=1, ctx=ctx)


def test_ssd_combined_gpu_without_D() raises:
    """Test ssd_combined on GPU without D."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_ssd_combined_gpu[
        DType.float32,
        4,  # DSTATE
        has_D=False,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=2, dim=4, seqlen=8, n_groups=1, ctx=ctx)


def test_ssd_combined_gpu_without_z() raises:
    """Test ssd_combined on GPU without z."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_ssd_combined_gpu[
        DType.float32,
        4,  # DSTATE
        has_D=True,
        has_z=False,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=2, dim=4, seqlen=8, n_groups=1, ctx=ctx)


def test_ssd_combined_gpu_without_delta_bias() raises:
    """Test ssd_combined on GPU without delta_bias."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_ssd_combined_gpu[
        DType.float32,
        4,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=False,
        delta_softplus=False,
    ](batch=2, dim=4, seqlen=8, n_groups=1, ctx=ctx)


def test_ssd_combined_gpu_with_delta_softplus() raises:
    """Test ssd_combined on GPU with delta_softplus."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_ssd_combined_gpu[
        DType.float32,
        4,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=True,
    ](batch=2, dim=4, seqlen=8, n_groups=1, ctx=ctx)


def test_ssd_combined_gpu_larger_shapes() raises:
    """Test ssd_combined on GPU with larger shapes."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_ssd_combined_gpu[
        DType.float32,
        8,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=4, dim=8, seqlen=16, n_groups=1, ctx=ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
