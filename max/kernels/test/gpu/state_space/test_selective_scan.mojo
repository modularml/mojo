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

from std.math import ceildiv
from max.gpu.host import DeviceContext
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from std.random import rand
from state_space.selective_scan import (
    selective_scan_fwd_cpu,
    selective_scan_fwd_gpu,
    selective_scan_update_cpu,
    selective_scan_update_gpu,
    Strides1D,
    Strides2D,
    Strides3D,
    Strides4D,
)
from std.testing import TestSuite, assert_almost_equal

from std.utils.index import Index


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def run_selective_scan_gpu[
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
    """Test selective scan GPU kernel against CPU reference."""
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

    # Initialize output buffers to zero
    for i in range(batch * dim * seqlen):
        output_cpu_h[i] = Scalar[dtype](0)
        output_gpu_h[i] = Scalar[dtype](0)
        out_z_cpu_h[i] = Scalar[dtype](0)
        out_z_gpu_h[i] = Scalar[dtype](0)
    for i in range(batch * dim * n_chunks * 2 * dstate):
        x_cpu_h[i] = Scalar[dtype](0)
        x_gpu_h[i] = Scalar[dtype](0)
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

    # Create LayoutTensors for initialization
    var u_init = LayoutTensor[dtype, layout_3d](
        u_h, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var delta_init = LayoutTensor[dtype, layout_3d](
        delta_h, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
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

    # Initialize input data
    rand[dtype](u_init.ptr, u_init.size())
    rand[dtype](delta_init.ptr, delta_init.size())
    rand[dtype](A_init.ptr, A_init.size())
    rand[dtype](B_init.ptr, B_init.size())
    rand[dtype](C_init.ptr, C_init.size())
    if has_D:
        rand[dtype](D_init.ptr, D_init.size())
    if has_z:
        rand[dtype](z_init.ptr, z_init.size())
    if has_delta_bias:
        rand[dtype](delta_bias_init.ptr, delta_bias_init.size())

    # Scale A to be negative for stability
    for i in range(dim * dstate):
        var val = A_h.load(i)
        A_h.store(i, Scalar[dtype](Float32(val) * -0.5))

    # Scale delta to be positive
    for i in range(batch * dim * seqlen):
        var val = delta_h.load(i)
        delta_h.store(i, Scalar[dtype](abs(Float32(val)) * 0.5))

    # Allocate device memory
    var output_cpu_d = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var output_gpu_d = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var x_cpu_d = ctx.enqueue_create_buffer[dtype](
        batch * dim * n_chunks * 2 * dstate
    )
    var x_gpu_d = ctx.enqueue_create_buffer[dtype](
        batch * dim * n_chunks * 2 * dstate
    )
    var out_z_cpu_d = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var out_z_gpu_d = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var u_d = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var delta_d = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var A_d = ctx.enqueue_create_buffer[dtype](dim * dstate)
    var B_d = ctx.enqueue_create_buffer[dtype](
        batch * n_groups * dstate * seqlen
    )
    var C_d = ctx.enqueue_create_buffer[dtype](
        batch * n_groups * dstate * seqlen
    )
    var D_d = ctx.enqueue_create_buffer[dtype](max(D_size, 1))
    var z_d = ctx.enqueue_create_buffer[dtype](max(z_size, 1))
    var delta_bias_d = ctx.enqueue_create_buffer[dtype](max(delta_bias_size, 1))

    # Copy to device
    ctx.enqueue_copy(u_d, u_h)
    ctx.enqueue_copy(delta_d, delta_h)
    ctx.enqueue_copy(A_d, A_h)
    ctx.enqueue_copy(B_d, B_h)
    ctx.enqueue_copy(C_d, C_h)
    if has_D:
        ctx.enqueue_copy(D_d, D_h)
    if has_z:
        ctx.enqueue_copy(z_d, z_h)
    if has_delta_bias:
        ctx.enqueue_copy(delta_bias_d, delta_bias_h)

    # Create LayoutTensors for CPU
    # Create CPU LayoutTensors with MutAnyOrigin for CPU function (using host memory)
    var _delta_bias_cpu_buf = LayoutTensor[dtype, layout_1d](
        delta_bias_h,
        RuntimeLayout[layout_1d].row_major(Index(delta_bias_size)),
    )

    # Create LayoutTensors for GPU
    var _output_gpu_buf = LayoutTensor[dtype, layout_3d](
        output_gpu_d,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )
    var _x_gpu_buf = LayoutTensor[dtype, layout_4d](
        x_gpu_d,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, dim, n_chunks, 2 * dstate)
        ),
    )
    var _out_z_gpu_buf = LayoutTensor[dtype, layout_3d](
        out_z_gpu_d,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )
    var _u_gpu_buf = LayoutTensor[dtype, layout_3d](
        u_d, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var _delta_gpu_buf = LayoutTensor[dtype, layout_3d](
        delta_d, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )
    var _A_gpu_buf = LayoutTensor[dtype, layout_2d](
        A_d, RuntimeLayout[layout_2d].row_major(Index(dim, dstate))
    )
    var _B_gpu_buf = LayoutTensor[dtype, layout_4d](
        B_d,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_groups, dstate, seqlen)
        ),
    )
    var _C_gpu_buf = LayoutTensor[dtype, layout_4d](
        C_d,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_groups, dstate, seqlen)
        ),
    )
    var _D_gpu_buf = LayoutTensor[dtype, layout_1d](
        D_d, RuntimeLayout[layout_1d].row_major(Index(D_size))
    )
    var _z_gpu_buf = LayoutTensor[dtype, layout_3d](
        z_d,
        RuntimeLayout[layout_3d].row_major(
            Index(
                batch if has_z else 0,
                dim if has_z else 0,
                seqlen if has_z else 0,
            )
        ),
    )
    var _delta_bias_gpu_buf = LayoutTensor[dtype, layout_1d](
        delta_bias_d, RuntimeLayout[layout_1d].row_major(Index(delta_bias_size))
    )

    # Strides for row-major layout
    var output_strides = Strides3D(dim * seqlen, seqlen, 1)
    var x_strides = Strides4D(
        dim * n_chunks * 2 * dstate, n_chunks * 2 * dstate, 2 * dstate, 1
    )
    var out_z_strides = Strides3D(dim * seqlen, seqlen, 1)
    var u_strides = Strides3D(dim * seqlen, seqlen, 1)
    var delta_strides = Strides3D(dim * seqlen, seqlen, 1)
    var A_strides = Strides2D(dstate, 1)
    var B_strides = Strides4D(
        n_groups * dstate * seqlen, dstate * seqlen, seqlen, 1
    )
    var C_strides = Strides4D(
        n_groups * dstate * seqlen, dstate * seqlen, seqlen, 1
    )
    var D_strides = Strides1D(1)
    var z_strides = Strides3D(dim * seqlen, seqlen, 1)
    var delta_bias_strides = Strides1D(1)

    comptime delta_softplus_int8: Int8 = Int8(1) if delta_softplus else Int8(0)

    # Create TileTensors for CPU kernel
    var output_cpu_tt = TileTensor(output_cpu_h, row_major(batch, dim, seqlen))
    var x_cpu_tt = TileTensor(
        x_cpu_h,
        row_major(batch, dim, n_chunks, 2 * dstate),
    )
    var out_z_cpu_tt = TileTensor(out_z_cpu_h, row_major(batch, dim, seqlen))
    var u_cpu_tt = TileTensor(u_h, row_major(batch, dim, seqlen))
    var delta_cpu_tt = TileTensor(delta_h, row_major(batch, dim, seqlen))
    var A_cpu_tt = TileTensor(A_h, row_major(dim, dstate))
    var B_cpu_tt = TileTensor(
        B_h,
        row_major(batch, n_groups, dstate, seqlen),
    )
    var C_cpu_tt = TileTensor(
        C_h,
        row_major(batch, n_groups, dstate, seqlen),
    )
    var D_cpu_tt = TileTensor(
        D_h,
        row_major(
            D_size,
        ),
    )
    var z_cpu_tt = TileTensor(
        z_h,
        row_major(
            (
                batch if has_z else 0,
                dim if has_z else 0,
                seqlen if has_z else 0,
            )
        ),
    )
    var delta_bias_cpu_tt = TileTensor(
        delta_bias_h,
        row_major(
            delta_bias_size,
        ),
    )

    # Run CPU kernel
    selective_scan_fwd_cpu[
        dtype,
        DSTATE,
    ](
        batch,
        dim,
        seqlen,
        group_size,
        delta_softplus_int8,
        output_cpu_tt,
        x_cpu_tt,
        out_z_cpu_tt,
        u_cpu_tt,
        delta_cpu_tt,
        A_cpu_tt,
        B_cpu_tt,
        C_cpu_tt,
        D_cpu_tt,
        z_cpu_tt,
        delta_bias_cpu_tt,
        output_strides,
        x_strides,
        out_z_strides,
        u_strides,
        delta_strides,
        A_strides,
        B_strides,
        C_strides,
        D_strides,
        z_strides,
        delta_bias_strides,
    )

    # Create TileTensors for GPU kernel
    var output_gpu_tt = TileTensor(
        output_gpu_d,
        row_major(batch, dim, seqlen),
    )
    var x_gpu_tt = TileTensor(
        x_gpu_d,
        row_major(batch, dim, n_chunks, 2 * dstate),
    )
    var out_z_gpu_tt = TileTensor(
        out_z_gpu_d,
        row_major(batch, dim, seqlen),
    )
    var u_gpu_tt = TileTensor(
        u_d,
        row_major(batch, dim, seqlen),
    )
    var delta_gpu_tt = TileTensor(
        delta_d,
        row_major(batch, dim, seqlen),
    )
    var A_gpu_tt = TileTensor(
        A_d,
        row_major(dim, dstate),
    )
    var B_gpu_tt = TileTensor(
        B_d,
        row_major(batch, n_groups, dstate, seqlen),
    )
    var C_gpu_tt = TileTensor(
        C_d,
        row_major(batch, n_groups, dstate, seqlen),
    )
    var D_gpu_tt = TileTensor(
        D_d,
        row_major(
            D_size,
        ),
    )
    var z_gpu_tt = TileTensor(
        z_d,
        row_major(
            (
                batch if has_z else 0,
                dim if has_z else 0,
                seqlen if has_z else 0,
            )
        ),
    )
    var delta_bias_gpu_tt = TileTensor(
        delta_bias_d,
        row_major(
            delta_bias_size,
        ),
    )

    # Run GPU kernel
    var total_batch_dim = batch * dim
    comptime BLOCK_SIZE = 128
    from std.math import ceildiv

    var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)

    var compiled_kernel = ctx.compile_function[
        selective_scan_fwd_gpu[
            dtype,
            DSTATE,
            output_gpu_tt.LayoutType,
            x_gpu_tt.LayoutType,
            out_z_gpu_tt.LayoutType,
            u_gpu_tt.LayoutType,
            delta_gpu_tt.LayoutType,
            A_gpu_tt.LayoutType,
            B_gpu_tt.LayoutType,
            C_gpu_tt.LayoutType,
            D_gpu_tt.LayoutType,
            z_gpu_tt.LayoutType,
            delta_bias_gpu_tt.LayoutType,
        ]
    ]()

    ctx.enqueue_function(
        compiled_kernel,
        Int32(total_batch_dim),
        Int32(batch),
        Int32(dim),
        Int32(seqlen),
        Int32(group_size),
        delta_softplus_int8,
        output_gpu_tt,
        x_gpu_tt,
        out_z_gpu_tt,
        u_gpu_tt,
        delta_gpu_tt,
        A_gpu_tt,
        B_gpu_tt,
        C_gpu_tt,
        D_gpu_tt,
        z_gpu_tt,
        delta_bias_gpu_tt,
        output_strides,
        x_strides,
        out_z_strides,
        u_strides,
        delta_strides,
        A_strides,
        B_strides,
        C_strides,
        D_strides,
        z_strides,
        delta_bias_strides,
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,),
    )

    # Copy GPU results back (CPU results are already in output_cpu_h)
    ctx.enqueue_copy(output_gpu_h, output_gpu_d)
    ctx.synchronize()

    # Compare results
    var flattened_size = batch * dim * seqlen
    for i in range(flattened_size):
        assert_almost_equal(
            output_cpu_h.load(i),
            output_gpu_h.load(i),
            rtol=rtol,
        )

    # Cleanup
    output_cpu_h.free()
    output_gpu_h.free()
    x_cpu_h.free()
    x_gpu_h.free()
    out_z_cpu_h.free()
    out_z_gpu_h.free()
    u_h.free()
    delta_h.free()
    A_h.free()
    B_h.free()
    C_h.free()
    D_h.free()
    z_h.free()
    delta_bias_h.free()


def run_selective_scan_update_gpu[
    dtype: DType,
    DSTATE: Int,
    has_D: Bool = True,
    has_z: Bool = True,
    has_delta_bias: Bool = True,
    delta_softplus: Bool = False,
](
    batch: Int,
    dim: Int,
    n_groups: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Test selective scan update GPU kernel against CPU reference."""
    comptime assert DSTATE <= 16, "DSTATE exceeds kernel limit"
    comptime dstate = DSTATE

    var group_size = dim // n_groups

    # Allocate host memory
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)

    var state_in_h = alloc[Scalar[dtype]](batch * dim * dstate)
    var state_out_gpu_h = alloc[Scalar[dtype]](batch * dim * dstate)
    var state_out_cpu_h = alloc[Scalar[dtype]](batch * dim * dstate)
    var output_gpu_h = alloc[Scalar[dtype]](batch * dim)
    var output_cpu_h = alloc[Scalar[dtype]](batch * dim)
    var x_h = alloc[Scalar[dtype]](batch * dim)
    var dt_h = alloc[Scalar[dtype]](batch * dim)
    var A_h = alloc[Scalar[dtype]](dim * dstate)
    var B_h = alloc[Scalar[dtype]](batch * n_groups * dstate)
    var C_h = alloc[Scalar[dtype]](batch * n_groups * dstate)
    var D_size = dim if has_D else 0
    var D_h = alloc[Scalar[dtype]](max(D_size, 1))
    var z_size = batch * dim if has_z else 0
    var z_h = alloc[Scalar[dtype]](max(z_size, 1))
    var dt_bias_size = dim if has_delta_bias else 0
    var dt_bias_h = alloc[Scalar[dtype]](max(dt_bias_size, 1))

    # Initialize output buffers to zero
    for i in range(batch * dim * dstate):
        state_out_gpu_h[i] = Scalar[dtype](0)
        state_out_cpu_h[i] = Scalar[dtype](0)
    for i in range(batch * dim):
        output_gpu_h[i] = Scalar[dtype](0)
        output_cpu_h[i] = Scalar[dtype](0)

    # Create LayoutTensors for initialization
    var state_in_init = LayoutTensor[dtype, layout_3d](
        state_in_h,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, dstate)),
    )
    var x_init = LayoutTensor[dtype, layout_2d](
        x_h, RuntimeLayout[layout_2d].row_major(Index(batch, dim))
    )
    var dt_init = LayoutTensor[dtype, layout_2d](
        dt_h, RuntimeLayout[layout_2d].row_major(Index(batch, dim))
    )
    var A_init = LayoutTensor[dtype, layout_2d](
        A_h, RuntimeLayout[layout_2d].row_major(Index(dim, dstate))
    )
    var B_init = LayoutTensor[dtype, layout_3d](
        B_h, RuntimeLayout[layout_3d].row_major(Index(batch, n_groups, dstate))
    )
    var C_init = LayoutTensor[dtype, layout_3d](
        C_h, RuntimeLayout[layout_3d].row_major(Index(batch, n_groups, dstate))
    )
    var D_init = LayoutTensor[dtype, layout_1d](
        D_h, RuntimeLayout[layout_1d].row_major(Index(D_size))
    )
    var z_init = LayoutTensor[dtype, layout_2d](
        z_h,
        RuntimeLayout[layout_2d].row_major(
            Index(batch if has_z else 0, dim if has_z else 0)
        ),
    )
    var dt_bias_init = LayoutTensor[dtype, layout_1d](
        dt_bias_h, RuntimeLayout[layout_1d].row_major(Index(dt_bias_size))
    )

    # Initialize input data
    rand[dtype](state_in_init.ptr, state_in_init.size())
    rand[dtype](x_init.ptr, x_init.size())
    rand[dtype](dt_init.ptr, dt_init.size())
    rand[dtype](A_init.ptr, A_init.size())
    rand[dtype](B_init.ptr, B_init.size())
    rand[dtype](C_init.ptr, C_init.size())
    if has_D:
        rand[dtype](D_init.ptr, D_init.size())
    if has_z:
        rand[dtype](z_init.ptr, z_init.size())
    if has_delta_bias:
        rand[dtype](dt_bias_init.ptr, dt_bias_init.size())

    # Scale A to be negative for stability
    for i in range(dim * dstate):
        var val = A_h.load(i)
        A_h.store(i, Scalar[dtype](Float32(val) * -0.5))

    # Copy state_in for CPU and GPU
    for i in range(batch * dim * dstate):
        state_out_cpu_h[i] = state_in_h[i]

    # Allocate device buffers
    var state_in_device = ctx.enqueue_create_buffer[dtype](batch * dim * dstate)
    var state_out_device = ctx.enqueue_create_buffer[dtype](
        batch * dim * dstate
    )
    var output_device = ctx.enqueue_create_buffer[dtype](batch * dim)
    var x_device = ctx.enqueue_create_buffer[dtype](batch * dim)
    var dt_device = ctx.enqueue_create_buffer[dtype](batch * dim)
    var A_device = ctx.enqueue_create_buffer[dtype](dim * dstate)
    var B_device = ctx.enqueue_create_buffer[dtype](batch * n_groups * dstate)
    var C_device = ctx.enqueue_create_buffer[dtype](batch * n_groups * dstate)
    var D_device = ctx.enqueue_create_buffer[dtype](max(D_size, 1))
    var z_device = ctx.enqueue_create_buffer[dtype](max(z_size, 1))
    var dt_bias_device = ctx.enqueue_create_buffer[dtype](max(dt_bias_size, 1))

    # Copy data to device
    with ctx.push_context():
        ctx.enqueue_copy(state_in_device, state_in_h)
        ctx.enqueue_copy(x_device, x_h)
        ctx.enqueue_copy(dt_device, dt_h)
        ctx.enqueue_copy(A_device, A_h)
        ctx.enqueue_copy(B_device, B_h)
        ctx.enqueue_copy(C_device, C_h)
        if has_D:
            ctx.enqueue_copy(D_device, D_h)
        if has_z:
            ctx.enqueue_copy(z_device, z_h)
        if has_delta_bias:
            ctx.enqueue_copy(dt_bias_device, dt_bias_h)

    # Create device tensors
    var _dt_bias_device_tensor = LayoutTensor[dtype, layout_1d](
        dt_bias_device,
        RuntimeLayout[layout_1d].row_major(Index(dt_bias_size)),
    )

    # Strides for row-major layout
    var state_out_strides = Strides3D(dim * dstate, dstate, 1)
    var output_strides = Strides2D(dim, 1)
    var state_in_strides = Strides3D(dim * dstate, dstate, 1)
    var x_strides = Strides2D(dim, 1)
    var dt_strides = Strides2D(dim, 1)
    var A_strides = Strides2D(dstate, 1)
    var B_strides = Strides3D(n_groups * dstate, dstate, 1)
    var C_strides = Strides3D(n_groups * dstate, dstate, 1)
    var D_strides = Strides1D(1)
    var z_strides = Strides2D(dim, 1)
    var dt_bias_strides = Strides1D(1)

    # Create TileTensors for GPU kernel
    var state_in_device_tt = TileTensor(
        state_in_device,
        row_major(batch, dim, dstate),
    )
    var state_out_device_tt = TileTensor(
        state_out_device,
        row_major(batch, dim, dstate),
    )
    var output_device_tt = TileTensor(
        output_device,
        row_major(batch, dim),
    )
    var x_device_tt = TileTensor(
        x_device,
        row_major(batch, dim),
    )
    var dt_device_tt = TileTensor(
        dt_device,
        row_major(batch, dim),
    )
    var A_device_tt = TileTensor(
        A_device,
        row_major(dim, dstate),
    )
    var B_device_tt = TileTensor(
        B_device,
        row_major(batch, n_groups, dstate),
    )
    var C_device_tt = TileTensor(
        C_device,
        row_major(batch, n_groups, dstate),
    )
    var D_device_tt = TileTensor(
        D_device,
        row_major(
            D_size,
        ),
    )
    var z_device_tt = TileTensor(
        z_device,
        row_major(
            (
                batch if has_z else 0,
                dim if has_z else 0,
            )
        ),
    )
    var dt_bias_device_tt = TileTensor(
        dt_bias_device,
        row_major(
            dt_bias_size,
        ),
    )

    # Run GPU kernel
    var total_batch_dim = batch * dim
    with ctx.push_context():
        var compiled_func = ctx.compile_function[
            selective_scan_update_gpu[
                dtype,
                DSTATE,
                state_out_device_tt.LayoutType,
                output_device_tt.LayoutType,
                state_in_device_tt.LayoutType,
                x_device_tt.LayoutType,
                dt_device_tt.LayoutType,
                A_device_tt.LayoutType,
                B_device_tt.LayoutType,
                C_device_tt.LayoutType,
                D_device_tt.LayoutType,
                z_device_tt.LayoutType,
                dt_bias_device_tt.LayoutType,
            ]
        ]()
        ctx.enqueue_function(
            compiled_func,
            Int32(total_batch_dim),
            Int32(batch),
            Int32(dim),
            Int32(group_size),
            Int8(1) if delta_softplus else Int8(0),
            state_out_device_tt,
            output_device_tt,
            state_in_device_tt,
            x_device_tt,
            dt_device_tt,
            A_device_tt,
            B_device_tt,
            C_device_tt,
            D_device_tt,
            z_device_tt,
            dt_bias_device_tt,
            state_out_strides,
            output_strides,
            state_in_strides,
            x_strides,
            dt_strides,
            A_strides,
            B_strides,
            C_strides,
            D_strides,
            z_strides,
            dt_bias_strides,
            grid_dim=(ceildiv(total_batch_dim, 256),),
            block_dim=(256,),
        )

    # Copy results back from device
    with ctx.push_context():
        ctx.enqueue_copy(state_out_gpu_h, state_out_device)
        ctx.enqueue_copy(output_gpu_h, output_device)
        ctx.synchronize()

    # Create TileTensors for CPU reference
    var state_out_cpu_tt = TileTensor(
        state_out_cpu_h, row_major(batch, dim, dstate)
    )
    var output_cpu_tt = TileTensor(output_cpu_h, row_major(batch, dim))
    var state_in_cpu_tt = TileTensor(state_in_h, row_major(batch, dim, dstate))
    var x_cpu_tt = TileTensor(x_h, row_major(batch, dim))
    var dt_cpu_tt = TileTensor(dt_h, row_major(batch, dim))
    var A_cpu_tt = TileTensor(A_h, row_major(dim, dstate))
    var B_cpu_tt = TileTensor(B_h, row_major(batch, n_groups, dstate))
    var C_cpu_tt = TileTensor(C_h, row_major(batch, n_groups, dstate))
    var D_cpu_tt = TileTensor(
        D_h,
        row_major(
            D_size,
        ),
    )
    var z_cpu_tt = TileTensor(
        z_h,
        row_major(
            (
                batch if has_z else 0,
                dim if has_z else 0,
            )
        ),
    )
    var dt_bias_cpu_tt = TileTensor(
        dt_bias_h,
        row_major(
            dt_bias_size,
        ),
    )

    # Run CPU reference
    selective_scan_update_cpu[
        dtype,
        DSTATE,
    ](
        batch,
        dim,
        group_size,
        Int8(1) if delta_softplus else Int8(0),
        state_out_cpu_tt,
        output_cpu_tt,
        state_in_cpu_tt,
        x_cpu_tt,
        dt_cpu_tt,
        A_cpu_tt,
        B_cpu_tt,
        C_cpu_tt,
        D_cpu_tt,
        z_cpu_tt,
        dt_bias_cpu_tt,
        state_out_strides,
        output_strides,
        state_in_strides,
        x_strides,
        dt_strides,
        A_strides,
        B_strides,
        C_strides,
        D_strides,
        z_strides,
        dt_bias_strides,
    )

    # Compare results
    var state_size = batch * dim * dstate
    for i in range(state_size):
        assert_almost_equal(
            state_out_gpu_h[i],
            state_out_cpu_h[i],
            rtol=rtol,
        )

    var output_size = batch * dim
    for i in range(output_size):
        assert_almost_equal(
            output_gpu_h[i],
            output_cpu_h[i],
            rtol=rtol,
        )

    # Cleanup
    state_in_h.free()
    state_out_gpu_h.free()
    state_out_cpu_h.free()
    output_gpu_h.free()
    output_cpu_h.free()
    x_h.free()
    dt_h.free()
    A_h.free()
    B_h.free()
    C_h.free()
    D_h.free()
    z_h.free()
    dt_bias_h.free()


# =============================================================================
# Test functions for selective scan forward (GPU)
# =============================================================================


def test_selective_scan_gpu_basic() raises:
    """Test basic selective scan GPU kernel."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_selective_scan_gpu[
        DType.float32,
        2,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=1, dim=2, seqlen=4, n_groups=1, ctx=ctx)


def test_selective_scan_gpu_without_D() raises:
    """Test selective scan GPU without D tensor."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_selective_scan_gpu[
        DType.float32,
        2,  # DSTATE
        has_D=False,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=1, dim=2, seqlen=4, n_groups=1, ctx=ctx)


def test_selective_scan_gpu_without_z() raises:
    """Test selective scan GPU without z tensor."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_selective_scan_gpu[
        DType.float32,
        2,  # DSTATE
        has_D=True,
        has_z=False,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=1, dim=2, seqlen=4, n_groups=1, ctx=ctx)


def test_selective_scan_gpu_with_delta_softplus() raises:
    """Test selective scan GPU with delta softplus activation."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_selective_scan_gpu[
        DType.float32,
        2,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=True,
    ](batch=1, dim=2, seqlen=4, n_groups=1, ctx=ctx)


def test_selective_scan_gpu_longer_sequence() raises:
    """Test selective scan GPU with longer sequence."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_selective_scan_gpu[
        DType.float32,
        4,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=1, dim=4, seqlen=16, n_groups=1, ctx=ctx)


def test_selective_scan_gpu_edge_case_seqlen() raises:
    """Test selective scan GPU with edge case sequence lengths."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    # CPU uses TILE_SIZE=4, so test edge cases around multiples of 4
    for seqlen in [5, 7]:
        run_selective_scan_gpu[
            DType.float32,
            2,  # DSTATE
            has_D=True,
            has_z=True,
            has_delta_bias=True,
            delta_softplus=False,
        ](batch=1, dim=2, seqlen=seqlen, n_groups=1, ctx=ctx)


def test_selective_scan_gpu_realistic_dimensions() raises:
    """Test selective scan GPU with realistic dimensions."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_selective_scan_gpu[
        DType.float32,
        8,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=True,
    ](batch=1, dim=64, seqlen=7, n_groups=1, ctx=ctx)


# =============================================================================
# Test functions for selective scan update (GPU)
# =============================================================================


def test_selective_scan_update_gpu_basic() raises:
    """Test basic selective scan update GPU kernel."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_selective_scan_update_gpu[
        DType.float32,
        2,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=1, dim=2, n_groups=1, ctx=ctx)


def test_selective_scan_update_gpu_without_D() raises:
    """Test selective scan update GPU without D tensor."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_selective_scan_update_gpu[
        DType.float32,
        2,  # DSTATE
        has_D=False,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=1, dim=2, n_groups=1, ctx=ctx)


def test_selective_scan_update_gpu_without_z() raises:
    """Test selective scan update GPU without z tensor."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_selective_scan_update_gpu[
        DType.float32,
        2,  # DSTATE
        has_D=True,
        has_z=False,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=1, dim=2, n_groups=1, ctx=ctx)


def test_selective_scan_update_gpu_with_delta_softplus() raises:
    """Test selective scan update GPU with delta softplus activation."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_selective_scan_update_gpu[
        DType.float32,
        2,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=True,
    ](batch=1, dim=2, n_groups=1, ctx=ctx)


def test_selective_scan_update_gpu_larger_dimensions() raises:
    """Test selective scan update GPU with larger dimensions."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_selective_scan_update_gpu[
        DType.float32,
        4,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=2, dim=4, n_groups=1, ctx=ctx)
