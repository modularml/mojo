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
"""Selective scan operation registrations for Mamba SSM.

This module registers the following ops:
- selective_scan_fwd: Full selective scan forward pass
- selective_scan_fwd_minimal: Minimal variant without optional tensors
- selective_scan_update: Single-step update for autoregressive inference
"""

from std.math import ceildiv

import extensibility
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu, is_gpu

from extensibility import InputTensor, OutputTensor
from std.utils.index import IndexList

from state_space.selective_scan import (
    selective_scan_fwd_cpu,
    selective_scan_fwd_gpu,
    selective_scan_fwd_cpu_minimal,
    selective_scan_fwd_gpu_minimal,
    selective_scan_update_cpu,
    selective_scan_update_gpu,
)


@extensibility.register("selective_scan_fwd")
struct SelectiveScanFwd[delta_softplus: Bool = False]:
    """Selective scan forward pass operation for Mamba SSM.

    Performs the selective scan computation used in Mamba state space models.
    This is the core operation that processes sequences through the SSM.

    Parameters:
        delta_softplus: If True, applies softplus activation to delta values.

    Tensor Shapes:
        - output: (batch, dim, seqlen) - Output tensor
        - x: (batch, dim, num_chunks, 2*dstate) - Checkpoint tensor for chunking
        - out_z: (batch, dim, seqlen) - Gated output (if z is provided)
        - u: (batch, dim, seqlen) - Input tensor
        - delta: (batch, dim, seqlen) - Time step tensor
        - A: (dim, dstate) - State transition matrix
        - B: (batch, n_groups, dstate, seqlen) - Input projection
        - C: (batch, n_groups, dstate, seqlen) - Output projection
        - D: (dim,) - Skip connection (optional, can be empty)
        - z: (batch, dim, seqlen) - Gating tensor (optional, can be empty)
        - delta_bias: (dim,) - Delta bias (optional, can be empty)
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        x: OutputTensor[dtype=dtype, rank=4, ...],
        out_z: OutputTensor[dtype=dtype, rank=3, ...],
        u: InputTensor[dtype=dtype, rank=3, ...],
        delta: InputTensor[dtype=dtype, rank=3, ...],
        A: InputTensor[dtype=dtype, rank=2, ...],
        B: InputTensor[dtype=dtype, rank=4, ...],
        C: InputTensor[dtype=dtype, rank=4, ...],
        D: InputTensor[dtype=dtype, rank=1, ...],
        z: InputTensor[dtype=dtype, rank=3, ...],
        delta_bias: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        if output.shape() != u.shape():
            raise Error("Output shape must match input u shape")

        var batch = output.dim_size(0)
        var dim = output.dim_size(1)
        var seqlen = output.dim_size(2)
        var dstate = A.dim_size(1)
        var n_groups = B.dim_size(1)
        var group_size = dim // n_groups

        var output_tt = output.to_tile_tensor[.int32]()
        var x_tt = x.to_tile_tensor[.int32]()
        var out_z_tt = out_z.to_tile_tensor[.int32]()
        var u_tt = u.to_tile_tensor[.int32]()
        var delta_tt = delta.to_tile_tensor[.int32]()
        var A_tt = A.to_tile_tensor[.int32]()
        var B_tt = B.to_tile_tensor[.int32]()
        var C_tt = C.to_tile_tensor[.int32]()
        var D_tt = D.to_tile_tensor[.int32]()
        var z_tt = z.to_tile_tensor[.int32]()
        var delta_bias_tt = delta_bias.to_tile_tensor[.int32]()

        var output_strides = output.strides()
        var x_strides = x.strides()
        var out_z_strides = out_z.strides()
        var u_strides = u.strides()
        var delta_strides = delta.strides()
        var A_strides = A.strides()
        var B_strides = B.strides()
        var C_strides = C.strides()
        var D_strides = D.strides()
        var z_strides = z.strides()
        var delta_bias_strides = delta_bias.strides()

        comptime delta_softplus_int8: Int8 = Int8(
            1
        ) if Self.delta_softplus else Int8(0)

        if dstate != 16 and dstate != 8:
            raise Error(
                "Unsupported dstate: " + String(dstate) + ". Expected 8 or 16."
            )

        # Dispatch runtime dstate to compile-time DSTATE for comptime for
        # loop unrolling and guaranteed register allocation on GPU.
        comptime if is_cpu[target]():
            if dstate == 16:
                selective_scan_fwd_cpu[
                    dtype,
                    16,
                ](
                    batch,
                    dim,
                    seqlen,
                    group_size,
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    out_z_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    delta_bias_tt,
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
                    Optional[DeviceContext](ctx),
                )
            else:
                selective_scan_fwd_cpu[
                    dtype,
                    8,
                ](
                    batch,
                    dim,
                    seqlen,
                    group_size,
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    out_z_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    delta_bias_tt,
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
                    Optional[DeviceContext](ctx),
                )
        elif is_gpu[target]():
            var gpu_ctx = ctx
            var total_batch_dim = batch * dim
            comptime BLOCK_SIZE = 128
            var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)

            if dstate == 16:
                comptime DSTATE_VAL = 16
                var compiled_kernel = gpu_ctx.compile_function[
                    selective_scan_fwd_gpu[
                        dtype,
                        DSTATE_VAL,
                        output_tt.LayoutType,
                        x_tt.LayoutType,
                        out_z_tt.LayoutType,
                        u_tt.LayoutType,
                        delta_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        delta_bias_tt.LayoutType,
                        output_tt.Engine,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    Int32(total_batch_dim),
                    Int32(batch),
                    Int32(dim),
                    Int32(seqlen),
                    Int32(group_size),
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    out_z_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    delta_bias_tt,
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
            else:
                comptime DSTATE_VAL = 8
                var compiled_kernel = gpu_ctx.compile_function[
                    selective_scan_fwd_gpu[
                        dtype,
                        DSTATE_VAL,
                        output_tt.LayoutType,
                        x_tt.LayoutType,
                        out_z_tt.LayoutType,
                        u_tt.LayoutType,
                        delta_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        delta_bias_tt.LayoutType,
                        output_tt.Engine,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    Int32(total_batch_dim),
                    Int32(batch),
                    Int32(dim),
                    Int32(seqlen),
                    Int32(group_size),
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    out_z_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    delta_bias_tt,
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
        else:
            raise Error("Unsupported target: " + target)


@extensibility.register_shape_function("selective_scan_fwd")
def selective_scan_fwd_shape[
    dtype: DType,
](
    u: InputTensor[dtype=dtype, rank=3, ...],
    delta: InputTensor[dtype=dtype, rank=3, ...],
    A: InputTensor[dtype=dtype, rank=2, ...],
    B: InputTensor[dtype=dtype, rank=4, ...],
    C: InputTensor[dtype=dtype, rank=4, ...],
    D: InputTensor[dtype=dtype, rank=1, ...],
    z: InputTensor[dtype=dtype, rank=3, ...],
    delta_bias: InputTensor[dtype=dtype, rank=1, ...],
) -> IndexList[3]:
    """Returns the output shape for the `selective_scan_fwd` op.

    The output of a selective scan forward pass has the same shape as the
    input sequence `u`: `(batch, dim, seqlen)`.

    Parameters:
        dtype: Element type of the input tensors (inferred).

    Args:
        u: Input sequence tensor with shape `(batch, dim, seqlen)`.
        delta: Time-step tensor with shape `(batch, dim, seqlen)`.
        A: State transition matrix with shape `(dim, dstate)`.
        B: Input projection with shape `(batch, n_groups, dstate, seqlen)`.
        C: Output projection with shape `(batch, n_groups, dstate, seqlen)`.
        D: Skip connection with shape `(dim,)`.
        z: Gating tensor with shape `(batch, dim, seqlen)`.
        delta_bias: Delta bias with shape `(dim,)`.

    Returns:
        The shape of the output tensor `(batch, dim, seqlen)`.
    """
    return u.shape()


@extensibility.register("selective_scan_fwd_minimal")
struct SelectiveScanFwdMinimal[delta_softplus: Bool = False]:
    """Minimal selective scan forward pass - no optional D, z, or delta_bias.

    This variant avoids passing empty tensors that could have null pointers.
    Use when D, z, and delta_bias are not provided.

    Parameters:
        delta_softplus: If True, applies softplus activation to delta values.

    Tensor Shapes:
        - output: (batch, dim, seqlen) - Output tensor
        - x: (batch, dim, num_chunks, 2*dstate) - Checkpoint tensor for chunking
        - u: (batch, dim, seqlen) - Input tensor
        - delta: (batch, dim, seqlen) - Time step tensor
        - A: (dim, dstate) - State transition matrix
        - B: (batch, n_groups, dstate, seqlen) - Input projection
        - C: (batch, n_groups, dstate, seqlen) - Output projection
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        x: OutputTensor[dtype=dtype, rank=4, ...],
        u: InputTensor[dtype=dtype, rank=3, ...],
        delta: InputTensor[dtype=dtype, rank=3, ...],
        A: InputTensor[dtype=dtype, rank=2, ...],
        B: InputTensor[dtype=dtype, rank=4, ...],
        C: InputTensor[dtype=dtype, rank=4, ...],
        ctx: DeviceContext,
    ) capturing raises:
        if output.shape() != u.shape():
            raise Error("Output shape must match input u shape")

        var batch = output.dim_size(0)
        var dim = output.dim_size(1)
        var seqlen = output.dim_size(2)
        var dstate = A.dim_size(1)
        var n_groups = B.dim_size(1)
        var group_size = dim // n_groups

        var output_tt = output.to_tile_tensor[.int32]()
        var x_tt = x.to_tile_tensor[.int32]()
        var u_tt = u.to_tile_tensor[.int32]()
        var delta_tt = delta.to_tile_tensor[.int32]()
        var A_tt = A.to_tile_tensor[.int32]()
        var B_tt = B.to_tile_tensor[.int32]()
        var C_tt = C.to_tile_tensor[.int32]()

        var output_strides = output.strides()
        var x_strides = x.strides()
        var u_strides = u.strides()
        var delta_strides = delta.strides()
        var A_strides = A.strides()
        var B_strides = B.strides()
        var C_strides = C.strides()

        comptime delta_softplus_int8: Int8 = Int8(
            1
        ) if Self.delta_softplus else Int8(0)

        if dstate != 16 and dstate != 8:
            raise Error(
                "Unsupported dstate: " + String(dstate) + ". Expected 8 or 16."
            )

        comptime if is_cpu[target]():
            if dstate == 16:
                selective_scan_fwd_cpu_minimal[
                    dtype,
                    16,
                ](
                    batch,
                    dim,
                    seqlen,
                    group_size,
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    output_strides,
                    x_strides,
                    u_strides,
                    delta_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    Optional[DeviceContext](ctx),
                )
            else:
                selective_scan_fwd_cpu_minimal[
                    dtype,
                    8,
                ](
                    batch,
                    dim,
                    seqlen,
                    group_size,
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    output_strides,
                    x_strides,
                    u_strides,
                    delta_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    Optional[DeviceContext](ctx),
                )
        elif is_gpu[target]():
            var gpu_ctx = ctx
            var total_batch_dim = batch * dim
            comptime BLOCK_SIZE = 128
            var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)

            if dstate == 16:
                comptime DSTATE_VAL = 16
                var compiled_kernel = gpu_ctx.compile_function[
                    selective_scan_fwd_gpu_minimal[
                        dtype,
                        DSTATE_VAL,
                        output_tt.LayoutType,
                        x_tt.LayoutType,
                        u_tt.LayoutType,
                        delta_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        output_tt.Engine,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    Int32(total_batch_dim),
                    Int32(batch),
                    Int32(dim),
                    Int32(seqlen),
                    Int32(group_size),
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    output_strides,
                    x_strides,
                    u_strides,
                    delta_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    grid_dim=(num_blocks),
                    block_dim=(BLOCK_SIZE),
                )
            else:
                comptime DSTATE_VAL = 8
                var compiled_kernel = gpu_ctx.compile_function[
                    selective_scan_fwd_gpu_minimal[
                        dtype,
                        DSTATE_VAL,
                        output_tt.LayoutType,
                        x_tt.LayoutType,
                        u_tt.LayoutType,
                        delta_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        output_tt.Engine,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    Int32(total_batch_dim),
                    Int32(batch),
                    Int32(dim),
                    Int32(seqlen),
                    Int32(group_size),
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    output_strides,
                    x_strides,
                    u_strides,
                    delta_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    grid_dim=(num_blocks),
                    block_dim=(BLOCK_SIZE),
                )
        else:
            raise Error("Unsupported target device")


@extensibility.register_shape_function("selective_scan_fwd_minimal")
def selective_scan_fwd_minimal_shape[
    dtype: DType,
](
    u: InputTensor[dtype=dtype, rank=3, ...],
    delta: InputTensor[dtype=dtype, rank=3, ...],
    A: InputTensor[dtype=dtype, rank=2, ...],
    B: InputTensor[dtype=dtype, rank=4, ...],
    C: InputTensor[dtype=dtype, rank=4, ...],
) -> IndexList[3]:
    """Returns the output shape for the `selective_scan_fwd_minimal` op.

    The output of the minimal selective scan forward pass has the same shape
    as the input sequence `u`: `(batch, dim, seqlen)`.

    Args:
        u: Input sequence tensor with shape `(batch, dim, seqlen)`.
        delta: Time-step tensor with shape `(batch, dim, seqlen)`.
        A: State transition matrix with shape `(dim, dstate)`.
        B: Input projection with shape `(batch, n_groups, dstate, seqlen)`.
        C: Output projection with shape `(batch, n_groups, dstate, seqlen)`.

    Returns:
        The shape of the output tensor `(batch, dim, seqlen)`.
    """
    return u.shape()


@extensibility.register("selective_scan_update")
struct SelectiveScanUpdate[delta_softplus: Bool = False]:
    """Selective scan update operation for autoregressive inference.

    Performs a single step of the SSM recurrence for incremental token generation.

    Parameters:
        delta_softplus: If True, applies softplus activation to delta values.

    Tensor Shapes:
        - state_out: (batch, dim, dstate) - Updated state output
        - output: (batch, dim) - Output tensor
        - state_in: (batch, dim, dstate) - Input state
        - x: (batch, dim) - Input tensor
        - dt: (batch, dim) - Time delta tensor
        - A: (dim, dstate) - State transition matrix
        - B: (batch, n_groups, dstate) - Input matrix
        - C: (batch, n_groups, dstate) - Output matrix
        - D: (dim,) - Skip connection (optional, can be empty)
        - z: (batch, dim) - Gating tensor (optional, can be empty)
        - dt_bias: (dim,) - Time delta bias (optional, can be empty)
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        state_out: OutputTensor[dtype=dtype, rank=3, ...],
        output: OutputTensor[dtype=dtype, rank=2, ...],
        state_in: InputTensor[dtype=dtype, rank=3, ...],
        x: InputTensor[dtype=dtype, rank=2, ...],
        dt: InputTensor[dtype=dtype, rank=2, ...],
        A: InputTensor[dtype=dtype, rank=2, ...],
        B: InputTensor[dtype=dtype, rank=3, ...],
        C: InputTensor[dtype=dtype, rank=3, ...],
        D: InputTensor[dtype=dtype, rank=1, ...],
        z: InputTensor[dtype=dtype, rank=2, ...],
        dt_bias: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        var batch = state_out.dim_size(0)
        var dim = state_out.dim_size(1)
        var dstate = state_out.dim_size(2)
        var n_groups = B.dim_size(1)
        var group_size = dim // n_groups

        var state_out_tt = state_out.to_tile_tensor[.int32]()
        var output_tt = output.to_tile_tensor[.int32]()
        var state_in_tt = state_in.to_tile_tensor[.int32]()
        var x_tt = x.to_tile_tensor[.int32]()
        var dt_tt = dt.to_tile_tensor[.int32]()
        var A_tt = A.to_tile_tensor[.int32]()
        var B_tt = B.to_tile_tensor[.int32]()
        var C_tt = C.to_tile_tensor[.int32]()
        var D_tt = D.to_tile_tensor[.int32]()
        var z_tt = z.to_tile_tensor[.int32]()
        var dt_bias_tt = dt_bias.to_tile_tensor[.int32]()

        var state_out_strides = state_out.strides()
        var output_strides = output.strides()
        var state_in_strides = state_in.strides()
        var x_strides = x.strides()
        var dt_strides = dt.strides()
        var A_strides = A.strides()
        var B_strides = B.strides()
        var C_strides = C.strides()
        var D_strides = D.strides()
        var z_strides = z.strides()
        var dt_bias_strides = dt_bias.strides()

        comptime delta_softplus_int8: Int8 = Int8(
            1
        ) if Self.delta_softplus else Int8(0)

        if dstate != 16 and dstate != 8:
            raise Error(
                "Unsupported dstate: " + String(dstate) + ". Expected 8 or 16."
            )

        comptime if is_cpu[target]():
            if dstate == 16:
                selective_scan_update_cpu[
                    dtype,
                    16,
                ](
                    batch,
                    dim,
                    group_size,
                    delta_softplus_int8,
                    state_out_tt,
                    output_tt,
                    state_in_tt,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    dt_bias_tt,
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
                    Optional[DeviceContext](ctx),
                )
            else:
                selective_scan_update_cpu[
                    dtype,
                    8,
                ](
                    batch,
                    dim,
                    group_size,
                    delta_softplus_int8,
                    state_out_tt,
                    output_tt,
                    state_in_tt,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    dt_bias_tt,
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
                    Optional[DeviceContext](ctx),
                )
        elif is_gpu[target]():
            var gpu_ctx = ctx
            var total_batch_dim = batch * dim
            comptime BLOCK_SIZE = 128
            var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)

            if dstate == 16:
                comptime DSTATE_VAL = 16
                var compiled_kernel = gpu_ctx.compile_function[
                    selective_scan_update_gpu[
                        dtype,
                        DSTATE_VAL,
                        state_out_tt.LayoutType,
                        output_tt.LayoutType,
                        state_in_tt.LayoutType,
                        x_tt.LayoutType,
                        dt_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        dt_bias_tt.LayoutType,
                        state_out_tt.Engine,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    Int32(total_batch_dim),
                    Int32(batch),
                    Int32(dim),
                    Int32(group_size),
                    delta_softplus_int8,
                    state_out_tt,
                    output_tt,
                    state_in_tt,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    dt_bias_tt,
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
                    grid_dim=(num_blocks,),
                    block_dim=(BLOCK_SIZE,),
                )
            else:
                comptime DSTATE_VAL = 8
                var compiled_kernel = gpu_ctx.compile_function[
                    selective_scan_update_gpu[
                        dtype,
                        DSTATE_VAL,
                        state_out_tt.LayoutType,
                        output_tt.LayoutType,
                        state_in_tt.LayoutType,
                        x_tt.LayoutType,
                        dt_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        dt_bias_tt.LayoutType,
                        state_out_tt.Engine,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    Int32(total_batch_dim),
                    Int32(batch),
                    Int32(dim),
                    Int32(group_size),
                    delta_softplus_int8,
                    state_out_tt,
                    output_tt,
                    state_in_tt,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    dt_bias_tt,
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
                    grid_dim=(num_blocks,),
                    block_dim=(BLOCK_SIZE,),
                )
        else:
            raise Error("Unsupported target: " + target)


@extensibility.register_shape_function("selective_scan_update")
def selective_scan_update_shape[
    dtype: DType,
](
    state_in: InputTensor[dtype=dtype, rank=3, ...],
    x: InputTensor[dtype=dtype, rank=2, ...],
    dt: InputTensor[dtype=dtype, rank=2, ...],
    A: InputTensor[dtype=dtype, rank=2, ...],
    B: InputTensor[dtype=dtype, rank=3, ...],
    C: InputTensor[dtype=dtype, rank=3, ...],
    D: InputTensor[dtype=dtype, rank=1, ...],
    z: InputTensor[dtype=dtype, rank=2, ...],
    dt_bias: InputTensor[dtype=dtype, rank=1, ...],
) -> Tuple[IndexList[3], IndexList[2]]:
    """Returns the output shapes for the `selective_scan_update` op.

    The update step produces two tensors: the updated SSM state and the
    single-step output for the new token.

    Args:
        state_in: Input SSM state with shape `(batch, dim, dstate)`.
        x: Input token with shape `(batch, dim)`.
        dt: Time-delta tensor with shape `(batch, dim)`.
        A: State transition matrix with shape `(dim, dstate)`.
        B: Input matrix with shape `(batch, n_groups, dstate)`.
        C: Output matrix with shape `(batch, n_groups, dstate)`.
        D: Skip connection with shape `(dim,)`.
        z: Gating tensor with shape `(batch, dim)`.
        dt_bias: Time-delta bias with shape `(dim,)`.

    Returns:
        A tuple of `(state_out_shape, output_shape)` where `state_out_shape`
        matches `state_in.shape()` and `output_shape` matches `x.shape()`.
    """
    return (state_in.shape(), x.shape())
