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

from std.math import ceildiv, exp, exp2, log, rsqrt

from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout, TileTensor, row_major
from layout._fillers import random
from state_space.selective_scan import (
    mamba_split_conv1d_scan_combined_cpu,
    mamba_split_conv1d_scan_combined_gpu,
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


def run_mamba_split_conv1d_scan_combined_gpu[
    dtype: DType,
    DSTATE: Int,
    has_D: Bool,
    has_rmsnorm: Bool,
    has_outproj: Bool,
    norm_before_gate: Bool,
    delta_softplus: Bool,
](
    batch: Int,
    seqlen: Int,
    dim: Int,
    nheads: Int,
    headdim: Int,
    ngroups: Int,
    width: Int,
    chunk_size: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    comptime dstate = DSTATE
    var n_chunks = ceildiv(seqlen, chunk_size)

    # Allocate host memory
    var zxbcdt_channels = 2 * dim + 2 * ngroups * dstate + nheads
    var zxbcdt_size = batch * seqlen * zxbcdt_channels
    var zxbcdt_h = ctx.enqueue_create_host_buffer[dtype](zxbcdt_size)
    var conv_weight_channels = dim + 2 * ngroups * dstate
    var conv_weight_size = conv_weight_channels * width
    var conv_weight_h = ctx.enqueue_create_host_buffer[dtype](conv_weight_size)
    var conv_bias_size = conv_weight_channels
    var conv_bias_h = ctx.enqueue_create_host_buffer[dtype](conv_bias_size)
    var dt_bias_size = nheads
    var dt_bias_h = ctx.enqueue_create_host_buffer[dtype](dt_bias_size)
    var A_size = nheads
    var A_h = ctx.enqueue_create_host_buffer[dtype](A_size)
    var D_size = nheads * headdim if has_D else 0
    var D_h = ctx.enqueue_create_host_buffer[dtype](max(D_size, 1))
    var x_size = batch * dim * n_chunks * 2 * dstate
    var x_h = ctx.enqueue_create_host_buffer[dtype](x_size)
    var out_z_size = batch * dim * seqlen
    var out_z_h = ctx.enqueue_create_host_buffer[dtype](out_z_size)
    var dt_size = batch * nheads * seqlen
    var dt_h = ctx.enqueue_create_host_buffer[dtype](dt_size)
    var B_size = batch * ngroups * dstate * seqlen
    var B_h = ctx.enqueue_create_host_buffer[dtype](B_size)
    var C_size = batch * ngroups * dstate * seqlen
    var C_h = ctx.enqueue_create_host_buffer[dtype](C_size)
    var z_size = batch * dim * seqlen
    var z_h = ctx.enqueue_create_host_buffer[dtype](z_size)
    var rmsnorm_weight_size = dim if has_rmsnorm else 0
    var rmsnorm_weight_h = ctx.enqueue_create_host_buffer[dtype](
        max(rmsnorm_weight_size, 1)
    )
    var out_dim = dim
    var outproj_weight_size = out_dim * dim if has_outproj else 0
    var outproj_weight_h = ctx.enqueue_create_host_buffer[dtype](
        max(outproj_weight_size, 1)
    )
    var outproj_bias_size = out_dim if has_outproj else 0
    var outproj_bias_h = ctx.enqueue_create_host_buffer[dtype](
        max(outproj_bias_size, 1)
    )
    var output_size = batch * seqlen * (out_dim if has_outproj else dim)
    var output_cpu_h = ctx.enqueue_create_host_buffer[dtype](output_size)
    var output_gpu_h = ctx.enqueue_create_host_buffer[dtype](output_size)

    # Create LayoutTensors for initialization
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout.row_major[1]()

    var zxbcdt_init = LayoutTensor[dtype, layout_3d](
        zxbcdt_h,
        RuntimeLayout[layout_3d].row_major(
            Index(batch, seqlen, zxbcdt_channels)
        ),
    )
    var conv_weight_init = LayoutTensor[dtype, layout_2d](
        conv_weight_h,
        RuntimeLayout[layout_2d].row_major(Index(conv_weight_channels, width)),
    )
    var conv_bias_init = LayoutTensor[dtype, layout_1d](
        conv_bias_h, RuntimeLayout[layout_1d].row_major(Index(conv_bias_size))
    )
    var dt_bias_init = LayoutTensor[dtype, layout_1d](
        dt_bias_h, RuntimeLayout[layout_1d].row_major(Index(dt_bias_size))
    )
    var A_init = LayoutTensor[dtype, layout_1d](
        A_h, RuntimeLayout[layout_1d].row_major(Index(A_size))
    )
    var D_init = LayoutTensor[dtype, layout_2d](
        D_h,
        RuntimeLayout[layout_2d].row_major(
            Index(
                nheads if has_D else 0,
                headdim if has_D and D_size > nheads else 0,
            )
        ),
    )
    var rmsnorm_weight_init = LayoutTensor[dtype, layout_1d](
        rmsnorm_weight_h,
        RuntimeLayout[layout_1d].row_major(Index(rmsnorm_weight_size)),
    )
    var outproj_weight_init = LayoutTensor[dtype, layout_2d](
        outproj_weight_h,
        RuntimeLayout[layout_2d].row_major(
            Index(out_dim if has_outproj else 0, dim if has_outproj else 0)
        ),
    )
    var outproj_bias_init = LayoutTensor[dtype, layout_1d](
        outproj_bias_h,
        RuntimeLayout[layout_1d].row_major(Index(outproj_bias_size)),
    )

    # Initialize with random data
    random(zxbcdt_init)
    random(conv_weight_init)
    random(conv_bias_init)
    random(dt_bias_init)
    random(A_init)
    if has_D:
        random(D_init)
    if has_rmsnorm:
        random(rmsnorm_weight_init)
        for i in range(dim):
            rmsnorm_weight_h[i] = abs(rmsnorm_weight_h[i]) + Scalar[dtype](0.1)
    if has_outproj:
        random(outproj_weight_init)
        random(outproj_bias_init)

    # Allocate GPU memory
    var zxbcdt_d = ctx.enqueue_create_buffer[dtype](zxbcdt_size)
    var conv_weight_d = ctx.enqueue_create_buffer[dtype](conv_weight_size)
    var conv_bias_d = ctx.enqueue_create_buffer[dtype](conv_bias_size)
    var dt_bias_d = ctx.enqueue_create_buffer[dtype](dt_bias_size)
    var A_d = ctx.enqueue_create_buffer[dtype](A_size)
    var D_d = ctx.enqueue_create_buffer[dtype](max(D_size, 1))
    var x_d = ctx.enqueue_create_buffer[dtype](x_size)
    var out_z_d = ctx.enqueue_create_buffer[dtype](out_z_size)
    var dt_d = ctx.enqueue_create_buffer[dtype](dt_size)
    var B_d = ctx.enqueue_create_buffer[dtype](B_size)
    var C_d = ctx.enqueue_create_buffer[dtype](C_size)
    var z_d = ctx.enqueue_create_buffer[dtype](z_size)
    var rmsnorm_weight_d = ctx.enqueue_create_buffer[dtype](
        max(rmsnorm_weight_size, 1)
    )
    var outproj_weight_d = ctx.enqueue_create_buffer[dtype](
        max(outproj_weight_size, 1)
    )
    var outproj_bias_d = ctx.enqueue_create_buffer[dtype](
        max(outproj_bias_size, 1)
    )
    var output_gpu_d = ctx.enqueue_create_buffer[dtype](output_size)

    # Copy to GPU
    with ctx.push_context():
        ctx.enqueue_copy(zxbcdt_d, zxbcdt_h)
        ctx.enqueue_copy(conv_weight_d, conv_weight_h)
        ctx.enqueue_copy(conv_bias_d, conv_bias_h)
        ctx.enqueue_copy(dt_bias_d, dt_bias_h)
        ctx.enqueue_copy(A_d, A_h)
        if has_D:
            ctx.enqueue_copy(D_d, D_h)
        if has_rmsnorm:
            ctx.enqueue_copy(rmsnorm_weight_d, rmsnorm_weight_h)
        if has_outproj:
            ctx.enqueue_copy(outproj_weight_d, outproj_weight_h)
            ctx.enqueue_copy(outproj_bias_d, outproj_bias_h)

    # Create TileTensors for GPU
    var zxbcdt_gpu_lt = TileTensor(
        zxbcdt_d, row_major(batch, seqlen, zxbcdt_channels)
    )
    var conv_weight_gpu_lt = TileTensor(
        conv_weight_d, row_major(conv_weight_channels, width)
    )
    var conv_bias_gpu_lt = TileTensor(conv_bias_d, row_major(conv_bias_size))
    var dt_bias_gpu_lt = TileTensor(dt_bias_d, row_major(dt_bias_size))
    var A_gpu_lt = TileTensor(A_d, row_major(A_size))
    var D_gpu_lt = TileTensor(
        D_d,
        row_major(
            (
                nheads if has_D else 0,
                headdim if has_D and D_size > nheads else 0,
            )
        ),
    )
    var x_gpu_lt = TileTensor(x_d, row_major(batch, dim, n_chunks, 2 * dstate))
    var out_z_gpu_lt = TileTensor(out_z_d, row_major(batch, dim, seqlen))
    var dt_gpu_lt = TileTensor(dt_d, row_major(batch, nheads, seqlen))
    var B_gpu_lt = TileTensor(B_d, row_major(batch, ngroups, dstate, seqlen))
    var C_gpu_lt = TileTensor(C_d, row_major(batch, ngroups, dstate, seqlen))
    var z_gpu_lt = TileTensor(z_d, row_major(batch, dim, seqlen))
    var rmsnorm_weight_gpu_lt = TileTensor(
        rmsnorm_weight_d, row_major(rmsnorm_weight_size)
    )
    var outproj_weight_gpu_lt = TileTensor(
        outproj_weight_d,
        row_major((out_dim if has_outproj else 0, dim if has_outproj else 0)),
    )
    var outproj_bias_gpu_lt = TileTensor(
        outproj_bias_d, row_major(outproj_bias_size)
    )
    var output_gpu_gpu_lt = TileTensor(
        output_gpu_d,
        row_major(batch, seqlen, out_dim if has_outproj else dim),
    )

    # Create CPU TileTensors for reference
    var zxbcdt_cpu_lt = TileTensor(
        zxbcdt_h, row_major(batch, seqlen, zxbcdt_channels)
    )
    var conv_weight_cpu_lt = TileTensor(
        conv_weight_h, row_major(conv_weight_channels, width)
    )
    var conv_bias_cpu_lt = TileTensor(conv_bias_h, row_major(conv_bias_size))
    var dt_bias_cpu_lt = TileTensor(dt_bias_h, row_major(dt_bias_size))
    var A_cpu_lt = TileTensor(A_h, row_major(A_size))
    var D_cpu_lt = TileTensor(
        D_h,
        row_major(
            (
                nheads if has_D else 0,
                headdim if has_D and D_size > nheads else 0,
            )
        ),
    )
    var x_cpu_lt = TileTensor(x_h, row_major(batch, dim, n_chunks, 2 * dstate))
    var out_z_cpu_lt = TileTensor(out_z_h, row_major(batch, dim, seqlen))
    var dt_cpu_lt = TileTensor(dt_h, row_major(batch, nheads, seqlen))
    var B_cpu_lt = TileTensor(B_h, row_major(batch, ngroups, dstate, seqlen))
    var C_cpu_lt = TileTensor(C_h, row_major(batch, ngroups, dstate, seqlen))
    var z_cpu_lt = TileTensor(z_h, row_major(batch, dim, seqlen))
    var rmsnorm_weight_cpu_lt = TileTensor(
        rmsnorm_weight_h, row_major(rmsnorm_weight_size)
    )
    var outproj_weight_cpu_lt = TileTensor(
        outproj_weight_h,
        row_major((out_dim if has_outproj else 0, dim if has_outproj else 0)),
    )
    var outproj_bias_cpu_lt = TileTensor(
        outproj_bias_h, row_major(outproj_bias_size)
    )
    var output_cpu_cpu_lt = TileTensor(
        output_cpu_h,
        row_major(batch, seqlen, out_dim if has_outproj else dim),
    )

    var epsilon = Float32(0.001)

    # Run CPU kernel
    mamba_split_conv1d_scan_combined_cpu[
        dtype,
        DSTATE,
        zxbcdt_cpu_lt.LayoutType,
        conv_weight_cpu_lt.LayoutType,
        conv_bias_cpu_lt.LayoutType,
        output_cpu_cpu_lt.LayoutType,
        x_cpu_lt.LayoutType,
        out_z_cpu_lt.LayoutType,
        dt_cpu_lt.LayoutType,
        A_cpu_lt.LayoutType,
        B_cpu_lt.LayoutType,
        C_cpu_lt.LayoutType,
        D_cpu_lt.LayoutType,
        z_cpu_lt.LayoutType,
        dt_bias_cpu_lt.LayoutType,
        rmsnorm_weight_cpu_lt.LayoutType,
        outproj_weight_cpu_lt.LayoutType,
        outproj_bias_cpu_lt.LayoutType,
    ](
        batch,
        seqlen,
        dim,
        nheads,
        headdim,
        ngroups,
        width,
        chunk_size,
        Int8(1) if delta_softplus else Int8(0),
        Int8(1) if norm_before_gate else Int8(0),
        Int8(1) if has_rmsnorm else Int8(0),
        Int8(1) if has_outproj else Int8(0),
        zxbcdt_cpu_lt.as_unsafe_any_origin(),
        conv_weight_cpu_lt.as_unsafe_any_origin(),
        conv_bias_cpu_lt.as_unsafe_any_origin(),
        dt_bias_cpu_lt.as_unsafe_any_origin(),
        A_cpu_lt.as_unsafe_any_origin(),
        D_cpu_lt.as_unsafe_any_origin(),
        x_cpu_lt.as_unsafe_any_origin(),
        out_z_cpu_lt.as_unsafe_any_origin(),
        dt_cpu_lt.as_unsafe_any_origin(),
        B_cpu_lt.as_unsafe_any_origin(),
        C_cpu_lt.as_unsafe_any_origin(),
        z_cpu_lt.as_unsafe_any_origin(),
        rmsnorm_weight_cpu_lt.as_unsafe_any_origin(),
        outproj_weight_cpu_lt.as_unsafe_any_origin(),
        outproj_bias_cpu_lt.as_unsafe_any_origin(),
        output_cpu_cpu_lt.as_unsafe_any_origin(),
        epsilon.cast[dtype](),
    )

    # Run GPU kernel
    var total_batch_dim = batch * dim
    comptime BLOCK_SIZE = 128
    var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)

    var compiled_kernel = ctx.compile_function[
        mamba_split_conv1d_scan_combined_gpu[
            dtype,
            DSTATE,
            zxbcdt_gpu_lt.LayoutType,
            conv_weight_gpu_lt.LayoutType,
            conv_bias_gpu_lt.LayoutType,
            output_gpu_gpu_lt.LayoutType,
            x_gpu_lt.LayoutType,
            out_z_gpu_lt.LayoutType,
            dt_gpu_lt.LayoutType,
            A_gpu_lt.LayoutType,
            B_gpu_lt.LayoutType,
            C_gpu_lt.LayoutType,
            D_gpu_lt.LayoutType,
            z_gpu_lt.LayoutType,
            dt_bias_gpu_lt.LayoutType,
            rmsnorm_weight_gpu_lt.LayoutType,
            outproj_weight_gpu_lt.LayoutType,
            outproj_bias_gpu_lt.LayoutType,
            zxbcdt_gpu_lt.Engine,
        ]
    ]()

    ctx.enqueue_function(
        compiled_kernel,
        Int32(total_batch_dim),
        Int32(batch),
        Int32(seqlen),
        Int32(dim),
        Int32(nheads),
        Int32(headdim),
        Int32(ngroups),
        Int32(width),
        Int32(chunk_size),
        Int8(1) if delta_softplus else Int8(0),
        Int8(1) if norm_before_gate else Int8(0),
        Int8(1) if has_rmsnorm else Int8(0),
        Int8(1) if has_outproj else Int8(0),
        zxbcdt_gpu_lt,
        conv_weight_gpu_lt,
        conv_bias_gpu_lt,
        dt_bias_gpu_lt,
        A_gpu_lt,
        D_gpu_lt,
        x_gpu_lt,
        out_z_gpu_lt,
        dt_gpu_lt,
        B_gpu_lt,
        C_gpu_lt,
        z_gpu_lt,
        rmsnorm_weight_gpu_lt,
        outproj_weight_gpu_lt,
        outproj_bias_gpu_lt,
        output_gpu_gpu_lt,
        epsilon.cast[dtype](),
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,),
    )

    # Wait for GPU kernel to complete
    ctx.synchronize()

    # Copy GPU results back (CPU results are already in host memory)
    with ctx.push_context():
        ctx.enqueue_copy(output_gpu_h, output_gpu_d)
    ctx.synchronize()

    # Compare GPU output vs CPU output
    for i in range(output_size):
        assert_almost_equal(
            output_cpu_h[i],
            output_gpu_h[i],
            rtol=rtol,
        )

    # ===--- Reference implementation for numerical verification ---=== #
    # Mirrors the kernel's 6-stage pipeline:
    # 1. Split zxbcdt -> z, xBC, dt
    # 2. Causal conv1d on x, B, C channels with SiLU activation
    # 3. Selective scan (SSM recurrence)
    # 4. Gating with z (optional RMSNorm)
    # 5. Store output (no outproj in current tests)
    var flattened_size = batch * seqlen * dim
    var output_ref_h = ctx.enqueue_create_host_buffer[dtype](flattened_size)

    # Channel offsets within zxbcdt
    var z_start = 0
    var xBC_start = dim
    var dt_start_ch = 2 * dim + 2 * ngroups * dstate

    for b_idx in range(batch):
        for d_idx in range(dim):
            var h, p = divmod(d_idx, headdim)
            var group_id = h // ngroups if ngroups > 1 else 0

            # Pre-load A value (same for all DSTATE entries within a head)
            var A_val_raw = Float32(A_h[h])
            var A_ref = SIMD[.float32, MAX_DSTATE](0.0)
            for n in range(dstate):
                A_ref[n] = A_val_raw * LOG2E

            # Load D value: D is (nheads, headdim)
            var D_val = Float32(0)
            if has_D:
                D_val = Float32(D_h[h * headdim + p])

            # Load dt_bias for this head
            var dt_bias_val = Float32(dt_bias_h[h])

            # Load rmsnorm weight for this dim
            var rmsnorm_w = Float32(0)
            if has_rmsnorm:
                rmsnorm_w = Float32(rmsnorm_weight_h[d_idx])

            # Initialize state for selective scan
            var state_ref = SIMD[.float32, MAX_DSTATE](0.0)

            for t in range(seqlen):
                # --- Step 1: Load z from zxbcdt ---
                var z_offset = (
                    b_idx * seqlen * zxbcdt_channels
                    + t * zxbcdt_channels
                    + (z_start + d_idx)
                )
                var z_val = Float32(zxbcdt_h[z_offset])

                # --- Step 2: Load dt, apply bias and softplus ---
                var dt_offset = (
                    b_idx * seqlen * zxbcdt_channels
                    + t * zxbcdt_channels
                    + (dt_start_ch + h)
                )
                var dt_val = Float32(zxbcdt_h[dt_offset])
                dt_val += dt_bias_val
                if delta_softplus:
                    dt_val = softplus_ref(dt_val)

                # --- Step 3: Causal conv1d for x channel ---
                var x_channel = d_idx
                var conv_sum_x = Float32(conv_bias_h[x_channel])
                for w in range(width):
                    var input_t = t - (width - 1 - w)
                    if input_t >= 0:
                        var xbc_off = (
                            b_idx * seqlen * zxbcdt_channels
                            + input_t * zxbcdt_channels
                            + (xBC_start + x_channel)
                        )
                        var wt_off = x_channel * width + w
                        conv_sum_x += Float32(zxbcdt_h[xbc_off]) * Float32(
                            conv_weight_h[wt_off]
                        )
                var x_val = silu_ref(conv_sum_x)

                # --- Step 4: Causal conv1d for B and C channels ---
                var B_vals = SIMD[.float32, MAX_DSTATE](0.0)
                var C_vals = SIMD[.float32, MAX_DSTATE](0.0)
                for n in range(dstate):
                    # B channel: dim + group_id * dstate + n (in xBC space)
                    var B_ch = dim + group_id * dstate + n
                    var B_conv = Float32(conv_bias_h[B_ch])
                    for w in range(width):
                        var input_t = t - (width - 1 - w)
                        if input_t >= 0:
                            var xbc_off = (
                                b_idx * seqlen * zxbcdt_channels
                                + input_t * zxbcdt_channels
                                + (xBC_start + B_ch)
                            )
                            var wt_off = B_ch * width + w
                            B_conv += Float32(zxbcdt_h[xbc_off]) * Float32(
                                conv_weight_h[wt_off]
                            )
                    B_vals[n] = silu_ref(B_conv)

                    # C channel: dim + ngroups*dstate + group_id*dstate + n
                    var C_ch = dim + ngroups * dstate + group_id * dstate + n
                    var C_conv = Float32(conv_bias_h[C_ch])
                    for w in range(width):
                        var input_t = t - (width - 1 - w)
                        if input_t >= 0:
                            var xbc_off = (
                                b_idx * seqlen * zxbcdt_channels
                                + input_t * zxbcdt_channels
                                + (xBC_start + C_ch)
                            )
                            var wt_off = C_ch * width + w
                            C_conv += Float32(zxbcdt_h[xbc_off]) * Float32(
                                conv_weight_h[wt_off]
                            )
                    C_vals[n] = silu_ref(C_conv)

                # --- Step 5: Selective scan ---
                var a_t = exp2(dt_val * A_ref)
                var b_t = B_vals * dt_val
                state_ref = state_ref * a_t + b_t
                var ss_output = (state_ref * C_vals).reduce_add()

                if has_D:
                    ss_output += D_val * x_val

                # --- Step 6: Apply gating and optional RMSNorm ---
                var out_val = ss_output
                if has_rmsnorm:
                    var eps = Float32(0.001)
                    if norm_before_gate:
                        var norm_val = (
                            rsqrt(out_val * out_val + eps) * rmsnorm_w
                        )
                        out_val = out_val * norm_val * silu_ref(z_val)
                    else:
                        var gated = out_val * silu_ref(z_val)
                        var norm_val = rsqrt(gated * gated + eps) * rmsnorm_w
                        out_val = gated * norm_val
                else:
                    out_val = out_val * silu_ref(z_val)

                # Store reference output: (batch, seqlen, dim) row-major
                var out_offset = b_idx * seqlen * dim + t * dim + d_idx
                output_ref_h[out_offset] = Scalar[dtype](out_val)

    # Compare CPU output vs reference
    for i in range(flattened_size):
        assert_almost_equal(
            output_cpu_h[i],
            output_ref_h[i],
            rtol=rtol,
        )

    # Device buffers are automatically freed when they go out of scope
    _ = zxbcdt_d^
    _ = conv_weight_d^
    _ = conv_bias_d^
    _ = dt_bias_d^
    _ = A_d^
    _ = D_d^
    _ = x_d^
    _ = out_z_d^
    _ = dt_d^
    _ = B_d^
    _ = C_d^
    _ = z_d^
    _ = rmsnorm_weight_d^
    _ = outproj_weight_d^
    _ = outproj_bias_d^
    _ = output_gpu_d^


def test_mamba_combined_gpu_basic() raises:
    """Test basic mamba_split_conv1d_scan_combined on GPU."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_mamba_split_conv1d_scan_combined_gpu[
        DType.float32,
        4,  # DSTATE
        has_D=True,
        has_rmsnorm=False,
        has_outproj=False,
        norm_before_gate=True,
        delta_softplus=True,
    ](
        batch=2,
        seqlen=8,
        dim=4,
        nheads=2,
        headdim=2,
        ngroups=1,
        width=4,
        chunk_size=4,
        ctx=ctx,
    )


def test_mamba_combined_gpu_without_D() raises:
    """Test mamba_split_conv1d_scan_combined on GPU without D."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_mamba_split_conv1d_scan_combined_gpu[
        DType.float32,
        4,  # DSTATE
        has_D=False,
        has_rmsnorm=False,
        has_outproj=False,
        norm_before_gate=True,
        delta_softplus=True,
    ](
        batch=2,
        seqlen=8,
        dim=4,
        nheads=2,
        headdim=2,
        ngroups=1,
        width=4,
        chunk_size=4,
        ctx=ctx,
    )


def test_mamba_combined_gpu_with_rmsnorm() raises:
    """Test mamba_split_conv1d_scan_combined on GPU with RMSNorm."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_mamba_split_conv1d_scan_combined_gpu[
        DType.float32,
        4,  # DSTATE
        has_D=True,
        has_rmsnorm=True,
        has_outproj=False,
        norm_before_gate=True,
        delta_softplus=True,
    ](
        batch=2,
        seqlen=8,
        dim=4,
        nheads=2,
        headdim=2,
        ngroups=1,
        width=4,
        chunk_size=4,
        ctx=ctx,
    )


def test_mamba_combined_gpu_norm_after_gate() raises:
    """Test mamba_split_conv1d_scan_combined on GPU with norm_after_gate."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_mamba_split_conv1d_scan_combined_gpu[
        DType.float32,
        4,  # DSTATE
        has_D=True,
        has_rmsnorm=False,
        has_outproj=False,
        norm_before_gate=False,
        delta_softplus=True,
    ](
        batch=2,
        seqlen=8,
        dim=4,
        nheads=2,
        headdim=2,
        ngroups=1,
        width=4,
        chunk_size=4,
        ctx=ctx,
    )


def test_mamba_combined_gpu_without_delta_softplus() raises:
    """Test mamba_split_conv1d_scan_combined on GPU without delta_softplus."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_mamba_split_conv1d_scan_combined_gpu[
        DType.float32,
        4,  # DSTATE
        has_D=True,
        has_rmsnorm=False,
        has_outproj=False,
        norm_before_gate=True,
        delta_softplus=False,
    ](
        batch=2,
        seqlen=8,
        dim=4,
        nheads=2,
        headdim=2,
        ngroups=1,
        width=4,
        chunk_size=4,
        ctx=ctx,
    )


def test_mamba_combined_gpu_larger_shapes() raises:
    """Test mamba_split_conv1d_scan_combined on GPU with larger shapes."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_mamba_split_conv1d_scan_combined_gpu[
        DType.float32,
        8,  # DSTATE
        has_D=True,
        has_rmsnorm=False,
        has_outproj=False,
        norm_before_gate=True,
        delta_softplus=True,
    ](
        batch=1,
        seqlen=32,
        dim=16,
        nheads=4,
        headdim=4,
        ngroups=2,
        width=4,
        chunk_size=8,
        ctx=ctx,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
