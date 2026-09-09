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

from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout._fillers import random
from state_space.selective_scan import (
    mamba_split_conv1d_scan_combined_cpu,
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


def run_mamba_split_conv1d_scan_combined[
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
    rtol: Float64 = 0.01,
) raises:
    comptime dstate = DSTATE
    var n_chunks = ceildiv(seqlen, chunk_size)

    # zxbcdt: (batch, seqlen, 2*dim + 2*ngroups*dstate + nheads)
    var zxbcdt_channels = 2 * dim + 2 * ngroups * dstate + nheads
    var zxbcdt_size = batch * seqlen * zxbcdt_channels
    var zxbcdt_heap = List(length=zxbcdt_size, fill=Scalar[dtype](0))
    comptime layout_3d = Layout.row_major[3]()
    var zxbcdt_h = TileTensor(
        zxbcdt_heap, row_major(batch, seqlen, zxbcdt_channels)
    )

    # conv_weight: (dim + 2*ngroups*dstate, width)
    var conv_weight_channels = dim + 2 * ngroups * dstate
    var conv_weight_size = conv_weight_channels * width
    var conv_weight_heap = List(length=conv_weight_size, fill=Scalar[dtype](0))
    comptime layout_2d = Layout.row_major[2]()
    var conv_weight_h = TileTensor(
        conv_weight_heap, row_major(conv_weight_channels, width)
    )

    # conv_bias: (dim + 2*ngroups*dstate,)
    var conv_bias_size = conv_weight_channels
    var conv_bias_heap = List(length=conv_bias_size, fill=Scalar[dtype](0))
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    var conv_bias_h = TileTensor(conv_bias_heap, row_major(conv_bias_size))

    # dt_bias: (nheads,)
    var dt_bias_size = nheads
    var dt_bias_heap = List(length=dt_bias_size, fill=Scalar[dtype](0))
    var dt_bias_h = TileTensor(dt_bias_heap, row_major(dt_bias_size))

    # A: (nheads,)
    var A_size = nheads
    var A_heap = List(length=A_size, fill=Scalar[dtype](0))
    var A_h = TileTensor(A_heap, row_major(A_size))

    # D: (nheads, headdim) or (nheads,)
    var D_size = nheads * headdim if has_D else 0
    var D_heap = List(length=max(D_size, 1), fill=Scalar[dtype](0))
    var D_h = TileTensor(
        D_heap,
        row_major(
            (
                nheads if has_D else 0,
                headdim if has_D and D_size > nheads else 0,
            )
        ),
    )

    # x: (batch, dim, num_chunks, 2*dstate)
    var x_size = batch * dim * n_chunks * 2 * dstate
    var x_heap = List(length=x_size, fill=Scalar[dtype](0))
    comptime layout_4d = Layout.row_major[4]()
    var x_h = TileTensor(x_heap, row_major(batch, dim, n_chunks, 2 * dstate))

    # out_z: (batch, dim, seqlen)
    var out_z_size = batch * dim * seqlen
    var out_z_heap = List(length=out_z_size, fill=Scalar[dtype](0))
    var out_z_h = TileTensor(out_z_heap, row_major(batch, dim, seqlen))

    # dt: (batch, nheads, seqlen)
    var dt_size = batch * nheads * seqlen
    var dt_heap = List(length=dt_size, fill=Scalar[dtype](0))
    var dt_h = TileTensor(dt_heap, row_major(batch, nheads, seqlen))

    # B: (batch, ngroups, dstate, seqlen)
    var B_size = batch * ngroups * dstate * seqlen
    var B_heap = List(length=B_size, fill=Scalar[dtype](0))
    var B_h = TileTensor(B_heap, row_major(batch, ngroups, dstate, seqlen))

    # C: (batch, ngroups, dstate, seqlen)
    var C_size = batch * ngroups * dstate * seqlen
    var C_heap = List(length=C_size, fill=Scalar[dtype](0))
    var C_h = TileTensor(C_heap, row_major(batch, ngroups, dstate, seqlen))

    # z: (batch, dim, seqlen)
    var z_size = batch * dim * seqlen
    var z_heap = List(length=z_size, fill=Scalar[dtype](0))
    var z_h = TileTensor(z_heap, row_major(batch, dim, seqlen))

    # rmsnorm_weight: (dim,)
    var rmsnorm_weight_size = dim if has_rmsnorm else 0
    var rmsnorm_weight_heap = List(
        length=max(rmsnorm_weight_size, 1), fill=Scalar[dtype](0)
    )
    var rmsnorm_weight_h = TileTensor(
        rmsnorm_weight_heap, row_major(rmsnorm_weight_size)
    )

    # outproj_weight: (out_dim, dim)
    var out_dim = dim  # For simplicity, use same as input dim
    var outproj_weight_size = out_dim * dim if has_outproj else 0
    var outproj_weight_heap = List(
        length=max(outproj_weight_size, 1), fill=Scalar[dtype](0)
    )
    var outproj_weight_h = TileTensor(
        outproj_weight_heap,
        row_major((out_dim if has_outproj else 0, dim if has_outproj else 0)),
    )

    # outproj_bias: (out_dim,)
    var outproj_bias_size = out_dim if has_outproj else 0
    var outproj_bias_heap = List(
        length=max(outproj_bias_size, 1), fill=Scalar[dtype](0)
    )
    var outproj_bias_h = TileTensor(
        outproj_bias_heap, row_major(outproj_bias_size)
    )

    # output: (batch, seqlen, dim) or (batch, seqlen, out_dim)
    var output_size = batch * seqlen * (out_dim if has_outproj else dim)
    var output_heap = List(length=output_size, fill=Scalar[dtype](0))
    var output_h = TileTensor(
        output_heap,
        row_major(batch, seqlen, out_dim if has_outproj else dim),
    )

    # Initialize data
    random(zxbcdt_h)
    random(conv_weight_h)
    random(conv_bias_h)
    random(dt_bias_h)
    random(A_h)
    if has_D:
        random(D_h)
    if has_rmsnorm:
        random(rmsnorm_weight_h)
        # Make positive
        for i in range(dim):
            rmsnorm_weight_h._storage[i] = abs(
                rmsnorm_weight_h._storage[i]
            ) + Scalar[dtype](0.1)
    if has_outproj:
        random(outproj_weight_h)
        random(outproj_bias_h)

    var epsilon = Float32(0.001)

    # Call kernel
    mamba_split_conv1d_scan_combined_cpu[
        dtype,
        DSTATE,
        zxbcdt_h.LayoutType,
        conv_weight_h.LayoutType,
        conv_bias_h.LayoutType,
        output_h.LayoutType,
        x_h.LayoutType,
        out_z_h.LayoutType,
        dt_h.LayoutType,
        A_h.LayoutType,
        B_h.LayoutType,
        C_h.LayoutType,
        D_h.LayoutType,
        z_h.LayoutType,
        dt_bias_h.LayoutType,
        rmsnorm_weight_h.LayoutType,
        outproj_weight_h.LayoutType,
        outproj_bias_h.LayoutType,
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
        zxbcdt_h,
        conv_weight_h,
        conv_bias_h,
        dt_bias_h,
        A_h,
        D_h,
        x_h,
        out_z_h,
        dt_h,
        B_h,
        C_h,
        z_h,
        rmsnorm_weight_h,
        outproj_weight_h,
        outproj_bias_h,
        output_h,
        epsilon.cast[dtype](),
    )

    # ===--- Reference implementation for numerical verification ---=== #
    # Mirrors the kernel's 6-stage pipeline:
    # 1. Split zxbcdt -> z, xBC, dt
    # 2. Causal conv1d on x, B, C channels with SiLU activation
    # 3. Selective scan (SSM recurrence)
    # 4. Gating with z (optional RMSNorm)
    # 5. Store output (no outproj in current tests)
    var output_ref_size = batch * seqlen * dim
    var output_ref_heap = List(length=output_ref_size, fill=Scalar[dtype](0))

    # Channel offsets within zxbcdt
    var z_start = 0
    var xBC_start = dim
    var dt_start_ch = 2 * dim + 2 * ngroups * dstate

    for b_idx in range(batch):
        for d_idx in range(dim):
            var h, p = divmod(d_idx, headdim)
            var group_id = h // ngroups if ngroups > 1 else 0

            # Pre-load A value (same for all DSTATE entries within a head)
            var A_val_raw = Float32(A_h._storage[h])
            var A_ref = SIMD[.float32, MAX_DSTATE](0.0)
            for n in range(dstate):
                A_ref[n] = A_val_raw * LOG2E

            # Load D value: D is (nheads, headdim)
            var D_val = Float32(0)
            if has_D:
                D_val = Float32(D_h._storage[h * headdim + p])

            # Load dt_bias for this head
            var dt_bias_val = Float32(dt_bias_h._storage[h])

            # Load rmsnorm weight for this dim
            var rmsnorm_w = Float32(0)
            if has_rmsnorm:
                rmsnorm_w = Float32(rmsnorm_weight_h._storage[d_idx])

            # Initialize state for selective scan
            var state_ref = SIMD[.float32, MAX_DSTATE](0.0)

            for t in range(seqlen):
                # --- Step 1: Load z from zxbcdt ---
                var z_offset = (
                    b_idx * seqlen * zxbcdt_channels
                    + t * zxbcdt_channels
                    + (z_start + d_idx)
                )
                var z_val = Float32(zxbcdt_h._storage[z_offset])

                # --- Step 2: Load dt, apply bias and softplus ---
                var dt_offset = (
                    b_idx * seqlen * zxbcdt_channels
                    + t * zxbcdt_channels
                    + (dt_start_ch + h)
                )
                var dt_val = Float32(zxbcdt_h._storage[dt_offset])
                dt_val += dt_bias_val
                if delta_softplus:
                    dt_val = softplus_ref(dt_val)

                # --- Step 3: Causal conv1d for x channel ---
                var x_channel = d_idx  # channel in xBC space
                var conv_sum_x = Float32(conv_bias_h._storage[x_channel])
                for w in range(width):
                    var input_t = t - (width - 1 - w)
                    if input_t >= 0:
                        var xbc_off = (
                            b_idx * seqlen * zxbcdt_channels
                            + input_t * zxbcdt_channels
                            + (xBC_start + x_channel)
                        )
                        var wt_off = x_channel * width + w
                        conv_sum_x += Float32(
                            zxbcdt_h._storage[xbc_off]
                        ) * Float32(conv_weight_h._storage[wt_off])
                var x_val = silu_ref(conv_sum_x)

                # --- Step 4: Causal conv1d for B and C channels ---
                var B_vals = SIMD[.float32, MAX_DSTATE](0.0)
                var C_vals = SIMD[.float32, MAX_DSTATE](0.0)
                for n in range(dstate):
                    # B channel: dim + group_id * dstate + n (in xBC space)
                    var B_ch = dim + group_id * dstate + n
                    var B_conv = Float32(conv_bias_h._storage[B_ch])
                    for w in range(width):
                        var input_t = t - (width - 1 - w)
                        if input_t >= 0:
                            var xbc_off = (
                                b_idx * seqlen * zxbcdt_channels
                                + input_t * zxbcdt_channels
                                + (xBC_start + B_ch)
                            )
                            var wt_off = B_ch * width + w
                            B_conv += Float32(
                                zxbcdt_h._storage[xbc_off]
                            ) * Float32(conv_weight_h._storage[wt_off])
                    B_vals[n] = silu_ref(B_conv)

                    # C channel: dim + ngroups*dstate + group_id*dstate + n
                    var C_ch = dim + ngroups * dstate + group_id * dstate + n
                    var C_conv = Float32(conv_bias_h._storage[C_ch])
                    for w in range(width):
                        var input_t = t - (width - 1 - w)
                        if input_t >= 0:
                            var xbc_off = (
                                b_idx * seqlen * zxbcdt_channels
                                + input_t * zxbcdt_channels
                                + (xBC_start + C_ch)
                            )
                            var wt_off = C_ch * width + w
                            C_conv += Float32(
                                zxbcdt_h._storage[xbc_off]
                            ) * Float32(conv_weight_h._storage[wt_off])
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
                    var eps = epsilon
                    if norm_before_gate:
                        # RMSNorm(x) * SiLU(z)
                        var norm_val = (
                            rsqrt(out_val * out_val + eps) * rmsnorm_w
                        )
                        out_val = out_val * norm_val * silu_ref(z_val)
                    else:
                        # RMSNorm(x * SiLU(z))
                        var gated = out_val * silu_ref(z_val)
                        var norm_val = rsqrt(gated * gated + eps) * rmsnorm_w
                        out_val = gated * norm_val
                else:
                    out_val = out_val * silu_ref(z_val)

                # Store reference output: (batch, seqlen, dim) row-major
                var out_offset = b_idx * seqlen * dim + t * dim + d_idx
                output_ref_heap[out_offset] = Scalar[dtype](out_val)

    # Compare kernel output vs reference
    for i in range(output_ref_size):
        assert_almost_equal(
            output_h._storage[i],
            output_ref_heap[i],
            rtol=rtol,
        )


def test_mamba_combined_basic() raises:
    """Test basic mamba_split_conv1d_scan_combined."""
    run_mamba_split_conv1d_scan_combined[
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
    )


def test_mamba_combined_without_D() raises:
    """Test mamba_split_conv1d_scan_combined without D."""
    run_mamba_split_conv1d_scan_combined[
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
    )


def test_mamba_combined_with_rmsnorm() raises:
    """Test mamba_split_conv1d_scan_combined with RMSNorm."""
    run_mamba_split_conv1d_scan_combined[
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
    )


def test_mamba_combined_norm_after_gate() raises:
    """Test mamba_split_conv1d_scan_combined with norm_after_gate."""
    run_mamba_split_conv1d_scan_combined[
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
    )


def test_mamba_combined_without_delta_softplus() raises:
    """Test mamba_split_conv1d_scan_combined without delta_softplus."""
    run_mamba_split_conv1d_scan_combined[
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
    )


def test_mamba_combined_larger_shapes() raises:
    """Test mamba_split_conv1d_scan_combined with larger shapes."""
    run_mamba_split_conv1d_scan_combined[
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
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
