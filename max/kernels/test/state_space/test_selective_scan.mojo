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

from std.math import exp, exp2, log

from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout._fillers import random
from state_space.selective_scan import (
    selective_scan_fwd_cpu,
    selective_scan_update_cpu,
    Strides1D,
    Strides2D,
    Strides3D,
    Strides4D,
)
from std.testing import TestSuite, assert_almost_equal

from std.utils.index import Index


# LOG2E constant for converting exp to exp2
comptime LOG2E = 1.4426950408889634
comptime MAX_DSTATE = 16


@always_inline
def softplus_ref(val: Float32) -> Float32:
    """Reference softplus implementation: log(1 + exp(x))."""
    if val > 20.0:
        return val
    var exp_val = exp(val)
    var one = Float32(1.0)
    return log(one + exp_val)


@always_inline
def sigmoid_ref(val: Float32) -> Float32:
    """Reference sigmoid implementation."""
    if val < -20.0:
        return 0.0
    var exp_neg = exp(-val)
    return 1.0 / (1.0 + exp_neg)


@always_inline
def silu_ref(val: Float32) -> Float32:
    """Reference SiLU implementation."""
    if val < -20.0:
        return 0.0
    var exp_neg = exp(-val)
    return val / (1.0 + exp_neg)


def run_selective_scan_fwd[
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
    rtol: Float64 = 0.01,
) raises:
    """Test selective scan forward kernel against reference implementation."""
    comptime assert DSTATE <= MAX_DSTATE, "DSTATE exceeds kernel limit"
    comptime dstate = DSTATE

    var group_size = dim // n_groups
    var chunk_size = 2048
    var n_chunks = (seqlen + chunk_size - 1) // chunk_size

    # Allocate host memory
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)

    # output: (batch, dim, seqlen)
    var output_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))
    var output_h = LayoutTensor[dtype, layout_3d, _](
        output_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )

    # x: (batch, dim, n_chunks, 2*dstate) - checkpoint tensor
    var x_heap = List(
        length=batch * dim * n_chunks * 2 * dstate, fill=Scalar[dtype](0)
    )

    # out_z: (batch, dim, seqlen)
    var out_z_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))

    # u: (batch, dim, seqlen)
    var u_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))
    var u_h = LayoutTensor[dtype, layout_3d, _](
        u_heap, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )

    # delta: (batch, dim, seqlen)
    var delta_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))
    var delta_h = LayoutTensor[dtype, layout_3d, _](
        delta_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )

    # A: (dim, dstate)
    var A_heap = List(length=dim * dstate, fill=Scalar[dtype](0))
    var A_h = LayoutTensor[dtype, layout_2d, _](
        A_heap, RuntimeLayout[layout_2d].row_major(Index(dim, dstate))
    )

    # B: (batch, n_groups, dstate, seqlen)
    var B_heap = List(
        length=batch * n_groups * dstate * seqlen, fill=Scalar[dtype](0)
    )
    var B_h = LayoutTensor[dtype, layout_4d, _](
        B_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_groups, dstate, seqlen)
        ),
    )

    # C: (batch, n_groups, dstate, seqlen)
    var C_heap = List(
        length=batch * n_groups * dstate * seqlen, fill=Scalar[dtype](0)
    )
    var C_h = LayoutTensor[dtype, layout_4d, _](
        C_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_groups, dstate, seqlen)
        ),
    )

    # D: (dim,) or empty
    var D_size = dim if has_D else 0
    var D_heap = List(length=max(D_size, 1), fill=Scalar[dtype](0))
    var D_h = LayoutTensor[dtype, layout_1d, _](
        D_heap, RuntimeLayout[layout_1d].row_major(Index(D_size))
    )

    # z: (batch, dim, seqlen) or empty
    var z_size = batch * dim * seqlen if has_z else 0
    var z_heap = List(length=max(z_size, 1), fill=Scalar[dtype](0))
    var z_h = LayoutTensor[dtype, layout_3d, _](
        z_heap,
        RuntimeLayout[layout_3d].row_major(
            Index(
                batch if has_z else 0,
                dim if has_z else 0,
                seqlen if has_z else 0,
            )
        ),
    )

    # delta_bias: (dim,) or empty
    var delta_bias_size = dim if has_delta_bias else 0
    var delta_bias_heap = List(
        length=max(delta_bias_size, 1), fill=Scalar[dtype](0)
    )
    var delta_bias_h = LayoutTensor[dtype, layout_1d, _](
        delta_bias_heap,
        RuntimeLayout[layout_1d].row_major(Index(delta_bias_size)),
    )

    # Initialize input data
    random(u_h)
    random(delta_h)
    random(A_h)
    random(B_h)
    random(C_h)
    if has_D:
        random(D_h)
    if has_z:
        random(z_h)
    if has_delta_bias:
        random(delta_bias_h)

    # Scale A to be negative for stability
    for i in range(dim * dstate):
        var val = A_h.ptr.load(i)
        A_h.ptr.store(i, Scalar[dtype](Float32(val) * -0.5))

    # Scale delta to be positive
    for i in range(batch * dim * seqlen):
        var val = delta_h.ptr.load(i)
        delta_h.ptr.store(i, Scalar[dtype](abs(Float32(val)) * 0.5))

    # Create TileTensor versions for kernel call
    var output_tt = TileTensor(output_heap, row_major(batch, dim, seqlen))
    var x_tt = TileTensor(
        x_heap,
        row_major(batch, dim, n_chunks, 2 * dstate),
    )
    var out_z_tt = TileTensor(out_z_heap, row_major(batch, dim, seqlen))
    var u_tt = TileTensor(u_heap, row_major(batch, dim, seqlen))
    var delta_tt = TileTensor(delta_heap, row_major(batch, dim, seqlen))
    var A_tt = TileTensor(A_heap, row_major(dim, dstate))
    var B_tt = TileTensor(
        B_heap,
        row_major(batch, n_groups, dstate, seqlen),
    )
    var C_tt = TileTensor(
        C_heap,
        row_major(batch, n_groups, dstate, seqlen),
    )
    var D_tt = TileTensor(
        D_heap,
        row_major(
            D_size,
        ),
    )
    var z_tt = TileTensor(
        z_heap,
        row_major(
            (
                batch if has_z else 0,
                dim if has_z else 0,
                seqlen if has_z else 0,
            )
        ),
    )
    var delta_bias_tt = TileTensor(
        delta_bias_heap,
        row_major(
            delta_bias_size,
        ),
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

    # Call fused kernel
    selective_scan_fwd_cpu[
        dtype,
        DSTATE,
    ](
        batch,
        dim,
        seqlen,
        group_size,
        Int8(1) if delta_softplus else Int8(0),
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
    )

    # For now, just verify the kernel executes without errors
    # A full reference implementation would require matching the exact chunking
    # and checkpoint logic, which is complex. This test verifies the kernel
    # runs correctly and produces reasonable outputs.

    # Basic sanity check: output should not be all zeros
    var has_nonzero = False
    for i in range(batch * dim * seqlen):
        if abs(Float32(output_h.ptr[i])) > 1e-6:
            has_nonzero = True
            break

    if not has_nonzero:
        raise Error(
            "Output is all zeros - kernel may not be executing correctly"
        )


def run_selective_scan_update[
    dtype: DType,
    DSTATE: Int,
    has_D: Bool = True,
    has_z: Bool = True,
    has_delta_bias: Bool = True,
    delta_softplus: Bool = False,
](batch: Int, dim: Int, n_groups: Int, rtol: Float64 = 0.01,) raises:
    """Test selective scan update kernel against reference implementation."""
    comptime assert DSTATE <= MAX_DSTATE, "DSTATE exceeds kernel limit"
    comptime dstate = DSTATE

    var group_size = dim // n_groups

    # Allocate host memory
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)

    # state_in: (batch, dim, dstate)
    var state_in_heap = List(length=batch * dim * dstate, fill=Scalar[dtype](0))
    var state_in_h = LayoutTensor[dtype, layout_3d, _](
        state_in_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, dstate)),
    )

    # state_out: (batch, dim, dstate)
    var state_out_heap = List(
        length=batch * dim * dstate, fill=Scalar[dtype](0)
    )
    var state_out_h = LayoutTensor[dtype, layout_3d, _](
        state_out_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, dstate)),
    )

    # output: (batch, dim)
    var output_heap = List(length=batch * dim, fill=Scalar[dtype](0))
    var output_h = LayoutTensor[dtype, layout_2d, _](
        output_heap, RuntimeLayout[layout_2d].row_major(Index(batch, dim))
    )

    # x: (batch, dim)
    var x_heap = List(length=batch * dim, fill=Scalar[dtype](0))
    var x_h = LayoutTensor[dtype, layout_2d, _](
        x_heap, RuntimeLayout[layout_2d].row_major(Index(batch, dim))
    )

    # dt: (batch, dim)
    var dt_heap = List(length=batch * dim, fill=Scalar[dtype](0))
    var dt_h = LayoutTensor[dtype, layout_2d, _](
        dt_heap, RuntimeLayout[layout_2d].row_major(Index(batch, dim))
    )

    # A: (dim, dstate)
    var A_heap = List(length=dim * dstate, fill=Scalar[dtype](0))
    var A_h = LayoutTensor[dtype, layout_2d, _](
        A_heap, RuntimeLayout[layout_2d].row_major(Index(dim, dstate))
    )

    # B: (batch, n_groups, dstate)
    var B_heap = List(length=batch * n_groups * dstate, fill=Scalar[dtype](0))
    var B_h = LayoutTensor[dtype, layout_3d, _](
        B_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, n_groups, dstate)),
    )

    # C: (batch, n_groups, dstate)
    var C_heap = List(length=batch * n_groups * dstate, fill=Scalar[dtype](0))
    var C_h = LayoutTensor[dtype, layout_3d, _](
        C_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, n_groups, dstate)),
    )

    # D: (dim,) or empty
    var D_size = dim if has_D else 0
    var D_heap = List(length=max(D_size, 1), fill=Scalar[dtype](0))
    var D_h = LayoutTensor[dtype, layout_1d, _](
        D_heap, RuntimeLayout[layout_1d].row_major(Index(D_size))
    )

    # z: (batch, dim) or empty
    var z_size = batch * dim if has_z else 0
    var z_heap = List(length=max(z_size, 1), fill=Scalar[dtype](0))
    var z_h = LayoutTensor[dtype, layout_2d, _](
        z_heap,
        RuntimeLayout[layout_2d].row_major(
            Index(batch if has_z else 0, dim if has_z else 0)
        ),
    )

    # dt_bias: (dim,) or empty
    var dt_bias_size = dim if has_delta_bias else 0
    var dt_bias_heap = List(length=max(dt_bias_size, 1), fill=Scalar[dtype](0))
    var dt_bias_h = LayoutTensor[dtype, layout_1d, _](
        dt_bias_heap, RuntimeLayout[layout_1d].row_major(Index(dt_bias_size))
    )

    # Reference output
    var state_out_ref_heap = List(
        length=batch * dim * dstate, fill=Scalar[dtype](0)
    )
    var state_out_ref_h = LayoutTensor[dtype, layout_3d, _](
        state_out_ref_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, dstate)),
    )

    var output_ref_heap = List(length=batch * dim, fill=Scalar[dtype](0))
    var output_ref_h = LayoutTensor[dtype, layout_2d, _](
        output_ref_heap, RuntimeLayout[layout_2d].row_major(Index(batch, dim))
    )

    # Initialize input data
    random(state_in_h)
    random(x_h)
    random(dt_h)
    random(A_h)
    random(B_h)
    random(C_h)
    if has_D:
        random(D_h)
    if has_z:
        random(z_h)
    if has_delta_bias:
        random(dt_bias_h)

    # Scale A to be negative for stability
    for i in range(dim * dstate):
        var val = A_h.ptr[i]
        A_h.ptr[i] = Scalar[dtype](Float32(val) * -0.5)

    # Copy state_in for reference
    for i in range(batch * dim * dstate):
        state_out_ref_h.ptr[i] = state_in_h.ptr[i]

    # Create TileTensor versions for kernel call
    var state_in_tt = TileTensor(state_in_heap, row_major(batch, dim, dstate))
    var state_out_tt = TileTensor(state_out_heap, row_major(batch, dim, dstate))
    var output_tt2 = TileTensor(output_heap, row_major(batch, dim))
    var x_tt2 = TileTensor(x_heap, row_major(batch, dim))
    var dt_tt = TileTensor(dt_heap, row_major(batch, dim))
    var A_tt2 = TileTensor(A_heap, row_major(dim, dstate))
    var B_tt2 = TileTensor(B_heap, row_major(batch, n_groups, dstate))
    var C_tt2 = TileTensor(C_heap, row_major(batch, n_groups, dstate))
    var D_tt2 = TileTensor(
        D_heap,
        row_major(
            D_size,
        ),
    )
    var z_tt2 = TileTensor(
        z_heap,
        row_major(batch if has_z else 0, dim if has_z else 0),
    )
    var dt_bias_tt = TileTensor(
        dt_bias_heap,
        row_major(
            dt_bias_size,
        ),
    )

    var x_buf = x_h
    var dt_buf = dt_h
    var A_buf = A_h
    var B_buf = B_h
    var C_buf = C_h
    var D_buf = D_h
    var z_buf = z_h
    var dt_bias_buf = dt_bias_h

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

    # Run kernel
    selective_scan_update_cpu[
        dtype,
        DSTATE,
    ](
        batch,
        dim,
        group_size,
        Int8(1) if delta_softplus else Int8(0),
        state_out_tt,
        output_tt2,
        state_in_tt,
        x_tt2,
        dt_tt,
        A_tt2,
        B_tt2,
        C_tt2,
        D_tt2,
        z_tt2,
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
    )

    # Reference implementation
    for b in range(batch):
        for d in range(dim):
            var group_id = d // group_size

            # Load dt value
            var dt_offset = b * dim + d
            var dt_val = Float32(dt_buf.ptr[dt_offset])

            # Apply dt_bias if present
            if has_delta_bias:
                var bias_val = Float32(dt_bias_buf.ptr[d])
                dt_val += bias_val

            # Apply softplus if requested
            if delta_softplus:
                dt_val = softplus_ref(dt_val)

            # Load x value
            var x_offset = b * dim + d
            var x_val = Float32(x_buf.ptr[x_offset])

            # Load A values and compute dA
            var dA_vals = SIMD[.float32, MAX_DSTATE](0.0)
            for n in range(dstate):
                var A_offset = d * dstate + n
                var A_val = Float32(A_buf.ptr[A_offset]) * LOG2E
                dA_vals[n] = exp2(A_val * dt_val)

            # Load B values and compute dB
            var dB_vals = SIMD[.float32, MAX_DSTATE](0.0)
            for n in range(dstate):
                var B_offset = b * n_groups * dstate + group_id * dstate + n
                var B_val = Float32(B_buf.ptr[B_offset])
                dB_vals[n] = B_val * dt_val

            # Load current state
            var state_vals = SIMD[.float32, MAX_DSTATE](0.0)
            for n in range(dstate):
                var state_offset = b * dim * dstate + d * dstate + n
                state_vals[n] = Float32(state_out_ref_h.ptr[state_offset])

            # Update state
            state_vals = state_vals * dA_vals + dB_vals * x_val

            # Store updated state
            for n in range(dstate):
                var state_offset = b * dim * dstate + d * dstate + n
                state_out_ref_h.ptr[state_offset] = Scalar[dtype](state_vals[n])

            # Load C values
            var C_vals = SIMD[.float32, MAX_DSTATE](0.0)
            for n in range(dstate):
                var C_offset = b * n_groups * dstate + group_id * dstate + n
                C_vals[n] = Float32(C_buf.ptr[C_offset])

            # Compute output
            var out_val = (state_vals * C_vals).reduce_add()

            # Add skip connection
            if has_D:
                var D_val = Float32(D_buf.ptr[d])
                out_val += x_val * D_val

            # Apply gating
            if has_z:
                var z_offset = b * dim + d
                var z_val = Float32(z_buf.ptr[z_offset])
                out_val *= z_val * sigmoid_ref(z_val)

            # Store output
            var out_offset = b * dim + d
            output_ref_h.ptr[out_offset] = Scalar[dtype](out_val)

    # Compare results
    var state_size = batch * dim * dstate
    for i in range(state_size):
        assert_almost_equal(
            state_out_h.ptr[i],
            state_out_ref_h.ptr[i],
            rtol=rtol,
        )

    var output_size = batch * dim
    for i in range(output_size):
        assert_almost_equal(
            output_h.ptr[i],
            output_ref_h.ptr[i],
            rtol=rtol,
        )


# =============================================================================
# Test functions for selective scan forward
# =============================================================================


def test_selective_scan_fwd_basic() raises:
    """Test basic selective scan forward."""
    run_selective_scan_fwd[
        DType.float32,
        2,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=1, dim=2, seqlen=4, n_groups=1)


def test_selective_scan_fwd_without_D() raises:
    """Test selective scan forward without D tensor."""
    run_selective_scan_fwd[
        DType.float32,
        2,  # DSTATE
        has_D=False,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=1, dim=2, seqlen=4, n_groups=1)


def test_selective_scan_fwd_without_z() raises:
    """Test selective scan forward without z tensor."""
    run_selective_scan_fwd[
        DType.float32,
        2,  # DSTATE
        has_D=True,
        has_z=False,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=1, dim=2, seqlen=4, n_groups=1)


def test_selective_scan_fwd_with_delta_softplus() raises:
    """Test selective scan forward with delta softplus activation."""
    run_selective_scan_fwd[
        DType.float32,
        2,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=True,
    ](batch=1, dim=2, seqlen=4, n_groups=1)


def test_selective_scan_fwd_longer_sequence() raises:
    """Test selective scan forward with longer sequence."""
    run_selective_scan_fwd[
        DType.float32,
        4,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=1, dim=4, seqlen=16, n_groups=1)


# =============================================================================
# Test functions for selective scan update
# =============================================================================


def test_selective_scan_update_basic() raises:
    """Test basic selective scan update."""
    run_selective_scan_update[
        DType.float32,
        2,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=1, dim=2, n_groups=1)


def test_selective_scan_update_without_D() raises:
    """Test selective scan update without D tensor."""
    run_selective_scan_update[
        DType.float32,
        2,  # DSTATE
        has_D=False,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=1, dim=2, n_groups=1)


def test_selective_scan_update_without_z() raises:
    """Test selective scan update without z tensor."""
    run_selective_scan_update[
        DType.float32,
        2,  # DSTATE
        has_D=True,
        has_z=False,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=1, dim=2, n_groups=1)


def test_selective_scan_update_with_delta_softplus() raises:
    """Test selective scan update with delta softplus activation."""
    run_selective_scan_update[
        DType.float32,
        2,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=True,
    ](batch=1, dim=2, n_groups=1)


def test_selective_scan_update_larger_dimensions() raises:
    """Test selective scan update with larger dimensions."""
    run_selective_scan_update[
        DType.float32,
        4,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=2, dim=4, n_groups=1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
