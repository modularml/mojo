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

from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout._fillers import random
from std.math import exp, exp2, log
from state_space.selective_scan import (
    ssd_combined_cpu,
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


def run_ssd_combined[
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
    """Test SSD combined kernel against reference implementation."""
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
    var output_h = TileTensor(output_heap, row_major(batch, dim, seqlen))

    # x: (batch, dim, num_chunks, 2*dstate) - checkpoint tensor
    var x_heap = List(
        length=batch * dim * n_chunks * 2 * dstate, fill=Scalar[dtype](0)
    )
    var x_h = TileTensor(x_heap, row_major(batch, dim, n_chunks, 2 * dstate))

    # out_z: (batch, dim, seqlen) - gated output
    var out_z_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))
    var out_z_h = TileTensor(out_z_heap, row_major(batch, dim, seqlen))

    # residual: (batch, dim, seqlen)
    var residual_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))
    var residual_h = TileTensor(residual_heap, row_major(batch, dim, seqlen))

    # u: (batch, dim, seqlen)
    var u_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))
    var u_h = TileTensor(u_heap, row_major(batch, dim, seqlen))

    # delta: (batch, dim, seqlen)
    var delta_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))
    var delta_h = TileTensor(delta_heap, row_major(batch, dim, seqlen))

    # A: (dim, dstate)
    var A_heap = List(length=dim * dstate, fill=Scalar[dtype](0))
    var A_h = TileTensor(A_heap, row_major(dim, dstate))

    # B: (batch, n_groups, dstate, seqlen)
    var B_heap = List(
        length=batch * n_groups * dstate * seqlen, fill=Scalar[dtype](0)
    )
    var B_h = TileTensor(B_heap, row_major(batch, n_groups, dstate, seqlen))

    # C: (batch, n_groups, dstate, seqlen)
    var C_heap = List(
        length=batch * n_groups * dstate * seqlen, fill=Scalar[dtype](0)
    )
    var C_h = TileTensor(C_heap, row_major(batch, n_groups, dstate, seqlen))

    # D: (dim,) - optional
    var D_size = dim if has_D else 0
    var D_heap = List(length=max(D_size, 1), fill=Scalar[dtype](0))
    var D_h = TileTensor(D_heap, row_major(D_size))

    # z: (batch, dim, seqlen) - optional
    var z_size = batch * dim * seqlen if has_z else 0
    var z_heap = List(length=max(z_size, 1), fill=Scalar[dtype](0))
    var z_h = TileTensor(
        z_heap,
        row_major(
            (
                batch if has_z else 0,
                dim if has_z else 0,
                seqlen if has_z else 0,
            )
        ),
    )

    # delta_bias: (dim,) - optional
    var delta_bias_size = dim if has_delta_bias else 0
    var delta_bias_heap = List(
        length=max(delta_bias_size, 1), fill=Scalar[dtype](0)
    )
    var delta_bias_h = TileTensor(delta_bias_heap, row_major(delta_bias_size))

    # gamma: (dim,) - for normalization
    var gamma_heap = List(length=dim, fill=Scalar[dtype](0))
    var gamma_h = TileTensor(gamma_heap, row_major(dim))

    # Initialize data
    random(u_h)
    random(delta_h)
    random(residual_h)
    random(A_h)
    random(B_h)
    random(C_h)
    if has_D:
        random(D_h)
    if has_z:
        random(z_h)
    if has_delta_bias:
        random(delta_bias_h)
    random(gamma_h)

    # Initialize gamma to positive values
    for i in range(dim):
        gamma_h._storage[i] = abs(gamma_h._storage[i]) + Scalar[dtype](0.1)

    var epsilon = Float32(0.001)
    var weight_offset = Scalar[dtype](0.0)

    # Call kernel
    ssd_combined_cpu[
        dtype,
        DSTATE,
        output_h.LayoutType,
        x_h.LayoutType,
        out_z_h.LayoutType,
        residual_h.LayoutType,
        u_h.LayoutType,
        delta_h.LayoutType,
        A_h.LayoutType,
        B_h.LayoutType,
        C_h.LayoutType,
        D_h.LayoutType,
        z_h.LayoutType,
        delta_bias_h.LayoutType,
        gamma_h.LayoutType,
    ](
        batch,
        dim,
        seqlen,
        group_size,
        Int8(1) if delta_softplus else Int8(0),
        output_h,
        x_h,
        out_z_h,
        residual_h,
        u_h,
        delta_h,
        A_h,
        B_h,
        C_h,
        D_h,
        z_h,
        delta_bias_h,
        gamma_h,
        epsilon.cast[dtype](),
        weight_offset,
    )

    # Reference implementation for numerical verification
    var ref_size = batch * dim * seqlen
    var output_ref_heap = List(length=ref_size, fill=Scalar[dtype](0))
    var out_z_ref_heap = List(length=ref_size, fill=Scalar[dtype](0))

    for b_idx in range(batch):
        for d_idx in range(dim):
            var group_id = d_idx // group_size

            # Pre-load A values with LOG2E scaling (matches kernel)
            var A_ref = SIMD[.float32, MAX_DSTATE](0.0)
            for n in range(dstate):
                A_ref[n] = Float32(A_h._storage[d_idx * dstate + n]) * LOG2E

            # Load per-dim scalars
            var gamma_val = Float32(gamma_h._storage[d_idx])
            var D_val = Float32(0)
            if has_D:
                D_val = Float32(D_h._storage[d_idx])
            var delta_bias_val = Float32(0)
            if has_delta_bias:
                delta_bias_val = Float32(delta_bias_h._storage[d_idx])
            var weight_offset_val = Float32(weight_offset)

            # Initialize state to zero
            var state_ref = SIMD[.float32, MAX_DSTATE](0.0)

            for t in range(seqlen):
                var off_3d = b_idx * dim * seqlen + d_idx * seqlen + t
                var u_val = Float32(u_h._storage[off_3d])
                var delta_val = Float32(delta_h._storage[off_3d])
                var residual_val = Float32(residual_h._storage[off_3d])

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
                    B_vals[n] = Float32(B_h._storage[bc_offset])
                    C_vals[n] = Float32(C_h._storage[bc_offset])

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
                    var z_val = Float32(z_h._storage[off_3d])
                    var out_z_val = normalized * silu_ref(z_val)
                    out_z_ref_heap[off_3d] = Scalar[dtype](out_z_val)
                    normalized = out_z_val

                output_ref_heap[off_3d] = Scalar[dtype](normalized)

    # Compare kernel output vs reference
    for i in range(ref_size):
        assert_almost_equal(
            output_h._storage[i],
            output_ref_heap[i],
            rtol=rtol,
        )

    # Verify out_z when z gating is enabled
    if has_z:
        for i in range(ref_size):
            assert_almost_equal(
                out_z_h._storage[i],
                out_z_ref_heap[i],
                rtol=rtol,
            )


def test_ssd_combined_basic() raises:
    """Test basic ssd_combined."""
    run_ssd_combined[
        DType.float32,
        4,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=2, dim=4, seqlen=8, n_groups=1)


def test_ssd_combined_without_D() raises:
    """Test ssd_combined without D."""
    run_ssd_combined[
        DType.float32,
        4,  # DSTATE
        has_D=False,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=2, dim=4, seqlen=8, n_groups=1)


def test_ssd_combined_without_z() raises:
    """Test ssd_combined without z."""
    run_ssd_combined[
        DType.float32,
        4,  # DSTATE
        has_D=True,
        has_z=False,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=2, dim=4, seqlen=8, n_groups=1)


def test_ssd_combined_without_delta_bias() raises:
    """Test ssd_combined without delta_bias."""
    run_ssd_combined[
        DType.float32,
        4,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=False,
        delta_softplus=False,
    ](batch=2, dim=4, seqlen=8, n_groups=1)


def test_ssd_combined_with_delta_softplus() raises:
    """Test ssd_combined with delta_softplus."""
    run_ssd_combined[
        DType.float32,
        4,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=True,
    ](batch=2, dim=4, seqlen=8, n_groups=1)


def test_ssd_combined_larger_shapes() raises:
    """Test ssd_combined with larger shapes."""
    run_ssd_combined[
        DType.float32,
        8,  # DSTATE
        has_D=True,
        has_z=True,
        has_delta_bias=True,
        delta_softplus=False,
    ](batch=4, dim=8, seqlen=16, n_groups=1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
