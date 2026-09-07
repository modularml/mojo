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

"""Tests for the Mamba-2 SSD varlen prefill scan.

Mirrors `test_varlen_selective_scan.mojo`: GPU kernel vs CPU reference (the
trusted reference for the per-token recurrence). Covers Mamba-2 grouping
(scalar A, grouped B/C, per-head dt+bias softplus), varlen ragged batches,
final-state output, and (crucially) varlen no-cross-sequence-bleed equivalence:
a packed ragged batch must equal independent per-sequence runs.
"""

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from std.random import rand
from state_space.mamba2_ssd_scan import (
    mamba2_ssd_chunk_scan_varlen_fwd_cpu,
    mamba2_ssd_chunk_scan_varlen_fwd_gpu,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_cpu,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_apple,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_dstate_split,
)
from max.gpu.host.info import B200
from std.testing import TestSuite, assert_almost_equal
from std.utils.index import Index, IndexList


def run_mamba2_ssd_fwd_gpu_vs_cpu[
    dtype: DType,
    DSTATE: Int,
    has_D: Bool = True,
    has_dt_bias: Bool = True,
    dt_softplus: Bool = True,
](
    nheads: Int,
    head_dim: Int,
    ngroups: Int,
    seq_lengths: IndexList,
    ctx: DeviceContext,
    rtol: Float64 = 0.02,
) raises:
    """Run the GPU kernel and CPU reference on identical inputs; assert match.
    """
    comptime dstate = DSTATE
    var batch = len(seq_lengths)
    var nheads_ngroups_ratio = nheads // ngroups

    var total_len = 0
    for i in range(batch):
        total_len += seq_lengths[i]

    # Host allocations.
    var x_h = alloc[Scalar[dtype]](total_len * nheads * head_dim)
    var dt_h = alloc[Scalar[dtype]](total_len * nheads)
    var A_h = alloc[Scalar[dtype]](nheads)
    var B_h = alloc[Scalar[dtype]](total_len * ngroups * dstate)
    var C_h = alloc[Scalar[dtype]](total_len * ngroups * dstate)
    var D_size = nheads if has_D else 0
    var D_h = alloc[Scalar[dtype]](max(D_size, 1))
    var dt_bias_size = nheads if has_dt_bias else 0
    var dt_bias_h = alloc[Scalar[dtype]](max(dt_bias_size, 1))
    var initial_states_h = alloc[Float32](1)  # empty (no init)
    var query_start_loc_h = alloc[Int32](batch + 1)
    var has_initial_state_h = alloc[Scalar[.bool]](1)  # empty

    var y_cpu_h = alloc[Scalar[dtype]](total_len * nheads * head_dim)
    var y_gpu_h = alloc[Scalar[dtype]](total_len * nheads * head_dim)
    var fs_cpu_h = alloc[Float32](batch * nheads * head_dim * dstate)
    var fs_gpu_h = alloc[Float32](batch * nheads * head_dim * dstate)

    # Initialise random inputs (uniform [0, 1)).
    rand(x_h, total_len * nheads * head_dim)
    rand(dt_h, total_len * nheads)
    rand(A_h, nheads)
    rand(B_h, total_len * ngroups * dstate)
    rand(C_h, total_len * ngroups * dstate)
    if has_D:
        rand(D_h, D_size)
    if has_dt_bias:
        rand(dt_bias_h, dt_bias_size)

    # A must be negative (A = -exp(A_log) in the reference). Scale to a stable
    # range so the recurrence does not blow up over the sequence.
    for i in range(nheads):
        A_h.store(i, Scalar[dtype](Float32(A_h.load(i)) * -1.0 - 0.1))

    # dt centred so softplus(dt + bias) is a reasonable positive step.
    for i in range(total_len * nheads):
        dt_h.store(i, Scalar[dtype](Float32(dt_h.load(i)) - 0.5))

    # query_start_loc cumulative.
    var cum = 0
    query_start_loc_h.store(0, Int32(0))
    for i in range(batch):
        cum += seq_lengths[i]
        query_start_loc_h.store(i + 1, Int32(cum))

    # TileTensors for CPU.
    var x_tt = TileTensor(x_h, row_major(total_len, nheads, head_dim))
    var dt_tt = TileTensor(dt_h, row_major(total_len, nheads))
    var A_tt = TileTensor(A_h, row_major(nheads))
    var B_tt = TileTensor(B_h, row_major(total_len, ngroups, dstate))
    var C_tt = TileTensor(C_h, row_major(total_len, ngroups, dstate))
    var D_tt = TileTensor(D_h, row_major(D_size))
    var dt_bias_tt = TileTensor(dt_bias_h, row_major(dt_bias_size))
    var initial_states_tt = TileTensor(initial_states_h, row_major(0, 0, 0, 0))
    var qsl_tt = TileTensor(query_start_loc_h, row_major(batch + 1))
    var his_tt = TileTensor(has_initial_state_h, row_major(0))
    var y_cpu_tt = TileTensor(y_cpu_h, row_major(total_len, nheads, head_dim))
    var fs_cpu_tt = TileTensor(
        fs_cpu_h, row_major(batch, nheads, head_dim, dstate)
    )

    var x_strides = IndexList[3](nheads * head_dim, head_dim, 1)
    var dt_strides = IndexList[2](nheads, 1)
    var A_strides = IndexList[1](1)
    var B_strides = IndexList[3](ngroups * dstate, dstate, 1)
    var C_strides = IndexList[3](ngroups * dstate, dstate, 1)
    var D_strides = IndexList[1](1)
    var dt_bias_strides = IndexList[1](1)
    var is_strides = IndexList[4](
        nheads * head_dim * dstate, head_dim * dstate, dstate, 1
    )
    var y_strides = IndexList[3](nheads * head_dim, head_dim, 1)
    var fs_strides = IndexList[4](
        nheads * head_dim * dstate, head_dim * dstate, dstate, 1
    )

    comptime dt_sp_int8 = Int8(1) if dt_softplus else Int8(0)

    # CPU reference.
    mamba2_ssd_chunk_scan_varlen_fwd_cpu[dtype, DSTATE](
        nheads,
        head_dim,
        ngroups,
        nheads_ngroups_ratio,
        batch,
        dt_sp_int8,
        x_tt,
        dt_tt,
        A_tt,
        B_tt,
        C_tt,
        D_tt,
        dt_bias_tt,
        initial_states_tt,
        y_cpu_tt,
        fs_cpu_tt,
        qsl_tt,
        his_tt,
        x_strides,
        dt_strides,
        A_strides,
        B_strides,
        C_strides,
        D_strides,
        dt_bias_strides,
        is_strides,
        y_strides,
        fs_strides,
    )

    # Device allocations + copies.
    var x_d = ctx.enqueue_create_buffer[dtype](total_len * nheads * head_dim)
    var dt_d = ctx.enqueue_create_buffer[dtype](total_len * nheads)
    var A_d = ctx.enqueue_create_buffer[dtype](nheads)
    var B_d = ctx.enqueue_create_buffer[dtype](total_len * ngroups * dstate)
    var C_d = ctx.enqueue_create_buffer[dtype](total_len * ngroups * dstate)
    var D_d = ctx.enqueue_create_buffer[dtype](max(D_size, 1))
    var dt_bias_d = ctx.enqueue_create_buffer[dtype](max(dt_bias_size, 1))
    var is_d = ctx.enqueue_create_buffer[.float32](1)
    var qsl_d = ctx.enqueue_create_buffer[.int32](batch + 1)
    var his_d = ctx.enqueue_create_buffer[.bool](1)
    var y_d = ctx.enqueue_create_buffer[dtype](total_len * nheads * head_dim)
    var fs_d = ctx.enqueue_create_buffer[.float32](
        batch * nheads * head_dim * dstate
    )

    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(dt_d, dt_h)
    ctx.enqueue_copy(A_d, A_h)
    ctx.enqueue_copy(B_d, B_h)
    ctx.enqueue_copy(C_d, C_h)
    if has_D:
        ctx.enqueue_copy(D_d, D_h)
    if has_dt_bias:
        ctx.enqueue_copy(dt_bias_d, dt_bias_h)
    ctx.enqueue_copy(qsl_d, query_start_loc_h)

    var x_gtt = TileTensor(x_d, row_major(total_len, nheads, head_dim))
    var dt_gtt = TileTensor(dt_d, row_major(total_len, nheads))
    var A_gtt = TileTensor(A_d, row_major(nheads))
    var B_gtt = TileTensor(B_d, row_major(total_len, ngroups, dstate))
    var C_gtt = TileTensor(C_d, row_major(total_len, ngroups, dstate))
    var D_gtt = TileTensor(D_d, row_major(D_size))
    var dt_bias_gtt = TileTensor(dt_bias_d, row_major(dt_bias_size))
    var is_gtt = TileTensor(is_d, row_major(0, 0, 0, 0))
    var qsl_gtt = TileTensor(qsl_d, row_major(batch + 1))
    var his_gtt = TileTensor(his_d, row_major(0))
    var y_gtt = TileTensor(y_d, row_major(total_len, nheads, head_dim))
    var fs_gtt = TileTensor(fs_d, row_major(batch, nheads, head_dim, dstate))

    comptime BLOCK_SIZE = 64
    var num_p_blocks = (head_dim + BLOCK_SIZE - 1) // BLOCK_SIZE

    var compiled = ctx.compile_function[
        mamba2_ssd_chunk_scan_varlen_fwd_gpu[
            dtype,
            DSTATE,
            x_gtt.LayoutType,
            dt_gtt.LayoutType,
            A_gtt.LayoutType,
            B_gtt.LayoutType,
            C_gtt.LayoutType,
            D_gtt.LayoutType,
            dt_bias_gtt.LayoutType,
            is_gtt.LayoutType,
            y_gtt.LayoutType,
            fs_gtt.LayoutType,
            qsl_gtt.LayoutType,
            his_gtt.LayoutType,
            x_gtt.Engine,
        ]
    ]()
    ctx.enqueue_function(
        compiled,
        Int32(nheads),
        Int32(head_dim),
        Int32(ngroups),
        Int32(nheads_ngroups_ratio),
        Int32(batch),
        dt_sp_int8,
        x_gtt,
        dt_gtt,
        A_gtt,
        B_gtt,
        C_gtt,
        D_gtt,
        dt_bias_gtt,
        is_gtt,
        y_gtt,
        fs_gtt,
        qsl_gtt,
        his_gtt,
        x_strides,
        dt_strides,
        A_strides,
        B_strides,
        C_strides,
        D_strides,
        dt_bias_strides,
        is_strides,
        y_strides,
        fs_strides,
        grid_dim=(num_p_blocks, nheads, batch),
        block_dim=(BLOCK_SIZE, 1, 1),
    )

    ctx.enqueue_copy(y_gpu_h, y_d)
    ctx.enqueue_copy(fs_gpu_h, fs_d)
    ctx.synchronize()

    # Compare y.
    for i in range(total_len * nheads * head_dim):
        assert_almost_equal(
            Float32(y_cpu_h.load(i)), Float32(y_gpu_h.load(i)), rtol=rtol
        )
    # Compare final_states (fp32).
    for i in range(batch * nheads * head_dim * dstate):
        assert_almost_equal(fs_cpu_h.load(i), fs_gpu_h.load(i), rtol=rtol)

    x_h.free()
    dt_h.free()
    A_h.free()
    B_h.free()
    C_h.free()
    D_h.free()
    dt_bias_h.free()
    initial_states_h.free()
    query_start_loc_h.free()
    has_initial_state_h.free()
    y_cpu_h.free()
    y_gpu_h.free()
    fs_cpu_h.free()
    fs_gpu_h.free()


def run_varlen_no_bleed_cpu[
    dtype: DType,
    DSTATE: Int,
](nheads: Int, head_dim: Int, ngroups: Int, seq_lengths: IndexList,) raises:
    """Varlen correctness: a packed ragged batch must equal independent
    per-sequence runs (no cross-sequence state bleed). Uses the CPU reference
    so this is a pure algorithmic-invariant check independent of GPU launch.
    """
    comptime dstate = DSTATE
    var batch = len(seq_lengths)
    var nheads_ngroups_ratio = nheads // ngroups

    var total_len = 0
    for i in range(batch):
        total_len += seq_lengths[i]

    var x_h = alloc[Scalar[dtype]](total_len * nheads * head_dim)
    var dt_h = alloc[Scalar[dtype]](total_len * nheads)
    var A_h = alloc[Scalar[dtype]](nheads)
    var B_h = alloc[Scalar[dtype]](total_len * ngroups * dstate)
    var C_h = alloc[Scalar[dtype]](total_len * ngroups * dstate)
    var D_h = alloc[Scalar[dtype]](nheads)
    var dt_bias_h = alloc[Scalar[dtype]](nheads)
    var is_h = alloc[Float32](1)
    var his_h = alloc[Scalar[.bool]](1)

    rand(x_h, total_len * nheads * head_dim)
    rand(dt_h, total_len * nheads)
    rand(A_h, nheads)
    rand(B_h, total_len * ngroups * dstate)
    rand(C_h, total_len * ngroups * dstate)
    rand(D_h, nheads)
    rand(dt_bias_h, nheads)
    for i in range(nheads):
        A_h.store(i, Scalar[dtype](Float32(A_h.load(i)) * -1.0 - 0.1))
    for i in range(total_len * nheads):
        dt_h.store(i, Scalar[dtype](Float32(dt_h.load(i)) - 0.5))

    var x_strides = IndexList[3](nheads * head_dim, head_dim, 1)
    var dt_strides = IndexList[2](nheads, 1)
    var A_strides = IndexList[1](1)
    var B_strides = IndexList[3](ngroups * dstate, dstate, 1)
    var C_strides = IndexList[3](ngroups * dstate, dstate, 1)
    var D_strides = IndexList[1](1)
    var dt_bias_strides = IndexList[1](1)
    var is_strides = IndexList[4](0, 0, 0, 0)

    var A_tt = TileTensor(A_h, row_major(nheads))
    var D_tt = TileTensor(D_h, row_major(nheads))
    var dt_bias_tt = TileTensor(dt_bias_h, row_major(nheads))
    var is_tt = TileTensor(is_h, row_major(0, 0, 0, 0))
    var his_tt = TileTensor(his_h, row_major(0))

    # ---- Packed run over the whole ragged batch ----
    var y_packed = alloc[Scalar[dtype]](total_len * nheads * head_dim)
    var fs_packed = alloc[Float32](batch * nheads * head_dim * dstate)
    var qsl_h = alloc[Int32](batch + 1)
    var cum = 0
    qsl_h.store(0, Int32(0))
    for i in range(batch):
        cum += seq_lengths[i]
        qsl_h.store(i + 1, Int32(cum))

    var x_tt = TileTensor(x_h, row_major(total_len, nheads, head_dim))
    var dt_tt = TileTensor(dt_h, row_major(total_len, nheads))
    var B_tt = TileTensor(B_h, row_major(total_len, ngroups, dstate))
    var C_tt = TileTensor(C_h, row_major(total_len, ngroups, dstate))
    var y_packed_tt = TileTensor(
        y_packed, row_major(total_len, nheads, head_dim)
    )
    var fs_packed_tt = TileTensor(
        fs_packed, row_major(batch, nheads, head_dim, dstate)
    )
    var qsl_tt = TileTensor(qsl_h, row_major(batch + 1))
    var y_strides = IndexList[3](nheads * head_dim, head_dim, 1)
    var fs_strides = IndexList[4](
        nheads * head_dim * dstate, head_dim * dstate, dstate, 1
    )

    mamba2_ssd_chunk_scan_varlen_fwd_cpu[dtype, DSTATE](
        nheads,
        head_dim,
        ngroups,
        nheads_ngroups_ratio,
        batch,
        Int8(1),
        x_tt,
        dt_tt,
        A_tt,
        B_tt,
        C_tt,
        D_tt,
        dt_bias_tt,
        is_tt,
        y_packed_tt,
        fs_packed_tt,
        qsl_tt,
        his_tt,
        x_strides,
        dt_strides,
        A_strides,
        B_strides,
        C_strides,
        D_strides,
        dt_bias_strides,
        is_strides,
        y_strides,
        fs_strides,
    )

    # ---- Independent per-sequence runs ----
    var y_indep = alloc[Scalar[dtype]](total_len * nheads * head_dim)
    var fs_indep = alloc[Float32](batch * nheads * head_dim * dstate)
    var off = 0
    for s in range(batch):
        var slen = seq_lengths[s]
        # single-sequence views (offsets into the packed buffers)
        var x_s = TileTensor(
            x_h + off * nheads * head_dim,
            row_major(slen, nheads, head_dim),
        )
        var dt_s = TileTensor(dt_h + off * nheads, row_major(slen, nheads))
        var B_s = TileTensor(
            B_h + off * ngroups * dstate, row_major(slen, ngroups, dstate)
        )
        var C_s = TileTensor(
            C_h + off * ngroups * dstate, row_major(slen, ngroups, dstate)
        )
        var y_s = TileTensor(
            y_indep + off * nheads * head_dim,
            row_major(slen, nheads, head_dim),
        )
        var fs_s = TileTensor(
            fs_indep + s * nheads * head_dim * dstate,
            row_major(1, nheads, head_dim, dstate),
        )
        var qsl_s_h = alloc[Int32](2)
        qsl_s_h.store(0, Int32(0))
        qsl_s_h.store(1, Int32(slen))
        var qsl_s = TileTensor(qsl_s_h, row_major(2))

        mamba2_ssd_chunk_scan_varlen_fwd_cpu[dtype, DSTATE](
            nheads,
            head_dim,
            ngroups,
            nheads_ngroups_ratio,
            1,
            Int8(1),
            x_s,
            dt_s,
            A_tt,
            B_s,
            C_s,
            D_tt,
            dt_bias_tt,
            is_tt,
            y_s,
            fs_s,
            qsl_s,
            his_tt,
            x_strides,
            dt_strides,
            A_strides,
            B_strides,
            C_strides,
            D_strides,
            dt_bias_strides,
            is_strides,
            y_strides,
            fs_strides,
        )
        qsl_s_h.free()
        off += slen

    # Packed must equal per-sequence (bit-identical: same fp32 ops, same order).
    for i in range(total_len * nheads * head_dim):
        assert_almost_equal(
            Float32(y_packed.load(i)), Float32(y_indep.load(i)), rtol=1e-5
        )
    for i in range(batch * nheads * head_dim * dstate):
        assert_almost_equal(fs_packed.load(i), fs_indep.load(i), rtol=1e-5)

    x_h.free()
    dt_h.free()
    A_h.free()
    B_h.free()
    C_h.free()
    D_h.free()
    dt_bias_h.free()
    is_h.free()
    his_h.free()
    y_packed.free()
    fs_packed.free()
    qsl_h.free()
    y_indep.free()
    fs_indep.free()


# =============================================================================
# Test entry points
# =============================================================================


def test_mamba2_ssd_small_parity_shape() raises:
    """Spec parity shape: nheads=8, head_dim=16, ngroups=2, dstate=16, 2-seq."""
    with DeviceContext() as ctx:
        if not ctx.is_compatible():
            return
        run_mamba2_ssd_fwd_gpu_vs_cpu[.bfloat16, 16](
            nheads=8,
            head_dim=16,
            ngroups=2,
            seq_lengths=Index(8, 5),
            ctx=ctx,
        )


def test_mamba2_ssd_fp32_exact() raises:
    """FP32 path, single sequence (tighter tolerance)."""
    with DeviceContext() as ctx:
        if not ctx.is_compatible():
            return
        run_mamba2_ssd_fwd_gpu_vs_cpu[.float32, 16](
            nheads=4,
            head_dim=8,
            ngroups=2,
            seq_lengths=Index(12),
            ctx=ctx,
            rtol=1e-4,
        )


def test_mamba2_ssd_variable_lengths() raises:
    """Ragged 3-sequence batch."""
    with DeviceContext() as ctx:
        if not ctx.is_compatible():
            return
        run_mamba2_ssd_fwd_gpu_vs_cpu[.bfloat16, 64](
            nheads=12,
            head_dim=16,
            ngroups=4,
            seq_lengths=Index(10, 6, 1),
            ctx=ctx,
        )


def test_mamba2_ssd_production_grouping() raises:
    """Production Nemotron-H grouping: 96 heads / 8 groups (12 heads/group),
    head_dim 80, dstate 128. Small seqlens to keep the test fast."""
    with DeviceContext() as ctx:
        if not ctx.is_compatible():
            return
        run_mamba2_ssd_fwd_gpu_vs_cpu[.bfloat16, 128](
            nheads=96,
            head_dim=80,
            ngroups=8,
            seq_lengths=Index(4, 3),
            ctx=ctx,
        )


def test_mamba2_ssd_no_D_no_bias() raises:
    """Optional D and dt_bias omitted."""
    with DeviceContext() as ctx:
        if not ctx.is_compatible():
            return
        run_mamba2_ssd_fwd_gpu_vs_cpu[
            DType.bfloat16, 16, has_D=False, has_dt_bias=False
        ](
            nheads=8,
            head_dim=16,
            ngroups=2,
            seq_lengths=Index(7, 7),
            ctx=ctx,
        )


def test_mamba2_ssd_varlen_no_cross_sequence_bleed() raises:
    """Packed ragged batch == per-sequence runs (CPU-reference invariant)."""
    run_varlen_no_bleed_cpu[.float32, 16](
        nheads=8,
        head_dim=16,
        ngroups=2,
        seq_lengths=Index(9, 4, 6),
    )


def run_mamba2_ssd_inplace_vs_functional[
    dtype: DType,
    DSTATE: Int,
    DSTATE_SPLIT: Int = 0,
    use_apple: Bool = False,
    # SSM-pool STORAGE dtype for the GPU run (the CPU references stay fp32).
    # Only the Apple kernel implements non-fp32 state.
    state_dtype: DType = .float32,
    # Seed the pool with (state_dtype-representable) random initial states and
    # set has_initial_state, exercising the state LOAD path — with a zero pool
    # and has_initial_state=False the initial-state load is dead code.
    init_state: Bool = False,
](
    nheads: Int,
    head_dim: Int,
    ngroups: Int,
    max_slots: Int,
    seq_lengths: IndexList,
    ctx: DeviceContext,
    rtol: Float64 = 0.02,
) raises:
    """Verify the inplace variant matches the functional variant.

    * ``y`` output must be numerically identical (same recurrence, just a
      different write-back path for final states).
    * After the inplace run, ``ssm_pool[slot, ...]`` at the used slots must
      match the ``final_states`` produced by the functional variant.

    ``DSTATE_SPLIT == 0`` runs the portable v1 one-thread-per-channel kernel
    (``mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu``) with its ``(64, 1, 1)``
    launch. ``DSTATE_SPLIT > 0`` runs the B200 cooperative-split kernel
    (``..._gpu_dstate_split``) with the production ``(DSTATE_SPLIT,
    CH_PER_BLOCK, 1)`` launch; the split kernel's output must equal the v1 /
    functional reference. The split path is only invoked from a B200-gated
    caller because its full-warp shuffle assumes warp width 32.

    ``state_dtype`` sets the GPU pool's storage dtype (the CPU references stay
    fp32); ``init_state`` seeds the pool and enables has_initial_state so the
    state LOAD path runs. See ``test_..._apple_bf16_state_vs_functional``.
    """
    comptime assert (
        state_dtype == .float32 or use_apple
    ), "non-fp32 SSM state is only implemented by the Apple kernel"
    comptime dstate = DSTATE
    var batch = len(seq_lengths)
    var nheads_ngroups_ratio = nheads // ngroups

    var total_len = 0
    for i in range(batch):
        total_len += seq_lengths[i]

    # Allocate host inputs shared by both variants.
    var x_h = alloc[Scalar[dtype]](total_len * nheads * head_dim)
    var dt_h = alloc[Scalar[dtype]](total_len * nheads)
    var A_h = alloc[Scalar[dtype]](nheads)
    var B_h = alloc[Scalar[dtype]](total_len * ngroups * dstate)
    var C_h = alloc[Scalar[dtype]](total_len * ngroups * dstate)
    var D_h = alloc[Scalar[dtype]](nheads)
    var dt_bias_h = alloc[Scalar[dtype]](nheads)
    var his_h = alloc[Scalar[.bool]](batch)
    var qsl_h = alloc[Int32](batch + 1)
    # slot_indices: identity mapping (slot b -> b) for simplicity.
    var slot_idx_h = alloc[UInt32](batch)
    # ssm_pool: [max_slots, nheads, head_dim, dstate] — zero-initialised.
    var pool_h = alloc[Float32](max_slots * nheads * head_dim * dstate)

    rand(x_h, total_len * nheads * head_dim)
    rand(dt_h, total_len * nheads)
    rand(A_h, nheads)
    rand(B_h, total_len * ngroups * dstate)
    rand(C_h, total_len * ngroups * dstate)
    rand(D_h, nheads)
    rand(dt_bias_h, nheads)
    for i in range(nheads):
        A_h.store(i, Scalar[dtype](Float32(A_h.load(i)) * -1.0 - 0.1))
    for i in range(total_len * nheads):
        dt_h.store(i, Scalar[dtype](Float32(dt_h.load(i)) - 0.5))
    # has_initial_state: all-False by default (pool starts at zero); all-True
    # under init_state (pool seeded below, state load path exercised).
    for i in range(batch):
        his_h.store(i, Scalar[.bool](init_state))
    # Slot indices: sequence b -> slot b.
    for i in range(batch):
        slot_idx_h.store(i, UInt32(i))
    # Initial states (functional-reference input + pool seed). Values are
    # pre-rounded to state_dtype so the fp32 references and a bf16 GPU pool
    # start from bit-identical states; the only remaining state_dtype effect
    # is then the single final write-back rounding.
    var is_count = (batch * nheads * head_dim * dstate) if init_state else 1
    var is_h = alloc[Float32](is_count)
    comptime if init_state:
        rand(is_h, is_count)
        for i in range(is_count):
            is_h.store(i, is_h.load(i).cast[state_dtype]().cast[.float32]())
    # Zero out pool, then (under init_state) seed slot b — identity slot
    # mapping — with sequence b's initial state so the inplace variants read
    # exactly what the functional reference receives via its initial_states.
    for i in range(max_slots * nheads * head_dim * dstate):
        pool_h.store(i, Float32(0.0))
    comptime if init_state:
        for i in range(batch * nheads * head_dim * dstate):
            pool_h.store(i, is_h.load(i))

    var cum = 0
    qsl_h.store(0, Int32(0))
    for i in range(batch):
        cum += seq_lengths[i]
        qsl_h.store(i + 1, Int32(cum))

    var x_strides = IndexList[3](nheads * head_dim, head_dim, 1)
    var dt_strides = IndexList[2](nheads, 1)
    var A_strides = IndexList[1](1)
    var B_strides = IndexList[3](ngroups * dstate, dstate, 1)
    var C_strides = IndexList[3](ngroups * dstate, dstate, 1)
    var D_strides = IndexList[1](1)
    var dt_bias_strides = IndexList[1](1)
    var y_strides = IndexList[3](nheads * head_dim, head_dim, 1)
    var fs_strides = IndexList[4](
        nheads * head_dim * dstate, head_dim * dstate, dstate, 1
    )
    var pool_strides = IndexList[4](
        nheads * head_dim * dstate, head_dim * dstate, dstate, 1
    )
    var is_strides = IndexList[4](
        nheads * head_dim * dstate, head_dim * dstate, dstate, 1
    )

    # ---- Functional variant (CPU reference) ----
    var y_ref_h = alloc[Scalar[dtype]](total_len * nheads * head_dim)
    var fs_ref_h = alloc[Float32](batch * nheads * head_dim * dstate)

    var x_tt = TileTensor(x_h, row_major(total_len, nheads, head_dim))
    var dt_tt = TileTensor(dt_h, row_major(total_len, nheads))
    var A_tt = TileTensor(A_h, row_major(nheads))
    var B_tt = TileTensor(B_h, row_major(total_len, ngroups, dstate))
    var C_tt = TileTensor(C_h, row_major(total_len, ngroups, dstate))
    var D_tt = TileTensor(D_h, row_major(nheads))
    var dt_bias_tt = TileTensor(dt_bias_h, row_major(nheads))
    # Empty (dim0 == 0) when init_state is False — the kernels key "has an
    # initial state" off the has_initial_state tensor's length.
    var func_is_dim0 = batch if init_state else 0
    var is_tt = TileTensor(
        is_h, row_major(func_is_dim0, nheads, head_dim, dstate)
    )
    var func_his_tt = TileTensor(his_h, row_major(func_is_dim0))
    var qsl_tt = TileTensor(qsl_h, row_major(batch + 1))
    var y_ref_tt = TileTensor(y_ref_h, row_major(total_len, nheads, head_dim))
    var fs_ref_tt = TileTensor(
        fs_ref_h, row_major(batch, nheads, head_dim, dstate)
    )

    mamba2_ssd_chunk_scan_varlen_fwd_cpu[dtype, DSTATE](
        nheads,
        head_dim,
        ngroups,
        nheads_ngroups_ratio,
        batch,
        Int8(1),
        x_tt,
        dt_tt,
        A_tt,
        B_tt,
        C_tt,
        D_tt,
        dt_bias_tt,
        is_tt,
        y_ref_tt,
        fs_ref_tt,
        qsl_tt,
        func_his_tt,
        x_strides,
        dt_strides,
        A_strides,
        B_strides,
        C_strides,
        D_strides,
        dt_bias_strides,
        is_strides,
        y_strides,
        fs_strides,
    )

    # ---- Inplace CPU variant ----
    var y_inplace_h = alloc[Scalar[dtype]](total_len * nheads * head_dim)
    var pool_tt = TileTensor(
        pool_h, row_major(max_slots, nheads, head_dim, dstate)
    )
    var slot_tt = TileTensor(slot_idx_h, row_major(batch))
    var his_tt = TileTensor(his_h, row_major(batch))
    var y_inplace_tt = TileTensor(
        y_inplace_h, row_major(total_len, nheads, head_dim)
    )

    mamba2_ssd_chunk_scan_varlen_fwd_inplace_cpu[dtype, DSTATE](
        nheads,
        head_dim,
        ngroups,
        nheads_ngroups_ratio,
        batch,
        Int8(1),
        x_tt,
        dt_tt,
        A_tt,
        B_tt,
        C_tt,
        D_tt,
        dt_bias_tt,
        y_inplace_tt,
        pool_tt,
        qsl_tt,
        his_tt,
        slot_tt,
        x_strides,
        dt_strides,
        A_strides,
        B_strides,
        C_strides,
        D_strides,
        dt_bias_strides,
        y_strides,
        pool_strides,
    )

    # y output must be numerically identical.
    for i in range(total_len * nheads * head_dim):
        assert_almost_equal(
            Float32(y_ref_h.load(i)),
            Float32(y_inplace_h.load(i)),
            rtol=rtol,
            msg="y mismatch at index " + String(i),
        )

    # After inplace run, pool[slot, ...] must equal fs_ref[batch, ...].
    for b in range(batch):
        var slot = Int(slot_idx_h.load(b))
        for i in range(nheads * head_dim * dstate):
            var pool_off = slot * nheads * head_dim * dstate + i
            var fs_off = b * nheads * head_dim * dstate + i
            assert_almost_equal(
                fs_ref_h.load(fs_off),
                pool_h.load(pool_off),
                rtol=rtol,
                msg="pool/final_states mismatch at batch="
                + String(b)
                + " i="
                + String(i),
            )

    # ---- GPU inplace variant ----
    var x_d = ctx.enqueue_create_buffer[dtype](total_len * nheads * head_dim)
    var dt_d = ctx.enqueue_create_buffer[dtype](total_len * nheads)
    var A_d = ctx.enqueue_create_buffer[dtype](nheads)
    var B_d = ctx.enqueue_create_buffer[dtype](total_len * ngroups * dstate)
    var C_d = ctx.enqueue_create_buffer[dtype](total_len * ngroups * dstate)
    var D_d = ctx.enqueue_create_buffer[dtype](nheads)
    var dt_bias_d = ctx.enqueue_create_buffer[dtype](nheads)
    var qsl_d = ctx.enqueue_create_buffer[.int32](batch + 1)
    var his_d = ctx.enqueue_create_buffer[.bool](batch)
    var slot_d = ctx.enqueue_create_buffer[.uint32](batch)
    var y_gpu_h = alloc[Scalar[dtype]](total_len * nheads * head_dim)
    var y_d = ctx.enqueue_create_buffer[dtype](total_len * nheads * head_dim)
    # GPU pool: state_dtype storage (fp32 default; bf16 exercises the Apple
    # bf16-state path), zero-initialised then seeded exactly like pool_h.
    var pool_d = ctx.enqueue_create_buffer[state_dtype](
        max_slots * nheads * head_dim * dstate
    )
    var init_pool = alloc[Scalar[state_dtype]](
        max_slots * nheads * head_dim * dstate
    )
    for i in range(max_slots * nheads * head_dim * dstate):
        init_pool.store(i, Scalar[state_dtype](0.0))
    comptime if init_state:
        # is_h values are state_dtype-representable, so this cast is exact.
        for i in range(batch * nheads * head_dim * dstate):
            init_pool.store(i, is_h.load(i).cast[state_dtype]())
    ctx.enqueue_copy(pool_d, init_pool)

    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(dt_d, dt_h)
    ctx.enqueue_copy(A_d, A_h)
    ctx.enqueue_copy(B_d, B_h)
    ctx.enqueue_copy(C_d, C_h)
    ctx.enqueue_copy(D_d, D_h)
    ctx.enqueue_copy(dt_bias_d, dt_bias_h)
    ctx.enqueue_copy(qsl_d, qsl_h)
    ctx.enqueue_copy(his_d, his_h)
    ctx.enqueue_copy(slot_d, slot_idx_h)

    var x_gtt = TileTensor(x_d, row_major(total_len, nheads, head_dim))
    var dt_gtt = TileTensor(dt_d, row_major(total_len, nheads))
    var A_gtt = TileTensor(A_d, row_major(nheads))
    var B_gtt = TileTensor(B_d, row_major(total_len, ngroups, dstate))
    var C_gtt = TileTensor(C_d, row_major(total_len, ngroups, dstate))
    var D_gtt = TileTensor(D_d, row_major(nheads))
    var dt_bias_gtt = TileTensor(dt_bias_d, row_major(nheads))
    var qsl_gtt = TileTensor(qsl_d, row_major(batch + 1))
    var his_gtt = TileTensor(his_d, row_major(batch))
    var slot_gtt = TileTensor(slot_d, row_major(batch))
    var y_gtt = TileTensor(y_d, row_major(total_len, nheads, head_dim))
    var pool_gtt = TileTensor(
        pool_d, row_major(max_slots, nheads, head_dim, dstate)
    )

    comptime if DSTATE_SPLIT > 0:
        # Mirror the production B200 split launch (kernels.mojo): DSTATE_SPLIT
        # threads cooperate per channel, CH_PER_BLOCK channels per 128-thread
        # block.
        comptime BLOCK_THREADS = 128
        comptime CH_PER_BLOCK = BLOCK_THREADS // DSTATE_SPLIT
        var num_p_blocks = (head_dim + CH_PER_BLOCK - 1) // CH_PER_BLOCK

        var compiled_split = ctx.compile_function[
            mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_dstate_split[
                dtype,
                DSTATE,
                x_gtt.LayoutType,
                dt_gtt.LayoutType,
                A_gtt.LayoutType,
                B_gtt.LayoutType,
                C_gtt.LayoutType,
                D_gtt.LayoutType,
                dt_bias_gtt.LayoutType,
                y_gtt.LayoutType,
                pool_gtt.LayoutType,
                qsl_gtt.LayoutType,
                his_gtt.LayoutType,
                slot_gtt.LayoutType,
                x_gtt.Engine,
                DSTATE_SPLIT,
            ]
        ]()
        ctx.enqueue_function(
            compiled_split,
            Int32(nheads),
            Int32(head_dim),
            Int32(ngroups),
            Int32(nheads_ngroups_ratio),
            Int32(batch),
            Int8(1),
            x_gtt,
            dt_gtt,
            A_gtt,
            B_gtt,
            C_gtt,
            D_gtt,
            dt_bias_gtt,
            y_gtt,
            pool_gtt,
            qsl_gtt,
            his_gtt,
            slot_gtt,
            x_strides,
            dt_strides,
            A_strides,
            B_strides,
            C_strides,
            D_strides,
            dt_bias_strides,
            y_strides,
            pool_strides,
            grid_dim=(num_p_blocks, nheads, batch),
            block_dim=(DSTATE_SPLIT, CH_PER_BLOCK, 1),
        )
    elif use_apple:
        # Apple-M5 vectorized-I/O kernel (same one-thread-per-channel launch as
        # v1). Portable fp32 SIMD, so it runs and validates on any GPU here.
        comptime BLOCK_SIZE = 64
        var num_p_blocks = (head_dim + BLOCK_SIZE - 1) // BLOCK_SIZE

        var compiled_apple = ctx.compile_function[
            mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_apple[
                dtype,
                DSTATE,
                x_gtt.LayoutType,
                dt_gtt.LayoutType,
                A_gtt.LayoutType,
                B_gtt.LayoutType,
                C_gtt.LayoutType,
                D_gtt.LayoutType,
                dt_bias_gtt.LayoutType,
                y_gtt.LayoutType,
                pool_gtt.LayoutType,
                qsl_gtt.LayoutType,
                his_gtt.LayoutType,
                slot_gtt.LayoutType,
                state_dtype,
                4,
                x_gtt.Engine,
            ]
        ]()
        ctx.enqueue_function(
            compiled_apple,
            Int32(nheads),
            Int32(head_dim),
            Int32(ngroups),
            Int32(nheads_ngroups_ratio),
            Int32(batch),
            Int8(1),
            x_gtt,
            dt_gtt,
            A_gtt,
            B_gtt,
            C_gtt,
            D_gtt,
            dt_bias_gtt,
            y_gtt,
            pool_gtt,
            qsl_gtt,
            his_gtt,
            slot_gtt,
            x_strides,
            dt_strides,
            A_strides,
            B_strides,
            C_strides,
            D_strides,
            dt_bias_strides,
            y_strides,
            pool_strides,
            grid_dim=(num_p_blocks, nheads, batch),
            block_dim=(BLOCK_SIZE, 1, 1),
        )
    else:
        comptime BLOCK_SIZE = 64
        var num_p_blocks = (head_dim + BLOCK_SIZE - 1) // BLOCK_SIZE

        var compiled_inplace = ctx.compile_function[
            mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu[
                dtype,
                DSTATE,
                x_gtt.LayoutType,
                dt_gtt.LayoutType,
                A_gtt.LayoutType,
                B_gtt.LayoutType,
                C_gtt.LayoutType,
                D_gtt.LayoutType,
                dt_bias_gtt.LayoutType,
                y_gtt.LayoutType,
                pool_gtt.LayoutType,
                qsl_gtt.LayoutType,
                his_gtt.LayoutType,
                slot_gtt.LayoutType,
                x_gtt.Engine,
            ]
        ]()
        ctx.enqueue_function(
            compiled_inplace,
            Int32(nheads),
            Int32(head_dim),
            Int32(ngroups),
            Int32(nheads_ngroups_ratio),
            Int32(batch),
            Int8(1),
            x_gtt,
            dt_gtt,
            A_gtt,
            B_gtt,
            C_gtt,
            D_gtt,
            dt_bias_gtt,
            y_gtt,
            pool_gtt,
            qsl_gtt,
            his_gtt,
            slot_gtt,
            x_strides,
            dt_strides,
            A_strides,
            B_strides,
            C_strides,
            D_strides,
            dt_bias_strides,
            y_strides,
            pool_strides,
            grid_dim=(num_p_blocks, nheads, batch),
            block_dim=(BLOCK_SIZE, 1, 1),
        )

    var pool_gpu_h = alloc[Scalar[state_dtype]](
        max_slots * nheads * head_dim * dstate
    )
    ctx.enqueue_copy(y_gpu_h, y_d)
    ctx.enqueue_copy(pool_gpu_h, pool_d)
    ctx.synchronize()

    # GPU y must match CPU functional reference.
    for i in range(total_len * nheads * head_dim):
        assert_almost_equal(
            Float32(y_ref_h.load(i)),
            Float32(y_gpu_h.load(i)),
            rtol=rtol,
            msg="GPU y mismatch at index " + String(i),
        )
    # GPU pool[slot, ...] must match final_states from CPU reference.
    for b in range(batch):
        var slot = Int(slot_idx_h.load(b))
        for i in range(nheads * head_dim * dstate):
            var pool_off = slot * nheads * head_dim * dstate + i
            var fs_off = b * nheads * head_dim * dstate + i
            # For bf16 state the GPU value carries one extra rounding (the
            # final write-back; <= 2^-9 relative), well inside rtol.
            assert_almost_equal(
                fs_ref_h.load(fs_off),
                Float32(pool_gpu_h.load(pool_off)),
                rtol=rtol,
                msg="GPU pool/fs mismatch at batch="
                + String(b)
                + " i="
                + String(i),
            )

    # Free all.
    x_h.free()
    dt_h.free()
    A_h.free()
    B_h.free()
    C_h.free()
    D_h.free()
    dt_bias_h.free()
    his_h.free()
    qsl_h.free()
    slot_idx_h.free()
    pool_h.free()
    is_h.free()
    y_ref_h.free()
    fs_ref_h.free()
    y_inplace_h.free()
    y_gpu_h.free()
    pool_gpu_h.free()
    init_pool.free()


def test_mamba2_ssd_inplace_vs_functional_small() raises:
    """Inplace variant: y and pool state match functional variant (small)."""
    with DeviceContext() as ctx:
        if not ctx.is_compatible():
            return
        run_mamba2_ssd_inplace_vs_functional[.bfloat16, 16](
            nheads=8,
            head_dim=16,
            ngroups=2,
            max_slots=8,
            seq_lengths=Index(6, 4),
            ctx=ctx,
        )


def test_mamba2_ssd_inplace_vs_functional_production() raises:
    """Inplace variant: production Nemotron-H grouping (96 heads, dstate=128).
    """
    with DeviceContext() as ctx:
        if not ctx.is_compatible():
            return
        run_mamba2_ssd_inplace_vs_functional[.bfloat16, 128](
            nheads=96,
            head_dim=80,
            ngroups=8,
            max_slots=4,
            seq_lengths=Index(3, 2),
            ctx=ctx,
        )


def test_mamba2_ssd_inplace_dstate_split_vs_functional() raises:
    """B200 cooperative DSTATE-split kernel: output must equal the functional
    reference across the DSTATE_SPLIT factors that the wrapper may pick.

    Gated to B200 (sm_100) because the split path uses a full-warp shuffle that
    assumes warp width 32; on AMD (wavefront 64) the split kernel is neither
    compiled nor launched, so this test is a no-op there (the v1 correctness
    tests above still cover AMD). Sweeps the production 96h/8g/dstate128 grouping
    plus the exact seqlen-1 decode shape the split targets, over DSTATE_SPLIT in
    {2, 4, 8} (8 = the production wrapper choice; 2/4 exercise other tile widths
    L = DSTATE / DSTATE_SPLIT).
    """
    with DeviceContext() as ctx:
        if not ctx.is_compatible():
            return

        comptime if ctx.default_device_info == B200:
            # Production grouping, ragged batch.
            run_mamba2_ssd_inplace_vs_functional[.bfloat16, 128, 2](
                nheads=96,
                head_dim=80,
                ngroups=8,
                max_slots=4,
                seq_lengths=Index(3, 2),
                ctx=ctx,
            )
            run_mamba2_ssd_inplace_vs_functional[.bfloat16, 128, 4](
                nheads=96,
                head_dim=80,
                ngroups=8,
                max_slots=4,
                seq_lengths=Index(3, 2),
                ctx=ctx,
            )
            run_mamba2_ssd_inplace_vs_functional[.bfloat16, 128, 8](
                nheads=96,
                head_dim=80,
                ngroups=8,
                max_slots=4,
                seq_lengths=Index(3, 2),
                ctx=ctx,
            )
            # Exact decode shape: seqlen-1 sequences (the shape the r7 occupancy
            # lever targets), at DSTATE_SPLIT=8 (the production choice).
            run_mamba2_ssd_inplace_vs_functional[.bfloat16, 128, 8](
                nheads=96,
                head_dim=80,
                ngroups=8,
                max_slots=8,
                seq_lengths=Index(1, 1, 1, 1),
                ctx=ctx,
            )
            # Small grouping / smaller dstate, to exercise DSTATE=16 tiling.
            run_mamba2_ssd_inplace_vs_functional[.bfloat16, 16, 4](
                nheads=8,
                head_dim=16,
                ngroups=2,
                max_slots=8,
                seq_lengths=Index(6, 4),
                ctx=ctx,
            )


def test_mamba2_ssd_inplace_apple_vs_functional() raises:
    """Apple-M5 vectorized-I/O kernel: y and pool state must equal the
    functional reference.

    The kernel is portable fp32 SIMD (no warp-width assumption), so it runs on
    any GPU here for cross-hardware numeric coverage; on Apple (cc==5) it is the
    production decode path wired at ``kernels.mojo``. Sweeps the production
    96h/8g/dstate128 grouping, the seqlen-1 decode shape, and a small DSTATE=16
    grouping (exercises ``NCHUNK = DSTATE / VEC`` chunk tiling at VEC=4).
    """
    with DeviceContext() as ctx:
        if not ctx.is_compatible():
            return
        run_mamba2_ssd_inplace_vs_functional[
            DType.bfloat16, 128, use_apple=True
        ](
            nheads=96,
            head_dim=80,
            ngroups=8,
            max_slots=4,
            seq_lengths=Index(3, 2),
            ctx=ctx,
        )
        # Exact seqlen-1 decode shape (the shape the Apple path targets).
        run_mamba2_ssd_inplace_vs_functional[
            DType.bfloat16, 128, use_apple=True
        ](
            nheads=96,
            head_dim=80,
            ngroups=8,
            max_slots=8,
            seq_lengths=Index(1, 1, 1, 1),
            ctx=ctx,
        )
        # Small grouping / smaller dstate, to exercise DSTATE=16 tiling.
        run_mamba2_ssd_inplace_vs_functional[
            DType.bfloat16, 16, use_apple=True
        ](
            nheads=8,
            head_dim=16,
            ngroups=2,
            max_slots=8,
            seq_lengths=Index(6, 4),
            ctx=ctx,
        )


def test_mamba2_ssd_inplace_apple_bf16_state_vs_functional() raises:
    """Apple kernel with bf16 SSM-state storage vs the fp32 references.

    ``init_state=True`` seeds the pool so BOTH bf16 boundary paths run: the
    initial-state load (bf16 -> fp32 widen) and the final write-back
    (fp32 -> bf16 round). Initial values are pre-rounded to bf16, so the only
    state_dtype effect vs the fp32 reference is the single final write-back
    rounding (<= 2^-9 relative) — a kernel that instead carried bf16 through
    the per-timestep recurrence would drift across the long-sequence case
    below (a canary only; the production gate is gsm8k + a long-generation
    drift check, since serving re-rounds the stored state every decode step).
    """
    with DeviceContext() as ctx:
        if not ctx.is_compatible():
            return
        # Production Nemotron-H grouping, ragged batch, bf16 state.
        run_mamba2_ssd_inplace_vs_functional[
            DType.bfloat16,
            128,
            use_apple=True,
            state_dtype=DType.bfloat16,
            init_state=True,
        ](
            nheads=96,
            head_dim=80,
            ngroups=8,
            max_slots=4,
            seq_lengths=Index(3, 2),
            ctx=ctx,
        )
        # Exact seqlen-1 decode shape (the shape the bf16 pool targets).
        run_mamba2_ssd_inplace_vs_functional[
            DType.bfloat16,
            128,
            use_apple=True,
            state_dtype=DType.bfloat16,
            init_state=True,
        ](
            nheads=96,
            head_dim=80,
            ngroups=8,
            max_slots=8,
            seq_lengths=Index(1, 1, 1, 1),
            ctx=ctx,
        )
        # Long-sequence fp32-accumulate canary (see docstring), small heads to
        # keep the CPU reference cheap.
        run_mamba2_ssd_inplace_apple_long_seq_case(ctx)
        # fp32 state with init_state: pins the initial-state LOAD path on the
        # unchanged fp32 route (previously only covered with a zero pool).
        run_mamba2_ssd_inplace_vs_functional[
            DType.bfloat16, 128, use_apple=True, init_state=True
        ](
            nheads=96,
            head_dim=80,
            ngroups=8,
            max_slots=4,
            seq_lengths=Index(3, 2),
            ctx=ctx,
        )


def run_mamba2_ssd_inplace_apple_long_seq_case(ctx: DeviceContext) raises:
    """Long-sequence bf16-state case, split out for readability."""
    run_mamba2_ssd_inplace_vs_functional[
        DType.bfloat16,
        128,
        use_apple=True,
        state_dtype=DType.bfloat16,
        init_state=True,
    ](
        nheads=8,
        head_dim=16,
        ngroups=2,
        max_slots=4,
        seq_lengths=Index(64, 33),
        ctx=ctx,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
