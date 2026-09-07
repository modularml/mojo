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

"""Mamba-2 SSD (state-space duality) varlen prefill scan.

Target hardware family: B200 (sm_100), but the kernel is architecture-agnostic
(plain elementwise + SIMD-over-dstate recurrence, no MMA/TMA/swizzle). bf16
in/out, fp32 accumulate, states fp32.

This is the varlen (ragged, `query_start_loc`) Mamba-2 SSD forward prefill op,
matching `mamba_chunk_scan_combined` semantics for the `NemotronHMamba2Mixer`
(Nemotron-H). It differs from the Mamba-1 ops in
`varlen_selective_scan.mojo` in three ways that the math requires:

  - `A` is a per-head SCALAR `(nheads,)` (shared across all head_dim channels
    and all dstate), not a per-channel `(dim, dstate)` diagonal.
  - `B`/`C` are GROUPED `(total_len, ngroups, dstate)`; `nheads/ngroups` heads
    share each group (`group_id = h // (nheads // ngroups)`).
  - `dt` is per-head `(total_len, nheads)` + per-head `dt_bias (nheads,)`,
    broadcast across head_dim; softplus is applied to `dt + dt_bias`.

Reference math (the source of truth for parity) is the HF `torch_forward` SSD
path (`segment_sum` / `reshape_into_chunks` / chunk-state recurrence). The SSD
chunked scan is a parallelism reformulation of the linear recurrence below; in
fp32 they are numerically equivalent. This first implementation carries the
state sequentially per `(head, head_dim)` channel (one thread per channel, SIMD
over dstate), mirroring the tiling style of `varlen_selective_scan_fwd_gpu`. A
chunk-tiled rewrite (segment_sum + matmul) is a follow-up perf slice; this op
is gated on CORRECTNESS first.

Per-token recurrence (per head `h`, head_dim channel `p`, group `g`):

    dt_t     = softplus(dt[t, h] + dt_bias[h])            # scalar per (t, h)
    dA_t     = exp(A[h] * dt_t)                           # scalar per (t, h)
    state_n  = state_n * dA_t + dt_t * B[t, g, n] * x[t, h, p]   # vector over n
    y[t,h,p] = sum_n C[t, g, n] * state_n  +  D[h] * x[t, h, p]

State resets to zero (or to `initial_states`) at each `query_start_loc`
boundary -- no cross-sequence bleed. `final_states (batch, nheads, head_dim,
dstate)` is written at each sequence end.
"""

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_dim,
    block_idx,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.primitives.warp import lane_group_sum
from max.algorithm import sync_parallelize
from std.math import exp2
from std.sys.info import align_of
from std.utils.index import IndexList
from std.utils.static_tuple import StaticTuple
from layout import DefaultEngine, TensorLayout, TensorEngine, TileTensor
from state_space.selective_scan import softplus

# LOG2E: convert exp(x) -> exp2(x * LOG2E) (faster on GPU), matching the
# convention in varlen_selective_scan.mojo.
comptime LOG2E = 1.4426950408889634
comptime MAX_DSTATE = 256  # Mamba-2 dstate <= 256

comptime Strides1D = IndexList[1]
comptime Strides2D = IndexList[2]
comptime Strides3D = IndexList[3]
comptime Strides4D = IndexList[4]


def mamba2_ssd_chunk_scan_varlen_fwd_gpu[
    kernel_dtype: DType,
    DSTATE: Int,
    x_LT: TensorLayout,
    dt_LT: TensorLayout,
    A_LT: TensorLayout,
    B_LT: TensorLayout,
    C_LT: TensorLayout,
    D_LT: TensorLayout,
    dt_bias_LT: TensorLayout,
    initial_states_LT: TensorLayout,
    y_LT: TensorLayout,
    final_states_LT: TensorLayout,
    query_start_loc_LT: TensorLayout,
    has_initial_state_LT: TensorLayout,
    # All operands are built from the same source (graph input tensors in
    # production, device buffers in the coverage test) so they share one
    # engine; a single param binds it for every tile argument.
    Engine: TensorEngine = DefaultEngine[element_width=1],
](
    nheads_dev: Int32,
    head_dim_dev: Int32,
    ngroups_dev: Int32,
    nheads_ngroups_ratio_dev: Int32,
    batch_dev: Int32,
    dt_softplus: Int8,
    # Tensors (varlen / ragged: time dim is the packed total_len)
    x: TileTensor[
        kernel_dtype, x_LT, MutUntrackedOrigin, Engine=Engine
    ],  # (total_len, nheads, head_dim)
    dt: TileTensor[
        kernel_dtype, dt_LT, MutUntrackedOrigin, Engine=Engine
    ],  # (total_len, nheads)
    A: TileTensor[
        kernel_dtype, A_LT, MutUntrackedOrigin, Engine=Engine
    ],  # (nheads,)
    B: TileTensor[
        kernel_dtype, B_LT, MutUntrackedOrigin, Engine=Engine
    ],  # (total_len, ngroups, dstate)
    C: TileTensor[
        kernel_dtype, C_LT, MutUntrackedOrigin, Engine=Engine
    ],  # (total_len, ngroups, dstate)
    D: TileTensor[
        kernel_dtype, D_LT, MutUntrackedOrigin, Engine=Engine
    ],  # (nheads,) optional
    dt_bias: TileTensor[
        kernel_dtype, dt_bias_LT, MutUntrackedOrigin, Engine=Engine
    ],  # (nheads,) optional
    initial_states: TileTensor[
        .float32, initial_states_LT, MutUntrackedOrigin, Engine=Engine
    ],  # (batch, nheads, head_dim, dstate) optional
    y: TileTensor[
        kernel_dtype, y_LT, MutUntrackedOrigin, Engine=Engine
    ],  # (total_len, nheads, head_dim)
    final_states: TileTensor[
        .float32, final_states_LT, MutUntrackedOrigin, Engine=Engine
    ],  # (batch, nheads, head_dim, dstate)
    query_start_loc: TileTensor[
        .int32, query_start_loc_LT, MutUntrackedOrigin, Engine=Engine
    ],  # (batch + 1,)
    has_initial_state: TileTensor[
        .bool, has_initial_state_LT, MutUntrackedOrigin, Engine=Engine
    ],  # (batch,) optional
    x_strides: Strides3D,  # (total_len, nheads, head_dim)
    dt_strides: Strides2D,  # (total_len, nheads)
    A_strides: Strides1D,  # (nheads,)
    B_strides: Strides3D,  # (total_len, ngroups, dstate)
    C_strides: Strides3D,  # (total_len, ngroups, dstate)
    D_strides: Strides1D,  # (nheads,)
    dt_bias_strides: Strides1D,  # (nheads,)
    initial_states_strides: Strides4D,  # (batch, nheads, head_dim, dstate)
    y_strides: Strides3D,  # (total_len, nheads, head_dim)
    final_states_strides: Strides4D,  # (batch, nheads, head_dim, dstate)
):
    """GPU kernel: Mamba-2 SSD varlen prefill scan, one thread per (head, channel).

    Grid: (ceildiv(head_dim, BLOCK), nheads, batch). Each thread owns one
    `(b, h, p)` channel, carries the `dstate`-vector state in registers, and
    walks its sequence `[seq_start, seq_end)` sequentially.
    """
    var nheads = Int(nheads_dev)
    var head_dim = Int(head_dim_dev)
    var nheads_ngroups_ratio = Int(nheads_ngroups_ratio_dev)
    var batch = Int(batch_dev)
    # block_idx.x * block_dim.x + thread_idx.x -> head_dim channel p
    var p = block_dim.x * block_idx.x + thread_idx.x
    var h = block_idx.y  # head
    var b = block_idx.z  # batch (sequence)

    if p >= head_dim or h >= nheads or b >= batch:
        return

    var has_D = Int(D.dim[0]()) > 0
    var has_dt_bias = Int(dt_bias.dim[0]()) > 0
    var has_init_tensor = Int(has_initial_state.dim[0]()) > 0
    var dt_softplus_bool = Bool(Int(dt_softplus) != 0)

    var group_id = h // nheads_ngroups_ratio

    # Sequence bounds for this batch element.
    var seq_start = Int(query_start_loc.raw_load(b))
    var seq_end = Int(query_start_loc.raw_load(b + 1))
    var seq_len = seq_end - seq_start
    if seq_len <= 0:
        return

    # Per-head scalar A, pre-multiplied by LOG2E for exp2.
    var A_val = (
        Scalar[kernel_dtype](A.raw_load(UInt32(h * A_strides[0]))).cast[
            DType.float32
        ]()
        * LOG2E
    )

    var dt_bias_val = Float32(0.0)
    if has_dt_bias:
        dt_bias_val = Scalar[kernel_dtype](
            dt_bias.raw_load(UInt32(h * dt_bias_strides[0]))
        ).cast[.float32]()

    var D_val = Float32(0.0)
    if has_D:
        D_val = Scalar[kernel_dtype](D.raw_load(UInt32(h * D_strides[0]))).cast[
            DType.float32
        ]()

    # State vector over dstate, fp32. Initialise from initial_states if present.
    var state = SIMD[.float32, MAX_DSTATE](0.0)
    var use_initial = False
    if has_init_tensor:
        use_initial = Bool(has_initial_state.raw_load(b))
    if use_initial:
        comptime for n in range(DSTATE):
            var off = UInt32(
                b * initial_states_strides[0]
                + h * initial_states_strides[1]
                + p * initial_states_strides[2]
                + n * initial_states_strides[3]
            )
            state[n] = initial_states.raw_load(off)

    # Sequential recurrence over the sequence.
    for t in range(seq_len):
        var gt = seq_start + t  # global (packed) time index

        # x[gt, h, p]
        var x_val = Scalar[kernel_dtype](
            x.raw_load(
                UInt32(gt * x_strides[0] + h * x_strides[1] + p * x_strides[2])
            )
        ).cast[.float32]()

        # dt[gt, h] (+ dt_bias), softplus -> per-(t,h) scalar (broadcast over p).
        var dt_val = Scalar[kernel_dtype](
            dt.raw_load(UInt32(gt * dt_strides[0] + h * dt_strides[1]))
        ).cast[.float32]()
        if has_dt_bias:
            dt_val += dt_bias_val
        if dt_softplus_bool:
            dt_val = softplus(dt_val)

        # dA = exp(A * dt) (scalar), dt_x = dt * x (discretised input).
        var dA = exp2(A_val * dt_val)
        var dt_x = dt_val * x_val

        # Load B and C rows for this (gt, group_id).
        var B_vals = SIMD[.float32, MAX_DSTATE](0.0)
        var C_vals = SIMD[.float32, MAX_DSTATE](0.0)
        comptime for n in range(DSTATE):
            B_vals[n] = Scalar[kernel_dtype](
                B.raw_load(
                    UInt32(
                        gt * B_strides[0]
                        + group_id * B_strides[1]
                        + n * B_strides[2]
                    )
                )
            ).cast[.float32]()
            C_vals[n] = Scalar[kernel_dtype](
                C.raw_load(
                    UInt32(
                        gt * C_strides[0]
                        + group_id * C_strides[1]
                        + n * C_strides[2]
                    )
                )
            ).cast[.float32]()

        # state_n = state_n * dA + (dt * x) * B_n   (vector over dstate)
        state = state * dA + B_vals * dt_x

        # y = sum_n C_n * state_n  + D * x
        var y_val = (state * C_vals).reduce_add()
        if has_D:
            y_val += D_val * x_val

        y.raw_store(
            UInt32(gt * y_strides[0] + h * y_strides[1] + p * y_strides[2]),
            Scalar[kernel_dtype](y_val.cast[kernel_dtype]()),
        )

    # Write final state (fp32) for chunked-prefill continuation / decode handoff.
    comptime for n in range(DSTATE):
        var off = UInt32(
            b * final_states_strides[0]
            + h * final_states_strides[1]
            + p * final_states_strides[2]
            + n * final_states_strides[3]
        )
        final_states.raw_store(off, state[n])


def mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu[
    kernel_dtype: DType,
    DSTATE: Int,
    x_LT: TensorLayout,
    dt_LT: TensorLayout,
    A_LT: TensorLayout,
    B_LT: TensorLayout,
    C_LT: TensorLayout,
    D_LT: TensorLayout,
    dt_bias_LT: TensorLayout,
    y_LT: TensorLayout,
    ssm_pool_LT: TensorLayout,
    query_start_loc_LT: TensorLayout,
    has_initial_state_LT: TensorLayout,
    cache_indices_LT: TensorLayout,
    # All operands share one engine (see the non-inplace variant); a
    # single param binds it for every tile argument.
    Engine: TensorEngine = DefaultEngine[element_width=1],
](
    nheads_dev: Int32,
    head_dim_dev: Int32,
    ngroups_dev: Int32,
    nheads_ngroups_ratio_dev: Int32,
    batch_dev: Int32,
    dt_softplus: Int8,
    x: TileTensor[kernel_dtype, x_LT, MutUntrackedOrigin, Engine=Engine],
    dt: TileTensor[kernel_dtype, dt_LT, MutUntrackedOrigin, Engine=Engine],
    A: TileTensor[kernel_dtype, A_LT, MutUntrackedOrigin, Engine=Engine],
    B: TileTensor[kernel_dtype, B_LT, MutUntrackedOrigin, Engine=Engine],
    C: TileTensor[kernel_dtype, C_LT, MutUntrackedOrigin, Engine=Engine],
    D: TileTensor[kernel_dtype, D_LT, MutUntrackedOrigin, Engine=Engine],
    dt_bias: TileTensor[
        kernel_dtype, dt_bias_LT, MutUntrackedOrigin, Engine=Engine
    ],
    y: TileTensor[kernel_dtype, y_LT, MutUntrackedOrigin, Engine=Engine],
    # ssm_pool: [max_slots, nheads, head_dim, dstate] fp32 — read for initial
    # state (when has_initial_state[b]) and written in-place at slot
    # cache_indices[b] (instead of a separate final_states output).
    ssm_pool: TileTensor[
        .float32, ssm_pool_LT, MutUntrackedOrigin, Engine=Engine
    ],
    query_start_loc: TileTensor[
        .int32, query_start_loc_LT, MutUntrackedOrigin, Engine=Engine
    ],
    has_initial_state: TileTensor[
        .bool, has_initial_state_LT, MutUntrackedOrigin, Engine=Engine
    ],
    cache_indices: TileTensor[
        .uint32, cache_indices_LT, MutUntrackedOrigin, Engine=Engine
    ],
    x_strides: Strides3D,
    dt_strides: Strides2D,
    A_strides: Strides1D,
    B_strides: Strides3D,
    C_strides: Strides3D,
    D_strides: Strides1D,
    dt_bias_strides: Strides1D,
    y_strides: Strides3D,
    ssm_pool_strides: Strides4D,
):
    """GPU kernel: Mamba-2 SSD varlen prefill scan with in-place SSM-pool write.

    Identical to ``mamba2_ssd_chunk_scan_varlen_fwd_gpu`` except final states
    are written directly into ``ssm_pool[cache_indices[b], ...]`` (fp32,
    [max_slots, nheads, head_dim, dstate]) instead of a separate
    ``final_states`` output tensor. This eliminates the graph-side
    gather/scatter_nd/buffer_store whole-pool round-trip.

    Grid: (ceildiv(head_dim, BLOCK), nheads, batch). Same launch shape as the
    non-inplace variant.

    Parameters:
        kernel_dtype: Element type of the input/output tensors `x`, `dt`,
            `A`, `B`, `C`, `D`, `dt_bias`, and `y`.
        DSTATE: State dimension per head; the `dstate` extent of `B`, `C`,
            and the SSM state vector.
        x_LT: Tensor layout of `x`.
        dt_LT: Tensor layout of `dt`.
        A_LT: Tensor layout of `A`.
        B_LT: Tensor layout of `B`.
        C_LT: Tensor layout of `C`.
        D_LT: Tensor layout of `D`.
        dt_bias_LT: Tensor layout of `dt_bias`.
        y_LT: Tensor layout of `y`.
        ssm_pool_LT: Tensor layout of `ssm_pool`.
        query_start_loc_LT: Tensor layout of `query_start_loc`.
        has_initial_state_LT: Tensor layout of `has_initial_state`.
        cache_indices_LT: Tensor layout of `cache_indices`.
        Engine: Engine shared by all tile operands (defaults to
            `DefaultEngine[element_width=1]`).

    Args:
        nheads_dev: Number of attention heads.
        head_dim_dev: Channel dimension per head; the `p` extent of `x`
            and `y`.
        ngroups_dev: Number of B/C groups; `nheads // ngroups` heads share
            each group.
        nheads_ngroups_ratio_dev: Ratio `nheads // ngroups` mapping head
            `h` to group `h // nheads_ngroups_ratio`.
        batch_dev: Number of sequences in the varlen batch.
        dt_softplus: Nonzero applies `softplus` to `dt + dt_bias`; zero
            skips it.
        x: Input sequence tensor of shape `(total_len, nheads, head_dim)`.
        dt: Step-size tensor of shape `(total_len, nheads)`.
        A: Per-head scalar recurrence diagonal of shape `(nheads,)`.
        B: Input-projection tensor of shape `(total_len, ngroups, dstate)`.
        C: Output-projection tensor of shape `(total_len, ngroups, dstate)`.
        D: Skip-connection per head of shape `(nheads,)`; may be empty to
            disable.
        dt_bias: Per-head bias added to `dt` of shape `(nheads,)`; may be
            empty to disable.
        y: Output tensor of shape `(total_len, nheads, head_dim)`, written
            in place.
        ssm_pool: State pool of shape `(max_slots, nheads, head_dim,
            dstate)` fp32; read for initial state and written in place at
            `cache_indices[b]`.
        query_start_loc: Cumulative sequence offsets of shape
            `(batch + 1,)`; sequence `b` spans
            `[query_start_loc[b], query_start_loc[b+1])`.
        has_initial_state: Per-sequence flag of shape `(batch,)`; when true,
            load the initial state from `ssm_pool[cache_indices[b]]`.
        cache_indices: Slot index per sequence of shape `(batch,)` into
            `ssm_pool` for the initial and final state.
        x_strides: Strides of `x` along `(total_len, nheads, head_dim)`.
        dt_strides: Strides of `dt` along `(total_len, nheads)`.
        A_strides: Strides of `A` along `(nheads,)`.
        B_strides: Strides of `B` along `(total_len, ngroups, dstate)`.
        C_strides: Strides of `C` along `(total_len, ngroups, dstate)`.
        D_strides: Strides of `D` along `(nheads,)`.
        dt_bias_strides: Strides of `dt_bias` along `(nheads,)`.
        y_strides: Strides of `y` along `(total_len, nheads, head_dim)`.
        ssm_pool_strides: Strides of `ssm_pool` along `(max_slots, nheads,
            head_dim, dstate)`.
    """
    var nheads = Int(nheads_dev)
    var head_dim = Int(head_dim_dev)
    var nheads_ngroups_ratio = Int(nheads_ngroups_ratio_dev)
    var batch = Int(batch_dev)
    var p = block_dim.x * block_idx.x + thread_idx.x
    var h = block_idx.y
    var b = block_idx.z

    if p >= head_dim or h >= nheads or b >= batch:
        return

    var has_D = Int(D.dim[0]()) > 0
    var has_dt_bias = Int(dt_bias.dim[0]()) > 0
    var has_init_tensor = Int(has_initial_state.dim[0]()) > 0
    var dt_softplus_bool = Bool(Int(dt_softplus) != 0)

    var group_id = h // nheads_ngroups_ratio

    var seq_start = Int(query_start_loc.raw_load(b))
    var seq_end = Int(query_start_loc.raw_load(b + 1))
    var seq_len = seq_end - seq_start
    if seq_len <= 0:
        return

    var A_val = (
        Scalar[kernel_dtype](A.raw_load(UInt32(h * A_strides[0]))).cast[
            DType.float32
        ]()
        * LOG2E
    )

    var dt_bias_val = Float32(0.0)
    if has_dt_bias:
        dt_bias_val = Scalar[kernel_dtype](
            dt_bias.raw_load(UInt32(h * dt_bias_strides[0]))
        ).cast[.float32]()

    var D_val = Float32(0.0)
    if has_D:
        D_val = Scalar[kernel_dtype](D.raw_load(UInt32(h * D_strides[0]))).cast[
            DType.float32
        ]()

    # Load initial state from ssm_pool at the slot for this sequence.
    var slot = Int(cache_indices.raw_load(b))
    var state = SIMD[.float32, MAX_DSTATE](0.0)
    var use_initial = False
    if has_init_tensor:
        use_initial = Bool(has_initial_state.raw_load(b))
    if use_initial:
        # Read initial state from ssm_pool[slot, h, p, n].
        comptime for n in range(DSTATE):
            var off = UInt32(
                slot * ssm_pool_strides[0]
                + h * ssm_pool_strides[1]
                + p * ssm_pool_strides[2]
                + n * ssm_pool_strides[3]
            )
            state[n] = ssm_pool.raw_load(off)

    for t in range(seq_len):
        var gt = seq_start + t

        var x_val = Scalar[kernel_dtype](
            x.raw_load(
                UInt32(gt * x_strides[0] + h * x_strides[1] + p * x_strides[2])
            )
        ).cast[.float32]()

        var dt_val = Scalar[kernel_dtype](
            dt.raw_load(UInt32(gt * dt_strides[0] + h * dt_strides[1]))
        ).cast[.float32]()
        if has_dt_bias:
            dt_val += dt_bias_val
        if dt_softplus_bool:
            dt_val = softplus(dt_val)

        var dA = exp2(A_val * dt_val)
        var dt_x = dt_val * x_val

        var B_vals = SIMD[.float32, MAX_DSTATE](0.0)
        var C_vals = SIMD[.float32, MAX_DSTATE](0.0)
        comptime for n in range(DSTATE):
            B_vals[n] = Scalar[kernel_dtype](
                B.raw_load(
                    UInt32(
                        gt * B_strides[0]
                        + group_id * B_strides[1]
                        + n * B_strides[2]
                    )
                )
            ).cast[.float32]()
            C_vals[n] = Scalar[kernel_dtype](
                C.raw_load(
                    UInt32(
                        gt * C_strides[0]
                        + group_id * C_strides[1]
                        + n * C_strides[2]
                    )
                )
            ).cast[.float32]()

        state = state * dA + B_vals * dt_x

        var y_val = (state * C_vals).reduce_add()
        if has_D:
            y_val += D_val * x_val

        y.raw_store(
            UInt32(gt * y_strides[0] + h * y_strides[1] + p * y_strides[2]),
            Scalar[kernel_dtype](y_val.cast[kernel_dtype]()),
        )

    # Write final state directly into ssm_pool at slot cache_indices[b].
    comptime for n in range(DSTATE):
        var off = UInt32(
            slot * ssm_pool_strides[0]
            + h * ssm_pool_strides[1]
            + p * ssm_pool_strides[2]
            + n * ssm_pool_strides[3]
        )
        ssm_pool.raw_store(off, state[n])


# NVIDIA B200 (sm_100) launch-bounds occupancy floor. This kernel is only ever
# launched with a 128-thread block (production kernels.mojo and the unit test
# both use `block_dim=(DSTATE_SPLIT, CH_PER_BLOCK, 1)` with
# DSTATE_SPLIT*CH_PER_BLOCK == 128), so `.maxntid 128` is exact.
#
# At the c32 (batch-32) decode regime this is the #1 decode GPU kernel and is
# REGISTER-capped, not launch-bound. For the production tile L == DSTATE //
# DSTATE_SPLIT == 128 // 8 == 16, the kernel allocates 149 registers uncapped
# (3 CTAs/SM, ~16% occupancy) yet fits a 128-register budget with ZERO spill
# (the 149->128 is dead slack; measured via `_ptxas_info_verbose`). Setting
# `.minnctapersm 4` forces that 128-reg budget: 4 CTAs * 128 threads * 128 regs
# == the 65536-register SM file, i.e. 4 CTAs/SM (~25% occupancy). Because the
# kernel is Long-Scoreboard / latency-bound (DRAM ~15% SoL, not bandwidth-
# bound), the extra resident CTAs hide global-load latency -- occupancy is the
# lever, and the zero-spill 128-reg point is the one that pays off.
#
# The floor is GATED on `L <= 16`: for larger tiles (e.g. DSTATE==256, split 8
# -> L==32) the hot fp32 state/B/C SIMD vectors of width L do NOT fit 128
# registers (they spill even uncapped at 255), so a 128-reg floor would only
# add spill traffic. Those tiles keep `.minnctapersm 1` (a no-op floor) and
# ptxas's natural register budget. This is a codegen (regalloc) hint only --
# output is bit-identical to the un-annotated kernel; the correctness gate is
# `test_mamba2_ssd_inplace_dstate_split_vs_functional`.
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(128))
)
@__llvm_metadata(
    `nvvm.minctasm`=SIMDLength(4) if (DSTATE // DSTATE_SPLIT)
    <= 16 else SIMDLength(1)
)
def mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_dstate_split[
    kernel_dtype: DType,
    DSTATE: Int,
    x_LT: TensorLayout,
    dt_LT: TensorLayout,
    A_LT: TensorLayout,
    B_LT: TensorLayout,
    C_LT: TensorLayout,
    D_LT: TensorLayout,
    dt_bias_LT: TensorLayout,
    y_LT: TensorLayout,
    ssm_pool_LT: TensorLayout,
    query_start_loc_LT: TensorLayout,
    has_initial_state_LT: TensorLayout,
    cache_indices_LT: TensorLayout,
    Engine: TensorEngine,
    DSTATE_SPLIT: Int = 1,
](
    nheads_dev: Int32,
    head_dim_dev: Int32,
    ngroups_dev: Int32,
    nheads_ngroups_ratio_dev: Int32,
    batch_dev: Int32,
    dt_softplus: Int8,
    x: TileTensor[kernel_dtype, x_LT, MutUntrackedOrigin, Engine=Engine],
    dt: TileTensor[kernel_dtype, dt_LT, MutUntrackedOrigin, Engine=Engine],
    A: TileTensor[kernel_dtype, A_LT, MutUntrackedOrigin, Engine=Engine],
    B: TileTensor[kernel_dtype, B_LT, MutUntrackedOrigin, Engine=Engine],
    C: TileTensor[kernel_dtype, C_LT, MutUntrackedOrigin, Engine=Engine],
    D: TileTensor[kernel_dtype, D_LT, MutUntrackedOrigin, Engine=Engine],
    dt_bias: TileTensor[
        kernel_dtype, dt_bias_LT, MutUntrackedOrigin, Engine=Engine
    ],
    y: TileTensor[kernel_dtype, y_LT, MutUntrackedOrigin, Engine=Engine],
    ssm_pool: TileTensor[
        .float32, ssm_pool_LT, MutUntrackedOrigin, Engine=Engine
    ],
    query_start_loc: TileTensor[
        .int32, query_start_loc_LT, MutUntrackedOrigin, Engine=Engine
    ],
    has_initial_state: TileTensor[
        .bool, has_initial_state_LT, MutUntrackedOrigin, Engine=Engine
    ],
    cache_indices: TileTensor[
        .uint32, cache_indices_LT, MutUntrackedOrigin, Engine=Engine
    ],
    x_strides: Strides3D,
    dt_strides: Strides2D,
    A_strides: Strides1D,
    B_strides: Strides3D,
    C_strides: Strides3D,
    D_strides: Strides1D,
    dt_bias_strides: Strides1D,
    y_strides: Strides3D,
    ssm_pool_strides: Strides4D,
):
    """GPU kernel: Mamba-2 SSD varlen in-place scan, cooperative DSTATE-split.

    NVIDIA B200 (sm_100) decode-occupancy variant. Numerically equivalent to
    ``mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu`` (the portable v1 kernel);
    it is gated on B200 in the wrapper because its output reduction uses a
    ``lane_group_sum`` full-warp shuffle butterfly and a 2D
    ``(DSTATE_SPLIT, CH_PER_BLOCK)`` block that assume a warp width of 32. On
    AMD (wavefront width 64) the lane-group layout is invalid, so AMD/non-B200
    runs the v1 kernel instead (this is why the round-2 split was reverted in
    07c5e0b7533; the gate is the portable restore).

    Grid: (ceildiv(head_dim, CH_PER_BLOCK), nheads, batch), where one block
    holds ``block_dim.y`` head_dim channels and ``block_dim.x == DSTATE_SPLIT``
    threads cooperate on each channel's DSTATE recurrence.

    Cooperative DSTATE-split (decode bs=1 occupancy lever, r7):

      The v1 kernel mapped one thread to one ``(h, p)`` head_dim channel and ran
      a fully serial ``DSTATE``-iteration fp32 recurrence in that thread. At
      decode bs=1 that exposed only ``nheads*head_dim`` work items, leaving the
      GPU ~96% idle (achieved occupancy ~4% on B200) -- the binding bottleneck.

      Here ``DSTATE_SPLIT`` threads cooperate on each channel: thread ``tx`` owns
      the contiguous DSTATE sub-tile ``[tx*L, (tx+1)*L)`` with ``L = DSTATE //
      DSTATE_SPLIT``. The recurrence ``state[n] = state[n]*dA + B[n]*dt_x`` is
      data-parallel across ``n`` so each thread runs it independently over its
      L lanes; the output ``y = sum_n state[n]*C[n]`` is a reduction, done as a
      per-thread partial ``reduce_add`` over its L lanes followed by a
      ``lane_group_sum[num_lanes=DSTATE_SPLIT, stride=1]`` butterfly within each
      contiguous DSTATE_SPLIT-lane group (warp-aligned because ``DSTATE_SPLIT``
      divides the warp and ``tx`` is the fastest-varying thread index). The
      group leader (``tx == 0``) adds the ``D*x`` skip term and stores ``y``.
      The final ssm_pool RMW is partitioned: each thread writes only its own L
      lanes (disjoint), so the in-place write stays correct.

      ``DSTATE_SPLIT == 1`` reproduces the v1 one-thread-per-channel mapping.
      The per-thread state / B / C SIMD vectors are sized to the tile width
      ``L`` (the v1 ``MAX_DSTATE``-wide vectors had always-zero upper lanes that
      only inflated register pressure and the recurrence/reduce width).

      Full-warp participation: ``lane_group_sum`` uses a full-warp shuffle, so
      out-of-range channel threads (``p >= head_dim``) must NOT early-return --
      they stay alive to keep every lane participating in the butterfly, and
      only their per-channel global loads/stores are guarded by ``active``. The
      ``h``/``b``/``seq_len`` guards are uniform across the whole warp (they do
      not depend on ``tx`` or ``ty``), so the warp agrees and the shuffle is not
      reached divergently.
    """
    var nheads = Int(nheads_dev)
    var head_dim = Int(head_dim_dev)
    var nheads_ngroups_ratio = Int(nheads_ngroups_ratio_dev)
    var batch = Int(batch_dev)
    comptime L = DSTATE // DSTATE_SPLIT
    var tx = thread_idx.x  # DSTATE sub-tile index in [0, DSTATE_SPLIT)
    var p = block_idx.x * block_dim.y + thread_idx.y
    var h = block_idx.y
    var b = block_idx.z

    # NOTE: do NOT early-return out-of-range channel threads (p >= head_dim) --
    # they must stay alive to participate in the full-warp lane_group_sum below.
    # The remaining guards (h, b, seq_len) are warp-uniform.
    if h >= nheads or b >= batch:
        return

    var active = p < head_dim
    var n_base = Int(tx) * L  # first DSTATE lane owned by this thread

    # Vectorized contiguous I/O lever (B200 decode). Each thread's L-lane
    # sub-tile of the ssm_pool state and of B/C is a contiguous run whenever the
    # innermost (dstate) stride is 1 -- the standard row-major layout every
    # caller uses. Move that run in ONE SIMD load/store instead of L scalar
    # raw_loads: this is the fix for the split=8 decode kernel being
    # L1TEX/mem-pipe-bound on many small scalar loads (ncu at the served decode
    # shape: SM SoL ~18%, DRAM SoL ~10% so NOT bandwidth-bound, mem-pipe SoL
    # ~75%, top stall Long Scoreboard ~9 cyc). The `*_contig` guards fall back
    # to the exact scalar path when a caller passes a non-contiguous stride, so
    # the result is bit-identical. `n_base` is a multiple of L and dstate is a
    # multiple of L, so the row-major base offset is L-aligned -- the SIMD
    # alignment below is satisfied for every dispatched DSTATE (16/64/128/256).
    comptime pool_align = align_of[SIMD[.float32, L]]()
    comptime bc_align = align_of[SIMD[kernel_dtype, L]]()
    var pool_contig = ssm_pool_strides[3] == 1
    var bc_contig = (B_strides[2] == 1) and (C_strides[2] == 1)

    var has_D = Int(D.dim[0]()) > 0
    var has_dt_bias = Int(dt_bias.dim[0]()) > 0
    var has_init_tensor = Int(has_initial_state.dim[0]()) > 0
    var dt_softplus_bool = Bool(Int(dt_softplus) != 0)

    var group_id = h // nheads_ngroups_ratio

    var seq_start = Int(query_start_loc.raw_load(b))
    var seq_end = Int(query_start_loc.raw_load(b + 1))
    var seq_len = seq_end - seq_start
    if seq_len <= 0:
        return

    var A_val = (
        Scalar[kernel_dtype](A.raw_load(UInt32(h * A_strides[0]))).cast[
            DType.float32
        ]()
        * LOG2E
    )

    var dt_bias_val = Float32(0.0)
    if has_dt_bias:
        dt_bias_val = Scalar[kernel_dtype](
            dt_bias.raw_load(UInt32(h * dt_bias_strides[0]))
        ).cast[.float32]()

    var D_val = Float32(0.0)
    if has_D:
        D_val = Scalar[kernel_dtype](D.raw_load(UInt32(h * D_strides[0]))).cast[
            DType.float32
        ]()

    # Load this thread's DSTATE sub-tile of the initial state from ssm_pool.
    var slot = Int(cache_indices.raw_load(b))
    var state = SIMD[.float32, L](0.0)
    var use_initial = False
    if has_init_tensor:
        use_initial = Bool(has_initial_state.raw_load(b))
    if active and use_initial:
        # Read this thread's contiguous L-lane sub-tile of the initial state
        # from ssm_pool[slot, h, p, n_base ..].
        if pool_contig:
            state = ssm_pool.raw_load[width=L, alignment=pool_align](
                UInt32(
                    slot * ssm_pool_strides[0]
                    + h * ssm_pool_strides[1]
                    + p * ssm_pool_strides[2]
                    + n_base * ssm_pool_strides[3]
                )
            )
        else:
            comptime for i in range(L):
                var off = UInt32(
                    slot * ssm_pool_strides[0]
                    + h * ssm_pool_strides[1]
                    + p * ssm_pool_strides[2]
                    + (n_base + i) * ssm_pool_strides[3]
                )
                state[i] = ssm_pool.raw_load(off)

    for t in range(seq_len):
        var gt = seq_start + t

        var x_val = Float32(0.0)
        var dt_x = Float32(0.0)
        var dA = Float32(0.0)
        var B_vals = SIMD[.float32, L](0.0)
        var C_vals = SIMD[.float32, L](0.0)
        if active:
            x_val = Scalar[kernel_dtype](
                x.raw_load(
                    UInt32(
                        gt * x_strides[0] + h * x_strides[1] + p * x_strides[2]
                    )
                )
            ).cast[.float32]()

            var dt_val = Scalar[kernel_dtype](
                dt.raw_load(UInt32(gt * dt_strides[0] + h * dt_strides[1]))
            ).cast[.float32]()
            if has_dt_bias:
                dt_val += dt_bias_val
            if dt_softplus_bool:
                dt_val = softplus(dt_val)

            dA = exp2(A_val * dt_val)
            dt_x = dt_val * x_val

            # This thread only loads B/C lanes for its own DSTATE sub-tile.
            if bc_contig:
                B_vals = B.raw_load[width=L, alignment=bc_align](
                    UInt32(
                        gt * B_strides[0]
                        + group_id * B_strides[1]
                        + n_base * B_strides[2]
                    )
                ).cast[.float32]()
                C_vals = C.raw_load[width=L, alignment=bc_align](
                    UInt32(
                        gt * C_strides[0]
                        + group_id * C_strides[1]
                        + n_base * C_strides[2]
                    )
                ).cast[.float32]()
            else:
                comptime for i in range(L):
                    var n = n_base + i
                    B_vals[i] = Scalar[kernel_dtype](
                        B.raw_load(
                            UInt32(
                                gt * B_strides[0]
                                + group_id * B_strides[1]
                                + n * B_strides[2]
                            )
                        )
                    ).cast[.float32]()
                    C_vals[i] = Scalar[kernel_dtype](
                        C.raw_load(
                            UInt32(
                                gt * C_strides[0]
                                + group_id * C_strides[1]
                                + n * C_strides[2]
                            )
                        )
                    ).cast[.float32]()

        state = state * dA + B_vals * dt_x

        var y_partial = (state * C_vals).reduce_add()
        # Combine the DSTATE_SPLIT partials within the contiguous lane group.
        # Every lane (including inactive channels) participates -- the full-warp
        # shuffle requires all lanes present; inactive results are discarded.
        comptime if DSTATE_SPLIT > 1:
            y_partial = lane_group_sum[num_lanes=DSTATE_SPLIT, stride=1](
                y_partial
            )

        if active and tx == 0:
            var y_val = y_partial
            if has_D:
                y_val += D_val * x_val
            y.raw_store(
                UInt32(gt * y_strides[0] + h * y_strides[1] + p * y_strides[2]),
                Scalar[kernel_dtype](y_val.cast[kernel_dtype]()),
            )

    # Write this thread's DSTATE sub-tile of the final state into ssm_pool at
    # slot cache_indices[b] (the DSTATE_SPLIT threads cover disjoint lanes).
    if active:
        if pool_contig:
            ssm_pool.raw_store[width=L, alignment=pool_align](
                UInt32(
                    slot * ssm_pool_strides[0]
                    + h * ssm_pool_strides[1]
                    + p * ssm_pool_strides[2]
                    + n_base * ssm_pool_strides[3]
                ),
                state,
            )
        else:
            comptime for i in range(L):
                var off = UInt32(
                    slot * ssm_pool_strides[0]
                    + h * ssm_pool_strides[1]
                    + p * ssm_pool_strides[2]
                    + (n_base + i) * ssm_pool_strides[3]
                )
                ssm_pool.raw_store(off, state[i])


struct DStateVecLoader[
    state_dtype: DType,
    kernel_dtype: DType,
    ssm_pool_LT: TensorLayout,
    B_LT: TensorLayout,
    C_LT: TensorLayout,
    Engine: TensorEngine,
    VEC: Int,
    NCHUNK: Int,
](ImplicitlyCopyable, Movable):
    """Owner of the Apple SSD kernel's vectorized dstate state / B / C I/O.

    Target hardware family: Apple silicon GPU (Metal 4). The B200
    ``..._dstate_split`` sibling performs the identical widen-on-load /
    round-on-store / vectorized-vs-scalar-fallback logic inline; it COULD reuse
    this owner later (it is parameterized on the storage dtypes / layouts /
    ``Engine``), but is not wired to it here (Apple-scoped change).

    Every DRAM <-> register transition on the innermost (dstate) axis has an
    owner here instead of a ``raw_load`` / ``raw_store`` scattered across the
    kernel body -- the owner-per-transition pattern (see ``VarlenConvIO`` in
    ``varlen_causal_conv1d.mojo``, ``Fp4WeightLoader`` in ``matmul2d_fp4.mojo``,
    and ``new-primitives/amd-tile-io-expert-objects``). Three verbs:

      - ``load_state``: widen ``ssm_pool`` storage -> fp32 register chunks
        (the ``NCHUNK`` ``VEC``-wide initial-state read).
      - ``load_bc``: widen one ``VEC``-lane B and C chunk -> fp32.
      - ``store_state``: round the fp32 state chunks -> ``ssm_pool`` storage.

    Each verb owns the ``VEC``-wide aligned ``raw_load`` / ``raw_store`` and the
    scalar fallback, so the recurrence in the kernel body indexes only fp32
    registers and issues no raw load/store. ``load_state`` / ``store_state``
    each own their contig branch, which wraps the whole ``NCHUNK`` loop (a
    verbatim relocation of the prior inline loops). ``load_bc`` runs inside the
    fused recurrence loop, so it takes the contig decision as a COMPTIME
    ``contig`` parameter: the caller hoists the runtime ``bc_contig`` branch
    ONCE outside the chunk loop, keeping the fast path a straight-line unrolled
    vec-load loop (a per-chunk runtime branch measured ~18% slower on M5). Every
    method is ``@always_inline``. The vectorized fast path (unit innermost
    stride) serves every row-major caller; a non-unit stride selects the exact
    v1 scalar gather/scatter, so the result is bit-identical for any layout.

    Parameters:
        state_dtype: ``ssm_pool`` storage dtype (fp32 or bf16).
        kernel_dtype: B / C element dtype.
        ssm_pool_LT: Layout type of the ``ssm_pool`` view.
        B_LT: Layout type of the ``B`` view.
        C_LT: Layout type of the ``C`` view.
        Engine: Shared engine of the views.
        VEC: SIMD load/store width over the contiguous dstate axis.
        NCHUNK: Number of ``VEC``-wide chunks spanning ``DSTATE``.
    """

    var ssm_pool: TileTensor[
        Self.state_dtype,
        Self.ssm_pool_LT,
        MutUntrackedOrigin,
        Engine=Self.Engine,
    ]
    var B: TileTensor[
        Self.kernel_dtype, Self.B_LT, MutUntrackedOrigin, Engine=Self.Engine
    ]
    var C: TileTensor[
        Self.kernel_dtype, Self.C_LT, MutUntrackedOrigin, Engine=Self.Engine
    ]
    # Innermost (dstate) strides for the scalar fallback gather/scatter.
    var pool_dstate_stride: Int
    var b_dstate_stride: Int
    var c_dstate_stride: Int
    # Row-major contiguity of the dstate axis (unit innermost stride). Runtime
    # because strides are runtime; a non-unit stride selects the scalar path.
    var pool_contig: Bool
    var bc_contig: Bool

    def __init__(
        out self,
        ssm_pool: TileTensor[
            Self.state_dtype,
            Self.ssm_pool_LT,
            MutUntrackedOrigin,
            Engine=Self.Engine,
        ],
        B: TileTensor[
            Self.kernel_dtype,
            Self.B_LT,
            MutUntrackedOrigin,
            Engine=Self.Engine,
        ],
        C: TileTensor[
            Self.kernel_dtype,
            Self.C_LT,
            MutUntrackedOrigin,
            Engine=Self.Engine,
        ],
        pool_dstate_stride: Int,
        b_dstate_stride: Int,
        c_dstate_stride: Int,
    ):
        self.ssm_pool = ssm_pool
        self.B = B
        self.C = C
        self.pool_dstate_stride = pool_dstate_stride
        self.b_dstate_stride = b_dstate_stride
        self.c_dstate_stride = c_dstate_stride
        self.pool_contig = pool_dstate_stride == 1
        self.bc_contig = (b_dstate_stride == 1) and (c_dstate_stride == 1)

    @always_inline
    def load_state(
        self,
        mut state: Array[SIMD[.float32, Self.VEC], Self.NCHUNK],
        pool_base: UInt32,
    ):
        """Fill fp32 ``state`` from ``ssm_pool[.., pool_base + n]`` (widening).

        The load widens to fp32 (a no-op when ``state_dtype`` is fp32); the
        recurrence downstream runs on the fp32 register copy.
        """
        comptime pool_align = align_of[SIMD[Self.state_dtype, Self.VEC]]()
        if self.pool_contig:
            comptime for c in range(Self.NCHUNK):
                state[c] = self.ssm_pool.raw_load[
                    width=Self.VEC, alignment=pool_align
                ](pool_base + UInt32(c * Self.VEC)).cast[.float32]()
        else:
            comptime for c in range(Self.NCHUNK):
                var chunk = SIMD[.float32, Self.VEC](0.0)
                comptime for i in range(Self.VEC):
                    chunk[i] = self.ssm_pool.raw_load(
                        pool_base
                        + UInt32((c * Self.VEC + i) * self.pool_dstate_stride)
                    ).cast[.float32]()
                state[c] = chunk

    @always_inline
    def load_bc[
        c: Int, contig: Bool
    ](
        self,
        b_base: UInt32,
        c_base: UInt32,
        mut b_c: SIMD[.float32, Self.VEC],
        mut c_c: SIMD[.float32, Self.VEC],
    ):
        """Load B and C chunk ``c`` (``VEC`` dstate lanes each), widening to fp32.

        ``contig`` is a COMPTIME parameter: the caller reads ``self.bc_contig``
        and hoists the runtime branch ONCE outside the ``NCHUNK`` chunk loop,
        then instantiates the pure vectorized (``contig=True``) or pure
        scalar-gather (``contig=False``) path here. A per-chunk *runtime* branch
        interleaves the cold scalar-gather code into the hot vectorized loop and
        measured ~18% slower on M5 (33.6 vs 28.4 us/launch at the decode shape),
        so the fast path must stay a straight-line unrolled vec-load loop.
        """
        comptime bc_align = align_of[SIMD[Self.kernel_dtype, Self.VEC]]()
        comptime if contig:
            b_c = self.B.raw_load[width=Self.VEC, alignment=bc_align](
                b_base + UInt32(c * Self.VEC)
            ).cast[.float32]()
            c_c = self.C.raw_load[width=Self.VEC, alignment=bc_align](
                c_base + UInt32(c * Self.VEC)
            ).cast[.float32]()
        else:
            comptime for i in range(Self.VEC):
                var n = c * Self.VEC + i
                b_c[i] = Scalar[Self.kernel_dtype](
                    self.B.raw_load(b_base + UInt32(n * self.b_dstate_stride))
                ).cast[.float32]()
                c_c[i] = Scalar[Self.kernel_dtype](
                    self.C.raw_load(c_base + UInt32(n * self.c_dstate_stride))
                ).cast[.float32]()

    @always_inline
    def store_state(
        self,
        state: Array[SIMD[.float32, Self.VEC], Self.NCHUNK],
        pool_wb: UInt32,
    ):
        """Round fp32 ``state`` to ``state_dtype`` and write back to ``ssm_pool``.

        The round happens once here at the final write-back (a no-op when
        ``state_dtype`` is fp32); the recurrent accumulator never leaves fp32.
        """
        comptime pool_align = align_of[SIMD[Self.state_dtype, Self.VEC]]()
        if self.pool_contig:
            comptime for c in range(Self.NCHUNK):
                self.ssm_pool.raw_store[width=Self.VEC, alignment=pool_align](
                    pool_wb + UInt32(c * Self.VEC),
                    state[c].cast[Self.state_dtype](),
                )
        else:
            comptime for c in range(Self.NCHUNK):
                comptime for i in range(Self.VEC):
                    self.ssm_pool.raw_store(
                        pool_wb
                        + UInt32((c * Self.VEC + i) * self.pool_dstate_stride),
                        state[c][i].cast[Self.state_dtype](),
                    )


def mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_apple[
    kernel_dtype: DType,
    DSTATE: Int,
    x_LT: TensorLayout,
    dt_LT: TensorLayout,
    A_LT: TensorLayout,
    B_LT: TensorLayout,
    C_LT: TensorLayout,
    D_LT: TensorLayout,
    dt_bias_LT: TensorLayout,
    y_LT: TensorLayout,
    ssm_pool_LT: TensorLayout,
    query_start_loc_LT: TensorLayout,
    has_initial_state_LT: TensorLayout,
    cache_indices_LT: TensorLayout,
    # SSM-state pool STORAGE dtype: fp32 (default) or bf16. bf16 halves the
    # dominant per-step pool traffic on Apple silicon; the recurrence still
    # accumulates in fp32 registers (loads widen, only the final write-back
    # rounds). See the docstring numerics contract.
    state_dtype: DType = .float32,
    # SIMD load/store width over the contiguous dstate axis. 4 fp32 = 16 B is
    # the proven M5 vector-load cap; 2/8 are sweepable in the microbench. NOTE:
    # VEC=8 is fp32-ONLY -- a width-8 bf16/fp16 load scalarizes on the M5 (AGX)
    # backend (`<8 x half>`/`<8 x bfloat>` under-alignment cliff), so the
    # shipping default VEC=4 is the safe choice across state dtypes.
    VEC: Int = 4,
    # All operands share one engine (see the v1 variant); a single
    # param binds it for every tile argument.
    Engine: TensorEngine = DefaultEngine[element_width=1],
](
    nheads_dev: Int32,
    head_dim_dev: Int32,
    ngroups_dev: Int32,
    nheads_ngroups_ratio_dev: Int32,
    batch_dev: Int32,
    dt_softplus: Int8,
    x: TileTensor[kernel_dtype, x_LT, MutUntrackedOrigin, Engine=Engine],
    dt: TileTensor[kernel_dtype, dt_LT, MutUntrackedOrigin, Engine=Engine],
    A: TileTensor[kernel_dtype, A_LT, MutUntrackedOrigin, Engine=Engine],
    B: TileTensor[kernel_dtype, B_LT, MutUntrackedOrigin, Engine=Engine],
    C: TileTensor[kernel_dtype, C_LT, MutUntrackedOrigin, Engine=Engine],
    D: TileTensor[kernel_dtype, D_LT, MutUntrackedOrigin, Engine=Engine],
    dt_bias: TileTensor[
        kernel_dtype, dt_bias_LT, MutUntrackedOrigin, Engine=Engine
    ],
    y: TileTensor[kernel_dtype, y_LT, MutUntrackedOrigin, Engine=Engine],
    ssm_pool: TileTensor[
        state_dtype, ssm_pool_LT, MutUntrackedOrigin, Engine=Engine
    ],
    query_start_loc: TileTensor[
        .int32, query_start_loc_LT, MutUntrackedOrigin, Engine=Engine
    ],
    has_initial_state: TileTensor[
        .bool, has_initial_state_LT, MutUntrackedOrigin, Engine=Engine
    ],
    cache_indices: TileTensor[
        .uint32, cache_indices_LT, MutUntrackedOrigin, Engine=Engine
    ],
    x_strides: Strides3D,
    dt_strides: Strides2D,
    A_strides: Strides1D,
    B_strides: Strides3D,
    C_strides: Strides3D,
    D_strides: Strides1D,
    dt_bias_strides: Strides1D,
    y_strides: Strides3D,
    ssm_pool_strides: Strides4D,
):
    """GPU kernel: Mamba-2 SSD varlen in-place scan, Apple-M5 vectorized I/O.

    Target hardware family: Apple silicon GPU (Metal 4, `compute_capability()
    == 5`). Numerically equivalent to
    ``mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu`` (the portable v1 kernel);
    same one-thread-per-``(b, h, p)``-channel mapping and launch geometry
    (grid ``(ceildiv(head_dim, BLOCK), nheads, batch)``, ``block (BLOCK,1,1)``).

    The only change from v1 is the state / B / C I/O. v1 walks the ``dstate``
    axis with a ``comptime for n in range(DSTATE)`` loop of scalar
    ``raw_load`` / ``raw_store`` (128 dependent scalar fp32 loads/thread for
    the state read + 128 for B + 128 for C + 128 scalar stores at the served
    decode shape), which is per-thread scalar-load latency bound on M5 (the
    scan realized ~15-17% of HBM at c32, flat across batch). This variant moves
    the innermost ``dstate`` run in ``VEC``-wide contiguous SIMD loads/stores
    over ``NCHUNK = DSTATE // VEC`` chunks -- the same vectorized-contiguous-I/O
    lever as the B200 ``..._dstate_split`` sibling, WITHOUT its warp-32
    ``lane_group_sum`` cooperative split (Metal simdgroup width; and c32
    occupancy is already full, so the split's bs=1 occupancy benefit does not
    apply here).

    Per-thread state is carried as ``Array[SIMD[fp32, VEC], NCHUNK]`` --
    native ``float4`` chunks rather than v1's monolithic ``SIMD[fp32,
    MAX_DSTATE=256]`` (whose upper ``MAX_DSTATE - DSTATE`` lanes were always
    zero and only inflated register pressure). The recurrence
    ``state = state*dA + B*dt_x`` and the output reduction ``y = sum_n
    state[n]*C[n]`` run per-chunk in fp32; fp32 wide-SIMD arithmetic is safe on
    the M5 backend (per `Kernels/claude_kb/entries/known-limitations/
    apple-m5-wide-16bit-simd-codegen-crash.md`: only >=24-lane 16-bit-element
    arithmetic crashes; B/C are cast to fp32 immediately after each load, so no
    wide bf16 arithmetic occurs). Uses neither `SIMD.insert` (no Apple-GPU
    codegen path in the stdlib) nor `SIMD.slice`.

    ``state_dtype`` selects the ``ssm_pool`` STORAGE dtype only (fp32 default;
    bf16 for the Apple decode path, halving the pool bytes — the pool
    read+write dominates the decode-step traffic). Numerics contract: pool
    loads widen to fp32 in registers, the full per-timestep recurrence
    ``state = state*dA + B*dt_x`` and the ``y`` reduction stay in fp32
    exactly as in the fp32 path, and rounding to ``state_dtype`` happens once
    per kernel invocation at the final write-back — never inside the
    recurrent accumulate. At ``state_dtype == float32`` both boundary casts
    are no-ops, so the fp32 path is byte-identical to before. Across decode
    steps the persistent state is thus re-rounded to bf16 once per step (each
    launch reloads the stored state); a single round is <= 2^-9 relative and
    the recurrence is contractive (``dA = exp(A*dt) < 1``, ``A < 0``), so the
    rounding does not compound. bf16 chunks at ``VEC = 4`` are 8 B
    ``<4 x bfloat>`` accesses, within the <=4-lane alignment-robust regime on
    M5 (see `.claude/agent-memory/mojo-kernel-engineer/
    apple-m5-width8-dtype-asymmetry.md`); the win is the halved bytes, not
    lane width.

    The ``*_contig`` guards fall back to the exact v1 scalar path when a caller
    passes a non-unit innermost stride, so the result matches v1 for any
    layout; the row-major callers (every production path) take the vectorized
    branch. ``DSTATE`` and every dispatched value (16/64/128/256) are multiples
    of ``VEC in {2,4,8}``, and the row-major dstate-axis base offset is a
    multiple of ``VEC``, so each SIMD access is naturally aligned.
    """
    comptime assert (
        DSTATE % VEC == 0
    ), "DSTATE must be a multiple of the SIMD I/O width VEC"
    # fp16 is deliberately excluded: the state magnitude is unbounded by the
    # recurrence (fp16 max 65504 risks overflow) and wide fp16 loads hit the
    # M5 scalarization trap. Only the two validated storage dtypes compile.
    comptime assert (
        state_dtype == .float32 or state_dtype == .bfloat16
    ), "state_dtype must be float32 or bfloat16"

    var nheads = Int(nheads_dev)
    var head_dim = Int(head_dim_dev)
    var nheads_ngroups_ratio = Int(nheads_ngroups_ratio_dev)
    var batch = Int(batch_dev)
    var p = block_dim.x * block_idx.x + thread_idx.x
    var h = block_idx.y
    var b = block_idx.z

    if p >= head_dim or h >= nheads or b >= batch:
        return

    var has_D = Int(D.dim[0]()) > 0
    var has_dt_bias = Int(dt_bias.dim[0]()) > 0
    var has_init_tensor = Int(has_initial_state.dim[0]()) > 0
    var dt_softplus_bool = Bool(Int(dt_softplus) != 0)

    var group_id = h // nheads_ngroups_ratio

    var seq_start = Int(query_start_loc.raw_load(b))
    var seq_end = Int(query_start_loc.raw_load(b + 1))
    var seq_len = seq_end - seq_start
    if seq_len <= 0:
        return

    var A_val = (
        Scalar[kernel_dtype](A.raw_load(UInt32(h * A_strides[0]))).cast[
            DType.float32
        ]()
        * LOG2E
    )

    var dt_bias_val = Float32(0.0)
    if has_dt_bias:
        dt_bias_val = Scalar[kernel_dtype](
            dt_bias.raw_load(UInt32(h * dt_bias_strides[0]))
        ).cast[.float32]()

    var D_val = Float32(0.0)
    if has_D:
        D_val = Scalar[kernel_dtype](D.raw_load(UInt32(h * D_strides[0]))).cast[
            DType.float32
        ]()

    comptime NCHUNK = DSTATE // VEC
    # This loader owns every width=VEC state / B / C DRAM<->register transition
    # (the raw_load/raw_store + the scalar fallback), so the kernel body below
    # indexes only fp32 registers. The innermost (dstate) strides pick the
    # vectorized fast path (unit stride, every row-major caller) or the exact
    # v1 scalar fallback (non-unit stride, bit-identical). The B/C `bc_contig`
    # branch is hoisted once in the recurrence loop below (comptime `contig`),
    # not per chunk -- a per-chunk branch regressed ~18% on M5.
    var loader = DStateVecLoader[
        state_dtype,
        kernel_dtype,
        ssm_pool_LT,
        B_LT,
        C_LT,
        Engine,
        VEC,
        NCHUNK,
    ](ssm_pool, B, C, ssm_pool_strides[3], B_strides[2], C_strides[2])

    # Per-thread dstate state as native VEC-wide fp32 chunks (see docstring).
    var state = Array[SIMD[.float32, VEC], NCHUNK](
        fill=SIMD[.float32, VEC](0.0)
    )

    var slot = Int(cache_indices.raw_load(b))
    var use_initial = False
    if has_init_tensor:
        use_initial = Bool(has_initial_state.raw_load(b))
    if use_initial:
        var pool_base = UInt32(
            slot * ssm_pool_strides[0]
            + h * ssm_pool_strides[1]
            + p * ssm_pool_strides[2]
        )
        # Widen to fp32 at the load boundary (no-op when state_dtype is fp32);
        # all recurrence math below runs on the fp32 register copy.
        loader.load_state(state, pool_base)

    for t in range(seq_len):
        var gt = seq_start + t

        var x_val = Scalar[kernel_dtype](
            x.raw_load(
                UInt32(gt * x_strides[0] + h * x_strides[1] + p * x_strides[2])
            )
        ).cast[.float32]()

        var dt_val = Scalar[kernel_dtype](
            dt.raw_load(UInt32(gt * dt_strides[0] + h * dt_strides[1]))
        ).cast[.float32]()
        if has_dt_bias:
            dt_val += dt_bias_val
        if dt_softplus_bool:
            dt_val = softplus(dt_val)

        var dA = exp2(A_val * dt_val)
        var dt_x = dt_val * x_val

        # y = sum_n C_n * state_n, accumulated lane-wise across chunks then
        # reduced once (fp32; associativity differs from v1's flat reduce by
        # <ULP-scale rounding, within the equivalence tolerance). The loader
        # owns each chunk's B/C widen-load; the recurrence and reduction stay
        # in fp32 registers. The runtime `bc_contig` branch is hoisted ONCE
        # here (not per chunk) and the comptime `contig` specializes the loader
        # so each arm is a straight-line unrolled loop -- a per-chunk runtime
        # branch measured ~18% slower on M5 (33.6 vs 28.4 us at the decode
        # shape) by interleaving cold scalar-gather code into the hot loop.
        var y_acc = SIMD[.float32, VEC](0.0)
        var B_base = UInt32(gt * B_strides[0] + group_id * B_strides[1])
        var C_base = UInt32(gt * C_strides[0] + group_id * C_strides[1])
        if loader.bc_contig:
            comptime for c in range(NCHUNK):
                var b_c = SIMD[.float32, VEC](0.0)
                var c_c = SIMD[.float32, VEC](0.0)
                loader.load_bc[c, True](B_base, C_base, b_c, c_c)
                var s_c = state[c] * dA + b_c * dt_x
                state[c] = s_c
                y_acc += s_c * c_c
        else:
            comptime for c in range(NCHUNK):
                var b_c = SIMD[.float32, VEC](0.0)
                var c_c = SIMD[.float32, VEC](0.0)
                loader.load_bc[c, False](B_base, C_base, b_c, c_c)
                var s_c = state[c] * dA + b_c * dt_x
                state[c] = s_c
                y_acc += s_c * c_c

        var y_val = y_acc.reduce_add()
        if has_D:
            y_val += D_val * x_val

        y.raw_store(
            UInt32(gt * y_strides[0] + h * y_strides[1] + p * y_strides[2]),
            Scalar[kernel_dtype](y_val.cast[kernel_dtype]()),
        )

    # Write final state back into ssm_pool at slot cache_indices[b].
    var pool_wb = UInt32(
        slot * ssm_pool_strides[0]
        + h * ssm_pool_strides[1]
        + p * ssm_pool_strides[2]
    )
    # Round to state_dtype only here, at the final write-back (no-op when
    # state_dtype is fp32); the recurrent accumulator itself never leaves fp32.
    loader.store_state(state, pool_wb)


def mamba2_ssd_chunk_scan_varlen_fwd_inplace_cpu[
    kernel_dtype: DType,
    DSTATE: Int,
](
    nheads: Int,
    head_dim: Int,
    ngroups: Int,
    nheads_ngroups_ratio: Int,
    batch: Int,
    dt_softplus: Int8,
    x: TileTensor[mut=False, kernel_dtype, ...],
    dt: TileTensor[mut=False, kernel_dtype, ...],
    A: TileTensor[mut=False, kernel_dtype, ...],
    B: TileTensor[mut=False, kernel_dtype, ...],
    C: TileTensor[mut=False, kernel_dtype, ...],
    D: TileTensor[mut=False, kernel_dtype, ...],
    dt_bias: TileTensor[mut=False, kernel_dtype, ...],
    y: TileTensor[mut=True, kernel_dtype, ...],
    # ssm_pool: [max_slots, nheads, head_dim, dstate] fp32 — read for initial
    # state (when has_initial_state[b]) and written in-place at slot
    # cache_indices[b] (instead of a separate final_states output).
    ssm_pool: TileTensor[mut=True, .float32, ...],
    query_start_loc: TileTensor[mut=False, .int32, ...],
    has_initial_state: TileTensor[mut=False, .bool, ...],
    cache_indices: TileTensor[mut=False, .uint32, ...],
    x_strides: Strides3D,
    dt_strides: Strides2D,
    A_strides: Strides1D,
    B_strides: Strides3D,
    C_strides: Strides3D,
    D_strides: Strides1D,
    dt_bias_strides: Strides1D,
    y_strides: Strides3D,
    ssm_pool_strides: Strides4D,
    ctx: Optional[DeviceContext] = None,
):
    """CPU reference: Mamba-2 SSD varlen scan with in-place SSM-pool write.

    Mirrors ``mamba2_ssd_chunk_scan_varlen_fwd_cpu`` but writes final states
    into ``ssm_pool[cache_indices[b], ...]`` directly.

    Parameters:
        kernel_dtype: Element type of the input/output tensors `x`, `dt`,
            `A`, `B`, `C`, `D`, `dt_bias`, and `y`.
        DSTATE: State dimension per head; the `dstate` extent of `B`, `C`,
            and the SSM state vector.

    Args:
        nheads: Number of attention heads.
        head_dim: Channel dimension per head; the `p` extent of `x` and `y`.
        ngroups: Number of B/C groups; `nheads // ngroups` heads share each
            group.
        nheads_ngroups_ratio: Ratio `nheads // ngroups` mapping head `h` to
            group `h // nheads_ngroups_ratio`.
        batch: Number of sequences in the varlen batch.
        dt_softplus: Nonzero applies `softplus` to `dt + dt_bias`; zero
            skips it.
        x: Input sequence tensor of shape `(total_len, nheads, head_dim)`.
        dt: Step-size tensor of shape `(total_len, nheads)`.
        A: Per-head scalar recurrence diagonal of shape `(nheads,)`.
        B: Input-projection tensor of shape `(total_len, ngroups, dstate)`.
        C: Output-projection tensor of shape `(total_len, ngroups, dstate)`.
        D: Skip-connection per head of shape `(nheads,)`; may be empty to
            disable.
        dt_bias: Per-head bias added to `dt` of shape `(nheads,)`; may be
            empty to disable.
        y: Output tensor of shape `(total_len, nheads, head_dim)`, written
            in place.
        ssm_pool: State pool of shape `(max_slots, nheads, head_dim,
            dstate)` fp32; read for initial state and written in place at
            `cache_indices[b]`.
        query_start_loc: Cumulative sequence offsets of shape
            `(batch + 1,)`; sequence `b` spans
            `[query_start_loc[b], query_start_loc[b+1])`.
        has_initial_state: Per-sequence flag of shape `(batch,)`; when true,
            load the initial state from `ssm_pool[cache_indices[b]]`.
        cache_indices: Slot index per sequence of shape `(batch,)` into
            `ssm_pool` for the initial and final state.
        x_strides: Strides of `x` along `(total_len, nheads, head_dim)`.
        dt_strides: Strides of `dt` along `(total_len, nheads)`.
        A_strides: Strides of `A` along `(nheads,)`.
        B_strides: Strides of `B` along `(total_len, ngroups, dstate)`.
        C_strides: Strides of `C` along `(total_len, ngroups, dstate)`.
        D_strides: Strides of `D` along `(nheads,)`.
        dt_bias_strides: Strides of `dt_bias` along `(nheads,)`.
        y_strides: Strides of `y` along `(total_len, nheads, head_dim)`.
        ssm_pool_strides: Strides of `ssm_pool` along `(max_slots, nheads,
            head_dim, dstate)`.
        ctx: Device context for the parallel worker pool (defaults to
            `None`).
    """
    var has_D = Int(D.dim[0]()) > 0
    var has_dt_bias = Int(dt_bias.dim[0]()) > 0
    var has_init_tensor = Int(has_initial_state.dim[0]()) > 0
    var dt_softplus_bool = Bool(Int(dt_softplus) != 0)

    def worker(idx: Int) {imm}:
        var b, remaining = divmod(idx, nheads * head_dim)
        var h, p = divmod(remaining, head_dim)

        var group_id = h // nheads_ngroups_ratio

        var seq_start = Int(query_start_loc.raw_load(b))
        var seq_end = Int(query_start_loc.raw_load(b + 1))
        var seq_len = seq_end - seq_start
        if seq_len <= 0:
            return

        var A_val = (
            Scalar[kernel_dtype](A.raw_load(UInt32(h * A_strides[0]))).cast[
                DType.float32
            ]()
            * LOG2E
        )

        var dt_bias_val = Float32(0.0)
        if has_dt_bias:
            dt_bias_val = Scalar[kernel_dtype](
                dt_bias.raw_load(UInt32(h * dt_bias_strides[0]))
            ).cast[.float32]()

        var D_val = Float32(0.0)
        if has_D:
            D_val = Scalar[kernel_dtype](
                D.raw_load(UInt32(h * D_strides[0]))
            ).cast[.float32]()

        var slot = Int(cache_indices.raw_load(b))
        var state = SIMD[.float32, MAX_DSTATE](0.0)
        var use_initial = False
        if has_init_tensor:
            use_initial = Bool(has_initial_state.raw_load(b))
        if use_initial:
            # Read initial state from ssm_pool[slot, h, p, n].
            comptime for n in range(DSTATE):
                var off = UInt32(
                    slot * ssm_pool_strides[0]
                    + h * ssm_pool_strides[1]
                    + p * ssm_pool_strides[2]
                    + n * ssm_pool_strides[3]
                )
                state[n] = ssm_pool.raw_load(off)

        for t in range(seq_len):
            var gt = seq_start + t

            var x_val = Scalar[kernel_dtype](
                x.raw_load(
                    UInt32(
                        gt * x_strides[0] + h * x_strides[1] + p * x_strides[2]
                    )
                )
            ).cast[.float32]()

            var dt_val = Scalar[kernel_dtype](
                dt.raw_load(UInt32(gt * dt_strides[0] + h * dt_strides[1]))
            ).cast[.float32]()
            if has_dt_bias:
                dt_val += dt_bias_val
            if dt_softplus_bool:
                dt_val = softplus(dt_val)

            var dA = exp2(A_val * dt_val)
            var dt_x = dt_val * x_val

            var B_vals = SIMD[.float32, MAX_DSTATE](0.0)
            var C_vals = SIMD[.float32, MAX_DSTATE](0.0)
            comptime for n in range(DSTATE):
                B_vals[n] = Scalar[kernel_dtype](
                    B.raw_load(
                        UInt32(
                            gt * B_strides[0]
                            + group_id * B_strides[1]
                            + n * B_strides[2]
                        )
                    )
                ).cast[.float32]()
                C_vals[n] = Scalar[kernel_dtype](
                    C.raw_load(
                        UInt32(
                            gt * C_strides[0]
                            + group_id * C_strides[1]
                            + n * C_strides[2]
                        )
                    )
                ).cast[.float32]()

            state = state * dA + B_vals * dt_x

            var y_val = (state * C_vals).reduce_add()
            if has_D:
                y_val += D_val * x_val

            y.raw_store(
                UInt32(gt * y_strides[0] + h * y_strides[1] + p * y_strides[2]),
                Scalar[kernel_dtype](y_val.cast[kernel_dtype]()),
            )

        # Write final state into ssm_pool at slot cache_indices[b].
        comptime for n in range(DSTATE):
            var off = UInt32(
                slot * ssm_pool_strides[0]
                + h * ssm_pool_strides[1]
                + p * ssm_pool_strides[2]
                + n * ssm_pool_strides[3]
            )
            ssm_pool.raw_store(off, state[n])

    sync_parallelize(worker, batch * nheads * head_dim, ctx)


def mamba2_ssd_chunk_scan_varlen_fwd_cpu[
    kernel_dtype: DType,
    DSTATE: Int,
](
    nheads: Int,
    head_dim: Int,
    ngroups: Int,
    nheads_ngroups_ratio: Int,
    batch: Int,
    dt_softplus: Int8,
    x: TileTensor[mut=False, kernel_dtype, ...],
    dt: TileTensor[mut=False, kernel_dtype, ...],
    A: TileTensor[mut=False, kernel_dtype, ...],
    B: TileTensor[mut=False, kernel_dtype, ...],
    C: TileTensor[mut=False, kernel_dtype, ...],
    D: TileTensor[mut=False, kernel_dtype, ...],
    dt_bias: TileTensor[mut=False, kernel_dtype, ...],
    initial_states: TileTensor[mut=False, .float32, ...],
    y: TileTensor[mut=True, kernel_dtype, ...],
    final_states: TileTensor[mut=True, .float32, ...],
    query_start_loc: TileTensor[mut=False, .int32, ...],
    has_initial_state: TileTensor[mut=False, .bool, ...],
    x_strides: Strides3D,
    dt_strides: Strides2D,
    A_strides: Strides1D,
    B_strides: Strides3D,
    C_strides: Strides3D,
    D_strides: Strides1D,
    dt_bias_strides: Strides1D,
    initial_states_strides: Strides4D,
    y_strides: Strides3D,
    final_states_strides: Strides4D,
    ctx: Optional[DeviceContext] = None,
):
    """CPU reference for the Mamba-2 SSD varlen prefill scan.

    This is the trusted reference for numerical-equivalence testing: it computes
    the same per-token recurrence in fp32. Parallelised over `(b, h, p)`.
    """
    var has_D = Int(D.dim[0]()) > 0
    var has_dt_bias = Int(dt_bias.dim[0]()) > 0
    var has_init_tensor = Int(has_initial_state.dim[0]()) > 0
    var dt_softplus_bool = Bool(Int(dt_softplus) != 0)

    def worker(idx: Int) {imm}:
        var b, remaining = divmod(idx, nheads * head_dim)
        var h, p = divmod(remaining, head_dim)

        var group_id = h // nheads_ngroups_ratio

        var seq_start = Int(query_start_loc.raw_load(b))
        var seq_end = Int(query_start_loc.raw_load(b + 1))
        var seq_len = seq_end - seq_start
        if seq_len <= 0:
            return

        var A_val = (
            Scalar[kernel_dtype](A.raw_load(UInt32(h * A_strides[0]))).cast[
                DType.float32
            ]()
            * LOG2E
        )

        var dt_bias_val = Float32(0.0)
        if has_dt_bias:
            dt_bias_val = Scalar[kernel_dtype](
                dt_bias.raw_load(UInt32(h * dt_bias_strides[0]))
            ).cast[.float32]()

        var D_val = Float32(0.0)
        if has_D:
            D_val = Scalar[kernel_dtype](
                D.raw_load(UInt32(h * D_strides[0]))
            ).cast[.float32]()

        var state = SIMD[.float32, MAX_DSTATE](0.0)
        var use_initial = False
        if has_init_tensor:
            use_initial = Bool(has_initial_state.raw_load(b))
        if use_initial:
            comptime for n in range(DSTATE):
                var off = UInt32(
                    b * initial_states_strides[0]
                    + h * initial_states_strides[1]
                    + p * initial_states_strides[2]
                    + n * initial_states_strides[3]
                )
                state[n] = initial_states.raw_load(off)

        for t in range(seq_len):
            var gt = seq_start + t

            var x_val = Scalar[kernel_dtype](
                x.raw_load(
                    UInt32(
                        gt * x_strides[0] + h * x_strides[1] + p * x_strides[2]
                    )
                )
            ).cast[.float32]()

            var dt_val = Scalar[kernel_dtype](
                dt.raw_load(UInt32(gt * dt_strides[0] + h * dt_strides[1]))
            ).cast[.float32]()
            if has_dt_bias:
                dt_val += dt_bias_val
            if dt_softplus_bool:
                dt_val = softplus(dt_val)

            var dA = exp2(A_val * dt_val)
            var dt_x = dt_val * x_val

            var B_vals = SIMD[.float32, MAX_DSTATE](0.0)
            var C_vals = SIMD[.float32, MAX_DSTATE](0.0)
            comptime for n in range(DSTATE):
                B_vals[n] = Scalar[kernel_dtype](
                    B.raw_load(
                        UInt32(
                            gt * B_strides[0]
                            + group_id * B_strides[1]
                            + n * B_strides[2]
                        )
                    )
                ).cast[.float32]()
                C_vals[n] = Scalar[kernel_dtype](
                    C.raw_load(
                        UInt32(
                            gt * C_strides[0]
                            + group_id * C_strides[1]
                            + n * C_strides[2]
                        )
                    )
                ).cast[.float32]()

            state = state * dA + B_vals * dt_x

            var y_val = (state * C_vals).reduce_add()
            if has_D:
                y_val += D_val * x_val

            y.raw_store(
                UInt32(gt * y_strides[0] + h * y_strides[1] + p * y_strides[2]),
                Scalar[kernel_dtype](y_val.cast[kernel_dtype]()),
            )

        comptime for n in range(DSTATE):
            var off = UInt32(
                b * final_states_strides[0]
                + h * final_states_strides[1]
                + p * final_states_strides[2]
                + n * final_states_strides[3]
            )
            final_states.raw_store(off, state[n])

    sync_parallelize(worker, batch * nheads * head_dim, ctx)
