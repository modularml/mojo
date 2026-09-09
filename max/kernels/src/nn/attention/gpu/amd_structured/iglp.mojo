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
"""IGroupLP `sched_group_barrier` aggregate-pair helpers for AMD MHA.

Comptime-recursive expansions of the reference's `sched_barrier_pairs<Pairs,
VALU_CNT, Group>` and `sched_barrier_exp_pairs<...>` C++ templates
(see the reference `attn/gqa_causal/kernel.cpp:44-56`).

These helpers prescribe IGroupLP groupings to LLVM's AMDGPU instruction
scheduler via the `llvm.amdgcn.sched.group.barrier` intrinsic. They were
defined identically across several attention kernels and pulled here to a
shared module: one place to fix as the language evolves, no duplication.
Current consumers: `mha_prefill_v2`, `mla_prefill`, `mla_prefill_v2`,
`mla_components`.

Per-kernel hint-pair parameters (which N, M for QK / PV / EXP cluster
types) are tuned via parameter sweep at the kernel. Only the helper
expansion logic is shared; the per-cluster `(N, M)` defaults belong with
each kernel (they're shape-dependent and kernel-specific).
"""

from std.sys import llvm_intrinsic

from max.gpu.sync import AMDScheduleBarrierMask, schedule_group_barrier


@fieldwise_init
struct AMDIGLPStrategy(Equatable, Intable, TrivialRegisterPassable):
    """Preset strategy values for the `llvm.amdgcn.iglp.opt` intrinsic.

    LLVM AMDGPU defines these as named presets in `AMDGPUIGroupLP.cpp`;
    the integer values are the second argument to the intrinsic. All
    presets assume MFMAs are present in the region; using on a
    non-MFMA region falls back gracefully but provides no constraints.

    Per AMD docs, mutually exclusive with `sched_group_barrier` in
    the same scheduling region; pick one or the other per cluster.
    """

    var _value: Int32

    comptime MFMA_SMALL_GEMM = Self(0)
    """`MFMASmallGemmOpt`: interleaves 2 DS reads per 1 MFMA."""

    comptime MFMA_SMALL_GEMM_SINGLE_WAVE = Self(1)
    """`MFMASmallGemmSingleWaveOpt`: single-wave variant for small GEMMs."""

    comptime MFMA_EXP_INTERLEAVE = Self(2)
    """`MFMAExpInterleaveOpt`: multi-phase attention preset (MFMA + `exp2`).

    Drives the MFMA/VALU/TRANS triple interleave that flash-attention-style
    softmax wants (used by `mla_prefill`).
    """

    comptime MFMA_EXP_SIMPLE_INTERLEAVE = Self(3)
    """`MFMAExpSimpleInterleaveOpt`: interleaves 1 TRANS per 1 MFMA."""

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return self._value != other._value

    def __int__(self) -> Int:
        return Int(self._value)


@always_inline
def _iglp_opt[strategy: AMDIGLPStrategy]() -> None:
    """Emits `llvm.amdgcn.iglp.opt(strategy)`: IGroupLP preset hint."""
    llvm_intrinsic["llvm.amdgcn.iglp.opt", NoneType](Int32(Int(strategy)))


@always_inline
@__parameter
def sched_barrier_pairs[pairs: Int, valu_cnt: Int, group: Int]() -> None:
    """Emits `pairs` schedule groups of shape `[1 MFMA, valu_cnt VALU]`.

    Each call expands to two `schedule_group_barrier` invocations under
    `sync_id=group`: one declaring "1 MFMA in this group" and one
    declaring "`valu_cnt` VALUs in this group". `pairs > 1` recurses to
    emit additional pairs in the same `sync_id`, all of which LLVM
    orders relative to each other.

    Note `valu_cnt` is the **count of VALU instructions in each group**,
    not a VALU-to-MFMA ratio; LLVM derives the interleave from the
    group declaration. Mojo's underlying
    `schedule_group_barrier(mask, size, sync_id)` docstring labels
    `size` as a "repeat count", that is misleading; it is the
    instruction count per group.

    Parameters:
        pairs: Number of (MFMA, VALU) groups to emit, all sharing
            `sync_id=group`.
        valu_cnt: VALU instructions per group.
        group: IGroupLP `sync_id`. Reuse within one cluster's pair
            sequence; pick distinct ids for distinct clusters so their
            constraints stay independent. 1-10 typical for MHA
            main+epilogue.
    """
    schedule_group_barrier(AMDScheduleBarrierMask.MFMA, Int32(1), Int32(group))
    schedule_group_barrier(
        AMDScheduleBarrierMask.VALU, Int32(valu_cnt), Int32(group)
    )
    comptime if pairs > 1:
        sched_barrier_pairs[pairs - 1, valu_cnt, group]()


@always_inline
@__parameter
def sched_dsread_valu_pairs[pairs: Int, valu_cnt: Int, group: Int]() -> None:
    """Emits `pairs` schedule groups of shape `[1 DS_READ, valu_cnt VALU]`.

    DS_READ variant of `sched_barrier_pairs` for clusters that have NO
    MFMAs but want to interleave LDS-reads with VALU work, typically
    V-load + causal-mask clusters (the MHA kernel's C5 / EPI_C5 / EPI_C9). The
    interleave hides the v_cmp→v_cndmask wait state (5 cycles, gated by
    `s_nop 1` if not filled) behind useful `ds_read_b64_tr_b16` work.

    Same `sync_id` rule as `sched_barrier_pairs`: pick a fresh group
    distinct from MFMA/VALU/TRANS pair sequences in other clusters.

    Parameters:
        pairs: Number of (DS_READ, VALU) groups to emit.
        valu_cnt: VALU instructions per group (typically 2 for one
            cmp+cndmask pair per ds_read).
        group: IGroupLP `sync_id`.
    """
    schedule_group_barrier(
        AMDScheduleBarrierMask.DS_READ, Int32(1), Int32(group)
    )
    schedule_group_barrier(
        AMDScheduleBarrierMask.VALU, Int32(valu_cnt), Int32(group)
    )
    comptime if pairs > 1:
        sched_dsread_valu_pairs[pairs - 1, valu_cnt, group]()


@always_inline
@__parameter
def sched_barrier_exp_pairs[pairs: Int, exp_cnt: Int, group: Int]() -> None:
    """Emits `pairs` schedule groups of shape `[1 MFMA, exp_cnt TRANS]`.

    TRANS variant of `sched_barrier_pairs` for `exp2` / softmax
    transcendental work that issues on the AMDGPU TRANS unit
    (mask `0x400` per LLVM AMDGPU). Pair this with
    `sched_barrier_pairs` under the same `sync_id` to declare both
    interleavings within one cluster; LLVM orders the declarations as
    a single sequence (see the reference
    `kernel.cpp:44-56` for the canonical pattern).

    Parameters:
        pairs: Number of (MFMA, TRANS) groups to emit.
        exp_cnt: TRANS instructions per group.
        group: IGroupLP `sync_id`, must match the companion
            `sched_barrier_pairs` call's `group` for combined ordering.
    """
    schedule_group_barrier(AMDScheduleBarrierMask.MFMA, Int32(1), Int32(group))
    schedule_group_barrier(
        AMDScheduleBarrierMask.TRANS, Int32(exp_cnt), Int32(group)
    )
    comptime if pairs > 1:
        sched_barrier_exp_pairs[pairs - 1, exp_cnt, group]()
