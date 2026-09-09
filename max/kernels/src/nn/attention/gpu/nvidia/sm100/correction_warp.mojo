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
"""Correction warp group logic for FA4 (SM100 Flash Attention)."""

from std.sys import size_of
from max.gpu import thread_idx
from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_ld,
    tcgen05_st,
    tcgen05_store_wait,
    tcgen05_fence_before,
)
from max.gpu.primitives.warp import _vote_nvidia_helper
from max.gpu.sync import umma_arrive_leader_cta
from linalg.matmul.gpu.sm100_structured.structured_kernels.tmem import (
    TmemAddress,
)
from nn.attention.gpu.nvidia.sm100.attention import FA4Config
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    mul_ftz,
    splitk_window,
    splitk_partition_idx,
    splitk_num_partitions,
)
from nn.attention.mha_mask import MHAMask
from .smem import SM100AttentionSMem


@always_inline
def fa4_correction[
    qkv_dtype: DType,
    rope_dtype_: Optional[DType],
    scale_dtype_: Optional[DType],
    MaskType: MHAMask,
    //,
    config: FA4Config[
        qkv_dtype, rope_dtype_=rope_dtype_, scale_dtype_=scale_dtype_
    ],
    page_size: Int,
    # Workspace (traditional/unfused) split-K: window the KV by a RUNTIME
    # partition count even at `config.splitk_partitions == 1`. Defaulted so
    # every non-workspace caller (incl. MLA) is byte-identical.
    workspace_split: Bool = False,
](
    smem: SM100AttentionSMem[config],
    tmem_addr: UInt32,
    seq_id: UInt32,
    score_row: UInt32,
    num_keys: UInt32,
    mask: MaskType,
    ws_num_partitions: UInt32 = 1,
):
    comptime accum_type = DType.float32
    comptime assert size_of[accum_type]() == 4
    comptime BN = config.BN

    var mbars = smem.misc_mbars()

    # Dummy arrives for the prologue iteration (no previous O to protect).
    # This satisfies the combined barrier's correction half for the first P@V.
    # Cluster-scope arrive so both CTAs' signals reach the leader.
    comptime if config.pair_cta:
        umma_arrive_leader_cta(mbars.combined_p_o_consumer(0))
        umma_arrive_leader_cta(mbars.combined_p_o_consumer(1))
    else:
        _ = mbars.combined_p_o_consumer(0)[].arrive()
        _ = mbars.combined_p_o_consumer(1)[].arrive()

    # `tmem_addr` passed in by register (read once post-barrier in the kernel
    # prologue); do NOT re-read `smem.tmem_addr_ptr()` here.
    var o0_tmem = TmemAddress(tmem_addr + UInt32(config.TMEM_O0))
    var o1_tmem = TmemAddress(tmem_addr + UInt32(config.TMEM_O1))
    var correction_smem_arg = smem.correction_smem()

    var pipeline_c0 = mbars.consumer_c0()
    var pipeline_c1 = mbars.consumer_c1()

    comptime BM_mask: Int = config.PairBM_eff()
    comptime num_q: Int = config.num_q

    # Pure O-rescale arithmetic (online-softmax correction of a previously
    # accumulated O tile by `c`). Depends only on `o_tmem`, `c_pair` and
    # comptime `config.correction_o_cols()` — no pipeline / warp-group state —
    # so it is shared verbatim by BOTH the single-O loop below and the two-WG
    # `_correction_step` closure further down. `@always_inline` => inlining
    # it back into either caller reproduces the identical instruction stream
    # (2-O / DeepSeek / MHA codegen is unchanged).
    #
    # Walks `correction_o_cols()` -- the PHYSICAL extent of one O accumulator,
    # not the logical depth. See its docstring: under shared-key the logical
    # depth runs 4x past the accumulator and silently corrupts.
    @__parameter
    @always_inline
    def _rescale_o(o_tmem: TmemAddress, c_pair: SIMD[.float32, 2]):
        comptime o_cols = config.correction_o_cols()
        comptime batch_size = 16 if o_cols % 16 == 0 else 8
        comptime assert o_cols % batch_size == 0
        # output is BM x depth
        comptime load_iters, load_remainder = divmod(o_cols, 2 * batch_size)
        comptime assert load_iters > 1
        comptime assert (load_remainder == batch_size) or (load_remainder == 0)

        var o_b0 = tcgen05_ld[
            datapaths=32,
            bits=32,
            repeat=batch_size,
            dtype=accum_type,
            pack=False,
            width=batch_size,
        ](o_tmem.addr)

        comptime for b in range(load_iters):
            comptime b0_offset0 = 2 * b * batch_size
            comptime b1_offset = b0_offset0 + batch_size
            comptime b0_offset1 = b1_offset + batch_size
            var o_b1 = tcgen05_ld[
                datapaths=32,
                bits=32,
                repeat=batch_size,
                dtype=accum_type,
                pack=False,
                width=batch_size,
            ]((o_tmem + b1_offset).addr)
            var o_b0_scaled = Array[Scalar[accum_type], batch_size](
                uninitialized=True
            )

            comptime for _i in range(0, batch_size, 2):
                var pair = mul_ftz(
                    SIMD[.float32, 2](o_b0[_i], o_b0[_i + 1]),
                    c_pair,
                )
                o_b0_scaled[_i] = pair[0]
                o_b0_scaled[_i + 1] = pair[1]
            tcgen05_st[datapaths=32, bits=32, repeat=batch_size, pack=False](
                (o_tmem + b0_offset0).addr, o_b0_scaled
            )

            comptime if b0_offset1 + batch_size <= o_cols:
                o_b0 = tcgen05_ld[
                    datapaths=32,
                    bits=32,
                    repeat=batch_size,
                    dtype=accum_type,
                    pack=False,
                    width=batch_size,
                ]((o_tmem + b0_offset1).addr)
            var o_b1_scaled = Array[Scalar[accum_type], batch_size](
                uninitialized=True
            )

            comptime for _i in range(0, batch_size, 2):
                var pair = mul_ftz(
                    SIMD[.float32, 2](o_b1[_i], o_b1[_i + 1]),
                    c_pair,
                )
                o_b1_scaled[_i] = pair[0]
                o_b1_scaled[_i + 1] = pair[1]
            tcgen05_st[datapaths=32, bits=32, repeat=batch_size, pack=False](
                (o_tmem + b1_offset).addr, o_b1_scaled
            )

        comptime if load_remainder > 0:
            comptime offset = 2 * batch_size * load_iters
            var o_b0_scaled_rem = Array[Scalar[accum_type], load_remainder](
                uninitialized=True
            )

            comptime for _i in range(0, load_remainder, 2):
                var pair = mul_ftz(
                    SIMD[.float32, 2](o_b0[_i], o_b0[_i + 1]),
                    c_pair,
                )
                o_b0_scaled_rem[_i] = pair[0]
                o_b0_scaled_rem[_i + 1] = pair[1]
            tcgen05_st[
                datapaths=32,
                bits=32,
                repeat=load_remainder,
                pack=False,
            ]((o_tmem + offset).addr, o_b0_scaled_rem)
        tcgen05_store_wait()
        tcgen05_fence_before()

    var change: Bool

    # ---- single-O correction (1Q wide-V) ----
    # Wide V forces single-O (aliased O0), so only WG0 runs: it folds EVERY
    # K-tile serially into the single O0 (WG1 no-op). The correction thus
    # (a) waits on WG0's O-producer barrier ONLY, via a ONE-stage pipeline
    # (`consumer_o0`) — the standard two-stage `consumer_o` would alternate
    # onto WG1's never-produced O barrier and deadlock — and (b) runs
    # `total_iters - 1` c0-only steps (the online-softmax rescale between
    # tiles; after WG0's softmax peel, one per remaining tile), with zero c1
    # steps. It shares only the pure `_rescale_o` arithmetic with the two-WG
    # path; the pipeline / step-count / control flow is self-contained and
    # the two-WG code below is untouched (byte-identical for MHA / DeepSeek /
    # 2-O and standard 1Q).
    comptime if config.single_o:
        var o0_tmem_so = TmemAddress(tmem_addr + UInt32(config.TMEM_O0))
        var pipeline_c0_so = mbars.consumer_c0()
        var pipeline_o_so = mbars.consumer_o0()
        var correction_smem_so = correction_smem_arg + UInt32(
            thread_idx.x
        ) % UInt32(WARPGROUP_SIZE)

        var t_left_so: UInt32 = (
            mask.total_iters[BM_mask, BN, page_size](
                seq_id, score_row, num_keys
            )
            - 1
        )
        while t_left_so != 0:
            t_left_so -= 1
            pipeline_c0_so.wait()
            var c_scalar = correction_smem_so[0]
            change = _vote_nvidia_helper(c_scalar < 1.0) != 0
            pipeline_o_so.wait()
            if change:
                _rescale_o(o0_tmem_so, SIMD[.float32, 2](c_scalar, c_scalar))

            pipeline_o_so.release()
            pipeline_c0_so.release()
        return

    # Two-WG (2Q and standard 1Q) O consumer — unchanged.
    var pipeline_o = mbars.consumer_o()

    # Per-WG main-loop commit counts (= softmax's main-loop iters after
    # peel, = MMA's post-peel P@V commits per side).
    # 2Q: both WGs share all T K-tiles, so c0 commits = c1 commits = T - 1.
    # 1Q: WG0 owns ceil(T/2) K-tiles, WG1 owns floor(T/2). After each
    #     WG's softmax peel consumes one, c0_iters = ceil(T/2) - 1 and
    #     c1_iters = max(0, floor(T/2) - 1). They differ by 0 (even T) or
    #     1 (odd T, WG0 takes the extra). c0_iters >= c1_iters always.
    var total_iters_runtime: UInt32 = mask.total_iters[BM_mask, BN, page_size](
        seq_id, score_row, num_keys
    )
    # Split-K (1Q): slice the combined tile count to this partition's window
    # before the per-WG c0/c1 split. Identical window to the other warps
    # (total_iters == last_masked_set_end for check_mask==False masks).
    comptime if config.num_q == 1 and (
        config.splitk_partitions > 1 or workspace_split
    ):
        var _np = splitk_num_partitions[config](ws_num_partitions)
        var _w = splitk_window(
            total_iters_runtime,
            _np,
            splitk_partition_idx(_np),
        )
        total_iters_runtime = _w[1] - _w[0]
        # Empty partition (front-load trailing window, T < num_partitions; or an
        # M6 idle CTA): no tiles. Return before the c0/c1 split -- at 0 tiles
        # `c0_iters = (0+1)//2 - 1` underflows to a huge UInt32, so the extra-c0
        # tail would spin on `pipeline_c0/o.wait()` the producer never posts.
        # softmax stages a neutral identity; terminal `cluster_sync()` still runs.
        if total_iters_runtime == 0:
            return
    # All-masked row (valid_length 0): total_iters == 0, so `c0_iters` underflows
    # and this warp waits on a tile the producer never posts. Split-K is guarded
    # above; guard the non-split path here.
    if total_iters_runtime == 0:
        return
    var c0_iters: UInt32
    var c1_iters: UInt32
    comptime if num_q == 1:
        c0_iters = (total_iters_runtime + 1) // 2 - 1
        var t_floor: UInt32 = total_iters_runtime // 2
        c1_iters = t_floor - 1 if t_floor > 0 else 0
    else:
        c0_iters = total_iters_runtime - 1
        c1_iters = total_iters_runtime - 1
    # Main loop iterates min(c0, c1) = c1; extra c0 iter(s) handled
    # after (0 for 2Q, 0 or 1 for 1Q).
    var iter_count: UInt32 = c1_iters
    var extra_c0_iters: UInt32 = c0_iters - c1_iters

    var correction_smem_0 = correction_smem_arg + UInt32(thread_idx.x) % UInt32(
        WARPGROUP_SIZE
    )
    var correction_smem_1 = correction_smem_0 + UInt32(WARPGROUP_SIZE)

    # The per-(c0 or c1) correction step. Inlined twice per outer iter
    # of the main loop (i=0 then i=1) for the WG0+WG1-paired tail, and
    # once more after the main loop for any extra c0-only iter (1Q
    # odd-T case where WG0 has one more main-loop commit than WG1).
    @__parameter
    @always_inline
    def _correction_step[i: Int]():
        # correct
        var c_scalar: Scalar[accum_type]

        comptime if i == 0:
            pipeline_c0.wait()
            c_scalar = correction_smem_0[0]
        else:
            pipeline_c1.wait()
            c_scalar = correction_smem_1[0]

        change = _vote_nvidia_helper(c_scalar < 1.0) != 0
        pipeline_o.wait()
        if change:
            var o_tmem: TmemAddress

            comptime if i == 0:
                o_tmem = o0_tmem
            else:
                o_tmem = o1_tmem

            _rescale_o(o_tmem, SIMD[.float32, 2](c_scalar, c_scalar))

        comptime if config.pair_cta:
            umma_arrive_leader_cta(pipeline_o.consumer_mbar())
            pipeline_o.step()
        else:
            pipeline_o.release()

        comptime if i == 0:
            pipeline_c0.release()
        else:
            pipeline_c1.release()

    while iter_count != 0:
        iter_count -= 1
        _correction_step[0]()
        _correction_step[1]()

    # 1Q odd-T: WG0 has one more main-loop commit than WG1. Run the
    # c0-only step to consume it. In 2Q (and 1Q even T) extra_c0_iters
    # is 0 and this loop is a no-op. The pipeline_o state after the
    # main loop is back at stage 0 (consumer_sub_stages=2 → wraps
    # every pair), so the next pipeline_o.wait inside `_correction_step[0]`
    # targets the o0 producer mbar as required.
    while extra_c0_iters != 0:
        extra_c0_iters -= 1
        _correction_step[0]()
