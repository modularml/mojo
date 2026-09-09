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

"""Fused reduce-scatter+RMSNorm -> GEMM benchmark for PDL overlap (bf16).

PDL makes the GEMM resident while the collective is still running, so the two
kernels genuinely overlap and neither can be timed on its own. Both are therefore
enqueued in one closure on one `DeviceContext` -- one stream -- and the reported
figure is the average per-iteration stream wall-clock across both.
`producer_only` / `consumer_only` exist for attribution only; summing them
double-counts the overlap.

Four configurations, identical in shape and buffers and differing only in PDL
wiring: neither side; consumer only; both; and both plus the consumer's pre-wait
weight prefetch. Consumer-only is what main runs, since the SM100 dispatcher
already passes `PDLLevel.ON` there, so it is the baseline to beat -- measuring
against "neither" would credit this change with overlap main already had.

Iterations run back-to-back, so a GEMM also overlaps the next iteration's
collective. That makes the result a steady-state pipelined figure rather than an
isolated pair, and a real decode step separates the two by EP dispatch and
attention -- so read the deltas as the mechanism's headroom, not as a predicted
end-to-end gain.

The consumer config is built here rather than read from the tuning table, which
enables weight prefetch nowhere. Weights stream through `CacheBustingBuffer` so
they are read cold; L2-resident weights would hide the very cost the prefetch
targets, and the window count is printed so a collapsed buffer cannot pass
unnoticed.
"""

from std.sys import (
    get_defined_dtype,
    get_defined_int,
    simd_width_of,
    size_of,
)

from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.random import rand

from max.benchmark import bench_multicontext, bencher_iter_custom
from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.primitives.grid_controls import PDLLevel

from comm import Signal, MAX_GPUS, group_start, group_end
from comm.reducescatter import ReduceScatterConfig
from comm.reducescatter_rmsnorm import reducescatter_rmsnorm
from comm.sync import enable_p2p, init_signal_buffer, is_p2p_enabled

from internal_utils import CacheBustingBuffer, arg_parse
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
)
from std.utils.index import Index


def bench_rs_norm_gemm_pdl[
    in_dtype: DType,
    ngpus: Int,
    num_cols: Int,
    gemm_n: Int,
    bm: Int = 64,
    bn: Int = 64,
    prefetch_tiles_n: Int = 2,
](num_rows: Int, mut b: Bench, list_of_ctx: List[DeviceContext]) raises:
    """Bench the fused RS+RMSNorm -> GEMM pair across the PDL configurations.

    Parameters:
        in_dtype: Element type of the collective and the GEMM (bf16).
        ngpus: Number of participating GPUs.
        num_cols: Hidden size; the collective's column count and the GEMM's K.
        gemm_n: N of the consumer GEMM (static, as the SM100 kernel requires).
        bm: Consumer GEMM block tile M.
        bn: Consumer GEMM block tile N.
        prefetch_tiles_n: Weight tiles the consumer issues before its PDL
            wait. Must not exceed the config's group pipeline stages.

    Args:
        num_rows: Total (pre-scatter) row count; each rank owns `num_rows/ngpus`.
        b: Benchmark harness.
        list_of_ctx: One context per participating GPU.
    """
    comptime assert (
        in_dtype == .bfloat16
    ), "the fused RS+RMSNorm producer is bf16-only"

    comptime simd_size = simd_width_of[in_dtype, target=get_gpu_target()]()
    var config = ReduceScatterConfig[in_dtype, ngpus](
        axis_size=num_rows, unit_numel=num_cols, threads_per_gpu=0
    )
    var length = num_rows * num_cols
    var epsilon = Float32(1e-6)
    # 1.0 matches the production norm this fusion replaces (Gemma-style offset).
    var weight_offset = Scalar[in_dtype](1.0)

    # --- Consumer GEMM config. cluster (2,1,1) is the minimal cta_group=2 pair,
    # so the grid stays legal at the small per-rank M this bench targets; a 4x4
    # cluster would need a grid the decode shard cannot fill. ---
    comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
    comptime BK = swizzle.bytes() // size_of[in_dtype]()
    comptime MMA_K = 16
    comptime block_tile_shape = Index(bm, bn, BK)
    comptime mma_shape = Index(2 * bm, 2 * bn, MMA_K)

    comptime gemm_config_off = MatmulConfig[in_dtype, in_dtype, in_dtype, True](
        cluster_shape=Index(2, 1, 1),
        mma_shape=mma_shape,
        cta_group=2,
    )
    comptime gemm_config_prefetch = MatmulConfig[
        in_dtype, in_dtype, in_dtype, True
    ](
        cluster_shape=Index(2, 1, 1),
        mma_shape=mma_shape,
        cta_group=2,
        prefetch_tiles_n=prefetch_tiles_n,
    )

    # --- Per-GPU buffers, all allocated before any timing. ---
    var cb_inputs = List[CacheBustingBuffer[in_dtype]]()
    var cb_weights = List[CacheBustingBuffer[in_dtype]]()
    var host_bufs = List[List[Scalar[in_dtype]]](capacity=ngpus)
    var weight_hosts = List[List[Scalar[in_dtype]]](capacity=ngpus)
    var normed = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var fused_sum = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var gamma_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var c_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var c_ref_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    var weight_elems = gemm_n * num_cols
    var gamma_host = List(length=num_cols, fill=Scalar[in_dtype](0))
    for c in range(num_cols):
        gamma_host[c] = (Float64(c + num_cols) / Float64(num_cols)).cast[
            in_dtype
        ]()

    var w_host = List[Scalar[in_dtype]](
        length=weight_elems, fill=Scalar[in_dtype](0)
    )
    rand(w_host.unsafe_ptr(), weight_elems)

    for i in range(ngpus):
        cb_inputs.append(
            CacheBustingBuffer[in_dtype](
                length, simd_size, list_of_ctx[i], True
            )
        )
        cb_weights.append(
            CacheBustingBuffer[in_dtype](
                weight_elems, simd_size, list_of_ctx[i], True
            )
        )

        # Fill the WHOLE cache-busting allocation, not just the first window: an
        # iteration landing on an unwritten window would read garbage, and NaNs
        # there make the bit-identity gate below vacuous.
        var h = List[Scalar[in_dtype]](
            unsafe_uninit_length=cb_inputs[i].alloc_size()
        )
        for j in range(cb_inputs[i].alloc_size()):
            h[j] = Scalar[in_dtype](i + 1) + Scalar[in_dtype](j % 251)
        list_of_ctx[i].enqueue_copy(cb_inputs[i].device_buffer(), h)
        host_bufs.append(h^)

        var wh = List[Scalar[in_dtype]](
            unsafe_uninit_length=cb_weights[i].alloc_size()
        )
        for j in range(cb_weights[i].alloc_size()):
            wh[j] = w_host[j % weight_elems]
        list_of_ctx[i].enqueue_copy(cb_weights[i].device_buffer(), wh)
        weight_hosts.append(wh^)

        if i == 0:
            print(
                "cache-busting windows: inputs=",
                cb_inputs[i].buffer_size // cb_inputs[i].stride,
                " weights=",
                cb_weights[i].buffer_size // cb_weights[i].stride,
                " (1 means the buster COLLAPSED -- raise budget_bytes)",
                sep="",
            )

        var alloc_i = config.rank_num_elements(i)
        if alloc_i < 1:
            alloc_i = 1
        normed.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i))
        fused_sum.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i)
        )

        gamma_dev.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](num_cols)
        )
        list_of_ctx[i].enqueue_copy(gamma_dev[i], gamma_host)

        var c_elems = config.rank_units(i) * gemm_n
        if c_elems < 1:
            c_elems = 1
        c_dev.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](c_elems))
        c_ref_dev.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](c_elems)
        )

        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[.uint8](size_of[Signal]())
        )
        rank_sigs[i] = (
            signal_buffers[i]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )

    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Producer views are fully dynamic (it takes `...`-generic TileTensors); the
    # consumer's operands are built with `Idx[...]` further down because the
    # SM100 kernel needs static N and K.
    comptime InTensorType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0, num_cols)))), ImmutAnyOrigin
    ]
    comptime OutShardType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0, num_cols)))), MutAnyOrigin
    ]
    comptime GammaShardType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0)))), ImmutAnyOrigin
    ]
    # `[N, K]` (transpose_b) with BOTH dims static, as the SM100 kernel needs.
    # Built from the rotating cache-bust window rather than the base pointer so
    # each iteration streams cold weights -- the cost the prefetch must hide.
    comptime WeightType = TileTensor[
        in_dtype,
        type_of(row_major(Coord(Idx[gemm_n], Idx[num_cols]))),
        ImmutAnyOrigin,
    ]
    var in_bufs = Array[InTensorType, ngpus](uninitialized=True)
    var normed_shards = Array[OutShardType, ngpus](
        fill_with=lambda (i: Int) {
            ref normed, imm config
        } -> OutShardType: OutShardType(
            normed[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )
    )
    var sum_shards = Array[OutShardType, ngpus](
        fill_with=lambda (i: Int) {
            ref fused_sum, imm config
        } -> OutShardType: OutShardType(
            fused_sum[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )
    )
    var gamma_shards = Array[GammaShardType, ngpus](
        fill_with=lambda (i: Int) -> GammaShardType: GammaShardType(
            rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                gamma_dev[i].unsafe_ptr()
            ),
            row_major(Coord(Index(num_cols))),
        )
    )

    var bench_name_prefix = String(
        "M=",
        num_rows,
        "/H=",
        num_cols,
        "/N=",
        gemm_n,
        "/TP",
        ngpus,
    )
    # Producer moves the full stream in and a shard out; consumer streams the
    # weights. Weights dominate at decode M, which is the point of the overlap.
    var total_bytes = (
        length * ngpus * size_of[in_dtype]()
        + weight_elems * size_of[in_dtype]()
    )

    # --- Integrity gate, before any timing: PDL + weight prefetch reorder
    # loads, never arithmetic, so the GEMM output must be BIT-identical between
    # PDL off and PDL+prefetch on. If this ever diverges, the overlap
    # is racing the producer's stores and every timing below is meaningless. ---
    comptime for verify_variant in range(2):
        comptime v_pdl = PDLLevel.OFF if verify_variant == 0 else PDLLevel.ON
        comptime v_config = (
            gemm_config_off if verify_variant == 0 else gemm_config_prefetch
        )
        # The collective must be enqueued on EVERY rank before anything that can
        # block the host, or its barrier can never resolve. The matmul launcher
        # builds TMA descriptors, so issuing it for rank 0 in this same loop
        # would sit between rank 0's and rank 1's collective launches and
        # deadlock the barrier (observed: GPU0 spinning at 100%, GPU1 idle).
        # Timing does not care -- `bench_multicontext` gives each rank its own
        # host thread -- but this single-threaded verify pass does, so the two
        # kernels are phase-separated here. Adjacency is irrelevant to a
        # value check.
        group_start()
        for i in range(ngpus):
            comptime for _j in range(ngpus):
                in_bufs[_j] = InTensorType(
                    rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                        cb_inputs[_j].offset_ptr(0)
                    ),
                    row_major(Coord(Index(num_rows, num_cols))),
                )
            reducescatter_rmsnorm[pdl_level=v_pdl](
                in_bufs,
                normed_shards[i],
                sum_shards[i],
                gamma_shards[i],
                epsilon,
                weight_offset,
                rank_sigs,
                list_of_ctx[i],
            )
        group_end()
        for i in range(ngpus):
            list_of_ctx[i].synchronize()

        for i in range(ngpus):
            var local_rows = config.rank_units(i)
            if local_rows > 0:
                blackwell_matmul_tma_umma_warp_specialized[
                    transpose_b=True, config=v_config, pdl_level=v_pdl
                ](
                    TileTensor(
                        c_dev[i] if verify_variant == 1 else c_ref_dev[i],
                        row_major(Coord(local_rows, Idx[gemm_n])),
                    ),
                    TileTensor(
                        normed[i],
                        row_major(Coord(local_rows, Idx[num_cols])),
                    ),
                    WeightType(
                        rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                            cb_weights[i].offset_ptr(0)
                        ),
                        row_major(Coord(Idx[gemm_n], Idx[num_cols])),
                    ),
                    list_of_ctx[i],
                )
        for i in range(ngpus):
            list_of_ctx[i].synchronize()

    var total_mismatch = 0
    for i in range(ngpus):
        var n_elems = config.rank_units(i) * gemm_n
        if n_elems == 0:
            continue
        var got = List[Scalar[in_dtype]](unsafe_uninit_length=n_elems)
        var want = List[Scalar[in_dtype]](unsafe_uninit_length=n_elems)
        list_of_ctx[i].enqueue_copy(got, c_dev[i])
        list_of_ctx[i].enqueue_copy(want, c_ref_dev[i])
        list_of_ctx[i].synchronize()
        for j in range(n_elems):
            if got[j] != want[j]:
                total_mismatch += 1
        _ = got^
        _ = want^

    if total_mismatch != 0:
        raise Error(
            String(
                "PDL+prefetch changed the GEMM result: ",
                total_mismatch,
                (
                    " mismatching elements with PDL off. The consumer is"
                    " reading the producer's output before it is written -- do"
                    " NOT trust any timing from this run."
                ),
            )
        )
    print("verified: PDL+prefetch GEMM output bit-identical to PDL-off")

    # 0: neither  1: consumer only (what main runs)  2: both
    # 3: both + weight prefetch  4/5: one kernel alone, for attribution
    comptime for variant in range(6):
        comptime prod_pdl = PDLLevel.OFF if variant <= 1 else PDLLevel.ON
        comptime cons_pdl = PDLLevel.OFF if variant == 0 else PDLLevel.ON
        comptime use_prefetch = variant == 3
        comptime producer_only = variant == 4
        comptime consumer_only = variant == 5
        comptime variant_name = (
            "pair_none" if variant
            == 0 else "pair_baseline_today" if variant
            == 1 else "pair_producer_pdl" if variant
            == 2 else "pair_producer_pdl_prefetch" if variant
            == 3 else "producer_only" if variant
            == 4 else "consumer_only"
        )

        @always_inline
        def bench_pair_iter(
            mut bench: Bencher, ctx: DeviceContext, ctx_idx: Int
        ) raises {mut in_bufs, imm}:
            var local_rows = config.rank_units(ctx_idx)

            @always_inline
            def call_fn(
                ctx_inner: DeviceContext, cache_iter: Int
            ) raises {mut in_bufs, imm}:
                comptime if not consumer_only:
                    comptime for _j in range(ngpus):
                        in_bufs[_j] = InTensorType(
                            rebind[
                                ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]
                            ](cb_inputs[_j].offset_ptr(cache_iter)),
                            row_major(Coord(Index(num_rows, num_cols))),
                        )
                    reducescatter_rmsnorm[pdl_level=prod_pdl](
                        in_bufs,
                        normed_shards[ctx_idx],
                        sum_shards[ctx_idx],
                        gamma_shards[ctx_idx],
                        epsilon,
                        weight_offset,
                        rank_sigs,
                        ctx_inner,
                    )

                comptime if not producer_only:
                    # A 0-row rank (ragged M) owns no shard, so there is no GEMM
                    # to run and a 0-M grid would be rejected at launch.
                    if local_rows > 0:
                        comptime gemm_config = (
                            gemm_config_prefetch if use_prefetch else gemm_config_off
                        )
                        blackwell_matmul_tma_umma_warp_specialized[
                            transpose_b=True,
                            config=gemm_config,
                            pdl_level=cons_pdl,
                        ](
                            TileTensor(
                                c_dev[ctx_idx],
                                row_major(Coord(local_rows, Idx[gemm_n])),
                            ),
                            TileTensor(
                                normed[ctx_idx],
                                row_major(Coord(local_rows, Idx[num_cols])),
                            ),
                            WeightType(
                                rebind[
                                    ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]
                                ](cb_weights[ctx_idx].offset_ptr(cache_iter)),
                                row_major(Coord(Idx[gemm_n], Idx[num_cols])),
                            ),
                            ctx_inner,
                        )

            bencher_iter_custom(bench, call_fn, ctx)

        bench_multicontext(
            b,
            bench_pair_iter,
            list_of_ctx,
            BenchId(variant_name, input_id=bench_name_prefix),
            [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
        )

    b.dump_report()
    _ = cb_inputs^
    _ = cb_weights^
    _ = normed^
    _ = fused_sum^
    _ = gamma_dev^
    _ = c_dev^
    _ = c_ref_dev^
    _ = signal_buffers^


def main() raises:
    comptime in_dtype = get_defined_dtype["dtype", .bfloat16]()
    comptime num_cols = get_defined_int["num_cols", 6144]()
    comptime gemm_n = get_defined_int["gemm_n", 4096]()
    comptime bm = get_defined_int["bm", 64]()
    comptime bn = get_defined_int["bn", 64]()
    comptime prefetch_tiles_n = get_defined_int["prefetch_tiles_n", 2]()
    comptime num_gpus = get_defined_int["num_gpus", 2]()

    var num_rows = Int(arg_parse("num_rows", 8))
    # >1 to get run-to-run spread: the deltas here are single-digit percent,
    # which one repetition cannot separate from box noise.
    var num_reps = Int(arg_parse("num_reps", 1))

    if DeviceContext.number_of_devices() < num_gpus:
        print("Not enough GPUs, skipping benchmark.")
        return

    _ = enable_p2p()
    if not is_p2p_enabled():
        print("P2P not enabled, skipping benchmark.")
        return

    var list_of_ctx = List[DeviceContext]()
    for i in range(num_gpus):
        list_of_ctx.append(DeviceContext(device_id=i))

    print(
        "Paired fused reduce-scatter+RMSNorm -> GEMM, PDL: ",
        num_gpus,
        " GPUs, ",
        in_dtype,
        ", M=",
        num_rows,
        " H=",
        num_cols,
        " N=",
        gemm_n,
        sep="",
    )

    var m = Bench(BenchConfig(num_repetitions=num_reps))
    bench_rs_norm_gemm_pdl[
        in_dtype,
        num_gpus,
        num_cols,
        gemm_n,
        bm=bm,
        bn=bn,
        prefetch_tiles_n=prefetch_tiles_n,
    ](num_rows, m, list_of_ctx)
