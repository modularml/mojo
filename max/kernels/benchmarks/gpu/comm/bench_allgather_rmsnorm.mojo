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

"""All-gather -> RMSNorm bench: two-launch baseline + fused kernel (bf16).

Times, per-GPU wall-clock (slowest GPU reported):
1. `allgather_only`                     -> t_AG (tuned interleaved standalone AG)
2. `rms_norm_full_cold`                 -> cold standalone norm on the full tensor
3. `allgather_then_rms_norm_chained`    -> t_chained: AG then a norm reading the
   live gathered output on the same stream (warm) -- the honest decode baseline
4. `allgather_rmsnorm_fused`            -> the fused kernel (always fused)
5. `allgather_rmsnorm_dispatch`         -> `_dispatch_ag_norm`, shape-gated auto

The "individual-sum" baseline (t_AG + t_norm) is derived by addition, not timed.
Baseline is the *tuned* standalone `allgather` (fabric-saturated ~160 GB/s on
CDNA4) + `rms_norm_gpu`; the fusion lever is eliding the separate norm launch.

Self-verifies: fused `normed_out` vs a host RMSNorm on the gathered bf16 row
(full-H divisor, mbc=True, cast bf16 last) and `sum_out` bit-identical to a
standalone all-gather (a pure gather/copy). For all-gather the residual is not an
f32 peer-sum, so the RS tight/loose split collapses to one oracle.
"""

from std.sys import (
    get_defined_bool,
    get_defined_dtype,
    get_defined_int,
    has_amd_gpu_accelerator,
    size_of,
    simd_width_of,
)

from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from comm import Signal, MAX_GPUS, group_start, group_end
from comm.allgather import allgather
from comm.reducescatter import ReduceScatterConfig
from comm.allgather_rmsnorm import allgather_rmsnorm, _dispatch_ag_norm
from comm.sync import enable_p2p, init_signal_buffer, is_p2p_enabled
from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from internal_utils import CacheBustingBuffer, arg_parse

from layout import Coord, TileTensor, row_major
from nn.normalization import rms_norm_gpu
from max.benchmark import bench_multicontext, bencher_iter_custom

from std.math import rsqrt
from std.utils.index import Index


@always_inline
def _gathered_value[in_dtype: DType](row: Int, col: Int) -> Scalar[in_dtype]:
    """Value at global (row, col) of the gathered stream (shard fill + oracle).
    """
    return Scalar[in_dtype](1 + (row % 13)) + Scalar[in_dtype](col % 251)


def _launch_norm_full[
    in_dtype: DType,
    num_cols: Int,
](
    in_ptr: MutPointer[Scalar[in_dtype], MutAnyOrigin],
    out_ptr: MutPointer[Scalar[in_dtype], MutAnyOrigin],
    gamma_ptr: ImmPointer[Scalar[in_dtype], ImmutAnyOrigin],
    rows: Int,
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    ctx: DeviceContext,
) raises:
    """Standalone RMSNorm over the full `[rows, num_cols]` tensor (M3
    input_layernorm is replicate -> norms all `rows` on every GPU). Fixed
    buffers, captured once (no per-iter rotation). Caller ensures `rows > 0`."""
    var gamma = TileTensor(gamma_ptr, row_major(Coord(Index(num_cols))))
    var in_buf = TileTensor(in_ptr, row_major(Coord(Index(rows, num_cols))))
    var out_buf = TileTensor(out_ptr, row_major(Coord(Index(rows, num_cols))))

    @always_inline
    @__copy_capture(in_buf)
    @__parameter
    def input_fn[width: Int](coords: Coord) -> SIMD[in_dtype, width]:
        return in_buf.raw_load[width=width](in_buf.layout(coords))

    @always_inline
    @__copy_capture(out_buf)
    @__parameter
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[in_dtype, width]) -> None:
        out_buf.raw_store[width=width, alignment=alignment](
            out_buf.layout(coords), val
        )

    rms_norm_gpu[2, input_fn, output_fn, multiply_before_cast=True](
        Coord(Index(rows, num_cols)), gamma, epsilon, weight_offset, ctx
    )


def _verify_results[
    in_dtype: DType,
    ngpus: Int,
    num_cols: Int,
](
    num_rows: Int,
    list_of_ctx: List[DeviceContext],
    signal_buffers: List[DeviceBuffer[.uint8]],
    cb_shards: List[CacheBustingBuffer[in_dtype]],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    gamma_dev: DeviceBuffer[in_dtype],
    gamma_host: List[Scalar[in_dtype]],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    config: ReduceScatterConfig[in_dtype, ngpus],
) raises:
    """Run the fused kernel + a standalone all-gather into fresh per-rank buffers,
    then gate the norm (vs host ref) and residual (fused sum vs standalone AG) on
    GPU 0 (outputs are replicated, so GPU 0 is representative)."""
    var full_n = num_rows * num_cols

    # Shard views (non-rotated, offset 0) and full output views.
    comptime ShardType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0, num_cols)))), ImmutAnyOrigin
    ]
    comptime FullType = TileTensor[
        mut=True,
        in_dtype,
        type_of(row_major(Coord(Index(0, num_cols)))),
        MutAnyOrigin,
    ]
    comptime GammaType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0)))), ImmutAnyOrigin
    ]
    var in_shards = Array[ShardType, ngpus](
        fill_with=lambda (i: Int) -> ShardType: ShardType(
            rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                cb_shards[i].offset_ptr(0)
            ),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )
    )
    var gamma_view = GammaType(
        rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
            gamma_dev.unsafe_ptr()
        ),
        row_major(Coord(Index(num_cols))),
    )

    var normed_r = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var sum_r = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var ag_r = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    for i in range(ngpus):
        normed_r.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](full_n))
        sum_r.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](full_n))
        ag_r.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](full_n))
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    group_start()
    for i in range(ngpus):
        var normed_view = FullType(
            normed_r[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(num_rows, num_cols))),
        )
        var sum_view = FullType(
            sum_r[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(num_rows, num_cols))),
        )
        allgather_rmsnorm(
            in_shards,
            normed_view,
            sum_view,
            gamma_view,
            epsilon,
            weight_offset,
            rank_sigs,
            list_of_ctx[i],
        )
    group_end()
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Standalone all-gather into ag_r for the residual compare.
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()
    group_start()
    for i in range(ngpus):
        var out_base = rebind[MutPointer[Scalar[in_dtype], MutAnyOrigin]](
            ag_r[i].unsafe_ptr()
        )
        var out_views = Array[FullType, ngpus](
            fill_with=lambda (src: Int) -> FullType: FullType(
                out_base + config.rank_unit_start(src) * num_cols,
                row_major(Coord(Index(config.rank_units(src), num_cols))),
            )
        )
        allgather(in_shards, out_views, rank_sigs, list_of_ctx[i], i)
    group_end()
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Host oracle + comparison on GPU 0.
    var woff = weight_offset.cast[.float32]()
    var normed_h = List[Scalar[in_dtype]](
        length=full_n, fill=Scalar[in_dtype](0)
    )
    var sum_h = List[Scalar[in_dtype]](length=full_n, fill=Scalar[in_dtype](0))
    var ag_h = List[Scalar[in_dtype]](length=full_n, fill=Scalar[in_dtype](0))
    list_of_ctx[0].enqueue_copy(normed_h, normed_r[0])
    list_of_ctx[0].enqueue_copy(sum_h, sum_r[0])
    list_of_ctx[0].enqueue_copy(ag_h, ag_r[0])
    list_of_ctx[0].synchronize()

    var max_ulp = 0
    var gt1_ulp = 0
    var sum_mismatch = 0
    for r in range(num_rows):
        var base = r * num_cols
        var m2 = Float32(0)
        for c in range(num_cols):
            var x = _gathered_value[in_dtype](r, c).cast[.float32]()
            m2 += x * x
        var nf = rsqrt(m2 / Float32(num_cols) + epsilon)
        for c in range(num_cols):
            var x = _gathered_value[in_dtype](r, c).cast[.float32]()
            var g_f = gamma_host[c].cast[.float32]() + woff
            var ref_v = ((x * nf) * g_f).cast[.bfloat16]()
            var gpu = normed_h[base + c].cast[.bfloat16]()
            var ulp = abs(Int(gpu.to_bits()) - Int(ref_v.to_bits()))
            if ulp > max_ulp:
                max_ulp = ulp
            if ulp > 1:
                gt1_ulp += 1
            if sum_h[base + c].to_bits() != ag_h[base + c].to_bits():
                sum_mismatch += 1

    var frac_gt1 = Float32(gt1_ulp) / Float32(full_n)
    print(
        "NORM(fused vs host ref): frac>1ULP =",
        frac_gt1 * 100.0,
        "%, max_ulp =",
        max_ulp,
        "| residual (sum_out vs standalone AG) mismatches =",
        sum_mismatch,
    )
    if frac_gt1 > 0.01 or max_ulp > 4:
        raise Error(
            String(
                "norm gate failed: frac>1ULP = ",
                frac_gt1 * 100.0,
                "%, max_ulp = ",
                max_ulp,
            )
        )
    comptime if has_amd_gpu_accelerator():
        if sum_mismatch != 0:
            raise Error(
                String(
                    (
                        "residual not bit-identical to standalone AG on AMD:"
                        " mismatches = "
                    ),
                    sum_mismatch,
                )
            )
    print("Verification PASSED")
    _ = normed_h^
    _ = sum_h^
    _ = ag_h^
    _ = normed_r^
    _ = sum_r^
    _ = ag_r^


def bench_allgather_rmsnorm[
    in_dtype: DType,
    ngpus: Int,
    num_cols: Int,
    quantize: Bool = False,
    cache_busting: Bool = True,
    verify: Bool = True,
](num_rows: Int, mut b: Bench, list_of_ctx: List[DeviceContext]) raises:
    comptime assert (
        not quantize
    ), "all-gather+RMSNorm baseline is bf16-only (quantize=False)"

    comptime simd_size = simd_width_of[in_dtype, target=get_gpu_target()]()
    var full_n = num_rows * num_cols

    # Ragged shard binning of the full stream (matches a standalone
    # reduce-scatter, so the gather concat order matches a standalone all-gather).
    var config = ReduceScatterConfig[in_dtype, ngpus](
        axis_size=num_rows, unit_numel=num_cols, threads_per_gpu=0
    )

    # Per-GPU shard inputs (cache-busted, rotated per iter). Full outputs +
    # a pre-filled cold full buffer for the standalone-norm variant.
    var cb_shards = List[CacheBustingBuffer[in_dtype]]()
    var cold_full = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var normed = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var sum_full = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var ag_full = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    for i in range(ngpus):
        var shard_rows = config.rank_units(i)
        var shard_len = shard_rows * num_cols
        if shard_len < 1:
            shard_len = (
                1  # 0-row ranks (M=1) still need a valid backing buffer.
            )
        cb_shards.append(
            CacheBustingBuffer[in_dtype](
                shard_len, simd_size, list_of_ctx[i], cache_busting
            )
        )

        # Fill each shard so global row `start_i+lr` matches `_gathered_value`.
        var start_i = config.rank_unit_start(i)
        var h = List[Scalar[in_dtype]](
            unsafe_uninit_length=cb_shards[i].alloc_size()
        )
        for j in range(cb_shards[i].alloc_size()):
            var local = j % shard_len
            var lr = local // num_cols
            var c = local % num_cols
            h[j] = _gathered_value[in_dtype](start_i + lr, c)
        list_of_ctx[i].enqueue_copy(cb_shards[i].device_buffer(), h)
        _ = h^

        normed.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](full_n))
        sum_full.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](full_n))
        ag_full.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](full_n))

        # Pre-fill the cold full buffer (the standalone-norm variant's input).
        cold_full.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](full_n))
        var cold_h = List[Scalar[in_dtype]](
            length=full_n, fill=Scalar[in_dtype](0)
        )
        for r in range(num_rows):
            for c in range(num_cols):
                cold_h[r * num_cols + c] = _gathered_value[in_dtype](r, c)
        list_of_ctx[i].enqueue_copy(cold_full[i], cold_h)
        _ = cold_h^

        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[.uint8](size_of[Signal]())
        )
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
        rank_sigs[i] = (
            signal_buffers[i]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )

    # Gamma (shared; each rank reads GPU 0 via P2P as the standalone norm does).
    var gamma_dev = list_of_ctx[0].enqueue_create_buffer[in_dtype](num_cols)
    var gamma_host = List(length=num_cols, fill=Scalar[in_dtype](0))
    for c in range(num_cols):
        gamma_host[c] = (Float64(c + num_cols) / Float64(num_cols)).cast[
            in_dtype
        ]()
    list_of_ctx[0].enqueue_copy(gamma_dev, gamma_host)
    var epsilon = Float32(1e-6)
    var weight_offset = Scalar[in_dtype](1.0)  # M3 Gemma-style
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    comptime ShardType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0, num_cols)))), ImmutAnyOrigin
    ]
    comptime FullType = TileTensor[
        mut=True,
        in_dtype,
        type_of(row_major(Coord(Index(0, num_cols)))),
        MutAnyOrigin,
    ]
    comptime GammaType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0)))), ImmutAnyOrigin
    ]
    var in_shards = Array[ShardType, ngpus](
        fill_with=lambda (i: Int) -> ShardType: ShardType(
            rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                cb_shards[i].unsafe_ptr()
            ),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )
    )
    var normed_views = Array[FullType, ngpus](
        fill_with=lambda (i: Int) {
            mut normed, imm num_rows
        } -> FullType: FullType(
            normed[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(num_rows, num_cols))),
        )
    )
    var sum_views = Array[FullType, ngpus](
        fill_with=lambda (i: Int) {
            mut sum_full, imm num_rows
        } -> FullType: FullType(
            sum_full[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(num_rows, num_cols))),
        )
    )
    var gamma_shard = GammaType(
        rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
            gamma_dev.unsafe_ptr()
        ),
        row_major(Coord(Index(num_cols))),
    )
    var rank_counts = Array[Int, ngpus](
        fill_with=lambda (i: Int) -> Int: config.rank_units(i)
    )

    # Per-GPU ptrs captured once for the timed closures.
    comptime PtrType = MutPointer[Scalar[in_dtype], MutAnyOrigin]
    var cold_ptrs = Array[PtrType, ngpus](
        fill_with=lambda (i: Int) {mut cold_full} -> PtrType: cold_full[i]
        .unsafe_ptr()
        .as_unsafe_any_origin()
    )
    var ag_ptrs = Array[PtrType, ngpus](
        fill_with=lambda (i: Int) {mut ag_full} -> PtrType: ag_full[i]
        .unsafe_ptr()
        .as_unsafe_any_origin()
    )
    var normed_ptrs = Array[PtrType, ngpus](
        fill_with=lambda (i: Int) {mut normed} -> PtrType: normed[i]
        .unsafe_ptr()
        .as_unsafe_any_origin()
    )
    var gamma_ptr = rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
        gamma_dev.unsafe_ptr().as_unsafe_any_origin()
    )
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    var total_bytes = full_n * size_of[in_dtype]()
    var bench_name_prefix = String(
        "allgather_rmsnorm/",
        in_dtype,
        "/",
        ngpus,
        "gpu/",
        num_rows,
        "x",
        num_cols,
    )

    @always_inline
    def _rebuild_shards(
        cache_iter: Int,
        mut in_shards: Array[ShardType, ngpus],
    ) {imm}:
        comptime for _j in range(ngpus):
            in_shards[_j] = ShardType(
                rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                    cb_shards[_j].offset_ptr(cache_iter)
                ),
                row_major(Coord(Index(rank_counts[_j], num_cols))),
            )

    # ===== Variant 1: all-gather only -> t_AG =====
    @always_inline
    def bench_ag_iter(
        mut bench: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises {mut in_shards, imm}:
        @always_inline
        def call_fn(
            ctx_inner: DeviceContext, cache_iter: Int
        ) raises {mut in_shards, imm}:
            _rebuild_shards(cache_iter, in_shards)
            # Rebuild per-rank all-gather output views (into ag_full) for the timed AG.
            var out_base = rebind[MutPointer[Scalar[in_dtype], MutAnyOrigin]](
                ag_full[ctx_idx].unsafe_ptr()
            )
            var out_views = Array[FullType, ngpus](
                fill_with=lambda (src: Int) -> FullType: FullType(
                    out_base + config.rank_unit_start(src) * num_cols,
                    row_major(Coord(Index(config.rank_units(src), num_cols))),
                )
            )
            allgather(in_shards, out_views, rank_sigs, ctx_inner, ctx_idx)

        bencher_iter_custom(bench, call_fn, ctx)

    bench_multicontext(
        b,
        bench_ag_iter,
        list_of_ctx,
        BenchId("allgather_only", input_id=bench_name_prefix),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    # ===== Variant 2: standalone RMSNorm on a cold full tensor -> t_norm =====
    @always_inline
    def bench_norm_cold_iter(
        mut bench: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises {imm}:
        @always_inline
        def call_fn(ctx_inner: DeviceContext, cache_iter: Int) raises {imm}:
            if num_rows > 0:
                _launch_norm_full[in_dtype, num_cols](
                    cold_ptrs[ctx_idx],
                    normed_ptrs[ctx_idx],
                    gamma_ptr,
                    num_rows,
                    epsilon,
                    weight_offset,
                    ctx_inner,
                )

        bencher_iter_custom(bench, call_fn, ctx)

    bench_multicontext(
        b,
        bench_norm_cold_iter,
        list_of_ctx,
        BenchId("rms_norm_full_cold", input_id=bench_name_prefix),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    # ===== Variant 3: AG then RMSNorm on the live gathered output -> t_chained =
    @always_inline
    def bench_chained_iter(
        mut bench: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises {mut in_shards, imm}:
        @always_inline
        def call_fn(
            ctx_inner: DeviceContext, cache_iter: Int
        ) raises {mut in_shards, imm}:
            _rebuild_shards(cache_iter, in_shards)
            # Rebuild per-rank all-gather output views (into ag_full) for the timed AG.
            var out_base = rebind[MutPointer[Scalar[in_dtype], MutAnyOrigin]](
                ag_full[ctx_idx].unsafe_ptr()
            )
            var out_views = Array[FullType, ngpus](
                fill_with=lambda (src: Int) -> FullType: FullType(
                    out_base + config.rank_unit_start(src) * num_cols,
                    row_major(Coord(Index(config.rank_units(src), num_cols))),
                )
            )
            allgather(in_shards, out_views, rank_sigs, ctx_inner, ctx_idx)
            if num_rows > 0:
                _launch_norm_full[in_dtype, num_cols](
                    ag_ptrs[ctx_idx],
                    normed_ptrs[ctx_idx],
                    gamma_ptr,
                    num_rows,
                    epsilon,
                    weight_offset,
                    ctx_inner,
                )

        bencher_iter_custom(bench, call_fn, ctx)

    bench_multicontext(
        b,
        bench_chained_iter,
        list_of_ctx,
        BenchId("allgather_then_rms_norm_chained", input_id=bench_name_prefix),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    # ===== Variant 4: fused all-gather + RMSNorm kernel -> t_fused =====
    @always_inline
    def bench_fused_iter(
        mut bench: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises {mut in_shards, imm}:
        @always_inline
        def call_fn(
            ctx_inner: DeviceContext, cache_iter: Int
        ) raises {mut in_shards, imm}:
            _rebuild_shards(cache_iter, in_shards)
            allgather_rmsnorm(
                in_shards,
                normed_views[ctx_idx],
                sum_views[ctx_idx],
                gamma_shard,
                epsilon,
                weight_offset,
                rank_sigs,
                ctx_inner,
            )

        bencher_iter_custom(bench, call_fn, ctx)

    bench_multicontext(
        b,
        bench_fused_iter,
        list_of_ctx,
        BenchId("allgather_rmsnorm_fused", input_id=bench_name_prefix),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    # ===== Variant 5: shape-gated dispatch =====
    @always_inline
    def bench_dispatch_iter(
        mut bench: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises {mut in_shards, imm}:
        @always_inline
        def call_fn(
            ctx_inner: DeviceContext, cache_iter: Int
        ) raises {mut in_shards, imm}:
            _rebuild_shards(cache_iter, in_shards)

            @always_inline
            def two_launch() raises capturing:
                var out_base = rebind[
                    MutPointer[Scalar[in_dtype], MutAnyOrigin]
                ](sum_full[ctx_idx].unsafe_ptr())
                var out_views = Array[FullType, ngpus](uninitialized=True)
                comptime for src in range(ngpus):
                    out_views[src] = FullType(
                        out_base + config.rank_unit_start(src) * num_cols,
                        row_major(
                            Coord(Index(config.rank_units(src), num_cols))
                        ),
                    )
                allgather(in_shards, out_views, rank_sigs, ctx_inner, ctx_idx)
                if num_rows > 0:
                    _launch_norm_full[in_dtype, num_cols](
                        rebind[MutPointer[Scalar[in_dtype], MutAnyOrigin]](
                            sum_full[ctx_idx].unsafe_ptr()
                        ),
                        normed_ptrs[ctx_idx],
                        gamma_ptr,
                        num_rows,
                        epsilon,
                        weight_offset,
                        ctx_inner,
                    )

            _dispatch_ag_norm[two_launch=two_launch](
                in_shards,
                normed_views[ctx_idx],
                sum_views[ctx_idx],
                gamma_shard,
                epsilon,
                weight_offset,
                rank_sigs,
                ctx_inner,
            )

        bencher_iter_custom(bench, call_fn, ctx)

    bench_multicontext(
        b,
        bench_dispatch_iter,
        list_of_ctx,
        BenchId("allgather_rmsnorm_dispatch", input_id=bench_name_prefix),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    b.dump_report()

    comptime if verify:
        _verify_results[in_dtype, ngpus, num_cols](
            num_rows,
            list_of_ctx,
            signal_buffers,
            cb_shards,
            rank_sigs,
            gamma_dev,
            gamma_host,
            epsilon,
            weight_offset,
            config,
        )

    _ = signal_buffers^
    _ = cb_shards^
    _ = cold_full^
    _ = normed^
    _ = sum_full^
    _ = ag_full^
    _ = gamma_dev^
    _ = gamma_host^


def main() raises:
    comptime in_dtype = get_defined_dtype["in_dtype", .bfloat16]()
    comptime quantize = get_defined_bool["quantize", False]()
    comptime num_gpus = get_defined_int["num_gpus", 4]()
    var num_rows = Int(arg_parse("num_rows", 1))
    comptime num_cols = get_defined_int["num_cols", 6144]()
    comptime cache_busting = get_defined_bool["cache_busting", True]()
    comptime verify = get_defined_bool["verify", True]()

    var num_devices = DeviceContext.number_of_devices()
    if num_devices < num_gpus:
        print(
            "Need", num_gpus, "GPUs but only found", num_devices, "- skipping."
        )
        return

    _ = enable_p2p()
    if not is_p2p_enabled():
        print("P2P not enabled, skipping benchmark.")
        return

    var list_of_ctx = List[DeviceContext]()
    for i in range(num_gpus):
        list_of_ctx.append(DeviceContext(device_id=i))

    print(
        "Benchmarking all-gather + RMSNorm (no quant):",
        num_gpus,
        "GPUs,",
        in_dtype,
        ",",
        num_rows,
        "x",
        num_cols,
    )

    var m = Bench(BenchConfig(num_repetitions=1))
    bench_allgather_rmsnorm[
        in_dtype, num_gpus, num_cols, quantize, cache_busting, verify
    ](num_rows, m, list_of_ctx)
