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

"""Reduce-scatter -> RMSNorm bench: two-launch baseline + fused kernel (bf16).

Times, per-GPU wall-clock (slowest GPU reported):
1. `reducescatter_only`                  -> t_RS
2. `rms_norm_shard_cold`                 -> cold standalone norm on the shard
3. `reducescatter_then_rms_norm_chained` -> t_chained: RS then a norm reading the
   live RS output on the same stream (warm) — the honest decode baseline
4. `reducescatter_rmsnorm_fused`         -> the fused kernel (always fused)
5. `reducescatter_rmsnorm_dispatch`      -> `_dispatch_rs_norm`, shape-gated auto
   (fused below `RS_NORM_FUSE_THRESHOLD`, two-launch above)

The prefill "cold-sum" baseline (t_RS + t_norm) is derived by addition, not timed.

Self-verifies both paths against two host references (full-H divisor,
multiply-before-cast, cast-to-bf16 last): a bf16-rounded ref (tight for the
two-launch, loose for fused) and an f32-throughout ref (tight for fused, loose
for two-launch), the gap being the mid-way bf16 rounding of the RS sum.
`sum_out` is also checked bit-identical to a standalone RS shard (AMD non-multimem).
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
from max.benchmark import bench_multicontext, bencher_iter_custom
from comm import Signal, MAX_GPUS, group_start, group_end
from comm.reducescatter import reducescatter, ReduceScatterConfig
from comm.reducescatter_rmsnorm import reducescatter_rmsnorm, _dispatch_rs_norm
from comm.sync import enable_p2p, init_signal_buffer, is_p2p_enabled
from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from internal_utils import CacheBustingBuffer, arg_parse

from layout import Coord, TileTensor, row_major
from nn.normalization import rms_norm_gpu

from std.math import rsqrt
from std.utils.index import Index


def _launch_norm[
    in_dtype: DType,
    num_cols: Int,
](
    in_ptr: MutPointer[Scalar[in_dtype], MutAnyOrigin],
    out_ptr: MutPointer[Scalar[in_dtype], MutAnyOrigin],
    gamma_ptr: ImmPointer[Scalar[in_dtype], ImmutAnyOrigin],
    local_rows: Int,
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    ctx: DeviceContext,
) raises:
    """Launch a standalone RMSNorm over `[local_rows, num_cols]`.

    `in_ptr`/`out_ptr` are fixed shard buffers, so the tensors are built once and
    captured (no per-iter rotation). Caller must ensure `local_rows > 0`.
    """
    var gamma = TileTensor(gamma_ptr, row_major(Coord(Index(num_cols))))
    var in_buf = TileTensor(
        in_ptr, row_major(Coord(Index(local_rows, num_cols)))
    )
    var out_buf = TileTensor(
        out_ptr, row_major(Coord(Index(local_rows, num_cols)))
    )

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

    rms_norm_gpu[
        2,
        input_fn,
        output_fn,
        multiply_before_cast=True,
    ](Coord(Index(local_rows, num_cols)), gamma, epsilon, weight_offset, ctx)


def _verify_results[
    in_dtype: DType,
    ngpus: Int,
    num_cols: Int,
](
    num_rows: Int,
    list_of_ctx: List[DeviceContext],
    signal_buffers: List[DeviceBuffer[.uint8]],
    cb_inputs: List[CacheBustingBuffer[in_dtype]],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    gamma_dev: DeviceBuffer[in_dtype],
    gamma_host: List[Scalar[in_dtype]],
    host_bufs: List[List[Scalar[in_dtype]]],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    config: ReduceScatterConfig[in_dtype, ngpus],
) raises:
    """Run both GPU paths into fresh buffers and compare each rank's normed shard
    against two host references over its owned rows.

    The two-launch path IS the bf16-rounded algorithm (RS stores the f32 peer-sum
    as bf16, the norm reads it back): tight vs the bf16-rounded ref (only
    block-reduce vs serial-f32 associativity noise), loose vs the f32-throughout
    ref (the mid-way bf16 rounding). Both refs use the full-H divisor and fold
    gamma in f32 before the final bf16 cast.
    """
    # Fresh buffers for both runs (>= 1 row so 0-row ranks aren't zero-size).
    # `v_rs_shard` doubles as the standalone RS shard `sum_out` is checked against.
    var v_rs_shard = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var v_normed = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var v_fused_normed = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var v_fused_sum = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    for i in range(ngpus):
        var alloc_i = config.rank_num_elements(i)
        if alloc_i < 1:
            alloc_i = 1
        v_rs_shard.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i)
        )
        v_normed.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i))
        v_fused_normed.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i)
        )
        v_fused_sum.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i)
        )

    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])

    # Launch 1: reduce-scatter (collective; 0-row ranks only clear the barrier).
    # Non-rotated inputs (offset 0) so the host oracle matches `host_bufs`.
    comptime InTensorType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0, num_cols)))), ImmutAnyOrigin
    ]
    comptime OutShardType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0, num_cols)))), MutAnyOrigin
    ]
    var in_bufs = Array[InTensorType, ngpus](
        fill_with=lambda (i: Int) -> InTensorType: InTensorType(
            rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                cb_inputs[i].offset_ptr(0)
            ),
            row_major(Coord(Index(num_rows, num_cols))),
        )
    )
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    group_start()
    for i in range(ngpus):
        var out_shard = OutShardType(
            v_rs_shard[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )
        reducescatter[dtype=in_dtype, ngpus=ngpus, axis=0](
            in_bufs, out_shard, rank_sigs, list_of_ctx[i]
        )
    group_end()
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Launch 2: standalone RMSNorm on each rank's bf16 RS shard.
    for i in range(ngpus):
        var local_rows = config.rank_units(i)
        if local_rows > 0:
            _launch_norm[in_dtype, num_cols](
                v_rs_shard[i].unsafe_ptr().as_unsafe_any_origin(),
                v_normed[i].unsafe_ptr().as_unsafe_any_origin(),
                rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                    gamma_dev.unsafe_ptr().as_unsafe_any_origin()
                ),
                local_rows,
                epsilon,
                weight_offset,
                list_of_ctx[i],
            )
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Launch 3: the fused kernel into fresh normed/sum shards. Re-init the
    # signal buffers so this collective gets clean barrier state.
    comptime GammaShardType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0)))), ImmutAnyOrigin
    ]
    var gamma_view = GammaShardType(
        rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
            gamma_dev.unsafe_ptr()
        ),
        row_major(Coord(Index(num_cols))),
    )
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    group_start()
    for i in range(ngpus):
        var normed_view = OutShardType(
            v_fused_normed[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )
        var sum_view = OutShardType(
            v_fused_sum[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )
        reducescatter_rmsnorm(
            in_bufs,
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

    # Host references + comparison (aggregate over all owned rows of all ranks).
    var woff = weight_offset.cast[.float32]()
    var errs_bf = (
        0  # cast-to-f32 exact-inequality count vs bf16-rounded (tight)
    )
    var errs_f32 = 0  # cast-to-f32 exact-inequality count vs f32 ref (loose)
    var total_elems = 0
    var max_ulp_bf = 0
    var max_ulp_f = 0
    var sum_ulp_bf = 0
    var sum_ulp_f = 0
    var gt1_ulp_bf = 0  # tight-ref elements with bf16-ULP distance > 1
    # Fused-kernel stats (tight vs bf16 ref, loose vs f32 ref).
    var fused_errs_f = 0  # exact-mismatch vs f32 ref (loose, reported)
    var fused_max_ulp_f = 0  # loose: fused vs f32 ref
    var fused_max_ulp_bf = 0  # tight: fused vs bf16 ref
    var fused_gt1_ulp_bf = 0  # tight-ref elements with bf16-ULP distance > 1
    var fused_sum_mismatch = 0  # sum_out vs standalone RS shard
    var fused_sum_max_ulp = 0
    var accum = List[Float32](length=num_cols, fill=Float32(0))

    for i in range(ngpus):
        var local_rows = config.rank_units(i)
        if local_rows == 0:
            continue
        var start = config.rank_unit_start(i)
        var n = local_rows * num_cols
        var gpu_h = List[Scalar[in_dtype]](length=n, fill=Scalar[in_dtype](0))
        var fnorm_h = List[Scalar[in_dtype]](length=n, fill=Scalar[in_dtype](0))
        var fsum_h = List[Scalar[in_dtype]](length=n, fill=Scalar[in_dtype](0))
        var rs_h = List[Scalar[in_dtype]](length=n, fill=Scalar[in_dtype](0))
        list_of_ctx[i].enqueue_copy(gpu_h, v_normed[i])
        list_of_ctx[i].enqueue_copy(fnorm_h, v_fused_normed[i])
        list_of_ctx[i].enqueue_copy(fsum_h, v_fused_sum[i])
        list_of_ctx[i].enqueue_copy(rs_h, v_rs_shard[i])
        list_of_ctx[i].synchronize()
        total_elems += n

        for rr in range(local_rows):
            var grow = start + rr
            var base = grow * num_cols

            # Pass 1: f32 peer-sum, and both mean-square accumulations.
            var m2_bf = Float32(0)
            var m2_f = Float32(0)
            for c in range(num_cols):
                var s = Float32(0)
                for g in range(ngpus):
                    s += host_bufs[g][base + c].cast[.float32]()
                accum[c] = s
                var xb = s.cast[.bfloat16]().cast[.float32]()
                m2_bf += xb * xb
                m2_f += s * s

            var nf_bf = rsqrt(m2_bf / Float32(num_cols) + epsilon)
            var nf_f = rsqrt(m2_f / Float32(num_cols) + epsilon)

            # Pass 2: normalize, fold gamma in f32, cast to bf16 last, compare.
            for c in range(num_cols):
                var s = accum[c]
                var xb = s.cast[.bfloat16]().cast[.float32]()
                var g_f = gamma_host[c].cast[.float32]() + woff
                var ref_bf16 = ((xb * nf_bf) * g_f).cast[.bfloat16]()
                var ref_f16 = ((s * nf_f) * g_f).cast[.bfloat16]()
                var gpu_bf16 = gpu_h[rr * num_cols + c].cast[.bfloat16]()

                # Exact-inequality counters (reported; loose oracle gates on ULP
                # magnitude, not this rate).
                if gpu_bf16.cast[.float32]() != ref_bf16.cast[.float32]():
                    errs_bf += 1
                if gpu_bf16.cast[.float32]() != ref_f16.cast[.float32]():
                    errs_f32 += 1

                # bf16-ULP via bit-reinterpret: outputs strictly positive, so
                # |bits_a - bits_b| is the ULP distance.
                var gpu_bits = Int(gpu_bf16.to_bits())
                var ulp_bf = abs(gpu_bits - Int(ref_bf16.to_bits()))
                var ulp_f = abs(gpu_bits - Int(ref_f16.to_bits()))
                sum_ulp_bf += ulp_bf
                sum_ulp_f += ulp_f
                if ulp_bf > max_ulp_bf:
                    max_ulp_bf = ulp_bf
                if ulp_f > max_ulp_f:
                    max_ulp_f = ulp_f
                if ulp_bf > 1:
                    gt1_ulp_bf += 1

                # Fused: rounds the sum to bf16 first, so tight vs ref_bf16 and
                # loose vs ref_f16 (matches the standalone path).
                var f_bits = Int(
                    fnorm_h[rr * num_cols + c].cast[.bfloat16]().to_bits()
                )
                var f_ulp_f = abs(f_bits - Int(ref_f16.to_bits()))
                var f_ulp_bf = abs(f_bits - Int(ref_bf16.to_bits()))
                if f_ulp_bf > fused_max_ulp_bf:
                    fused_max_ulp_bf = f_ulp_bf
                if f_ulp_bf > 1:
                    fused_gt1_ulp_bf += 1
                if f_ulp_f > fused_max_ulp_f:
                    fused_max_ulp_f = f_ulp_f
                if (
                    fnorm_h[rr * num_cols + c].cast[.float32]()
                    != ref_f16.cast[.float32]()
                ):
                    fused_errs_f += 1

                # sum_out vs the standalone RS shard (bit-identical on AMD).
                var f_sum_bits = Int(
                    fsum_h[rr * num_cols + c].cast[.bfloat16]().to_bits()
                )
                var rs_bits = Int(
                    rs_h[rr * num_cols + c].cast[.bfloat16]().to_bits()
                )
                var s_ulp = abs(f_sum_bits - rs_bits)
                if s_ulp != 0:
                    fused_sum_mismatch += 1
                if s_ulp > fused_sum_max_ulp:
                    fused_sum_max_ulp = s_ulp
        _ = gpu_h^
        _ = fnorm_h^
        _ = fsum_h^
        _ = rs_h^

    var rate_bf = Float32(errs_bf) / Float32(total_elems)
    var rate_f = Float32(errs_f32) / Float32(total_elems)
    var frac_gt1_bf = Float32(gt1_ulp_bf) / Float32(total_elems)
    var mean_ulp_bf = Float32(sum_ulp_bf) / Float32(total_elems)
    var mean_ulp_f = Float32(sum_ulp_f) / Float32(total_elems)

    # Loose oracle gates on bf16-ULP MAGNITUDE, not exact-mismatch RATE: mid-way
    # bf16 rounding of the ~1010 peer sum shifts 1-2 ULP on ~24% of elements, so
    # an exact-rate<=3% gate false-fails. A real bug is large (>>4 ULP).
    print(
        (
            "TIGHT (GPU two-launch vs host bf16-rounded ref, same algorithm):"
            " exact-match ="
        ),
        (1.0 - rate_bf) * 100.0,
        "%, max_ulp =",
        max_ulp_bf,
        ", fraction>1ULP =",
        frac_gt1_bf * 100.0,
        "%, mean_ulp =",
        mean_ulp_bf,
    )
    print(
        "LOOSE (GPU two-launch vs host f32 ref): exact-mismatch =",
        rate_f * 100.0,
        "% [fusion-accuracy headroom], max_ulp =",
        max_ulp_f,
        ", mean_ulp =",
        mean_ulp_f,
    )

    # Tight gate: same algorithm, bit-equal apart from block-reduce vs
    # serial-f32 associativity in the mean-square (~1-ULP nf wobble).
    if frac_gt1_bf > 0.01 or max_ulp_bf > 4:
        raise Error(
            String(
                "tight bf16-rounded-ref gate failed: fraction>1ULP = ",
                frac_gt1_bf * 100.0,
                "%, max_ulp = ",
                max_ulp_bf,
            )
        )
    # Loose gate: bounded bf16-ULP magnitude (legit rounding small, a bug large).
    if max_ulp_f > 4:
        raise Error(
            String("loose f32-ref ULP gate failed: max_ulp = ", max_ulp_f)
        )

    # --- Fused-kernel gates (tight vs bf16 ref, loose vs f32 ref). ---
    var fused_frac_gt1_bf = Float32(fused_gt1_ulp_bf) / Float32(total_elems)
    var fused_rate_f = Float32(fused_errs_f) / Float32(total_elems)
    print(
        "FUSED TIGHT (kernel vs host bf16 ref): frac>1ULP =",
        fused_frac_gt1_bf * 100.0,
        "%, max_ulp =",
        fused_max_ulp_bf,
    )
    print(
        "FUSED LOOSE (kernel vs host f32 ref): exact-mismatch =",
        fused_rate_f * 100.0,
        "% [mid-way rounding divergence], max_ulp =",
        fused_max_ulp_f,
        "| sum_out mismatches =",
        fused_sum_mismatch,
        ", max_ulp =",
        fused_sum_max_ulp,
    )
    if fused_frac_gt1_bf > 0.01 or fused_max_ulp_bf > 4:
        raise Error(
            String(
                "fused tight bf16-ref gate failed: frac>1ULP = ",
                fused_frac_gt1_bf * 100.0,
                "%, max_ulp = ",
                fused_max_ulp_bf,
            )
        )
    if fused_max_ulp_f > 4:
        raise Error(
            String(
                "fused loose f32-ref ULP gate failed: max_ulp = ",
                fused_max_ulp_f,
            )
        )
    # sum_out: bit-identical on AMD non-multimem; 1-ULP tolerance on NVIDIA.
    comptime if has_amd_gpu_accelerator():
        if fused_sum_mismatch != 0:
            raise Error(
                String(
                    (
                        "fused sum_out not bit-identical to standalone RS on"
                        " AMD: mismatches = "
                    ),
                    fused_sum_mismatch,
                    ", max_ulp = ",
                    fused_sum_max_ulp,
                )
            )
    else:
        if fused_sum_max_ulp > 1:
            raise Error(
                String(
                    (
                        "fused sum_out exceeds 1-ULP tolerance vs standalone"
                        " RS: max_ulp = "
                    ),
                    fused_sum_max_ulp,
                )
            )

    print("Verification PASSED")
    _ = v_rs_shard^
    _ = v_normed^
    _ = v_fused_normed^
    _ = v_fused_sum^
    _ = accum^


def bench_reducescatter_rmsnorm[
    in_dtype: DType,
    ngpus: Int,
    num_cols: Int,
    quantize: Bool = False,
    cache_busting: Bool = True,
    verify: Bool = True,
](num_rows: Int, mut b: Bench, list_of_ctx: List[DeviceContext]) raises:
    # This baseline is bf16-only; there is no FP8 / scale machinery here.
    comptime assert (
        not quantize
    ), "reduce-scatter+RMSNorm baseline is bf16-only (quantize=False)"

    var length = num_rows * num_cols
    comptime simd_size = simd_width_of[in_dtype, target=get_gpu_target()]()

    # Ragged partition: remainder rows go to low ranks (M=1 -> rank 0 owns the
    # row, ranks 1-3 own none). Other M here are divisible by ngpus.
    var config = ReduceScatterConfig[in_dtype, ngpus](
        axis_size=num_rows, unit_numel=num_cols, threads_per_gpu=0
    )

    # --- Per-GPU buffers (set up once, before timing) ---
    # RS inputs (cache-busted, rotated per iter); the two derived reduce-scatter
    # variants reuse these.
    var cb_inputs = List[CacheBustingBuffer[in_dtype]]()
    var host_bufs = List[List[Scalar[in_dtype]]](capacity=ngpus)

    # RS output shard, a pre-filled cold shard for the standalone-norm variant,
    # and the normed output. >= 1 row so 0-row ranks aren't zero-size.
    var rs_out_shard = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var cold_shard = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var normed = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    # Fused-kernel sum_out (residual) shard; normed reuses `normed` above.
    var fused_sum = List[DeviceBuffer[in_dtype]](capacity=ngpus)

    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    for i in range(ngpus):
        cb_inputs.append(
            CacheBustingBuffer[in_dtype](
                length, simd_size, list_of_ctx[i], cache_busting
            )
        )

        var alloc_i = config.rank_num_elements(i)
        if alloc_i < 1:
            alloc_i = 1
        rs_out_shard.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i)
        )
        cold_shard.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i)
        )
        normed.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i))
        fused_sum.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i)
        )

        # Distinct per-GPU data so the peer sum is non-trivial: the sum reaches
        # ~1010 (bf16 granule 4), so the two-launch's mid-way bf16 store is
        # genuinely lossy (~1 ULP), the headroom the fused kernel recovers.
        # Positive keeps the bf16-ULP-via-bits verify clean.
        var h = List[Scalar[in_dtype]](
            unsafe_uninit_length=cb_inputs[0].alloc_size()
        )
        for j in range(cb_inputs[0].alloc_size()):
            h[j] = Scalar[in_dtype](i + 1) + Scalar[in_dtype](j % 251)
        list_of_ctx[i].enqueue_copy(cb_inputs[i].device_buffer(), h)
        host_bufs.append(h^)

        # Pre-fill the cold shard (fixed buffer: cold at prefill when it exceeds
        # L2, launch-floor-dominated at decode — no cache-bust needed).
        var cold_h = List[Scalar[in_dtype]](
            length=alloc_i, fill=Scalar[in_dtype](0)
        )
        for j in range(alloc_i):
            cold_h[j] = Scalar[in_dtype](i + 1) + Scalar[in_dtype](j % 251)
        list_of_ctx[i].enqueue_copy(cold_shard[i], cold_h)
        _ = cold_h^

        # Plain reduce-scatter needs only the Signal (no all-gather scratch).
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

    # Gamma weights (shared, read-only; GPU 0 is fine).
    var gamma_dev = list_of_ctx[0].enqueue_create_buffer[in_dtype](num_cols)
    var gamma_host = List(length=num_cols, fill=Scalar[in_dtype](0))
    for i in range(num_cols):
        gamma_host[i] = (Float64(i + num_cols) / Float64(num_cols)).cast[
            in_dtype
        ]()
    list_of_ctx[0].enqueue_copy(gamma_dev, gamma_host)
    var epsilon = Float32(1e-6)
    var weight_offset = Scalar[in_dtype](0.0)

    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # RS input / output-shard tensor views. Inputs rotate per iter (as a direct
    # Array arg); output shards are fixed (RS overwrites the storage).
    comptime InTensorType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0, num_cols)))), ImmutAnyOrigin
    ]
    comptime OutShardType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0, num_cols)))), MutAnyOrigin
    ]
    comptime GammaShardType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0)))), ImmutAnyOrigin
    ]
    var in_bufs = Array[InTensorType, ngpus](
        fill_with=lambda (i: Int) -> InTensorType: InTensorType(
            rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                cb_inputs[i].unsafe_ptr()
            ),
            row_major(Coord(Index(num_rows, num_cols))),
        )
    )
    var out_shards = Array[OutShardType, ngpus](
        fill_with=lambda (i: Int) {
            ref rs_out_shard, imm config
        } -> OutShardType: OutShardType(
            rs_out_shard[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )
    )
    # Fused-kernel output-shard views: normed + sum, both [rank_units, cols].
    var normed_shards = Array[OutShardType, ngpus](
        fill_with=lambda (i: Int) {
            ref normed, imm config
        } -> OutShardType: OutShardType(
            normed[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )
    )
    var fused_sum_shards = Array[OutShardType, ngpus](
        fill_with=lambda (i: Int) {
            ref fused_sum, imm config
        } -> OutShardType: OutShardType(
            fused_sum[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )
    )
    # Shared gamma view (GPU 0; peer-read on other ranks via P2P, matching the
    # standalone-norm variant).
    var gamma_shard = GammaShardType(
        rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
            gamma_dev.unsafe_ptr()
        ),
        row_major(Coord(Index(num_cols))),
    )
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    comptime InDTypePointer = Pointer[Scalar[in_dtype], MutAnyOrigin]

    # Per-GPU norm shard pointers, captured once for the timed closures.
    var cold_ptrs = Array[MutPointer[Scalar[in_dtype], MutAnyOrigin], ngpus](
        fill_with=lambda (i: Int) {ref} -> InDTypePointer: cold_shard[i]
        .unsafe_ptr()
        .as_unsafe_any_origin()
    )
    var rs_out_ptrs = Array[MutPointer[Scalar[in_dtype], MutAnyOrigin], ngpus](
        fill_with=lambda (i: Int) {ref} -> InDTypePointer: rs_out_shard[i]
        .unsafe_ptr()
        .as_unsafe_any_origin()
    )
    var normed_ptrs = Array[MutPointer[Scalar[in_dtype], MutAnyOrigin], ngpus](
        fill_with=lambda (i: Int) {ref} -> InDTypePointer: normed[i]
        .unsafe_ptr()
        .as_unsafe_any_origin()
    )
    var gamma_ptr = rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
        gamma_dev.unsafe_ptr().as_unsafe_any_origin()
    )

    var total_bytes = ngpus * length * size_of[in_dtype]()
    var bench_name_prefix = String(
        "reducescatter_rmsnorm/",
        in_dtype,
        "/",
        ngpus,
        "gpu/",
        num_rows,
        "x",
        num_cols,
    )

    # ===== Variant 1: reduce-scatter only -> t_RS =====
    @always_inline
    def bench_rs_iter(
        mut bench: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises {mut in_bufs, imm}:
        @always_inline
        def call_fn(
            ctx_inner: DeviceContext, cache_iter: Int
        ) raises {mut in_bufs, imm}:
            comptime for _j in range(ngpus):
                in_bufs[_j] = InTensorType(
                    rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                        cb_inputs[_j].offset_ptr(cache_iter)
                    ),
                    row_major(Coord(Index(num_rows, num_cols))),
                )
            reducescatter[dtype=in_dtype, ngpus=ngpus, axis=0](
                in_bufs, out_shards[ctx_idx], rank_sigs, ctx_inner
            )

        bencher_iter_custom(bench, call_fn, ctx)

    bench_multicontext(
        b,
        bench_rs_iter,
        list_of_ctx,
        BenchId("reducescatter_only", input_id=bench_name_prefix),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    # ===== Variant 2: standalone RMSNorm on a cold shard -> t_norm(shard) =====
    @always_inline
    def bench_norm_cold_iter(
        mut bench: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises {imm}:
        var local_rows = config.rank_units(ctx_idx)

        @always_inline
        def call_fn(ctx_inner: DeviceContext, cache_iter: Int) raises {imm}:
            if local_rows > 0:
                _launch_norm[in_dtype, num_cols](
                    cold_ptrs[ctx_idx],
                    normed_ptrs[ctx_idx],
                    gamma_ptr,
                    local_rows,
                    epsilon,
                    weight_offset,
                    ctx_inner,
                )

        bencher_iter_custom(bench, call_fn, ctx)

    bench_multicontext(
        b,
        bench_norm_cold_iter,
        list_of_ctx,
        BenchId("rms_norm_shard_cold", input_id=bench_name_prefix),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    # ===== Variant 3: RS then RMSNorm on the live RS output -> t_chained =====
    @always_inline
    def bench_chained_iter(
        mut bench: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises {mut in_bufs, imm}:
        var local_rows = config.rank_units(ctx_idx)

        @always_inline
        def call_fn(
            ctx_inner: DeviceContext, cache_iter: Int
        ) raises {mut in_bufs, imm}:
            comptime for _j in range(ngpus):
                in_bufs[_j] = InTensorType(
                    rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                        cb_inputs[_j].offset_ptr(cache_iter)
                    ),
                    row_major(Coord(Index(num_rows, num_cols))),
                )
            reducescatter[dtype=in_dtype, ngpus=ngpus, axis=0](
                in_bufs, out_shards[ctx_idx], rank_sigs, ctx_inner
            )
            if local_rows > 0:
                _launch_norm[in_dtype, num_cols](
                    rs_out_ptrs[ctx_idx],
                    normed_ptrs[ctx_idx],
                    gamma_ptr,
                    local_rows,
                    epsilon,
                    weight_offset,
                    ctx_inner,
                )

        bencher_iter_custom(bench, call_fn, ctx)

    bench_multicontext(
        b,
        bench_chained_iter,
        list_of_ctx,
        BenchId(
            "reducescatter_then_rms_norm_chained", input_id=bench_name_prefix
        ),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    # ===== Variant 4: fused reduce-scatter + RMSNorm kernel -> t_fused =====
    @always_inline
    def bench_fused_iter(
        mut bench: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises {mut in_bufs, imm}:
        @always_inline
        def call_fn(
            ctx_inner: DeviceContext, cache_iter: Int
        ) raises {mut in_bufs, imm}:
            comptime for _j in range(ngpus):
                in_bufs[_j] = InTensorType(
                    rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                        cb_inputs[_j].offset_ptr(cache_iter)
                    ),
                    row_major(Coord(Index(num_rows, num_cols))),
                )
            reducescatter_rmsnorm(
                in_bufs,
                normed_shards[ctx_idx],
                fused_sum_shards[ctx_idx],
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
        BenchId("reducescatter_rmsnorm_fused", input_id=bench_name_prefix),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    # ===== Variant 5: shape-gated dispatch =====
    # Two-launch path is caller-supplied so `comm` stays free of `nn`; the
    # dispatch auto-routes on per-rank shard size (fused below
    # `RS_NORM_FUSE_THRESHOLD`, two-launch above).
    @always_inline
    def bench_dispatch_iter(
        mut bench: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises {mut in_bufs, imm}:
        var local_rows = config.rank_units(ctx_idx)

        @always_inline
        def call_fn(
            ctx_inner: DeviceContext, cache_iter: Int
        ) raises {mut in_bufs, imm}:
            comptime for _j in range(ngpus):
                in_bufs[_j] = InTensorType(
                    rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                        cb_inputs[_j].offset_ptr(cache_iter)
                    ),
                    row_major(Coord(Index(num_rows, num_cols))),
                )

            @__parameter
            @always_inline
            def two_launch() raises:
                reducescatter[dtype=in_dtype, ngpus=ngpus, axis=0](
                    in_bufs, out_shards[ctx_idx], rank_sigs, ctx_inner
                )
                if local_rows > 0:
                    _launch_norm[in_dtype, num_cols](
                        rs_out_ptrs[ctx_idx],
                        normed_ptrs[ctx_idx],
                        gamma_ptr,
                        local_rows,
                        epsilon,
                        weight_offset,
                        ctx_inner,
                    )

            _dispatch_rs_norm[two_launch=two_launch](
                in_bufs,
                normed_shards[ctx_idx],
                fused_sum_shards[ctx_idx],
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
        BenchId("reducescatter_rmsnorm_dispatch", input_id=bench_name_prefix),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    b.dump_report()

    comptime if verify:
        _verify_results[in_dtype, ngpus, num_cols](
            num_rows,
            list_of_ctx,
            signal_buffers,
            cb_inputs,
            rank_sigs,
            gamma_dev,
            gamma_host,
            host_bufs,
            epsilon,
            weight_offset,
            config,
        )

    _ = host_bufs^
    _ = signal_buffers^
    _ = cb_inputs^
    _ = rs_out_shard^
    _ = cold_shard^
    _ = normed^
    _ = fused_sum^
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

    # Enable P2P between all GPU pairs before the read-only status check.
    _ = enable_p2p()
    if not is_p2p_enabled():
        print("P2P not enabled, skipping benchmark.")
        return

    var list_of_ctx = List[DeviceContext]()
    for i in range(num_gpus):
        list_of_ctx.append(DeviceContext(device_id=i))

    print(
        "Benchmarking reduce-scatter + RMSNorm (no quant):",
        num_gpus,
        "GPUs,",
        in_dtype,
        ",",
        num_rows,
        "x",
        num_cols,
    )

    var m = Bench(BenchConfig(num_repetitions=1))
    bench_reducescatter_rmsnorm[
        in_dtype, num_gpus, num_cols, quantize, cache_busting, verify
    ](num_rows, m, list_of_ctx)
