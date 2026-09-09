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

"""RMSNorm -> GEMM benchmark for PDL overlap (single GPU, bf16).

PDL makes the GEMM resident while the norm is still running, so the two kernels
genuinely overlap and neither can be timed on its own. Both are therefore
enqueued in one closure on one `DeviceContext` -- one stream -- and the reported
figure is the average per-iteration stream wall-clock across both. `norm_only` /
`gemm_only` exist for attribution only; summing them double-counts the overlap.

Five paired configurations, identical in shape and buffers, differing only in
where the norm releases its dependents and whether the GEMM prefetches weight
tiles before its PDL wait. `pair_baseline_today` is the one to beat: main
already runs the GEMM at `PDLLevel.ON` and the norm releases at its end, so
measuring against `pair_none` would credit this change with overlap main already
has.

Weights stream through `CacheBustingBuffer` so they are read cold, and the
window count is printed so a collapsed buffer cannot pass unnoticed. The norm's
input and output are deliberately left cache-warm: at decode sizes both are well
under a megabyte and are written by the immediately preceding kernel in
production too, so busting them would model a residency the real pipeline does
not have.

Iterations run back-to-back, so a GEMM also overlaps the next iteration's norm.
That makes the result a steady-state pipelined figure rather than an isolated
pair -- read the deltas as the mechanism's headroom, not as a predicted
end-to-end gain.
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

from max.benchmark import bencher_iter_custom
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.primitives.grid_controls import PDLLevel

from internal_utils import CacheBustingBuffer, arg_parse
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
)
from nn.normalization import rms_norm_gpu
from std.utils.index import Index

# Paired variants, then the two single-kernel attribution runs.
comptime NUM_PAIRS = 5
comptime NUM_VARIANTS = NUM_PAIRS + 2


def bench_rms_norm_gemm_pdl[
    dtype: DType,
    num_cols: Int,
    gemm_n: Int,
    mma_m: Int = 128,
    mma_n: Int = 32,
    cta_group: Int = 2,
    k_group_size: Int = 2,
    num_pipeline_stages: Int = 16,
    swap_ab: Bool = False,
    prefetch_tiles_n: Int = 2,
](num_rows: Int, mut b: Bench, ctx: DeviceContext) raises:
    """Bench the RMSNorm -> GEMM pair across the PDL configurations.

    Parameters:
        dtype: Element type of the norm and the GEMM (bf16).
        num_cols: Hidden size; the norm's column count and the GEMM's K.
        gemm_n: N of the consumer GEMM (static, as the SM100 kernel requires).
        mma_m: Consumer GEMM MMA tile M.
        mma_n: Consumer GEMM MMA tile N.
        cta_group: Consumer GEMM CTA group.
        k_group_size: Consumer GEMM K tiles per pipeline stage.
        num_pipeline_stages: Consumer GEMM input pipeline depth.
        swap_ab: Whether the consumer GEMM swaps its A and B operands.
        prefetch_tiles_n: Weight tiles the consumer issues before its PDL wait.
            Must not exceed the config's group pipeline stages.

    Args:
        num_rows: Token count (batch x tokens per step).
        b: Benchmark harness.
        ctx: Device context.
    """
    comptime assert dtype == .bfloat16, "this bench is bf16-only"

    comptime simd_size = simd_width_of[dtype, target=get_gpu_target()]()
    comptime MMA_K = 16

    comptime gemm_config_off = MatmulConfig[dtype, dtype, dtype, True](
        cluster_shape=Index(cta_group, 1, 1),
        mma_shape=Index(mma_m, mma_n, MMA_K),
        cta_group=cta_group,
        num_pipeline_stages=num_pipeline_stages,
        num_accum_pipeline_stages=1,
        num_clc_pipeline_stages=0,
        k_group_size=k_group_size,
        AB_swapped=swap_ab,
    )
    comptime gemm_config_prefetch = MatmulConfig[dtype, dtype, dtype, True](
        cluster_shape=Index(cta_group, 1, 1),
        mma_shape=Index(mma_m, mma_n, MMA_K),
        cta_group=cta_group,
        num_pipeline_stages=num_pipeline_stages,
        num_accum_pipeline_stages=1,
        num_clc_pipeline_stages=0,
        k_group_size=k_group_size,
        AB_swapped=swap_ab,
        prefetch_tiles_n=prefetch_tiles_n,
    )

    var a_elems = num_rows * num_cols
    var w_elems = gemm_n * num_cols
    var c_elems = num_rows * gemm_n
    var epsilon = Float32(1e-5)
    # 0.0 matches the plain RMSNorm this pair models (no Gemma-style offset).
    var weight_offset = Scalar[dtype](0.0)

    var a_host = List[Scalar[dtype]](unsafe_uninit_length=a_elems)
    rand(a_host.unsafe_ptr(), a_elems)
    var gamma_host = List(length=num_cols, fill=Scalar[dtype](0))
    for c in range(num_cols):
        gamma_host[c] = (Float64(c + num_cols) / Float64(num_cols)).cast[
            dtype
        ]()
    var w_base = List[Scalar[dtype]](unsafe_uninit_length=w_elems)
    rand(w_base.unsafe_ptr(), w_elems)

    var cb_weights = CacheBustingBuffer[dtype](w_elems, simd_size, ctx, True)
    print(
        "cache-busting windows: weights=",
        cb_weights.buffer_size // cb_weights.stride,
        " (1 means the buster COLLAPSED -- raise budget_bytes)",
        sep="",
    )
    # Fill the WHOLE allocation, not just the first window: an iteration landing
    # on an unwritten window would read garbage, and NaNs there make the
    # bit-identity gate below vacuous.
    var w_all = List[Scalar[dtype]](
        unsafe_uninit_length=cb_weights.alloc_size()
    )
    for j in range(cb_weights.alloc_size()):
        w_all[j] = w_base[j % w_elems]
    ctx.enqueue_copy(cb_weights.device_buffer(), w_all)

    var a_raw_dev = ctx.enqueue_create_buffer[dtype](a_elems)
    var a_normed_dev = ctx.enqueue_create_buffer[dtype](a_elems)
    var gamma_dev = ctx.enqueue_create_buffer[dtype](num_cols)
    var c_dev = ctx.enqueue_create_buffer[dtype](c_elems)
    var c_ref_dev = ctx.enqueue_create_buffer[dtype](c_elems)
    ctx.enqueue_copy(a_raw_dev, a_host)
    ctx.enqueue_copy(gamma_dev, gamma_host)

    var a_raw = TileTensor(a_raw_dev, row_major(Coord(num_rows, Idx[num_cols])))
    var a_normed = TileTensor(
        a_normed_dev, row_major(Coord(num_rows, Idx[num_cols]))
    )
    var gamma = TileTensor(gamma_dev, row_major(Coord(Idx[num_cols])))
    var c_out = TileTensor(c_dev, row_major(Coord(num_rows, Idx[gemm_n])))
    var c_ref = TileTensor(c_ref_dev, row_major(Coord(num_rows, Idx[gemm_n])))
    var norm_shape = Index(num_rows, num_cols)

    # `[N, K]` (transpose_b) with BOTH dims static, as the SM100 kernel needs.
    # Built from the rotating cache-bust window rather than the base pointer so
    # each iteration streams cold weights -- the cost the prefetch must hide.
    comptime WeightType = TileTensor[
        dtype,
        type_of(row_major(Coord(Idx[gemm_n], Idx[num_cols]))),
        ImmutAnyOrigin,
    ]

    comptime ARawType = type_of(a_raw)
    comptime ANormedType = type_of(a_normed)
    comptime GammaType = type_of(gamma)
    comptime CBWeightsType = type_of(cb_weights)
    comptime NormShapeType = type_of(norm_shape)

    @always_inline
    @__copy_capture(a_normed)
    @__parameter
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) -> None:
        a_normed.raw_store[width=width, alignment=alignment](
            a_normed.layout(coords), val
        )

    @always_inline
    def run_pair[
        norm_pdl: PDLLevel,
        gemm_pdl: PDLLevel,
        use_prefetch: Bool,
        norm_only: Bool = False,
        gemm_only: Bool = False,
    ](
        ctx_inner: DeviceContext,
        cache_iter: Int,
        mut c_tile: TileTensor[mut=True, dtype, ...],
        a_raw: ARawType,
        a_normed: ANormedType,
        mut cb_weights: CBWeightsType,
        gamma: GammaType,
        epsilon: Float32,
        weight_offset: Scalar[dtype],
        norm_shape: NormShapeType,
    ) raises {}:
        @always_inline
        @__copy_capture(a_raw)
        @__parameter
        def input_fn[width: Int](coords: Coord) -> SIMD[dtype, width]:
            return a_raw.raw_load[width=width](a_raw.layout(coords))

        comptime if not gemm_only:
            rms_norm_gpu[
                2,
                input_fn,
                output_fn,
                multiply_before_cast=True,
                pdl_level=norm_pdl,
            ](Coord(norm_shape), gamma, epsilon, weight_offset, ctx_inner)

        comptime if not norm_only:
            comptime gemm_config = (
                gemm_config_prefetch if use_prefetch else gemm_config_off
            )
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=True, config=gemm_config, pdl_level=gemm_pdl
            ](
                c_tile,
                a_normed,
                WeightType(
                    rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                        cb_weights.offset_ptr(cache_iter)
                    ),
                    row_major(Coord(Idx[gemm_n], Idx[num_cols])),
                ),
                ctx_inner,
            )

    # --- Integrity gate, before any timing: PDL and weight prefetch reorder
    # loads, never arithmetic, so the GEMM output must be BIT-identical across
    # all variants. If this diverges, the GEMM is reading the norm's output
    # before it is written and every timing below is meaningless. ---
    run_pair[PDLLevel.OFF, PDLLevel.OFF, False](
        ctx,
        0,
        c_ref,
        a_raw,
        a_normed,
        cb_weights,
        gamma,
        epsilon,
        weight_offset,
        norm_shape,
    )
    ctx.synchronize()
    var want = List[Scalar[dtype]](unsafe_uninit_length=c_elems)
    ctx.enqueue_copy(want, c_ref_dev)
    ctx.synchronize()

    comptime for v in range(1, NUM_PAIRS):
        comptime norm_pdl = (
            PDLLevel.OVERLAP_AT_END if v <= 2 else PDLLevel.OVERLAP_AT_BEGINNING
        )
        comptime use_pf = v == 2 or v == 4
        run_pair[norm_pdl, PDLLevel.ON, use_pf](
            ctx,
            0,
            c_out,
            a_raw,
            a_normed,
            cb_weights,
            gamma,
            epsilon,
            weight_offset,
            norm_shape,
        )
        ctx.synchronize()
        var got = List[Scalar[dtype]](unsafe_uninit_length=c_elems)
        ctx.enqueue_copy(got, c_dev)
        ctx.synchronize()
        var mismatch = 0
        for j in range(c_elems):
            if got[j] != want[j]:
                mismatch += 1
        if mismatch != 0:
            raise Error(
                String(
                    "variant ",
                    v,
                    " changed the GEMM result: ",
                    mismatch,
                    (
                        " mismatching elements vs PDL off. The GEMM is reading"
                        " the norm's output before it is written -- do NOT"
                        " trust any timing from this run."
                    ),
                )
            )
        _ = got^
    print("verified: every PDL variant is bit-identical to PDL off")

    var bench_prefix = String(
        "M=", num_rows, "/K=", num_cols, "/N=", gemm_n, "/pf=", prefetch_tiles_n
    )
    # The norm moves its stream in and out; the GEMM streams the weights.
    # Weights dominate at decode M, which is the point of the overlap.
    var total_bytes = (2 * a_elems + w_elems) * size_of[dtype]()

    comptime for v in range(NUM_VARIANTS):
        comptime solo = v >= NUM_PAIRS
        comptime norm_pdl = (
            PDLLevel.OFF if v == 0
            or solo else PDLLevel.OVERLAP_AT_END if v
            <= 2 else PDLLevel.OVERLAP_AT_BEGINNING
        )
        comptime gemm_pdl = PDLLevel.OFF if v == 0 or solo else PDLLevel.ON
        comptime use_pf = v == 2 or v == 4
        comptime vname = (
            "pair_none" if v
            == 0 else "pair_baseline_today" if v
            == 1 else "pair_end_prefetch" if v
            == 2 else "pair_early_launch" if v
            == 3 else "pair_early_prefetch" if v
            == 4 else "norm_only" if v
            == 5 else "gemm_only"
        )

        @always_inline
        def call_fn(
            ctx_inner: DeviceContext, cache_iter: Int
        ) raises {
            mut c_out,
            mut cb_weights,
            imm a_raw,
            imm a_normed,
            imm gamma,
            var epsilon,
            var weight_offset,
            var norm_shape,
        }:
            run_pair[
                norm_pdl,
                gemm_pdl,
                use_pf,
                norm_only=v == 5,
                gemm_only=v == 6,
            ](
                ctx_inner,
                cache_iter,
                c_out,
                a_raw,
                a_normed,
                cb_weights,
                gamma,
                epsilon,
                weight_offset,
                norm_shape,
            )

        @always_inline
        def bench_fn(mut bench: Bencher) raises {imm}:
            bencher_iter_custom(bench, call_fn, ctx)

        b.bench_function(
            bench_fn,
            BenchId(vname, input_id=bench_prefix),
            [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
        )

    b.dump_report()
    _ = a_host^
    _ = gamma_host^
    _ = w_base^
    _ = w_all^
    _ = want^
    _ = cb_weights^
    _ = a_raw_dev^
    _ = a_normed_dev^
    _ = gamma_dev^
    _ = c_dev^
    _ = c_ref_dev^


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .bfloat16]()
    comptime num_cols = get_defined_int["num_cols", 6144]()
    comptime gemm_n = get_defined_int["gemm_n", 2624]()
    comptime mma_m = get_defined_int["mma_m", 128]()
    comptime mma_n = get_defined_int["mma_n", 32]()
    comptime cta_group = get_defined_int["cta_group", 2]()
    comptime k_group_size = get_defined_int["k_group_size", 2]()
    comptime num_pipeline_stages = get_defined_int["num_pipeline_stages", 16]()
    comptime swap_ab = get_defined_int["swap_ab", 0]() != 0
    comptime prefetch_tiles_n = get_defined_int["prefetch_tiles_n", 2]()

    # GLM 5.2 decode: batch 8 x (1 target + 5 speculative) tokens.
    var num_rows = Int(arg_parse("num_rows", 48))
    # >1 to get run-to-run spread: the deltas here are single-digit percent,
    # which one repetition cannot separate from box noise.
    var num_reps = Int(arg_parse("num_reps", 3))

    with DeviceContext() as ctx:
        print(
            "Paired RMSNorm -> GEMM, PDL: ",
            dtype,
            ", M=",
            num_rows,
            " K=",
            num_cols,
            " N=",
            gemm_n,
            sep="",
        )
        var m = Bench(BenchConfig(num_repetitions=num_reps))
        bench_rms_norm_gemm_pdl[
            dtype,
            num_cols,
            gemm_n,
            mma_m=mma_m,
            mma_n=mma_n,
            cta_group=cta_group,
            k_group_size=k_group_size,
            num_pipeline_stages=num_pipeline_stages,
            swap_ab=swap_ab,
            prefetch_tiles_n=prefetch_tiles_n,
        ](num_rows, m, ctx)
