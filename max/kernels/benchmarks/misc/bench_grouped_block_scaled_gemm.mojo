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
"""Performance benchmarking for grouped block-scaled GEMM (sm100_structured).

Benchmarks the grouped_block_scaled_matmul kernel with realistic MoE shapes
from DeepSeek-V2 and other models.

Usage:
    # Basic benchmark (standalone)
    mojo bench_grouped_block_scaled_gemm.mojo

    # With kbench (autotuning)
    ./bazelw run //max/kernels/benchmarks/autotune:kbench -- \
        max/kernels/benchmarks/autotune/bench_grouped_block_scaled_gemm.yaml
"""

from std.math import ceildiv
from std.sys import get_defined_int

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from std.random import rand, seed
from std.utils import Index
from internal_utils import arg_parse
from layout import CoordLike, Coord, Idx, TileTensor, row_major

from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    NVFP4_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    MXFP8_SF_VECTOR_SIZE,
    NVFP4_SF_VECTOR_SIZE,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    BlockScaledMatmulConfig,
    GEMMKind,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled.grouped_block_scaled_matmul import (
    grouped_block_scaled_matmul,
)


# =============================================================================
# Benchmark Configuration
# =============================================================================


def _get_run_name[
    in_type: DType,
    out_type: DType,
](num_groups: Int, m_per_group: Int, n: Int, k: Int, cta_group: Int) -> String:
    var mode_str = "1SM" if cta_group == 1 else "2SM"
    return String(
        "grouped_block_scaled_gemm(",
        in_type,
        "->",
        out_type,
        ") : ",
        num_groups,
        " x ",
        m_per_group,
        " x ",
        n,
        " x ",
        k,
        " [",
        mode_str,
        "]",
    )


# =============================================================================
# Main Benchmark Function
# =============================================================================


def bench_grouped_block_scaled_gemm[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    scales_dtype: DType,
    num_groups: Int,
    transpose_b: Bool,
    cta_group: Int = 1,
    k_group_size: Int = 1,
    block_swizzle_size: Int = 8,
    scaling_kind: UMMAKind = UMMAKind.KIND_MXF8F6F4,
    sf_vector_size: Int = MXFP8_SF_VECTOR_SIZE,
](
    ctx: DeviceContext,
    mut bench: Bench,
    m: MType,
    n: NType,
    k: KType,
    m_override: Int = 0,
) raises:
    """Benchmark grouped block-scaled GEMM with production-style configuration.

    Args:
        ctx: Device context.
        bench: Benchmark instance.
        m: M dimension.
        n: N dimension.
        k: K dimension.
        m_override: Runtime M value. When > 0, overrides Int(m.value()) for
            allocations and problem sizes. Used by kbench with dynamic M.
    """
    comptime SF_VECTOR_SIZE = sf_vector_size
    comptime max_groups = num_groups
    # FP4 packs 2 values per byte, so K dimension arrays are halved
    comptime is_fp4 = (a_type == DType.uint8)

    # MMA shape and cluster shape
    comptime mma_shape = Index(256, 256, 32) if cta_group == 2 else Index(
        128, 128, 32
    )
    comptime cluster_shape = Index(2, 1, 1) if cta_group == 2 else Index(
        1, 1, 1
    )

    # Use m_override if provided, otherwise Int(m.value()) (compile-time)
    var M = m_override if m_override > 0 else Int(m.value())

    # For FP4, K dimension in arrays is halved (2 values packed per byte)
    comptime k_pack = 2 if is_fp4 else 1
    comptime K_ARRAY = KType.static_value // k_pack
    var k_array_val = Int(k.value()) // k_pack

    # Compute sizes (using packed K for FP4)
    var a_size = M * k_array_val
    var b_size = (
        Int(n.value())
        * k_array_val if transpose_b else k_array_val
        * Int(n.value())
    )
    var c_size = M * Int(n.value())

    # Scale factors always use logical K
    var a_scales_total = (
        ceildiv(M, SF_MN_GROUP_SIZE)
        * ceildiv(Int(k.value()), SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )
    var b_scales_total = (
        ceildiv(Int(n.value()), SF_MN_GROUP_SIZE)
        * ceildiv(Int(k.value()), SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )

    # Allocate tensors. Scale-factor buffers are filled with 1.0 (identity
    # scaling); the rest start at zero and get overwritten below.
    var a_host = List(length=a_size, fill=Scalar[a_type](0))
    var b_host = List(length=b_size, fill=Scalar[b_type](0))
    var c_host = List(length=c_size, fill=Scalar[c_type](0))
    var sfa_host = List(
        length=a_scales_total, fill=Float32(1.0).cast[scales_dtype]()
    )
    var sfb_host = List(
        length=b_scales_total, fill=Float32(1.0).cast[scales_dtype]()
    )

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var sfa_device = ctx.enqueue_create_buffer[scales_dtype](a_scales_total)
    var sfb_device = ctx.enqueue_create_buffer[scales_dtype](b_scales_total)

    # Initialize
    seed(42)
    rand(a_host)
    rand(b_host)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)
    ctx.enqueue_copy(c_device, c_host)
    ctx.enqueue_copy(sfa_device, sfa_host)
    ctx.enqueue_copy(sfb_device, sfb_host)
    ctx.synchronize()

    # Create TileTensors - 3D with batch=1
    var a_template = TileTensor(
        a_device,
        row_major(Coord(Idx[1], M, Idx[K_ARRAY])),
    )
    var b_template = TileTensor(
        b_device,
        row_major(
            Coord(
                Idx[1],
                Idx[NType.static_value if transpose_b else K_ARRAY],
                Idx[K_ARRAY if transpose_b else NType.static_value],
            )
        ),
    )
    var c_template = TileTensor(
        c_device,
        row_major(Coord(Idx[1], M, n)),
    )

    # Scale factor template tensors - 5D with batch=1 and merged last dims
    var sfa_template = TileTensor(
        sfa_device,
        row_major(
            Coord(
                Idx[1],
                ceildiv(M, SF_MN_GROUP_SIZE),
                Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1] * SF_ATOM_K],
            )
        ),
    )
    var sfb_template = TileTensor(
        sfb_device,
        row_major(
            Coord(
                Idx[1],
                ceildiv(Int(n.value()), SF_MN_GROUP_SIZE),
                Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1] * SF_ATOM_K],
            )
        ),
    )

    # Setup pointer arrays
    var problem_sizes_host = List(length=max_groups * 4, fill=Int32(0))
    for g in range(max_groups):
        problem_sizes_host[g * 4 + 0] = Int32(M)
        problem_sizes_host[g * 4 + 1] = Int32(Int(n.value()))
        problem_sizes_host[g * 4 + 2] = Int32(Int(k.value()))
        problem_sizes_host[g * 4 + 3] = 1

    var problem_sizes_device = ctx.enqueue_create_buffer[.int32](max_groups * 4)
    ctx.enqueue_copy(problem_sizes_device, problem_sizes_host)

    var a_ptrs_host = List(
        length=max_groups, fill=UInt64(Int(a_device.unsafe_ptr()))
    )
    var b_ptrs_host = List(
        length=max_groups, fill=UInt64(Int(b_device.unsafe_ptr()))
    )
    var c_ptrs_host = List(
        length=max_groups, fill=UInt64(Int(c_device.unsafe_ptr()))
    )
    var sfa_ptrs_host = List(
        length=max_groups, fill=UInt64(Int(sfa_device.unsafe_ptr()))
    )
    var sfb_ptrs_host = List(
        length=max_groups, fill=UInt64(Int(sfb_device.unsafe_ptr()))
    )

    var a_ptrs_device = ctx.enqueue_create_buffer[.uint64](max_groups)
    var b_ptrs_device = ctx.enqueue_create_buffer[.uint64](max_groups)
    var c_ptrs_device = ctx.enqueue_create_buffer[.uint64](max_groups)
    var sfa_ptrs_device = ctx.enqueue_create_buffer[.uint64](max_groups)
    var sfb_ptrs_device = ctx.enqueue_create_buffer[.uint64](max_groups)

    ctx.enqueue_copy(a_ptrs_device, a_ptrs_host)
    ctx.enqueue_copy(b_ptrs_device, b_ptrs_host)
    ctx.enqueue_copy(c_ptrs_device, c_ptrs_host)
    ctx.enqueue_copy(sfa_ptrs_device, sfa_ptrs_host)
    ctx.enqueue_copy(sfb_ptrs_device, sfb_ptrs_host)
    ctx.synchronize()

    var problem_sizes_tensor_host = TileTensor(
        problem_sizes_host,
        row_major(Coord(Idx[max_groups], Idx[4])),
    )

    var a_ptrs_tensor = TileTensor(
        a_ptrs_device,
        row_major(Coord(Idx[max_groups], Idx[1])),
    )
    var b_ptrs_tensor = TileTensor(
        b_ptrs_device,
        row_major(Coord(Idx[max_groups], Idx[1])),
    )
    var c_ptrs_tensor = TileTensor(
        c_ptrs_device,
        row_major(Coord(Idx[max_groups], Idx[1])),
    )
    var sfa_ptrs_tensor = TileTensor(
        sfa_ptrs_device,
        row_major(Coord(Idx[max_groups], Idx[1])),
    )
    var sfb_ptrs_tensor = TileTensor(
        sfb_ptrs_device,
        row_major(Coord(Idx[max_groups], Idx[1])),
    )

    comptime BM = mma_shape[0] // cta_group
    comptime BN = mma_shape[1] // cta_group
    var tiles_per_group = ceildiv(M, BM) * ceildiv(Int(n.value()), BN)
    var total_tiles = tiles_per_group * num_groups

    # Configuration matching production benchmark
    comptime config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ](
        scaling_kind=scaling_kind,
        cluster_shape=cluster_shape,
        mma_shape=mma_shape,
        block_swizzle_size=block_swizzle_size,
        cta_group=cta_group,
        k_group_size=k_group_size,
        num_accum_pipeline_stages=2,
        gemm_kind=GEMMKind.GMM,
    )

    # Total FLOPs for all groups
    var total_flops = 2 * M * Int(n.value()) * Int(k.value()) * num_groups

    @always_inline
    def bench_func(
        mut bencher: Bencher,
    ) {
        var a_ptrs_tensor,
        var b_ptrs_tensor,
        var c_ptrs_tensor,
        var sfa_ptrs_tensor,
        var sfb_ptrs_tensor,
        var problem_sizes_tensor_host,
        var a_template,
        var b_template,
        var c_template,
        var sfa_template,
        var sfb_template,
        var total_tiles,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
            grouped_block_scaled_matmul[
                transpose_b=transpose_b,
                max_groups=max_groups,
                config=config,
            ](
                a_ptrs_tensor,
                b_ptrs_tensor,
                c_ptrs_tensor,
                sfa_ptrs_tensor,
                sfb_ptrs_tensor,
                problem_sizes_tensor_host,
                num_groups,
                total_tiles,
                a_template,
                b_template,
                c_template,
                sfa_template,
                sfb_template,
                ctx,
            )

        bencher_iter_custom(bencher, kernel_launch, ctx)

    bench.bench_function(
        bench_func,
        BenchId(
            _get_run_name[a_type, c_type](
                num_groups, M, Int(n.value()), Int(k.value()), cta_group
            )
        ),
        [ThroughputMeasure(BenchMetric.flops, total_flops)],
    )
    _ = sfb_ptrs_host^
    _ = sfa_ptrs_host^
    _ = c_ptrs_host^
    _ = b_ptrs_host^
    _ = a_ptrs_host^
    _ = problem_sizes_host^
    _ = sfb_host^
    _ = sfa_host^
    _ = c_host^
    _ = b_host^
    _ = a_host^


# =============================================================================
# Main Entry Point
# =============================================================================


def main() raises:
    # Compile-time parameters (from kbench YAML or defaults)
    comptime N = get_defined_int["N", 0]()
    comptime K = get_defined_int["K", 0]()
    comptime num_groups = get_defined_int["num_groups", 0]()
    comptime cta_group = get_defined_int["cta_group", 1]()
    comptime k_group_size = get_defined_int["k_group_size", 1]()
    comptime block_swizzle_size = get_defined_int["block_swizzle_size", 8]()

    # Runtime parameters (from kbench YAML $-prefixed or defaults)
    var M = Int(arg_parse("M", 0))

    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime scales_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    var b = Bench()

    with DeviceContext() as ctx:
        comptime if N > 0 and K > 0 and num_groups > 0:
            # kbench mode: use env parameters
            bench_grouped_block_scaled_gemm[
                a_type,
                b_type,
                c_type,
                scales_dtype,
                num_groups=num_groups,
                transpose_b=transpose_b,
                cta_group=cta_group,
                k_group_size=k_group_size,
                block_swizzle_size=block_swizzle_size,
            ](ctx, b, Idx[0], Idx[N], Idx[K], M)
        else:
            # Standalone mode: run default shapes
            print("=" * 70)
            print("Benchmark: Grouped Block-Scaled GEMM (sm100_structured)")
            print("=" * 70)

            print("\n=== DeepSeek-V2 MoE Decode Shapes (small M) ===")

            bench_grouped_block_scaled_gemm[
                a_type,
                b_type,
                c_type,
                scales_dtype,
                num_groups=32,
                transpose_b=transpose_b,
                cta_group=1,
                k_group_size=1,
                block_swizzle_size=8,
            ](ctx, b, Idx[128], Idx[4096], Idx[7168])

            bench_grouped_block_scaled_gemm[
                a_type,
                b_type,
                c_type,
                scales_dtype,
                num_groups=32,
                transpose_b=transpose_b,
                cta_group=1,
                k_group_size=1,
                block_swizzle_size=8,
            ](ctx, b, Idx[128], Idx[7168], Idx[2048])

            print("\n=== DeepSeek-V2 MoE Prefill Shapes (large M) ===")

            bench_grouped_block_scaled_gemm[
                a_type,
                b_type,
                c_type,
                scales_dtype,
                num_groups=32,
                transpose_b=transpose_b,
                cta_group=1,
                k_group_size=1,
                block_swizzle_size=8,
            ](ctx, b, Idx[4096], Idx[4096], Idx[7168])

            bench_grouped_block_scaled_gemm[
                a_type,
                b_type,
                c_type,
                scales_dtype,
                num_groups=32,
                transpose_b=transpose_b,
                cta_group=1,
                k_group_size=1,
                block_swizzle_size=8,
            ](ctx, b, Idx[4096], Idx[7168], Idx[2048])

            # === NVFP4 Benchmarks ===
            comptime fp4_a_type = DType.uint8
            comptime fp4_b_type = DType.uint8
            comptime fp4_c_type = DType.bfloat16
            comptime fp4_scales_dtype = NVFP4_SF_DTYPE

            print("\n=== NVFP4 DeepSeek-V2 MoE Decode Shapes (small M) ===")

            bench_grouped_block_scaled_gemm[
                fp4_a_type,
                fp4_b_type,
                fp4_c_type,
                fp4_scales_dtype,
                num_groups=32,
                transpose_b=transpose_b,
                cta_group=1,
                k_group_size=1,
                block_swizzle_size=8,
                scaling_kind=UMMAKind.KIND_MXF4NVF4,
                sf_vector_size=NVFP4_SF_VECTOR_SIZE,
            ](ctx, b, Idx[128], Idx[4096], Idx[7168])

            bench_grouped_block_scaled_gemm[
                fp4_a_type,
                fp4_b_type,
                fp4_c_type,
                fp4_scales_dtype,
                num_groups=32,
                transpose_b=transpose_b,
                cta_group=1,
                k_group_size=1,
                block_swizzle_size=8,
                scaling_kind=UMMAKind.KIND_MXF4NVF4,
                sf_vector_size=NVFP4_SF_VECTOR_SIZE,
            ](ctx, b, Idx[128], Idx[7168], Idx[2048])

            print("\n=== NVFP4 DeepSeek-V2 MoE Prefill Shapes (large M) ===")

            bench_grouped_block_scaled_gemm[
                fp4_a_type,
                fp4_b_type,
                fp4_c_type,
                fp4_scales_dtype,
                num_groups=32,
                transpose_b=transpose_b,
                cta_group=1,
                k_group_size=1,
                block_swizzle_size=8,
                scaling_kind=UMMAKind.KIND_MXF4NVF4,
                sf_vector_size=NVFP4_SF_VECTOR_SIZE,
            ](ctx, b, Idx[4096], Idx[4096], Idx[7168])

            bench_grouped_block_scaled_gemm[
                fp4_a_type,
                fp4_b_type,
                fp4_c_type,
                fp4_scales_dtype,
                num_groups=32,
                transpose_b=transpose_b,
                cta_group=1,
                k_group_size=1,
                block_swizzle_size=8,
                scaling_kind=UMMAKind.KIND_MXF4NVF4,
                sf_vector_size=NVFP4_SF_VECTOR_SIZE,
            ](ctx, b, Idx[4096], Idx[7168], Idx[2048])

    b.dump_report()
    print("\n" + "=" * 70)
