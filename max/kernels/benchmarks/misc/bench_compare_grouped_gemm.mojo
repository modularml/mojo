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
"""Apples-to-apples comparison: cuBLAS (per-group) vs Structured Grouped GEMM.

Both sides use identical block-scaled data formats (MXFP8 or NVFP4):
- MXFP8: float8_e4m3fn A/B with UE8M0 scale factors
- NVFP4: uint8 (packed FP4) A/B with float8_e4m3fn scale factors
- C: bfloat16

The cuBLAS baseline calls vendor_blas.matmul once per group (sequentially),
which is what you'd do without a grouped/persistent kernel. The structured
kernel processes all groups in a single persistent launch.
"""

from std.math import ceildiv

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
from layout import CoordLike, Coord, Idx, TileTensor, row_major

import linalg.matmul.vendor.blas as vendor_blas
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    NVFP4_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    MXFP8_SF_VECTOR_SIZE,
    NVFP4_SF_VECTOR_SIZE,
)

# Structured kernel
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    BlockScaledMatmulConfig,
    GEMMKind,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled.grouped_block_scaled_matmul import (
    grouped_block_scaled_matmul,
)


def bench_cublas_per_group[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    scales_dtype: DType,
    num_groups: Int,
    sf_vector_size: Int = MXFP8_SF_VECTOR_SIZE,
](ctx: DeviceContext, mut bench: Bench, m: MType, n: NType, k: KType) raises:
    """Benchmark cuBLAS called once per group (sequential baseline).

    This represents the naive approach without a grouped/persistent kernel:
    iterate over groups and call cuBLAS matmul for each one.
    """

    comptime SF_VECTOR_SIZE = sf_vector_size
    comptime transpose_b = True
    comptime is_fp4 = (a_type == DType.uint8)
    comptime k_pack = 2 if is_fp4 else 1
    comptime K_ARRAY = KType.static_value // k_pack
    var k_array_val = Int(k.value()) // k_pack

    var a_size = Int(m.value()) * k_array_val
    var b_size = Int(n.value()) * k_array_val
    var c_size = Int(m.value()) * Int(n.value())

    var a_scales_total = (
        ceildiv(Int(m.value()), SF_MN_GROUP_SIZE)
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

    var a_host = List(length=a_size, fill=Scalar[a_type](0))
    var b_host = List(length=b_size, fill=Scalar[b_type](0))
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

    seed(42)
    rand(a_host)
    rand(b_host)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)
    ctx.enqueue_copy(sfa_device, sfa_host)
    ctx.enqueue_copy(sfb_device, sfb_host)
    ctx.synchronize()

    var a_shape = Coord(m, Idx[K_ARRAY])
    var b_shape = Coord(n, Idx[K_ARRAY])
    var c_shape = Coord(m, n)
    var a_scales_shape = Coord(
        ceildiv(Int(m.value()), SF_MN_GROUP_SIZE),
        Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    var b_scales_shape = Coord(
        ceildiv(Int(n.value()), SF_MN_GROUP_SIZE),
        Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )

    var a_tensor = TileTensor(a_device, row_major(a_shape))
    var b_tensor = TileTensor(b_device, row_major(b_shape))
    var c_tensor = TileTensor(c_device, row_major(c_shape))
    var sfa_tensor = TileTensor(sfa_device, row_major(a_scales_shape))
    var sfb_tensor = TileTensor(sfb_device, row_major(b_scales_shape))

    # FLOPs use logical K (not packed)
    var total_flops = (
        2 * Int(m.value()) * Int(n.value()) * Int(k.value()) * num_groups
    )

    @always_inline
    def bench_func(
        mut bencher: Bencher,
    ) {
        var a_tensor,
        var b_tensor,
        var c_tensor,
        var sfa_tensor,
        var sfb_tensor,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
            # Call cuBLAS once per group (sequential)
            for _g in range(num_groups):
                vendor_blas.matmul(
                    ctx,
                    c_tensor,
                    a_tensor,
                    b_tensor,
                    a_scales=sfa_tensor.as_immut(),
                    b_scales=sfb_tensor.as_immut(),
                    transpose_b=transpose_b,
                    c_row_major=True,
                )

        bencher_iter_custom(bencher, kernel_launch, ctx)

    var fmt = String("NVFP4") if is_fp4 else String("MXFP8")

    bench.bench_function(
        bench_func,
        BenchId(
            String(
                "cuBLAS(",
                fmt,
                ",per-group) : ",
                num_groups,
                " x ",
                Int(m.value()),
                " x ",
                Int(n.value()),
                " x ",
                Int(k.value()),
            )
        ),
        [ThroughputMeasure(BenchMetric.flops, total_flops)],
    )
    _ = sfb_host^
    _ = sfa_host^
    _ = b_host^
    _ = a_host^


def bench_structured_kernel[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    scales_dtype: DType,
    num_groups: Int,
    mma_m: Int = 128,
    mma_n: Int = 128,
    k_grp_size: Int = 1,
    scaling_kind: UMMAKind = UMMAKind.KIND_MXF8F6F4,
    sf_vector_size: Int = MXFP8_SF_VECTOR_SIZE,
](ctx: DeviceContext, mut bench: Bench, m: MType, n: NType, k: KType) raises:
    """Benchmark structured grouped_block_scaled_matmul."""

    comptime SF_VECTOR_SIZE = sf_vector_size
    comptime max_groups = num_groups
    comptime transpose_b = True
    comptime mma_shape = Index(mma_m, mma_n, 32)
    comptime cluster_shape = Index(1, 1, 1)
    comptime is_fp4 = (a_type == DType.uint8)
    comptime k_pack = 2 if is_fp4 else 1
    comptime K_ARRAY = KType.static_value // k_pack
    var k_array_val = Int(k.value()) // k_pack

    var a_size = Int(m.value()) * k_array_val
    var b_size = Int(n.value()) * k_array_val
    var c_size = Int(m.value()) * Int(n.value())

    var a_scales_total = (
        ceildiv(Int(m.value()), SF_MN_GROUP_SIZE)
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

    var a_host = List(length=a_size, fill=Scalar[a_type](0))
    var b_host = List(length=b_size, fill=Scalar[b_type](0))
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

    seed(42)
    rand(a_host)
    rand(b_host)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)
    ctx.enqueue_copy(sfa_device, sfa_host)
    ctx.enqueue_copy(sfb_device, sfb_host)
    ctx.synchronize()

    # Template tensors - 3D with batch=1
    var a_template = TileTensor(
        a_device,
        row_major(Coord(Idx[1], m, Idx[K_ARRAY])),
    )
    var b_template = TileTensor(
        b_device,
        row_major(Coord(Idx[1], n, Idx[K_ARRAY])),
    )
    var c_template = TileTensor(
        c_device,
        row_major(Coord(Idx[1], m, n)),
    )

    # Scale factor template tensors - 5D with batch=1 and merged last dims
    var sfa_template = TileTensor(
        sfa_device,
        row_major(
            Coord(
                Idx[1],
                ceildiv(Int(m.value()), SF_MN_GROUP_SIZE),
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

    var problem_sizes_host = List(length=max_groups * 4, fill=Int32(0))
    for g in range(max_groups):
        problem_sizes_host[g * 4 + 0] = Int32(Int(m.value()))
        problem_sizes_host[g * 4 + 1] = Int32(Int(n.value()))
        problem_sizes_host[g * 4 + 2] = Int32(Int(k.value()))  # Logical K
        problem_sizes_host[g * 4 + 3] = 1

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

    var problem_sizes_tensor = TileTensor(
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

    comptime BM = mma_shape[0]
    comptime BN = mma_shape[1]
    var tiles_per_group = ceildiv(Int(m.value()), BM) * ceildiv(
        Int(n.value()), BN
    )
    var total_tiles = tiles_per_group * num_groups

    comptime config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ](
        scaling_kind=scaling_kind,
        cluster_shape=cluster_shape,
        mma_shape=mma_shape,
        block_swizzle_size=8,
        cta_group=1,
        k_group_size=k_grp_size,
        num_accum_pipeline_stages=2,
        gemm_kind=GEMMKind.GMM,
    )

    # FLOPs use logical K (not packed)
    var total_flops = (
        2 * Int(m.value()) * Int(n.value()) * Int(k.value()) * num_groups
    )

    @always_inline
    def bench_func(
        mut bencher: Bencher,
    ) {
        var a_ptrs_tensor,
        var b_ptrs_tensor,
        var c_ptrs_tensor,
        var sfa_ptrs_tensor,
        var sfb_ptrs_tensor,
        var problem_sizes_tensor,
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
                problem_sizes_tensor,
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

    var fmt = String("NVFP4") if is_fp4 else String("MXFP8")

    bench.bench_function(
        bench_func,
        BenchId(
            String(
                "STRUCTURED(",
                fmt,
                ",",
                mma_m,
                "x",
                mma_n,
                ",k_grp=",
                k_grp_size,
                ") : ",
                num_groups,
                " x ",
                Int(m.value()),
                " x ",
                Int(n.value()),
                " x ",
                Int(k.value()),
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
    _ = b_host^
    _ = a_host^


def main() raises:
    print("=" * 70)
    print("Comparison: cuBLAS (per-group) vs Structured Grouped GEMM")
    print("=" * 70)
    print()
    print("Both sides use identical block-scaled formats (MXFP8 and NVFP4).")
    print("cuBLAS baseline: vendor_blas.matmul called once per group")
    print("Structured:      grouped_block_scaled_matmul (persistent)")
    print()

    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime scales_dtype = MXFP8_SF_DTYPE

    comptime fp4_a_type = DType.uint8
    comptime fp4_b_type = DType.uint8
    comptime fp4_c_type = DType.bfloat16
    comptime fp4_scales_dtype = NVFP4_SF_DTYPE

    var b = Bench()

    with DeviceContext() as ctx:
        # =====================================================================
        # MXFP8: DeepSeek-V2 Prefill: 32 groups x 4096 x 4096 x 7168
        # =====================================================================
        print("=== MXFP8 Prefill: 32 groups x 4096 x 4096 x 7168 ===")

        bench_cublas_per_group[
            a_type,
            b_type,
            c_type,
            scales_dtype,
            num_groups=32,
        ](ctx, b, Idx[4096], Idx[4096], Idx[7168])

        bench_structured_kernel[
            a_type,
            b_type,
            c_type,
            scales_dtype,
            num_groups=32,
            mma_m=128,
            mma_n=128,
            k_grp_size=1,
        ](ctx, b, Idx[4096], Idx[4096], Idx[7168])

        # =====================================================================
        # MXFP8: DeepSeek-V2 Decode: 32 groups x 128 x 4096 x 7168
        # =====================================================================
        print("\n=== MXFP8 Decode: 32 groups x 128 x 4096 x 7168 ===")

        bench_cublas_per_group[
            a_type,
            b_type,
            c_type,
            scales_dtype,
            num_groups=32,
        ](ctx, b, Idx[128], Idx[4096], Idx[7168])

        bench_structured_kernel[
            a_type,
            b_type,
            c_type,
            scales_dtype,
            num_groups=32,
            mma_m=128,
            mma_n=128,
            k_grp_size=1,
        ](ctx, b, Idx[128], Idx[4096], Idx[7168])

        # =====================================================================
        # NVFP4: DeepSeek-V2 Prefill: 32 groups x 4096 x 4096 x 7168
        # =====================================================================
        print("\n=== NVFP4 Prefill: 32 groups x 4096 x 4096 x 7168 ===")

        bench_cublas_per_group[
            fp4_a_type,
            fp4_b_type,
            fp4_c_type,
            fp4_scales_dtype,
            num_groups=32,
            sf_vector_size=NVFP4_SF_VECTOR_SIZE,
        ](ctx, b, Idx[4096], Idx[4096], Idx[7168])

        bench_structured_kernel[
            fp4_a_type,
            fp4_b_type,
            fp4_c_type,
            fp4_scales_dtype,
            num_groups=32,
            mma_m=128,
            mma_n=128,
            k_grp_size=1,
            scaling_kind=UMMAKind.KIND_MXF4NVF4,
            sf_vector_size=NVFP4_SF_VECTOR_SIZE,
        ](ctx, b, Idx[4096], Idx[4096], Idx[7168])

        # =====================================================================
        # NVFP4: DeepSeek-V2 Decode: 32 groups x 128 x 4096 x 7168
        # =====================================================================
        print("\n=== NVFP4 Decode: 32 groups x 128 x 4096 x 7168 ===")

        bench_cublas_per_group[
            fp4_a_type,
            fp4_b_type,
            fp4_c_type,
            fp4_scales_dtype,
            num_groups=32,
            sf_vector_size=NVFP4_SF_VECTOR_SIZE,
        ](ctx, b, Idx[128], Idx[4096], Idx[7168])

        bench_structured_kernel[
            fp4_a_type,
            fp4_b_type,
            fp4_c_type,
            fp4_scales_dtype,
            num_groups=32,
            mma_m=128,
            mma_n=128,
            k_grp_size=1,
            scaling_kind=UMMAKind.KIND_MXF4NVF4,
            sf_vector_size=NVFP4_SF_VECTOR_SIZE,
        ](ctx, b, Idx[128], Idx[4096], Idx[7168])

    b.dump_report()
    print()
    print("=" * 70)
