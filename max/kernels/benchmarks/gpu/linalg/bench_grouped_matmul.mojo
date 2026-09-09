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

from std.os import abort
from std.math import ceildiv
from std.sys import (
    get_defined_bool,
    get_defined_dtype,
    get_defined_int,
    get_defined_string,
    size_of,
)

import linalg.matmul.vendor.blas as vendor_blas
from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from internal_utils import arg_parse
from internal_utils._utils import (
    InitializationType,
    init_vector_launch,
    _init_block_scaled_scales_launch,
)
from linalg.grouped_matmul import grouped_matmul
from linalg.matmul.gpu.sm100.config import MatmulConfig
from max.gpu.primitives.grid_controls import PDLLevel
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_nvfp4_dispatch,
    grouped_matmul_mxfp8_dispatch,
)
from linalg.grouped_matmul_sm100_blockwise_fp8 import (
    grouped_matmul_sm100_blockwise_scaled_fp8_persistent,
)
from layout import Coord, Idx, TileTensor, row_major
from linalg.utils import elementwise_epilogue_type

from std.utils import Index, IndexList
from std.collections import Optional

from linalg.fp4_utils import (
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
)


def _get_run_name[
    in_type: DType,
    out_type: DType,
    *,
    b_in_type: DType = in_type,
    use_vendor_blas: Bool,
    has_epilogue: Bool = False,
](num_active_experts: Int, total_num_tokens: Int, N: Int, K: Int) -> String:
    var vendor_str = "vendor_gmm" if use_vendor_blas else "gmm"
    # W4A8 activations and weights differ (E4M3 vs packed E2M1); name both
    # so the report line distinguishes it from the same-dtype configs.
    var type_str: String
    if b_in_type == in_type:
        type_str = String("(", in_type, " -> ", out_type, ") : ")
    else:
        type_str = String(
            "(", in_type, "/", b_in_type, " -> ", out_type, ") : "
        )
    # num_active_experts
    var num_active_experts_str = String(num_active_experts)
    # total_num_tokens
    var total_num_tokens_str = String(total_num_tokens)
    # N
    var n_str = String(N)
    # K
    var k_str = String(K)
    # has_epilogue
    var has_epilogue_str = String(" with epilogue" if has_epilogue else "")

    return String(
        vendor_str,
        type_str,
        num_active_experts_str,
        " x ",
        total_num_tokens_str,
        " x ",
        n_str,
        " x ",
        k_str,
        has_epilogue_str,
    )


comptime epilogue_func_type = def[
    dtype: DType, width: SIMDLength, *, alignment: Int = 1
](SIMD[dtype, width]) capturing -> SIMD[dtype, width]


@always_inline
def test_epilogue[
    dtype: DType
](m: Int, n: Int, val: Scalar[dtype]) -> Scalar[dtype]:
    return val + 4 * (Scalar[dtype]((m + n) % 21 - 10))


@always_inline
@__parameter
def add_two[
    dtype: DType,
    width: SIMDLength,
    *,
    alignment: Int = 1,
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return val + 2


def bench_grouped_matmul[
    _in_type: DType,
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],
    /,
    *,
    b_in_type: DType = _in_type,
    use_vendor_blas: Bool = False,
    has_epilogue: Bool = False,
    scaling_kind_str: String = "1d2d",
    override: Bool = False,
    AB_swapped: Bool = True,
    mma_bn: Int = 8,
    cta_group: Int = 1,
    num_pipeline_stages: Int = -1,
    pdl_level: Int = 0,
](
    ctx: DeviceContext,
    mut bench: Bench,
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids_input: List[Int],
    init_type: InitializationType,
) raises:
    comptime N = expert_shape[0]
    comptime K = expert_shape[1]

    comptime is_fp4e2m1 = _in_type == DType.float4_e2m1fn
    comptime in_type = DType.uint8 if is_fp4e2m1 else _in_type  # TODO: (KERN-2238): Replace with float4-e2m1fn
    comptime a_type = in_type
    # W4A8 is the one pair where A and B don't share a dtype: A stays
    # unpacked (float8_e4m3fn) while B is nibble-packed (uint8). Every other
    # config keeps both operands in lockstep with `is_fp4e2m1`, same as A.
    comptime b_type = DType.uint8 if is_fp4e2m1 else b_in_type
    comptime c_type = out_type

    # Total and max number of tokens
    var total_num_tokens = 0
    var max_num_tokens_by_expert = 0
    for num_tokens in num_tokens_by_expert:
        total_num_tokens += num_tokens
        max_num_tokens_by_expert = max(max_num_tokens_by_expert, num_tokens)
    var M = total_num_tokens
    var total_flops = 2 * M * N * K

    # Print parsed inputs for verification (before any GPU work).
    var tok_str = "[" + ", ".join(num_tokens_by_expert) + "]"
    var eid_str = "[" + ", ".join(expert_ids_input) + "]"
    print(
        "Config: num_active_experts=",
        num_active_experts,
        " N=",
        N,
        " K=",
        K,
        " num_experts=",
        num_experts,
        sep="",
    )
    print(
        "  tokens_by_expert(len=",
        len(num_tokens_by_expert),
        " sum=",
        total_num_tokens,
        "): ",
        tok_str,
        sep="",
    )
    print(
        "  expert_ids(len=",
        len(expert_ids_input),
        "): ",
        eid_str,
        sep="",
    )

    def _ri(v: Int) -> Int64:
        return Int64(v)

    # Define shapes and sizes
    # For fp4, data is stored as uint8 (2 fp4 values per byte), so K dimension
    # is halved. W4A8 packs only B, so A and B need independent packed-K
    # extents; every other config has a_packed_K == b_packed_K.
    comptime a_packed_K = K // 2 if a_type == DType.uint8 else K
    comptime b_packed_K = K // 2 if b_type == DType.uint8 else K
    var a_size = total_num_tokens * a_packed_K
    var c_size = total_num_tokens * N
    var b_size = num_experts * N * b_packed_K

    # Host allocations
    var a_offsets_host_ptr = List(length=num_active_experts + 1, fill=UInt32(0))
    var a_scale_offsets_ptr = List(length=num_active_experts, fill=UInt32(0))
    var expert_ids_host_ptr = List(length=num_active_experts, fill=Int32(0))

    # Setup offsets and expert ids
    var a_scale_dim0 = 0
    for i in range(num_active_experts):
        var num_tokens = num_tokens_by_expert[i]
        a_scale_offsets_ptr[i] = UInt32(
            a_scale_dim0
            - Int(a_offsets_host_ptr[i] // UInt32(SF_MN_GROUP_SIZE))
        )
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[i] + UInt32(num_tokens)
        a_scale_dim0 += ceildiv(num_tokens, SF_MN_GROUP_SIZE)
        expert_ids_host_ptr[i] = Int32(expert_ids_input[i])

        # This alignment is specific to the 1d2d blockwise-scaled kernel's
        # float32 scale loads; the block-scaled (mxf8f6f4/nvfp4) path pads
        # ragged experts to SF_MN_GROUP_SIZE via `a_scale_offsets` instead
        # and has no such per-4-token constraint.
        comptime if in_type == .float8_e4m3fn and scaling_kind_str == "1d2d":
            comptime a_scale_alignment = 16 // size_of[DType.float32]()
            if num_tokens % a_scale_alignment != 0:
                abort(
                    "num_tokens=num_tokens_by_expert["
                    + String(i)
                    + "]="
                    + String(num_tokens)
                    + " must be divisible by a_scale_alignment="
                    + String(a_scale_alignment)
                )

    # Device allocations
    var a_dev_buffer = ctx.enqueue_create_buffer[a_type](a_size)
    var b_dev_buffer = ctx.enqueue_create_buffer[b_type](b_size)
    var c_dev_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var a_offsets_dev_buffer = ctx.enqueue_create_buffer[.uint32](
        num_active_experts + 1
    )
    var expert_ids_dev_buffer = ctx.enqueue_create_buffer[.int32](
        num_active_experts
    )

    var a_dev = TileTensor(
        a_dev_buffer,
        row_major(Coord(_ri(total_num_tokens), Idx[a_packed_K])),
    ).as_unsafe_any_origin()
    var b_dev = TileTensor(
        b_dev_buffer,
        row_major(Coord(Idx[num_experts], Idx[N], Idx[b_packed_K])),
    ).as_unsafe_any_origin()
    var c_dev = TileTensor(
        c_dev_buffer,
        row_major(Coord(_ri(total_num_tokens), Idx[N])),
    ).as_unsafe_any_origin()
    var a_offsets_dev = TileTensor(
        a_offsets_dev_buffer,
        row_major(Coord(_ri(num_active_experts + 1))),
    ).as_unsafe_any_origin()
    var expert_ids_dev = TileTensor(
        expert_ids_dev_buffer,
        row_major(Coord(_ri(num_active_experts))),
    ).as_unsafe_any_origin()

    # Initialize data on the device
    init_vector_launch[a_type](a_dev_buffer, a_size, init_type, ctx)
    init_vector_launch[b_type](b_dev_buffer, b_size, init_type, ctx)

    # Move host-initialized data to device
    ctx.enqueue_copy(a_offsets_dev_buffer, a_offsets_host_ptr)
    ctx.enqueue_copy(expert_ids_dev_buffer, expert_ids_host_ptr)

    @always_inline
    @__copy_capture(c_dev)
    @__parameter
    def epilogue_fn[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]) -> None:
        var new_val = val

        comptime for i in range(width):
            new_val[i] = test_epilogue(idx[0], idx[1] + i, val[i])

        c_dev.store_linear[width=width, alignment=alignment](
            idx, new_val.cast[out_type]()
        )

    comptime if is_fp4e2m1:
        comptime assert (
            scaling_kind_str == "nvfp4"
        ), "Only support nvfp4 scaling kind for float4-e2m1fn"

        var a_scale_offsets_dev_buffer = ctx.enqueue_create_buffer[
            DType.uint32
        ](num_active_experts)
        var a_scale_offsets_dev = TileTensor(
            a_scale_offsets_dev_buffer,
            row_major(Coord(_ri(num_active_experts))),
        ).as_unsafe_any_origin()
        ctx.enqueue_copy(a_scale_offsets_dev_buffer, a_scale_offsets_ptr)

        # Calculate scales dimensions
        var a_scales_size = (
            a_scale_dim0
            * ceildiv(K, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
            * SF_ATOM_M[0]
            * SF_ATOM_M[1]
            * SF_ATOM_K
        )
        var b_scales_size = (
            num_experts
            * ceildiv(N, SF_MN_GROUP_SIZE)
            * ceildiv(K, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
            * SF_ATOM_M[0]
            * SF_ATOM_M[1]
            * SF_ATOM_K
        )

        # Scales device allocations
        var a_scales_dev_buffer = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](
            a_scales_size
        )
        var b_scales_dev_buffer = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](
            b_scales_size
        )

        init_vector_launch[NVFP4_SF_DTYPE](
            a_scales_dev_buffer,
            a_scales_size,
            init_type,
            ctx,
        )
        init_vector_launch[NVFP4_SF_DTYPE](
            b_scales_dev_buffer,
            b_scales_size,
            init_type,
            ctx,
        )

        # Build TileTensors for scale factors directly from device buffer
        # pointers + row_major layouts.
        # This avoids a compiler bug with ceildiv(...) in type params.
        comptime k_groups = ceildiv(K, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
        comptime n_groups = ceildiv(N, SF_MN_GROUP_SIZE)
        var a_scales_tt = TileTensor(
            a_scales_dev_buffer,
            row_major(
                Coord(
                    Int64(a_scale_dim0),
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        ).as_unsafe_any_origin()
        var b_scales_tt = TileTensor(
            b_scales_dev_buffer,
            row_major(
                Coord(
                    Idx[num_experts],
                    Idx[n_groups],
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        ).as_unsafe_any_origin()

        var expert_scales_dev_buffer = ctx.enqueue_create_buffer[.float32](
            num_experts
        )
        var expert_scales_host_ptr = List(length=num_experts, fill=Float32(0))
        for i in range(num_experts):
            expert_scales_host_ptr[i] = 1.0 + Float32(i + 1) / Float32(
                num_experts
            )
        ctx.enqueue_copy(expert_scales_dev_buffer, expert_scales_host_ptr)
        var expert_scales_tt = TileTensor(
            expert_scales_dev_buffer,
            row_major(Coord(Int64(num_experts))),
        ).as_unsafe_any_origin()

        @always_inline
        def bench_func_nvfp4(
            mut bench: Bencher,
        ) {
            var a_dev,
            var b_dev,
            var c_dev,
            var a_offsets_dev,
            var expert_ids_dev,
            var a_scale_offsets_dev,
            var a_scales_tt,
            var b_scales_tt,
            var expert_scales_tt,
            imm,
        }:
            @always_inline
            def kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
                comptime if use_vendor_blas:
                    # TODO: Implement vendor grouped matmul
                    pass

                else:
                    comptime transpose_b = True
                    comptime _pdl_level = PDLLevel(pdl_level)
                    grouped_matmul_nvfp4_dispatch[
                        transpose_b=transpose_b,
                        override=override,
                        AB_swapped=AB_swapped,
                        mma_bn=mma_bn,
                        cta_group=cta_group,
                        num_pipeline_stages=num_pipeline_stages,
                        pdl_level=_pdl_level,
                    ](
                        c_dev,
                        a_dev,
                        b_dev,
                        a_scales_tt,
                        b_scales_tt,
                        a_offsets_dev,
                        a_scale_offsets_dev,
                        expert_ids_dev,
                        expert_scales_tt,
                        num_active_experts,
                        total_num_tokens,
                        ctx,
                    )

            bencher_iter_custom(bench, kernel_launch, ctx)

        bench.bench_function(
            bench_func_nvfp4,
            BenchId(
                _get_run_name[
                    _in_type,
                    out_type,
                    use_vendor_blas=use_vendor_blas,
                    has_epilogue=has_epilogue,
                ](
                    num_active_experts,
                    total_num_tokens,
                    N,
                    K,
                )
            ),
            [
                ThroughputMeasure(
                    BenchMetric.flops,
                    total_flops,
                )
            ],
        )

        _ = a_scales_dev_buffer^
        _ = b_scales_dev_buffer^
        _ = a_scale_offsets_dev_buffer^
        _ = expert_scales_dev_buffer^
        _ = expert_scales_host_ptr^

    elif scaling_kind_str == "mxf8f6f4":
        # Grouped block-scaled matmul under kind::mxf8f6f4: E4M3 activations
        # (always unpacked) against either E4M3 (plain MXFP8, `b_in_type` =
        # float8_e4m3fn) or nibble-packed E2M1 weights (W4A8, `b_in_type` =
        # uint8). Both share this one launcher (`grouped_matmul_mxfp8_dispatch`)
        # so a comparison between the two isolates exactly what the packed-FP4
        # TMA unpack costs/saves in the matmul, without also varying the
        # dispatch tactic (mma_bn/cta_group/AB_swapped) between them.
        comptime assert (
            a_type == .float8_e4m3fn
        ), "mxf8f6f4 scaling kind requires float8_e4m3fn activations"

        var a_scale_offsets_dev_buffer = ctx.enqueue_create_buffer[
            DType.uint32
        ](num_active_experts)
        var a_scale_offsets_dev = TileTensor(
            a_scale_offsets_dev_buffer,
            row_major(Coord(_ri(num_active_experts))),
        ).as_unsafe_any_origin()
        ctx.enqueue_copy(a_scale_offsets_dev_buffer, a_scale_offsets_ptr)

        # Calculate scales dimensions
        var a_scales_size = (
            a_scale_dim0
            * ceildiv(K, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K)
            * SF_ATOM_M[0]
            * SF_ATOM_M[1]
            * SF_ATOM_K
        )
        var b_scales_size = (
            num_experts
            * ceildiv(N, SF_MN_GROUP_SIZE)
            * ceildiv(K, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K)
            * SF_ATOM_M[0]
            * SF_ATOM_M[1]
            * SF_ATOM_K
        )

        # Scales device allocations. E8M0 scales need exponent-valued random
        # fill (`_init_block_scaled_scales_launch`), not a raw uniform-[0,1)
        # cast (`init_vector_launch`), which would collapse to the smallest
        # representable exponent almost everywhere.
        var a_scales_dev_buffer = ctx.enqueue_create_buffer[MXFP8_SF_DTYPE](
            a_scales_size
        )
        var b_scales_dev_buffer = ctx.enqueue_create_buffer[MXFP8_SF_DTYPE](
            b_scales_size
        )

        _init_block_scaled_scales_launch[MXFP8_SF_DTYPE](
            a_scales_dev_buffer, a_scales_size, ctx
        )
        _init_block_scaled_scales_launch[MXFP8_SF_DTYPE](
            b_scales_dev_buffer, b_scales_size, ctx
        )

        # Build TileTensors for scale factors directly from device buffer
        # pointers + row_major layouts.
        # This avoids a compiler bug with ceildiv(...) in type params.
        comptime k_groups = ceildiv(K, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K)
        comptime n_groups = ceildiv(N, SF_MN_GROUP_SIZE)
        var a_scales_tt = TileTensor(
            a_scales_dev_buffer,
            row_major(
                Coord(
                    Int64(a_scale_dim0),
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        ).as_unsafe_any_origin()
        var b_scales_tt = TileTensor(
            b_scales_dev_buffer,
            row_major(
                Coord(
                    Idx[num_experts],
                    Idx[n_groups],
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        ).as_unsafe_any_origin()

        var expert_scales_dev_buffer = ctx.enqueue_create_buffer[.float32](
            num_experts
        )
        var expert_scales_host_ptr = List(length=num_experts, fill=Float32(0))
        for i in range(num_experts):
            expert_scales_host_ptr[i] = 1.0 + Float32(i + 1) / Float32(
                num_experts
            )
        ctx.enqueue_copy(expert_scales_dev_buffer, expert_scales_host_ptr)
        var expert_scales_tt = TileTensor(
            expert_scales_dev_buffer,
            row_major(Coord(Int64(num_experts))),
        ).as_unsafe_any_origin()

        @always_inline
        def bench_func_mxf8f6f4(
            mut bench: Bencher,
        ) {
            var a_dev,
            var b_dev,
            var c_dev,
            var a_offsets_dev,
            var expert_ids_dev,
            var a_scale_offsets_dev,
            var a_scales_tt,
            var b_scales_tt,
            var expert_scales_tt,
            imm,
        }:
            @always_inline
            def kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
                comptime if use_vendor_blas:
                    # TODO: Implement vendor grouped matmul
                    pass

                else:
                    comptime transpose_b = True
                    comptime _pdl_level = PDLLevel(pdl_level)
                    # No `override`/mma_bn/cta_group knobs here: the
                    # production dispatcher (grouped_matmul_mxfp8_dispatch)
                    # always regime-classifies on (N, K, avg_m); that is the
                    # tactic we want measured for both A and B configs.
                    grouped_matmul_mxfp8_dispatch[
                        transpose_b=transpose_b,
                        pdl_level=_pdl_level,
                    ](
                        c_dev,
                        a_dev,
                        b_dev,
                        a_scales_tt,
                        b_scales_tt,
                        a_offsets_dev,
                        a_scale_offsets_dev,
                        expert_ids_dev,
                        expert_scales_tt,
                        num_active_experts,
                        total_num_tokens,
                        ctx,
                    )

            bencher_iter_custom(bench, kernel_launch, ctx)

        bench.bench_function(
            bench_func_mxf8f6f4,
            BenchId(
                _get_run_name[
                    _in_type,
                    out_type,
                    b_in_type=b_type,
                    use_vendor_blas=use_vendor_blas,
                    has_epilogue=has_epilogue,
                ](
                    num_active_experts,
                    total_num_tokens,
                    N,
                    K,
                )
            ),
            [
                ThroughputMeasure(
                    BenchMetric.flops,
                    total_flops,
                )
            ],
        )

        _ = a_scales_dev_buffer^
        _ = b_scales_dev_buffer^
        _ = a_scale_offsets_dev_buffer^
        _ = expert_scales_dev_buffer^
        _ = expert_scales_host_ptr^

    elif in_type == .float8_e4m3fn:
        comptime assert (
            scaling_kind_str == "1d2d"
        ), "Only support 1d2d scaling kind for float8_e4m3fn"
        comptime BLOCK_SCALE_K = 128
        var a_scales_size = (K // BLOCK_SCALE_K) * total_num_tokens
        var b_scales_size = (
            num_experts * (N // BLOCK_SCALE_K) * (K // BLOCK_SCALE_K)
        )

        # Scales device allocations
        var a_scales_dev_buffer = ctx.enqueue_create_buffer[.float32](
            a_scales_size
        )
        var b_scales_dev_buffer = ctx.enqueue_create_buffer[.float32](
            b_scales_size
        )

        var a_scales_dev = TileTensor(
            a_scales_dev_buffer,
            row_major(Coord(Idx[K // BLOCK_SCALE_K], _ri(total_num_tokens))),
        ).as_unsafe_any_origin()
        var b_scales_dev = TileTensor(
            b_scales_dev_buffer,
            row_major(
                Coord(
                    Idx[num_experts],
                    Idx[N // BLOCK_SCALE_K],
                    Idx[K // BLOCK_SCALE_K],
                )
            ),
        ).as_unsafe_any_origin()

        init_vector_launch[.float32](
            a_scales_dev_buffer,
            a_scales_size,
            init_type,
            ctx,
        )
        init_vector_launch[.float32](
            b_scales_dev_buffer,
            b_scales_size,
            init_type,
            ctx,
        )

        @always_inline
        def bench_func_fp8_1d2d(
            mut bench: Bencher,
        ) {
            var a_dev,
            var b_dev,
            var c_dev,
            var a_offsets_dev,
            var expert_ids_dev,
            var a_scales_dev,
            var b_scales_dev,
            imm,
        }:
            @always_inline
            def kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
                comptime if use_vendor_blas:
                    # TODO: Implement vendor grouped matmul
                    pass

                else:
                    comptime umma_shape = Index(64, 64, 32)
                    comptime transpose_b = True
                    comptime config = MatmulConfig[
                        a_type, b_type, c_type, transpose_b
                    ](
                        cluster_shape=Index(1, 1, 1),
                        mma_shape=umma_shape,
                        cta_group=1,
                        AB_swapped=AB_swapped,
                        k_group_size=1,
                    )
                    grouped_matmul_sm100_blockwise_scaled_fp8_persistent[
                        config=config,
                        elementwise_lambda_fn=Optional[
                            elementwise_epilogue_type
                        ](epilogue_fn) if has_epilogue else None,
                    ](
                        c_dev,
                        a_dev,
                        b_dev,
                        a_scales_dev,
                        b_scales_dev,
                        a_offsets_dev,
                        expert_ids_dev,
                        max_num_tokens_by_expert,
                        num_active_experts,
                        ctx,
                    )

            bencher_iter_custom(bench, kernel_launch, ctx)

        bench.bench_function(
            bench_func_fp8_1d2d,
            BenchId(
                _get_run_name[
                    in_type,
                    out_type,
                    use_vendor_blas=use_vendor_blas,
                    has_epilogue=has_epilogue,
                ](
                    num_active_experts,
                    total_num_tokens,
                    N,
                    K,
                )
            ),
            # TODO: Pick relevant benchmetric
            [
                ThroughputMeasure(
                    BenchMetric.flops,
                    total_flops,
                )
            ],
        )

        _ = a_scales_dev_buffer^
        _ = b_scales_dev_buffer^
    else:

        @always_inline
        def bench_func(
            mut bench: Bencher,
        ) {
            var a_dev,
            var b_dev,
            var c_dev,
            var a_offsets_dev,
            var expert_ids_dev,
            imm,
        }:
            @always_inline
            def kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
                comptime if use_vendor_blas:
                    # TODO: Implement vendor grouped matmul
                    pass

                else:
                    grouped_matmul[
                        elementwise_lambda_fn=Optional[
                            elementwise_epilogue_type
                        ](epilogue_fn) if has_epilogue else None,
                    ](
                        c_dev,
                        a_dev,
                        b_dev,
                        a_offsets_dev,
                        expert_ids_dev,
                        max_num_tokens_by_expert,
                        num_active_experts,
                        ctx,
                    )

            bencher_iter_custom(bench, kernel_launch, ctx)

        bench.bench_function(
            bench_func,
            BenchId(
                _get_run_name[
                    in_type,
                    out_type,
                    use_vendor_blas=use_vendor_blas,
                    has_epilogue=has_epilogue,
                ](
                    num_active_experts,
                    total_num_tokens,
                    N,
                    K,
                )
            ),
            # TODO: Pick relevant benchmetric
            [
                ThroughputMeasure(
                    BenchMetric.flops,
                    total_flops,
                )
            ],
        )

    # Consume device buffers
    _ = a_dev_buffer^
    _ = b_dev_buffer^
    _ = c_dev_buffer^
    _ = a_offsets_dev_buffer^
    _ = expert_ids_dev_buffer^
    _ = expert_ids_host_ptr^
    _ = a_scale_offsets_ptr^
    _ = a_offsets_host_ptr^


def create_grouped_matmul_bench[
    in_type: DType,
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],
    /,
    *,
    b_in_type: DType = in_type,
    use_vendor_blas: Bool = False,
    has_epilogue: Bool = False,
    scaling_kind_str: String = "1d2d",
    override: Bool = False,
    AB_swapped: Bool = True,
    mma_bn: Int = 8,
    cta_group: Int = 1,
    num_pipeline_stages: Int = -1,
    pdl_level: Int = 0,
](
    ctx: DeviceContext,
    mut bench: Bench,
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    init_type: InitializationType,
) raises:
    bench_grouped_matmul[
        in_type,
        out_type,
        num_experts,
        expert_shape,
        b_in_type=b_in_type,
        use_vendor_blas=use_vendor_blas,
        has_epilogue=has_epilogue,
        scaling_kind_str=scaling_kind_str,
        override=override,
        AB_swapped=AB_swapped,
        mma_bn=mma_bn,
        cta_group=cta_group,
        num_pipeline_stages=num_pipeline_stages,
        pdl_level=pdl_level,
    ](
        ctx,
        bench,
        num_active_experts,
        num_tokens_by_expert,
        expert_ids,
        init_type,
    )


def string_to_list(string: String) raises -> List[Int]:
    var s = string.strip("[]")
    var list = List[Int]()
    for i in s.split(","):
        try:
            list.append(Int(i))
        except:
            continue
    return list^


def main() raises:
    comptime in_type = get_defined_dtype["in_type", .bfloat16]()
    comptime out_type = get_defined_dtype["out_type", .bfloat16]()
    # Defaults to `in_type` so every existing (same-dtype) config is
    # unaffected; W4A8 passes `-D b_in_type=uint8` explicitly.
    comptime b_in_type = get_defined_dtype["b_in_type", in_type]()
    comptime scaling_kind_str = get_defined_string["scaling_kind", "1d2d"]()

    var num_active_experts = Int(arg_parse("num_active_experts", 1))
    var num_tokens_by_expert_string = String(
        arg_parse("num_tokens_by_expert", "256")
    )
    var expert_ids_string = String(arg_parse("expert_ids", "0"))

    var num_tokens_by_expert = string_to_list(num_tokens_by_expert_string)
    var expert_ids = string_to_list(expert_ids_string)

    comptime N = get_defined_int["N", 256]()
    comptime K = get_defined_int["K", 256]()
    comptime num_experts = get_defined_int["num_experts", 1]()

    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    comptime use_vendor_blas = get_defined_bool["use_vendor_blas", False]()
    comptime has_epilogue = get_defined_bool["has_epilogue", False]()
    comptime override = get_defined_bool["override", False]()
    comptime AB_swapped = get_defined_bool["AB_swapped", True]()
    comptime mma_bn = get_defined_int["mma_bn", 8]()
    comptime cta_group = get_defined_int["cta_group", 1]()
    comptime num_pipeline_stages = get_defined_int["num_pipeline_stages", -1]()
    comptime pdl_level = get_defined_int["pdl_level", 0]()

    var b = Bench()
    comptime expert_shape = IndexList[2](N, K)

    with DeviceContext() as ctx:
        create_grouped_matmul_bench[
            in_type,
            out_type,
            num_experts,
            expert_shape,
            b_in_type=b_in_type,
            use_vendor_blas=use_vendor_blas,
            has_epilogue=has_epilogue,
            scaling_kind_str=scaling_kind_str,
            override=override,
            AB_swapped=AB_swapped,
            mma_bn=mma_bn,
            cta_group=cta_group,
            num_pipeline_stages=num_pipeline_stages,
            pdl_level=pdl_level,
        ](
            ctx,
            b,
            num_active_experts,
            num_tokens_by_expert,
            expert_ids,
            init_type,
        )

    b.dump_report()
