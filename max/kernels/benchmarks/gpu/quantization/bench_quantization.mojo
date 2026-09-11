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

from std.sys import (
    get_defined_dtype,
    get_defined_int,
    size_of,
    get_defined_bool,
)
from std.math import ceildiv, divmod
from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    ThroughputMeasure,
    BenchMetric,
)
from max.gpu.host import DeviceContext
from layout import (
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    lt_to_tt,
    row_major,
)
from layout._fillers import random
from max.gpu.host.info import _is_sm10x_gpu

from std.utils.index import IndexList
from linalg.fp4_utils import (
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    MXFP8_SF_DTYPE,
)
from linalg.block_scaled_quantization import (
    quantize_dynamic_scaled_fp4fp8,
    quantize_dynamic_scaled_fp4_async,
    grouped_quantize_dynamic_scaled_fp4_async,
)
from internal_utils import arg_parse


def bench_1d1d_quantization[
    in_dtype: DType, cols: Int, use_async: Bool, is_fp4: Bool
](ctx: DeviceContext, mut b: Bench, fn_name: String, rows: Int) raises:
    comptime out_dtype = DType.uint8 if is_fp4 else DType.float8_e4m3fn
    comptime scales_dtype = NVFP4_SF_DTYPE if is_fp4 else MXFP8_SF_DTYPE
    comptime SF_VECTOR_SIZE = NVFP4_SF_VECTOR_SIZE if is_fp4 else MXFP8_SF_VECTOR_SIZE

    comptime input_static_shape = Layout.row_major(UNKNOWN_VALUE, cols)
    var input_dynamic_shape = IndexList[2](rows, cols)
    var input_runtime_layout = RuntimeLayout[input_static_shape].row_major(
        input_dynamic_shape
    )
    var in_device = ctx.enqueue_create_buffer[in_dtype](
        input_dynamic_shape.flattened_length()
    )
    var input_tensor = LayoutTensor[in_dtype, input_static_shape](
        in_device, input_runtime_layout
    )

    # Output tensor layout and buffer
    comptime output_static_shape = Layout.row_major(
        UNKNOWN_VALUE,
        ceildiv(cols, 2),
    )
    var output_dynamic_shape = IndexList[2](rows, ceildiv(cols, 2))
    var output_runtime_layout = RuntimeLayout[output_static_shape].row_major(
        output_dynamic_shape
    )
    var out_device = ctx.enqueue_create_buffer[out_dtype](
        output_dynamic_shape.flattened_length()
    )
    var output_tensor = LayoutTensor[out_dtype, output_static_shape](
        out_device, output_runtime_layout
    )

    # Scales tensor layout and buffer
    var scales_shape = IndexList[5](
        ceildiv(rows, SF_MN_GROUP_SIZE),
        ceildiv(cols, SF_VECTOR_SIZE * SF_ATOM_K),
        SF_ATOM_M[0],
        SF_ATOM_M[1],
        SF_ATOM_K,
    )
    comptime scales_static_layout = Layout.row_major(
        UNKNOWN_VALUE,
        ceildiv(cols, SF_VECTOR_SIZE * SF_ATOM_K),
        SF_ATOM_M[0],
        SF_ATOM_M[1],
        SF_ATOM_K,
    )
    var scales_runtime_layout = RuntimeLayout[scales_static_layout].row_major(
        scales_shape
    )
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](
        scales_shape.flattened_length()
    )
    var scales_tensor = LayoutTensor[scales_dtype, scales_static_layout](
        scales_device, scales_runtime_layout
    )

    # Initialize input with random data and output with zeros on host
    with in_device.map_to_host() as in_host:
        var in_host_tensor = LayoutTensor[in_dtype, input_static_shape](
            in_host, input_runtime_layout
        )
        random(in_host_tensor)

    @always_inline
    def bench_fn(
        mut b: Bencher,
    ) raises {var input_tensor, var output_tensor, var scales_tensor, imm,}:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            # Run the quantization kernel
            comptime if use_async:
                quantize_dynamic_scaled_fp4_async[
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE
                ](
                    ctx,
                    lt_to_tt(output_tensor).as_unsafe_any_origin(),
                    lt_to_tt(scales_tensor).as_unsafe_any_origin(),
                    lt_to_tt(input_tensor).as_unsafe_any_origin(),
                )
            else:
                quantize_dynamic_scaled_fp4fp8[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    ctx,
                    lt_to_tt(output_tensor).as_unsafe_any_origin(),
                    lt_to_tt(scales_tensor).as_unsafe_any_origin(),
                    lt_to_tt(input_tensor).as_unsafe_any_origin(),
                    num_cols=cols,
                    num_cols_padded=cols,
                )

        bencher_iter_custom(b, kernel_launch, ctx)

    var bytes = ThroughputMeasure(
        BenchMetric.bytes,
        (rows * cols) * size_of[in_dtype]()
        + (rows * cols // (2 if is_fp4 else 1)) * size_of[out_dtype]()
        + (
            ceildiv(rows, SF_MN_GROUP_SIZE)
            * ceildiv(cols, SF_VECTOR_SIZE * SF_ATOM_K)
        )
        * (SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K)
        * size_of[scales_dtype](),
    )

    b.bench_function(
        bench_fn,
        BenchId(
            "1d1d_quantization",
            input_id=String(
                fn_name,
                in_dtype,
                out_dtype,
                scales_dtype,
                SF_VECTOR_SIZE,
                rows,
                cols,
                sep="/",
            ),
        ),
        [bytes],
        # fixed_iterations=1,
    )

    ctx.synchronize()


def bench_grouped_quantization[
    in_dtype: DType, cols: Int, num_experts: Int, spread: Bool = True
](ctx: DeviceContext, mut b: Bench, rows: Int) raises:
    """Times the grouped NVFP4 quantize across expert-occupancy shapes.

    Cost here tracks occupied scale tiles rather than rows, because a block
    covers a whole 128-row tile whatever the tile holds. Sweeping
    `num_experts` at a fixed `rows` moves the tile count while the payload
    stays put, and sweeping `rows` at a fixed expert count does the reverse,
    so the two separate. Expect the expert sweep to move the number and the
    row sweep to barely register.
    """
    comptime out_dtype = DType.uint8
    comptime scales_dtype = NVFP4_SF_DTYPE
    comptime SF_VECTOR_SIZE = NVFP4_SF_VECTOR_SIZE
    comptime K_tiles = ceildiv(cols, SF_VECTOR_SIZE * SF_ATOM_K)

    # `spread` shares the rows evenly across experts, which is what decode
    # looks like: top_k x batch routed pairs land on nearly that many distinct
    # experts, each holding a row or two and so each needing its own scale
    # tile. That makes the occupied tile count track the expert count rather
    # than the row count. The `spread=False` layout parks every row on one
    # expert and is the easy case, kept only as a contrast.
    var host_row_offsets = alloc[UInt32](num_experts + 1)
    var host_scales_offsets = alloc[UInt32](num_experts)
    var host_expert_ids = alloc[Int32](num_experts)
    var host_sf = alloc[Float32](num_experts)
    var tile_starts = alloc[Int](num_experts + 1)

    host_row_offsets[0] = 0
    tile_starts[0] = 0
    var base_count, remainder = divmod(rows, num_experts)
    for i in range(num_experts):
        var count: Int
        comptime if spread:
            count = base_count + (1 if i < remainder else 0)
        else:
            count = rows if i == 0 else 0
        host_row_offsets[i + 1] = host_row_offsets[i] + UInt32(count)
        tile_starts[i + 1] = tile_starts[i] + ceildiv(count, SF_MN_GROUP_SIZE)
        host_scales_offsets[i] = UInt32(
            tile_starts[i] - Int(host_row_offsets[i]) // SF_MN_GROUP_SIZE
        )
        host_expert_ids[i] = Int32(i)
        host_sf[i] = 1.0

    # Occupied tiles, and the worst case the host must allocate for.
    var occupied_tiles = tile_starts[num_experts]
    var total_m_tiles = ceildiv(rows, SF_MN_GROUP_SIZE) + min(num_experts, rows)

    var dev_in = ctx.enqueue_create_buffer[in_dtype](rows * cols)
    var dev_out = ctx.enqueue_create_buffer[out_dtype](rows * ceildiv(cols, 2))
    var dev_scales = ctx.enqueue_create_buffer[scales_dtype](
        total_m_tiles * K_tiles * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    )
    var dev_row_offsets = ctx.enqueue_create_buffer[.uint32](num_experts + 1)
    var dev_scales_offsets = ctx.enqueue_create_buffer[.uint32](num_experts)
    var dev_expert_ids = ctx.enqueue_create_buffer[.int32](num_experts)
    var dev_sf = ctx.enqueue_create_buffer[.float32](num_experts)

    # The group max feeds a data-dependent branch, so the input has to be
    # defined: an all-zero buffer skips the reciprocal path that real
    # activations always take.
    var host_input = alloc[Scalar[in_dtype]](rows * cols)
    var host_input_tensor = TileTensor(
        host_input, row_major(Coord(rows, Idx[cols]))
    )
    random(host_input_tensor, min=-1.0, max=1.0)
    ctx.enqueue_copy(dev_in, host_input)

    ctx.enqueue_copy(dev_row_offsets, host_row_offsets)
    ctx.enqueue_copy(dev_scales_offsets, host_scales_offsets)
    ctx.enqueue_copy(dev_expert_ids, host_expert_ids)
    ctx.enqueue_copy(dev_sf, host_sf)

    var in_tensor = TileTensor(dev_in, row_major(Coord(rows, Idx[cols])))
    var out_tensor = TileTensor(
        dev_out, row_major(Coord(rows, Idx[ceildiv(cols, 2)]))
    )
    var scales_tensor = TileTensor(
        dev_scales,
        row_major(
            Coord(
                total_m_tiles,
                Idx[K_tiles],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            )
        ),
    )
    var row_offsets_t = TileTensor(
        dev_row_offsets, row_major(Coord(Idx[num_experts + 1]))
    )
    var scales_offsets_t = TileTensor(
        dev_scales_offsets, row_major(Coord(Idx[num_experts]))
    )
    var expert_ids_t = TileTensor(
        dev_expert_ids, row_major(Coord(Idx[num_experts]))
    )
    var sf_t = TileTensor(dev_sf, row_major(Coord(Idx[num_experts])))

    @always_inline
    def bench_fn(
        mut b: Bencher,
    ) raises {
        var out_tensor,
        var scales_tensor,
        var in_tensor,
        var row_offsets_t,
        var scales_offsets_t,
        var expert_ids_t,
        var sf_t,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            grouped_quantize_dynamic_scaled_fp4_async(
                out_tensor,
                scales_tensor,
                in_tensor,
                row_offsets_t,
                scales_offsets_t,
                expert_ids_t,
                sf_t,
                ctx,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    b.bench_function(
        bench_fn,
        BenchId(
            "grouped_quantization",
            input_id=String(
                "rows",
                rows,
                "cols",
                cols,
                "experts",
                num_experts,
                "spread" if spread else "single",
                "occupied",
                occupied_tiles,
                "alloc",
                total_m_tiles,
                sep="/",
            ),
        ),
    )
    ctx.synchronize()


def main() raises:
    comptime in_dtype = get_defined_dtype["dtype", .bfloat16]()

    var rows = Int(arg_parse("M", 1))
    comptime cols = get_defined_int["N", 4096]()
    comptime use_async = get_defined_bool["use_async", True]()
    comptime is_fp4 = get_defined_bool["is_fp4", True]()
    # The grouped cases carry their own row counts, so they ignore `M` and
    # would repeat identically across an `M` sweep. Opt in with -D grouped=1.
    comptime grouped = get_defined_bool["grouped", False]()

    with DeviceContext() as ctx:
        comptime if _is_sm10x_gpu(ctx.default_device_info):
            var m = Bench(BenchConfig(num_repetitions=1))
            comptime if grouped:
                # Decode: batch 32 x top_k 6 routed pairs over 256 experts.
                bench_grouped_quantization[in_dtype, cols, 256](ctx, m, 192)
                bench_grouped_quantization[in_dtype, cols, 128](ctx, m, 192)
                bench_grouped_quantization[in_dtype, cols, 32](ctx, m, 192)
                bench_grouped_quantization[in_dtype, cols, 8](ctx, m, 192)
                # Smaller decode batch: 8 x top_k 6 routed pairs. This sits
                # just above the batching crossover, so it checks the
                # threshold picks the faster path there rather than only for
                # large batches.
                bench_grouped_quantization[in_dtype, cols, 48](ctx, m, 48)
                # Prefill-scale row counts, at both expert counts the shipping
                # models use.
                bench_grouped_quantization[in_dtype, cols, 256](ctx, m, 768)
                bench_grouped_quantization[in_dtype, cols, 256](ctx, m, 3072)
                bench_grouped_quantization[in_dtype, cols, 128](ctx, m, 768)
                bench_grouped_quantization[in_dtype, cols, 128](ctx, m, 3072)
                # Single-expert contrast: every row on one expert, so the
                # occupied tile count collapses to the row tiles.
                bench_grouped_quantization[in_dtype, cols, 256, spread=False](
                    ctx, m, 192
                )
            else:
                bench_1d1d_quantization[in_dtype, cols, use_async, is_fp4](
                    ctx, m, "1d1d_quantization", rows
                )
            m.dump_report()
        else:
            print("this benchmark is only supported on NVIDIA SM100")
