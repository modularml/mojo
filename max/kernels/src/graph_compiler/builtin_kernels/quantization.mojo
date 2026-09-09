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


# ===-----------------------------------------------------------------------===#
# General imports
# ===-----------------------------------------------------------------------===#

"""Registers quantization and dequantization graph ops backed by the `quantization` and `linalg` kernels."""

from std.sys.info import size_of
import extensibility

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#

from max.gpu.host import DeviceContext
from layout.tile_tensor import row_major
from max.gpu.host.info import is_cpu, is_gpu
from internal_utils.fp8_utils import fp8_quantize
from builtin_primitives.primitives import foreach
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    UNKNOWN_VALUE,
    coord_to_index_list,
    row_major,
)
from nn.normalization import (
    rms_norm_fused_quantize_dynamic_scaled_fp8,
)
from std.utils.coord import ComptimeInt
from std.utils.index import IndexList
from linalg.fp8_quantization import (
    quantize_dynamic_scaled_fp8,
    quantize_tensor_dynamic_scaled_fp8,
)
from linalg.block_scaled_quantization import (
    quantize_dynamic_block_scaled,
    grouped_quantize_dynamic_scaled_fp4_async,
    block_scales_interleave,
    quantize_dynamic_block_scaled_mxfp4,
)
from linalg.matmul.gpu.amd import (
    Shuffler,
    block_scaled_grouped_matmul_amd,
)
from linalg.mxfp4_dequant import dequant_mxfp4
from linalg.fp6_utils import FP6Format
from linalg.fp6_quantization import quantize_mxfp6_amd
from linalg.mxfp6_dequant import dequant_mxfp6
from nn.bicubic import resize_bicubic
from nn.kv_cache import generic_get_paged_cache
from nn.kv_cache_ragged import unfused_qkv_matmul_ragged_paged_gguf_quantized
from nn.resize import (
    CoordinateTransformationMode,
    RoundMode,
    resize_linear,
    resize_nearest_neighbor,
)
from quantization import (
    Q4sym,
    block_Q4_K,
    block_Q6_K,
    block_QK_K,
    q4_k_dequantize_impl,
    q6_k_dequantize_impl,
)
from quantization.qmatmul import matmul_qint4, matmul_qint4_pack_b
from quantization.qmatmul_gpu import (
    gpu_qint4_repack_GPTQ,
    gpu_qint4_repack_Q4_0,
    matmul_gpu_qint4,
)
from quantization.qmatmul_k import (
    matmul_Q4_K,
    matmul_Q4_K_pack_b,
    matmul_Q6_K,
    matmul_Q6_K_pack_b,
)
from extensibility import InputTensor, OutputTensor, Tensor
from extensibility import TileTensorable
from extensibility import (
    _FusedInputTensor as FusedInputTensor,
)
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)
from std.logger import Logger

comptime logger = Logger()

from std.utils import IndexList
from std.utils.index import Index

# ===-----------------------------------------------------------------------===#
from .kernels import *


@extensibility.register(
    "mo.composite.rms_norm_fused_quantize_dynamic_scaled_fp8"
)
struct RMSNormFusedQuantizeDynamicScaledFP8:
    """Registers the `mo.composite.rms_norm_fused_quantize_dynamic_scaled_fp8` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        input_dtype: DType,
        output_dtype: DType,
        scale_dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_dtype, rank=rank, ...],
        scales: OutputTensor[dtype=scale_dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=input_dtype, rank=rank, ...],
        gamma: InputTensor[dtype=input_dtype, rank=1, ...],
        epsilon: Float32,
        weight_offset: Scalar[dtype=input_dtype],
        scale_ub: Float32,
        ctx: DeviceContext,
    ) capturing raises:
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")

        var out_t = output.to_tile_tensor[.int64]()

        # The scale output holds one value per input row, laid out
        # [1, rows]. View it as rank-1 and index it by row number.
        var in_shape = input.shape()
        var rows = in_shape.flattened_length() // in_shape[rank - 1]
        var scale_t = TileTensor(
            scales.to_tile_tensor[.int64]()._storage,
            row_major(Coord(rows)),
        )

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var input} -> SIMD[input_dtype, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                coords
            )

        @always_inline
        def output_fn[
            width: SIMDLength, alignment: Int
        ](coords: Coord, val: SIMD[output_dtype, width]) {var out_t}:
            out_t.store[width=width, alignment=alignment](coords, val)

        @always_inline
        def scale_fn(
            coords: Coord, val: Scalar[scale_dtype]
        ) {var scale_t, var in_shape}:
            var row = 0
            comptime for i in range(rank - 1):
                row = row * in_shape[i] + Int(coords[i].value())
            scale_t.store_linear[width=1, alignment=1](IndexList[1](row), val)

        # Static row width (when known) enables the register-cached row path.
        comptime cols = Int(input.static_spec.shape_tuple[rank - 1])

        comptime if cols != UNKNOWN_VALUE:
            rms_norm_fused_quantize_dynamic_scaled_fp8[
                input_dtype, output_dtype, scale_dtype, rank, target=target
            ](
                input_fn,
                output_fn,
                scale_fn,
                input.shape_coord(),
                ComptimeInt[cols](),
                gamma.to_tile_tensor[.int64](),
                epsilon.cast[input_dtype](),
                weight_offset,
                scale_ub,
                ctx,
            )
        else:
            rms_norm_fused_quantize_dynamic_scaled_fp8[
                input_dtype, output_dtype, scale_dtype, rank, target=target
            ](
                input_fn,
                output_fn,
                scale_fn,
                input.shape_coord(),
                Int(Int(input.shape()[rank - 1])),
                gamma.to_tile_tensor[.int64](),
                epsilon.cast[input_dtype](),
                weight_offset,
                scale_ub,
                ctx,
            )


@extensibility.register_shape_function(
    "mo.composite.rms_norm_fused_quantize_dynamic_scaled_fp8"
)
def composite_rms_norm_fused_quantize_dynamic_scaled_fp8_shape[
    input_dtype: DType,
](
    input: Some[Tensor],
    gamma: Some[Tensor],
    epsilon: Float32,
    weight_offset: Scalar[dtype=input_dtype],
    scale_ub: Float32,
) -> IndexList[type_of(input).rank]:
    """Computes the output shapes for the fused RMS norm and dynamic scaled FP8 quantization op.
    """
    comptime assert (
        type_of(gamma).dtype == type_of(input).dtype
    ), "gamma dtype must match input dtype"
    comptime assert type_of(gamma).rank == 1, "gamma must be rank 1"
    comptime assert (
        input_dtype == type_of(input).dtype
    ), "weight_offset dtype must match input dtype"
    return rebind[IndexList[type_of(input).rank]](
        coord_to_index_list(input.shape().tuple())
    )


@extensibility.register("mo.resize.nearest")
struct ResizeNearest:
    """Registers the `mo.resize.nearest` graph op with the graph compiler."""

    @staticmethod
    def execute[
        coordinate_transform_mode: Int,
        round_mode: Int,
        rank: Int,
        dtype: DType,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        size: InputTensor[rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        resize_nearest_neighbor[
            CoordinateTransformationMode(coordinate_transform_mode),
            RoundMode(round_mode),
        ](
            input.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
            ctx,
        )


@extensibility.register_shape_function("mo.resize.nearest")
def resize_nearest_shape(
    input: Some[Tensor], size: Some[TileTensorable]
) -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.resize.nearest` graph op."""
    comptime assert type_of(size).rank == 1, "size must be rank 1"
    var size_tt = size.to_tile_tensor()
    var shape = IndexList[type_of(input).rank]()
    for i in range(type_of(input).rank):
        shape[i] = Int(size_tt[i])

    return shape


@extensibility.register("mo.resize.linear")
struct ResizeLinear:
    """Registers the `mo.resize.linear` graph op with the graph compiler."""

    @staticmethod
    def execute[
        coordinate_transform_mode: Int,
        antialias: Bool,
        rank: Int,
        dtype: DType,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        size: InputTensor[rank=1, ...],
    ):
        resize_linear[
            CoordinateTransformationMode(coordinate_transform_mode), antialias
        ](
            input.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
        )


@extensibility.register_shape_function("mo.resize.linear")
def resize_linear_shape(
    input: Some[Tensor], size: Some[TileTensorable]
) -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.resize.linear` graph op."""
    comptime assert type_of(size).rank == 1, "size must be rank 1"
    var size_tt = size.to_tile_tensor()
    var shape = IndexList[type_of(input).rank]()
    for i in range(type_of(input).rank):
        shape[i] = Int(size_tt[i])

    return shape


@extensibility.register("mo.resize.bicubic")
struct ResizeBicubic:
    """Registers the `mo.resize.bicubic` graph op with the graph compiler."""

    @staticmethod
    def execute[
        rank: Int,
        dtype: DType,
        target: StaticString,
        //,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        size: InputTensor[rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        resize_bicubic[dtype=dtype, target=target](
            output.to_tile_tensor[.int64](),
            input.to_tile_tensor[.int64](),
            ctx,
        )


@extensibility.register_shape_function("mo.resize.bicubic")
def resize_bicubic_shape(
    input: Some[Tensor], size: Some[TileTensorable]
) -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.resize.bicubic` graph op."""
    comptime assert type_of(size).rank == 1, "size must be rank 1"
    var size_tt = size.to_tile_tensor()
    var shape = IndexList[type_of(input).rank]()
    for i in range(type_of(input).rank):
        shape[i] = Int(size_tt[i])

    return shape


@extensibility.register("ggml_q4_0_dequantize")
struct GGMLQ40Dequantize:
    """Registers the `ggml_q4_0_dequantize` graph op with the graph compiler."""

    @staticmethod
    @always_inline
    def execute[
        _trace_name: StaticString,
    ](
        output: OutputTensor[dtype=.float32, rank=2, ...],
        input: InputTensor[dtype=.uint8, rank=2, ...],
    ) raises:
        var input_tt = input.to_tile_tensor[.int64]()
        var output_tt = output.to_tile_tensor[.int64]()
        Q4sym[group_size=32].dequantize_and_write_to_tensor(
            input_tt,
            output_tt,
            output.shape(),
        )


@extensibility.register_shape_function("ggml_q4_0_dequantize")
def ggml_q4_0_dequantize_shape(input: Some[Tensor]) -> IndexList[2]:
    """Computes the output shape for the `ggml_q4_0_dequantize` graph op."""
    comptime assert type_of(input).dtype == .uint8, "input must be uint8"
    comptime assert type_of(input).rank == 2, "input must be rank 2"
    comptime block_nbytes = size_of[Q4sym[group_size=32]]()
    comptime quants_per_block = 32
    var shape = input.shape()
    var dim0 = Int(coord_to_index_list(shape.tuple())[0])
    var num_block_per_batch = (Int(shape.product()) // dim0) // block_nbytes
    return (dim0, quants_per_block * num_block_per_batch)


@extensibility.register("vroom_q4_0_matmul")
struct VroomQ40Matmul:
    """Registers the `vroom_q4_0_matmul` graph op with the graph compiler."""

    @staticmethod
    @always_inline
    def execute[
        _trace_name: StaticString,
        target: StaticString,
    ](
        c: OutputTensor[dtype=.float32, rank=2, ...],
        a: InputTensor[dtype=.float32, rank=2, ...],
        b: InputTensor[dtype=.uint8, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_cpu[target](), "only valid on CPUs"
        matmul_qint4[32](
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            c.to_tile_tensor[.int64](),
            Optional[DeviceContext](ctx),
        )


@extensibility.register_shape_function("vroom_q4_0_matmul")
def vroom_q4_0_matmul_shape(a: Some[Tensor], b: Some[Tensor]) -> IndexList[2]:
    """Computes the output shape for the `vroom_q4_0_matmul` graph op."""
    comptime assert type_of(a).dtype == .float32, "a must be float32"
    comptime assert type_of(a).rank == 2, "a must be rank 2"
    comptime assert type_of(b).dtype == .uint8, "b must be uint8"
    comptime assert type_of(b).rank == 2, "b must be rank 2"
    return IndexList[2](
        Int(coord_to_index_list(a.shape().tuple())[0]),
        Int(coord_to_index_list(b.shape().tuple())[0]),
    )


@extensibility.register("vroom_q4_0_repack_weights")
struct VroomQ40RepackWeights:
    """Registers the `vroom_q4_0_repack_weights` graph op with the graph compiler.
    """

    @staticmethod
    @always_inline
    def execute[
        _trace_name: StaticString,
    ](
        b_packed: OutputTensor[dtype=.uint8, rank=2, ...],
        b: InputTensor[dtype=.uint8, rank=2, ...],
    ) raises:
        matmul_qint4_pack_b[32](
            b.to_tile_tensor[.int64](),
            b_packed.to_tile_tensor[.int64](),
        )


@extensibility.register_shape_function("vroom_q4_0_repack_weights")
def vroom_q4_0_repack_weights_shape(
    b: Some[Tensor],
) -> IndexList[type_of(b).rank]:
    """Computes the output shape for the `vroom_q4_0_repack_weights` graph op.
    """
    comptime assert type_of(b).dtype == .uint8, "b must be uint8"
    comptime assert type_of(b).rank == 2, "b must be rank 2"
    return rebind[IndexList[type_of(b).rank]](
        coord_to_index_list(b.shape().tuple())
    )


@extensibility.register("ggml_q4_k_dequantize")
struct GGMLQ4KDequantize:
    """Registers the `ggml_q4_k_dequantize` graph op with the graph compiler."""

    @staticmethod
    @always_inline
    def execute[
        _trace_name: StaticString,
    ](
        output: OutputTensor[dtype=.float32, rank=2, ...],
        input: InputTensor[dtype=.uint8, rank=2, ...],
    ) raises:
        q4_k_dequantize_impl(
            input.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
        )


@extensibility.register_shape_function("ggml_q4_k_dequantize")
def ggml_q4_k_dequantize_shape(input: Some[Tensor]) -> IndexList[2]:
    """Computes the output shape for the `ggml_q4_k_dequantize` graph op."""
    comptime assert type_of(input).dtype == .uint8, "input must be uint8"
    comptime assert type_of(input).rank == 2, "input must be rank 2"
    comptime block_nbytes = size_of[block_Q4_K]()
    comptime elements_per_block = block_QK_K.quantized_k

    var shape = input.shape()
    var dim0 = Int(coord_to_index_list(shape.tuple())[0])
    var num_block_per_batch = (Int(shape.product()) // dim0) // block_nbytes

    return (
        dim0,
        elements_per_block * num_block_per_batch,
    )


@extensibility.register("vroom_q4_k_matmul")
struct VroomQ4KMatmul:
    """Registers the `vroom_q4_k_matmul` graph op with the graph compiler."""

    @staticmethod
    @always_inline
    def execute[
        _trace_name: StaticString,
        target: StaticString,
    ](
        c: OutputTensor[dtype=.float32, rank=2, ...],
        a: InputTensor[dtype=.float32, rank=2, ...],
        b: InputTensor[dtype=.uint8, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_cpu[target](), "only valid on CPUs"
        matmul_Q4_K(
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            c.to_tile_tensor[.int64](),
            Optional[DeviceContext](ctx),
        )


@extensibility.register_shape_function("vroom_q4_k_matmul")
def vroom_q4_k_matmul_shape(a: Some[Tensor], b: Some[Tensor]) -> IndexList[2]:
    """Computes the output shape for the `vroom_q4_k_matmul` graph op."""
    comptime assert type_of(a).dtype == .float32, "a must be float32"
    comptime assert type_of(a).rank == 2, "a must be rank 2"
    comptime assert type_of(b).dtype == .uint8, "b must be uint8"
    comptime assert type_of(b).rank == 2, "b must be rank 2"
    return IndexList[2](
        Int(coord_to_index_list(a.shape().tuple())[0]),
        Int(coord_to_index_list(b.shape().tuple())[0]),
    )


@extensibility.register("vroom_q4_k_repack_weights")
struct VroomQ4KRepackWeights:
    """Registers the `vroom_q4_k_repack_weights` graph op with the graph compiler.
    """

    @staticmethod
    @always_inline
    def execute[
        _trace_name: StaticString,
    ](
        b_packed: OutputTensor[dtype=.uint8, rank=2, ...],
        b: InputTensor[dtype=.uint8, rank=2, ...],
    ) raises:
        matmul_Q4_K_pack_b(
            b.to_tile_tensor[.int64](),
            b_packed.to_tile_tensor[.int64](),
        )


@extensibility.register_shape_function("vroom_q4_k_repack_weights")
def vroom_q4_k_repack_weights_shape(
    b: Some[Tensor],
) -> IndexList[type_of(b).rank]:
    """Computes the output shape for the `vroom_q4_k_repack_weights` graph op.
    """
    comptime assert type_of(b).dtype == .uint8, "b must be uint8"
    comptime assert type_of(b).rank == 2, "b must be rank 2"
    return rebind[IndexList[type_of(b).rank]](
        coord_to_index_list(b.shape().tuple())
    )


@extensibility.register("ggml_q6_k_dequantize")
struct GGMLQ6KDequantize:
    """Registers the `ggml_q6_k_dequantize` graph op with the graph compiler."""

    @staticmethod
    @always_inline
    def execute[
        _trace_name: StaticString,
    ](
        output: OutputTensor[dtype=.float32, rank=2, ...],
        input: InputTensor[dtype=.uint8, rank=2, ...],
    ) raises:
        var input_tt = input.to_tile_tensor[.int64]()
        var output_tt = output.to_tile_tensor[.int64]()
        q6_k_dequantize_impl(
            input_tt,
            output_tt,
            output.shape(),
        )


@extensibility.register_shape_function("ggml_q6_k_dequantize")
def ggml_q6_k_dequantize_shape(input: Some[Tensor]) -> IndexList[2]:
    """Computes the output shape for the `ggml_q6_k_dequantize` graph op."""
    comptime assert type_of(input).dtype == .uint8, "input must be uint8"
    comptime assert type_of(input).rank == 2, "input must be rank 2"
    comptime block_nbytes = size_of[block_Q6_K]()
    comptime elements_per_block = block_QK_K.quantized_k

    var shape = input.shape()
    var dim0 = Int(coord_to_index_list(shape.tuple())[0])
    var num_block_per_batch = (Int(shape.product()) // dim0) // block_nbytes

    return (
        dim0,
        elements_per_block * num_block_per_batch,
    )


@extensibility.register("vroom_q6_k_matmul")
struct VroomQ6KMatmul:
    """Registers the `vroom_q6_k_matmul` graph op with the graph compiler."""

    @staticmethod
    @always_inline
    def execute[
        _trace_name: StaticString,
        target: StaticString,
    ](
        c: OutputTensor[dtype=.float32, rank=2, ...],
        a: InputTensor[dtype=.float32, rank=2, ...],
        b: InputTensor[dtype=.uint8, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_cpu[target](), "only valid on CPUs"
        matmul_Q6_K(
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            c.to_tile_tensor[.int64](),
            Optional[DeviceContext](ctx),
        )


@extensibility.register_shape_function("vroom_q6_k_matmul")
def vroom_q6_k_matmul_shape(a: Some[Tensor], b: Some[Tensor]) -> IndexList[2]:
    """Computes the output shape for the `vroom_q6_k_matmul` graph op."""
    comptime assert type_of(a).dtype == .float32, "a must be float32"
    comptime assert type_of(a).rank == 2, "a must be rank 2"
    comptime assert type_of(b).dtype == .uint8, "b must be uint8"
    comptime assert type_of(b).rank == 2, "b must be rank 2"
    return IndexList[2](
        Int(coord_to_index_list(a.shape().tuple())[0]),
        Int(coord_to_index_list(b.shape().tuple())[0]),
    )


@extensibility.register("vroom_q6_k_repack_weights")
struct VroomQ6KRepackWeights:
    """Registers the `vroom_q6_k_repack_weights` graph op with the graph compiler.
    """

    @staticmethod
    @always_inline
    def execute[
        _trace_name: StaticString,
    ](
        b_packed: OutputTensor[dtype=.uint8, rank=2, ...],
        b: InputTensor[dtype=.uint8, rank=2, ...],
    ) raises:
        matmul_Q6_K_pack_b(
            b.to_tile_tensor[.int64](),
            b_packed.to_tile_tensor[.int64](),
        )


@extensibility.register_shape_function("vroom_q6_k_repack_weights")
def vroom_q6_k_repack_weights_shape(
    b: Some[Tensor],
) -> IndexList[type_of(b).rank]:
    """Computes the output shape for the `vroom_q6_k_repack_weights` graph op.
    """
    comptime assert type_of(b).dtype == .uint8, "b must be uint8"
    comptime assert type_of(b).rank == 2, "b must be rank 2"
    return rebind[IndexList[type_of(b).rank]](
        coord_to_index_list(b.shape().tuple())
    )


@extensibility.register("qmatmul_b4_g32")
struct QMatmulGPU_b4_g32:
    """Registers the `qmatmul_b4_g32` graph op with the graph compiler."""

    @staticmethod
    @always_inline
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        c: OutputTensor[dtype=.bfloat16, rank=2, ...],
        a: InputTensor[dtype=.bfloat16, rank=2, ...],
        b: InputTensor[dtype=.uint8, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        matmul_gpu_qint4[32, target](
            c.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            ctx,
        )


@extensibility.register_shape_function("qmatmul_b4_g32")
def qmatmul_b4_g32_shape(a: Some[Tensor], b: Some[Tensor]) -> IndexList[2]:
    """Computes the output shape for the `qmatmul_b4_g32` graph op."""
    comptime assert type_of(a).rank == 2, "a must be rank 2"
    comptime assert type_of(b).rank == 2, "b must be rank 2"
    return IndexList[2](
        Int(coord_to_index_list(a.shape().tuple())[0]),
        Int(coord_to_index_list(b.shape().tuple())[0]),
    )


@extensibility.register("qmatmul_b4_g128")
struct QMatmulGPU_b4_g128:
    """Registers the `qmatmul_b4_g128` graph op with the graph compiler."""

    @staticmethod
    @always_inline
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        c: OutputTensor[dtype=.bfloat16, rank=2, ...],
        a: InputTensor[dtype=.bfloat16, rank=2, ...],
        b: InputTensor[dtype=.uint8, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        matmul_gpu_qint4[128, target](
            c.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            ctx,
        )


@extensibility.register_shape_function("qmatmul_b4_g128")
def qmatmul_b4_g128_shape(a: Some[Tensor], b: Some[Tensor]) -> IndexList[2]:
    """Computes the output shape for the `qmatmul_b4_g128` graph op."""
    comptime assert type_of(a).rank == 2, "a must be rank 2"
    comptime assert type_of(b).rank == 2, "b must be rank 2"
    return IndexList[2](
        Int(coord_to_index_list(a.shape().tuple())[0]),
        Int(coord_to_index_list(b.shape().tuple())[0]),
    )


@extensibility.register("GGUF_gpu_repack_q4_0")
struct QMatmulGPURepackGGUF:
    """Registers the `GGUF_gpu_repack_q4_0` graph op with the graph compiler."""

    @staticmethod
    @always_inline
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        b_packed: OutputTensor[dtype=.uint8, rank=2, ...],
        b: InputTensor[dtype=.uint8, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        gpu_qint4_repack_Q4_0[target](
            b.to_tile_tensor(), b_packed.to_tile_tensor(), ctx
        )


@extensibility.register_shape_function("GGUF_gpu_repack_q4_0")
def GGUF_gpu_repack_q4_0_shape(
    b: Some[Tensor],
) -> IndexList[type_of(b).rank]:
    """Computes the output shape for the `GGUF_gpu_repack_q4_0` graph op."""
    comptime assert type_of(b).dtype == .uint8, "b must be uint8"
    comptime assert type_of(b).rank == 2, "b must be rank 2"
    return rebind[IndexList[type_of(b).rank]](
        coord_to_index_list(b.shape().tuple())
    )


@extensibility.register("GPTQ_gpu_repack_b4_g128")
struct QMatmulGPURepackGPTQ_b4_g128:
    """Registers the `GPTQ_gpu_repack_b4_g128` graph op with the graph compiler.
    """

    @staticmethod
    @always_inline
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        b_packed: OutputTensor[dtype=.uint8, rank=2, ...],
        b: InputTensor[dtype=.uint8, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        gpu_qint4_repack_GPTQ[128, target](
            b.to_tile_tensor(), b_packed.to_tile_tensor(), ctx=ctx
        )


@extensibility.register_shape_function("GPTQ_gpu_repack_b4_g128")
def GPTQ_gpu_repack_b4_g128_shape(b: Some[Tensor]) -> IndexList[2]:
    """Computes the output shape for the `GPTQ_gpu_repack_b4_g128` graph op."""
    comptime assert type_of(b).dtype == .uint8, "b must be uint8"
    comptime assert type_of(b).rank == 2, "b must be rank 2"
    var shape = b.shape()
    var shape_list = coord_to_index_list(shape.tuple())
    return IndexList[2](Int(shape_list[1]), Int(shape_list[0]))


@extensibility.register("GPTQ_gpu_repack_b4_g128_desc_act")
struct QMatmulGPURepackGPTQ_b4_g128_desc_act:
    """Registers the `GPTQ_gpu_repack_b4_g128_desc_act` graph op with the graph compiler.
    """

    @staticmethod
    @always_inline
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        b_packed: OutputTensor[dtype=.uint8, rank=2, ...],
        b: InputTensor[dtype=.uint8, rank=2, ...],
        perm_idx: InputTensor[dtype=.int32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        var perm_idx_lt = perm_idx.to_layout_tensor()
        gpu_qint4_repack_GPTQ[128, target](
            b.to_tile_tensor(),
            b_packed.to_tile_tensor(),
            LayoutTensor[.int32, Layout.row_major(UNKNOWN_VALUE)](
                perm_idx_lt.ptr,
                RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                    perm_idx_lt.runtime_layout.shape.value.canonicalize()
                ),
            ).as_imm(),
            ctx=ctx,
        )


@extensibility.register_shape_function("GPTQ_gpu_repack_b4_g128_desc_act")
def GPTQ_gpu_repack_b4_g128_desc_act_shape(
    b: Some[Tensor], perm_idx: Some[Tensor]
) -> IndexList[2]:
    """Computes the output shape for the `GPTQ_gpu_repack_b4_g128_desc_act` graph op.
    """
    comptime assert type_of(b).dtype == .uint8, "b must be uint8"
    comptime assert type_of(b).rank == 2, "b must be rank 2"
    comptime assert type_of(perm_idx).dtype == .int32, "perm_idx must be int32"
    comptime assert type_of(perm_idx).rank == 1, "perm_idx must be rank 1"
    var shape = b.shape()
    var shape_list = coord_to_index_list(shape.tuple())
    return IndexList[2](Int(shape_list[1]), Int(shape_list[0]))


@extensibility.register("mo.quantize.dynamic.block.scaled")
struct Struct_quantize_dynamic_block_scaled:
    """Registers the `mo.quantize.dynamic.block.scaled` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        out_dtype: DType,
        scales_type: DType,
        in_dtype: DType,
        //,
        scales_rank: Int,
        SF_VECTOR_SIZE: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=out_dtype, rank=2, ...],
        scales: OutputTensor[dtype=scales_type, rank=scales_rank, ...],
        input: InputTensor[dtype=in_dtype, rank=2, ...],
        tensor_sf: Float32,
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), (
            "quantize dynamic block scaled only support GPUs with native"
            " block scaled support"
        )

        quantize_dynamic_block_scaled[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            target=target,
        ](
            output.to_tile_tensor[.int64](),
            scales.to_tile_tensor[.int64](),
            input.to_tile_tensor[.int64](),
            tensor_sf,
            context,
        )


@extensibility.register("mo.grouped.quantize.dynamic.block.scaled")
struct Struct_grouped_quantize_dynamic_block_scaled:
    """Registers the `mo.grouped.quantize.dynamic.block.scaled` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        out_dtype: DType,
        scales_type: DType,
        in_dtype: DType,
        //,
        scales_rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=out_dtype, rank=2, ...],
        scales: OutputTensor[dtype=scales_type, rank=scales_rank, ...],
        input: InputTensor[dtype=in_dtype, rank=2, ...],
        row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        scales_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        expert_ids: InputTensor[dtype=.int32, rank=1, ...],
        sf_tensor: InputTensor[dtype=.float32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "grouped quantize dynamic block scaled only supports GPUs"

        grouped_quantize_dynamic_scaled_fp4_async(
            output.to_tile_tensor[.int64](),
            scales.to_tile_tensor[.int64](),
            input.to_tile_tensor[.int64](),
            row_offsets.to_tile_tensor[.int64](),
            scales_offsets.to_tile_tensor[.int64](),
            expert_ids.to_tile_tensor[.int64](),
            sf_tensor.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("mo.quantize.dynamic.block.scaled.mxfp4")
struct Struct_quantize_dynamic_block_scaled_mxfp4:
    """Registers the `mo.quantize.dynamic.block.scaled.mxfp4` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        in_dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=.uint8, rank=2, ...],
        scales: OutputTensor[dtype=.float8_e8m0fnu, rank=2, ...],
        input: InputTensor[dtype=in_dtype, rank=2, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), (
            "quantize dynamic block scaled only support GPUs with native"
            " block scaled support"
        )

        quantize_dynamic_block_scaled_mxfp4(
            output.to_tile_tensor[.int64](),
            scales.to_tile_tensor[.int64](),
            input.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("mo.dequant.mxfp4")
struct Struct_dequant_mxfp4:
    """Registers the `mo.dequant.mxfp4` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        out_type: DType,
        in_type: DType,
        scales_type: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=out_type, rank=2, ...],
        input: InputTensor[dtype=in_type, rank=2, ...],
        scales: InputTensor[dtype=scales_type, rank=2, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "MXFP4 dequant only supports GPUs"
        comptime assert out_type in (
            DType.bfloat16,
            DType.float8_e4m3fn,
        ), "MXFP4 dequant output must be bfloat16 or float8_e4m3fn"
        comptime assert (
            in_type == .uint8
        ), "MXFP4 dequant input must be uint8 (packed FP4)"
        comptime assert (
            scales_type == .float8_e8m0fnu
        ), "MXFP4 dequant scales must be float8_e8m0fnu"

        var in_tt = input.to_tile_tensor[.int64]()
        var scales_tt = scales.to_tile_tensor[.int64]()
        var out_tt = output.to_tile_tensor[.int64]()

        var num_rows = Int(in_tt.dim[0]())
        # num_cols is the unpacked column count (2x packed)
        var num_cols = Int(in_tt.dim[1]()) * 2

        dequant_mxfp4(
            context,
            out_tt,
            in_tt,
            scales_tt,
            num_rows=num_rows,
            num_cols=num_cols,
        )


@extensibility.register("mo.quantize.dynamic.block.scaled.mxfp6")
struct Struct_quantize_dynamic_block_scaled_mxfp6:
    """Registers the `mo.quantize.dynamic.block.scaled.mxfp6` graph op."""

    @always_inline
    @staticmethod
    def execute[
        in_dtype: DType,
        //,
        FP6_FORMAT: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=.uint8, rank=2, ...],
        scales: OutputTensor[dtype=.float8_e8m0fnu, rank=2, ...],
        input: InputTensor[dtype=in_dtype, rank=2, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), (
            "MXFP6 quantization requires a GPU with native block-scaled support"
        )
        comptime assert FP6_FORMAT in (
            0,
            1,
        ), "FP6_FORMAT must be 0 (E2M3) or 1 (E3M2)"

        quantize_mxfp6_amd[FP6Format(FP6_FORMAT)](
            context,
            output.to_tile_tensor[.int64](),
            scales.to_tile_tensor[.int64](),
            input.to_tile_tensor[.int64](),
        )


@extensibility.register("mo.dequant.mxfp6")
struct Struct_dequant_mxfp6:
    """Registers the `mo.dequant.mxfp6` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        out_type: DType,
        in_type: DType,
        scales_type: DType,
        //,
        FP6_FORMAT: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=out_type, rank=2, ...],
        input: InputTensor[dtype=in_type, rank=2, ...],
        scales: InputTensor[dtype=scales_type, rank=2, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "MXFP6 dequant only supports GPUs"
        comptime assert out_type in (
            DType.bfloat16,
            DType.float8_e4m3fn,
        ), "MXFP6 dequant output must be bfloat16 or float8_e4m3fn"
        comptime assert (
            in_type == .uint8
        ), "MXFP6 dequant input must be uint8 (packed FP6)"
        comptime assert (
            scales_type == .float8_e8m0fnu
        ), "MXFP6 dequant scales must be float8_e8m0fnu"
        comptime assert FP6_FORMAT in (
            0,
            1,
        ), "FP6_FORMAT must be 0 (E2M3) or 1 (E3M2)"

        var in_tt = input.to_tile_tensor[.int64]()
        var scales_tt = scales.to_tile_tensor[.int64]()
        var out_tt = output.to_tile_tensor[.int64]()

        var num_rows = Int(in_tt.dim[0]())
        var num_cols = (Int(in_tt.dim[1]()) * 8) // 6

        dequant_mxfp6[FP6Format(FP6_FORMAT)](
            context,
            out_tt,
            in_tt,
            scales_tt,
            num_rows=num_rows,
            num_cols=num_cols,
        )


@extensibility.register("mo.interleave.block.scales")
struct Struct_interleave_block_scales:
    """Registers the `mo.interleave.block.scales` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        scales_type: DType,
        //,
        SF_VECTOR_SIZE: Int,
        target: StaticString,
    ](
        output_scales: OutputTensor[dtype=scales_type, rank=5, ...],
        input_scales: InputTensor[dtype=scales_type, rank=2, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), (
            "quantize dynamic block scaled only support GPUs with native"
            " block scaled support"
        )

        block_scales_interleave[SF_VECTOR_SIZE=SF_VECTOR_SIZE, target=target](
            output_scales.to_tile_tensor[.int64](),
            input_scales.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("mo.block.scaled.preshuffle.b.5d")
struct Struct_block_scaled_preshuffle_b_5d:
    """Run the AMD CDNA4 MXFP4 B 5D preshuffle as a custom op.

    Used to pre-bake weights into `Shuffler[E].b_5d_grouped_layout` (the
    layout the `block_scaled_grouped_matmul_amd_preb` reader expects) without
    paying the >1 h CPU-side numpy shuffle on every model load.
    """

    @always_inline
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=.uint8, rank=3, ...],
        input: InputTensor[dtype=.uint8, rank=3, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "mo.block.scaled.preshuffle.b.5d is GPU-only (AMD CDNA4 consumer)"

        var raw_tt = input.to_tile_tensor[.int64]()
        var dst_tt = output.to_tile_tensor[.int64]()
        comptime E = type_of(raw_tt).static_shape[0]
        comptime N = type_of(raw_tt).static_shape[1]
        comptime K_BYTES = type_of(raw_tt).static_shape[2]
        Shuffler[E].preshuffle_b_5d[N=N, K_BYTES=K_BYTES](
            raw_tt, dst_tt, context
        )


@extensibility.register("mo.block.scaled.preshuffle.scale.4d_per_expert")
struct Struct_block_scaled_preshuffle_scale_4d_per_expert:
    """Per-step A-scale preshuffle for the AMD CDNA4 preb grouped matmul.

    Takes row-major E8M0 A-scales `[total_tokens, K_SCALES]` and writes
    cell-packed scales into per-expert fixed-stride slots of size
    `max_padded_M = align_up(max_num_tokens_per_expert, 32)`. The
    `block_scaled_grouped_matmul_amd_preb` kernel reads slot `e * max_padded_M`
    for expert slot `e`. Inactive slots and pad rows are left untouched
    by this kernel; the matmul's per-expert tight V# bound guards
    out-of-range reads.
    """

    @always_inline
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=.float8_e8m0fnu, rank=2, ...],
        input: InputTensor[dtype=.float8_e8m0fnu, rank=2, ...],
        expert_start_indices: InputTensor[dtype=.uint32, rank=1, ...],
        max_num_tokens_per_expert: UInt32,
        num_active_experts: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "mo.block.scaled.preshuffle.scale.4d_per_expert is GPU-only"

        # E8M0 bytes feed the launcher as raw uint8 (the cell-packing is
        # byte-level). Bitcast the input/output tile pointers so dtype
        # metadata matches the launcher's `DType.uint8` TileTensor sig.
        var raw_e8 = input.to_tile_tensor[.int64]()
        var dst_e8 = output.to_tile_tensor[.int64]()
        var raw_tt = TileTensor[mut=False](
            raw_e8._storage.bitcast[UInt8](), raw_e8.layout
        )
        var dst_tt = TileTensor[mut=True](
            dst_e8._storage.bitcast[UInt8](), dst_e8.layout
        )
        var a_off_tt = expert_start_indices.to_tile_tensor[.int64]()
        comptime K_SCALES = type_of(raw_tt).static_shape[1]
        # Persistent grid: one CTA per WG slot, grid-strides real tiles.
        # `cu_count * 2` matches the matmul's persistent dispatch (see
        # `PreShuffledBGroupedGEMM.total_wg`).
        comptime total_wg = context.default_device_info.sm_count * 2
        Shuffler[1].preshuffle_grouped_scale_4d_gpu[K_SCALES=K_SCALES](
            raw_tt,
            dst_tt,
            a_off_tt,
            Int(num_active_experts),
            Int(max_num_tokens_per_expert),
            total_wg,
            context,
        )


@extensibility.register("mo.unfused_qkv_matmul.ragged.paged.gguf_quantized")
struct Struct_unfused_qkv_matmul_ragged_paged_gguf_quantized:
    """Registers the `mo.unfused_qkv_matmul.ragged.paged.gguf_quantized` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        quantization_encoding_q: StaticString,
        quantization_encoding_k: StaticString,
        quantization_encoding_v: StaticString,
    ](
        output: OutputTensor[dtype=.float32, rank=2, ...],
        hidden_state: InputTensor[dtype=.float32, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        q_weight: InputTensor[dtype=.uint8, rank=2, ...],
        k_weight: InputTensor[dtype=.uint8, rank=2, ...],
        v_weight: InputTensor[dtype=.uint8, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=.float32, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        unfused_qkv_matmul_ragged_paged_gguf_quantized[
            quantization_encoding_q,
            quantization_encoding_k,
            quantization_encoding_v,
        ](
            hidden_state.to_layout_tensor(),
            input_row_offsets.to_layout_tensor(),
            q_weight.to_layout_tensor(),
            k_weight.to_layout_tensor(),
            v_weight.to_layout_tensor(),
            kv_collection,
            layer_idx,
            output.to_layout_tensor(),
            ctx,
        )


@extensibility.register("mo.quantize_static_scaled_float8")
struct QuantizeStaticScaledFloat8[*, scale_is_inverted: Bool]:
    """Registers the `mo.quantize_static_scaled_float8` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        input_type: DType,
        output_type: DType,
        scale_type: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_type, rank=2, ...],
        input: FusedInputTensor[dtype=input_type, rank=2, ...],
        scale: Scalar[scale_type],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert is_gpu[target](), "only valid on GPUs"
        comptime assert output_type in (
            DType.float8_e4m3fn,
            DType.float8_e4m3fnuz,
        ), "output dtype should be float8_e4m3fn or float8_e4m3fnuz"

        # A single-use elementwise producer feeding this quantize (MLP relu2,
        # the gated-group-norm final cast, residual casts) fuses INTO this
        # load lambda, saving one kernel launch + one full-width HBM
        # materialization per FP8 Linear activation. Math is bit-identical to
        # the standalone `quantize_static_scaled_fp8` path: cast to f32, then
        # `fp8_quantize(v, 1.0/scale)`. The original kernel ignored the
        # `scale_is_inverted` param and always used `1.0/scale`; preserved here.
        var inversed_scale = 1.0 / scale.cast[.float32]()

        @always_inline
        def quant_fn[
            width: Int, element_alignment: Int
        ](idx: IndexList[2]) {var input, var inversed_scale} -> SIMD[
            output_type, width
        ]:
            var v = input._fused_load[
                width, element_alignment=element_alignment
            ](idx).cast[.float32]()
            return fp8_quantize[output_type, use_clamp=True](v, inversed_scale)

        foreach[
            target=target,
            _trace_name="scaled_fp8_quant",
        ](quant_fn, output, ctx)


@extensibility.register("mo.quantize_tensor_dynamic_scaled_float8")
struct QuantizeTensorDynamicScaledFloat8:
    """Registers the `mo.quantize_tensor_dynamic_scaled_float8` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        input_type: DType,
        scales_type: DType,
        output_type: DType,
        //,
        group_size_or_per_token: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_type, rank=2, ...],
        scales: OutputTensor[dtype=scales_type, rank=2, ...],
        input: FusedInputTensor[dtype=input_type, rank=2, ...],
        scale_ub: Float32,
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](row: Int, col: Int) {var input} -> SIMD[input_type, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                Index(row, col)
            )

        quantize_tensor_dynamic_scaled_fp8[
            in_dtype=input_type,
            group_size_or_per_token=group_size_or_per_token,
            num_cols=Int(input.static_spec.shape_tuple[1]),
        ](
            input_fn,
            output.to_tile_tensor[.int64](),
            scales.to_tile_tensor[.int64](),
            scale_ub,
            ctx,
            num_rows=input.dim_size(0),
        )


@extensibility.register("mo.quantize_dynamic_scaled_float8")
struct QuantizeDynamicScaledFloat8:
    """Registers the `mo.quantize_dynamic_scaled_float8` graph op with the graph compiler.
    """

    @__parameter
    @always_inline
    @staticmethod
    def execute[
        input_type: DType,
        scales_type: DType,
        output_type: DType,
        //,
        group_size_or_per_token: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_type, rank=2, ...],
        scales: OutputTensor[dtype=scales_type, rank=2, ...],
        input: FusedInputTensor[dtype=input_type, rank=2, ...],
        scale_ub: Float32,
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](row: Int, col: Int) {var input} -> SIMD[input_type, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                Index(row, col)
            )

        quantize_dynamic_scaled_fp8[
            in_dtype=input_type,
            group_size_or_per_token=group_size_or_per_token,
            num_cols=Int(input.static_spec.shape_tuple[1]),
        ](
            input_fn,
            output.to_tile_tensor[.int64](),
            scales.to_tile_tensor[.int64](),
            scale_ub,
            ctx,
            num_rows=input.dim_size(0),
        )
