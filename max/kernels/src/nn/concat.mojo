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
"""Implements tensor concatenation along a specified axis for CPU and GPU targets."""

from std.collections import Optional
from std.math import align_down, align_up, ceildiv, divmod

from std.sys._build import is_debug_build
from std.sys.info import CompilationTarget, simd_width_of, size_of

from max.algorithm.functional import (
    _get_start_indices_of_nth_subvolume,
    dual_elementwise,
    elementwise,
    sync_parallelize,
)
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from max.gpu.host.info import is_cpu, is_valid_target
from layout import (
    Coord,
    TensorLayout,
    TensorEngine,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from std.memory import unsafe_memcpy
from max.runtime.tracing import Trace, TraceLevel, get_safe_task_id

from std.utils import IndexList, StaticTuple, product

from .gather_scatter import normalize_neg_index

# Reuse the static-divisor subvolume decomposition the layer_norm / rms_norm /
# softmax migration introduced (the static-folding counterpart of
# `_get_start_indices_of_nth_subvolume`). When an output dim is statically
# known in the `Coord` *type* carried by the tensor layout, the `divmod`
# strength-reduces to a magic-multiply + shift (SASS `IMAD.WIDE`/`SHF`, no
# `IDIV`/`MUFU.RCP`); dynamic dims fall back to the runtime divide.
from .shapes import _get_start_indices_of_nth_subvolume_static

comptime elementwise_epilogue_type = def[
    c_type: DType, rank: Int, width: SIMDLength = 1, *, alignment: Int = 1
](IndexList[rank], SIMD[c_type, width]) capturing -> None


@always_inline
@__parameter
def preferred_simd_width[dtype: DType]() -> Int:
    """SIMD scalar count for fused GPU concat vectorization.

    Uses 32-byte global loads on ``sm_100a``; otherwise the target's native
    ``simd_width_of`` for ``dtype`` on the active GPU compilation target.

    Parameters:
        dtype: Element type used to compute the SIMD scalar width.
    """
    return (
        32
        // size_of[dtype]() if CompilationTarget[get_gpu_target()]._is_arch[
            "sm_100a"
        ]() else simd_width_of[dtype, target=get_gpu_target()]()
    )


# ===-----------------------------------------------------------------------===#
# concat
# ===-----------------------------------------------------------------------===#


@always_inline
def memcpy_or_fuse[
    rank: Int,
    dtype: DType,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    dest_data: MutPointer[Int8, _],
    out_byte_offset: Int,
    src_data: ImmPointer[Int8, _],
    n: Int,
    out_shape: IndexList[rank, ...],
) raises:
    """Copies ``n`` bytes from ``src_data`` into ``dest_data`` at ``out_byte_offset``, applying ``epilogue_fn`` elementwise when present.

    When no epilogue function is supplied, this performs a plain ``memcpy`` of
    ``n`` bytes. When an epilogue function is supplied, the source bytes are
    reinterpreted as typed elements and the epilogue is applied scalar-by-scalar
    so that fused concat can transform values while copying them into the
    output buffer.

    Parameters:
        rank: Number of dimensions in the output tensor used for epilogue
            indexing.
        dtype: Element type of the tensors being copied.
        epilogue_fn: Optional elementwise function applied to each copied
            element; when absent, a plain byte ``memcpy`` is performed.

    Args:
        dest_data: Destination byte buffer to write into.
        out_byte_offset: Byte offset into ``dest_data`` where the copy starts.
        src_data: Source byte buffer to read from.
        n: Number of bytes to copy.
        out_shape: Shape of the output tensor used to compute multi-dimensional
            indices for the epilogue function.
    """
    comptime if not epilogue_fn:
        unsafe_memcpy(
            dest=dest_data.unsafe_offset(out_byte_offset), src=src_data, count=n
        )
    else:
        comptime func = epilogue_fn.value()
        comptime simd_width = simd_width_of[dtype]()

        var typed_offset = out_byte_offset // size_of[dtype]()
        var typed_len = n // size_of[dtype]()
        assert (
            n % size_of[dtype]() == 0
            and out_byte_offset % size_of[dtype]() == 0
        ), "offset and length must be dividable by size_of[dtype]"

        # Cast
        var shape_1d = Coord(typed_len)
        var typed_src = src_data.unsafe_bitcast[Scalar[dtype]]()
        var input = TileTensor(
            typed_src,
            row_major(Coord(shape_1d)),
        )

        @always_inline
        def epilogue_wrapper[
            simd_width: Int, alignment: Int = 1
        ](index: Coord) {var}:
            var load = input.load[width=simd_width, alignment=1](index)

            var out_index = _get_start_indices_of_nth_subvolume[0](
                Int(index[0].value()) + typed_offset,
                out_shape,
            )

            func[dtype, rank, simd_width](
                out_index.cast[.int64](),
                load,
            )
            return

        # We must run scalar to be conservative. This is because the fused
        # output lambda might operate on views (e.g., broadcast) that does not
        # always work with indices produced from a linearized address.
        elementwise[simd_width=1](
            epilogue_wrapper, shape_1d, DeviceContext(api="cpu")
        )


@fieldwise_init
struct _Span(TrivialRegisterPassable):
    var start: Int
    var end: Int

    @always_inline("nodebug")
    def empty(self) -> Bool:
        return not (self.start < self.end)

    @always_inline("nodebug")
    def intersect(self, other: Self) -> Self:
        return Self(max(self.start, other.start), min(self.end, other.end))


@fieldwise_init
struct _CanonicallyReshapedBuffer[mut: Bool, //, origin: Origin[mut=mut]](
    TrivialRegisterPassable
):
    var data: Pointer[Int8, Self.origin]
    var h: Int
    var w: Int
    var c: Int


def _canonical_reshape[
    dtype: DType
](
    buf: TileTensor[dtype, address_space=.GENERIC, ...],
    axis: Int,
) -> _CanonicallyReshapedBuffer[buf.origin]:
    var shape = coord_to_index_list(buf.layout.shape_coord())
    var h = product(shape, 0, axis)
    var w = Int(buf.dim(axis))
    var c = product(shape, axis + 1, buf.rank) * size_of[dtype]()
    return _CanonicallyReshapedBuffer(buf.ptr.unsafe_bitcast[Int8](), h, w, c)


def _canonical_reshape_output[
    input_origin: ImmOrigin,
    InputLayoutType: TensorLayout,
    //,
    dtype: DType,
](
    out_buf: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    axis: Int,
    inputs: List[TileTensor[dtype, InputLayoutType, input_origin]],
) -> _CanonicallyReshapedBuffer[out_buf.origin]:
    var input0_canon = _canonical_reshape(inputs[0], axis)
    var out_w = input0_canon.w
    for i in range(1, len(inputs)):
        out_w += Int(inputs[i].dim(axis))
    return _CanonicallyReshapedBuffer(
        out_buf.ptr.unsafe_bitcast[Int8](),
        input0_canon.h,
        out_w,
        input0_canon.c,
    )


def _concat_parallel[
    input_origin: ImmOrigin,
    InputLayoutType: TensorLayout,
    //,
    dtype: DType,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    axis: Int,
    inputs: List[TileTensor[dtype, InputLayoutType, input_origin]],
    ctx: Optional[DeviceContext] = None,
) raises:
    var output_canon = _canonical_reshape_output(output, axis, inputs)

    var output_h = output_canon.h
    var output_w = output_canon.w
    var output_c = output_canon.c
    var output_wc = output_w * output_c
    var output_data = output_canon.data

    var total_output_bytes = output_h * output_wc
    var output_shape = rebind[IndexList[output.rank]](
        coord_to_index_list(output.layout.shape_coord())
    )

    comptime KB = 1024
    comptime parallel_chunk_size = 64 * KB  # TODO autotune
    var num_chunks = ceildiv(total_output_bytes, parallel_chunk_size)

    def do_chunk(
        chunk_index: Int,
    ) raises {
        var total_output_bytes,
        var output_h,
        var output_c,
        var output_data,
        var output_wc,
        var output_shape,
        imm inputs,
        imm axis,
    }:
        # "Amount" refers to byte-offsets into logical copy order, not into
        # output buffer.
        var chunk_start_amount = chunk_index * parallel_chunk_size
        var chunk_end_amount = min(
            (chunk_index + 1) * parallel_chunk_size, total_output_bytes
        )
        var chunk_span = _Span(chunk_start_amount, chunk_end_amount)

        var amount_traversed = 0
        var output_wc_offset = 0

        for input_index in range(len(inputs)):
            var input = inputs[input_index]
            var input_canon = _canonical_reshape(input, axis)
            var input_h = input_canon.h
            var input_w = input_canon.w
            var input_c = input_canon.c
            var input_wc = input_w * input_c
            var input_data = input_canon.data
            assert input_h == output_h, "input_h != output_h"
            assert input_c == output_c, "input_c != output_c"
            var input_byte_size = input_h * input_wc

            var input_span = _Span(
                amount_traversed, amount_traversed + input_byte_size
            )
            var overlap_span = chunk_span.intersect(input_span)

            if not overlap_span.empty():
                # These are offsets of what we're trying to compute relative to
                # the start of the input buffer.
                var overlap_rel_start = overlap_span.start - input_span.start
                var overlap_rel_end = overlap_span.end - input_span.start
                # These are offsets into the input, chopping off the ends so as
                # to align to an integral 'h' index.
                var overlap_full_rel_start = align_up(
                    overlap_rel_start, input_wc
                )
                var overlap_full_rel_end = align_down(overlap_rel_end, input_wc)

                if overlap_full_rel_end < overlap_full_rel_start:
                    # If we hit here, this was probably a bad chunking choice,
                    # but var's handle it correctly anyways.
                    memcpy_or_fuse[output.rank, dtype, epilogue_fn](
                        output_data,
                        output_wc_offset
                        + overlap_rel_start // input_wc * output_wc
                        + overlap_rel_start % input_wc,
                        input_data.unsafe_offset(overlap_rel_start),
                        overlap_rel_end - overlap_rel_start,
                        output_shape,
                    )
                else:
                    # OK, we have maybe stragglers on the start and end, and a
                    # nice solid middle section -- var's handle those
                    # separately.
                    # First, leading stragglers:
                    memcpy_or_fuse[output.rank, dtype, epilogue_fn](
                        output_data,
                        output_wc_offset
                        + overlap_rel_start // input_wc * output_wc
                        + overlap_rel_start % input_wc,
                        input_data.unsafe_offset(overlap_rel_start),
                        overlap_full_rel_start - overlap_rel_start,
                        output_shape,
                    )
                    # Now, fully-aligned sections:
                    var in_ptr = input_data.unsafe_offset(
                        overlap_full_rel_start
                    )
                    var end_in_ptr = input_data.unsafe_offset(
                        overlap_full_rel_end
                    )
                    var out_ptr_offset = (
                        output_wc_offset
                        + overlap_full_rel_start // input_wc * output_wc
                    )

                    while in_ptr < end_in_ptr:
                        memcpy_or_fuse[output.rank, dtype, epilogue_fn](
                            output_data,
                            out_ptr_offset,
                            in_ptr,
                            input_wc,
                            output_shape,
                        )
                        in_ptr = in_ptr.unsafe_offset(input_wc)
                        out_ptr_offset += output_wc
                    # Lastly, trailing stragglers:
                    memcpy_or_fuse[output.rank, dtype, epilogue_fn](
                        output_data,
                        out_ptr_offset,
                        in_ptr,
                        overlap_rel_end - overlap_full_rel_end,
                        output_shape,
                    )

            amount_traversed += input_byte_size
            output_wc_offset += input_wc

        assert (
            amount_traversed == total_output_bytes
        ), "amount_traversed != total_output_bytes"

    # The do_chunk closure captures the stack allocated Buffer,
    # so this kernel must be run synchronously.
    sync_parallelize(do_chunk, num_chunks, ctx)


@always_inline
def _concat[
    input_origin: ImmOrigin,
    InputLayoutType: TensorLayout,
    //,
    dtype: DType,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    axis: Int,
    inputs: List[TileTensor[dtype, InputLayoutType, input_origin]],
) raises:
    """Concatenate inputs along axis and store in output.

    This simplifies the implementation by reshaping the output and inputs into 3D
    buffers. input i has dims [h, wi, c]. The output has dims [h, sum(wi), c] where
    i ranges from [0, num_inputs).

    Reshaping the buffer does not change the memory layout. After reshaping to 3D
    it is easy to visualize that the inputs can be copied in w x c sized
    contiguous slices along the h dimension.

    """

    var h = product(
        coord_to_index_list(inputs[0].layout.shape_coord()), 0, axis
    )
    var c = product(
        coord_to_index_list(inputs[0].layout.shape_coord()),
        axis + 1,
        output.rank,
    )

    var w_out: Int = 0
    for i in range(len(inputs)):
        w_out += Int(inputs[i].dim(axis))

    var stride_h_out = w_out * c
    var stride_w_out = c

    var w_offset: Int = 0
    for i in range(len(inputs)):
        # copy one w x c slice along h at a time
        var w = Int(inputs[i].dim(axis))
        for j in range(h):
            var input_offset = j * w * c
            var output_offset = j * stride_h_out + w_offset * stride_w_out
            # these slices are contiguous
            memcpy_or_fuse[output.rank, dtype, epilogue_fn](
                output.ptr.unsafe_bitcast[Int8](),
                output_offset * size_of[dtype](),
                inputs[i]
                .ptr.unsafe_offset(input_offset)
                .unsafe_bitcast[Int8](),
                w * c * size_of[dtype](),
                rebind[IndexList[output.rank]](
                    coord_to_index_list(output.layout.shape_coord())
                ),
            )
        w_offset += w


@always_inline
def _concat_inner[
    input_origin: ImmOrigin,
    InputLayoutType: TensorLayout,
    //,
    dtype: DType,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    inputs: List[TileTensor[dtype, InputLayoutType, input_origin]],
) raises:
    var num_elems_copied: Int = 0
    for i in range(len(inputs)):
        var buffer_len = inputs[i].num_elements()
        memcpy_or_fuse[output.rank, dtype, epilogue_fn](
            output.ptr.unsafe_bitcast[Int8](),
            num_elems_copied * size_of[dtype](),
            inputs[i].ptr.unsafe_bitcast[Int8](),
            buffer_len * size_of[dtype](),
            rebind[IndexList[output.rank]](
                coord_to_index_list(output.layout.shape_coord())
            ),
        )
        num_elems_copied += buffer_len


@always_inline
def _check_input_consistency[
    input_origin: ImmOrigin,
    InputLayoutType: TensorLayout,
    //,
    dtype: DType,
](axis: Int, inputs: List[TileTensor[dtype, InputLayoutType, input_origin]],):
    comptime if not is_debug_build():
        return
    # check inputs have same rank and same dims except for axis dim
    for i in range(len(inputs)):
        for j in range(inputs[i].rank):
            assert j == axis or inputs[0].dim(j) == inputs[i].dim(j), (
                "all concat inputs must have the same dimensions in the"
                " non-concat axes"
            )


@always_inline
def _concat_serial[
    input_origin: ImmOrigin,
    InputLayoutType: TensorLayout,
    //,
    dtype: DType,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    axis: Int,
    inputs: List[TileTensor[dtype, InputLayoutType, input_origin]],
) raises:
    _check_input_consistency[dtype](axis, inputs)

    var all_outer_dims_singvaron = True
    for i in range(axis):
        if inputs[0].dim(i) == 1:
            continue

        all_outer_dims_singvaron = False
        break

    if all_outer_dims_singvaron:
        _concat_inner[dtype, epilogue_fn](output, inputs)
        return

    _concat[dtype, epilogue_fn](output, axis, inputs)


@always_inline
def _concat_cpu[
    input_origin: ImmOrigin,
    InputLayoutType: TensorLayout,
    //,
    dtype: DType,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    axis: Int,
    inputs: List[TileTensor[dtype, InputLayoutType, input_origin]],
    ctx: Optional[DeviceContext] = None,
) raises:
    _check_input_consistency[dtype](axis, inputs)

    @always_inline
    def dispatch_serial(unused_thread_idx: Int) raises {imm}:
        _concat_serial[dtype, epilogue_fn](output, axis, inputs)

    comptime KB = 1024
    comptime min_work_for_parallel = 128 * KB  # TODO: autotune

    var output_bytes = output.num_elements() * size_of[dtype]()

    if output_bytes < min_work_for_parallel:
        # The dispatch_serial closure captures the stack allocated
        # Buffer, so this kernel must be run synchronously.
        sync_parallelize(dispatch_serial, 1, ctx)
    else:
        _concat_parallel[epilogue_fn=epilogue_fn](output, axis, inputs, ctx=ctx)


@always_inline
def concat_shape[
    input_origin: ImmOrigin,
    InputLayoutType: TensorLayout,
    //,
    input_type: DType,
](
    input_bufs: List[TileTensor[input_type, InputLayoutType, input_origin]],
    axis: Int,
) raises -> IndexList[InputLayoutType.rank]:
    """
    Compute the output shape of a `pad` operation, and assert the inputs are
    compatible.

    Parameters:
        input_origin: Origin of the input tensor.
        InputLayoutType: Layout type of the input tensor.
        input_type: Type of the input tensor.

    Args:
        input_bufs: The input tensors list.
        axis: The axis.

    Returns:
        The output shape.
    """

    # extract hyper parameters
    var normalized_axis = normalize_neg_index(axis, InputLayoutType.rank)

    @__parameter
    @always_inline
    def shape_equal_ignore_axis(
        s1: IndexList[InputLayoutType.rank],
        s2: IndexList[InputLayoutType.rank],
    ) -> Bool:
        for i in range(InputLayoutType.rank):
            if i != axis and s1[i] != s2[i]:
                return False
        return True

    var concat_axis_dim_sum = 0
    for i in range(len(input_bufs)):
        concat_axis_dim_sum += Int(input_bufs[i].dim(normalized_axis))
        if not shape_equal_ignore_axis(
            rebind[IndexList[InputLayoutType.rank]](
                coord_to_index_list(input_bufs[0].layout.shape_coord())
            ),
            rebind[IndexList[InputLayoutType.rank]](
                coord_to_index_list(input_bufs[i].layout.shape_coord())
            ),
        ):
            raise Error(
                "[concat_from_list] input shapes must match except at concat"
                " axis"
            )

    # compute and return the output shape
    var output_shape = rebind[IndexList[InputLayoutType.rank]](
        coord_to_index_list(input_bufs[0].layout.shape_coord())
    )
    output_shape[normalized_axis] = concat_axis_dim_sum
    return output_shape


@always_inline
def concat[
    input_origin: ImmOrigin,
    InputLayoutType: TensorLayout,
    //,
    dtype: DType,
    target: StaticString = "cpu",
    epilogue_fn: Optional[elementwise_epilogue_type] = None,
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    axis: Int,
    inputs: StaticTuple[
        TileTensor[dtype, InputLayoutType, input_origin],
        ...,
    ],
    context: DeviceContext,
) raises:
    """Concatenates ``inputs`` along ``axis`` into ``output`` for the given ``target``.

    Dispatches to the CPU or GPU concat implementation based on ``target`` and
    applies ``epilogue_fn`` elementwise to each copied element when supplied.
    Returns early when the output tensor is empty.

    Parameters:
        input_origin: Origin of the input tensors (inferred).
        InputLayoutType: Layout type of the input tensors (inferred).
        dtype: Element type of the input and output tensors.
        target: Target device to dispatch to (defaults to ``"cpu"``).
        epilogue_fn: Optional elementwise function applied to each copied
            element (defaults to ``None``).

    Args:
        output: Destination tensor that receives the concatenated result.
        axis: Axis along which to concatenate the inputs.
        inputs: Static tuple of input tensors to concatenate.
        context: Device context used to schedule the work.
    """
    comptime assert is_valid_target[target](), "not a valid target"

    with Trace[TraceLevel.OP, target=target](
        "concat", task_id=get_safe_task_id(context)
    ):
        # Exit early if the tensors are empty.
        if output.num_elements() == 0:
            return
        comptime if is_cpu[target]():
            comptime InputTileType = TileTensor[
                dtype, InputLayoutType, input_origin
            ]
            var inputVec = List(
                length=inputs.size,
                fill_with=lambda (i: Int) -> InputTileType: inputs[i],
            )

            # Dynamic input length is required by `mo.concat_from_list`
            # TODO: Should we just provide a separate implementation for
            # `concat_from_list`, since dynamic input size does not work with
            # static sized input lambda tuple.
            _concat_cpu[dtype, epilogue_fn](
                output,
                axis,
                inputVec,
                ctx=Optional[DeviceContext](context),
            )
        else:
            _concat_gpu[dtype, epilogue_fn](
                # This is safe since `output` being an arg will keep the origin alive
                # for the duration of this call.
                output,
                axis,
                inputs,
                context,
            )


@__name(t"concat_gpu_flat_{dtype}_ax{axis}_w{vec_width}")
def _concat_gpu_flat_kernel[
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    OutputEngine: TensorEngine,
    InputLayoutType: TensorLayout,
    input_origin: ImmOrigin,
    InputEngine: TensorEngine,
    //,
    dtype: DType,
    num_inputs: Int,
    axis: Int,
    rank: Int,
    block_size: Int,
    vec_width: Int,
](
    output: TileTensor[
        dtype, OutputLayoutType, output_origin, Engine=OutputEngine
    ],
    inputs: StaticTuple[
        TileTensor[dtype, InputLayoutType, input_origin, Engine=InputEngine],
        num_inputs,
    ],
    inner_size: Int32,
    total_concat_dim: Int32,
    total_vec_items: Int32,
):
    """Flat-indexing GPU kernel for concat.

    Decomposes the output into canonical [outer, concat, inner] dimensions and
    uses flat pointer arithmetic to avoid multi-dimensional index decomposition
    and TileTensor coordinate-to-offset conversion overhead.
    """
    var _inner_size = Int(inner_size)
    var _total_concat_dim = Int(total_concat_dim)
    var _total_vec_items = Int(total_vec_items)
    var tid = block_idx.x * block_size + thread_idx.x
    if tid >= _total_vec_items:
        return

    var vec_idx = tid * vec_width

    # Decompose flat index into (outer, concat, inner) coordinates.
    var remaining, inner_idx = divmod(vec_idx, _inner_size)
    var outer_idx, concat_idx = divmod(remaining, _total_concat_dim)

    # Find which input this concat_idx belongs to and compute source offset.
    # Alignment is guaranteed: vec_idx is a multiple of vec_width, and
    # in_offset is a multiple of vec_width when inner_size % vec_width == 0
    # (enforced by the caller).
    var acc = 0
    comptime for i in range(num_inputs):
        var input_concat_dim = Int(inputs[i].dim(axis))
        if concat_idx < acc + input_concat_dim:
            var local_concat = concat_idx - acc
            var in_offset = (
                outer_idx * input_concat_dim + local_concat
            ) * _inner_size + inner_idx
            output.raw_store[alignment=vec_width](
                vec_idx,
                inputs[i].raw_load[
                    width=vec_width, alignment=vec_width, invariant=True
                ](in_offset),
            )
            return
        acc += input_concat_dim


@__name(t"concat_inner_most_single_dim_{dtype}")
def _concat_inner_most_single_dim[
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    OutputEngine: TensorEngine,
    InputLayoutType: TensorLayout,
    input_origin: ImmOrigin,
    InputEngine: TensorEngine,
    //,
    dtype: DType,
    num_inputs: Int,
    block_size: Int,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    output: TileTensor[
        dtype, OutputLayoutType, output_origin, Engine=OutputEngine
    ],
    inputs: StaticTuple[
        TileTensor[dtype, InputLayoutType, input_origin, Engine=InputEngine],
        num_inputs,
    ],
):
    var idx = block_idx.x * block_size + thread_idx.x
    # One thread per "row" of the concat inputs (last dim is 1 on each input).
    # `output.num_elements()` includes the stacked concat axis and must not be
    # used here; extra tail threads from `ceildiv` in `_concat_gpu` must exit
    # before subvolume indexing.
    var row_count = inputs[0].num_elements()
    if idx >= row_count:
        return

    # Static-divisor row -> n-D decomposition: the output's outer dims that are
    # statically known in `OutputLayoutType` fold the per-thread `divmod` to a
    # magic-multiply + shift (no `IDIV`); dynamic dims fall back to the runtime
    # divide read from the `Coord`'s leaf values. Behavior is bit-identical to
    # `_get_start_indices_of_nth_subvolume[1]`.
    var index = _get_start_indices_of_nth_subvolume_static(
        idx, output.layout.shape_coord()
    )
    var in_coord = Coord(index)

    comptime for i in range(num_inputs):
        var out_index = rebind[IndexList[output.rank]](index.canonicalize())
        out_index[output.rank - 1] = i
        var out_coord = Coord(out_index)

        comptime if epilogue_fn:
            comptime func = epilogue_fn.value()
            func[dtype, output.rank, 1](
                out_index, inputs[i].load[width=1](in_coord)
            )
        else:
            output.store(out_coord, inputs[i].load[width=1](in_coord))


@always_inline
def _concat_gpu_elementwise[
    input_origin: ImmOrigin,
    InputLayoutType: TensorLayout,
    InputEngine: TensorEngine,
    //,
    dtype: DType,
    num_inputs: Int,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, Engine=_, ...],
    axis: Int,
    inputs: StaticTuple[
        TileTensor[dtype, InputLayoutType, input_origin, Engine=InputEngine],
        num_inputs,
    ],
    ctx: DeviceContext,
) raises:
    # Without parameter dispatch there are 2 extra stack allocations in the GPU kernel
    comptime for i in range(output.rank):
        if i == axis:
            return _concat_gpu_elementwise[axis=i, epilogue_fn=epilogue_fn](
                output, inputs, ctx
            )


@always_inline
def _concat_gpu_elementwise[
    input_origin: ImmOrigin,
    InputLayoutType: TensorLayout,
    InputEngine: TensorEngine,
    //,
    axis: Int,
    dtype: DType,
    num_inputs: Int,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, Engine=_, ...],
    inputs: StaticTuple[
        TileTensor[dtype, InputLayoutType, input_origin, Engine=InputEngine],
        num_inputs,
    ],
    ctx: DeviceContext,
) raises:
    # Fast path: use flat-indexing kernel to avoid multi-dimensional index
    # decomposition overhead. Only when no epilogue function is needed.
    comptime if not epilogue_fn:
        var output_shape = coord_to_index_list(output.layout.shape_coord())
        var inner_size = 1
        comptime for dim_idx in range(axis + 1, output.rank):
            inner_size *= Int(output_shape[dim_idx])
        var total_concat_dim = Int(output_shape[axis])

        # Target 128-bit (16 byte) vector loads for optimal memory
        # throughput. This gives vec_width=4 for f32, 8 for f16/bf16,
        # 2 for f64, etc.
        comptime _vec_width = 16 // size_of[dtype]()
        comptime _block_size = 256

        @__parameter
        @always_inline
        def _launch_flat[_vw: Int]() raises:
            comptime kernel_fn = _concat_gpu_flat_kernel[
                OutputLayoutType=output.LayoutType,
                output_origin=output.origin,
                OutputEngine=output.Engine,
                InputLayoutType=InputLayoutType,
                input_origin=input_origin,
                InputEngine=InputEngine,
                dtype,
                num_inputs,
                axis,
                output.rank,
                _block_size,
                _vw,
            ]
            var total_vec_items = output.num_elements() // _vw
            ctx.enqueue_function[kernel_fn](
                output,
                inputs,
                Int32(inner_size),
                Int32(total_concat_dim),
                Int32(total_vec_items),
                grid_dim=(ceildiv(total_vec_items, _block_size),),
                block_dim=(_block_size,),
            )

        # Check if _vec_width-wide loads are safe:
        # - Non-innermost: inner dims must be aligned.
        # - Innermost: all input concat dims must be aligned.
        var can_vec = True
        comptime if axis != output.rank - 1:
            can_vec = inner_size % _vec_width == 0
        else:
            comptime for i in range(num_inputs):
                if Int(inputs[i].dim(axis)) % _vec_width != 0:
                    can_vec = False

        if can_vec:
            _launch_flat[_vec_width]()
            return

        # Fallbacks: try 4-wide for non-innermost, scalar for innermost.
        comptime if axis != output.rank - 1 and _vec_width > 4:
            if inner_size % 4 == 0:
                _launch_flat[4]()
                return
        comptime if axis == output.rank - 1:
            _launch_flat[1]()
            return

    # Fallback: elementwise approach (used when epilogue is present or inner
    # dimensions are not aligned for vectorization).
    #
    # Copy-capture `inputs` and `output` into the device-kernel closure. On
    # Metal a by-reference capture leaves the GPU kernel holding host-side
    # pointers, so it reads garbage/zeros. Copy-capture brings the device
    # pointers into the closure.
    @always_inline
    def per_output_elem[
        simd_width: Int, alignment: Int = 1
    ](out_index: Coord) {var}:
        var in_index = coord_to_index_list(out_index)

        comptime for i in range(num_inputs):
            var input = inputs[i]
            var input_shape = coord_to_index_list(input.layout.shape_coord())

            if Int(in_index[axis].value()) < input_shape[axis]:
                var in_coord = Coord(in_index)

                comptime if epilogue_fn:
                    comptime func = epilogue_fn.value()
                    func[dtype, out_index.rank, simd_width](
                        coord_to_index_list(out_index),
                        input.load[width=simd_width](in_coord),
                    )
                else:
                    output.store[width=simd_width](
                        out_index,
                        input.load[width=simd_width](in_coord),
                    )
                return

            in_index[axis] -= input_shape[axis]

    # When axis != rank-1, the SIMD group spans the innermost (non-concat)
    # dimension, so all elements belong to the same input and we can safely
    # use vectorized loads/stores (float4 = 128-bit transactions).
    comptime if axis != output.rank - 1:
        elementwise[4, target="gpu", _trace_description="concat"](
            per_output_elem, output.layout.shape_coord(), ctx
        )
    else:
        elementwise[1, target="gpu", _trace_description="concat"](
            per_output_elem, output.layout.shape_coord(), ctx
        )


@always_inline
def _concat_gpu[
    input_origin: ImmOrigin,
    InputLayoutType: TensorLayout,
    InputEngine: TensorEngine,
    //,
    dtype: DType,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, Engine=_, ...],
    axis: Int,
    inputs: StaticTuple[
        TileTensor[dtype, InputLayoutType, input_origin, Engine=InputEngine],
        ...,
    ],
    ctx: DeviceContext,
) raises:
    comptime num_inputs = inputs.size
    # Size of outer dims, if 1 we should memcpy to the output buffer.
    var outer_dims = 1
    for i in range(axis):
        # Use input[0], all dims should be equal except axis.
        outer_dims *= Int(inputs[0].dim(i))

    @__parameter
    @always_inline
    def _concat_buffers_contiguously() raises:
        var input_size = 0

        comptime for i in range(num_inputs):
            # Skip empty inputs.
            if inputs[i].num_elements() > 0:
                # TODO: Owning = True or False?
                var outp = DeviceBuffer(
                    ctx,
                    output.ptr.unsafe_offset(input_size),
                    inputs[i].num_elements(),
                    owning=False,
                )
                var inp = DeviceBuffer(
                    ctx,
                    inputs[i].ptr,
                    inputs[i].num_elements(),
                    owning=False,
                )
                ctx.enqueue_copy(
                    outp,
                    inp,
                )

                input_size += inputs[i].num_elements()

    # If outer_dims are ones and it is not a fused kernel, use device-to-device
    # copies.
    comptime if not epilogue_fn:
        if outer_dims == 1:
            return _concat_buffers_contiguously()

    if axis == output.rank - 1:
        var inner_most_unit_dim = True
        for i in range(num_inputs):
            if inputs[i].dim(axis) != 1:
                inner_most_unit_dim = False
                break

        if inner_most_unit_dim:
            comptime block_size = 32
            comptime kernel = _concat_inner_most_single_dim[
                OutputLayoutType=output.LayoutType,
                output_origin=output.origin,
                OutputEngine=output.Engine,
                InputLayoutType=InputLayoutType,
                input_origin=input_origin,
                InputEngine=InputEngine,
                dtype,
                num_inputs,
                block_size,
                epilogue_fn,
            ]

            return ctx.enqueue_function[kernel](
                output,
                inputs,
                grid_dim=(ceildiv(inputs[0].num_elements(), block_size),),
                block_dim=(block_size),
            )

    _concat_gpu_elementwise[epilogue_fn=epilogue_fn](output, axis, inputs, ctx)


@always_inline
def _fused_concat_cpu[
    rank: Int,
    dtype: DType,
    input_fn: def[input_index: Int, width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) capturing -> SIMD[dtype, width],
    output_0_fn: elementwise_epilogue_type,
    size: Int,
](
    axis: Int,
    input_shapes: StaticTuple[IndexList[rank], size],
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, Engine=_, ...],
    ctx: Optional[DeviceContext],
) raises:
    var offset = 0

    comptime for i in range(input_shapes.size):
        var input_shape = input_shapes[i]

        @always_inline
        def elementwise_wrapper[
            _width: Int, alignment: Int = 1
        ](indices: Coord) {var}:
            var c = rebind[IndexList[rank]](coord_to_index_list(indices))
            c[axis] += offset

            # Call the input/output lambda for fused concat kernel.
            output_0_fn[dtype, rank, width=_width, alignment=1](
                c,
                input_fn[i, _width, rank, alignment](
                    rebind[IndexList[rank]](coord_to_index_list(indices))
                ),
            )

        # TODO: we can use simd_width > 0 if all inputs are aligned.
        var device_ctx = ctx.value() if ctx else DeviceContext(api="cpu")
        elementwise[
            1,
            _trace_description="concat_fused",
        ](elementwise_wrapper, Coord(input_shape), device_ctx)
        offset = offset + input_shape[axis]


@always_inline
@__name(t"fused_concat_inner_most_single_dim_{dtype}")
def _fused_concat_inner_most_single_dim[
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    OutputEngine: TensorEngine,
    //,
    rank: Int,
    dtype: DType,
    block_size: Int,
    input_fn: def[input_index: Int, width: Int, _rank: Int, alignment: Int = 1](
        IndexList[_rank]
    ) capturing -> SIMD[dtype, width],
    output_0_fn: elementwise_epilogue_type,
    size: Int,
](
    input_shapes: StaticTuple[IndexList[rank], size],
    output: TileTensor[
        dtype, OutputLayoutType, output_origin, Engine=OutputEngine
    ],
):
    comptime num_inputs = input_shapes.size

    var idx = block_idx.x * block_size + thread_idx.x
    if idx >= product(input_shapes[0], rank):
        return

    # Static-divisor row -> n-D decomposition: the FusedConcatSlice
    # hot site. The graph compiler threads a `TileTensor` whose layout
    # `_shape_types` preserve the statically-known MLIR output dims, so the
    # per-thread `divmod` over the outer dims folds to magic-multiply + shift
    # (no `IDIV`). Dynamic dims fall back to the runtime divide; behavior is
    # bit-identical to `_get_start_indices_of_nth_subvolume[1]`.
    var index = _get_start_indices_of_nth_subvolume_static(
        idx, output.layout.shape_coord()
    )

    comptime for i in range(num_inputs):
        var out_index = index
        out_index[rank - 1] = i

        output_0_fn[dtype, rank, width=1](
            rebind[IndexList[rank]](out_index.canonicalize()),
            input_fn[i, 1, rank](rebind[IndexList[rank]](index.canonicalize())),
        )


@always_inline
@__name(t"fused_dual_concat_inner_most_single_dim_{dtype}")
def _fused_dual_concat_inner_most_single_dim[
    OutputLayoutType0: TensorLayout,
    output_origin_0: MutOrigin,
    OutputEngine0: TensorEngine,
    OutputLayoutType1: TensorLayout,
    output_origin_1: MutOrigin,
    OutputEngine1: TensorEngine,
    //,
    rank: Int,
    dtype: DType,
    block_size: Int,
    input_fn_0: def[
        input_index: Int, width: Int, _rank: Int, alignment: Int = 1
    ](IndexList[_rank]) capturing -> SIMD[dtype, width],
    output_0_fn: elementwise_epilogue_type,
    size_0: Int,
    input_fn_1: def[
        input_index: Int, width: Int, _rank: Int, alignment: Int = 1
    ](IndexList[_rank]) capturing -> SIMD[dtype, width],
    output_1_fn: elementwise_epilogue_type,
    size_1: Int,
](
    input_shapes_0: StaticTuple[IndexList[rank], size_0],
    output_0: TileTensor[
        dtype, OutputLayoutType0, output_origin_0, Engine=OutputEngine0
    ],
    input_shapes_1: StaticTuple[IndexList[rank], size_1],
    output_1: TileTensor[
        dtype, OutputLayoutType1, output_origin_1, Engine=OutputEngine1
    ],
):
    """Dual-concat kernel: two independent inner-most single-dim concats
    execute in the same kernel launch. Every thread processes both concats,
    so there is no intra-block branching beyond the standard bounds check.
    """
    var idx = block_idx.x * block_size + thread_idx.x

    if idx < product(input_shapes_0[0], rank):
        # Static-divisor row -> n-D decomposition; folds the per-
        # thread `divmod` over `output_0`'s statically-known outer dims.
        var index = _get_start_indices_of_nth_subvolume_static(
            idx, output_0.layout.shape_coord()
        )

        comptime for i in range(size_0):
            var out_index = index
            out_index[rank - 1] = i

            output_0_fn[dtype, rank, width=1](
                rebind[IndexList[rank]](out_index.canonicalize()),
                input_fn_0[i, 1, rank](
                    rebind[IndexList[rank]](index.canonicalize())
                ),
            )

    if idx < product(input_shapes_1[0], rank):
        # Static-divisor row -> n-D decomposition; folds the per-
        # thread `divmod` over `output_1`'s statically-known outer dims.
        var index = _get_start_indices_of_nth_subvolume_static(
            idx, output_1.layout.shape_coord()
        )

        comptime for i in range(size_1):
            var out_index = index
            out_index[rank - 1] = i

            output_1_fn[dtype, rank, width=1](
                rebind[IndexList[rank]](out_index.canonicalize()),
                input_fn_1[i, 1, rank](
                    rebind[IndexList[rank]](index.canonicalize())
                ),
            )


@always_inline
def _fused_dual_concat_gpu[
    rank: Int,
    dtype: DType,
    input_fn_0: def[
        input_index: Int, width: Int, _rank: Int, alignment: Int = 1
    ](IndexList[_rank]) capturing -> SIMD[dtype, width],
    output_0_fn: elementwise_epilogue_type,
    size_0: Int,
    input_fn_1: def[
        input_index: Int, width: Int, _rank: Int, alignment: Int = 1
    ](IndexList[_rank]) capturing -> SIMD[dtype, width],
    output_1_fn: elementwise_epilogue_type,
    size_1: Int,
    output_layout_0: TensorLayout,
    output_layout_1: TensorLayout,
](
    input_shapes_0: StaticTuple[IndexList[rank], size_0],
    output_0: TileTensor[mut=True, dtype, output_layout_0, _, Engine=_],
    input_shapes_1: StaticTuple[IndexList[rank], size_1],
    output_1: TileTensor[mut=True, dtype, output_layout_1, _, Engine=_],
    ctx: DeviceContext,
) raises:
    """Launch the dual-concat kernel for two inner-most single-dim concats.

    Both concats must satisfy the same preconditions as the single-concat
    variant: axis == rank-1, each input has size 1 in the concat dim, and
    all inputs within each group share the same shape.
    """
    comptime block_size = 64
    comptime kernel = _fused_dual_concat_inner_most_single_dim[
        OutputLayoutType0=output_0.LayoutType,
        output_origin_0=output_0.origin,
        OutputEngine0=output_0.Engine,
        OutputLayoutType1=output_1.LayoutType,
        output_origin_1=output_1.origin,
        OutputEngine1=output_1.Engine,
        rank,
        dtype,
        block_size,
        input_fn_0,
        output_0_fn,
        size_0,
        input_fn_1,
        output_1_fn,
        size_1,
    ]

    var num_elems_0 = product(input_shapes_0[0], input_shapes_0[0].size)
    var num_elems_1 = product(input_shapes_1[0], input_shapes_1[0].size)
    var max_elems = num_elems_0 if num_elems_0 > num_elems_1 else num_elems_1

    ctx.enqueue_function[kernel](
        input_shapes_0,
        output_0,
        input_shapes_1,
        output_1,
        grid_dim=(ceildiv(max_elems, block_size)),
        block_dim=block_size,
    )


@always_inline
def _fused_concat_gpu_elementwise[
    axis: Int,
    rank: Int,
    dtype: DType,
    input_fn: def[input_index: Int, width: Int, _rank: Int, alignment: Int = 1](
        IndexList[_rank]
    ) capturing -> SIMD[dtype, width],
    output_0_fn: elementwise_epilogue_type,
    size: Int,
](
    input_shapes: StaticTuple[IndexList[rank], size],
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, Engine=_, ...],
    ctx: DeviceContext,
) raises:
    comptime num_inputs = input_shapes.size

    @always_inline
    def per_output_elem[
        simd_width: Int, alignment: Int = 1
    ](out_index: Coord) {var}:
        var in_index = coord_to_index_list(out_index)

        comptime for i in range(num_inputs):
            var input_shape = input_shapes[i]

            if Int(in_index[axis].value()) < input_shape[axis]:
                output_0_fn[dtype, width=simd_width, alignment=alignment](
                    coord_to_index_list(out_index),
                    input_fn[i, simd_width, alignment=alignment](in_index),
                )
                return

            in_index[axis] -= input_shape[axis]

    # When axis != rank-1, the SIMD group spans the innermost (non-concat)
    # dimension, so we can use vectorized 32B loads/stores on sm_100a
    comptime if axis != rank - 1:
        comptime _vec_width = preferred_simd_width[dtype]()
        var inner_size = 1
        comptime for dim_idx in range(axis + 1, rank):
            inner_size *= Int(input_shapes[0][dim_idx])

        if _vec_width > 1 and inner_size % _vec_width == 0:
            elementwise[
                _vec_width,
                target="gpu",
                _trace_description="concat_fused",
            ](per_output_elem, output.layout.shape_coord(), ctx)
        elif inner_size % 4 == 0:
            elementwise[
                4,
                target="gpu",
                _trace_description="concat_fused",
            ](per_output_elem, output.layout.shape_coord(), ctx)
        else:
            elementwise[
                1,
                target="gpu",
                _trace_description="concat_fused",
            ](per_output_elem, output.layout.shape_coord(), ctx)
    else:
        comptime simd_width = preferred_simd_width[dtype]()

        # Check if all inputs are aligned to the target SIMD width.
        var use_simd_width = True
        comptime for i in range(num_inputs):
            if input_shapes[i][axis] % simd_width != 0:
                use_simd_width = False

        if use_simd_width:
            elementwise[
                simd_width,
                target="gpu",
                _trace_description="concat_fused",
            ](per_output_elem, output.layout.shape_coord(), ctx)
        else:
            elementwise[
                1,
                target="gpu",
                _trace_description="concat_fused",
            ](per_output_elem, output.layout.shape_coord(), ctx)


@always_inline
def _fused_dual_concat_gpu_elementwise[
    axis: Int,
    rank: Int,
    dtype: DType,
    input_fn_0: def[
        input_index: Int, width: Int, _rank: Int, alignment: Int = 1
    ](IndexList[_rank]) capturing -> SIMD[dtype, width],
    output_0_fn: elementwise_epilogue_type,
    size_0: Int,
    input_fn_1: def[
        input_index: Int, width: Int, _rank: Int, alignment: Int = 1
    ](IndexList[_rank]) capturing -> SIMD[dtype, width],
    output_1_fn: elementwise_epilogue_type,
    size_1: Int,
](
    input_shapes_0: StaticTuple[IndexList[rank], size_0],
    output_0: TileTensor[
        mut=True, dtype, address_space=.GENERIC, Engine=_, ...
    ],
    input_shapes_1: StaticTuple[IndexList[rank], size_1],
    output_1: TileTensor[
        mut=True, dtype, address_space=.GENERIC, Engine=_, ...
    ],
    ctx: DeviceContext,
) raises:
    """Fuses two independent concat operations into a single GPU kernel launch
    via `dual_elementwise`. Each concat gets its own closure; the dual
    elementwise infrastructure handles iteration, SIMD width, and grid sizing.
    """

    @__parameter
    @always_inline
    def per_output_elem_0[
        simd_width: Int, alignment: Int = 1
    ](out_index: Coord):
        var in_index = coord_to_index_list(out_index)
        var out_idx = in_index
        comptime for i in range(size_0):
            var input_shape = input_shapes_0[i]
            if Int(in_index[axis].value()) < Int(input_shape[axis].value()):
                output_0_fn[
                    dtype, out_index.rank, width=simd_width, alignment=alignment
                ](
                    out_idx,
                    input_fn_0[
                        i, simd_width, out_index.rank, alignment=alignment
                    ](in_index),
                )
                return
            in_index[axis] -= input_shape[axis]

    @__parameter
    @always_inline
    def per_output_elem_1[
        simd_width: Int, alignment: Int = 1
    ](out_index: Coord):
        var in_index = coord_to_index_list(out_index)
        var out_idx = in_index
        comptime for i in range(size_1):
            var input_shape = input_shapes_1[i]
            if in_index[axis] < input_shape[axis]:
                output_1_fn[
                    dtype, out_index.rank, width=simd_width, alignment=alignment
                ](
                    out_idx,
                    input_fn_1[
                        i, simd_width, out_index.rank, alignment=alignment
                    ](in_index),
                )
                return
            in_index[axis] -= input_shape[axis]

    # Build IndexList[rank] explicitly so both shapes share the same type.
    var _s0 = coord_to_index_list(output_0.layout.shape_coord())
    var _s1 = coord_to_index_list(output_1.layout.shape_coord())
    var output_shape_0 = IndexList[rank]()
    var output_shape_1 = IndexList[rank]()
    comptime for d in range(rank):
        output_shape_0[d] = Int(_s0[d])
        output_shape_1[d] = Int(_s1[d])

    comptime if axis != rank - 1:
        comptime _vec_width = preferred_simd_width[dtype]()
        var inner_size = 1
        comptime for dim_idx in range(axis + 1, rank):
            inner_size *= Int(input_shapes_0[0][dim_idx])

        if _vec_width > 1 and inner_size % _vec_width == 0:
            dual_elementwise[
                per_output_elem_0,
                per_output_elem_1,
                _vec_width,
                target="gpu",
                _trace_description="dual_concat_fused",
            ](Coord(output_shape_0), Coord(output_shape_1), ctx)
        elif inner_size % 4 == 0:
            dual_elementwise[
                per_output_elem_0,
                per_output_elem_1,
                4,
                target="gpu",
                _trace_description="dual_concat_fused",
            ](Coord(output_shape_0), Coord(output_shape_1), ctx)
        else:
            dual_elementwise[
                per_output_elem_0,
                per_output_elem_1,
                1,
                target="gpu",
                _trace_description="dual_concat_fused",
            ](Coord(output_shape_0), Coord(output_shape_1), ctx)
    else:
        comptime simd_width = preferred_simd_width[dtype]()

        # All inputs from both sets must be aligned for vectorized access.
        var use_simd_width = True
        comptime for i in range(size_0):
            if input_shapes_0[i][axis] % simd_width != 0:
                use_simd_width = False
        comptime for i in range(size_1):
            if input_shapes_1[i][axis] % simd_width != 0:
                use_simd_width = False

        if use_simd_width:
            dual_elementwise[
                per_output_elem_0,
                per_output_elem_1,
                simd_width,
                target="gpu",
                _trace_description="dual_concat_fused",
            ](Coord(output_shape_0), Coord(output_shape_1), ctx)
        else:
            dual_elementwise[
                per_output_elem_0,
                per_output_elem_1,
                1,
                target="gpu",
                _trace_description="dual_concat_fused",
            ](Coord(output_shape_0), Coord(output_shape_1), ctx)


@always_inline
def _fused_concat_gpu[
    rank: Int,
    dtype: DType,
    input_fn: def[input_index: Int, width: Int, _rank: Int, alignment: Int = 1](
        IndexList[_rank]
    ) capturing -> SIMD[dtype, width],
    output_0_fn: elementwise_epilogue_type,
    size: Int,
    output_layout: TensorLayout,
](
    axis: Int,
    input_shapes: StaticTuple[IndexList[rank], size],
    output: TileTensor[mut=True, dtype, output_layout, _, Engine=_],
    ctx: DeviceContext,
) raises:
    comptime num_inputs = input_shapes.size

    if axis == rank - 1:
        var inner_most_unit_dim = True
        for i in range(num_inputs):
            if (
                input_shapes[i][axis] != 1
                or not input_shapes[i] == input_shapes[0]
            ):
                inner_most_unit_dim = False
                break

        if inner_most_unit_dim:
            comptime block_size = 32
            comptime kernel = _fused_concat_inner_most_single_dim[
                OutputLayoutType=output.LayoutType,
                output_origin=output.origin,
                OutputEngine=output.Engine,
                rank,
                dtype,
                block_size,
                input_fn,
                output_0_fn,
                size,
            ]

            return ctx.enqueue_function[kernel](
                input_shapes,
                output,
                grid_dim=(
                    ceildiv(
                        product(input_shapes[0], input_shapes[0].size),
                        block_size,
                    )
                ),
                block_dim=(block_size),
            )

    # Without parameter dispatch there are 2 extra stack allocations in the GPU kernel
    comptime for i in range(rank):
        if i == axis:
            return _fused_concat_gpu_elementwise[
                i,
                rank,
                dtype,
                input_fn,
                output_0_fn,
                size,
            ](input_shapes, output, ctx)


@always_inline
def _fused_dual_concat_gpu[
    rank: Int,
    dtype: DType,
    input_fn_0: def[
        input_index: Int, width: Int, _rank: Int, alignment: Int = 1
    ](IndexList[_rank]) capturing -> SIMD[dtype, width],
    output_0_fn: elementwise_epilogue_type,
    size_0: Int,
    input_fn_1: def[
        input_index: Int, width: Int, _rank: Int, alignment: Int = 1
    ](IndexList[_rank]) capturing -> SIMD[dtype, width],
    output_1_fn: elementwise_epilogue_type,
    size_1: Int,
    output_layout_0: TensorLayout,
    output_layout_1: TensorLayout,
](
    axis: Int,
    input_shapes_0: StaticTuple[IndexList[rank], size_0],
    output_0: TileTensor[mut=True, dtype, output_layout_0, _, Engine=_],
    input_shapes_1: StaticTuple[IndexList[rank], size_1],
    output_1: TileTensor[mut=True, dtype, output_layout_1, _, Engine=_],
    ctx: DeviceContext,
) raises:
    if axis == rank - 1:
        var inner_most_unit_dim = True
        for i in range(size_0):
            if (
                input_shapes_0[i][axis] != 1
                or not input_shapes_0[i] == input_shapes_0[0]
            ):
                inner_most_unit_dim = False
                break
        if inner_most_unit_dim:
            for i in range(size_1):
                if (
                    input_shapes_1[i][axis] != 1
                    or not input_shapes_1[i] == input_shapes_1[0]
                ):
                    inner_most_unit_dim = False
                    break

        if inner_most_unit_dim:
            return _fused_dual_concat_gpu[
                rank,
                dtype,
                input_fn_0,
                output_0_fn,
                size_0,
                input_fn_1,
                output_1_fn,
                size_1,
                output_layout_0,
                output_layout_1,
            ](
                input_shapes_0,
                output_0,
                input_shapes_1,
                output_1,
                ctx,
            )

    comptime for i in range(rank):
        if i == axis:
            return _fused_dual_concat_gpu_elementwise[
                i,
                rank,
                dtype,
                input_fn_0,
                output_0_fn,
                size_0,
                input_fn_1,
                output_1_fn,
                size_1,
            ](
                input_shapes_0,
                output_0,
                input_shapes_1,
                output_1,
                ctx,
            )


@always_inline
def fused_concat[
    dtype: DType,
    rank: Int,
    input_fn: def[input_index: Int, width: Int, _rank: Int, alignment: Int = 1](
        IndexList[_rank]
    ) capturing -> SIMD[dtype, width],
    output_0_fn: elementwise_epilogue_type,
    output_layout: TensorLayout,
    *,
    axis: Int,
    target: StaticString = "cpu",
](
    input_shapes: StaticTuple[IndexList[rank], _],
    output: TileTensor[mut=True, dtype, output_layout, _, Engine=_],
    ctx: DeviceContext,
) raises:
    """Concatenates inputs produced by ``input_fn`` along ``axis`` into ``output``, applying ``output_0_fn`` to each element.

    Instead of reading from concrete input tensors, the fused variant drives the
    concat from a caller-supplied ``input_fn`` that produces values on demand,
    allowing the concat to be fused with preceding elementwise operations.
    Dispatches to the CPU or GPU fused implementation based on ``target`` and
    returns early when the output tensor is empty.

    Parameters:
        dtype: Element type of the input and output tensors.
        rank: Number of dimensions in the input and output tensors.
        input_fn: Function that produces input element values on demand,
            indexed by input position.
        output_0_fn: Epilogue function applied to each produced element before
            storing.
        output_layout: Layout type of the output tensor.
        axis: Axis along which to concatenate the inputs.
        target: Target device to dispatch to (defaults to ``"cpu"``).

    Args:
        input_shapes: Static tuple of per-input shapes describing each logical
            input that ``input_fn`` produces.
        output: Destination tensor that receives the concatenated result.
        ctx: Device context used to schedule the work.
    """
    comptime assert is_valid_target[target](), "not a valid target"

    with Trace[TraceLevel.OP, target=target](
        "concat", task_id=get_safe_task_id(ctx)
    ):
        # Exit early if the tensors are empty.
        if output.num_elements() == 0:
            return
        comptime if is_cpu[target]():
            return _fused_concat_cpu[
                rank,
                dtype,
                input_fn,
                output_0_fn,
            ](axis, input_shapes, output, Optional[DeviceContext](ctx))
        else:
            return _fused_concat_gpu[rank, dtype, input_fn, output_0_fn](
                axis,
                input_shapes,
                output.as_unsafe_any_origin(),
                ctx,
            )
