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

from std.sys.info import _current_target, simd_width_of
from std.math.uutils import ufloordiv

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.info import is_cpu
from layout import LayoutTensor, TileTensor
from layout import Coord, Idx, coord_to_index_list

from std.utils import IndexList


@always_inline
def get_batch_from_row_offsets(
    row_offsets: LayoutTensor[mut=False, .uint32, ...], tok_idx: Int
) -> Int:
    """Calculate the batch_idx for the given flattened token_idx using row_offsets.
    """
    var row_offsets_size = row_offsets.size()

    assert tok_idx >= 0 and tok_idx < Int(
        row_offsets[row_offsets_size - 1]
    ), "tok_idx is out of range of row_offsets"

    var low = 0
    var high: Int = row_offsets_size - 1
    while low + 1 != high:
        var mid = ufloordiv(low + high, 2)

        if tok_idx >= Int(row_offsets[mid]):
            low = mid
        else:
            high = mid

    return low


@always_inline
def get_batch_from_row_offsets(
    row_offsets: TileTensor[mut=False, .uint32, ...], tok_idx: Int
) -> Int:
    """Calculate the batch_idx for the given flattened token_idx using row_offsets.
    """
    comptime assert row_offsets.flat_rank == 1

    var row_offsets_size = row_offsets.num_elements()

    assert tok_idx >= 0 and tok_idx < Int(
        row_offsets[row_offsets_size - 1]
    ), "tok_idx is out of range of row_offsets"

    var low = 0
    var high: Int = row_offsets_size - 1
    while low + 1 != high:
        var mid = ufloordiv(low + high, 2)

        if tok_idx >= Int(row_offsets[mid]):
            low = mid
        else:
            high = mid

    return low


@always_inline
def get_batch_and_token_idx_from_row_offsets(
    row_offsets: TileTensor[mut=False, .uint32, ...], tok_idx: Int
) -> Tuple[Int, Int]:
    """Calculate the batch_idx for the given flattened token_idx using row_offsets.
    """
    comptime assert row_offsets.flat_rank == 1

    var batch_idx = get_batch_from_row_offsets(row_offsets, tok_idx)
    return batch_idx, Int(tok_idx - Int(row_offsets[batch_idx]))


def merge_ragged_tensors[
    rank: Int,
    dtype: DType,
    //,
    target: StaticString = "cpu",
](
    c: TileTensor[mut=True, dtype, ...],
    c_row_offsets: TileTensor[mut=True, .uint32, ...],
    a: TileTensor[mut=False, dtype, ...],
    a_row_offsets: TileTensor[mut=False, .uint32, ...],
    b: TileTensor[mut=False, dtype, ...],
    b_row_offsets: TileTensor[mut=False, .uint32, ...],
    ctx: DeviceContext,
) raises:
    comptime assert c.flat_rank == rank, "c.flat_rank must equal rank"
    comptime assert a.flat_rank == rank, "a.flat_rank must equal rank"
    comptime assert b.flat_rank == rank, "b.flat_rank must equal rank"
    comptime assert (
        c_row_offsets.flat_rank == 1
    ), "c_row_offsets.flat_rank must be 1"
    comptime assert (
        a_row_offsets.flat_rank == 1
    ), "a_row_offsets.flat_rank must be 1"
    comptime assert (
        b_row_offsets.flat_rank == 1
    ), "b_row_offsets.flat_rank must be 1"

    @always_inline
    def merge_fn[width: Int, alignment: Int = 1](idx: Coord) {var}:
        comptime assert idx.rank == rank, "Invalid rank passed to the kernel"

        var a_tensor_size = Int(a.dim[0]())
        var is_tensor_a = Int(idx[0].value()) < a_tensor_size

        var batch_id: Int
        var src_idx: IndexList[rank] = rebind[IndexList[rank]](
            coord_to_index_list(idx)
        )
        if is_tensor_a:
            batch_id = get_batch_from_row_offsets(a_row_offsets, src_idx[0])
        else:
            src_idx[0] = Int(idx[0].value()) - a_tensor_size
            batch_id = get_batch_from_row_offsets(b_row_offsets, src_idx[0])

        var dst_idx: IndexList[rank] = rebind[IndexList[rank]](
            coord_to_index_list(idx)
        )
        var dst_row_idx: Int = src_idx[0]

        if is_tensor_a:
            dst_row_idx += Int(b_row_offsets[batch_id])
        else:
            dst_row_idx += Int(a_row_offsets[batch_id + 1])

        dst_idx[0] = dst_row_idx

        # Compute flat offsets for pointer load/store (Horner form).
        # Inner dimensions are the same across a, b, and c.
        @always_inline
        @__parameter
        def _flat_offset[r: Int](index: IndexList[r]) -> Int:
            comptime assert r == rank
            var flat = index[0]
            comptime for i in range(1, rank):
                flat = flat * Int(c.dim[i]()) + index[i]
            return flat

        # The elementwise function takes care of handling the scenario where
        # tensors' last dimension is not multiple of simdwidth. It will call
        # this `merge_fn`function with width = 1 for the last few elements.
        var val: SIMD[dtype, width]
        if is_tensor_a:
            val = a.load[width=width](Coord(src_idx))
        else:
            val = b.load[width=width](Coord(src_idx))

        c.store[width=width](Coord(dst_idx), val)

        # Update the row offsets if this is the first element of the batch
        var is_first_element = is_tensor_a and src_idx[0] == Int(
            a_row_offsets[batch_id]
        )

        comptime for i in range(1, rank):
            if idx[i].value() != 0:
                is_first_element = False

        if is_first_element:
            c_row_offsets[batch_id] = UInt32(dst_row_idx)

            # If this is the last batch, also update the last row offset to the total size
            if batch_id == Int(c_row_offsets.dim[0]()) - 2:
                var total_size = Int(a.dim[0]()) + Int(b.dim[0]())
                c_row_offsets[batch_id + 1] = UInt32(total_size)

    comptime compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    comptime target_simd_width = simd_width_of[dtype, target=compile_target]()
    comptime kernel_simd_width = 1 if rank == 1 else target_simd_width

    elementwise[
        simd_width=kernel_simd_width,
        target=target,
        _trace_description="merge_ragged_tensors",
    ](merge_fn, c.layout.shape_coord(), ctx)


def eagle_prefill_shift_tokens[
    dtype: DType,
    //,
    target: StaticString = "cpu",
](
    output: TileTensor[mut=True, dtype, ...],
    tokens: TileTensor[mut=False, dtype, ...],
    offsets: TileTensor[mut=False, .uint32, ...],
    shift_next_tokens: TileTensor[mut=False, dtype, ...],
    ctx: DeviceContext,
) raises:
    """Shift ragged tokens left by 1 per request, appending bonus tokens."""
    comptime assert output.flat_rank == 1
    comptime assert tokens.flat_rank == 1
    comptime assert offsets.flat_rank == 1
    comptime assert shift_next_tokens.flat_rank == 1

    @always_inline
    def shift_fn[width: Int, alignment: Int = 1](idx: Coord) {var}:
        comptime assert idx.rank == 1

        var i = Int(idx[0].value())

        # Shift left by 1 per batch, append bonus token
        var batch_id = get_batch_from_row_offsets(offsets, i)
        var end = Int(offsets[batch_id + 1])

        if i < end - 1:
            # Not the last position: copy from next position
            output.store(Coord(i), tokens.load[width=1](Coord(i + 1)))
        else:
            # Last position in batch: append shift_next_tokens
            output.store(
                Coord(i),
                shift_next_tokens.load[width=1](Coord(batch_id)),
            )

    var shape = Coord(Int(output.dim[0]()))

    elementwise[
        simd_width=1,
        target=target,
        _trace_description="eagle_prefill_shift_tokens",
    ](shift_fn, shape, ctx)
