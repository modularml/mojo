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
"""CPU implementations of elementwise functions."""

from std.math import ceildiv

from std.utils.coord import Coord, coord_to_index_list
from std.utils.index import IndexList

from max.gpu.host import DeviceContext
from std.algorithm.backend.vectorize import vectorize

from .parallelize import _get_num_workers, sync_parallelize
from max.algorithm.functional import _get_start_indices_of_nth_subvolume


# ===-----------------------------------------------------------------------===#
# Elementwise CPU implementations
# ===-----------------------------------------------------------------------===#


@always_inline
def _elementwise_impl_cpu[
    simd_width: Int,
    FuncType: def[width: Int, alignment: Int = 1](Coord) -> None,
    *,
    trace_description: StaticString,
](func: FuncType, *, shape: Coord, ctx: Optional[DeviceContext] = None,):
    """Dispatches elementwise execution on CPU to the 1D or ND implementation
    based on the rank of the input shape.

    Parameters:
        simd_width: The SIMD vector width to use.
        FuncType: The body function type.
        trace_description: Description of the trace.

    Args:
        func: The closure carrying the captured state of the body function.
        shape: The shape of the buffer.
        ctx: Optional CPU DeviceContext to execute the tasks on.
    """

    comptime impl = _elementwise_impl_cpu_1d if shape.rank == 1 else _elementwise_impl_cpu_nd
    impl[simd_width](func, shape, ctx)


@always_inline
def _elementwise_impl_cpu_1d[
    simd_width: Int,
    FuncType: def[width: Int, alignment: Int = 1](Coord) -> None,
](func: FuncType, shape: Coord, ctx: Optional[DeviceContext] = None,):
    """Executes `func[width, rank](indices)`, possibly using sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns when
    all sub-tasks have completed.

    Parameters:
        simd_width: The SIMD vector width to use.
        FuncType: The body function type.

    Args:
        func: The closure carrying the captured state of the body function.
        shape: The shape of the buffer.
        ctx: Optional CPU DeviceContext to execute the tasks on.
    """
    comptime assert shape.rank == 1, "Specialization for 1D"

    comptime unroll_factor = 8  # TODO: Comeup with a cost heuristic.

    var problem_size = shape.product()

    var num_workers = _get_num_workers(problem_size, ctx=ctx)
    var chunk_size = ceildiv(problem_size, num_workers)

    @always_inline
    def task_func(i: Int) {imm}:
        var start_offset = i * chunk_size
        var end_offset = min((i + 1) * chunk_size, problem_size)
        var len = end_offset - start_offset

        @always_inline
        def func_wrapper[
            simd_width: Int
        ](idx: Int) {imm start_offset, imm func,}:
            var offset = start_offset + idx
            func[simd_width](Coord(offset))

        vectorize[simd_width, unroll_factor=unroll_factor](len, func_wrapper)

    sync_parallelize(task_func, num_workers, ctx)


@always_inline
def _elementwise_impl_cpu_nd[
    simd_width: Int,
    FuncType: def[width: Int, alignment: Int = 1](Coord) -> None,
](func: FuncType, shape: Coord, ctx: Optional[DeviceContext] = None,):
    """Executes `func[width, rank](indices)`, possibly using sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns
    when all sub-tasks have completed.

    Parameters:
        simd_width: The SIMD vector width to use.
        FuncType: The body function type.

    Args:
        func: The closure carrying the captured state of the body function.
        shape: The shape of the buffer.
        ctx: Optional CPU DeviceContext to execute the tasks on.
    """
    comptime assert shape.rank > 1, "Specialization for ND where N > 1"
    comptime rank = shape.rank

    # If we know we won't do any work, return early
    if shape[rank - 1].value() == 0:
        return

    comptime unroll_factor = 8  # TODO: Comeup with a cost heuristic.

    # Strategy: we parallelize over all dimensions except the innermost and
    # vectorize over the innermost dimension. We unroll the innermost dimension
    # by a factor of unroll_factor.

    # Compute the number of workers to allocate based on ALL work, not just
    # the dimensions we split across.
    var total_size = shape.product()

    var num_workers = _get_num_workers(total_size, ctx=ctx)
    var parallelism_size = total_size // SIMDLength(shape[rank - 1].value())
    var chunk_size = ceildiv(parallelism_size, num_workers)

    @always_inline
    def task_func(i: Int) {imm}:
        var start_parallel_offset = i * chunk_size
        var end_parallel_offset = min((i + 1) * chunk_size, parallelism_size)

        var len = end_parallel_offset - start_parallel_offset
        if len <= 0:
            return

        var indices = IndexList[rank]()

        @always_inline
        def func_wrapper_nd[
            simd_width: Int
        ](idx: Int) {mut indices, imm func, imm}:
            indices[rank - 1] = idx
            func[simd_width](Coord(indices.canonicalize()))

        for parallel_offset in range(
            start_parallel_offset, end_parallel_offset
        ):
            indices = _get_start_indices_of_nth_subvolume(
                parallel_offset, coord_to_index_list(shape)
            )

            # We vectorize over the innermost dimension.
            vectorize[simd_width, unroll_factor=unroll_factor](
                SIMDLength(shape[rank - 1].value()), func_wrapper_nd
            )

    sync_parallelize(task_func, num_workers, ctx)
