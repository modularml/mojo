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
"""CPU implementations of parallelization functions."""

from std.math import ceildiv
from std.os import abort
from std.utils.index import IndexList

from max.runtime.asyncrt import parallelism_level
from max.runtime import tracing
from max.runtime.tracing import Trace, TraceLevel

from std.utils.numerics import FlushDenormals

from max.gpu.host import DeviceContext

# ===-----------------------------------------------------------------------===#
# Parallelize
# ===-----------------------------------------------------------------------===#


@always_inline
def sync_parallelize[
    FuncType: def(Int) raises -> None,
](func: FuncType, num_work_items: Int, ctx: Optional[DeviceContext] = None):
    """Executes func(0) ... func(num_work_items-1) as parallel sub-tasks,
    and returns when all are complete.

    Non-raising closures still bind this signature. Exceptions raised by
    `func` abort rather than propagating to the caller.

    Parameters:
        FuncType: The body function type.

    Args:
        func: The closure carrying the captured state of the body function.
        num_work_items: Number of parallel tasks.
        ctx: The CPU DeviceContext to enqueue the work on.
    """
    # We have no tasks, so do nothing.
    if num_work_items <= 0:
        # No-op
        return

    # If profiling is enabled, and the caller's thread has an active profile
    # entry, each sub-task will also be profiled with a reference back to the
    # parent. Otherwise parent_id will be zero.
    var parent_id = tracing.get_current_trace_id[TraceLevel.THREAD]()

    @always_inline
    def func_wrapped(i: Int) {imm}:
        with FlushDenormals():
            try:
                with Trace[TraceLevel.THREAD, target=StaticString("cpu")](
                    "task", task_id=i, parent_id=parent_id
                ):
                    func(i)
            except e:
                abort(String(e))

    if num_work_items == 1:
        # Just run inline.
        func_wrapped(0)
        return

    try:
        var cpu_ctx = ctx.or_else(DeviceContext(api="cpu"))
        cpu_ctx.enqueue_cpu_range(func_wrapped, count=num_work_items)
        cpu_ctx.synchronize()
    except e:
        abort(String(e))


@always_inline
def parallelize[
    FuncType: def(Int) -> None,
](func: FuncType, num_work_items: Int, ctx: Optional[DeviceContext] = None):
    """Executes func(0) ... func(num_work_items-1) as sub-tasks in parallel, and
    returns when all are complete.

    Parameters:
        FuncType: The body function type.

    Args:
        func: The closure carrying the captured state of the body function.
        num_work_items: Number of parallel tasks.
        ctx: Optional CPU DeviceContext to execute the work on.
    """
    _parallelize_impl(func, num_work_items, parallelism_level(ctx), ctx)


@always_inline
def parallelize[
    FuncType: def(Int) -> None,
](
    func: FuncType,
    num_work_items: Int,
    num_workers: Int,
    ctx: Optional[DeviceContext] = None,
):
    """Executes func(0) ... func(num_work_items-1) as sub-tasks in parallel, and
    returns when all are complete.

    Parameters:
        FuncType: The body function type.

    Args:
        func: The closure carrying the captured state of the body function.
        num_work_items: Number of parallel tasks.
        num_workers: The number of workers to use for execution.
        ctx: Optional CPU DeviceContext to execute the work on.
    """
    _parallelize_impl(func, num_work_items, num_workers, ctx)


@always_inline
def _parallelize_impl[
    FuncType: def(Int) -> None,
](
    func: FuncType,
    num_work_items: Int,
    num_workers: Int,
    ctx: Optional[DeviceContext] = None,
):
    """Distributes work items across workers by coalescing consecutive items
    into chunks and executing them in parallel via `sync_parallelize`.

    Parameters:
        FuncType: The body function type.

    Args:
        func: The closure carrying the captured state of the body function.
        num_work_items: Number of parallel tasks.
        num_workers: The number of workers to use for execution.
        ctx: Optional CPU DeviceContext to execute the work on.
    """
    assert num_workers > 0, "Number of workers must be positive"
    # Calculate how many items are picked up by each worker.
    var chunk_size, extra_items = divmod(num_work_items, num_workers)

    # We coalesce consecutive groups of work items into a single dispatch by
    # using the coarse_grained_func below.
    @always_inline
    def coarse_grained_func(
        thread_idx: Int,
    ) {imm func, imm chunk_size, imm extra_items,}:
        # Calculate the consecutive range of work items this invocation is
        # responsible for.
        var start_idx = thread_idx * chunk_size + min(thread_idx, extra_items)
        for i in range(chunk_size + Int(thread_idx < extra_items)):
            func(start_idx + i)

    sync_parallelize(coarse_grained_func, num_workers, ctx)


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


@always_inline
def _get_num_workers(
    problem_size: Int,
    grain_size: Int = 32768,
    ctx: Optional[DeviceContext] = None,
) -> Int:
    """Returns a number of workers to run in parallel for given problem_size,
    accounting for the available worker threads of the current runtime.

    Args:
        problem_size: The number of parallel tasks.
        grain_size: Minimum number of elements to warrant an additional thread.
        ctx: The context to execute the work on.

    Returns:
        The number of workers to run in parallel.
    """
    # default grain_size copied from https://github.com/pytorch/pytorch/blob/20dfce591ce88bc957ffcd0c8dc7d5f7611a4a3b/aten/src/ATen/TensorIterator.h#L86
    # Ensure at least one worker is always returned to avoid division by zero.
    return max(
        1, min(parallelism_level(ctx), ceildiv(problem_size, grain_size))
    )


# ===-----------------------------------------------------------------------===#
# parallelize_over_rows
# ===-----------------------------------------------------------------------===#


def parallelize_over_rows[
    FuncType: def(Int, Int) -> None,
](
    func: FuncType,
    shape: IndexList,
    axis: Int,
    grain_size: Int,
    ctx: Optional[DeviceContext] = None,
):
    """Parallelize func over non-axis dims of shape.

    Parameters:
        FuncType: The body function type.

    Args:
        func: The closure carrying the captured state of the body function.
        shape: Shape to parallelize over.
        axis: Rows are slices along the axis dimension of shape.
        grain_size: The minimum number of elements to warrant using an additional thread.
        ctx: Optional CPU DeviceContext to execute the work on.
    """
    # If we know we will have no work, return early
    if shape[axis] == 0:
        return

    var total_size = shape.flattened_length()
    var num_rows = total_size // shape[axis]

    var num_workers = min(
        num_rows,
        _get_num_workers(total_size, grain_size, ctx),
    )
    var chunk_size = ceildiv(num_rows, num_workers)

    @always_inline
    def task_func(
        task_id: Int,
    ) {imm func, imm chunk_size, imm num_rows,}:
        var start_row = task_id * chunk_size
        var end_row = min((task_id + 1) * chunk_size, num_rows)

        func(start_row, end_row)

    sync_parallelize(task_func, num_workers, ctx)
