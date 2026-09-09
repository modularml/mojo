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

"""Implements `unsafe_parallel_memcpy`.

You can import these APIs from the `algorithm` package. For example:

```mojo
from max.algorithm import unsafe_parallel_memcpy
```
"""

from . import sync_parallelize
from std.math import ceildiv

from std.memory import unsafe_memcpy
from std.runtime import parallelism_level


def unsafe_parallel_memcpy[
    dtype: DType
](
    *,
    dest: Pointer[mut=True, Scalar[dtype], _],
    src: Pointer[Scalar[dtype], _],
    count: Int,
    count_per_task: Int,
    num_tasks: Int,
):
    """Copies `count` elements from a memory buffer `src` to `dest` in parallel
    by spawning `num_tasks` tasks each copying `count_per_task` elements.

    Parameters:
        dtype: The element dtype.

    Args:
        dest: The destination buffer.
        src: The source buffer.
        count: Number of elements in the buffer.
        count_per_task: Task size.
        num_tasks: Number of tasks to run in parallel.
    """
    if count == 0:
        return

    @always_inline
    def _parallel_copy(thread_id: Int) {imm}:
        var begin = count_per_task * thread_id
        var end = min(
            count_per_task * (thread_id + 1),
            count,
        )
        if begin >= count:
            return
        var to_copy = end - begin
        if to_copy <= 0:
            return

        unsafe_memcpy(
            dest=dest.unsafe_offset(begin),
            src=src.unsafe_offset(begin),
            count=to_copy,
        )

    sync_parallelize(_parallel_copy, num_tasks)


def unsafe_parallel_memcpy[
    dtype: DType,
](
    *,
    dest: Pointer[mut=True, Scalar[dtype], _],
    src: Pointer[Scalar[dtype], _],
    count: Int,
):
    """Copies `count` elements from a memory buffer `src` to `dest` in parallel.

    Parameters:
        dtype: The element dtype.

    Args:
        dest: The destination pointer.
        src: The source pointer.
        count: The number of elements to copy.
    """

    # TODO: Find a heuristic to replace the magic number.
    comptime min_work_per_task = 1024
    comptime min_work_for_parallel = 4 * min_work_per_task

    # If number of elements to be copied is less than minimum preset (4048),
    # then use default unsafe_memcpy.
    if count < min_work_for_parallel:
        unsafe_memcpy(dest=dest, src=src, count=count)
    else:
        var work_units = ceildiv(count, min_work_per_task)
        var num_tasks = min(work_units, parallelism_level())
        var work_block_size = ceildiv(work_units, num_tasks)

        unsafe_parallel_memcpy(
            dest=dest,
            src=src,
            count=count,
            count_per_task=work_block_size * min_work_per_task,
            num_tasks=num_tasks,
        )


@deprecated(use=unsafe_parallel_memcpy)
def parallel_memcpy[
    dtype: DType
](
    *,
    dest: OptionalPointer[mut=True, Scalar[dtype], _],
    src: OptionalPointer[Scalar[dtype], _],
    count: Int,
    count_per_task: Int,
    num_tasks: Int,
):
    """Copies `count` elements from a memory buffer `src` to `dest` in parallel
    by spawning `num_tasks` tasks each copying `count_per_task` elements.

    Parameters:
        dtype: The element dtype.

    Args:
        dest: The destination buffer.
        src: The source buffer.
        count: Number of elements in the buffer.
        count_per_task: Task size.
        num_tasks: Number of tasks to run in parallel.

    Safety:
        `dest` or `src` can only be `None` when `count == 0`.
    """
    unsafe_parallel_memcpy(
        dest=dest.unsafe_value(),
        src=src.unsafe_value(),
        count=count,
        count_per_task=count_per_task,
        num_tasks=num_tasks,
    )


@deprecated(use=unsafe_parallel_memcpy)
def parallel_memcpy[
    dtype: DType,
](
    *,
    dest: OptionalPointer[mut=True, Scalar[dtype], _],
    src: OptionalPointer[Scalar[dtype], _],
    count: Int,
):
    """Copies `count` elements from a memory buffer `src` to `dest` in parallel.

    Parameters:
        dtype: The element dtype.

    Args:
        dest: The destination pointer.
        src: The source pointer.
        count: The number of elements to copy.

    Safety:
        `dest` or `src` can only be `None` when `count == 0`.
    """
    if count == 0:
        return

    unsafe_parallel_memcpy(
        dest=dest.unsafe_value(), src=src.unsafe_value(), count=count
    )
