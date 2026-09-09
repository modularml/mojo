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

from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.python._cpython import GILAcquired, GILReleased
from std.os import abort
import std.math
from max.algorithm.functional import parallelize
from std.sys.info import num_physical_cores


@export
def PyInit_mojo_module() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojo_module")
        m.def_function[plus_one]("plus_one")
        m.def_function[parallel_wrapper](
            "parallel_wrapper", docstring="Parallelizing function"
        )
        return m.finalize()
    except e:
        abort(t"failed to create Python module: {e}")


def plus_one(arg: PythonObject) raises -> PythonObject:
    return arg + 1


def parallel_wrapper(array: PythonObject) raises -> PythonObject:
    comptime do_parallelize = True
    var array_len = len(array)
    var num_cores = num_physical_cores()
    var chunk_size, remainder = divmod(array_len, num_cores)

    def calc_max(i: Int) {imm} -> None:
        ref cpython = Python().cpython()
        # Each worker needs to hold the GIL to access python objects.
        # It is more efficient to only use Mojo native data structures in worker threads.
        with GILAcquired(Python(cpython)):
            try:
                var start_idx = i * chunk_size
                var is_last_thread = i == num_cores - 1
                var work_size = (
                    chunk_size + remainder
                ) if is_last_thread else chunk_size
                if work_size == 0:
                    return
                var end_idx = start_idx + work_size

                var max_val = array[start_idx]
                for j in range(start_idx + 1, end_idx):
                    if array[j] > max_val:
                        max_val = array[j]

                array[start_idx] = max_val
            except e:
                pass

    ref cpython = Python().cpython()

    comptime if do_parallelize:
        # Save the current thread state to avoid holding the GIL for the parallel loop.
        with GILReleased(Python(cpython)):
            parallelize(calc_max, num_cores)
    else:
        for i in range(0, num_cores):
            calc_max(i)

    var final_max = array[0]
    for i in range(1, num_cores):
        var idx = i * chunk_size
        if idx < array_len and array[idx] > final_max:
            final_max = array[idx]
    array[0] = final_max

    return array
