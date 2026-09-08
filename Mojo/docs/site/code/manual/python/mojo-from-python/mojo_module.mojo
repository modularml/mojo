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

import std.math
from std.os import abort

from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


@export
def PyInit_mojo_module() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojo_module")
        m.def_function[factorial]("factorial", docstring="Compute n!")
        return m.finalize()
    except e:
        abort(String("error creating Python Mojo module:", e))


def factorial(py_obj: PythonObject) raises -> PythonObject:
    # Raises an exception if `py_obj` is not convertible to a Mojo `Int`.
    var n = Int(py=py_obj)

    return std.math.factorial(n)
