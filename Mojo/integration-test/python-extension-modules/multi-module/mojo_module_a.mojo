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

from common import TestStruct
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


@export
def PyInit_mojo_module_a() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojo_module_a")
        _ = (
            m.add_type[TestStruct]("TestStruct")
            .def_init_defaultable[TestStruct]()
            .def_method[TestStruct.set_a]("set_a")
            .def_method[TestStruct.set_b]("set_b")
        )
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
