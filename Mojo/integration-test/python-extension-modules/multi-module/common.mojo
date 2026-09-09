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
from std.python import PythonObject


struct TestStruct(Defaultable, Movable, Writable):
    var a: Int
    var b: Int

    def __init__(out self):
        self.a = 0
        self.b = 0

    def __init__(out self, a: Int, b: Int):
        self.a = a
        self.b = b

    def print(self) -> None:
        print(self.a, self.b)

    @staticmethod
    def set_a(py_self: PythonObject, a: PythonObject):
        try:
            Self._get_self_ptr(py_self)[].a = Int(py=a)
        except e:
            abort(String("failed to set a: ", a))

    @staticmethod
    def set_b(py_self: PythonObject, b: PythonObject):
        try:
            Self._get_self_ptr(py_self)[].b = Int(py=b)
        except e:
            abort(String("failed to set b: ", b))

    @staticmethod
    def _get_self_ptr(
        py_self: PythonObject,
    ) -> Pointer[Self, MutAnyOrigin]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(
                String(
                    (
                        "Python method receiver object did not have the"
                        " expected type:"
                    ),
                    e,
                )
            )
