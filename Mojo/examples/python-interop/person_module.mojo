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

# DOC: Mojo/docs/site/manual/python/mojo-from-python.mdx

from std.os import abort

from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


@fieldwise_init
struct Person(Movable, Writable):
    var name: String
    var age: Int

    @staticmethod
    def py_init(
        out self: Person, args: PythonObject, kwargs: PythonObject
    ) raises:
        # Validate argument count
        if len(args) != 2:
            raise Error("Person() takes exactly 2 arguments")

        # Convert Python arguments to Mojo types
        var name = String(py=args[0])
        var age = Int(py=args[1])

        self = Self(name, age)


@export
def PyInit_person_module() abi("C") -> PythonObject:
    try:
        var mb = PythonModuleBuilder("person_module")

        _ = mb.add_type[Person]("Person").def_py_init[Person.py_init]()

        return mb.finalize()
    except e:
        abort(String("error creating Python Mojo module:", e))
