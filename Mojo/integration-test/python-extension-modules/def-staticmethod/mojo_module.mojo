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

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder


@export
def PyInit_mojo_module() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("mojo_module")

        _ = (
            b.add_type[Dummy]("Dummy")
            .def_init_defaultable[Dummy]()
            # def_staticmethod with return, raising
            .def_staticmethod[Dummy.takes_zero_raises_returns](
                "takes_zero_raises_returns"
            )
            .def_staticmethod[Dummy.takes_one_raises_returns](
                "takes_one_raises_returns"
            )
            .def_staticmethod[Dummy.takes_two_raises_returns](
                "takes_two_raises_returns"
            )
            .def_staticmethod[Dummy.takes_three_raises_returns](
                "takes_three_raises_returns"
            )
            # def_staticmethod with return, not raising
            .def_staticmethod[Dummy.takes_zero_returns]("takes_zero_returns")
            .def_staticmethod[Dummy.takes_one_returns]("takes_one_returns")
            .def_staticmethod[Dummy.takes_two_returns]("takes_two_returns")
            .def_staticmethod[Dummy.takes_three_returns]("takes_three_returns")
            # # def_staticmethod with no return, raising
            .def_staticmethod[Dummy.takes_zero_raises]("takes_zero_raises")
            .def_staticmethod[Dummy.takes_one_raises]("takes_one_raises")
            .def_staticmethod[Dummy.takes_two_raises]("takes_two_raises")
            .def_staticmethod[Dummy.takes_three_raises]("takes_three_raises")
            # # def_staticmethod with no return, not raising
            .def_staticmethod[Dummy.takes_zero]("takes_zero")
            .def_staticmethod[Dummy.takes_one]("takes_one")
            .def_staticmethod[Dummy.takes_two]("takes_two")
            .def_staticmethod[Dummy.takes_three]("takes_three")
            # def_staticmethod with kwargs
            .def_staticmethod[Dummy.sum_pos_arg_and_kwargs](
                "sum_pos_arg_and_kwargs"
            )
            .def_staticmethod[Dummy.append_kwarg]("append_kwarg")
            .def_staticmethod[Dummy.ignore_kwargs]("ignore_kwargs")
        )

        return b.finalize()
    except e:
        abort(String("failed to create Python module: ", e))


@fieldwise_init
struct Dummy(Defaultable, Movable, Writable):
    @staticmethod
    def takes_zero_raises_returns() raises -> PythonObject:
        var s = Python().evaluate("getattr(sys.modules['test_module'], 's')")
        if s != "just a python string":
            raise Error(String("`s` must be 'just a python string'"))

        return PythonObject("just another python string")

    @staticmethod
    def takes_one_raises_returns(
        a: PythonObject,
    ) raises -> PythonObject:
        if a != PythonObject("foo"):
            raise Error(String("input must be 'foo'"))
        return a

    @staticmethod
    def takes_two_raises_returns(
        a: PythonObject, b: PythonObject
    ) raises -> PythonObject:
        if a != PythonObject("foo"):
            raise Error(String("first input must be 'foo'"))
        return a + b

    @staticmethod
    def takes_three_raises_returns(
        a: PythonObject, b: PythonObject, c: PythonObject
    ) raises -> PythonObject:
        if a != PythonObject("foo"):
            raise Error(String("first input must be 'foo'"))
        return a + b + c

    @staticmethod
    def takes_zero_returns() -> PythonObject:
        try:
            return Self.takes_zero_raises_returns()
        except e:
            abort(String("Unexpected Python error: ", e))

    @staticmethod
    def takes_one_returns(a: PythonObject) -> PythonObject:
        try:
            return Self.takes_one_raises_returns(a)
        except e:
            abort(String("Unexpected Python error: ", e))

    @staticmethod
    def takes_two_returns(a: PythonObject, b: PythonObject) -> PythonObject:
        try:
            return Self.takes_two_raises_returns(a, b)
        except e:
            abort(String("Unexpected Python error: ", e))

    @staticmethod
    def takes_three_returns(
        a: PythonObject, b: PythonObject, c: PythonObject
    ) -> PythonObject:
        try:
            return Self.takes_three_raises_returns(a, b, c)
        except e:
            abort(String("Unexpected Python error: ", e))

    @staticmethod
    def takes_zero_raises() raises:
        var s = Python().evaluate("getattr(sys.modules['test_module'], 's')")
        if s != "just a python string":
            raise Error(String("`s` must be 'just a python string'"))

        _ = Python().eval(
            "setattr(sys.modules['test_module'], 's', 'Hark! A mojo function"
            " calling into Python, called from Python!')"
        )

    @staticmethod
    def takes_one_raises(list_obj: PythonObject) raises:
        if len(list_obj) != 3:
            raise Error(String("list_obj must have length 3"))
        list_obj[PythonObject(0)] = PythonObject("baz")

    @staticmethod
    def takes_two_raises(list_obj: PythonObject, obj: PythonObject) raises:
        if len(list_obj) != 3:
            raise Error(String("list_obj must have length 3"))
        list_obj[PythonObject(0)] = obj

    @staticmethod
    def takes_three_raises(
        list_obj: PythonObject, obj: PythonObject, obj2: PythonObject
    ) raises:
        if len(list_obj) != 3:
            raise Error(String("list_obj must have length 3"))
        list_obj[PythonObject(0)] = obj + obj2

    @staticmethod
    def takes_zero():
        try:
            Self.takes_zero_raises()
        except e:
            abort(String("Unexpected Python error: ", e))

    @staticmethod
    def takes_one(list_obj: PythonObject):
        try:
            Self.takes_one_raises(list_obj)
        except e:
            abort(String("Unexpected Python error: ", e))

    @staticmethod
    def takes_two(list_obj: PythonObject, obj: PythonObject):
        try:
            Self.takes_two_raises(list_obj, obj)
        except e:
            abort(String("Unexpected Python error: ", e))

    @staticmethod
    def takes_three(
        list_obj: PythonObject, obj: PythonObject, obj2: PythonObject
    ):
        try:
            Self.takes_three_raises(list_obj, obj, obj2)
        except e:
            abort(String("Unexpected Python error: ", e))

    @staticmethod
    def sum_pos_arg_and_kwargs(
        base: PythonObject, var **kwargs: PythonObject
    ) raises -> PythonObject:
        var total = Int(py=base)
        for entry in kwargs.items():
            total += Int(py=entry.value)
        return PythonObject(total)

    @staticmethod
    def append_kwarg(list_obj: PythonObject, var **kwargs: PythonObject) raises:
        list_obj.append(kwargs["value"])

    @staticmethod
    def ignore_kwargs(var **kwargs: PythonObject):
        pass
