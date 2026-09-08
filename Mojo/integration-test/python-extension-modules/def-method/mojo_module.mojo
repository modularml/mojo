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
            b.add_type[Person]("Person")
            .def_init_defaultable[Person]()
            # def_method with return, raising
            .def_method[Person.get_name]("get_name")
            .def_method[Person.split_name]("split_name")
            .def_method[Person._with]("_with")
            # def_method with return, not raising
            .def_method[Person.get_age]("get_age")
            .def_method[Person._get_birth_year]("_get_birth_year")
            .def_method[Person._with_first_last_name]("_with_first_last_name")
            # def_method with no return, raising
            .def_method[Person.erase_name]("erase_name")
            .def_method[Person.set_age]("set_age")
            .def_method[Person.set_name_and_age]("set_name_and_age")
            # def_method with no return, not raising
            .def_method[Person.reset]("reset")
            .def_method[Person.set_name]("set_name")
            .def_method[Person._set_age_from_dates]("_set_age_from_dates")
            # def_method using automatic self downcasting
            .def_method[Person.set_name_auto]("set_name_auto")
            .def_method[Person.get_name_auto]("get_name_auto")
            .def_method[Person.increment_age_auto]("increment_age_auto")
            .def_method[Person.reset_auto]("reset_auto")
            # kwargs test methods
            .def_method[Person.sum_kwargs_ints]("sum_kwargs_ints")
            .def_py_method[Person.sum_kwargs_ints_py]("sum_kwargs_ints_py")
            # auto-convert self + kwargs test method
            .def_method[Person.add_kwargs_to_age_auto]("add_kwargs_to_age_auto")
            .def_method[Person.set_age_from_kwargs_auto](
                "set_age_from_kwargs_auto"
            )
        )
        return b.finalize()
    except e:
        abort(String("failed to create Python module: ", e))


@fieldwise_init
struct Person(Defaultable, ImplicitlyCopyable, Writable):
    var name: String
    var age: Int

    def __init__(out self):
        self.name = "John Smith"
        self.age = 123

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

    @staticmethod
    def get_name(py_self: PythonObject) raises -> PythonObject:
        # TODO: replace with property once we have them
        var self_ptr = Self._get_self_ptr(py_self)

        var s = Python().evaluate(
            "hasattr(sys.modules['test_module'], 'deny_name')"
        )
        if s:
            raise Error("name cannot be accessed")

        return PythonObject(self_ptr[].name)

    @staticmethod
    def split_name(
        py_self: PythonObject, sep: PythonObject
    ) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        var names = List[PythonObject]()
        for name in self_ptr[].name.split(String(py=sep)):
            names.append(PythonObject(name))
        return Python.list(names^)

    @staticmethod
    def _with(
        py_self: PythonObject, name: PythonObject, age: PythonObject
    ) raises -> PythonObject:
        Self.set_name_and_age(py_self, name, age)
        return py_self

    @staticmethod
    def get_age(py_self: PythonObject) -> PythonObject:
        # TODO: replace with property once we have them
        return PythonObject(Self._get_self_ptr(py_self)[].age)

    @staticmethod
    def _get_birth_year(
        py_self: PythonObject, this_year: PythonObject
    ) -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        try:
            return PythonObject(Int(py=this_year) - self_ptr[].age)
        except e:
            abort(String("failed to get birth year: ", e))

    @staticmethod
    def _with_first_last_name(
        py_self: PythonObject, first_name: PythonObject, last_name: PythonObject
    ) -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        try:
            self_ptr[].name = String(py=first_name) + " " + String(py=last_name)
        except e:
            abort(String("failed to set name: ", e))
        return py_self

    @staticmethod
    def erase_name(py_self: PythonObject) raises:
        var self_ptr = Self._get_self_ptr(py_self)
        if not self_ptr[].name:
            raise Error("cannot erase name if it's already empty")

        self_ptr[].name = String()

    @staticmethod
    def set_age(py_self: PythonObject, age: PythonObject) raises:
        var self_ptr = Self._get_self_ptr(py_self)
        try:
            self_ptr[].age = Int(py=age)
        except e:
            raise Error("cannot set age to ", String(py=age))

    @staticmethod
    def set_name_and_age(
        py_self: PythonObject, name: PythonObject, age: PythonObject
    ) raises:
        var self_ptr = Self._get_self_ptr(py_self)
        self_ptr[].name = String(py=name)
        Self.set_age(py_self, age)

    @staticmethod
    def reset(py_self: PythonObject):
        var self_ptr = Self._get_self_ptr(py_self)
        self_ptr[].name = "John Smith"
        self_ptr[].age = 123

    @staticmethod
    def set_name(py_self: PythonObject, name: PythonObject):
        # TODO: replace with property once we have them
        try:
            Self._get_self_ptr(py_self)[].name = String(py=name)
        except e:
            abort(String("failed to set name: ", e))

    @staticmethod
    def _set_age_from_dates(
        py_self: PythonObject, birth_year: PythonObject, this_year: PythonObject
    ):
        var self_ptr = Self._get_self_ptr(py_self)
        try:
            self_ptr[].age = Int(py=this_year) - Int(py=birth_year)
        except e:
            abort(String("failed to set age: ", e))

    @staticmethod
    def set_name_auto(
        self_ptr: Pointer[Self, MutAnyOrigin],
        name: PythonObject,
    ):
        try:
            self_ptr[].name = String(py=name)
        except e:
            abort(String("failed to set name: ", e))

    @staticmethod
    def get_name_auto(
        self_ptr: Pointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        return PythonObject(self_ptr[].name)

    @staticmethod
    def increment_age_auto(
        self_ptr: Pointer[Self, MutAnyOrigin],
        increment: PythonObject,
    ) raises -> PythonObject:
        self_ptr[].age += Int(py=increment)
        return PythonObject(self_ptr[].age)

    @staticmethod
    def reset_auto(self_ptr: Pointer[Self, MutAnyOrigin]):
        self_ptr[].name = "Auto Reset Person"
        self_ptr[].age = 999

    @staticmethod
    def sum_kwargs_ints(
        py_self: PythonObject, var **kwargs: PythonObject
    ) raises -> PythonObject:
        """Test method that takes kwargs, adds them to person's age and returns the new age.
        """
        var self_ptr = Self._get_self_ptr(py_self)
        return Self.add_kwargs_to_age_auto(self_ptr, **kwargs^)

    @staticmethod
    def sum_kwargs_ints_py(
        py_self: PythonObject, py_args: PythonObject, py_kwargs: PythonObject
    ) raises -> PythonObject:
        """Test def_py_method that takes kwargs, adds them to person's age and returns the new age.
        """
        var self_ptr = Self._get_self_ptr(py_self)
        var total = 0
        if py_kwargs._obj_ptr:
            for entry in py_kwargs.values():
                total += Int(py=entry)
        self_ptr[].age += total
        return PythonObject(self_ptr[].age)

    @staticmethod
    def add_kwargs_to_age_auto(
        self_ptr: Pointer[Self, MutAnyOrigin], var **kwargs: PythonObject
    ) raises -> PythonObject:
        """Test method with auto-convert self + kwargs that adds kwargs to person's age.
        """
        var total = 0
        for entry in kwargs.items():
            var value = entry.value
            total += Int(py=value)

        self_ptr[].age += total
        return PythonObject(self_ptr[].age)

    @staticmethod
    def set_age_from_kwargs_auto(
        self_ptr: Pointer[Self, MutAnyOrigin],
        base: PythonObject,
        var **kwargs: PythonObject,
    ) raises:
        self_ptr[].age = Int(py=base)
        for entry in kwargs.items():
            self_ptr[].age += Int(py=entry.value)
