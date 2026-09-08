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
"""Provides infrastructure for creating Python bindings to Mojo code.

This module implements the core machinery for exposing Mojo functions and types
to Python through CPython's C API. It includes builder types for constructing
Python modules and type objects, wrapper functions for converting between Mojo
and Python calling conventions, and utilities for argument validation and type
conversion. This enables seamless bidirectional interoperability between Mojo
and Python code.
"""

from . import ConvertibleFromPython
from std.ffi import _Global, c_int, c_char
from std.sys.info import size_of
from std.collections import StringDict

from std.builtin._startup import _ensure_runtime_init
from std.builtin.variadics import _call_with_dynamic_pack_pointers
from std.reflection import reflect
from std.memory import OpaquePointer, unsafe_stack_allocation
from std.python import Python, PythonObject
from std.python._cpython import (
    GILAcquired,
    Py_TPFLAGS_DEFAULT,
    Py_ssize_t,
    PyCFunction,
    PyCFunctionFast,
    PyCFunctionWithKeywords,
    PyMethodDef,
    PyObject,
    PyObjectPtr,
    PyType_Slot,
    PyType_Spec,
    PyTypeObject,
    PyTypeObjectPtr,
)
from std.python.python_object import _unsafe_alloc, _unsafe_init

from std.utils import Variant

# ===-----------------------------------------------------------------------===#
# Global `PyTypeObject` Registration
# ===-----------------------------------------------------------------------===#

comptime MOJO_PYTHON_TYPE_OBJECTS = _Global[
    StorageType=Dict[StaticString, PythonObject],
    "MOJO_PYTHON_TYPE_OBJECTS",
    Dict[StaticString, PythonObject].__init__,
]
"""Mapping of Mojo type identifiers to unique `PyTypeObject*` binding
that Mojo type to this CPython interpreter instance."""


def _register_py_type_object(
    type_id: StaticString, var type_obj: PythonObject
) raises:
    """Register a Python type object for the identified Mojo type.

    The provided Python type object describes how a wrapped Mojo value can
    be used from within Python code.

    Args:
        type_id: The unique type id of a Mojo type.
        type_obj: The Python type object that binds the Mojo type identified
          by `type_id`.

    Raises:
        If a Python type object has already been registered in the current
        session for the provided type id.
    """
    var type_dict = MOJO_PYTHON_TYPE_OBJECTS.get_or_create_ptr()

    if type_id in type_dict[]:
        raise Error(
            (
                "Error building multiple Python type objects bound to"
                " Mojo type with id: "
            ),
            type_id,
        )

    type_dict[][type_id] = type_obj^


def lookup_py_type_object[T: AnyType]() raises -> PythonObject:
    """Retrieve a reference to the unique Python type describing Python objects
    containing Mojo values of type `T`.

    This function looks up the Python type object that was previously registered
    for the Mojo type `T` using a `PythonTypeBuilder`. The returned type object
    can be used to create Python objects that wrap Mojo values of type `T`.

    Parameters:
        T: The Mojo type to look up.

    Returns:
        A `PythonObject` representing the Python type object that binds the Mojo
        type `T` to the current CPython interpreter instance.

    Raises:
        If no `PythonTypeBuilder` was ever finalized for type `T`, or if no
        Python type object has been registered for the provided type identifier.
    """
    var type_dict = MOJO_PYTHON_TYPE_OBJECTS.get_or_create_ptr()

    # FIXME(MSTDL-1580):
    #   This should use a unique compiler type ID, not the Python name of this
    #   type.

    comptime type_name = reflect[T].name[qualified_builtins=True]()
    var entry = type_dict[].find(type_name)
    if entry:
        return entry.take()

    raise Error(
        "No Python type object registered for Mojo type with name: ",
        reflect[T].name(),
    )


# ===-----------------------------------------------------------------------===#
# Mojo Object
# ===-----------------------------------------------------------------------===#

# https://docs.python.org/3/c-api/typeobj.html#slot-type-typedefs


struct PyMojoObject[T: Deinitable](Movable where conforms_to(T, Movable)):
    """Storage backing a PyObject* wrapping a Mojo value.

    This struct represents the C-level layout of a Python object that contains
    a wrapped Mojo value. It must be ABI-compatible with CPython's PyObject
    structure to enable seamless interoperability between Mojo and Python.

    The struct follows Python's object model where all Python objects begin
    with a PyObject header (ob_base), followed by type-specific data. In this
    case, the type-specific data is a Mojo value of type T.

    See https://docs.python.org/3/c-api/structures.html#c.PyObject for more details.

    Parameters:
        T: The Mojo type being wrapped. Can be any type that satisfies `AnyType`.
    """

    var ob_base: PyObject
    """The standard Python object header containing reference count and type information.

    This must be the first field to maintain ABI compatibility with Python's object layout.
    All Python objects begin with this header structure.
    """

    var mojo_value: Self.T
    """The actual Mojo value being wrapped and exposed to Python.

    This field stores the Mojo data that Python code can interact with through
    the registered type methods and bindings.
    """

    # TODO(MSTDL-467): Replace with Optional[T] when Optional doesn't require Copyable anymore.
    var is_initialized: Bool
    """Whether the Mojo value has been initialized."""


def _tp_dealloc_wrapper[T: Deinitable](py_self: PyObjectPtr) abi("C"):
    """Python-compatible wrapper for deallocating a `PyMojoObject`.

    This function serves as the tp_dealloc slot for Python type objects that
    wrap Mojo values. It properly destroys the wrapped Mojo value and frees
    the Python object memory.

    Parameters:
        T: The wrapped Mojo type.

    Args:
        py_self: Pointer to the Python object to be deallocated.
    """
    ref cpython = Python().cpython()

    ref self = py_self.bitcast[PyMojoObject[T]]().value()[]

    # TODO(MSTDL-633):
    #   Is this always safe? Wrap in GIL, because this could
    #   evaluate arbitrary code?
    if self.is_initialized:
        Pointer(to=self.mojo_value).unsafe_deinit_pointee()

    cpython.PyObject_Free(py_self.bitcast[NoneType]())


def _tp_repr_wrapper[
    T: Deinitable
](py_self: PyObjectPtr) abi("C") -> PyObjectPtr:
    """Python-compatible wrapper for generating string representation of a
    `PyMojoObject`.

    This function serves as the `tp_repr` slot for Python type objects that
    wrap Mojo values. It calls the Mojo `repr()` function on the wrapped value
    and returns the result as a Python string object.

    Parameters:
        T: The wrapped Mojo type that must be `Writable`.

    Args:
        py_self: Pointer to the Python object to get representation for.

    Returns:
        A new Python string object containing the string representation,
        or null pointer if an error occurs.
    """
    ref cpython = Python().cpython()

    ref self = py_self.bitcast[PyMojoObject[T]]().value()[]

    var repr_str = String()
    if self.is_initialized:
        comptime assert conforms_to(
            T, Writable
        ), "_tp_repr_wrapper requires conformance to Writable."
        self.mojo_value.write_repr_to(repr_str)
    else:
        repr_str = String(t"<uninitialized {reflect[T].name()}>")

    return cpython.PyUnicode_DecodeUTF8(repr_str)


# ===-----------------------------------------------------------------------===#
# Builders
# ===-----------------------------------------------------------------------===#

comptime PyFunctionRaising = def(
    mut PythonObject, mut PythonObject
) thin raises -> PythonObject
"""The generic function type for raising Python bindings.

The first argument is the self object, and the second argument is a tuple of the
positional arguments. These functions always return a Python object (could be a
`None` object).
"""

comptime PyFunctionWithKeywordsRaising = def(
    mut PythonObject, mut PythonObject, mut PythonObject
) thin raises -> PythonObject
"""The generic function type for raising Python bindings with keyword arguments.

The first argument is the self object, the second argument is a tuple of the
positional arguments, and the third argument is a dictionary of the keyword arguments.
"""

comptime GenericPyFunction = Variant[
    PyFunctionRaising,
    PyFunctionWithKeywordsRaising,
]
"""A variant type that can hold either a PyFunctionRaising or PyFunctionWithKeywordsRaising."""


struct PythonModuleBuilder:
    """A builder for creating Python modules with Mojo function and type bindings.

    This builder provides a high-level API for declaring Python bindings for Mojo
    functions and types within a Python module. It manages the registration of
    functions, types, and their associated metadata, then finalizes everything
    into a complete Python module object.

    The builder follows a declarative pattern where you:
    1. Create a builder instance with a module name
    2. Add function bindings using `def_function()`, `def_py_function()`, `def_py_c_function()`
    3. Add type bindings using `add_type[T]()` and configure them
    4. Call `finalize()` to finish building the Python module.

    Example:
        ```mojo
        from std.python import PythonObject
        from std.python.bindings import PythonModuleBuilder

        def my_func(arg: PythonObject) -> PythonObject:
            return arg

        var builder = PythonModuleBuilder("my_module")
        builder.def_function[my_func]("my_func", "Documentation for my_func")

        var module = builder.finalize()
        ```

    Note:
        After calling `finalize()`, the builder's internal state is cleared and
        it should not be reused for creating additional modules.

        TODO: This should be enforced programmatically in the future.
    """

    var module: PythonObject
    """The Python module being built."""

    var functions: List[PyMethodDef]
    """List of function definitions that will be exposed in the module."""

    var type_builders: List[PythonTypeBuilder]
    """List of type builders for types that will be exposed in the module."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    def __init__(out self, name: StaticString) raises:
        """Construct a Python module builder with the given module name.

        Args:
            name: The name of the module.

        Raises:
            If the module creation fails.
        """
        self = Self(Python().create_module(name))

    def __init__(out self, module: PythonObject):
        """Construct a Python module builder with the given module.

        Args:
            module: The module to build.
        """
        self.module = module
        self.functions = []
        self.type_builders = []

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def add_type[
        T: Deinitable
    ](mut self, type_name: StaticString) -> ref[
        self.type_builders[0]
    ] PythonTypeBuilder:
        """Add a type to the module and return a builder for it.

        Parameters:
            T: The mojo type to bind in the module.

        Args:
            type_name: The name of the type to expose in the module.

        Returns:
            A reference to a type builder registered in the module builder.
        """
        self.type_builders.append(PythonTypeBuilder.bind[T](type_name))
        return self.type_builders[len(self.type_builders) - 1]

    def def_py_c_function(
        mut self,
        func: PyCFunction,
        func_name: StaticString,
        docstring: StaticString = "",
    ):
        """Declare a binding for a function with PyCFunction signature in the
        module.

        Args:
            func: The function to declare a binding for.
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        self.functions.append(PyMethodDef.function(func, func_name, docstring))

    def def_py_c_function(
        mut self,
        func: PyCFunctionWithKeywords,
        func_name: StaticString,
        docstring: StaticString = "",
    ):
        """Declare a binding for a function with PyCFunctionWithKeywords signature in the
        module.

        Args:
            func: The function to declare a binding for.
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        self.functions.append(PyMethodDef.function(func, func_name, docstring))

    def def_py_c_function(
        mut self,
        func: PyCFunctionFast,
        func_name: StaticString,
        docstring: StaticString = "",
    ):
        """Declare a binding for a function with `PyCFunctionFast` signature
        (`METH_FASTCALL`) in the module.

        Args:
            func: The function to declare a binding for.
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        self.functions.append(PyMethodDef.function(func, func_name, docstring))

    def def_py_function[
        func: PyFunctionRaising
    ](mut self, func_name: StaticString, docstring: StaticString = ""):
        """Declare a binding for a function with PyFunctionRaising signature in
        the module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        self._generic_def_py_function[func](func_name, docstring)

    def def_py_function[
        func: PyFunctionWithKeywordsRaising
    ](mut self, func_name: StaticString, docstring: StaticString = ""):
        """Declare a binding for a function with PyFunctionWithKeywordsRaising signature in
        the module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        self._generic_def_py_function[func](func_name, docstring)

    def _generic_def_py_function[
        func: GenericPyFunction
    ](mut self, func_name: StaticString, docstring: StaticString = ""):
        self.def_py_c_function(
            _py_c_function_wrapper[func], func_name, docstring
        )

    def def_function[
        PyArgs: TypeList[Trait=type_of(PythonObject), ...],
        RetType: Deinitable & Movable,
        //,
        func: def(
            * args: * PyArgs, var ** kwargs: PythonObject
        ) raises thin -> RetType,
    ](mut self, func_name: StaticString, docstring: StaticString = "") where (
        RetType == PythonObject or RetType == type_of(None),
        "function return type must be PythonObject or None",
    ):
        """Declare a binding for a module-level function.

        Accepts functions with PythonObject arguments, can optionally
        return a PythonObject, and can raise. Functions can also accept keyword
        arguments via `var **kwargs: PythonObject`.

        Non-kwargs callables register through CPython's `METH_FASTCALL`
        calling convention; kwargs-accepting callables use
        `METH_VARARGS | METH_KEYWORDS`.

        Example signatures:
        ```mojo
        from std.python import PythonObject

        def func(arg1: PythonObject) -> PythonObject: ...
        def func(arg1: PythonObject, arg2: PythonObject) raises: ...
        def func(var **kwargs: PythonObject) -> PythonObject: ...
        def func(arg1: PythonObject, var **kwargs: PythonObject) raises: ...
        ```

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """
        # Keyword-accepting functions still go through the
        # `METH_VARARGS | METH_KEYWORDS` dispatch path. The
        # corresponding `METH_FASTCALL | METH_KEYWORDS` protocol (with
        # `kwnames`) is a separate vectorcall shape and is not
        # implemented here yet.
        self._generic_def_py_function[_py_kwargs_function_wrapper[func]()](
            func_name, docstring
        )

    # Hidden overload that supports functions with non-kwargs signatures. Avoids
    # repeating the full docstring for a minor implementation detail.
    @doc_hidden
    def def_function[
        PyArgs: TypeList[Trait=type_of(PythonObject), ...],
        RetType: Deinitable & Movable,
        //,
        func: def(* args: * PyArgs) raises thin -> RetType,
    ](mut self, func_name: StaticString, docstring: StaticString = "") where (
        RetType == PythonObject or RetType == type_of(None),
        "function return type must be PythonObject or None",
    ):
        self.def_py_c_function(
            _py_function_fastcall_wrapper[func],
            func_name,
            docstring,
        )

    def finalize(mut self) raises -> PythonObject:
        """Finalize the module builder, creating the module object.


        All types and functions added to the builder will be built and exposed
        in the module. After calling this method, the builder's internal state
        is cleared and it should not be reused for creating additional modules.

        Returns:
            The finalized Python module containing all registered functions and types.

        Raises:
            If the module creation fails or if we fail to add any of the
            declared functions or types to the module.
        """

        var functions = self.functions^
        self.functions = List[PyMethodDef]()

        Python.add_functions(self.module, functions^)

        for ref builder in self.type_builders:
            builder.finalize(self.module)
        self.type_builders.clear()

        _ensure_runtime_init()

        return self.module


struct PythonTypeBuilder(Copyable):
    """A builder for a Python 'type' binding.

    This is typically used to build a type description of a `PyMojoObject[T]`.

    This builder is used to declare method bindings for a Python type, and then
    create the type binding.

    Finalizing builder created with `PythonTypeObject.bind[T]()` will globally
    register the resulting Python 'type' object as the single canonical type
    object for the Mojo type `T`. Subsequent attempts to register a Python type
    for `T` will raise an exception.

    Registering a Python type object for `T` is necessary to be able to
    construct a `PythonObject` from an instance of `T`, or to downcast an
    existing `PythonObject` to a pointer to the inner `T` value.
    """

    var type_name: StaticString
    """The name the type will be exposed as in the Python module."""

    var _type_id: Optional[StaticString]
    """The unique type identifier for the Mojo type being bound, if any."""

    var basicsize: Int
    """The required allocation size to hold an instance of this type as a Python object."""

    var _slots: Dict[Int, OptionalPointer[NoneType, MutUntrackedOrigin]]
    """Dictionary of Python type slots that define the behavior of the type, mapping slot number to function pointer."""

    var methods: List[PyMethodDef]
    """List of method definitions that will be exposed on the Python type."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    def __init__(out self, type_name: StaticString, *, basicsize: Int):
        """Construct a new builder for a Python type binding.

        Args:
            type_name: The name the type will be exposed as in the Python module.
            basicsize: The required allocation size to hold an instance of this
              type as a Python object.
        """

        self.type_name = type_name
        self._type_id = None
        self.basicsize = basicsize
        self._slots = {}
        self.methods = []

    @staticmethod
    def bind[T: Deinitable](type_name: StaticString) -> PythonTypeBuilder:
        """Construct a new builder for a Python type that binds a Mojo type.

        Parameters:
            T: The mojo type to bind.

        Args:
            type_name: The name the type will be exposed as in the Python module.

        Returns:
            A new type builder instance.
        """
        var b = PythonTypeBuilder(
            type_name,
            basicsize=size_of[PyMojoObject[T]](),
        )
        b._insert_slot(PyType_Slot.tp_new(_py_new_function_wrapper[T]))
        b._insert_slot(PyType_Slot.tp_init(_py_init_function_nonregistered))
        b._insert_slot(PyType_Slot.tp_dealloc(_tp_dealloc_wrapper[T]))
        b._insert_slot(PyType_Slot.tp_repr(_tp_repr_wrapper[T]))

        b.methods = List[PyMethodDef]()
        b._type_id = reflect[T].name[qualified_builtins=True]()

        return b^

    def finalize(mut self, module: PythonObject) raises:
        """Finalize the builder and add the created type to a Python module.

        This method completes the type building process by calling the
        parameterless `finalize()` method to create the Python type object, then
        automatically adds the resulting type to the specified Python module
        using the builder's configured type name. After successful completion,
        the builder's method list is cleared to prevent accidental reuse.

        This is a convenience method that combines type finalization and module
        registration in a single operation, which is the most common use case
        when creating Python-accessible Mojo types.

        Args:
            module: The Python module to which the finalized type will be added.
                The type will be accessible from Python code that imports this
                module using the name specified during builder construction.

        Raises:
            If the type object creation fails (see `finalize()` for details) or
            if adding the type to the module fails, typically due to name
            conflicts or module state issues.

        Note:
            After calling this method, the builder's internal state is modified
            (methods list is cleared), so the builder should not be reused for
            creating additional type objects. If you need the type object for
            further operations, use the parameterless `finalize()` method
            instead and manually add it to the module.
        """
        ref cpython = Python().cpython()

        if self.methods:
            self.methods.append(PyMethodDef())  # Zeroed item as terminator
            # FIXME: Avoid leaking the methods data pointer in this way.
            var methods_ptr = (
                self.methods.unsafe_take_allocation().unsafe_leak()
            )
            self._insert_slot(PyType_Slot.tp_methods(methods_ptr))

        # Convert _slots dictionary to a list of PyType_Slot structs
        var slots = List[PyType_Slot]()
        for slot_entry in self._slots.items():
            slots.append(PyType_Slot(c_int(slot_entry.key), slot_entry.value))

        # Zeroed item terminator
        slots.append(PyType_Slot.null())

        var type_spec = PyType_Spec(
            # FIXME(MOCO-1306): This should be `T.__name__`.
            self.type_name.as_c_string_slice(),
            c_int(self.basicsize),
            0,
            Py_TPFLAGS_DEFAULT,
            # Note: This pointer is only "read-only" by PyType_FromSpec.
            slots.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        )

        # Construct a Python 'type' object from our type spec.
        var type_obj_ptr = cpython.PyType_FromSpec(
            Pointer(to=type_spec).as_unsafe_any_origin()
        )

        if not type_obj_ptr:
            raise cpython.get_error()

        var type_obj = PythonObject(from_owned=type_obj_ptr)

        # Every Mojo type that is exposed to Python must have EXACTLY ONE
        # `PyTypeObject` instance that represents it. That is important for
        # correctness. This check here ensures that the user is not accidentally
        # creating multiple `PyTypeObject` instances that bind the same Mojo
        # type.
        var type_id = self._type_id
        if type_id:
            _register_py_type_object(type_id[], type_obj)

        Python.add_object(module, self.type_name, type_obj)
        self.methods.clear()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def _insert_slot(mut self, slot: PyType_Slot):
        """Insert a slot into the type builder.
        If the slot is already present, it will be replaced.

        Args:
            slot: The PyType_Slot to insert.
        """
        self._slots[Int(slot.slot)] = slot.pfunc

    def def_init_defaultable[
        T: Defaultable & Movable & Deinitable,
    ](mut self) raises -> ref[self] Self:
        """Declare a binding for the `__init__` method of the type which
        initializes the type with a default value.

        Parameters:
            T: The Mojo type to bind, which must be `Defaultable` and `Movable`.

        Returns:
            A reference to self for method chaining.

        Raises:
            If the slot insertion fails.
        """

        @always_inline
        def default_init_func(
            out self: T, args: PythonObject, kwargs: PythonObject
        ) raises:
            if len(args) > 0 or kwargs._obj_ptr:
                raise "unexpected arguments passed to default initializer function of wrapped Mojo type"
            self = T()

        self._insert_slot(
            PyType_Slot.tp_init(_py_init_function_wrapper[T, default_init_func])
        )
        return self

    def def_py_init[
        T: Movable & Deinitable,
        //,
        init_func: def(out T, args: PythonObject, kwargs: PythonObject) thin,
    ](mut self) raises -> ref[self] Self:
        """Declare a binding for the `__init__` method of the type.

        Parameters:
            T: The Mojo type to bind.
            init_func: The initialization function to bind.

        Returns:
            A reference to self for method chaining.

        Raises:
            If the slot insertion fails.
        """
        return self.def_py_init[_raising_py_init_wrapper[T, init_func]]()

    def def_py_init[
        T: Movable & Deinitable,
        //,
        init_func: def(
            out T, args: PythonObject, kwargs: PythonObject
        ) thin raises,
    ](mut self) raises -> ref[self] Self:
        """Declare a binding for the `__init__` method of the type.

        Parameters:
            T: The Mojo type to bind.
            init_func: The initialization function to bind (may raise).

        Returns:
            A reference to self for method chaining.

        Raises:
            If the slot insertion fails.
        """
        self._insert_slot(
            PyType_Slot.tp_init(_py_init_function_wrapper[T, init_func])
        )
        return self

    def def_py_c_method[
        static_method: Bool = False
    ](
        mut self,
        method: PyCFunction,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref[self] Self:
        """Declare a binding for a method with PyObjectPtr signature for the
        type.

        Parameters:
            static_method: Whether the method is exposed as a staticmethod.
                Default is False. Note that CPython will pass a null pointer for
                the first argument for static methods (i.e. instead of passing
                the self object). See [METH_STATIC](https://docs.python.org/3/c-api/structures.html#c.METH_STATIC).

        Args:
            method: The method to declare a binding for.
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """
        self.methods.append(
            PyMethodDef.function[static_method](method, method_name, docstring)
        )
        return self

    def def_py_c_method[
        static_method: Bool = False
    ](
        mut self,
        method: PyCFunctionWithKeywords,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref[self] Self:
        """Declare a binding for a method with PyCFunctionWithKeywords signature for the
        type.

        Parameters:
            static_method: Whether the method is exposed as a staticmethod.
                Default is False. Note that CPython will pass a null pointer for
                the first argument for static methods (i.e. instead of passing
                the self object). See [METH_STATIC](https://docs.python.org/3/c-api/structures.html#c.METH_STATIC).

        Args:
            method: The method to declare a binding for.
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        self.methods.append(
            PyMethodDef.function[static_method](method, method_name, docstring)
        )
        return self

    def def_py_c_method[
        static_method: Bool = False
    ](
        mut self,
        method: PyCFunctionFast,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref[self] Self:
        """Declare a binding for a method with `PyCFunctionFast` signature
        (`METH_FASTCALL`) for the type.

        Parameters:
            static_method: Whether the method is exposed as a staticmethod.
                Default is False.

        Args:
            method: The fastcall method to declare a binding for.
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        self.methods.append(
            PyMethodDef.function[static_method](method, method_name, docstring)
        )
        return self

    def def_py_method[
        method: PyFunctionRaising, static_method: Bool = False
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref[self] Self:
        """Declare a binding for a method with PyFunctionRaising signature.

        Accepts methods with signature: `def (mut PythonObject, mut PythonObject) thin raises -> PythonObject`
        where the first arg is self and the second is a tuple of arguments.

        Parameters:
            method: The method to declare a binding for.
            static_method: Whether the method is exposed as a staticmethod.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        return self._generic_def_py_method[method, static_method](
            method_name, docstring
        )

    def def_py_method[
        method: PyFunctionWithKeywordsRaising, static_method: Bool = False
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref[self] Self:
        """Declare a binding for a method with PyFunctionWithKeywordsRaising signature.

        Accepts methods with signature:
        `def (mut PythonObject, mut PythonObject, mut PythonObject) thin raises -> PythonObject`
        where the first arg is self, the second is a tuple of arguments, and the third is a dict of keyword arguments.

        Parameters:
            method: The method to declare a binding for.
            static_method: Whether the method is exposed as a staticmethod.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        return self._generic_def_py_method[method, static_method](
            method_name, docstring
        )

    def _generic_def_py_method[
        method: GenericPyFunction,
        static_method: Bool = False,
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = "",
    ) -> ref[self] Self:
        return self.def_py_c_method[static_method](
            _py_c_function_wrapper[method], method_name, docstring
        )

    def def_method[
        SelfType: Deinitable,
        PyArgs: TypeList[Trait=type_of(PythonObject), ...],
        RetType: Deinitable & Movable,
        //,
        method: def(
            self_: Pointer[SelfType, MutUnsafeAnyOrigin], * args: * PyArgs,
            var ** kwargs: PythonObject,
        ) raises thin -> RetType,
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = "",
    ) -> ref[self] Self where (
        RetType == PythonObject or RetType == type_of(None),
        "function return type must be PythonObject or None",
    ):
        """Declares a binding for a method with an automatically downcast self.

        The method receives a pointer to the wrapped Mojo value. Methods that
        need generic Python object access can receive `PythonObject` instead.

        Non-kwargs methods register through CPython's `METH_FASTCALL`
        calling convention; kwargs-accepting methods use
        `METH_VARARGS | METH_KEYWORDS`.

        Example signatures:
        ```mojo
        from std.python import PythonObject

        def method(
            self: Pointer[Self, MutUnsafeAnyOrigin],
            arg: PythonObject,
            var **kwargs: PythonObject,
        ) raises -> PythonObject: ...
        ```

        Parameters:
            SelfType: The wrapped Mojo self type.
            PyArgs: The method's positional Python argument types.
            RetType: The method's return type.
            method: The method to declare a binding for.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """
        return self._generic_def_py_method[
            _py_kwargs_method_wrapper[method](),
            static_method=False,
        ](method_name, docstring)

    # Hidden overload to support slightly different signatures. Avoids
    # repeating the full docstring for a minor implementation detail.
    @doc_hidden
    def def_method[
        PyArgs: TypeList[Trait=type_of(PythonObject), ...],
        RetType: Deinitable & Movable,
        //,
        method: def(
            self_: PythonObject, * args: * PyArgs, var ** kwargs: PythonObject
        ) raises thin -> RetType,
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = "",
    ) -> ref[self] Self where (
        RetType == PythonObject or RetType == type_of(None),
        "function return type must be PythonObject or None",
    ):
        return self._generic_def_py_method[
            _py_kwargs_method_wrapper[method](),
            static_method=False,
        ](method_name, docstring)

    # Hidden overload to support slightly different signatures. Avoids
    # repeating the full docstring for a minor implementation detail.
    @doc_hidden
    def def_method[
        SelfType: Deinitable,
        PyArgs: TypeList[Trait=type_of(PythonObject), ...],
        RetType: Deinitable & Movable,
        //,
        method: def(
            self_: Pointer[SelfType, MutUnsafeAnyOrigin], * args: * PyArgs
        ) raises thin -> RetType,
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = "",
    ) -> ref[self] Self where (
        RetType == PythonObject or RetType == type_of(None),
        "function return type must be PythonObject or None",
    ):
        return self.def_py_c_method[static_method=False](
            _py_method_typed_fastcall_wrapper[method],
            method_name,
            docstring,
        )

    # Hidden overload to support slightly different signatures. Avoids
    # repeating the full docstring for a minor implementation detail.
    @doc_hidden
    def def_method[
        PyArgs: TypeList[Trait=type_of(PythonObject), ...],
        RetType: Deinitable & Movable,
        //,
        method: def(
            self_: PythonObject, * args: * PyArgs
        ) raises thin -> RetType,
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = "",
    ) -> ref[self] Self where (
        RetType == PythonObject or RetType == type_of(None),
        "function return type must be PythonObject or None",
    ):
        return self.def_py_c_method[static_method=False](
            _py_method_fastcall_wrapper[method],
            method_name,
            docstring,
        )

    def def_staticmethod[
        PyArgs: TypeList[Trait=type_of(PythonObject), ...],
        RetType: Deinitable & Movable,
        //,
        method: def(
            * args: * PyArgs, var ** kwargs: PythonObject
        ) raises thin -> RetType,
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref[self] Self where (
        RetType == PythonObject or RetType == type_of(None),
        "function return type must be PythonObject or None",
    ):
        """Declares a binding for a static method with optional keyword arguments.

        Accepts methods with `PythonObject` positional and keyword arguments.
        The method can return a `PythonObject` or `None`, and can raise.

        Non-kwargs static methods register through CPython's `METH_FASTCALL`
        calling convention; kwargs-accepting static methods use
        `METH_VARARGS | METH_KEYWORDS`.

        Example signatures:
        ```mojo
        from std.python import PythonObject

        def static_method(arg1: PythonObject) -> PythonObject: ...
        def static_method(arg1: PythonObject, arg2: PythonObject) raises: ...
        def static_method(
            arg: PythonObject, var **kwargs: PythonObject
        ) raises -> PythonObject: ...
        ```

        Parameters:
            method: The static method to declare a binding for.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """
        return self._generic_def_py_method[
            _py_kwargs_function_wrapper[method](), static_method=True
        ](method_name, docstring)

    # Hidden overload that supports functions with non-kwargs signatures. Avoids
    # repeating the full docstring for a minor implementation detail.
    @doc_hidden
    def def_staticmethod[
        PyArgs: TypeList[Trait=type_of(PythonObject), ...],
        RetType: Deinitable & Movable,
        //,
        method: def(* args: * PyArgs) raises thin -> RetType,
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref[self] Self where (
        RetType == PythonObject or RetType == type_of(None),
        "function return type must be PythonObject or None",
    ):
        return self.def_py_c_method[static_method=True](
            _py_function_fastcall_wrapper[method],
            method_name,
            docstring,
        )


# ===-----------------------------------------------------------------------===#
# Error Translation
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct ExceptionType(TrivialRegisterPassable):
    """A CPython global exception type used to translate a Mojo `Error` into a
    Python exception.
    """

    var global_name: StaticString
    """The name of the backing CPython global, for example `PyExc_TypeError`."""

    comptime Exception = Self("PyExc_Exception")
    """The base `Exception` type."""

    comptime TypeError = Self("PyExc_TypeError")
    """The `TypeError` type."""

    comptime ValueError = Self("PyExc_ValueError")
    """The `ValueError` type."""


def _set_python_error(
    e: Error, exc_type: ExceptionType = ExceptionType.Exception
):
    """Set the active Python exception from a Mojo `Error`.

    Translates `e` into a Python exception of type `exc_type` via
    `PyErr_SetString`, leaving the CPython error indicator set so the
    enclosing `PyCFunction` wrapper can signal the failure to CPython.

    Args:
        e: The Mojo error to translate.
        exc_type: The CPython global exception type to set. Defaults to
            `ExceptionType.Exception`.
    """
    ref cpython = Python().cpython()
    var error_message = String(e)
    var error_type = cpython.get_error_global(exc_type.global_name)
    cpython.PyErr_SetString(
        error_type,
        error_message.as_c_string_slice().ptr().as_unsafe_any_origin(),
    )


def raise_python_exception(
    e: Error, exc_type: ExceptionType = ExceptionType.Exception
) -> PyObjectPtr:
    """Translate a Mojo `Error` into a Python exception and return NULL.

    Sets the active Python exception via `PyErr_SetString` so that the
    calling `PyCFunction` wrapper can return the resulting null `PyObjectPtr`
    to signal the error to CPython.

    Example:

    ```mojo
    from std.python import PythonObject
    from std.python._cpython import PyObjectPtr
    from std.python.bindings import raise_python_exception

    def do_work(args: PyObjectPtr) -> PythonObject:
        # Your wrapper's real work, which may raise a Mojo `Error`.
        return PythonObject(from_borrowed=args)

    def my_wrapper(py_self: PyObjectPtr, args: PyObjectPtr) -> PyObjectPtr:
        try:
            return do_work(args).steal_data()
        except e:
            return raise_python_exception(e)
    ```

    Args:
        e: The Mojo error to translate.
        exc_type: The CPython global exception type to set. Defaults to
            `ExceptionType.Exception`.

    Returns:
        A null `PyObjectPtr`, which signals the error to CPython.
    """
    _set_python_error(e, exc_type)
    return PyObjectPtr()


# ===-----------------------------------------------------------------------===#
# PyCFunction Wrappers
# ===-----------------------------------------------------------------------===#


def _py_init_function_nonregistered(
    py_self_ptr: PyObjectPtr, args_ptr: PyObjectPtr, kwargs_ptr: PyObjectPtr
) abi("C") -> c_int:
    ref cpython = Python().cpython()
    var error_type = cpython.get_error_global("PyExc_TypeError")
    cpython.PyErr_SetString(
        error_type,
        "No initializer registered for this type. Use def_py_init() or"
        " def_init_defaultable() to register an initializer.".as_c_string_slice()
        .ptr()
        .as_unsafe_any_origin(),
    )
    return -1


def _py_new_function_wrapper[
    T: AnyType
](subtype: PyTypeObjectPtr, args_ptr: PyObjectPtr, kwargs_ptr: PyObjectPtr) abi(
    "C"
) -> PyObjectPtr:
    try:
        return _unsafe_alloc[T](subtype)
    except e:
        return raise_python_exception(e, ExceptionType.TypeError)


def _py_init_function_wrapper[
    T: Movable & Deinitable,
    init_func: def(out T, args: PythonObject, kwargs: PythonObject) thin raises,
](py_self: PyObjectPtr, args_ptr: PyObjectPtr, kwargs_ptr: PyObjectPtr) abi(
    "C"
) -> c_int:
    """Wrapper function that adapts a Mojo `PyInitFunction` to be callable from
    Python.
    """

    var kwargs = PythonObject(from_borrowed=kwargs_ptr)
    var args = PythonObject(from_borrowed=args_ptr)

    try:
        var value = init_func(args, kwargs)
        _unsafe_init(py_self, value^)
        return 0

    except e:
        # TODO(MSTDL-933): Add custom 'MojoError' type, and raise it here.
        _set_python_error(e, ExceptionType.ValueError)
        return -1


@always_inline
def _raising_py_init_wrapper[
    T: Movable & Deinitable,
    init_func: def(args: PythonObject, kwargs: PythonObject) thin -> T,
](out t: T, args: PythonObject, kwargs: PythonObject) raises:
    t = init_func(args, kwargs)


@always_inline
def _py_c_function_wrapper[
    user_func: GenericPyFunction
](py_self_ptr: PyObjectPtr, args_ptr: PyObjectPtr, kwargs_ptr: PyObjectPtr) abi(
    "C"
) -> PyObjectPtr:
    """
    1. Wraps a raw Python C function to convert raw `PyObjectPtr`s to `PythonObject`s.
    `PythonObject`s are managed objects which automatically handle reference counting,
    and are the preferred way to interact with Python objects in Mojo.

    2. Catches exceptions thrown by user supplied functions and converts them to Python exceptions.

    Parameters:
        user_func: The Mojo function to wrap.
    Args:
        py_self_ptr: Pointer to the Python object representing 'self' in the
            method call. This is borrowed from the caller.
        args_ptr: Pointer to a Python tuple containing the positional arguments
            passed to the function. This is borrowed from the caller.
        kwargs_ptr: Optional pointer to a Python dictionary containing the keyword arguments
            passed to the function. This is borrowed from the caller.

    Returns:
        A new Python object pointer containing the result of the user function,
        or a null pointer if an error occurred during execution.

    Note:
        This function carefully manages object ownership according to Python's
        reference counting rules. The input arguments are borrowed references
        that must not be decremented, while the return value is a new reference
        that the caller will own.
    """

    #   > When a C function is called from Python, it borrows references to its
    #   > arguments from the caller. The caller owns a reference to the object,
    #   > so the read-only reference's lifetime is guaranteed until the function
    #   > returns. Only when such a read-only reference must be stored or passed
    #   > on, it must be turned into an owned reference by calling Py_INCREF().
    #   >
    #   >  -- https://docs.python.org/3/extending/extending.html#ownership-rules
    #
    # We turn these into owned references, knowing that their destructors will
    # appropriately decrement the reference count.

    var py_self = PythonObject(from_borrowed=py_self_ptr)
    var args = PythonObject(from_borrowed=args_ptr)

    # SAFETY:
    #   Call the user provided function, and take ownership of the
    #   PyObjectPtr of the returned PythonObject.
    #
    # CPython holds the GIL across the lifetime of an extension-function
    # call (PEP 311; CPython's call protocol always enters with the GIL
    # held), so we do not acquire it here. Acquiring it would just cost
    # an extra PyGILState_Ensure/Release round-trip per call.

    try:
        comptime if user_func.isa[PyFunctionRaising]():
            return user_func.unsafe_get[PyFunctionRaising]()(
                py_self, args
            ).steal_data()
        elif user_func.isa[PyFunctionWithKeywordsRaising]():
            var kwargs = PythonObject(from_borrowed=kwargs_ptr)
            return user_func.unsafe_get[PyFunctionWithKeywordsRaising]()(
                py_self, args, kwargs
            ).steal_data()
        else:
            comptime assert False, "unknown `GenericPyFunction` variant"
    except e:
        # Return a NULL `PyObject*`, with the Python error indicator set.
        return raise_python_exception(e)


def _convert_kwargs(
    py_kwargs: PythonObject,
) raises -> StringDict[PythonObject]:
    """Convert a Python dictionary to a StringDict.

    Args:
        py_kwargs: Python dictionary containing keyword arguments.

    Returns:
        A StringDict containing the keyword arguments.
    """
    var result = StringDict[PythonObject]()

    # Handle the case where kwargs is None or empty
    if not py_kwargs._obj_ptr:
        return result^

    # Iterate through the Python dictionary and populate StringDict
    var items = py_kwargs.items()
    for item in items:
        var key = item[0]
        var value = item[1]
        var key_str = String(key)
        result[key_str] = value

    return result^


@always_inline
def _py_kwargs_function_wrapper[
    PyArgs: TypeList[Trait=type_of(PythonObject), ...],
    RetType: Deinitable & Movable,
    //,
    func: def(
        * args: * PyArgs, var ** kwargs: PythonObject
    ) raises thin -> RetType,
]() -> GenericPyFunction:
    """Converts a kwargs-accepting user signature to a GenericPyFunction.

    Wraps the user's function in a `METH_VARARGS | METH_KEYWORDS` dispatch
    shim. Non-kwargs callables go through `_py_function_fastcall_wrapper`
    (METH_FASTCALL) instead.
    """

    @always_inline
    def wrapper_with_kwargs(
        mut py_self: PythonObject,
        mut py_args: PythonObject,
        mut py_kwargs: PythonObject,
    ) raises -> PythonObject:
        _ = py_self
        return _dispatch_python_object_kwargs_function[func](py_args, py_kwargs)

    return GenericPyFunction(wrapper_with_kwargs)


@always_inline
def _py_kwargs_method_wrapper[
    SelfType: Deinitable,
    PyArgs: TypeList[Trait=type_of(PythonObject), ...],
    RetType: Deinitable & Movable,
    //,
    method: def(
        self_: Pointer[SelfType, MutUnsafeAnyOrigin], * args: * PyArgs,
        var ** kwargs: PythonObject,
    ) raises thin -> RetType,
]() -> GenericPyFunction:
    @always_inline
    def wrapper_with_kwargs(
        mut py_self: PythonObject,
        mut py_args: PythonObject,
        mut py_kwargs: PythonObject,
    ) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[SelfType]()
        return _dispatch_python_object_kwargs_method[method](
            self_ptr, py_args, py_kwargs
        )

    return GenericPyFunction(wrapper_with_kwargs)


@always_inline
def _py_kwargs_method_wrapper[
    PyArgs: TypeList[Trait=type_of(PythonObject), ...],
    RetType: Deinitable & Movable,
    //,
    method: def(
        self_: PythonObject, * args: * PyArgs, var ** kwargs: PythonObject
    ) raises thin -> RetType,
]() -> GenericPyFunction:
    @always_inline
    def wrapper_with_kwargs(
        mut py_self: PythonObject,
        mut py_args: PythonObject,
        mut py_kwargs: PythonObject,
    ) raises -> PythonObject:
        return _dispatch_python_object_kwargs_method[method](
            py_self, py_args, py_kwargs
        )

    return GenericPyFunction(wrapper_with_kwargs)


# ===-----------------------------------------------------------------------===#
# METH_FASTCALL Wrappers
# ===-----------------------------------------------------------------------===#


@always_inline
def _py_function_fastcall_wrapper[
    PyArgs: TypeList[Trait=type_of(PythonObject), ...],
    RetType: Deinitable & Movable,
    //,
    func: def(* args: * PyArgs) raises thin -> RetType,
](
    py_self_ptr: PyObjectPtr,
    args: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    """Build a `METH_FASTCALL`-shaped wrapper around a non-kwargs user Python
    function.

    CPython will invoke the returned wrapper through the `PyCFunctionFast`
    calling convention: `self` is a borrowed `PyObject*`, `args` is a
    borrowed C array of `PyObject*` of length `nargs`, and no tuple is
    ever constructed for the positional arguments. The wrapper forwards
    `args` directly to `_dispatch_python_object_function`, which reads
    `args[i]` without going through the tuple-mapping protocol.
    Exceptions raised by the user function are translated into Python
    exceptions and signaled by returning a NULL `PyObject*`.

    Parameters:
        func: The wrapped Mojo function being registered with the module
            or type.

    Returns:
        A function value of type `PyCFunctionFast` suitable for passing to
        `PyMethodDef.function` for `METH_FASTCALL` registration.
    """

    _ = py_self_ptr

    try:
        # CPython's vectorcall protocol (PEP 590) guarantees `args` is
        # non-null for every METH_FASTCALL invocation, including the
        # `nargs == 0` case (CPython hands the callee a pointer into a
        # cached empty tuple). `_dispatch_fast` therefore accepts a plain
        # `Pointer` rather than `OptionalPointer`.
        return _dispatch_python_object_function[func](
            args, Int(nargs)
        ).steal_data()
    except e:
        # Return a NULL `PyObject*`, with the Python error indicator set.
        return raise_python_exception(e)


@always_inline
def _py_method_typed_fastcall_wrapper[
    SelfType: Deinitable,
    PyArgs: TypeList[Trait=type_of(PythonObject), ...],
    RetType: Deinitable & Movable,
    //,
    method: def(
        self_: Pointer[SelfType, MutUnsafeAnyOrigin], * args: * PyArgs
    ) raises thin -> RetType,
](
    py_self_ptr: PyObjectPtr,
    args: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        var py_self = PythonObject(from_borrowed=py_self_ptr)
        var self_ptr = py_self.downcast_value_ptr[SelfType]()
        return _dispatch_python_object_method[method](
            self_ptr, args, Int(nargs)
        ).steal_data()
    except e:
        return raise_python_exception(e)


@always_inline
def _py_method_fastcall_wrapper[
    PyArgs: TypeList[Trait=type_of(PythonObject), ...],
    RetType: Deinitable & Movable,
    //,
    method: def(self_: PythonObject, * args: * PyArgs) raises thin -> RetType,
](
    py_self_ptr: PyObjectPtr,
    args: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        var py_self = PythonObject(from_borrowed=py_self_ptr)
        return _dispatch_python_object_method[method](
            py_self, args, Int(nargs)
        ).steal_data()
    except e:
        return raise_python_exception(e)


# ===-----------------------------------------------------------------------===#
# VariadicPack PythonObject function calling wrappers
# ===-----------------------------------------------------------------------===#
#
# Using variadics and return type generics, we can effectively support calling
# user functions of any arity and either `-> None` or `-> PythonObject` return
# type. However, in Mojo today we can't easily abstract over:
#
#   1) presence of a trailing `**kwargs` argument
#   2) leading method `SelfArg` type, which may or may not be `PythonObject`
#
# The four function "shapes" that we can't currently abstract over:
#
#     def(*args: *PyArgs) raises thin -> RetType
#     def(*args: *PyArgs, var **kwargs: PythonObject) raises thin -> RetType
#     def(self_: SelfArg, *args: *PyArgs) raises thin -> RetType
#     def(self_: SelfArg, *args: *PyArgs, var **kwargs: PythonObject) raises thin -> RetType,
#
# are each handled by a dedicated dispatcher below.


@always_inline("nodebug")
def _dispatch_python_object_function[
    PyArgs: TypeList[Trait=type_of(PythonObject), ...],
    RetType: Deinitable & Movable,
    //,
    func: def(* args: * PyArgs) raises thin -> RetType,
](
    args: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Int,
) raises -> PythonObject:
    check_arguments_arity(PyArgs.length, nargs)

    def get_arg_ptr[
        idx: Int
    ]() {args} -> Pointer[PyArgs[idx], MutUnsafeAnyOrigin]:
        var p: Pointer[PyObjectPtr, _] = Pointer(to=args[unsafe_offset=idx])
        return rebind_var[Pointer[PyArgs[idx], MutUnsafeAnyOrigin]](
            p.unsafe_bitcast[PythonObject]().as_unsafe_any_origin()
        )

    var result = _call_with_dynamic_pack_pointers[
        ArgTrait=type_of(PythonObject), func
    ](get_arg_ptr)

    return _return_python_object(result^)


@always_inline("nodebug")
def _dispatch_python_object_method[
    SelfArg: Deinitable,
    PyArgs: TypeList[Trait=type_of(PythonObject), ...],
    RetType: Deinitable & Movable,
    //,
    method: def(self_: SelfArg, * args: * PyArgs) raises thin -> RetType,
](
    self_arg: SelfArg,
    args: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Int,
) raises -> PythonObject:
    check_arguments_arity(PyArgs.length, nargs)

    comptime ToPointer[
        T: type_of(PythonObject)
    ]: ImplicitlyCopyable & Deinitable = Pointer[T, MutUnsafeAnyOrigin]
    var pointers: Tuple[*PyArgs.map[ToPointer]()]
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(pointers))
    comptime for i in range(PyArgs.length):
        var p: Pointer[PyObjectPtr, _] = Pointer(to=args[unsafe_offset=i])
        pointers[i] = rebind[type_of(pointers[i])](
            p.unsafe_bitcast[PythonObject]().as_unsafe_any_origin()
        )

    comptime BorrowedPack = VariadicPack[
        origin=MutUnsafeAnyOrigin,
        element_trait=type_of(PythonObject),
        False,
        *PyArgs,
    ]
    var borrowed = BorrowedPack(
        __mlir_op.`lit.ref.pack.from_pointer_pack`[
            _type=BorrowedPack._mlir_type
        ](pointers._mlir_value)
    )
    var result = method(self_arg, *borrowed)

    return _return_python_object(result^)


# TODO: Combine this overload if/when it's possible to make
#       `_call_with_dynamic_pack_pointers` generic over presence of kwargs.
@always_inline("nodebug")
def _dispatch_python_object_kwargs_function[
    PyArgs: TypeList[Trait=type_of(PythonObject), ...],
    RetType: Deinitable & Movable,
    //,
    func: def(
        * args: * PyArgs, var ** kwargs: PythonObject
    ) raises thin -> RetType,
](py_args: PythonObject, py_kwargs: PythonObject) raises -> PythonObject:
    check_arguments_arity(PyArgs.length, py_args)

    var positional_args = Array[PythonObject, PyArgs.length](uninitialized=True)
    comptime for i in range(PyArgs.length):
        positional_args.unsafe_ptr().unsafe_offset(i).unsafe_write(py_args[i])

    comptime ToPointer[
        T: type_of(PythonObject)
    ]: ImplicitlyCopyable & Deinitable = Pointer[T, MutUnsafeAnyOrigin]
    var pointers: Tuple[*PyArgs.map[ToPointer]()]
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(pointers))
    comptime for i in range(PyArgs.length):
        var element = (
            positional_args.unsafe_ptr().unsafe_offset(i).as_unsafe_any_origin()
        )
        pointers[i] = rebind[type_of(pointers[i])](element)

    comptime BorrowedPack = VariadicPack[
        origin=MutUnsafeAnyOrigin,
        element_trait=type_of(PythonObject),
        False,
        *PyArgs,
    ]
    var borrowed = BorrowedPack(
        __mlir_op.`lit.ref.pack.from_pointer_pack`[
            _type=BorrowedPack._mlir_type
        ](pointers._mlir_value)
    )
    var kwargs = _convert_kwargs(py_kwargs)
    var result = func(*borrowed, **kwargs^)

    return _return_python_object(result^)


# TODO: Combine this overload if/when it's possible to make
#       `_call_with_dynamic_pack_pointers` generic over presence of `SelfArg`
#       and kwargs.
@always_inline("nodebug")
def _dispatch_python_object_kwargs_method[
    SelfArg: Deinitable,
    PyArgs: TypeList[Trait=type_of(PythonObject), ...],
    RetType: Deinitable & Movable,
    //,
    method: def(
        self_: SelfArg, * args: * PyArgs,
        var ** kwargs: PythonObject,
    ) raises thin -> RetType,
](
    self_arg: SelfArg,
    py_args: PythonObject,
    py_kwargs: PythonObject,
) raises -> PythonObject:
    check_arguments_arity(PyArgs.length, py_args)

    var positional_args = Array[PythonObject, PyArgs.length](uninitialized=True)
    comptime for i in range(PyArgs.length):
        positional_args.unsafe_ptr().unsafe_offset(i).unsafe_write(py_args[i])

    comptime ToPointer[
        T: type_of(PythonObject)
    ]: ImplicitlyCopyable & Deinitable = Pointer[T, MutUnsafeAnyOrigin]
    var pointers: Tuple[*PyArgs.map[ToPointer]()]
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(pointers))
    comptime for i in range(PyArgs.length):
        var element = (
            positional_args.unsafe_ptr().unsafe_offset(i).as_unsafe_any_origin()
        )
        pointers[i] = rebind[type_of(pointers[i])](element)

    comptime BorrowedPack = VariadicPack[
        origin=MutUnsafeAnyOrigin,
        element_trait=type_of(PythonObject),
        False,
        *PyArgs,
    ]
    var borrowed = BorrowedPack(
        __mlir_op.`lit.ref.pack.from_pointer_pack`[
            _type=BorrowedPack._mlir_type
        ](pointers._mlir_value)
    )
    var kwargs = _convert_kwargs(py_kwargs)
    var result = method(self_arg, *borrowed, **kwargs^)

    return _return_python_object(result^)


def _return_python_object[
    RetType: Movable & Deinitable
](var result: RetType,) -> PythonObject:
    comptime if RetType == PythonObject:
        # TODO: This rebind shouldn't be necessary
        return rebind_var[PythonObject](result^)
    else:
        comptime assert RetType == type_of(None)
        return PythonObject(None)


# ===-----------------------------------------------------------------------===#
# Utilities for building Python bindings
# ===-----------------------------------------------------------------------===#


def check_arguments_arity(
    arity: Int,
    args: PythonObject,
) raises:
    """Validate that the provided arguments match the expected function arity.

    This function checks if the number of arguments in the provided tuple object
    matches the expected arity for a function call. If the counts don't match,
    it raises a descriptive error message similar to Python's built-in TypeError
    messages.

    Args:
        arity: The expected number of arguments for the function.
        args: A tuple containing the actual arguments passed to the function.

    Raises:
        If the argument count doesn't match the expected arity. The error
               message follows Python's convention for `TypeError` messages,
               indicating whether too few or too many arguments were provided.
    """
    # This overload (and the `(arity, arg_count)` overload below) exists
    # because the generic dispatch templates in `_python_func.mojo`
    # (`_dispatch_kwargs`, `_dispatch_fast`) don't have access to the
    # registered function name. The wrappers that invoke them
    # (`_py_kwargs_function_wrapper`, `_py_function_fastcall_wrapper`) decay to
    # `thin -> ...` C function pointers for CPython's `PyMethodDef`
    # table, which means they cannot capture any runtime state - so a
    # runtime `func_name: StringSlice` cannot be threaded down to the
    # dispatch site. Threading it as a `comptime` parameter would force
    # `func_name` to be `comptime` on `def_function` / `def_method` /
    # `def_staticmethod`, a breaking change for thousands of callers.
    # The fallback name keeps the existing error-message shape; remove
    # this overload only if/when the language gains runtime-string-to-
    # `comptime` promotion (or the wrapper closures gain a way to
    # smuggle state through CPython's call protocol).
    return check_arguments_arity(arity, args, "<mojo function>")


def check_arguments_arity(
    arity: Int,
    args: PythonObject,
    func_name: StringSlice,
) raises:
    """Validate that the provided arguments match the expected function arity.

    This function checks if the number of arguments in the provided tuple object
    matches the expected arity for a function call. If the counts don't match,
    it raises a descriptive error message similar to Python's built-in TypeError
    messages.

    Args:
        arity: The expected number of arguments for the function.
        args: A tuple containing the actual arguments passed to the function.
        func_name: The name of the function being called, used in error messages
                  to provide better debugging information.

    Raises:
        If the argument count doesn't match the expected arity. The error
               message follows Python's convention for TypeError messages,
               indicating whether too few or too many arguments were provided,
               along with the specific function name.
    """

    return check_arguments_arity(arity, len(args), func_name)


def check_arguments_arity(
    arity: Int,
    arg_count: Int,
) raises:
    """Validate that the provided argument count matches the expected arity.

    Fastcall-friendly overload: takes the already-known number of positional
    arguments rather than computing it from a tuple object. Used by the
    `METH_FASTCALL` dispatch path, which receives `nargs: Py_ssize_t`
    directly from CPython and never materializes a tuple.

    Args:
        arity: The expected number of arguments for the function.
        arg_count: The actual number of arguments passed to the function.

    Raises:
        If `arg_count` differs from `arity`. The error message follows
        Python's `TypeError` convention.
    """
    # See the `(arity, args: PythonObject)` overload above for why this
    # no-`func_name` form exists - same `thin` C-function-pointer
    # constraint applies to the METH_FASTCALL dispatch path.
    return check_arguments_arity(arity, arg_count, "<mojo function>")


def check_arguments_arity(
    arity: Int,
    arg_count: Int,
    func_name: StringSlice,
) raises:
    """Validate that the provided argument count matches the expected arity.

    Fastcall-friendly overload that accepts a precomputed arg count and a
    function name. See the `arg_count`-only overload for the rationale.

    Args:
        arity: The expected number of arguments for the function.
        arg_count: The actual number of arguments passed to the function.
        func_name: The name of the function being called, used in error
            messages.

    Raises:
        If `arg_count` differs from `arity`.
    """
    # The error messages raised below are intended to be similar to the
    # equivalent errors in Python.
    if arg_count != arity:
        if arg_count < arity:
            var missing_arg_count = arity - arg_count

            raise Error(
                "TypeError: ",
                func_name,
                "() missing ",
                missing_arg_count,
                " required positional ",
                _pluralize(missing_arg_count, "argument", "arguments"),
            )
        else:
            raise Error(
                "TypeError: ",
                func_name,
                "() takes ",
                arity,
                " positional ",
                _pluralize(arity, "argument", "arguments"),
                " but ",
                arg_count,
                " were given",
            )


def check_and_get_arg[
    T: Deinitable
](func_name: StaticString, py_args: PythonObject, index: Int) raises -> Pointer[
    T, MutUnsafeAnyOrigin
]:
    """Get the argument at the given index and downcast it to a given Mojo type.

    Parameters:
        T: The Mojo type to downcast the argument to.

    Args:
        func_name: The name of the function referenced in the error message if
            the downcast fails.
        py_args: The Python tuple object containing the arguments.
        index: The index of the argument.

    Returns:
        A pointer to the Mojo value contained in the argument.

    Raises:
        If the argument cannot be downcast to the given type.
    """
    return py_args[index].downcast_value_ptr[T](func=func_name)


def _try_convert_arg[
    T: ConvertibleFromPython
](
    func_name: StringSlice, py_args: PythonObject, argidx: Int, out result: T
) raises:
    try:
        result = T(py=py_args[argidx])
    except convert_err:
        raise Error(
            "TypeError: ",
            func_name,
            "() expected argument at position ",
            argidx,
            " to be instance of (or convertible to) Mojo '",
            reflect[T].name(),
            "'; got '",
            _get_type_name(py_args[argidx]),
            "'. (Note: attempted conversion failed due to: ",
            convert_err,
            ")",
        )


# NOTE:
#   @always_inline is needed so that the unsafe_stack_allocation() that
#   appears in the definition below is valid in the _callers_ stack frame,
#   effectively allowing us to "return" a pointer to stack-allocated data
#   from this function.
@always_inline
def check_and_get_or_convert_arg[
    T: ConvertibleFromPython
](func_name: StaticString, py_args: PythonObject, index: Int) raises -> Pointer[
    T, MutUnsafeAnyOrigin
]:
    """Get the argument at the given index and convert it to a given Mojo type.

    If the argument cannot be directly downcast to the given type, it will be
    converted to it.

    Parameters:
        T: The Mojo type to downcast or convert the argument to.

    Args:
        func_name: The name of the function referenced in the error message if
            the downcast fails.
        py_args: The Python tuple object containing the arguments.
        index: The index of the argument.

    Returns:
        A pointer to the Mojo value contained in or converted from the argument.

    Raises:
        If the argument cannot be downcast or converted to the given type.
    """

    # Stack space to hold a converted value for this argument, if needed.
    var converted_arg_ptr = unsafe_stack_allocation[
        1, T
    ]().as_unsafe_any_origin()

    try:
        return check_and_get_arg[T](func_name, py_args, index)
    except e:
        converted_arg_ptr.unsafe_write(
            _try_convert_arg[T](
                func_name,
                py_args,
                index,
            )
        )
        # Return a pointer to stack data. Only valid because this function is
        # @always_inline.
        return converted_arg_ptr


def _get_type_name(obj: PythonObject) raises -> String:
    ref cpython = Python().cpython()

    var actual_type = cpython.Py_TYPE(obj._obj_ptr)
    var actual_type_name = PythonObject(
        from_owned=cpython.PyType_GetName(actual_type)
    )

    return String(actual_type_name)


def _pluralize(
    count: Int,
    singular: StaticString,
    plural: StaticString,
) -> StaticString:
    return singular if count == 1 else plural
