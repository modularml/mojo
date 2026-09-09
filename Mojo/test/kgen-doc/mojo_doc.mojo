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

"""
This is a module summary, that
spills over to the next line."""

# RUN: kgen-doc %s | FileCheck %s

# Check that no diagnostics are output:
# RUN: kgen-doc %s 2>&1 | FileCheck %s --allow-empty --check-prefix CHECK-DIAG
# CHECK-DIAG-NOT: warning

# """
# This is a module summary, that
# spills over to the next line."""

from layout.int_tuple import *
from std.sys.info import is_nvidia_gpu
from std.memory import Pointer
from std.builtin.simd import Float4_e2m1fn, Float8_e4m3fn


# CHECK:  "aliases": [
# CHECK:   "kind": "alias",
# CHECK:   "name": "alias_Type",
# CHECK:   "path": "/mojo_doc/#alias_type",
# CHECK:   "summary": "An example alias of a Type.",
# CHECK:   "value": "Int"
comptime alias_Type = Int
"""An example alias of a Type."""

# CHECK:  "kind": "alias",
# CHECK:  "name": "alias_Value",
# CHECK:  "path": "/mojo_doc/#alias_value",
# CHECK:  "summary": "An example alias of a Value.",
# CHECK:  "value": "10"
comptime alias_Value = 10
"""An example alias of a Value."""


# CHECK:  "kind": "alias",
# CHECK:  "name": "alias_construct",
# CHECK:  "path": "/mojo_doc/#alias_construct",
# CHECK:  "value": "IntTuple(Int(0), Int(1), Int(2), Int(3), Int(4))"
comptime alias_construct = IntTuple(0, 1, 2, 3, 4)


# CHECK:  "kind": "alias",
# CHECK:  "name": "alias_cond",
# CHECK:  "path": "/mojo_doc/#alias_cond",
# CHECK:  "value": "Int(2) if is_nvidia_gpu() else Int(1)"
comptime alias_cond = 2 if is_nvidia_gpu() else 1

# CHECK:  "kind": "alias",
# CHECK:  "name": "alias_fn",
# CHECK:  "path": "/mojo_doc/#alias_fn",
# CHECK:  "value": "def(Int, Int) thin -> None"
comptime alias_fn = def(Int, Int) thin -> None


# CHECK:  "kind": "alias",
# CHECK:  "name": "alias_str",
# CHECK:  "path": "/mojo_doc/#alias_str",
# CHECK:  "value": "\"\""
comptime alias_str = ""


# CHECK:  "deprecated": "deprecated alias",
# CHECK:  "kind": "alias",
# CHECK:  "name": "deprecated_alias",
# CHECK:  "path": "/mojo_doc/#deprecated_alias",
@deprecated("deprecated alias")
comptime deprecated_alias = 1

# Issue #5361: kgen-doc crashes on alias of parametrized function with origin

# CHECK-LABEL: "name": "parametric_ref_origin_alias",
# CHECK: "value": "fn_literal"
comptime parametric_ref_origin_alias = parametric_ref_origin[2]


# CHECK:  "description": "",
# CHECK:  "functions": [
# CHECK:  "name": "empty_fn",
# CHECK:  "overloads": [
# CHECK:      "description": "The is some kind of description."
# CHECK:      "signature": "def empty_fn()"
# CHECK:      "summary": "This is a function summary."
# CHECK:  ]
def empty_fn():
    """This is a function summary.

    The is some kind of description.
    """
    return


# CHECK-NOT:  fn_hidden


@doc_hidden
def fn_hidden() -> Int:
    """This is a function summary.

    The is some kind of description.

    Returns:
        An Int.
    """
    return 33


# CHECK-NOT: alias_hidden


@doc_hidden
comptime alias_hidden = 42


# CHECK-NOT: hidden_field


struct PublicStruct:
    @doc_hidden
    var hidden_field: Int


# CHECK-NOT: _private_member


comptime _private_member = ""


# CHECK-NOT: __double_underscore_private_member


comptime __double_underscore_private_member = ""


# CHECK-LABEL:  "name": "fn_that_async",
# CHECK:  "overloads":
# CHECK:      "async": true
# CHECK:      "returns": {
# CHECK:        "doc": "An Int.",
# CHECK:        "path": "/std/builtin/simd/#int",
# CHECK:        "type": "Int"
# CHECK:      },
# CHECK:      "signature": "def fn_that_async() -> Int"
# CHECK:      "summary": "This is a function summary."


async def fn_that_async() -> Int:
    """This is a function summary.

    The is some kind of description.

    Returns:
        An Int.
    """
    return 33


# CHECK:  "kind": "function",
# CHECK:  "name": "fn_that_raises",
# CHECK:  "overloads":
# CHECK:      "raises": true
# CHECK:      "raisesDoc": "Raises an exception when it wants to.\n"
# CHECK:      "returns": {
# CHECK:        "doc": "An Int.",
# CHECK:        "path": "/std/builtin/simd/#int",
# CHECK:        "type": "Int"
# CHECK:      },
# CHECK:      "signature": "def fn_that_raises() -> Int"
# CHECK:      "summary": "This is a function summary."


def fn_that_raises() raises -> Int:
    """This is a function summary.

    The is some kind of description.

    Raises:
        Raises an exception when it wants to.

    Returns:
        An Int.
    """
    return 33


# CHECK:  "kind": "function",
# CHECK:  "name": "fn_with_args",
# CHECK:  "overloads":
# CHECK:      "args":
# CHECK:          "description": "This is an argument."
# CHECK:          "name": "arg"
# CHECK:          "path": "/std/builtin/simd/#int"
# CHECK:          "type": "Int"
# CHECK:          "convention": "mut"
# CHECK:          "description": "This is an mut arg."
# CHECK:          "name": "inoutArg"
# CHECK:          "path": "/std/builtin/simd/#int"
# CHECK:          "type": "Int"
# CHECK:          "convention": "var"
# CHECK:          "description": "This is an owned arg."
# CHECK:          "name": "ownedArg"
# CHECK:          "path": "/std/builtin/simd/#int"
# CHECK:          "type": "Int"
# CHECK:          "convention": "imm"
# CHECK:          "description": "This is a borrowedArg."
# CHECK:          "name": "borrowedArg"
# CHECK:          "path": "/std/builtin/simd/#int"
# CHECK:          "type": "Int"
# CHECK:      "signature": "def fn_with_args(arg: Int, mut inoutArg: Int, var ownedArg: Int, borrowedArg: Int)",
# CHECK:      "summary": "This is a function summary."


def fn_with_args(
    arg: Int,
    mut inoutArg: Int,
    var ownedArg: Int,
    imm borrowedArg: Int,
):
    """This is a function summary.

    The is some kind of description.

    Args:
        arg: This is an argument.
        inoutArg: This is an mut arg.
        ownedArg: This is an owned arg.
        borrowedArg: This is a borrowedArg.
    """
    return


# CHECK-LABEL:  "name": "fn_with_overload",
# CHECK:  "overloads": [
# CHECK:      "signature": "def fn_with_overload()",
# CHECK:      "signature": "def fn_with_overload(arg: Int)",


def fn_with_overload():
    """This is a function summary.

    The is some kind of description.
    """
    return


def fn_with_overload(arg: Int):
    """This is a function summary.

    The is some kind of description.

    Args:
        arg: This is an argument.
    """
    return


# CHECK: "kind": "function",
# CHECK: "name": "fn_with_parameter_references",
# CHECK: "overloads":
# CHECK:     "signature": "def fn_with_parameter_references[arg1_type: TrivialRegisterPassable, arg2_type: TrivialRegisterPassable](func: {{.*}}, arg1: arg1_type, arg2: arg2_type)"


def fn_with_parameter_references[
    arg1_type: TrivialRegisterPassable,
    arg2_type: TrivialRegisterPassable,
](
    func: __mlir_type[`(`, +arg1_type, `,`, +arg2_type, `) -> ()`],
    arg1: arg1_type,
    arg2: arg2_type,
):
    pass


# CHECK:  "kind": "function",
# CHECK:  "name": "fn_with_params",
# CHECK:  "overloads":
# CHECK:      "parameters": [
# CHECK:          "description": "This is a parameter."
# CHECK:          "name": "param"
# CHECK:          "passingKind": "pos",
# CHECK:          "type": "__mlir_type.`!kgen.dtype`"
# CHECK:          "description": "This is a second parameter."
# CHECK:          "name": "param2"
# CHECK:          "passingKind": "kw",
# CHECK:          "type": "__mlir_type.`!kgen.dtype`"
# CHECK:      "signature": "def fn_with_params[param: __mlir_type.`!kgen.dtype`, /, *, param2: __mlir_type.`!kgen.dtype`]()"


def fn_with_params[
    param: __mlir_type.`!kgen.dtype`, /, *, param2: __mlir_type.`!kgen.dtype`
]():
    """This is a function summary.

    The is some kind of description.

    Parameters:
        param: This is a parameter.
        param2: This is a second parameter.
    """
    return


# CHECK: "kind": "function",
# CHECK: "name": "fn_with_params_and_return",
# CHECK: "overloads":
# CHECK:     "args":
# CHECK:         "description": "This is an argument."
# CHECK:         "name": "arg"
# CHECK:         "path": "/std/builtin/simd/#int"
# CHECK:         "type": "Int"
# CHECK:     "returns": {
# CHECK:       "doc": "This is a return value.",
# CHECK:       "path": "/std/builtin/simd/#int",
# CHECK:       "type": "Int"
# CHECK:     },
# CHECK:     "signature": "def fn_with_params_and_return(arg: Int) -> Int"


def fn_with_params_and_return(arg: Int) -> Int:
    """This is a function summary.

    The is some kind of description.

    Args:
        arg: This is an argument.

    Returns:
        This is a return value.
    """
    return arg


# CHECK: "kind": "function",
# CHECK: "name": "fn_with_fn_param_and_arg",
# CHECK: "overloads":
# CHECK:     "args":
# CHECK:         "name": "arg_fn"
# CHECK:         "type": "def(S, S) capturing thin -> Bool"
# CHECK:     "parameters":
# CHECK:         "name": "T"
# CHECK:         "path": "/std/traits/anytype/AnyType"
# CHECK:         "type": "AnyType"
# CHECK:         "name": "param_fn"
# CHECK:         "type": "def(T, T) capturing thin -> Bool"
# CHECK:         "default": "T"
# CHECK:         "name": "S"
# CHECK:         "path": "/std/traits/anytype/AnyType"
# CHECK:         "type": "AnyType"
# CHECK:     "returns": {
# CHECK:       "type": "S"
# CHECK:     },
# CHECK:     "signature": "def fn_with_fn_param_and_arg[T: AnyType, param_fn: def(T, T) capturing thin -> Bool, S: AnyType = T](arg_fn: def(S, S) capturing thin -> Bool) -> S"


def fn_with_fn_param_and_arg[
    T: AnyType,
    param_fn: def(T, T) capturing[_] -> Bool,
    S: AnyType = T,
](arg_fn: def(S, S) capturing[_] -> Bool) -> S:
    pass


# CHECK: "kind": "function",
# CHECK: "name": "logsoftmax",
# CHECK:     "args":
# CHECK:         "name": "output"
# CHECK:         "type": "Pointer[Scalar[dtype]]"
# CHECK:     "parameters":
# CHECK:         "name": "origins"
# CHECK:         "type": "OriginSet"
# CHECK:         "name": "input_fn_1d"
# CHECK:         "type": "def[_simd_width: Int](Int) capturing thin -> SIMD[dtype, _simd_width]"


def logsoftmax[
    simd_width: Int,
    buffer_size: Int,
    dtype: DType,
    origins: OriginSet,
    input_fn_1d: def[_simd_width: Int](Int) capturing[origins] -> SIMD[
        dtype, _simd_width
    ],
](output: Pointer[Scalar[dtype], _]) raises:
    pass


# COM: Verify that closure parameter types render with their full signature,
# COM: not just the bare keyword.
# CHECK: "signature": "def vectorize_unified[func: def[width: Int](idx: Int) -> None](closure: func)",


def vectorize_unified[func: def[width: Int](idx: Int) -> None](closure: func):
    pass


# FIXME(MOCO-3730): sugar regression.
# CHECK: "signature": "def tile_and_unswitch[workgroup_function: def[width: Int, sw: Bool](Int, Int) capturing thin -> None, *tile_size_list: Int](offset: Int, upperbound: Int)",


def tile_and_unswitch[
    workgroup_function: Static1DTileUnswitchUnitFunc,
    *tile_size_list: Int,
](offset: Int, upperbound: Int):
    pass


comptime Static1DTileUnswitchUnitFunc = def[width: Int, sw: Bool](
    Int, Int
) capturing[_] -> None


@fieldwise_init
struct MyStruct[x: Int]:
    pass


# CHECK-LABEL: "name": "fn_with_implicit_params",
# CHECK: "args":
# CHECK: {
# CHECK:     "name": "arg",
# CHECK:     "path": "/mojo_doc/MyStruct",
# CHECK:     "type": "MyStruct"
# CHECK: }
# CHECK: "parameters":
# CHECK: {
# CHECK:     "description": "Explicitly declared function parameter.",
# CHECK:     "kind": "parameter",
# CHECK:     "name": "p",
# CHECK:     "path": "/std/builtin/simd/#int",
# CHECK:     "type": "Int"
# CHECK: }
# CHECK: "signature": "def fn_with_implicit_params[p: Int](arg: MyStruct)"


def fn_with_implicit_params[p: Int](arg: MyStruct):
    """
    An autoparameterized function with documentation.

    Parameters:
      p: Explicitly declared function parameter.

    Args:
      arg: An argument whose declared type induces an implicit parameter.
    """
    pass


# CHECK:  "name": "pos_only_print",
# CHECK:  "overloads":
# CHECK:      "args":
# CHECK:          "name": "x"
# CHECK:          "passingKind": "pos",
# CHECK:          "path": "/std/collections/string/string/String"
# CHECK:          "type": "String"

# CHECK:          "name": "sep"
# CHECK:          "passingKind": "pos_or_kw",
# CHECK:          "path": "/std/collections/string/string/String"
# CHECK:          "type": "String"
# CHECK:      "signature": "def pos_only_print(x: String, /, sep: String)",


def pos_only_print(x: String, /, sep: String):
    """Prints a String type.

    Args:
        x: The String to print.
        sep: The separator.
    """
    pass


# CHECK:  "name": "keyword_only_prod",
# CHECK:  "overloads":
# CHECK:      "args":
# CHECK:          "name": "a"
# CHECK:          "passingKind": "pos",
# CHECK:          "path": "/std/builtin/simd/#int"

# CHECK:          "name": "b"
# CHECK:          "passingKind": "pos",
# CHECK:          "path": "/std/builtin/simd/#int"

# CHECK:          "name": "offset"
# CHECK:          "passingKind": "kw",
# CHECK:          "path": "/std/builtin/simd/#int"
# CHECK:      "signature": "def keyword_only_prod(a: Int, b: Int, /, *, offset: Int)",


def keyword_only_prod(a: Int, b: Int, /, *, offset: Int):
    """Multiply and add an offset.

    Args:
        a: First factor.
        b: Second factor.
        offset: The offset to be added.
    """
    pass


# CHECK:  "name": "default_args_and_params",
# CHECK:  "overloads":
# CHECK:      "args":
# CHECK:          "default": "Int(2)",
# CHECK:          "name": "b"
# CHECK:          "path": "/std/builtin/simd/#int"

# CHECK:          "default": "Int(3)",
# CHECK:          "name": "c"
# CHECK:          "path": "/std/builtin/simd/#int"

# CHECK:      "parameters":
# CHECK:          "default": "Int(1)",
# CHECK:          "name": "a"
# CHECK:          "path": "/std/builtin/simd/#int"
# CHECK:      "signature": "def default_args_and_params[a: Int = Int(1)](b: Int = Int(2), /, *, c: Int = Int(3))",


def default_args_and_params[a: Int = 1](b: Int = 2, /, *, c: Int = 3):
    """Test default handling.

    Parameters:
        a: Param.

    Args:
        b: Arg.
        c: Arg.
    """
    pass


# CHECK: "name": "variadic_pack",
# CHECK: "overloads":
# CHECK:     "args":
# CHECK:         "name": "*vals",
# CHECK:         "passingKind": "pos_or_kw",
# CHECK:         "type": "*Ts.values"

# CHECK:     "parameters":
# CHECK:         "name": "*Ts",
# CHECK:         "passingKind": "pos_or_kw",
# CHECK:         "path": "/std/traits/anytype/AnyType",
# CHECK:         "type": "AnyType"

# CHECK:     "signature": "def variadic_pack[*Ts: AnyType](*vals: *Ts.values)",


def variadic_pack[*Ts: AnyType](*vals: *Ts):
    """Test variadic pack argument type printing.

    Parameters:
        Ts: The list of types.

    Args:
        vals: Variadic pack arguments.
    """
    pass


# CHECK: "name": "variadic_params_args",
# CHECK: "overloads":
# CHECK:     "args":
# CHECK:         "name": "*vals",
# CHECK:         "passingKind": "pos_or_kw",
# CHECK:         "path": "/std/builtin/simd/#int",
# CHECK:         "type": "Int"

# CHECK:         "name": "**kwargs",
# CHECK:         "passingKind": "kw",
# CHECK:         "path": "/std/collections/string/string/String",
# CHECK:         "type": "String"

# CHECK:     "parameters":
# CHECK:         "name": "*nums",
# CHECK:         "passingKind": "pos_or_kw",
# CHECK:         "type": "Int"

# CHECK:     "signature": "def variadic_params_args[*nums: Int](*vals: Int, *, var **kwargs: String)",


def variadic_params_args[*nums: Int](*vals: Int, var **kwargs: String):
    """Test variadic argument/parameter type printing.

    Parameters:
        nums: Variadic parameters.

    Args:
        vals: Variadic arguments.
        kwargs: Variadic keyword arguments.
    """
    pass


# CHECK: "name": "parameter_with_escaped_mlir_name",
# CHECK: "overloads":
# CHECK:     "args":
# CHECK:         "name": "value",
# CHECK:         "type": "type"

# CHECK:     "parameters":
# CHECK:         "kind": "parameter",
# CHECK:         "name": "type",
# CHECK:         "path": "/std/traits/anytype/AnyType",

# CHECK:     "signature": "def parameter_with_escaped_mlir_name[type: AnyType](value: type)",


def parameter_with_escaped_mlir_name[type: AnyType](value: type):
    pass


##===----------------------------------------------------------------------===##
# Ref args and results.
##===----------------------------------------------------------------------===##

# MOTO-516: [doc generation] Print 'ref' arguments and results in docgen

# CHECK-LABEL: "name": "fn_with_anon_refs",
# CHECK: "args": [
# CHECK-NEXT:     {
# CHECK-NEXT:       "convention": "ref",


# CHECK:     "signature": "def fn_with_anon_refs(ref ref_arg1: TrivialRegisterPassable) -> ref[ref_arg1] TrivialRegisterPassable"
def fn_with_anon_refs(
    ref ref_arg1: TrivialRegisterPassable,
) -> ref[ref_arg1] TrivialRegisterPassable:
    pass


# CHECK-LABEL: "name": "fn_with_named_refs",
# CHECK:     "signature": "def fn_with_named_refs[life: MutOrigin](ref[life] ref_arg1: TrivialRegisterPassable) -> ref[life] TrivialRegisterPassable",
def fn_with_named_refs[
    life: MutOrigin
](ref[life] ref_arg1: TrivialRegisterPassable) -> ref[
    origin_of(ref_arg1)
] TrivialRegisterPassable:
    pass


# MOTO-870: Improve doc gen of struct Origin parameters
# CHECK-LABEL: "name": "fn_with_origins",
# CHECK:     "signature": "def fn_with_origins[o1: Origin[mut=o1.mut], o2: MutOrigin](ref[o1] arg1: Int, ref[o2] arg2: Int) -> ref[o1] Int",
def fn_with_origins[
    o1: Origin, o2: Origin[mut=True]
](ref[o1] arg1: Int, ref[o2] arg2: Int) -> ref[arg1] Int:
    pass


# MOTO-870: Improve doc gen of struct Origin parameters
# CHECK-LABEL: "name": "fn_with_mult_result_origins",
# CHECK:     "signature": "def fn_with_mult_result_origins(ref arg1: Int, ref arg2: Int) -> ref[arg1, arg2] Int",
def fn_with_mult_result_origins(
    ref arg1: Int, ref arg2: Int
) -> ref[arg1, arg2] Int:
    pass


# CHECK-LABEL: "name": "fn_with_named_result",
# CHECK:     "signature": "def fn_with_named_result(a: Int, out res: String)",
def fn_with_named_result(a: Int, out res: String):
    res = ""


# CHECK: "kind": "function"
# CHECK: "overloads":
# CHECK:    "deprecated": "deprecated function"
# CHECK:    "name": "deprecated_function"
@deprecated("deprecated function")
def deprecated_function():
    pass


# MOTO-418: Improve AST type printing of `reversed` in API docs
# CHECK-LABEL: "name": "dep_type"
# CHECK: "args":
# CHECK:   "name": "value",
# CHECK:   "path": "/mojo_doc/UsesParameter",
# CHECK:   "type": "UsesParameter[K]"
# CHECK: "parameters":
# CHECK:   "name": "K",
# CHECK:   "path": "/std/traits/anytype/AnyType",
# CHECK:   "type": "AnyType"
# CHECK: "returns": {
# CHECK:   "type": "ref[value] UsesParameter[K]"
# CHECK: },
# CHECK: "signature": "def dep_type[K: AnyType](ref value: UsesParameter[K]) -> ref[value] UsesParameter[K]",
struct UsesParameter[A: AnyType]:
    pass


def dep_type[
    K: AnyType
](ref value: UsesParameter[K]) -> ref[value] UsesParameter[K]:
    return value


# Check that we dump optional default values correctly.
from std.collections.optional import Optional


# CHECK-LABEL: "name": "optional_default_arg_none"
# CHECK: "signature": "def optional_default_arg_none(input: Optional[Int64] = None)"
def optional_default_arg_none(input: Optional[Int64] = None):
    pass


# CHECK-LABEL: "name": "optional_default_arg_none2"
# CHECK: "signature": "def optional_default_arg_none2(input: Optional[SIMD[.int64, 4]] = None)"
def optional_default_arg_none2(input: Optional[SIMD[.int64, 4]] = None):
    pass


# CHECK-LABEL: "name": "optional_default_arg_13"
# CHECK: "signature": "def optional_default_arg_13(input: Optional[Int64] = Int64(13))"
def optional_default_arg_13(input: Optional[Int64] = Int64(13)):
    pass


# CHECK-LABEL: "name": "simd_scalar_alias"
# CHECK: "signature": "def simd_scalar_alias[dt: DType](input: Scalar[dt])"
def simd_scalar_alias[dt: DType](input: Scalar[dt]):
    pass


# CHECK-LABEL: "name": "simd_scalar_sugar"
# CHECK: "signature": "def simd_scalar_sugar(b: Int, c: Int8, d: UInt8, e: BFloat16, f: Float64, g: Float4_e2m1fn, h: Float8_e4m3fn)"
# Int & Bool are not SIMD scalars (yet). Once they are, they'll be added here.
def simd_scalar_sugar(
    b: Int,
    c: Int8,
    d: UInt8,
    e: BFloat16,
    f: Float64,
    g: Float4_e2m1fn,
    h: Float8_e4m3fn,
):
    pass


# CHECK-LABEL: "name": "parametric_ref_origin",
# CHECK: "signature": "def parametric_ref_origin[b: Int](ref c: Int)",
def parametric_ref_origin[b: Int](ref c: Int):
    pass


# ===----------------------------------------------------------------------=== #
# Struct documentation
# ===----------------------------------------------------------------------=== #

# CHECK:  "kind": "module",
# CHECK:  "name": "mojo_doc",

# CHECK-LABEL:  "structs": [


# MOTO-516: [doc generation] Print 'ref' arguments and results in docgen
# CHECK:  "convention": "register_passable_trivial",
struct HMyUnsafePointer[
    T: AnyType,
    address_space: AddressSpace = .GENERIC,
](TrivialRegisterPassable):
    # CHECK: "signature": "def __getitem__(self) -> ref[MutUnsafeAnyOrigin, address_space] T",
    def __getitem__(
        self,
    ) -> ref[MutAnyOrigin, Self.address_space] Self.T:
        pass

    # CHECK: "signature": "def address_of(ref[address_space] arg: T) -> Self",
    @staticmethod
    def address_of(ref[Self.address_space] arg: Self.T) -> Self:
        pass


# CHECK:  "signature": "struct HMyUnsafePointer[T: AnyType, address_space: AddressSpace = .GENERIC]",


struct HList[T: ImplicitlyCopyable]:
    # FIXME: self_is_mut is wrong.
    # CHECK: "signature": "def __getitem__(ref self, idx: Int) -> ref[self_is_mut] T",
    def __getitem__(ref self, idx: Int) -> ref[self] Self.T:
        pass


# FIXME(MOTO-692): This should say `T: ImplicitlyCopyable`.
# CHECK: "signature": "struct HList[T: ImplicitlyCopyable]",


# Check that we don't generate any synthesized thunk methods
# from the trait usage.
# CHECK-NOT: "name": "thunk_

# Check that special functions are ordered first, and with the correct
# prioritization (i.e. not just name based).
# CHECK:  "kind": "function",
# CHECK:  "name": "__init__",
# CHECK:     "signature": "def __init__(out self)",
# CHECK:  "signature": "def __init__(out self, *, copy: Self)",
# CHECK:  "name": "__deinit__",

# CHECK: "name": "__add__",
# CHECK: "overloads":
# CHECK:      "signature": "def __add__(self, other: Self) -> Self"

# CHECK:  "name": "__len__",

# CHECK:  "kind": "function",
# CHECK:  "name": "fn_with_by_conventions",
# CHECK:  "overloads"
# CHECK:      "args"
# CHECK:          "description": "This is a by-ref argument."
# CHECK:          "name": "arg"
# CHECK:          "type": "Self"
# CHECK:          "description": "This is a variadic argument."
# CHECK:          "name": "*args",
# CHECK:          "type": "Self"
# CHECK:      "constraints": "This describes the method's constraints.\n",
# CHECK:      "description": ""
# CHECK:      "returns": {
# CHECK:        "doc": "This is a by-ref return value.",
# CHECK:        "type": "Self"
# CHECK:      },
# CHECK:      "signature": "def fn_with_by_conventions(mut self, mut arg: Self, mut *args: Self) -> Self",
# CHECK:      "summary": "This is a function summary."
# CHECK:  "kind": "struct",
# CHECK:  "name": "InMemoryStruct",
# CHECK:  "parentTraits": [
# CHECK-NEXT:   {
# CHECK-NEXT:     "name": "AnyType",
# CHECK-NEXT:     "path": "/std/traits/anytype/AnyType"
# CHECK-NEXT:   },
# CHECK-NEXT:   {
# CHECK-NEXT:     "name": "Copyable",
# CHECK-NEXT:     "path": "/std/traits/copyable/Copyable"
# CHECK-NEXT:   },
# CHECK-NEXT:   {
# CHECK-NEXT:     "name": "Deinitable",
# CHECK-NEXT:     "path": "/std/traits/deinitable/Deinitable"
# CHECK-NEXT:   },
# CHECK-NEXT:   {
# CHECK-NEXT:     "name": "ImplicitlyCopyable",
# CHECK-NEXT:     "path": "/std/traits/copyable/ImplicitlyCopyable"
# CHECK-NEXT:   },
# CHECK-NEXT:   {
# CHECK-NEXT:     "name": "Movable",
# CHECK-NEXT:     "path": "/std/traits/movable/Movable"
# CHECK-NEXT:   },
# CHECK-NEXT:   {
# CHECK-NEXT:     "name": "Sized",
# CHECK-NEXT:     "path": "/std/builtin/len/Sized"
# CHECK-NEXT:   }
# CHECK:  "signature": "struct InMemoryStruct"


struct InMemoryStruct(ImplicitlyCopyable, Sized):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __deinit__(deinit self):
        pass

    def __add__(self, other: Self) -> Self:
        return other

    def __len__(self) -> Int:
        return 0

    def fn_with_by_conventions(
        mut self, mut arg: InMemoryStruct, mut *args: InMemoryStruct
    ) -> InMemoryStruct:
        """This is a function summary.

        Constraints:
            This describes the method's constraints.

        Args:
            arg: This is a by-ref argument.
            args: This is a variadic argument.

        Returns:
            This is a by-ref return value.
        """
        return arg


# CHECK:  "constraints": "This describes the struct's constraints.\n",
# CHECK:  "convention": "{{.*}}",
# CHECK:  "description": "The is some kind of description.\n",
# CHECK:      "kind": "function",
# CHECK:      "name": "fn_with_self_param",
# CHECK:          "parameters": [
# CHECK:              "description": "This is a Self parameter."
# CHECK:              "name": "param"
# CHECK:              "type": "Self"
# CHECK:          "signature": "def fn_with_self_param[param: Self](self)"
# CHECK:  "kind": "struct",
# CHECK:  "name": "ParameterClass",
# CHECK:  "parameters": [
# CHECK:      "description": "This is a parameter."
# CHECK:      "name": "_type"
# CHECK:      "type": "__mlir_type.`!kgen.dtype`"
# CHECK:  ],
# CHECK:  "signature": "struct ParameterClass[_type: __mlir_type.`!kgen.dtype`]",
# CHECK:  "summary": "This is a class summary."


struct ParameterClass[_type: __mlir_type.`!kgen.dtype`](RegisterPassable):
    """This is a class summary.

    The is some kind of description.

    Constraints:
        This describes the struct's constraints.

    Parameters:
        _type: This is a parameter.
    """

    def fn_with_self_param[param: ParameterClass[Self._type]](self):
        """A summary.

        Parameters:
            param: This is a Self parameter.
        """
        return


# CHECK:  "kind": "struct",
# CHECK:  "name": "StructWithDefault",
# CHECK:  "parameters":
# CHECK:      "default": "Int(1)",
# CHECK:      "name": "a",
# CHECK:      "path": "/std/builtin/simd/#int"


struct StructWithDefault[a: Int = 1]:
    pass


# Test that struct parameters with auto-parameterized types (like Scalar, which
# expands to SIMD[dtype, 1]) are printed with the full "a.dtype" form rather
# than just "dtype". This requires the DiagnosticDeclContextChanger to be active
# during type generation (not only during constraint generation).
# CHECK: "name": "StructWithAutoParamScalar",
# CHECK: "signature": "struct StructWithAutoParamScalar[a: Scalar[a.dtype]]"
@fieldwise_init
struct StructWithAutoParamScalar[a: Scalar]:
    """A struct with an auto-parameterized Scalar parameter.

    Parameters:
        a: A scalar value whose type induces an implicit dtype parameter.
    """

    pass


# CHECK:  "kind": "struct",
# CHECK:  "name": "StructWithFnParam",
# CHECK:  "parameters":
# CHECK:      "name": "T"
# CHECK:      "path": "/std/traits/anytype/AnyType"
# CHECK:      "type": "AnyType"
# CHECK:      "name": "param_fn"
# CHECK:      "type": "def(T, T) capturing thin -> Bool"
# CHECK:      "default": "T",
# CHECK:      "name": "S"
# CHECK:      "path": "/std/traits/anytype/AnyType"
# CHECK:      "type": "AnyType"
# CHECK: "signature": "struct StructWithFnParam[T: AnyType, param_fn: def(T, T) capturing thin -> Bool, S: AnyType = T]",


struct StructWithFnParam[
    T: AnyType,
    param_fn: def(T, T) capturing[_] -> Bool,
    S: AnyType = T,
]:
    pass


# CHECK-NOT: "name": "HiddenStruct"


@doc_hidden
struct HiddenStruct:
    """A struct that should be hidden from docs."""

    pass


# CHECK: "deprecated": "deprecated struct"
# CHECK: "name": "DeprecatedStruct"
@deprecated("deprecated struct")
struct DeprecatedStruct:
    pass


# CHECK:  "summary": "This is a module summary, that spills over to the next line."

##===----------------------------------------------------------------------===##
# Traits
##===----------------------------------------------------------------------===##

# CHECK:  "traits": [
# CHECK:    "description": "The is some kind of description.",
# CHECK:    "functions":
# CHECK:      "hasDefaultImplementation": false,
# CHECK:      "kind": "function",
# Check that we don't generate inherited methods (like __deinit__ from AnyType).
# CHECK-NOT: "name": "__deinit__"
# CHECK:      "name": "f",
# CHECK:      "summary": "This is a trait function doc."
# CHECK:      "hasDefaultImplementation": true,
# CHECK:      "name": "f_default",
# CHECK:    "kind": "trait",
# CHECK:    "name": "Trait",
# CHECK:    "summary": "This is a trait doc."


trait Trait:
    """This is a trait doc.

    The is some kind of description.
    """

    def f(self):
        """This is a trait function doc."""
        ...

    def f_default(self) -> Int:
        """A function with a default implementation."""
        return 0


# CHECK: "deprecated": "deprecated trait"
# CHECK: "name": "DeprecatedTrait"
@deprecated("deprecated trait")
trait DeprecatedTrait:
    pass


# Check that we include version information in the generated JSON.
# CHECK: "version":
