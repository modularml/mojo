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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import enum
from collections.abc import Callable, Sequence
from typing import overload

import max._core
import max._core.dialects.builtin
from max.mlir import Context, Location

# C++ overloads on different int types look the same in Python, ignore these
# mypy: disable-error-code="overload-cannot-match"

DiagnosticHandler = Callable

class ArgConvention(enum.Enum):
    imm = 0

    imm_mem = 1

    owned = 2

    owned_in_mem = 3

    deinit_mem = 4

    mut = 5

    ref = 6

    mutref = 7

    byref_result = 8

    byref_error = 9

class ArgConventionAttr(max._core.Attribute):
    def __init__(self, arg0: Context, arg1: ArgConvention, /) -> None: ...
    @property
    def value(self) -> ArgConvention: ...

class FnEffects(enum.Enum):
    none = 0

    throws = 1

    async_ = 2

    capturing = 4

    refresult = 32

    cabi = 512

class POC(enum.Enum):
    add = 0

    mul = 1

    mul_no_wrap = 2

    and_ = 3

    or_ = 4

    xor = 5

    max = 6

    min = 7

    shl = 8

    shr = 9

    div = 10

    mod = 11

    eq = 12

    lt = 13

    le = 14

    cond = 16

    current_target = 17

    target_has_feature = 18

    target_get_field = 19

    accelerator_arch = 20

    cross_compilation = 21

    get_env = 22

    get_sizeof = 23

    get_alignof = 24

    apply = 25

    apply_result_slot = 26

    rebind = 27

    ptr_bitcast = 34

    load_from_mem = 35

    variadic_ptr_map = 36

    variadic_ptrremove_map = 37

    attr_to_str = 39

    data_to_str = 40

    string_address = 41

    str_concat = 42

    function_get_arg_types = 43

    div_s = 44

    div_u = 45

    ceil_div_s = 46

    ceil_div_u = 47

    floor_div_s = 48

    rem_s = 49

    rem_u = 50

class POCAttr(max._core.Attribute):
    def __init__(self, arg0: Context, arg1: POC, /) -> None: ...
    @property
    def value(self) -> POC: ...

class PassingKind(enum.Enum):
    pos_or_kw = 0

    pos = 1

    kw = 2

    implicit = 3

    inferred = 4

class VariadicKind(enum.Enum):
    not_vararg = 0

    pos_vararg = 1

    pack_vararg = 2

    kw_vararg = 3

class CastFromBuiltinAttr(max._core.Attribute):
    """
    The `#kgen.cast_from_builtin` attribute converts a builtin MLIR type to a
    POP type.
    """

    @overload
    def __init__(self, arg: max._core.dialects.builtin.TypedAttr) -> None: ...
    @overload
    def __init__(
        self, arg: max._core.dialects.builtin.TypedAttr, type: SIMDType
    ) -> None: ...
    @overload
    def __init__(
        self, arg: max._core.dialects.builtin.TypedAttr, type: SIMDType
    ) -> None: ...
    @property
    def arg(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def type(self) -> SIMDType: ...

class CastToBuiltinAttr(max._core.Attribute):
    """
    The `#kgen.cast_to_builtin` attribute converts a POP type to a builtin MLIR
    type.
    """

    @overload
    def __init__(self, arg: max._core.dialects.builtin.TypedAttr) -> None: ...
    @overload
    def __init__(
        self, arg: max._core.dialects.builtin.TypedAttr, type: max._core.Type
    ) -> None: ...
    @overload
    def __init__(
        self, arg: max._core.dialects.builtin.TypedAttr, type: max._core.Type
    ) -> None: ...
    @property
    def arg(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def type(self) -> max._core.Type | None: ...

class ConstraintAttr(max._core.Attribute):
    """
    The `#kgen.constraint` attribute represents a proposition that should hold
    and a location for where this constraint was declared, which is useful for
    error reporting.

    The proposition is an i1-typed parameter expression.

    The optional message is a user-provided string (from
    `where (cond, "message")` syntax) that is surfaced in the diagnostic when
    the constraint fails. It is null for constraints without a user message
    (including all compiler-synthesized constraints).

    Example:

    ```mlir
    #kgen.constraint<1, loc("file.mojo":10:5)>
    #kgen.constraint<1, loc("file.mojo":10:5), "must be positive">
    ```
    """

    @overload
    def __init__(
        self,
        proposition: max._core.dialects.builtin.TypedAttr,
        loc: max._core.LocationAttr,
        message: max._core.dialects.builtin.StringAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        proposition: max._core.dialects.builtin.TypedAttr,
        loc: max._core.LocationAttr,
        message: max._core.dialects.builtin.StringAttr,
    ) -> None: ...
    @property
    def proposition(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def loc(self) -> max._core.LocationAttr: ...
    @property
    def message(self) -> max._core.dialects.builtin.StringAttr: ...

class ParamDeclArrayAttr(max._core.Attribute):
    @overload
    def __init__(self, param_decl: ParamDeclAttr) -> None: ...
    @overload
    def __init__(self, value: Sequence[ParamDeclAttr]) -> None: ...
    @property
    def value(self) -> Sequence[ParamDeclAttr]: ...

class ParamDeclAttr(max._core.Attribute):
    """
    This is a declaration of a parameter in the meta-programming domain for
    generators and related infrastructure.  These are typically owned by
    kgen.generator instances or other things that produce new attributes.
    """

    @overload
    def __init__(self, ref: ParamDeclRefAttr) -> None: ...
    @overload
    def __init__(
        self, name: max._core.dialects.builtin.StringAttr, type: max._core.Type
    ) -> None: ...
    @overload
    def __init__(self, name: str, type: max._core.Type) -> None: ...
    @overload
    def __init__(
        self, name: max._core.dialects.builtin.StringAttr, type: max._core.Type
    ) -> None: ...
    @property
    def name(self) -> max._core.dialects.builtin.StringAttr: ...
    @property
    def type(self) -> max._core.Type | None: ...

class ParamDeclRefAttr(max._core.Attribute):
    """
    The `#kgen.param.decl.ref` attribute is a typed attribute that represents
    a reference to a declared parameter. It contains the type of the referenced
    parameter and its name.

    Example:

    ```mlir
    // A reference to parameter "p" with type "i1".
    #kgen.param.decl.ref<"p"> : i1
    ```

    There are special rules governing when these can appear, or when
    ParamIndexRefAttr must be used instead, see DCRTODS.
    """

    @overload
    def __init__(self, decl: ParamDeclAttr) -> None: ...
    @overload
    def __init__(
        self, name: max._core.dialects.builtin.StringAttr, type: max._core.Type
    ) -> None: ...
    @overload
    def __init__(self, name: str, type: max._core.Type) -> None: ...
    @overload
    def __init__(
        self, name: max._core.dialects.builtin.StringAttr, type: max._core.Type
    ) -> None: ...
    @property
    def name(self) -> max._core.dialects.builtin.StringAttr: ...
    @property
    def type(self) -> max._core.Type | None: ...

class ParamIndexRefAttr(max._core.Attribute):
    """
    The `#kgen.param.index.ref` attribute is a reference to an input
    parameter of an enclosing signature. This attribute can only be used inside
    a `GeneratorType`. This attribute contains two fields:

    - depth: Which containing signature contains the parameter we're referring
      to. Non-negative integer. Zero means the nearest containing signature, one
      means the signature containing that one, etc.
      Note they cannot refer to any op's parameter-decls, and you cannot always
      use a depth to refer to surrounding scopes, see DCRTODS.
    - index: index of the parameter decl in that signature (non-negative
      integer).

    Example:

    ```mlir
    // Second input parameter of the nearest signature.
    #kgen.param.index.ref<0, 1> : index
    // First input parameter of next enclosing signature.
    #kgen.param.index.ref<1, 0> : !lit.struct<@Int>
    ```

    The latter would appear in something like this:

    ```
    comptime bar: def[
      D: DType,
      N: Int,
      f: def[Y: AnyType](Y, SIMD[N, D])->None
    ](...) = ...
    ```

    The `SIMD[N, D]`'s `N` is a #kgen.param.index.ref<1, 0> : !lit.struct<@Int>.

    But, per DCRTODS, it can NOT be used inside a generator's body to refer to
    one of the generator's parameters, like this:

    ```
    def foo[X: AnyType](x: X):
        comptime zork: def[...(
          # Cannot have: #kgen.param.index.ref<1, 0> : !lit.struct<@Int>
        )->None = ...
    ```

    These depths must be carefully handled and adjusted when dealing with
    multiple signatures or scopes, see STCHDDDOS.
    """

    @overload
    def __init__(self, index: int, type: max._core.Type) -> None: ...
    @overload
    def __init__(
        self, depth: int, index: int, type: max._core.Type
    ) -> None: ...
    @overload
    def __init__(
        self, depth: int, index: int, type: max._core.Type
    ) -> None: ...
    @property
    def depth(self) -> int: ...
    @property
    def index(self) -> int: ...
    @property
    def type(self) -> max._core.Type | None: ...

class ParamOperatorAttr(max._core.Attribute):
    @overload
    def __init__(
        self,
        opcode: POC,
        operands: Sequence[max._core.dialects.builtin.TypedAttr],
    ) -> None: ...
    @overload
    def __init__(
        self,
        opcode: POC,
        operands: Sequence[max._core.dialects.builtin.TypedAttr],
        type: max._core.Type,
    ) -> None: ...
    @overload
    def __init__(
        self,
        opcode: POC,
        lhs: max._core.dialects.builtin.TypedAttr,
        rhs: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        opcode: POC,
        operands: Sequence[max._core.dialects.builtin.TypedAttr],
        type: max._core.Type,
    ) -> None: ...
    @property
    def opcode(self) -> POC: ...
    @property
    def operands(self) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...
    @property
    def type(self) -> max._core.Type | None: ...

class ParameterExprArrayAttr(max._core.Attribute):
    def __init__(
        self, value: Sequence[max._core.dialects.builtin.TypedAttr]
    ) -> None: ...
    @property
    def value(self) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...

class PogListAttr(max._core.Attribute):
    """
    The `#kgen.pog_list` attribute contains metadata about an argument or
    parameter list of a function, including names, passing kinds, default
    values, and information about variadic arguments.
    The positional default values correspond to the trailing positional
    (i.e. pos-only or pos-or-kw) args/params, and the keyword-only default
    values similarly correspond to the trailing keyword-only args/params.

    This attribute is the `metadata` field of `GeneratorType`.

    Example:

    ```mlir
    #kgen.pog_list<
      ["a", "b", "c", "d"],
      [pos, pos_or_kw, kw, kw],
    >
    ```

    The `origVariadicConvention` indicates whether the original argument
    convention of a VariadicList or VariadicPack, e.g. "mut *args: Int".

    Optional `bodyConstraints` holds constraints that are enforced by the body
    of the generator type that carries this list as metadata.
    """

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, num_pogs: int) -> None: ...
    @overload
    def __init__(self, pogs: Sequence[PogMetadataAttr]) -> None: ...
    @overload
    def __init__(
        self, num_pogs: int, body_constraints: Sequence[ConstraintAttr]
    ) -> None: ...
    @overload
    def __init__(
        self,
        names: Sequence[max._core.dialects.builtin.StringAttr],
        passing_kinds: Sequence[PassingKind],
    ) -> None: ...
    @overload
    def __init__(
        self,
        pogs: Sequence[PogMetadataAttr],
        body_constraints: Sequence[ConstraintAttr],
        orig_variadic_convention: ArgConvention,
    ) -> None: ...
    @overload
    def __init__(
        self,
        names: Sequence[max._core.dialects.builtin.StringAttr],
        passing_kinds: Sequence[PassingKind],
        variadics: Sequence[VariadicKind],
        defaults: Sequence[max._core.dialects.builtin.TypedAttr],
        orig_variadic_convention: ArgConvention | None,
        body_constraints: Sequence[ConstraintAttr],
    ) -> None: ...
    @property
    def pogs(self) -> Sequence[PogMetadataAttr]: ...
    @property
    def body_constraints(self) -> Sequence[ConstraintAttr]: ...
    @property
    def orig_variadic_convention(self) -> ArgConvention: ...

class PogMetadataAttr(max._core.Attribute):
    """
    The `#kgen.pog_metadata` attribute contains metadata about an argument or
    parameter of a function, including the name, passing kind, variadicness, and
    the default value (if present).

    Example:

    ```mlir
    #kgen.pog_metadata<"some_keyword_param", pos_or_kw, false, 42>
    #kgen.pog_metadata<"some_variadic_param", pos_or_kw, true>
    ```
    """

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(
        self,
        name: max._core.dialects.builtin.StringAttr,
        passing_kind: PassingKind,
        variadic: VariadicKind = VariadicKind.not_vararg,
        default_value: max._core.dialects.builtin.TypedAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        name: max._core.dialects.builtin.StringAttr,
        passing_kind: PassingKind,
        variadic: VariadicKind,
        default_value: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def name(self) -> max._core.dialects.builtin.StringAttr: ...
    @property
    def passing_kind(self) -> PassingKind: ...
    @property
    def variadic(self) -> VariadicKind: ...
    @property
    def default_value(self) -> max._core.dialects.builtin.TypedAttr: ...

class SIMDAttr(max._core.Attribute):
    """
    The `#kgen.simd` attribute represents a constant SIMD vector value. It
    contains `N` values of a particular dtype. Only integer, floating point, and
    bool dtypes are supported.

    Example:

    ```mlir
    #kgen.simd<1, 2> : !kgen.simd<2, si32>
    #kgen.simd<1.5, 2.5> : !kgen.simd<2, f64>
    #kgen.simd<true, false> : !kgen.simd<2, bool>
    ```

    When all values of the SIMD vector are equal, the attribute has a special
    splat syntax:

    ```mlir
    #kgen<simd 0> : !kgen.simd<4, si32>
    #kgen<simd "1.5"> : !kgen.simd<4, f32>
    #kgen<simd false> : !kgen.simd<4, bool>
    ```
    """

    @overload
    def __init__(
        self, values: Sequence[_DTypeValue], type: SIMDType
    ) -> None: ...
    @overload
    def __init__(self, value: _DTypeValue, type: SIMDType) -> None: ...
    @overload
    def __init__(self, int_val: int, type: SIMDType) -> None: ...
    @overload
    def __init__(
        self, values: Sequence[_DTypeValue], type: SIMDType
    ) -> None: ...
    @property
    def values(self) -> Sequence[_DTypeValue]: ...
    @property
    def type(self) -> SIMDType: ...

class ParamDeclareOp(max._core.Operation):
    """
    The `kgen.param.declare` operation declares a single parameter and binds
    its value to a parameter expression. The parameter is visible within and
    below the scope of the enclosing `DeclInterface` operation.

    Example:

    ```mlir
    kgen.param.declare A = <5>
    kgen.param.declare A_plus_one = <add(A, 1)>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        param_decl: ParamDeclAttr,
        value: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def param_decl(self) -> ParamDeclAttr: ...
    @param_decl.setter
    def param_decl(self, arg: ParamDeclAttr, /) -> None: ...
    @property
    def value(self) -> max._core.dialects.builtin.TypedAttr: ...
    @value.setter
    def value(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...

class GeneratorType(max._core.Type):
    """
    This type describes a generator (i.e. a parameterized value). It describes
    the generator's parameter signature (the types of input parameters), and the
    generator's output type (potentially in terms of the input parameters).
    """

    @overload
    def __init__(
        self,
        input_param_types: Sequence[max._core.Type],
        body: max._core.Type,
        param_list_attrs: max._core.Attribute = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        input_param_types: Sequence[max._core.Type],
        body: max._core.Type,
        param_list_attrs: PogListAttr,
    ) -> None: ...
    @property
    def input_param_types(self) -> Sequence[max._core.Type]: ...
    @property
    def body(self) -> max._core.Type | None: ...
    @property
    def param_list_attrs(self) -> PogListAttr: ...

class SIMDType(max._core.Type):
    """This type is parameterized with a size and a !kgen.dtype type."""

    @overload
    def __init__(
        self,
        size: max._core.dialects.builtin.TypedAttr,
        dtype: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @overload
    def __init__(
        self, size: int, dtype: max._core.dialects.builtin.TypedAttr
    ) -> None: ...
    @overload
    def __init__(
        self, size: max._core.dialects.builtin.TypedAttr, dtype: _KGENDType
    ) -> None: ...
    @overload
    def __init__(
        self,
        size: max._core.dialects.builtin.TypedAttr,
        d_type: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @overload
    def __init__(self, size: int, dtype: _KGENDType) -> None: ...
    @property
    def size(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def d_type(self) -> max._core.dialects.builtin.TypedAttr: ...

class FuncTypeGeneratorType(GeneratorType):
    def __init__(
        self,
        input_param_types: Sequence[max._core.Type],
        fn_type: max._core.dialects.builtin.FunctionType,
        arg_convs: Sequence[ArgConvention] = [],
        effects: FnEffects = FnEffects.none,
        fn_metadata: max._core.Attribute = ...,
        gen_metadata: max._core.Attribute = ...,
        arg_list_attrs: max._core.Attribute = ...,
    ) -> None: ...

class _KGENDType:
    @staticmethod
    def get_int(arg0: int, arg1: bool, /) -> _KGENDType: ...

class _DTypeValue: ...
class ParamDefValue: ...
class ParameterEvaluator: ...
