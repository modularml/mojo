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

# RUN: %parse-mojo-isolated %s | FileCheck %s
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values \
# RUN:     | FileCheck %s --check-prefix=INLINE


# CHECK-LABEL: lit.fn @"empty_def()"() -> !kgen.none
# CHECK: lit.end_fn
def empty_def():
    pass


# Thin function types whose only unbound parameters are singletons (origins,
# including those implied by a variadic pack) are valid runtime arguments.

# CHECK-LABEL: lit.fn @"pack_fn_as_arg
def pack_fn_as_arg[*Args: AnyType](func: def(*args: *Args) thin):
    pass


struct PackFnType[*Args: AnyType]:
    comptime type = def(*args: *Self.Args) thin


# CHECK-LABEL: lit.fn @"pack_fn_alias_as_arg
def pack_fn_alias_as_arg[*Args: AnyType](func: PackFnType[*Args].type):
    pass


# Explicit Origin parameters are singletons, so this is a valid runtime
# argument (unlike bare `def(ref arg: Int)`, which invents a Bool mutability
# parameter).

# CHECK-LABEL: lit.fn @"origin_fn_as_arg
def origin_fn_as_arg(func: def[a: MutOrigin](ref [a] arg: Int) thin -> None):
    pass


# CHECK-LABEL: lit.fn @"slash
# CHECK-SAME: (%a: !Int, |, %b: !Int)
def slash(a: Int, /, b: Int):
    pass


# CHECK-LABEL: lit.fn @"trailing_slash
# CHECK-SAME: (%a: !Int, |)
def trailing_slash(a: Int, /):
    pass


# CHECK-LABEL: lit.fn @"star
# CHECK-SAME: (%a: !Int, *, %b: !Int)
def star(a: Int, *, b: Int):
    pass


# CHECK-LABEL: lit.fn @"leading_star
# CHECK-SAME: (*, %a: !Int)
def leading_star(*, a: Int):
    pass


# CHECK-LABEL: lit.fn @"star_and_slash
# CHECK-LABEL: (%a: !Int, |, *, %b: !Int)
def star_and_slash(a: Int, /, *, b: Int):
    pass


# CHECK-LABEL: lit.fn @"star_and_slash_2
# CHECK-SAME: (%a: !Int, |, %b: !Int, *, %c: !Int)
def star_and_slash_2(a: Int, /, b: Int, *, c: Int):
    pass


# CHECK-LABEL: lit.fn @"default_args
# CHECK-SAME: (%a: !Int, %b: !Int = {:scalar<index> 8}, *, %c: !Int, %d: !Int = {:scalar<index> 9})
def default_args(a: Int, b: Int = 8, *, c: Int, d: Int = 9):
    pass


# CHECK-LABEL: lit.fn @"variadic_and_kw_only
# CHECK-SAME: (%a: !Int, %b: !Int, %args: !lit.ref<!lit.struct<#VariadicList{{.*}}> imm_mem|pos_vararg, *, %c: !Int, %d: !Int = {:scalar<index> 9})
def variadic_and_kw_only(
    a: Int, b: Int, *args: Int, c: Int, d: Int = 9
):
    pass


# CHECK-LABEL: lit.fn @"variadic_arg_after_default
# CHECK-SAME: (%a: !Int, %b: !Int = {:scalar<index> 0}, %args: !lit.ref<!lit.struct<#VariadicList{{.*}}> imm_mem|pos_vararg = :none *?,
# CHECK-SAME:  *, %c: !Int, %d: !Int = {:scalar<index> 1}, %kwargs: {{.*}}|kw_vararg = :none *?)
def variadic_arg_after_default(
    a: Int,
    b: Int = 0,
    *args: Int,
    c: Int,
    d: Int = 1,
    var **kwargs: Int,
):
    pass


# CHECK-LABEL: lit.fn @"variadic_param_after_default
# CHECK-SAME: a: !Int, b: !Int = {:scalar<index> 0}, args: {{.*}} pos_vararg = :none *?, *, c: !Int, d: !Int = {:scalar<index> 1}>()
def variadic_param_after_default[
    a: Int, b: Int = 0, *args: Int, c: Int, d: Int = 1
]():
    pass


# CHECK-LABEL: lit.fn @"inferred_params
# CHECK-SAME: <x: !Int, y: !Int, +>
def inferred_params[x: Int, y: Int, //]():
    # CHECK-NEXT: !lit.generator<<"x": !Int, "y": !Int, +>() -> !kgen.none> = <@
    comptime def_type: def[x: Int, y: Int, //] () thin -> None = inferred_params


# CHECK-LABEL: lit.fn @"inferred_params_regular
# CHECK-SAME: <x: !Int, +, y: !Int>
def inferred_params_regular[x: Int, //, y: Int]():
    # CHECK-NEXT: !lit.generator<<"x": !Int, +, "y": !Int>() -> !kgen.none> = <@
    comptime def_type: def[
        x: Int, //, y: Int
    ] () thin -> None = inferred_params_regular


# CHECK-LABEL: lit.fn @"inferred_params_pos_only
# CHECK-SAME: <x: !Int, +, y: !Int = {:scalar<index> 1}, |>
def inferred_params_pos_only[x: Int, //, y: Int = 1, /]():
    pass


# CHECK-LABEL: lit.fn @"inferred_params_kw_only
# CHECK-SAME: <x: !Int, +, *, y: !Int>
def inferred_params_kw_only[x: Int, //, *, y: Int]():
    pass


# ===----------------------------------------------------------------------=== #
# Test that def arguments are assignable and we get the right number of copies.
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct MemoryOnly(ImplicitlyCopyable):
    pass


@fieldwise_init
struct NonTrivialReg(ImplicitlyCopyable, RegisterPassable):
    pass


struct TypeWithParametricSelf(Movable where False):
    def method(ref self):
        pass


struct ValueWithTypeWithParametricSelf(Movable where False):
    var member: TypeWithParametricSelf


# CHECK-LABEL: test_def_arg_box_mbvalue
def test_def_arg_box_mbvalue(
    a: TypeWithParametricSelf, b: ValueWithTypeWithParametricSelf
) raises:
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}method{{.*}}(%a)
    # CHECK-NEXT: %xyz = lit.var.decl "xyz"
    # CHECK-NEXT: lit.ref.store [[TMP]], %xyz
    var xyz = a.method()

    # MOCO-715: failed to infer implicit parameter 'mut' of argument 'self' type 'Pointer
    # CHECK-NEXT: [[MEMBERREF:%.*]] = lit.ref.struct.ger %b[member]
    _ = b.member.method()


def returnsMultiple() -> Tuple[Int, MemoryOnly]:
    pass


# MOCO-687: Unable to destructure multiple outputs in a def function without explicit var declarations
# CHECK-LABEL: test_multi_tuple_def_value
def test_multi_tuple_def_value() raises:
    # CHECK: %a = lit.var.decl "a"
    # CHECK: %b = lit.var.decl "b"
    var a, b = returnsMultiple()


# CHECK-LABEL: lit.fn @"ref_result
def ref_result(mut x: MemoryOnly) -> ref [x] MemoryOnly:
    # CHECK-NEXT: lit.return %x : !lit.ref<!MemoryOnly, mut *"x`">
    return x


# CHECK-LABEL: lit.fn @"def_ref_result
def def_ref_result(mut x: MemoryOnly) raises -> ref [x] MemoryOnly:
    # CHECK-NEXT: lit.ref.store %x, %__result__
    # CHECK-NEXT: %simd = kgen.param.constant: scalar<bool> = <false>
    # CHECK-NEXT: lit.return %simd
    return x


# CHECK-LABEL: lit.fn @"use_ref_result
def use_ref_result() raises:
    # CHECK-NEXT: %a = lit.var.decl "a"
    # CHECK-NEXT: lit.call {{.*}}MemoryOnly::@"__init__{{.*}}(%a)
    var a = MemoryOnly()

    # CHECK-NEXT: [[REF:%.*]] = lit.call {{.*}}ref_result{{.*}}(%a)
    ref_result(a) = MemoryOnly()
    # CHECK-NEXT: lit.call {{.*}}MemoryOnly::@"__init__{{.*}}([[REF]])

    # CHECK-NEXT: %__call_result_tmp__ = lit.var.decl
    # CHECK-NEXT: lit.call {{.*}}decls::@"def_ref_result{{.*}}(%a, %__error__, %__call_result_tmp__)
    # CHECK-NEXT: [[REF:%.*]] = lit.load.consume %__call_result_tmp__
    # CHECK-NEXT: lit.call {{.*}}MemoryOnly::@"__init__{{.*}}([[REF]])
    def_ref_result(a) = MemoryOnly()


# CHECK-LABEL: lit.fn @"return_def_arg_box
def return_def_arg_box(abc: MemoryOnly) raises -> ref [abc] MemoryOnly:
    # CHECK-NEXT: lit.ref.store %abc, %__result__
    return abc


# CHECK-LABEL: lit.fn @"foldable_requires_1
# CHECK-SAME: {
# CHECK-SAME:  ne(:scalar<index> #lit.struct.extract<:!Int x, "_mlir_value">, 0))
def foldable_requires_1[x: Int]()
    where x:
        pass


# CHECK-LABEL: lit.fn @"foldable_requires_2
# CHECK-SAME: {
# CHECK-SAME:   ge(:scalar<index> #lit.struct.extract<:!Int y, "_mlir_value">, 11))
# CHECK-SAME:   lt(:scalar<index> #lit.struct.extract<:!Int x, "_mlir_value">, 1))
def foldable_requires_2[x: Int, y: Int]()
    where y > 10
    where x < 1:
        pass


# CHECK-LABEL: lit.fn @"foldable_requires_passthru
def foldable_requires_passthru[a: Int, b: Int]()
    where a > 10  # test comment
    where b < 1
    where b       # test another comment
    where a:      # test another comment
        foldable_requires_2[b, a]()
        foldable_requires_1[a]()
        foldable_requires_1[b]()


# CHECK-LABEL: lit.fn @"foldable_requires_param_if
def foldable_requires_param_if[a: Int, b: Int]():
    comptime if a > 10:
        comptime if b < 1:
            foldable_requires_2[b, a]()
    elif a < 1:
        comptime if b > 10:
            foldable_requires_2[a, b]()
    elif a:
        foldable_requires_1[a]()
    elif b:
        foldable_requires_1[b]()
    else:
        foldable_requires_1[42]()


struct FN_LITERAL_RET[x: Int, y: Int]():
    pass


def test_fn_literal_type[x: Int, y: Int]() -> FN_LITERAL_RET[x, y]:
    pass

# CHECK:      lit.alias.decl *"test_fn_literal_type_type{{.*}}": {{.*}} =
# CHECK-SAME: !kgen.func.literal<:!lit.fn<[1]({{.*}}!lit.struct<#RET <:!Int *(0,0), :!Int *(0,1)>>{{.*}}) -> !kgen.none>
# CHECK-SAME: #kgen.func.symbol<@{{.*}}::@"test_fn_literal_type[::SIMD[DType.int, 1],::SIMD[DType.int, 1]]()"<:!Int *(0,0), :!Int *(0,1)>>
comptime test_fn_literal_type_type = type_of(test_fn_literal_type)

# CHECK:      lit.alias.decl *"test_fn_literal_type_type_1{{.*}}": {{.*}} =
# CHECK-SAME: !kgen.func.literal<:!lit.fn<[1]({{.*}}!lit.struct<#RET <:!Int {:scalar<index> 1}, :!Int *(0,0)>>{{.*}}) -> !kgen.none>
# CHECK-SAME: #kgen.func.symbol<@{{.*}}::@"test_fn_literal_type[::SIMD[DType.int, 1],::SIMD[DType.int, 1]]()"<:!Int {:scalar<index> 1}, :!Int *(0,0)>>
comptime test_fn_literal_type_type_1 = type_of(test_fn_literal_type[1, _])

# CHECK:      lit.alias.decl *"test_fn_literal_type_type_2{{.*}}": {{.*}} =
# CHECK-SAME: !kgen.func.literal<:!lit.fn<[1]({{.*}}!lit.struct<#RET <:!Int {:scalar<index> 1}, :!Int {:scalar<index> 2}>>{{.*}}) -> !kgen.none>
# CHECK-SAME: #kgen.func.symbol<@{{.*}}::@"test_fn_literal_type[::SIMD[DType.int, 1],::SIMD[DType.int, 1]]()"<:!Int {:scalar<index> 1}, :!Int {:scalar<index> 2}>>
comptime test_fn_literal_type_type_2 = type_of(test_fn_literal_type[1, 2])

# CHECK-LABEL:      lit.fn @"test_fn_literal_parameter{{.*}}"
def test_fn_literal_parameter[x : test_fn_literal_type_type_1]():
    # CHECK:      lit.alias.decl *"t{{[^"]*}}":
    # CHECK-SAME: !kgen.func.literal<:!lit.fn<[1]({{.*}}!lit.struct<#RET <:!Int {:scalar<index> 1}, :!Int {:scalar<index> 3}>>{{.*}}) -> !kgen.none>
    # CHECK-SAME: #kgen.func.symbol<@{{.*}}::@"test_fn_literal_type[::SIMD[DType.int, 1],::SIMD[DType.int, 1]]()"<:!Int {:scalar<index> 1}, :!Int {:scalar<index> 3}>>
    # CHECK-SAME: = <bind_params({{.*}}, :!Int {:scalar<index> 3})>
    comptime t = x[3]


def test_fn_literal_call():
    # CHECK: lit.call @{{.*}}::@"test_fn_literal_type[::SIMD[DType.int, 1],::SIMD[DType.int, 1]]()"{{.*}}<:!Int {:scalar<index> 1}, :!Int {:scalar<index> 2}>
    var _ = test_fn_literal_type[1, 2]()


trait FooTrait:
    def foo[x: Int, y: Int](self) -> Int:
        ...


@fieldwise_init
struct Foo[x: Int, y: Int](FooTrait, Movable where False):
    # CHECK: kgen.witness "foo[::SIMD[DType.int, 1],::SIMD[DType.int, 1]]($0)"
    def foo[q: Int, z: Int](self) -> Int:
        return Self.x + Self.y


# Ensure nested function types mangle independently of their enclosing
# scope.

# INLINE-LABEL: lit.trait.decl @PluginHooks
# INLINE: lit.alias.decl *"print_emit_fn{{[`0-9]*}}":
# INLINE-SAME: !lit.generator<<"[[OMUT:O.mut`2x]]": !Bool,
# INLINE-SAME: "[[OORIGIN:O._mlir_origin`2x1]]": origin<
trait PluginHooks:
    comptime print_emit_fn: Optional[def[O: Origin](str: StringSlice[O]) thin]


# INLINE-LABEL: lit.struct.decl @DefaultPlugin
# INLINE: lit.alias.decl *"print_emit_fn{{[`0-9]*}}":
# INLINE-SAME: !lit.generator<<"[[OMUT]]": !Bool,
# INLINE-SAME: "[[OORIGIN]]": origin<
# INLINE: kgen.conformance{{.*}}@PluginHooks
# INLINE: kgen.witness "print_emit_fn"
struct DefaultPlugin(PluginHooks, Movable where False):
    comptime print_emit_fn: Optional[def[O: Origin](str: StringSlice[O]) thin] = (
        None
    )
