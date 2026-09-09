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


# Parametric Mojo function. The operand types `!kgen.scalar<T>` are
# parametric but textually parseable — `%{type_of(x)}` substitutes to
# `!kgen.scalar<T>` and the op lowers directly without `kgen.deferred`.
#
# CHECK-LABEL: lit.fn @"param_op_type
# CHECK: pop.add %{{[^ ]+}}, %{{[^ ]+}} : !kgen.scalar<T>
def param_op_type[
    T: __mlir_type.`!kgen.dtype`
](
    x: __mlir_type[`!kgen.scalar<`, T, `>`],
    y: __mlir_type[`!kgen.scalar<`, T, `>`],
) -> __mlir_type[`!kgen.scalar<`, T, `>`]:
    return __mlir_op[`pop.add %{x}, %{y} : %{type_of(x)}`]


# Caller instantiates with `T = si32`.
# CHECK-LABEL: lit.fn @"use_si32
# CHECK: lit.call {{.*}}<:dtype si32>
def use_si32(
    x: __mlir_type.`!kgen.scalar<si32>`, y: __mlir_type.`!kgen.scalar<si32>`
) -> __mlir_type.`!kgen.scalar<si32>`:
    return param_op_type[
        __mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`
    ](x, y)


# Multi-result in a parametric function. `index` arithmetic is single-result,
# so a two-result `builtin.unrealized_conversion_cast` exercises the Tuple
# binding through the f-string path (output types differ from the inputs so
# the cast is not a foldable no-op).
# CHECK-LABEL: lit.fn @"param_multi_result
# CHECK: builtin.unrealized_conversion_cast %{{[^ ]+}}, %{{[^ ]+}} : index, index to i32, i32
def param_multi_result[
    T: __mlir_type.`!kgen.dtype`
](
    x: __mlir_type.index, y: __mlir_type.index
) -> Tuple[__mlir_type.i32, __mlir_type.i32]:
    return __mlir_op[
        `builtin.unrealized_conversion_cast %{x}, %{y} : %{type_of(x)}, %{type_of(y)} to i32, i32`,
        _type=Tuple[__mlir_type.i32, __mlir_type.i32],
    ]


# Caller instantiates `T = si32`.
# CHECK-LABEL: lit.fn @"use_param_multi
# CHECK: lit.call {{.*}}<:dtype si32>
def use_param_multi(
    x: __mlir_type.index, y: __mlir_type.index
) -> Tuple[__mlir_type.i32, __mlir_type.i32]:
    return param_multi_result[
        __mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`
    ](x, y)


# `%{name}` for a comptime/parametric Mojo value stringifies it with
# type elided (matching `__mlir_deferred_attr`'s stringify-with-elide
# semantics). The placeholder lands on the emitted `kgen.deferred` as
# a `%param<N>` marker plus a `fstring_params` array; the elaborator
# substitutes the concrete string after `N` is bound at the call site.
# CHECK-LABEL: lit.fn @"param_literal_attr
# CHECK: kgen.deferred "index.constant 0 {tag = %param<0> : i64}"
# CHECK-SAME: fstring_params = [#kgen<to_string_deferred(#kgen.param.decl.ref<"N">
def param_literal_attr[N: Int](x: __mlir_type.index) -> __mlir_type.index:
    return __mlir_op[
        `index.constant 0 {tag = %{N} : i64}`, _type=__mlir_type.index
    ]


# Caller binds `N = 7`. The elaborator path is exercised end-to-end at
# kgen-elaborate time (covered by integration tests); the parser-only
# snapshot just records the call.
# CHECK-LABEL: lit.fn @"use_literal_seven
# CHECK: lit.call {{.*}}<:!Int {:scalar<index> 7}>
def use_literal_seven(x: __mlir_type.index) -> __mlir_type.index:
    return param_literal_attr[7](x)


# Result type computed at comptime, then substituted into the template
# via `%{T}` (same `%param<N>` mechanism as a comptime Int — the PValue is
# stringified with type elided). `index` ops are typeless, so a typed
# `llvm.mlir.constant` carries the substituted result type. The f-string
# emits `kgen.deferred` because the PValue resolves only after binding.
# CHECK-LABEL: lit.fn @"comptime_type_value
# CHECK: kgen.deferred "llvm.mlir.constant(42 : %param<0>) : %param<0>"
# CHECK-SAME: fstring_params = [
def comptime_type_value() -> __mlir_type.i32:
    comptime T = __mlir_type.i32
    return __mlir_op[`llvm.mlir.constant(42 : %{T}) : %{T}`, _type=T]


# Conditional comptime alias: picks an MLIR type based on the parameter.
# Mirrors `_dtype_to_llvm_type_f8[dtype]` in stdlib/std/_gpu/_utils.mojo.
comptime _selected_type[
    wide: Bool
] = __mlir_type.i64 if wide else __mlir_type.i32


# `T` resolves to either `i64` or `i32` once `wide` is bound. The
# f-string captures the conditional as `kgen.param.expr<cond, ...>`
# inside `fstring_params`; the elaborator picks the branch after
# binding and inlines the type text into `%param<0>`.
# CHECK-LABEL: lit.fn @"conditional_comptime_type
# CHECK: kgen.deferred "llvm.mlir.constant(0 : %param<0>) : %param<0>"
# CHECK-SAME: fstring_params = [{{.*}}param.expr<cond
def conditional_comptime_type[wide: Bool]() -> _selected_type[wide]:
    comptime T = _selected_type[wide]
    return __mlir_op[`llvm.mlir.constant(0 : %{T}) : %{T}`, _type=T]
