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

# RUN: kgen -elaborate -O0 %s -S | FileCheck %s

# End-to-end elaboration of the deferred `__mlir_op[`...`]` f-string path: the
# parser stashes a `kgen.deferred` template, and the elaborator concretizes
# result types, re-substitutes parametric operand types / `%param` markers, and
# re-parses the op via `lowerFStringMLIROp` after parameter binding.


# Parametric operand type: `%{type_of(x)}` survives to `kgen.deferred` and is
# re-substituted with the concrete `!kgen.scalar<T>` per instantiation.
def param_add[
    T: __mlir_type.`!kgen.dtype`
](
    x: __mlir_type[`!kgen.scalar<`, T, `>`],
    y: __mlir_type[`!kgen.scalar<`, T, `>`],
) -> __mlir_type[`!kgen.scalar<`, T, `>`]:
    return __mlir_op[`pop.add %{x}, %{y} : %{type_of(x)}`]


# Multi-result op: both results are wired back through a Tuple. `index`
# arithmetic is single-result, so a two-result
# `builtin.unrealized_conversion_cast` exercises the Tuple path (output types
# differ from the inputs so the cast is not a foldable no-op).
def param_mul[
    T: __mlir_type.`!kgen.dtype`
](x: __mlir_type.index, y: __mlir_type.index) -> Tuple[
    __mlir_type.i32, __mlir_type.i32
]:
    return __mlir_op[
        `builtin.unrealized_conversion_cast %{x}, %{y} : %{type_of(x)}, %{type_of(y)} to i32, i32`,
        _type=Tuple[__mlir_type.i32, __mlir_type.i32],
    ]


# `%param` substitution: a comptime MLIR attribute is stringified by the
# elaborator and inlined into the template before re-parsing.
def lit_attr[T: __mlir_type.`!kgen.dtype`]() -> __mlir_type.index:
    comptime v = __mlir_attr.`42 : index`
    return __mlir_op[`index.constant %{v}`, _type=__mlir_type.index]


# Regression: a placeholder immediately followed by a digit. The bracketed
# `%param<N>` marker terminates the index, so the trailing `0` stays template
# text and the constant is `50`. With the old unterminated `%param0` form the
# `0` was absorbed into the index, silently dropping it (here) or running off
# the end of `fstring_params`.
def lit_then_digit() -> __mlir_type.index:
    comptime v = __mlir_attr.`5 : index`
    return __mlir_op[`index.constant %{v}0`, _type=__mlir_type.index]


# Deferred *operand* type: the operand/result types are themselves
# `__mlir_deferred_type[...]`, so `%{type_of(a)}` survives to `kgen.deferred` and
# is re-substituted with the concrete `iN` once `width` binds (here `i32`).
@no_inline
def deferred_operand_type[
    width: Int
](
    a: __mlir_deferred_type[`i`, +width.__mlir_index__()],
    b: __mlir_deferred_type[`i`, +width.__mlir_index__()],
) -> __mlir_deferred_type[`i`, +width.__mlir_index__()]:
    return __mlir_op[
        `llvm.add %{a}, %{b} : %{type_of(a)}`,
        _type=__mlir_deferred_type[`i`, +width.__mlir_index__()],
    ]


# CHECK-DAG: pop.add %{{.*}}, %{{.*}} : !kgen.scalar<si32>
# CHECK-DAG: pop.add %{{.*}}, %{{.*}} : !kgen.scalar<f32>
# CHECK-DAG: builtin.unrealized_conversion_cast %{{.*}}, %{{.*}} : index, index to i32, i32
# CHECK-DAG: index.constant 42
# CHECK-DAG: index.constant 50
# CHECK-DAG: llvm.add %{{.*}}, %{{.*}} : i32
@export
def top(
    a: __mlir_type.`!kgen.scalar<si32>`,
    b: __mlir_type.`!kgen.scalar<si32>`,
    c: __mlir_type.`!kgen.scalar<f32>`,
    i: __mlir_type.index,
    j: __mlir_type.index,
    p: __mlir_type.i32,
    q: __mlir_type.i32,
) abi("Mojo") -> __mlir_type.index:
    var s = param_add[__mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`](
        a, b
    )
    var f = param_add[__mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`](
        c, c
    )
    var m = param_mul[__mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`](
        i, j
    )
    _ = deferred_operand_type[32](p, q)
    _ = lit_then_digit()
    return lit_attr[__mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`]()
