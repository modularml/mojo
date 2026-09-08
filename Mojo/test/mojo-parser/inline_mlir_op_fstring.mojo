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


# CHECK-LABEL: lit.fn @"test_basic
def test_basic(x: __mlir_type.index, y: __mlir_type.index) -> __mlir_type.index:
    # `%{name}` looks `name` up in the Mojo scope and dispatches on its
    # value kind: SSA-representable values (function args, locals) become
    # operand bindings (`%argN`); parametric/comptime PValues become
    # literal substitutions (handled deferred — see
    # `inline_mlir_op_fstring_deferred.mojo`). Here `x` and `y` are SSA.
    # CHECK: index.add %{{[^ ]+}}, %{{[^ ]+}}
    return __mlir_op[`index.add %{x}, %{y}`]


# CHECK-LABEL: lit.fn @"test_arg_type
def test_arg_type(
    x: __mlir_type.index, y: __mlir_type.index
) -> __mlir_type.index:
    # `%{type_of(x)}` substitutes the textual MLIR type of the value bound to `x`.
    # `index.*` arithmetic is typeless, so a cast op carries the type slot.
    # CHECK: builtin.unrealized_conversion_cast %{{[^ ]+}} : index to index
    return __mlir_op[`builtin.unrealized_conversion_cast %{x} : %{type_of(x)} to index`]


# CHECK-LABEL: lit.fn @"test_result_type
def test_result_type(
    x: __mlir_type.index, y: __mlir_type.index
) -> __mlir_type.index:
    # `_type=` is verified against the parsed op's result types; the type
    # itself is inlined into the template.
    # CHECK: index.add %{{[^ ]+}}, %{{[^ ]+}}
    return __mlir_op[`index.add %{x}, %{y}`, _type=__mlir_type.index]


# CHECK-LABEL: lit.fn @"test_type_first
def test_type_first(
    x: __mlir_type.index, y: __mlir_type.index
) -> __mlir_type.index:
    # `_type=` is accepted before the positional template.
    # CHECK: index.add %{{[^ ]+}}, %{{[^ ]+}}
    return __mlir_op[_type=__mlir_type.index, `index.add %{x}, %{y}`]


# CHECK-LABEL: lit.fn @"test_repeated_ref
def test_repeated_ref(x: __mlir_type.index) -> __mlir_type.index:
    # `%{x}` appearing twice resolves to the same operand (one block-arg
    # slot, two textual references).
    # CHECK: index.add %{{[^ ]+}}, %{{[^ ]+}}
    return __mlir_op[`index.add %{x}, %{x}`]


# CHECK-LABEL: lit.fn @"test_no_operands
def test_no_operands() -> __mlir_type.index:
    # No `%{…}` operand placeholders — types live entirely in the template.
    # CHECK: index.constant 42
    return __mlir_op[`index.constant 42`, _type=__mlir_type.index]


# CHECK-LABEL: lit.fn @"test_string_attr_with_percent
def test_string_attr_with_percent(
    x: __mlir_type.index, y: __mlir_type.index
) -> __mlir_type.index:
    # `%{...}` inside a string literal must NOT be substituted.
    # CHECK: index.add %{{[^ ]+}}, %{{[^ ]+}} {note = "%{x} stays as is"}
    return __mlir_op[
        `index.add %{x}, %{y} {note = "%{x} stays as is"}`
    ]


# CHECK-LABEL: lit.fn @"test_multi_result
def test_multi_result(
    x: __mlir_type.index, y: __mlir_type.index
) -> Tuple[__mlir_type.i32, __mlir_type.i32]:
    # `index` arithmetic is single-result, so a two-result
    # `builtin.unrealized_conversion_cast` exercises the Tuple path. `_type=Tuple`
    # carries both results; output types differ from the inputs so the cast is
    # not a foldable no-op.
    # CHECK: builtin.unrealized_conversion_cast %{{[^ ]+}}, %{{[^ ]+}} : index, index to i32, i32
    return __mlir_op[
        `builtin.unrealized_conversion_cast %{x}, %{y} : %{type_of(x)}, %{type_of(y)} to i32, i32`,
        _type=Tuple[__mlir_type.i32, __mlir_type.i32],
    ]


# CHECK-LABEL: lit.fn @"test_chunked_template
def test_chunked_template(
    x: __mlir_type.index, y: __mlir_type.index
) -> __mlir_type.index:
    # Multiple positional backtick chunks are concatenated at parse time,
    # so long templates can wrap across source lines. Equivalent to
    # `__mlir_op[`index.add %{x}, %{y}`]`.
    # CHECK: index.add %{{[^ ]+}}, %{{[^ ]+}}
    return __mlir_op[
        `index.add `,
        `%{x}, %{y}`,
    ]
