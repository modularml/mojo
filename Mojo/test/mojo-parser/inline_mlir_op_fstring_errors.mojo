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

# RUN: %parse-mojo-isolated -verify-diagnostics %s


def test_unknown_kwarg(x: __mlir_type.index):
    # `_properties` and friends are not accepted in the new syntax.
    # expected-error @+1 {{only accepts backtick template chunks and an optional '_type=' keyword}}
    var _r = __mlir_op[`index.constant 1`, _properties=__mlir_attr.`{}`]


def test_missing_template(x: __mlir_type.index):
    # Must have a positional template.
    # expected-error @+1 {{requires a backtick f-string template}}
    var _r = __mlir_op[_type=__mlir_type.index]


def test_unterminated_placeholder(x: __mlir_type.index):
    # `%{` without a matching `}` anywhere downstream is rejected.
    # expected-error @+1 {{unterminated `%{...}` placeholder}}
    var _r = __mlir_op[`index.add %{x still no brace`]


def test_unknown_name(x: __mlir_type.index):
    # `%{nope}` references an identifier that isn't in scope.
    # The error comes from Mojo's normal name-resolution path.
    # expected-error @+1 {{use of unknown declaration 'nope'}}
    var _r = __mlir_op[`index.add %{x}, %{nope}`]


def test_bare_type_rejected(x: __mlir_type.index, y: __mlir_type.index):
    # `%type` was dropped; inline the result type text in the template.
    # expected-error @+1 {{`%type` placeholder is no longer supported}}
    var _r = __mlir_op[`index.add %{x}, %{y} : %type`, _type=__mlir_type.index]


def test_parse_failure(x: __mlir_type.index, y: __mlir_type.index):
    # Bogus op text — parser fails, error reports against the Mojo source.
    # expected-error @+1 {{failed to parse f-string MLIR op}}
    var _r = __mlir_op[`index.add %{x} totallybogus %{y}`]


def test_old_type_suffix_rejected(x: __mlir_type.index):
    # The previous `%{name:type}` syntax was replaced by `%{type_of(name)}`.
    # The `:` is no longer a valid identifier character, so it's rejected.
    # expected-error @+1 {{invalid identifier in `%{...}` placeholder: 'x:type'}}
    var _r = __mlir_op[`index.add %{x:type}, %{x}`]


def test_raw_arg_rejected(x: __mlir_type.index):
    # `%arg<N>` is an internal placeholder; reference operands via `%{name}`.
    # expected-error @+1 {{placeholder in __mlir_op template}}
    var _r = __mlir_op[`index.add %arg0, %{x}`]


def test_raw_type_of_rejected(x: __mlir_type.index):
    # Raw `%type_of(...)` is reserved; use `%{type_of(name)}`.
    # expected-error @+1 {{placeholder in __mlir_op template}}
    var _r = __mlir_op[`index.constant 0 : %type_of(arg0)`]


def test_multiple_ops_rejected():
    # A template that parses to more than one op is rejected rather than
    # silently keeping only the first. (Two backtick chunks are concatenated,
    # so the template holds two ops separated by a space.)
    # expected-error @+1 {{expected exactly one op}}
    var _r = __mlir_op[`%r = index.constant 0 `, `%s = index.constant 1`]


def test_error_names_fstring():
    # Lowering failures name the offending template in the diagnostic.
    # expected-error @+1 {{failed to parse f-string MLIR op in f-string `totally bogus`}}
    var _r = __mlir_op[`totally bogus`]


def test_error_names_fstring_with_placeholders(
    x: __mlir_type.index, y: __mlir_type.index
):
    # The diagnostic shows the original `%{name}` source, not the rewritten
    # `%arg<N>` form.
    # expected-error @+1 {{failed to parse f-string MLIR op in f-string `index.add %{x} bogus %{y}`}}
    var _r = __mlir_op[`index.add %{x} bogus %{y}`]
