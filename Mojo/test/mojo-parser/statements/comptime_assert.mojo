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

##===----------------------------------------------------------------------===##
# comptime assert
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.fn @"test_assert_with_message
def test_assert_with_message[cond: Bool]():
    # CHECK: kgen.param.assert <{{.*}}#lit.struct.extract<:!Bool cond, "_mlir_value">{{.*}}>, data_to_str({{.*}}"custom error message"
    comptime assert cond, "custom error message"


# CHECK-LABEL: lit.fn @"test_assert_with_long_message
def test_assert_with_long_message[cond: Bool]():
    # CHECK: kgen.param.assert <{{.*}}#lit.struct.extract<:!Bool cond, "_mlir_value">{{.*}}>, data_to_str({{.*}}"custom error message with long message and more"
    comptime assert cond, "custom error message"
                          " with long message "
                          "and more"


# CHECK-LABEL: lit.fn @"test_assert_with_message_parameter
def test_assert_with_message_parameter[x: Int]():
    # CHECK: kgen.param.assert <{{.*}}#lit.struct.extract<:!Int x, "_mlir_value">{{.*}}>, data_to_str{{.*}}@String::@"__init__
    comptime assert x, String(x)


# CHECK-LABEL: lit.fn @"test_assert_with_param_expr
def test_assert_with_param_expr[x: Int, y: Int]():
    # CHECK: kgen.param.assert <{{.*}}eq(:scalar<index> #lit.struct.extract<:!Int x, "_mlir_value">, #lit.struct.extract<:!Int y, "_mlir_value">))>, ""
    comptime assert x == y


def requires_natural[x: Int](y: Int) where x >= 0:
    pass


# CHECK-LABEL: lit.fn @"test_assert_enables_where_constraint
def test_assert_enables_where_constraint[x: Int](y: Int):
    # First assert that x >= 0
    # CHECK: kgen.param.assert <{{.*}}ge(:scalar<index> #lit.struct.extract<:!Int x, "_mlir_value">, 0))>, ""
    comptime assert x >= 0

    # Now we can call a function that requires x >= 0 via where clause
    # CHECK: lit.call {{.*}}@"requires_natural
    requires_natural[x](y)


# CHECK-LABEL: lit.fn @"test_assert_with_tstring_message
def test_assert_with_tstring_message[x: Int]():
    # CHECK: kgen.param.assert <{{.*}}>, data_to_str({{.*}}__make_tstring
    comptime assert x, t"expected positive, got {x}"


# CHECK-LABEL: lit.fn @"test_always_true_warning
def test_always_true_warning():
    # CHECK-NOT: kgen.param.assert
    comptime assert 2 > 1, "this assert is useless"


##===----------------------------------------------------------------------===##
# comptime assert contradicts where-clause: overload disambiguation
##===----------------------------------------------------------------------===##
# A comptime assert that proves P should cause any overload with `where not P`
# to be treated as violated (not merely "unprovable"), so the complementary
# overload with `where P` is selected unambiguously.

# Non-builtin function whose result is opaque to the constant folder; this
# exercises the symbolic contradiction path rather than trivial constant folding.
def is_prime_like(x: Int) -> Bool:
    return x > 1


def overloaded_on_primeness[N: Int](x: Int) -> Int where is_prime_like(N):
    return x + N


def overloaded_on_primeness[N: Int](x: Int) -> Int where not is_prime_like(N):
    return x - N


# CHECK-LABEL: lit.fn @"test_assert_contradicts_where_selects_first_overload
def test_assert_contradicts_where_selects_first_overload[N: Int](x: Int) -> Int:
    # Assumption: is_prime_like(N) is true.
    comptime assert is_prime_like(N)
    # The second overload's `where not is_prime_like(N)` is contradicted by the
    # assumption above, so it is violated (not unprovable) and the first overload
    # is the unique candidate.
    # CHECK: lit.call {{.*}}@"overloaded_on_primeness
    return overloaded_on_primeness[N](x)
