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

##===----------------------------------------------------------------------===##
# comptime assert errors
##===----------------------------------------------------------------------===##


struct NotBool(Movable where False):
    pass


def test_non_bool_type_error[x: NotBool]():
    # expected-error @below {{'NotBool' does not implement the '__bool__' method}}
    comptime assert x


def test_runtime_expr_error(x: Int, y: Int):
    # expected-error @below {{cannot use a dynamic value in 'comptime assert' expression}}
    comptime assert x == y


def test_non_string_literal_message_error():
    # expected-error @below {{cannot implicitly convert}}
    comptime assert True, 42


struct Notdef[x: Bool](Movable where False):
    # expected-error @below {{'comptime assert' must be inside a function; move this into a function body}}
    comptime assert x


# expected-note @below {{cannot prove constraint}}
# expected-note @below {{constraint declared here}}
# expected-note @below {{function declared here}}
def requires_natural[x: Int]() where x >= 0:
    pass


def requires_not_natural[x: Int]() where not (x >= 0):
    pass


def requires_positive[x: Int]() where x > 0:
    pass


def requires_not_positive[x: Int]() where not (x > 0):
    pass


def requires_zero[x: Int]() where x == 0:
    pass


def requires_not_zero[x: Int]() where not (x == 0):
    pass


def test_assert_injects_assumption_correctly[x: Int]():
    comptime if x > 10:
        comptime assert x >= 0

        # This is OK.
        requires_natural[x]()
    else:
        # expected-error @below {{invalid call to 'requires_natural': lacking evidence to prove correctness}}
        # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
        requires_natural[x]()

    # expected-error @below {{invalid call to 'requires_natural': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    requires_natural[x]()


# COM: MOCO-3725: the else branch must not inherit the positive condition.
# COM: The else branch gets NOT(x >= 0) as an assumption, which contradicts the
# COM: `where x >= 0` constraint, so the call is "violated", not "unprovable".
def test_else_branch_does_not_leak_then_assumption[x: Int]():
    comptime if x >= 0:
        requires_natural[x]()
    else:
        # expected-error @below {{invalid call to 'requires_natural': violated constraint}}
        requires_natural[x]()


def test_else_branch_can_use_inverse_constraint[x: Int]():
    comptime if x >= 0:
        requires_natural[x]()
    else:
        requires_not_natural[x]()


# COM: Nested comptime-if lowering should accumulate prior false assumptions.
def test_multibranch_branch_assumptions[x: Int]():
    comptime if x > 0:
        requires_positive[x]()
    elif x == 0:
        requires_not_positive[x]()
        requires_zero[x]()
    else:
        requires_not_positive[x]()
        requires_not_zero[x]()


def test_newly_created_scope[x: Int]():
    # expected-error @below {{invalid call to 'requires_natural': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    comptime y = requires_natural[x]()

    # COM: This assert should not validate the above statement.
    comptime assert x >= 0


def test_always_false_no_warning():
    comptime assert 2 < 1


def test_always_true_no_warning():
    comptime assert 2 > 1
