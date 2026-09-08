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

# Negative cases for body-constraint discharge: bindings whose body
# constraints are concrete and violated, or concrete and unprovable,
# should fail with a diagnostic at the binding site.

# RUN: %parse-mojo-isolated -verify-diagnostics %s


##===----------------------------------------------------------------------===##
# Single-variable alias, violation at full binding
##===----------------------------------------------------------------------===##


comptime pos_only[x: Int]
    # expected-note @below {{constraint declared here evaluated to False}}
    where x > 0 = x


def use_single_var_violation():
    # expected-error @below {{violated constraint}}
    comptime bad = pos_only[-1]


##===----------------------------------------------------------------------===##
# Multi-variable alias, violation at full binding
##===----------------------------------------------------------------------===##
# Both parameters are bound to literals that make `x < y` false; the
# constraint is concrete and provably violated.


comptime ordered_pair[x: Int, y: Int]
    # expected-note @below {{constraint declared here evaluated to False}}
    where x < y = x + y


def use_multi_var_violation():
    # expected-error @below {{violated constraint}}
    comptime bad = ordered_pair[5, 3]


##===----------------------------------------------------------------------===##
# Multi-variable alias, violation by the concrete part of a partial bind
##===----------------------------------------------------------------------===##
# `where x > 0` references only `x`. Binding `x = -1` makes that part
# of the constraint concrete and false even though `y` is still
# unbound; the partial binding fails.


comptime needs_x_positive[x: Int, y: Int]
    # expected-note @below {{constraint declared here evaluated to False}}
    where x > 0 = x + y


def use_partial_bind_violation():
    # expected-error @below {{violated constraint}}
    comptime bad = needs_x_positive[-1, _]


##===----------------------------------------------------------------------===##
# Single-variable alias, parametric binding without evidence
##===----------------------------------------------------------------------===##
# The bound value is the caller's parameter `K`. After substitution the
# constraint is `K > 0`, which is concrete (no unbound parameters) but
# unprovable here because the caller carries no `where K > 0` clause.


comptime needs_positive[x: Int]
    # expected-note @below {{constraint declared here needs evidence for}}
    where x > 0 = x


def use_unprovable[K: Int]():
    # expected-error @below {{lacking evidence to prove correctness}}
    comptime z = needs_positive[K]


##===----------------------------------------------------------------------===##
# `and` constraint, both conjuncts concrete, one false
##===----------------------------------------------------------------------===##
# `and(x > 0, y > 0)` with `x = 5, y = -1` folds to False, so the
# binding fails.


comptime and_pos[x: Int, y: Int]
    # expected-note @below {{constraint declared here evaluated to False}}
    where x > 0 and y > 0 = x + y


def use_and_full_violation():
    # expected-error @below {{violated constraint}}
    comptime bad = and_pos[5, -1]


##===----------------------------------------------------------------------===##
# `or` constraint, both disjuncts concrete and false
##===----------------------------------------------------------------------===##


comptime or_pos[x: Int, y: Int]
    # expected-note @below {{constraint declared here evaluated to False}}
    where x > 0 or y > 0 = x + y


def use_or_full_violation():
    # expected-error @below {{violated constraint}}
    comptime bad = or_pos[-1, -2]


##===----------------------------------------------------------------------===##
# Function symbol, violation at full binding
##===----------------------------------------------------------------------===##


# expected-note @below {{function declared here}}
def need_pos[x: Int, y: Int]()
    # expected-note @below {{constraint declared here evaluated to False}}
    where x > 0:
    pass


def use_fn_violation():
    # expected-error @below {{violated constraint}}
    need_pos[-1, 3]()
