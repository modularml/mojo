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

# Test that the optional message on a `where` clause
# (`where (condition, "message")`) is surfaced in constraint diagnostics: in
# the "evaluated to False" note when a constraint is violated, and in the
# "needs evidence for" note when a constraint is inconclusive.
#
# Each section uses its own struct/function decl so per-section diagnostic
# notes don't bleed across test sites (mirrors the convention in
# `struct_body_constraints.mojo`).

# RUN: %parse-mojo-isolated -verify-diagnostics %s


##===----------------------------------------------------------------------===##
# Satisfied constraint with a message - positive case
##===----------------------------------------------------------------------===##
# The message must not affect constraint checking on the happy path.


struct SatisfiedStruct[N: Int]
    where (N > 0, "N must be positive"):
    pass


def use_satisfied_struct():
    var x: SatisfiedStruct[5]


##===----------------------------------------------------------------------===##
# Violated struct body constraint - message in the note
##===----------------------------------------------------------------------===##


# expected-note @below {{'ViolatedStruct' declared here}}
struct ViolatedStruct[N: Int]
    # expected-note @below {{constraint declared here evaluated to False, expected '(N > Int(0))': N must be positive}}
    where (N > 0, "N must be positive"):
    pass


def use_violated_struct():
    # expected-error @below {{violated constraint}}
    var x: ViolatedStruct[-1]


##===----------------------------------------------------------------------===##
# Violated function constraint - message in the note
##===----------------------------------------------------------------------===##


# expected-note @below {{function declared here}}
def gated_fn[sc: Int]()
    # expected-note @below {{constraint declared here evaluated to False, expected '(sc > Int(1))': scaling factor must be greater than 1}}
    where (sc > 1, "scaling factor must be greater than 1"):
    pass


def use_gated_fn():
    # expected-error @below {{invalid call to 'gated_fn': violated constraint}}
    gated_fn[0]()


##===----------------------------------------------------------------------===##
# Violated compound (and) constraint - one clause keeps one message
##===----------------------------------------------------------------------===##
# `where (A and B, "msg")` is a single constraint carrying the message.


# expected-note @below {{'CompoundStruct' declared here}}
struct CompoundStruct[N: Int]
    # expected-note @below {{constraint declared here evaluated to False, expected '(N < Int(10)) if (N > Int(0)) else (N > Int(0))': N must be in (0, 10)}}
    where (N > 0 and N < 10, "N must be in (0, 10)"):
    pass


def use_compound_struct():
    # expected-error @below {{violated constraint}}
    var x: CompoundStruct[-1]


##===----------------------------------------------------------------------===##
# Inconclusive constraint - message in the "needs evidence" note
##===----------------------------------------------------------------------===##


# Helper that represents an unprovable constraint since it is not
# always_inline("builtin").
# expected-note @below {{cannot evaluate call to non-builtin function declared here}}
def is_prime(x: Int) -> Bool:
    return x > 1


# expected-note @below {{cannot prove constraint}}
def inconclusive_fn[x: Int]()
    # expected-note @below {{constraint declared here needs evidence for 'is_prime(Int(2))': x must be prime}}
    where (is_prime(x), "x must be prime"):
    pass


def use_inconclusive_fn():
    # expected-error @below {{invalid call to 'inconclusive_fn': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    inconclusive_fn[2]()


##===----------------------------------------------------------------------===##
# Violated alias (comptime) constraint - message in the note
##===----------------------------------------------------------------------===##


comptime GatedAlias[N: Int]
    # expected-note @below {{constraint declared here evaluated to False, expected '(N > Int(0))': alias requires a positive N}}
    where (N > 0, "alias requires a positive N") = N


def use_gated_alias():
    # expected-error @below {{violated constraint}}
    comptime x = GatedAlias[-1]
