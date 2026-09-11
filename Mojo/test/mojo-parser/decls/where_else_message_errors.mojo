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

# Test that a message written `where <condition> else "<message>"` reaches the
# constraint diagnostics exactly as the equivalent `where (condition,
# "message")` does. The two forms differ only in surface syntax, so these are
# the `where_message_errors.mojo` and `where_message_conformance.mojo` cases
# respelled, and the expected notes are character-for-character the same.
#
# Each section uses its own struct/function decl so per-section diagnostic
# notes don't bleed across test sites.

# RUN: %parse-mojo-isolated -verify-diagnostics %s


##===----------------------------------------------------------------------===##
# Satisfied constraint with a message - positive case
##===----------------------------------------------------------------------===##
# The message must not affect constraint checking on the happy path.


struct SatisfiedStruct[N: Int]
    where N > 0 else "N must be positive":
    pass


def use_satisfied_struct():
    var x: SatisfiedStruct[5]


##===----------------------------------------------------------------------===##
# Violated struct body constraint - message in the note
##===----------------------------------------------------------------------===##


# expected-note @below {{'ViolatedStruct' declared here}}
struct ViolatedStruct[N: Int]
    # expected-note @below {{constraint declared here evaluated to False, expected '(N > Int(0))': N must be positive}}
    where N > 0 else "N must be positive":
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
    where sc > 1 else "scaling factor must be greater than 1":
    pass


def use_gated_fn():
    # expected-error @below {{invalid call to 'gated_fn': violated constraint}}
    gated_fn[0]()


##===----------------------------------------------------------------------===##
# A parenthesized message wrapping across lines reaches the note joined up
##===----------------------------------------------------------------------===##


# expected-note @below {{function declared here}}
def wrapped_msg_fn[w: Int]()
    # expected-note @below {{constraint declared here evaluated to False, expected '(w > Int(0))': the width must be positive and no wider than a warp}}
    where w > 0 else (
        "the width must be positive "
        "and no wider than a warp"
    ):
    pass


def use_wrapped_msg_fn():
    # expected-error @below {{invalid call to 'wrapped_msg_fn': violated constraint}}
    wrapped_msg_fn[0]()


##===----------------------------------------------------------------------===##
# Violated conditional conformance - message on the conformance diagnostic
##===----------------------------------------------------------------------===##


trait Marker:
    pass


struct No:
    pass


struct GenBox[T: Deinitable](
    # expected-note @below {{failed constraint: GenBox[T] requires T to be a Marker}}
    Marker where conforms_to(T, Marker) else "GenBox[T] requires T to be a Marker"
):
    pass


# expected-note @below {{function declared here}}
def wants_marker[U: Marker & Deinitable](x: U):
    pass


def use_gen_box(b: GenBox[No]):
    # expected-error @below {{does not conform to trait 'Deinitable & Marker'}}
    wants_marker(b)
