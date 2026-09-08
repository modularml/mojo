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

# Test parsing of optional string-literal messages on `where` clauses
# (`where (condition, "message")`). The message is stored on the emitted
# `#kgen.constraint` attribute (printed as its trailing `"..."` field) and is
# surfaced in the diagnostic when the constraint fails (see
# `where_message_errors.mojo`).
#
# The message lives inside the parentheses, so the trailing comma that
# separates the next conformance entry is unambiguous.
#
# Covered forms: trailing function constraints, trailing struct constraints,
# multi-clause `where (a, "x") where (b, "y")`, adjacent string-literal
# concatenation, conformance-list conditional-conformance constraints, and
# `alias`/`comptime` declarations.

# RUN: %parse-mojo-isolated %s | FileCheck %s


trait Base:
    pass


trait Extra:
    pass


##===----------------------------------------------------------------------===##
# Trailing struct constraint with a message
##===----------------------------------------------------------------------===##
# CHECK-LABEL: lit.struct.decl @StructMsg
# CHECK-SAME: "N must be positive"
struct StructMsg[N: Int] where (N > 0, "N must be positive"):
    pass


##===----------------------------------------------------------------------===##
# Trailing function constraint with a message
##===----------------------------------------------------------------------===##
# CHECK-LABEL: lit.fn @"fn_msg
# CHECK-SAME: "scaling factor must be greater than 1"
def fn_msg[sc: Int]() where (sc > 1, "scaling factor must be greater than 1"):
    pass


##===----------------------------------------------------------------------===##
# Multiple where clauses, each with its own message
##===----------------------------------------------------------------------===##
# CHECK-LABEL: lit.fn @"multi_clause
# CHECK-SAME: "a positive"
# CHECK-SAME: "b positive"
def multi_clause[a: Int, b: Int]() where (a > 0, "a positive") where (
    b > 0, "b positive"
):
    pass


##===----------------------------------------------------------------------===##
# A parenthesized single condition (no message) is left as the condition, not
# mistaken for a message clause. This guards the paren-single vs paren-tuple
# discrimination in `extractParenthesizedMessage`.
##===----------------------------------------------------------------------===##
# CHECK-LABEL: lit.fn @"wrapping_cond
# A message would print as the constraint's trailing `, "..."` field; assert
# there is none.
# CHECK-NOT: constraint<{{.*}}, "
def wrapping_cond[x: Int]() where (x > 0 and x < 10):
    pass


##===----------------------------------------------------------------------===##
# Adjacent string literals concatenate (Python-style)
##===----------------------------------------------------------------------===##
# CHECK-LABEL: lit.fn @"concat_msg
# CHECK-SAME: "part one part two"
def concat_msg[x: Int]() where (x > 0, "part one " "part two"):
    pass


##===----------------------------------------------------------------------===##
# Conformance-list conditional conformance with a message
##===----------------------------------------------------------------------===##
# The constrained trait alias is printed at the top of the output, so match
# it with CHECK-DAG rather than relying on position.
# CHECK-DAG: where #kgen.constraint<{{.*}}, "wrapped type must be extra">
struct CondConfMsg[T: Base](
    Extra where (conforms_to(T, Extra), "wrapped type must be extra"),
):
    pass


##===----------------------------------------------------------------------===##
# Conditional conformance without a message still parses as two entries
##===----------------------------------------------------------------------===##
# CHECK-DAG: lit.struct.decl @CondConfNoMsg
struct CondConfNoMsg[T: Base](
    Extra where conforms_to(T, Extra), Base,
):
    pass


##===----------------------------------------------------------------------===##
# A conformance constraint with a user message, followed by another entry: the
# message is inside the parentheses, so the separating comma is unambiguous.
##===----------------------------------------------------------------------===##
# CHECK-DAG: where #kgen.constraint<{{.*}}, "must be extra when wrapped">
struct CondConfMsgThenEntry[T: Base](
    Extra where (conforms_to(T, Extra), "must be extra when wrapped"), Base,
):
    pass


##===----------------------------------------------------------------------===##
# `comptime` (alias) declaration with a `where` clause that has a user message
##===----------------------------------------------------------------------===##
# CHECK-DAG: lit.alias.decl *"AliasMsg{{.*}}"T must be extra"
comptime AliasMsg[T: AnyType] where (conforms_to(T, Extra), "T must be extra") = T
