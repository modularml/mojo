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

# Test parsing of the `where <condition> else "<message>"` spelling of the
# optional message on a `where` clause. It is an alternative surface syntax for
# `where (condition, "message")` (see `where_message.mojo`) and lowers to the
# same `#kgen.constraint` message field, printed as its trailing `"..."`.
#
# `else` is a hard keyword, so unlike the bare-comma form that this replaces it
# is unambiguous against the comma that separates conformance-list entries.
#
# Covered forms: trailing function constraints, trailing struct constraints,
# multi-clause, adjacent string-literal concatenation, a parenthesized message
# spanning lines, conformance-list conditional conformance, `thin` function
# types, `alias`/`comptime` declarations, and a ternary condition whose own
# `else` must not be mistaken for the message separator.

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
struct StructMsg[N: Int] where N > 0 else "N must be positive":
    pass


##===----------------------------------------------------------------------===##
# Trailing function constraint with a message
##===----------------------------------------------------------------------===##
# CHECK-LABEL: lit.fn @"fn_msg
# CHECK-SAME: "scaling factor must be greater than 1"
def fn_msg[sc: Int]() where sc > 1 else "scaling factor must be greater than 1":
    pass


##===----------------------------------------------------------------------===##
# Multiple where clauses, each with its own message
##===----------------------------------------------------------------------===##
# CHECK-LABEL: lit.fn @"multi_clause
# CHECK-SAME: "a positive"
# CHECK-SAME: "b positive"
def multi_clause[a: Int, b: Int]() where a > 0 else "a positive" where (
    b > 0
) else "b positive":
    pass


##===----------------------------------------------------------------------===##
# The two spellings mix freely across clauses on one signature
##===----------------------------------------------------------------------===##
# CHECK-LABEL: lit.fn @"mixed_spellings
# CHECK-SAME: "paren form"
# CHECK-SAME: "else form"
def mixed_spellings[a: Int, b: Int]() where (a > 0, "paren form") where (
    b > 0
) else "else form":
    pass


##===----------------------------------------------------------------------===##
# A parenthesized condition keeps its message; the parens are the condition's,
# not a message tuple's.
##===----------------------------------------------------------------------===##
# CHECK-LABEL: lit.fn @"paren_cond
# CHECK-SAME: "x must be a single digit"
def paren_cond[x: Int]() where (x > 0 and x < 10) else "x must be a single digit":
    pass


##===----------------------------------------------------------------------===##
# Adjacent string literals concatenate (Python-style)
##===----------------------------------------------------------------------===##
# CHECK-LABEL: lit.fn @"concat_msg
# CHECK-SAME: "part one part two"
def concat_msg[x: Int]() where x > 0 else "part one " "part two":
    pass


##===----------------------------------------------------------------------===##
# Parentheses around the message let a long one wrap across lines
##===----------------------------------------------------------------------===##
# CHECK-LABEL: lit.fn @"wrapped_msg
# CHECK-SAME: "the width must be positive and no wider than a warp"
def wrapped_msg[w: Int]() where w > 0 else (
    "the width must be positive "
    "and no wider than a warp"
):
    pass


##===----------------------------------------------------------------------===##
# A ternary condition owns its own `else`; a second `else` is the message.
##===----------------------------------------------------------------------===##
# CHECK-LABEL: lit.fn @"ternary_cond_no_msg
# A message would print as the constraint's trailing `, "..."` field; assert
# there is none.
# CHECK-NOT: constraint<{{.*}}, "
def ternary_cond_no_msg[A: Int, B: Int, C: Int]() where (C <= B) if (
    A <= B
) else (C <= Int(0)):
    pass


# CHECK-LABEL: lit.fn @"ternary_cond_msg
# CHECK-SAME: "C is bounded by B, or by zero"
def ternary_cond_msg[A: Int, B: Int, C: Int]() where (C <= B) if (
    A <= B
) else (C <= Int(0)) else "C is bounded by B, or by zero":
    pass


##===----------------------------------------------------------------------===##
# Conformance-list conditional conformance with a message
##===----------------------------------------------------------------------===##
# The constrained trait alias is printed at the top of the output, so match
# it with CHECK-DAG rather than relying on position.
# CHECK-DAG: where #kgen.constraint<{{.*}}, "wrapped type must be extra">
struct CondConfMsg[T: Base](
    Extra where conforms_to(T, Extra) else "wrapped type must be extra",
):
    pass


##===----------------------------------------------------------------------===##
# A conformance constraint with a message, followed by another entry: `else` is
# a keyword, so the separating comma is unambiguous.
##===----------------------------------------------------------------------===##
# CHECK-DAG: where #kgen.constraint<{{.*}}, "must be extra when wrapped">
struct CondConfMsgThenEntry[T: Base](
    Extra where conforms_to(T, Extra) else "must be extra when wrapped", Base,
):
    pass


##===----------------------------------------------------------------------===##
# `thin` function type carrying a constrained message
##===----------------------------------------------------------------------===##
# CHECK-DAG: lit.alias.decl *"Kernel{{.*}}"width must be positive"
comptime Kernel = def[w: Int]() thin -> None where w > 0 else (
    "width must be positive"
)


# The constraint is part of the type, so it shows up in the mangled name too.
# CHECK-DAG: lit.fn @"takes_kernel[def[{{.*}}]() thin -> None where (
def takes_kernel[F: Kernel]():
    F[4]()


##===----------------------------------------------------------------------===##
# `comptime` (alias) declaration with a `where` clause that has a message
##===----------------------------------------------------------------------===##
# CHECK-DAG: lit.alias.decl *"AliasMsg{{.*}}"T must be extra"
comptime AliasMsg[T: AnyType] where conforms_to(T, Extra) else "T must be extra" = T
