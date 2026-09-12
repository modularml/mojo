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

# Test negative parse cases for the `where <condition> else "<message>"` form.
# As with `where (condition, "message")`, only string literals are accepted:
# a non-literal expression would need comptime evaluation that the parser
# cannot perform, so it is rejected with the same targeted diagnostic (see
# `where_message_parse_errors.mojo`).

# RUN: %parse-mojo-isolated -verify-diagnostics %s


trait Base:
    pass


trait Extra:
    pass


##===----------------------------------------------------------------------===##
# Non-literal message expression (identifier)
##===----------------------------------------------------------------------===##


# expected-error @below {{the message in a 'where' clause must be a string literal}}
def non_literal_message[x: Int]() where x > 0 else x:
    pass


##===----------------------------------------------------------------------===##
# Non-literal message expression (alias reference)
##===----------------------------------------------------------------------===##


comptime MSG = "must be positive"


# expected-error @below {{the message in a 'where' clause must be a string literal}}
def alias_message[x: Int]() where x > 0 else MSG:
    pass


##===----------------------------------------------------------------------===##
# A t-string is not a string literal
##===----------------------------------------------------------------------===##


# expected-error @below {{the message in a 'where' clause must be a string literal}}
def t_string_message[x: Int]() where x > 0 else t"x is {x}":
    pass


##===----------------------------------------------------------------------===##
# Parentheses may wrap the message, but not turn it into an expression
##===----------------------------------------------------------------------===##


# expected-error @below {{the message in a 'where' clause must be a string literal}}
def paren_non_literal[x: Int]() where x > 0 else (x):
    pass


##===----------------------------------------------------------------------===##
# Non-literal message on a struct trailing constraint
##===----------------------------------------------------------------------===##


# expected-error @below {{the message in a 'where' clause must be a string literal}}
struct StructNonLiteral[N: Int] where N > 0 else N < 10:
    pass


##===----------------------------------------------------------------------===##
# Non-literal message on a conformance-list constraint
##===----------------------------------------------------------------------===##


# expected-error @below {{the message in a 'where' clause must be a string literal}}
struct CondConfNonLiteral[T: Base](Extra where conforms_to(T, Extra) else T):
    pass


##===----------------------------------------------------------------------===##
# The message may be written once, in one form or the other -- not both
##===----------------------------------------------------------------------===##


# expected-error @below {{a 'where' clause takes at most one message: prefer 'where condition else "message"'}}
def both_forms[x: Int]() where (x > 0, "paren form") else "else form":
    pass


struct CondConfBothForms[T: Base](
    # expected-error @below {{a 'where' clause takes at most one message: prefer 'where condition else "message"'}}
    Extra where (conforms_to(T, Extra), "paren form") else "else form"
):
    pass


##===----------------------------------------------------------------------===##
# `else` with the message left out entirely
##===----------------------------------------------------------------------===##


# expected-error @below {{expected a string literal message after 'else'}}
def missing_message[x: Int]() where x > 0 else:
    pass


# The conformance-list entry separator terminates the clause the same way.
struct CondConfMissingMessage[T: Base](
    # expected-error @below {{expected a string literal message after 'else'}}
    Extra where conforms_to(T, Extra) else, Base
):
    pass


# expected-error @below {{expected a string literal message after 'else'}}
comptime AliasMissingMessage[T: AnyType] where conforms_to(T, Extra) else = T


##===----------------------------------------------------------------------===##
# A dedented `else` is the start of a new statement, not this clause's message
##===----------------------------------------------------------------------===##

# The dedented `else` lands at file scope, and a bare file-scope statement that
# fails to parse ends the module parse (unlike a bad decl body, which is
# contained by decl boundary scanning). So this case must stay last: anything
# below it would silently produce no diagnostics at all.


# expected-error @+1 {{expected ':' in function definition}}
def dedented_else[x: Int]() where x > 0
# expected-error @below {{unexpected token in expression}}
else "":
    pass
