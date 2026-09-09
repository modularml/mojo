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

# Test negative parse cases for the optional message on `where` clauses. A
# message is written `where (condition, "message")`. For now only string
# literals are accepted as the message: a non-literal expression would need
# comptime evaluation that the parser cannot perform, so it is rejected with a
# targeted diagnostic.

# RUN: %parse-mojo-isolated -verify-diagnostics %s


##===----------------------------------------------------------------------===##
# Non-literal message expression (identifier)
##===----------------------------------------------------------------------===##


# expected-error @below {{the message in a 'where' clause must be a string literal}}
def non_literal_message[x: Int]() where (x > 0, x):
    pass


##===----------------------------------------------------------------------===##
# Non-literal message expression (alias reference)
##===----------------------------------------------------------------------===##


comptime MSG = "must be positive"


# expected-error @below {{the message in a 'where' clause must be a string literal}}
def alias_message[x: Int]() where (x > 0, MSG):
    pass


##===----------------------------------------------------------------------===##
# Non-literal message on a struct trailing constraint
##===----------------------------------------------------------------------===##


# expected-error @below {{the message in a 'where' clause must be a string literal}}
struct StructNonLiteral[N: Int] where (N > 0, N < 10):
    pass


##===----------------------------------------------------------------------===##
# Non-literal message on a conformance-list constraint
##===----------------------------------------------------------------------===##


trait Base:
    pass


trait Extra:
    pass


# expected-error @below {{the message in a 'where' clause must be a string literal}}
struct CondConfNonLiteral[T: Base](Extra where (conforms_to(T, Extra), T)):
    pass


##===----------------------------------------------------------------------===##
# Wrong tuple arity: a `where` clause takes a condition and an optional message
##===----------------------------------------------------------------------===##


# expected-error @below {{a 'where' clause takes a condition and an optional message: 'where (condition, "message")'}}
def too_many_elements[x: Int]() where (x > 0, "msg", "extra"):
    pass
