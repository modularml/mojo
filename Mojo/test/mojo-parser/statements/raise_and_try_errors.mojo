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
# Raise
##===----------------------------------------------------------------------===##


def raisingFunction() raises:
    pass


# expected-note @below {{or mark surrounding function as 'raises'}}
def callRaisingFunction():
    # expected-error @below {{cannot call function that may raise in a context that cannot raise}}
    # expected-note @below {{try surrounding the call in a 'try' block}}
    raisingFunction()


def cannotReRaise() raises:
    # expected-error @below {{'raise' must live within an 'except' block or a function marked 'raises'}}
    raise


def cannotRaise(err: Error):
    # expected-error @below {{'raise' requires a surrounding 'try' block or the enclosing function to declare 'raises'}}
    raise err


# Issue #12358
def raise_bad_type() raises:
    raise 42  # expected-error {{cannot implicitly convert 'IntLiteral[42]' value to 'Error'}}


def raises_arg(x: String) raises Pointer[String, origin_of(x)]:
    pass


# MOCO-3000
def origin_scope_example():
    try:
        var key = 42  # expected-note {{origin declared here}}
        # expected-error @+1 {{inferred error type 'Pointer[Int, origin_of(key)]' captures origin 'origin_of(key)' from within try body; it is not in scope in except body}}
        raise Pointer(to=key)
    except e:
        _ = e[]  # isn't valid.

    try:
        var str = String()  # expected-note {{origin declared here}}
        # expected-error @+1 {{inferred error type 'Pointer[String, origin_of(str)]' captures origin 'origin_of(str)' from within try body; it is not in scope in except body}}
        raises_arg(str)
    except e2:
        _ = e2[]  # isn't valid.

    try:
        # expected-error @below {{inferred error type 'Pointer[String, origin_of(__call_result_tmp__)]' captures origin of temporary from within try body; it is not in scope in except body}}
        # expected-note @below {{origin declared here}}
        raises_arg(String())
    except e3:
        _ = e3[]  # isn't valid.
