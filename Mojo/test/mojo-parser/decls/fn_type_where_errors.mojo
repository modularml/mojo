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


def violated_at_explicit_bind[
    # expected-note @below {{constraint declared here evaluated to False, expected '(w > Int(0))': width must be positive}}
    F: def[w: Int](Int) thin -> None where (w > 0, "width must be positive")
]():
    # expected-error @below {{violated constraint}}
    F[0](0)


# The constraint rides on the returned function type, so it is checked at the
# indirect call rather than at the call to the returning function.
# expected-note @below {{constraint declared here evaluated to False, expected '(Int(0) > Int(0))'}}
def violated_through_returned_value[n: Int]() -> def() thin -> None where n > 0:
    def inner():
        pass

    return inner


def call_returned_value():
    # expected-error @below {{invalid indirect call: violated constraint}}
    violated_through_returned_value[0]()()


# A `where` clause inside a function type's parameter list points at the
# trailing form.
def inline_where_in_fn_type[
    # expected-error @below {{'where' clauses inside parameter lists are no longer supported}}
    # expected-note @below {{use a trailing 'where' clause after the result type of a 'thin' function type instead}}
    f: def[a: Int where a > 0]() thin -> None
]():
    pass


# A `where` clause in a function type's argument list constrains nothing.
def where_in_fn_type_args[
    # expected-error @below {{'where' clauses must be used with parameters and cannot be used with arguments}}
    f: def(x: Int where x > 0) thin -> None
]():
    pass
