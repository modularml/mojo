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

# RUN: %parse-mojo-isolated %s -verify-diagnostics

# MOCO-3472: a non-empty tuple literal used in type position must be
# diagnosed directly, instead of falling through to Tuple's constructor-call
# machinery and failing with a confusing trait-conformance error.


# expected-error @+1 {{expected a type, found a tuple value; use 'Tuple[...]' to write a tuple type}}
def returns_tuple_type() -> (Int, Int):
    return (Int(1), Int(2))


def takes_tuple_type_param(
    # expected-error @+1 {{expected a type, found a tuple value; use 'Tuple[...]' to write a tuple type}}
    x: (Int, Int)
):
    pass


def local_tuple_type_annotation():
    # expected-error @+1 {{expected a type, found a tuple value; use 'Tuple[...]' to write a tuple type}}
    var x: (Int, Int)


def single_element_tuple_type():
    # expected-error @+1 {{expected a type, found a tuple value; use 'Tuple[...]' to write a tuple type}}
    var x: (Int,)


# The empty tuple is the one legitimate type/value ambiguity and must
# continue to work as sugar for 'Tuple[]'.
def returns_empty_tuple_type() -> ():
    return ()
