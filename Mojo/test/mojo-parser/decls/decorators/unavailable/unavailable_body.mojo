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

# Tests that @unavailable functions accept '...' as their body even when they
# have a non-None return type, and reject any other body content.

# RUN: %parse-mojo-isolated -verify-diagnostics %s


# ===----------------------------------------------------------------------=== #
# Test: '...' body is accepted for @unavailable function with no return type
# ===----------------------------------------------------------------------=== #


@unavailable("don't use")
def ok_unavailable_void():
    ...


# ===----------------------------------------------------------------------=== #
# Test: '...' body is accepted for @unavailable function with a return type
# ===----------------------------------------------------------------------=== #


@unavailable("don't use")
def ok_unavailable_int_return() -> Int:
    ...


@unavailable("don't use")
def ok_unavailable_string_return() -> String:
    ...


# ===----------------------------------------------------------------------=== #
# Test: '...' body is accepted for @unavailable method with a return type
# ===----------------------------------------------------------------------=== #


struct StringLike(Movable where False):
    @unavailable("no length for 'StringLike'; use byte_length() or codepoint_length() instead")
    def __len__(self) -> Int:
        ...


# ===----------------------------------------------------------------------=== #
# Test: non-'...' body is rejected on @unavailable function
# ===----------------------------------------------------------------------=== #


@unavailable("don't use")
# expected-note @below {{in 'rejects_pass_body', declared here}}
def rejects_pass_body():
    # expected-error @below {{unexpected function body in @unavailable function declaration, use '...'}}
    pass


@unavailable("don't use")
# expected-note @below {{in 'rejects_return_body', declared here}}
def rejects_return_body() -> Int:
    # expected-error @below {{unexpected function body in @unavailable function declaration, use '...'}}
    return 42
