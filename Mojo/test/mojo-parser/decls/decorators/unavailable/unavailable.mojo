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

# Tests for @unavailable decorator IR generation (LIT tests).
# For error emission tests, see unavailable_errors.mojo.
#
# @unavailable is only supported on functions and methods. Tests for the
# restriction on structs/traits/comptime aliases live in
# unavailable_errors.mojo.

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s


# ===----------------------------------------------------------------------=== #
# Test: Basic @unavailable IR generation on a function
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.fn @"unavailable_func
# CHECK-SAME: unavailableInfo = #lit.unavailable<"func">
@unavailable("func")
def unavailable_func():
    ...


# ===----------------------------------------------------------------------=== #
# Test: @unavailable(use=...) on a function
# ===----------------------------------------------------------------------=== #


def unavailable_func_target():
    pass


# CHECK-LABEL: lit.fn @"unavailable_func_use
# CHECK-SAME: unavailableInfo = #lit.unavailable<"'unavailable_func_use' is unavailable, use 'unavailable_func_target' instead", "unavailable_func_target">
@unavailable(use=unavailable_func_target)
def unavailable_func_use():
    ...


# ===----------------------------------------------------------------------=== #
# Test: @unavailable(use=...) on methods
# ===----------------------------------------------------------------------=== #


struct MethodUnavailableTest(Movable where False):
    def replacement_method(self):
        pass

    # CHECK-LABEL: lit.fn @"unavailable_method_use
    # CHECK-SAME: unavailableInfo = #lit.unavailable<"'unavailable_method_use' is unavailable, use 'replacement_method' instead", "replacement_method">
    @unavailable(use=replacement_method)
    def unavailable_method_use(self):
        ...


struct StaticMethodUnavailableTest(Movable where False):
    @staticmethod
    def replacement_static():
        pass

    # CHECK-LABEL: lit.fn @"unavailable_static_use
    # CHECK-SAME: unavailableInfo = #lit.unavailable<"'unavailable_static_use' is unavailable, use 'replacement_static' instead", "replacement_static">
    @staticmethod
    @unavailable(use=replacement_static)
    def unavailable_static_use():
        ...


# ===----------------------------------------------------------------------=== #
# Test: @unavailable on a method with a non-None return type, body is `...`
# ===----------------------------------------------------------------------=== #


struct StringLike(Movable where False):
    # CHECK-LABEL: lit.fn @"__len__
    # CHECK-SAME: unavailableInfo = #lit.unavailable<"no length for 'StringLike'; use byte_length() or codepoint_length() instead">
    @unavailable("no length for 'StringLike'; use byte_length() or codepoint_length() instead")
    def __len__(self) -> Int:
        ...
