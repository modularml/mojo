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

# ===----------------------------------------------------------------------=== #
#
# File originates from:
#   Repo:   git@github.com:psf/black.git
#   Commit: d4a85643a465f5fae2113d07d22d021d4af4795a
#   Path:   tests/data/simple_cases/class_blank_parentheses.py
#
# ===----------------------------------------------------------------------=== #


class SimpleClassWithBlankParentheses:
    pass


class ClassWithSpaceParentheses:
    first_test_data = 90
    second_test_data = 100

    def test_func(self):
        return None


class ClassWithEmptyFunc(object):
    def func_with_blank_parentheses():
        return 5


def public_func_with_blank_parentheses():
    return None


def class_under_the_func_with_blank_parentheses():
    class InsideFunc:
        pass


class NormalClass:
    def func_for_testing(self, first, second):
        sum = first + second
        return sum


# output


class SimpleClassWithBlankParentheses:
    pass


class ClassWithSpaceParentheses:
    first_test_data = 90
    second_test_data = 100

    def test_func(self):
        return None


class ClassWithEmptyFunc(object):
    def func_with_blank_parentheses():
        return 5


def public_func_with_blank_parentheses():
    return None


def class_under_the_func_with_blank_parentheses():
    class InsideFunc:
        pass


class NormalClass:
    def func_for_testing(self, first, second):
        sum = first + second
        return sum
