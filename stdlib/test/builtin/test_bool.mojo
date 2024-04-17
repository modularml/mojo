# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo -debug-level full %s

from testing import assert_equal


def test_bool_cast_to_int():
    assert_equal(False.__int__(), 0)
    assert_equal(True.__int__(), 1)

    assert_equal(int(False), 0)
    assert_equal(int(True), 1)


def test_bool_none():
    var test = None
    assert_equal(bool(None), False)
    assert_equal(bool(test), False)


def test_bool_abs():
    assert_equal(False.__abs__(), 0)
    assert_equal(True.__abs__(), 1)


def test_bool_add():
    assert_equal(True.__add__(1), 2)
    assert_equal(False.__add__(1), 1)


def test_bool_float():
    assert_equal(True.__float__(), 1.0)
    assert_equal(False.__float__(), 0.0)


def main():
    test_bool_cast_to_int()
    test_bool_none()
    test_bool_abs()
    test_bool_add()
    test_bool_float()
