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

from testing import assert_equal, assert_true


def test_bool_cast_to_int():
    assert_equal(False.__int__(), 0)
    assert_equal(True.__int__(), 1)

    assert_equal(int(False), 0)
    assert_equal(int(True), 1)


def test_bool_none():
    var test = None
    assert_equal(bool(None), False)
    assert_equal(bool(test), False)


@value
struct MyTrue:
    fn __bool__(self) -> Bool:
        return True


fn takes_bool(cond: Bool) -> Bool:
    return cond


def test_convert_from_boolable():
    assert_true(takes_bool(MyTrue()))


def test_bool_to_string():
    assert_equal(str(True), "True")
    assert_equal(str(False), "False")


def main():
    test_bool_cast_to_int()
    test_bool_none()
    test_convert_from_boolable()
    test_bool_to_string()
