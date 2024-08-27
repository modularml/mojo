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
# RUN: %mojo %s
from testing import assert_equal, assert_true


def test_copy_reference_explicitly():
    var a = List[Int](1, 2, 3)

    var b = Reference(a)

    var c = Reference(other=b)

    c[][0] = 4
    assert_equal(a[0], 4)
    assert_equal(b[][0], 4)
    assert_equal(c[][0], 4)


def test_equality():
    var a = List[Int](1, 2, 3)
    var b = List[Int](4, 5, 6)

    assert_true(Reference(a) == Reference(a))
    assert_true(Reference(b) == Reference(b))
    assert_true(Reference(a) != Reference(b))


def test_str():
    var a = Int(42)
    var a_ref = Reference(a)
    assert_true(str(a_ref).startswith("0x"))


def main():
    test_copy_reference_explicitly()
    test_equality()
    test_str()
