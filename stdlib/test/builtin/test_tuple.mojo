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

from testing import assert_true, assert_false
from sys import os_is_macos


def test_tuple_contains():
    var a = (123, True, "Mojo is awesome")

    assert_true("Mojo is awesome" in a)
    assert_true(a.__contains__("Mojo is awesome"))

    assert_false("Hello world" in a)
    assert_false(a.__contains__("Hello world"))

    assert_true(123 in a)
    assert_true(a.__contains__(123))

    assert_true(True in a)
    assert_true(a.__contains__(True))

    assert_false(False in a)
    assert_false(a.__contains__(False))

    assert_false(a.__contains__(1))
    assert_false(a.__contains__(0))
    assert_false(1 in a)
    assert_false(0 in a)

    var b = (False, True)
    assert_true(True in b)
    assert_true(b.__contains__(True))
    assert_true(False in b)
    assert_true(b.__contains__(False))
    assert_false(b.__contains__(1))
    assert_false(b.__contains__(0))

    var c = (1, 0)
    assert_false(c.__contains__(True))
    assert_false(c.__contains__(False))
    assert_false(True in c)
    assert_false(False in c)

    var d = (123, True, String("Mojo is awesome"))

    assert_true(String("Mojo is awesome") in d)
    assert_true(d.__contains__(String("Mojo is awesome")))

    assert_false(String("Hello world") in d)
    assert_false(d.__contains__(String("Hello world")))


def main():
    # FIXME(MSTDL-516)
    if not os_is_macos():
        test_tuple_contains()
