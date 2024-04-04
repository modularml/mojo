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

from testing import assert_equal, assert_raises

from utils import StringRef


def test_strref_from_start():
    var str = StringRef("Hello")

    assert_equal(5, len(str))

    assert_equal(str._from_start(0), "Hello")
    assert_equal(str._from_start(1), "ello")
    assert_equal(str._from_start(4), "o")
    assert_equal(str._from_start(5), "")
    assert_equal(str._from_start(10), "")

    assert_equal(str._from_start(-1), "o")
    assert_equal(str._from_start(-3), "llo")
    assert_equal(str._from_start(-5), "Hello")
    assert_equal(str._from_start(-10), "Hello")


def test_intable():
    assert_equal(int(StringRef("123")), 123)

    with assert_raises():
        int(StringRef("hi"))


def main():
    test_strref_from_start()
    test_intable()
