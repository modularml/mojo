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

from utils import InlineArray, Span
from collections.list import List
from testing import *


def test_span_list_int():
    var l = List[Int](1, 2, 3, 4, 5, 6, 7)
    var s = Span(l)
    assert_equal(len(s), len(l))
    for i in range(len(s)):
        assert_equal(l[i], s[i])
    # subslice
    var s2 = s[2:]
    assert_equal(s2[0], l[2])
    assert_equal(s2[1], l[3])
    assert_equal(s2[2], l[4])
    assert_equal(s2[3], l[5])
    assert_equal(s[-1], l[-1])


def test_span_list_str():
    var l = List[String]("a", "b", "c", "d", "e", "f", "g")
    var s = Span(l)
    assert_equal(len(s), len(l))
    for i in range(len(s)):
        assert_equal(l[i], s[i])
    # subslice
    var s2 = s[2:]
    assert_equal(s2[0], l[2])
    assert_equal(s2[1], l[3])
    assert_equal(s2[2], l[4])
    assert_equal(s2[3], l[5])


def test_span_array_int():
    var l = InlineArray[Int, 7](1, 2, 3, 4, 5, 6, 7)
    var s = Span(l)
    assert_equal(len(s), len(l))
    for i in range(len(s)):
        assert_equal(l[i], s[i])
    # subslice
    var s2 = s[2:]
    assert_equal(s2[0], l[2])
    assert_equal(s2[1], l[3])
    assert_equal(s2[2], l[4])
    assert_equal(s2[3], l[5])


def test_span_array_str():
    var l = InlineArray[String, 7]("a", "b", "c", "d", "e", "f", "g")
    var s = Span(l)
    assert_equal(len(s), len(l))
    for i in range(len(s)):
        assert_equal(l[i], s[i])
    # subslice
    var s2 = s[2:]
    assert_equal(s2[0], l[2])
    assert_equal(s2[1], l[3])
    assert_equal(s2[2], l[4])
    assert_equal(s2[3], l[5])


def main():
    test_span_list_int()
    test_span_list_str()
    test_span_array_int()
    test_span_array_str()
