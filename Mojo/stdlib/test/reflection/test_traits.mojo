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
"""Tests for `TypeList.all_conforms_to` trait-conformance checks."""

from std.builtin.variadics import TypeList
from std.testing import assert_true, assert_false
from std.testing import TestSuite
from test_utils import ExplicitDelOnly


struct NoConformances(Movable where False):
    pass


def test_all_conforms_to[Trait: type_of(AnyType)]() raises:
    assert_true(TypeList.of[Int]().all_conforms_to[Trait]())
    assert_true(TypeList.of[Int, String, Float64]().all_conforms_to[Trait]())
    assert_false(TypeList.of[Int, NoConformances]().all_conforms_to[Trait]())
    assert_false(TypeList.of[NoConformances]().all_conforms_to[Trait]())


def main() raises:
    test_all_conforms_to[Writable]()
    test_all_conforms_to[Movable]()
    test_all_conforms_to[Copyable]()
    test_all_conforms_to[ImplicitlyCopyable]()
    test_all_conforms_to[Defaultable]()
    test_all_conforms_to[Equatable]()
    test_all_conforms_to[Hashable]()
