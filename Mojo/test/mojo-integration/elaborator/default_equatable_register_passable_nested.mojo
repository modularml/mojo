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
# RUN: mojo %s

# Test that the default Equatable implementation works for nested
# RegisterPassable structs. This is a regression test for MOCO-4259.

from std.testing import assert_true, assert_false


# Multi-field RegisterPassable struct
@fieldwise_init
struct Inner(Equatable, RegisterPassable):
    var x: Int
    var y: Int


# Single-field RegisterPassable struct containing a multi-field struct.
# This is the problematic case: Outer is isSingleElement() due to having
# one field, so it gets "flattened" to Inner's representation. The default
# Equatable implementation must handle this correctly.
@fieldwise_init
struct Outer(Equatable, RegisterPassable):
    var inner: Inner


def main() raises:
    # Test Inner (multi-field) - this should work
    var i1 = Inner(1, 2)
    var i2 = Inner(1, 2)
    var i3 = Inner(3, 4)

    assert_true(i1 == i2, "equal Inner values should be equal")
    assert_false(i1 == i3, "different Inner values should not be equal")

    # Test Outer (nested) - this was failing due to struct flattening
    var o1 = Outer(Inner(1, 2))
    var o2 = Outer(Inner(1, 2))
    var o3 = Outer(Inner(3, 4))

    assert_true(o1 == o2, "equal Outer values should be equal")
    assert_false(o1 == o3, "different Outer values should not be equal")
