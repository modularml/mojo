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

# Test that the default Equatable implementation works for single-field
# structs of TrivialRegisterPassable.

from std.testing import assert_true, assert_false


@fieldwise_init
struct SingleFieldTrivial(Equatable, TrivialRegisterPassable):
    var value: Int


def main() raises:
    var a = SingleFieldTrivial(42)
    var b = SingleFieldTrivial(42)
    var c = SingleFieldTrivial(10)

    assert_true(a == b, "equal values should be equal")
    assert_false(a == c, "different values should not be equal")
    assert_true(a != c, "different values should be not-equal")
    assert_false(a != b, "equal values should not be not-equal")
