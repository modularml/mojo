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

from os import Atomic
from testing import assert_equal


fn test_atomic() raises:
    var atom: Atomic[DType.index] = 3

    assert_equal(atom.value, 3)

    assert_equal(atom.load(), 3)

    atom += 4
    assert_equal(atom.value, 7)

    atom -= 4
    assert_equal(atom.value, 3)

    atom.max(0)
    assert_equal(atom.value, 3)

    atom.max(42)
    assert_equal(atom.value, 42)

    atom.min(3)
    assert_equal(atom.value, 3)

    atom.min(0)
    assert_equal(atom.value, 0)


fn test_atomic_floating_point() raises:
    var atom: Atomic[DType.float32] = Float32(3.0)

    assert_equal(atom.value, 3.0)

    atom += 4
    assert_equal(atom.value, 7.0)

    atom -= 4
    assert_equal(atom.value, 3.0)

    atom.max(0)
    assert_equal(atom.value, 3.0)

    atom.max(42)
    assert_equal(atom.value, 42.0)

    atom.min(3)
    assert_equal(atom.value, 3.0)

    atom.min(0)
    assert_equal(atom.value, 0.0)


def test_atomic_move_constructor():
    var atom: Atomic[DType.index] = 3
    var atom2 = atom^
    assert_equal(atom2.value, 3)
    atom2 += 4
    assert_equal(atom2.value, 7)
    atom2 -= 4
    assert_equal(atom2.value, 3)
    atom2.max(0)
    assert_equal(atom2.value, 3)
    atom2.max(42)
    assert_equal(atom2.value, 42)
    atom2.min(3)
    assert_equal(atom2.value, 3)
    atom2.min(0)
    assert_equal(atom2.value, 0)


def main():
    test_atomic()
    test_atomic_floating_point()
    test_atomic_move_constructor()
