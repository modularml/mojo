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
# RUN: %mojo -debug-level full %s | FileCheck %s

from os.atomic import Atomic


# CHECK-LABEL: test_atomic
fn test_atomic():
    print("== test_atomic")

    var atom: Atomic[DType.index] = 3

    # CHECK: 3
    print(atom.load())

    # CHECK: 3
    print(atom.value)

    atom += 4

    # CHECK: 7
    print(atom.value)

    atom -= 4

    # CHECK: 3
    print(atom.value)

    # CHECK: 3
    atom.max(0)
    print(atom.value)

    # CHECK: 42
    atom.max(42)
    print(atom.value)

    # CHECK: 3
    atom.min(3)
    print(atom.value)

    # CHECK: 0
    atom.min(0)
    print(atom.value)


# CHECK-LABEL: test_atomic_floating_point
fn test_atomic_floating_point():
    print("== test_atomic_floating_point")

    var atom: Atomic[DType.float32] = Float32(3.0)

    # CHECK: 3.0
    print(atom.value)

    atom += 4

    # CHECK: 7.0
    print(atom.value)

    atom -= 4

    # CHECK: 3.0
    print(atom.value)

    # CHECK: 3.0
    atom.max(0)
    print(atom.value)

    # CHECK: 42.0
    atom.max(42)
    print(atom.value)

    # CHECK: 3.0
    atom.min(3)
    print(atom.value)

    # CHECK: 0.0
    atom.min(0)
    print(atom.value)


fn main():
    test_atomic()
    test_atomic_floating_point()
