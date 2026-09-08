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

# RUN: %mojo %s | FileCheck %s

from std.collections import Deque
from std.collections.deque import _DequeIter


def not_a_list[
    T: Copyable & Deinitable
](ref value: Deque[T]) -> _DequeIter[T, origin_of(value), False]:
    return value.__reversed__()


def needs_default[
    IterableTypeA: Iterable
](ref iterable_a: IterableTypeA) -> IterableTypeA.IteratorType[
    origin_of(iterable_a)
]:
    return iter(iterable_a)


def main() raises:
    # COM: The motivating bug. Ensure no regression.
    # CHECK: 77 88
    # CHECK: 99 66
    for a, b in zip([77, 99], [88, 66]):
        print(a, b)

    # COM: the type cannot be inferred from the trait, fallback to list.
    # CHECK: 4
    # CHECK: 5
    # CHECK: 6
    for c in needs_default([4, 5, 6]):
        print(c)

    # COM: Ensure the specified type gets priority and does not fallback to list.
    # CHECK: 3
    # CHECK: 2
    # CHECK: 1
    for d in not_a_list([1, 2, 3]):
        print(d)

    # COM: Dictionary literals also work
    # CHECK: honda
    # CHECK: accord
    for key in needs_default({"honda": 3, "accord": 1}):
        print(key)
