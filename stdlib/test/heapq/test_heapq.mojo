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

from heapq import heapify, heappush, heappop
from testing import assert_equal


def main():
    var l = List(3, 6, 7, 8, 3, 7)
    heapify(l)
    # TODO: use __eq__ when implemented for List
    assert_equal(l.__str__(), "[3, 3, 7, 8, 6, 7]")
    heappush(l, 5)
    assert_equal(l.__str__(), "[3, 3, 5, 8, 6, 7, 7]")

    assert_equal(heappop(l), 3)
    assert_equal(l.__str__(), "[3, 6, 5, 8, 7, 7]")

    l = List(57, 3467, 734, 4, 6, 8, 236, 367, 236, 75, 87)
    heapify(l)
    assert_equal(l.__str__(), "[4, 6, 8, 236, 57, 734, 236, 367, 3467, 75, 87]")

    # sanity check that nothing bad happens
    var l2 = List[Int]()
    heapify(l2)
