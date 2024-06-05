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

from collections import List

from testing import assert_equal


fn sum_items(data: List[Int8]) -> Int:
    var sum: Int = 0
    for i in range(len(data)):
        sum += int(data[i])
    return sum


fn make_abcd_vector() -> List[Int8]:
    return List[Int8](97, 98, 99, 100)


def main():
    var vec = make_abcd_vector()
    assert_equal(sum_items(vec), 394)
