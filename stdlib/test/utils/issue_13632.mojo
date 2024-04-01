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

from collections import List


fn sum_items(data: List[Int8]) -> Int:
    var sum: Int = 0
    for i in range(len(data)):
        sum += int(data[i])
    return sum


fn make_abcd_vector() -> List[Int8]:
    return List[Int8](97, 98, 99, 100)


fn main():
    var vec = make_abcd_vector()
    # CHECK: 394
    print(sum_items(vec))
