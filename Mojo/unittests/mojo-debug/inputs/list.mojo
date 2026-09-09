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

from debug_test_utils import keep_alive


struct Point(TrivialRegisterPassable):
    var x: Int
    var y: Int

    def __init__(out self, x: Int, y: Int):
        self.x = x
        self.y = y


def main():
    var point_vec = List[Point](capacity=3)
    var p1 = Point(1, -1)
    var p2 = Point(2, -2)
    var p3 = Point(3, -3)
    point_vec.append(p1)
    point_vec.append(p2)
    point_vec.append(p3)
    var value = point_vec[0].x  # breakpoint
    print(value)

    var int_vec = List[Int](capacity=3)
    int_vec.append(1)
    int_vec.append(2)
    int_vec.append(3)
    print(len(int_vec))  # breakpoint

    for i in range(0, 100):
        int_vec.append(i)
    keep_alive(int_vec)  # breakpoint
