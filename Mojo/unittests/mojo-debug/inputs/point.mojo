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


struct Point(TrivialRegisterPassable):
    var x: Int
    var y: Int

    def __init__(out self, x: Int, y: Int):
        self.x = x  # breakpoint
        self.y = y
        return


def main():
    var p1 = Point(1, -1)
    var p2 = Point(2, -2)
    print(p1.x, p2.y)
