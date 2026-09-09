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


def test(st: String):
    print(st)  # breakpoint


struct Point(TrivialRegisterPassable):
    var x: Int
    var y: Int

    def __init__(out self, x: Int, y: Int):
        self.x = x
        self.y = y


def main():
    var p2 = Point(2, 2)
    var literal: StaticString = "string_literal"
    var s1 = "let_string"
    var s2 = String()
    for i in range(0, 100):
        s2 += String(i)
    var s3 = String()
    test(s2)
    var s4 = Pointer(to=s2)
    print(literal, s1, s2, s3, end="")  # breakpoint
    print(s4)
    _ = p2
