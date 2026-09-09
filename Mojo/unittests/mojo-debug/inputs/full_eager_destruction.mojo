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


def get_string() -> String:
    var s = "hel"
    s += "lo"  # defeat the string literal optimization.
    return s


def get_number() -> Int:
    return 8


def take_string(var s: String):
    print(s)  # breakpoint


def take_number(var i: Int):
    print(i)


def main():
    var text = get_string()
    print(text)  # breakpoint

    var number = get_number()
    for i in range(2):
        print(number)  # breakpoint
    print(0)  # breakpoint

    var simd = SIMD[.int16, 4](1, 2, 3, 4)
    if simd[0] < 0:
        print(simd)
    else:
        print(0)  # breakpoint

    var text_moved = get_string()
    take_string(text_moved^)  # breakpoint

    var text_copied = get_string()
    take_string(text_copied)  # breakpoint

    var text_before = get_string()
    var text_after = text_before^
    print(text_after)  # breakpoint

    var number2 = get_number()
    take_number(number2)  # breakpoint
    print(0)  # breakpoint
