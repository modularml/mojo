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

from std.testing import *


def construct_1() raises:
    var s: String = "Testing"
    s += " Mojo strings"
    assert_true(s == "Testing Mojo strings")


def construct_2() raises:
    var s = "Items in list: " + String(5)
    assert_true(s == "Items in list: 5")

    s = String("Items in list: ", 5)
    assert_true(s == "Items in list: 5")


def format_1() raises:
    var s = "{0} {1} {0}".format("Mojo", 1.125)
    assert_true(s == "Mojo 1.125 Mojo")
    var s2 = "{} {}".format(True, "hello world")
    assert_true(s2 == "True hello world")


def template_1() raises:
    var count = 3
    var items = "apples"
    var template = t"Give me {count} {items}."  # Template string
    assert_true(String(template) == "Give me 3 apples.")

    var x = 41
    assert_true(String(t"The answer is {x + 1}") == "The answer is 42")

    var name = "Nate"
    var template2 = t"Hello, {name}!"  # template creation
    assert_true(String(template2) == "Hello, Nate!")

    var list: List[Int] = [1, 2, 3]
    assert_true(String(t"{list[0] + list[1]}") == "3")

    # Literal braces use {{ and }}.
    assert_true(String(t"Use {{braces}} like this") == "Use {braces} like this")


def main() raises:
    construct_1()
    construct_2()
    format_1()
    template_1()
