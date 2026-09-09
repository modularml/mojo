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


def dynamic_function(i: Int) -> Int:
    return i % 4


def process(i: Int):
    pass


def lookup_fn(count: Int):
    comptime list_of_values: List[Int] = [1, 3, 5, 7]

    for i in range(count):
        # Some computation, doesn't matter what it is.
        var idx = dynamic_function(i)

        # This is the problem
        var tmp: List[Int] = materialize[list_of_values]()
        var lookup = tmp[idx]
        # tmp is destroyed here

        # Use the value
        process(lookup)


def lookup_fn2(count: Int):
    comptime list_of_values: List[Int] = [1, 3, 5, 7]

    var list = materialize[list_of_values]()
    for i in range(count):
        # Some computation, doesn't matter what it is.
        var idx = dynamic_function(i)

        var lookup = list[idx]

        # Use the value
        process(lookup)


def main():
    lookup_fn(4)
    lookup_fn2(4)

    comptime str_literal = "Hello"  # at compile time, a StringLiteral
    var str = str_literal  # at run time, a String.
    var static_str: StaticString = str_literal  # or a StaticString

    # end of examples
    _, _ = str, static_str
