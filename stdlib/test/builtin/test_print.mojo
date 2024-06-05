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
# RUN: %mojo %s | FileCheck %s


import sys

from memory import DTypePointer

from utils import StaticIntTuple, StringRef


# CHECK-LABEL: test_print
fn test_print():
    print("== test_print")

    # CHECK: Hello
    print("Hello")

    # CHECK: World
    print("World", flush=True)

    var hello: StringRef = "Hello,"
    var world: String = "world!"
    var f: Bool = False
    # CHECK: > Hello, world! 42 True False
    print(">", hello, world, 42, True, f)

    # CHECK: > 3.14000{{[0-9]+}} 99.90000{{[0-9]+}} -129.29018{{[0-9]+}} (1, 2, 3)
    var float32: Float32 = 99.9
    var float64: Float64 = -129.2901823
    print("> ", end="")
    print(3.14, float32, float64, StaticIntTuple[3](1, 2, 3), end="")
    print()

    # CHECK: > 9223372036854775806
    print(">", 9223372036854775806)

    var pi = 3.1415916535897743
    # CHECK: > 3.1415916535{{[0-9]+}}
    print(">", pi)
    var x = (pi - 3.141591) * 1e6
    # CHECK: > 0.6535{{[0-9]+}}
    print(">", x)

    # CHECK: Hello world
    print(String("Hello world"))


# CHECK-LABEL: test_print_end
fn test_print_end():
    print("== test_print_end")
    # CHECK: Hello World
    print("Hello", end=" World\n")


# CHECK-LABEL: test_print_sep
fn test_print_sep():
    print("== test_print_sep")

    # CHECK: a/b/c
    print("a", "b", "c", sep="/")

    # CHECK: a/1/2xx
    print("a", 1, 2, sep="/", end="xx\n")


fn main():
    test_print()
    test_print_end()
    test_print_sep()
