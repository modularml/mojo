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


def main():
    # While loop
    var fib_prev = 0
    var fib_curr = 1

    print(fib_prev, end="")
    while fib_curr < 50:
        print(",", fib_curr, end="")
        fib_prev, fib_curr = fib_curr, fib_prev + fib_curr

    print()

    # While loop with continue statement
    var n = 0
    while n < 5:
        n += 1
        if n == 3:
            continue
        print(n, end=", ")

    print()

    # While loop with break statement
    n = 0
    while n < 5:
        n += 1
        if n == 3:
            break
        print(n, end=", ")

    print()

    # While loop with else statement
    n = 5
    while n < 4:
        print(n)
        n += 1
    else:
        print("Loop completed")

    print()

    # While loop with break and else statement
    n = 0
    while n < 5:
        n += 1
        if n == 3:
            break
        print(n)
    else:
        print("Executing else clause")
