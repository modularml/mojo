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

# RUN: %mojo -debug-level full %s | FileCheck %s


def diverge_comptime(i: Int) -> Int:
    var t: Int
    # Intentionally have then and else branches mismatch logic for testing.
    if __is_run_in_comptime_interpreter:
        t = i + 40
    else:
        t = i + 1

    return t + 1


def test_is_run_in_comptime_interpreter():
    # CHECK-LABEL: testing test_is_run_in_comptime_interpreter
    print("testing test_is_run_in_comptime_interpreter")

    # CHECK: interpret value: 42
    comptime a = diverge_comptime(1)
    print("interpret value:", a)

    # CHECK: runtime value: 3
    var b = diverge_comptime(1)
    print("runtime value:", b)


def might_throw(cond: Bool) -> Int:
    var result = 0
    try:
        if cond:
            raise "something"

        result = 4
    except e:
        return String(e).byte_length() * 4

    else:
        result += 1
    finally:
        result += 2

    return result


# MOCO-246: Test EH at comptime.
def test_exception_handling():
    # CHECK-LABEL: test_exception_handling
    print("test_exception_handling")

    # CHECK: interpret value: 36
    comptime a = might_throw(True)
    print("interpret value:", a)

    # CHECK: run value: 36
    print("run value:", might_throw(True))

    # CHECK: interpret value: 7
    comptime b = might_throw(False)
    print("interpret value:", b)

    # CHECK: run value: 7
    print("run value:", might_throw(False))


def main():
    test_is_run_in_comptime_interpreter()
    test_exception_handling()
