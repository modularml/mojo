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
# RUN: %mojo %s 2>&1 1>/dev/null | FileCheck %s --check-prefix=CHECK-STDERR


import std.sys

from std.testing import TestSuite


# CHECK-LABEL: test_print_stderr
def test_print_stderr() raises:
    # CHECK-STDERR: stderr
    print("stderr", file=std.sys.stderr)
    # CHECK-STDERR: a/b/c
    print("a", "b", "c", sep="/", file=std.sys.stderr)
    # CHECK-STDERR: world
    print("world", flush=True, file=std.sys.stderr)
    # CHECK-STDERR: helloworld
    print("hello", end="world", file=std.sys.stderr)
    # CHECK-STDERR: hello world
    print("hello world", file=std.sys.stderr)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
