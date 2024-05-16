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
# RUN: %mojo %s 2>&1 1>/dev/null | FileCheck %s --check-prefix=CHECK-STDERR


import sys


# CHECK-LABEL: test_print_stderr
fn test_print_stderr():
    # CHECK-STDERR: stderr
    print("stderr", file=sys.stderr)
    # CHECK-STDERR: a/b/c
    print("a", "b", "c", sep="/", file=sys.stderr)
    # CHECK-STDERR: world
    print("world", flush=True, file=sys.stderr)
    # CHECK-STDERR: helloworld
    print("hello", end="world", file=sys.stderr)
    # CHECK-STDERR: hello world
    print(String("hello world"), file=sys.stderr)


fn main():
    test_print_stderr()
