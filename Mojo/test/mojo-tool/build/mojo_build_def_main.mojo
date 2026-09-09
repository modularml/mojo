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

# RUN: %mojo-build %s -o %t.bin
# RUN: not %t.bin --arg1 1> %t.out.txt 2> %t.err.txt
# RUN cat %t.out.txt | FileCheck %s --check-prefix OUT
# RUN cat %t.err.txt | FileCheck %s --check-prefix ERR

from std.sys import argv, stderr


def main() raises:
    # OUT: This was called inside of `def` main
    print("This was called inside of `def` main")

    # ERR: This is printed to stderr
    print("This is printed to stderr", file=stderr)

    # OUT: --arg1
    print(argv()[1])

    # ERR: Unhandled exception caught during execution: main raised an error
    raise Error("main raised an error")
