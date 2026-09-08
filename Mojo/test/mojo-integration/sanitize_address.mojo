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
# This turns on asan itself
# UNSUPPORTED: asan
# RUN: %mojo-build -O0 --sanitize address %s -o %t
# RUN: export ASAN_OPTIONS=abort_on_error=1
# RUN: not not %t 2>&1 | FileCheck %s

from std.compile import compile_info


def main() raises:
    # CHECK: ERROR: AddressSanitizer: heap-buffer-overflow
    # CHECK: WRITE of size 8
    # CHECK: #0 {{.*}} in sanitize_address::main() {{.*}}sanitize_address.mojo:[[@LINE+3]]
    # CHECK: is located 0 bytes after 8-byte region
    var p = alloc[Int]({count = 1}).unsafe_leak()
    p[unsafe_offset=1] = 4
