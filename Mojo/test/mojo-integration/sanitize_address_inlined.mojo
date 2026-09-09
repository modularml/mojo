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
# Verify that ASAN stack traces on macOS include inlined function source
# locations. At O3, user functions are inlined into main and the DWARF has
# proper DW_TAG_inlined_subroutine entries. This test ensures the ASAN runtime
# uses atos -i (supported since LLVM 22, llvm/llvm-project#170815) so that
# inlined frames are reported.
# This turns on asan itself
# UNSUPPORTED: asan, system-linux

# Build at O3 (default) where step_b is inlined into main.
# RUN: %mojo-build --sanitize address %s -o %t
# RUN: export ASAN_OPTIONS=abort_on_error=1
# RUN: not not %t 2>&1 | FileCheck %s

from std.sys import argv


# CHECK: ERROR: AddressSanitizer: heap-buffer-overflow
# CHECK: WRITE of size 8
# The key assertion: the inlined function name and source location must
# appear in the ASAN stack trace, not just "main" or "<unknown module>".
# CHECK: in {{.*}}step_b{{.*}} {{.*}}sanitize_address_inlined.mojo:[[@LINE+5]]


def step_b(n: Int):
    var p = alloc[Int]({count = 1}).unsafe_leak()
    p[unsafe_offset=n] = 42  # OOB write — ASAN reports this.
    print(
        p[unsafe_offset=0]
    )  # Side effect to prevent dead-code elimination at O3.


def main():
    step_b(argv()[0].byte_length())
