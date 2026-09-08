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

# RUN: %bare-mojo build --target-triple=i686-unknown-linux-gnu --target-cpu=i686 --emit=llvm %s -o - | FileCheck %s

# Test that C types have correct sizes on 32-bit targets (ILP32 data model).
# On 32-bit systems:
#   - c_long is 32-bit (4 bytes)
#   - c_long_long is 64-bit (8 bytes)
#
# Reference: https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models
#
# Before this fix, compiling for 32-bit targets would fail with:
#   constraint failed: size of C `long` is unknown on this target

from std.ffi import c_long, c_long_long, c_ulong, c_ulong_long
from std.sys.info import size_of

# CHECK: target datalayout = "e-m:e-p:32:32


def main() raises:
    # Compile-time assertions to verify correct sizes on 32-bit target
    # ILP32: long is 32-bit (4 bytes), long long is 64-bit (8 bytes)
    comptime assert size_of[c_long]() == 4, "c_long should be 4 bytes on 32-bit"
    comptime assert (
        size_of[c_ulong]() == 4
    ), "c_ulong should be 4 bytes on 32-bit"
    comptime assert (
        size_of[c_long_long]() == 8
    ), "c_long_long should be 8 bytes on 32-bit"
    comptime assert (
        size_of[c_ulong_long]() == 8
    ), "c_ulong_long should be 8 bytes on 32-bit"

    # Use the types to ensure they work
    var a: c_long = 1
    var b: c_ulong = 2
    var c: c_long_long = 3
    var d: c_ulong_long = 4
    _ = a
    _ = b
    _ = c
    _ = d
