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

# RUN: not %mojo %s 2>&1 | FileCheck %s

from std.ffi import OwnedDLHandle, c_int


def main() raises:
    # No path: opens the current process, so this needs no platform-specific
    # library name.
    var lib = OwnedDLHandle()
    var sym = lib.get_symbol[c_int]("errno")
    # Consuming the handle runs `dlclose`. A symbol pointer borrows the handle,
    # so that must be rejected while the pointer is still live.
    # CHECK: value 'lib' cannot be consumed, because it is used later
    var moved = lib^
    _ = sym
    _ = moved^
