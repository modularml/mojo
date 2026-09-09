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


def write_through(lib: OwnedDLHandle) raises:
    # `lib` is an immutable borrow, so the symbol it resolves is read-only.
    # CHECK: expression must be mutable in assignment
    lib.get_symbol[c_int]("errno").value()[] = 0


def main() raises:
    write_through(OwnedDLHandle())
