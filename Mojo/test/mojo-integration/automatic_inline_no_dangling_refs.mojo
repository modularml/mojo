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
#
# Regression test for MOCO-3737: storing a `def` into a struct field of a thin
# function pointer type, when the same function is also referenced directly at
# another site in the same body, would leave the function pointer's backing
# declaration dangling after AutomaticInline.
#
# RUN: %mojo %s | FileCheck %s

from std.memory import Pointer


comptime PackFn = def(
    Pointer[UInt8, MutAnyOrigin],
    Pointer[UInt8, MutAnyOrigin],
    Int,
    Int,
) thin -> None


def pack_noop(
    src: Pointer[UInt8, MutAnyOrigin],
    dst: Pointer[UInt8, MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    pass


@fieldwise_init
struct Entry(Copyable):
    var pack_fn: PackFn


def main():
    var out = List[Entry]()
    # Referencing `pack_noop` twice in the same body — once indirectly via a
    # local and once directly — used to cause one of the references to be
    # dropped by `AutomaticInline` while the other remained.
    var nop = pack_noop
    out.append(Entry(pack_fn=nop))
    out.append(Entry(pack_fn=pack_noop))
    print("len:", len(out))
    # CHECK: len: 2
