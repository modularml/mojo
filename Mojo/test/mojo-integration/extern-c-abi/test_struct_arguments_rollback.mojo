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
# C reference: c_abi_test_rollback.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_rollback
# RUN: %t.dir/test_rollback | FileCheck %s

from std.ffi import external_call
from std.memory import Pointer, alloc, dealloc


# 12-byte struct: two eightbytes, both INTEGER class (IntegerPair).
@fieldwise_init
struct Int3(TrivialRegisterPassable):
    var x: Int32
    var y: Int32
    var z: Int32


# Control: struct passed first, so it fits in registers.
def test_struct_early():
    var v = Int3(Int32(11), Int32(22), Int32(33))
    var status = external_call["check_struct_early", Int32](v)
    print("struct_early:", status)


# CHECK: struct_early: 0


# Struct passed after five integer-class args: must roll back to the stack.
# Under the old per-argument lowering the struct was split, corrupting it.
def test_struct_after_five():
    var p0_alloc = alloc[Int32]({count = 1})
    var p0 = p0_alloc.unsafe_ptr()
    var p1_alloc = alloc[Int32]({count = 1})
    var p1 = p1_alloc.unsafe_ptr()
    var v = Int3(Int32(11), Int32(22), Int32(33))
    var status = external_call["check_struct_after_five", Int32](
        p0, p1, Int32(101), Int32(202), Int32(303), v
    )
    print("struct_after_five:", status)
    dealloc(p0_alloc^)
    dealloc(p1_alloc^)


# CHECK: struct_after_five: 0


def main():
    test_struct_early()
    test_struct_after_five()
