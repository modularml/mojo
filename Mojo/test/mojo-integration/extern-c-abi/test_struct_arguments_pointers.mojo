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
# C reference: c_abi_test_ptr_structs.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_pointers
# RUN: %t.dir/test_pointers | FileCheck %s

from std.ffi import external_call
from std.memory import Pointer, alloc, dealloc


# ============================================================================
# 16-byte struct (pointer + Int32 + padding)
# ARM64 AAPCS: IntegerPair class (two registers)
# ============================================================================
@fieldwise_init
struct PtrInt32Struct(TrivialRegisterPassable):
    var p: Pointer[Int32, MutUntrackedOrigin]
    var i: Int32


def test_ptr_int32():
    var p_alloc = alloc[Int32]({count = 1})
    var p = p_alloc.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    p[] = 999
    var s = PtrInt32Struct(p, Int32(200))
    var result = external_call["c_func_ptr_int32", PtrInt32Struct](s)
    # C increments pointer by 1 byte and int by 1
    print("ptr_int32:", result.i)
    # Verify original allocation still accessible
    print("ptr_int32_val:", p[])
    dealloc(p_alloc^)


# CHECK: ptr_int32: 201
# CHECK: ptr_int32_val: 999


# ============================================================================
# 24-byte struct (three pointers) - MEMORY class
# ============================================================================
@fieldwise_init
struct ThreePtrStruct(TrivialRegisterPassable):
    var a: Pointer[Int32, MutUntrackedOrigin]

    var b: Pointer[Int32, MutUntrackedOrigin]

    var c: Pointer[Int32, MutUntrackedOrigin]


def test_three_ptr():
    var pa_alloc = alloc[Int32]({count = 1})
    var pa = pa_alloc.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var pb_alloc = alloc[Int32]({count = 1})
    var pb = pb_alloc.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var pc_alloc = alloc[Int32]({count = 1})
    var pc = pc_alloc.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    pa[] = 111
    pb[] = 222
    pc[] = 333
    var s = ThreePtrStruct(pa, pb, pc)
    var result = external_call["c_func_three_ptr", ThreePtrStruct](s)
    # C advances each pointer by 1 byte, but original allocations are intact
    print("three_ptr:", pa[], pb[], pc[])
    dealloc(pa_alloc^)
    dealloc(pb_alloc^)
    dealloc(pc_alloc^)


# CHECK: three_ptr: 111 222 333


# ============================================================================
# Main - run all tests
# ============================================================================
def main():
    test_ptr_int32()

    test_three_ptr()
