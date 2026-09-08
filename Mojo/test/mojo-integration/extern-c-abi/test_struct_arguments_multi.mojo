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
# C reference: c_abi_test_multi_args.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_multi
# RUN: %t.dir/test_multi | FileCheck %s
#
# The sibling test files each pass a single struct, which pins classification
# but not allocation. These calls mix aggregates with scalars, so a struct has
# to find its slot with part of the register file already consumed.

from std.ffi import external_call


# ============================================================================
# Scalar preceding a 36-byte MEMORY-class struct
# ============================================================================
@fieldwise_init
struct MultiInt36(TrivialRegisterPassable):
    var a: Int32
    var b: Int32
    var c: Int32
    var d: Int32
    var e: Int32
    var f: Int32
    var g: Int32
    var h: Int32
    var i: Int32


def test_scalar_then_memory_struct():
    var s = MultiInt36(1, 2, 3, 4, 5, 6, 7, 8, 9)
    var result = external_call["c_func_scalar_then_memory_struct", Int64](
        Int32(7), s
    )
    print("scalar_then_memory_struct:", result)


# CHECK: scalar_then_memory_struct: 7987654321


# ============================================================================
# Scalar preceding a 4-byte struct with one byte of tail padding
# ============================================================================
@fieldwise_init
struct MultiPadShort(TrivialRegisterPassable):
    var a: Int16
    var b: Int8


def test_scalar_then_padded_small():
    var s = MultiPadShort(3, 5)
    var result = external_call["c_func_scalar_then_padded_small", Int64](
        Int32(7), s
    )
    print("scalar_then_padded_small:", result)


# CHECK: scalar_then_padded_small: 70305


# ============================================================================
# HFA arriving with the SIMD registers consumed
# ============================================================================
@fieldwise_init
struct MultiHfa2D(TrivialRegisterPassable):
    var a: Float64
    var b: Float64


# TODO(MOCO-4611): Add the seven-leading-double case, which leaves one SIMD
# register — one fewer than the HFA needs — so AAPCS must burn it and pass the
# aggregate whole on the stack. Mojo splits it there instead.
def test_eight_doubles_then_hfa2d():
    """No SIMD register is left at all, which reaches plain stack passing."""
    var s = MultiHfa2D(1.5, 2.25)
    var result = external_call["c_func_eight_doubles_then_hfa2d", Float64](
        Float64(1.0),
        Float64(2.0),
        Float64(3.0),
        Float64(4.0),
        Float64(5.0),
        Float64(6.0),
        Float64(7.0),
        Float64(8.0),
        s,
    )
    print("eight_doubles_then_hfa2d:", result)


# CHECK: eight_doubles_then_hfa2d: 2604.0


# ============================================================================
# Trailing scalar after an HFA
# ============================================================================
@fieldwise_init
struct MultiHfa4F(TrivialRegisterPassable):
    var a: Float32
    var b: Float32
    var c: Float32
    var d: Float32


def test_hfa4f_then_scalar():
    """The scalar must take the next free SIMD register instead of
    overwriting a member of the aggregate."""
    var s = MultiHfa4F(1.5, 2.25, 3.0, 4.5)
    var result = external_call["c_func_hfa4f_then_scalar", MultiHfa4F](
        s, Float32(2.0)
    )
    print("hfa4f_then_scalar:", result.a, result.b, result.c, result.d)


# CHECK: hfa4f_then_scalar: 3.0 4.5 6.0 9.0


# ============================================================================
# 24-byte homogeneous double aggregate: AAPCS D0-D2, SysV MEMORY
# ============================================================================
@fieldwise_init
struct MultiHfa3D(TrivialRegisterPassable):
    var a: Float64
    var b: Float64
    var c: Float64


def test_hfa3d_argument():
    var s = MultiHfa3D(1.5, 2.25, 3.5)
    var result = external_call["c_func_hfa3d_sum", Float64](s)
    print("hfa3d_argument:", result)


# CHECK: hfa3d_argument: 3502251.5


def test_hfa3d_two_scalars_return():
    """Two scalar arguments and an aggregate return in one signature."""
    var result = external_call["c_func_hfa3d_make", MultiHfa3D](
        Float64(10.0), Float64(0.5)
    )
    print("hfa3d_two_scalars_return:", result.a, result.b, result.c)


# CHECK: hfa3d_two_scalars_return: 10.5 11.0 11.5


# ============================================================================
# Main - run all tests
# ============================================================================
def main():
    test_scalar_then_memory_struct()
    test_scalar_then_padded_small()
    test_eight_doubles_then_hfa2d()
    test_hfa4f_then_scalar()
    test_hfa3d_argument()
    test_hfa3d_two_scalars_return()
