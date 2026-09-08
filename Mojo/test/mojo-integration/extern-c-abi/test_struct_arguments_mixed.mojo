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
# C reference: c_abi_test_float_structs.c
# Consolidated test for mixed int/float struct arguments
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_mixed
# RUN: %t.dir/test_mixed | FileCheck %s

from std.ffi import external_call


# ============================================================================
# 8-byte mixed (Int32 + Float32)
# On ARM64 AAPCS: non-HFA, coerced to i64 (single GPR)
# On x86-64 SysV: single eightbyte, INTEGER wins over SSE
# ============================================================================
@fieldwise_init
struct MixedIntFloat8(TrivialRegisterPassable):
    var i: Int32
    var f: Float32


def test_mixed_if_8byte():
    var s = MixedIntFloat8(10, 10.5)
    var result = external_call["c_func_mixed_if_8byte", MixedIntFloat8](s)
    print("mixed_if_8byte:", result.i, result.f)


# CHECK: mixed_if_8byte: 11 11.5


# ============================================================================
# 12-byte mixed (Int32 + Float64) - with padding
# C layout: {int32_t i; /* 4 bytes pad */ double d;} = 16 bytes
# On ARM64: IntegerPair (two GPRs)
# ============================================================================
@fieldwise_init
struct MixedIntDouble12(TrivialRegisterPassable):
    var i: Int32
    var d: Float64


def test_mixed_id_12byte():
    var s = MixedIntDouble12(10, 100.5)
    var result = external_call["c_func_mixed_id_12byte", MixedIntDouble12](s)
    print("mixed_id_12byte:", result.i, result.d)


# CHECK: mixed_id_12byte: 11 101.5


# ============================================================================
# 16-byte mixed (Int64 + Float64)
# On ARM64: IntegerPair (two GPRs)
# ============================================================================
@fieldwise_init
struct MixedIntDouble16(TrivialRegisterPassable):
    var i: Int64
    var d: Float64


def test_mixed_id_16byte():
    var s = MixedIntDouble16(10, 100.5)
    var result = external_call["c_func_mixed_id_16byte", MixedIntDouble16](s)
    print("mixed_id_16byte:", result.i, result.d)


# CHECK: mixed_id_16byte: 11 101.5


# ============================================================================
# 16-byte mixed (Float64 + Int64)
# On ARM64: IntegerPair (two GPRs)
# ============================================================================
@fieldwise_init
struct MixedDoubleInt16(TrivialRegisterPassable):
    var d: Float64
    var i: Int64


def test_mixed_di_16byte():
    var s = MixedDoubleInt16(100.5, 10)
    var result = external_call["c_func_mixed_di_16byte", MixedDoubleInt16](s)
    print("mixed_di_16byte:", result.d, result.i)


# CHECK: mixed_di_16byte: 101.5 11


# ============================================================================
# 24-byte complex mixed (Int32 + Float32 + Float64 + Int32) - MEMORY class
# C struct: {int32_t i1; float f; double d; int32_t i2;} = 24 bytes
# On ARM64: >16 bytes, passed by pointer
# ============================================================================
@fieldwise_init
struct MixedComplex24(TrivialRegisterPassable):
    var i1: Int32
    var f: Float32
    var d: Float64
    var i2: Int32


def test_mixed_complex_24byte():
    var s = MixedComplex24(10, 10.5, 20.5, 30)
    var result = external_call["c_func_mixed_complex_24byte", MixedComplex24](s)
    print("mixed_complex_24byte:", result.i1, result.f, result.d, result.i2)


# CHECK: mixed_complex_24byte: 11 11.5 21.5 31


# ============================================================================
# Main - run all tests
# ============================================================================
def main():
    test_mixed_if_8byte()
    test_mixed_id_12byte()
    test_mixed_id_16byte()
    test_mixed_di_16byte()
    test_mixed_complex_24byte()
