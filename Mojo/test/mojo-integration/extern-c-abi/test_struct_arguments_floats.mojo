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
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_floats
# RUN: %t.dir/test_floats | FileCheck %s

from std.ffi import external_call


# ============================================================================
# 4-byte float struct (single Float32)
# ============================================================================
@fieldwise_init
struct FloatStruct4(TrivialRegisterPassable):
    var f: Float32


def test_float_4byte():
    var s = FloatStruct4(10.5)
    var result = external_call["c_func_float_4byte", FloatStruct4](s)
    print("float_4byte:", result.f)


# CHECK: float_4byte: 11.5


# ============================================================================
# 8-byte float struct (two Float32)
# ============================================================================
@fieldwise_init
struct FloatStruct8(TrivialRegisterPassable):
    var f1: Float32
    var f2: Float32


def test_float_8byte():
    var s = FloatStruct8(10.5, 20.5)
    var result = external_call["c_func_float_8byte", FloatStruct8](s)
    print("float_8byte:", result.f1, result.f2)


# CHECK: float_8byte: 11.5 21.5


# ============================================================================
# 8-byte double struct (single Float64)
# ============================================================================
@fieldwise_init
struct DoubleStruct8(TrivialRegisterPassable):
    var d: Float64


def test_double_8byte():
    var s = DoubleStruct8(100.25)
    var result = external_call["c_func_double_8byte", DoubleStruct8](s)
    print("double_8byte:", result.d)


# CHECK: double_8byte: 101.25


# ============================================================================
# 12-byte float struct (three Float32)
# ============================================================================
@fieldwise_init
struct FloatStruct12(TrivialRegisterPassable):
    var f1: Float32
    var f2: Float32
    var f3: Float32


def test_float_12byte():
    var s = FloatStruct12(10.5, 20.5, 30.5)
    var result = external_call["c_func_float_12byte", FloatStruct12](s)
    print("float_12byte:", result.f1, result.f2, result.f3)


# CHECK: float_12byte: 11.5 21.5 31.5


# ============================================================================
# 16-byte float struct (four Float32)
# ============================================================================
@fieldwise_init
struct FloatStruct16(TrivialRegisterPassable):
    var f1: Float32
    var f2: Float32
    var f3: Float32
    var f4: Float32


def test_float_16byte():
    var s = FloatStruct16(10.5, 20.5, 30.5, 40.5)
    var result = external_call["c_func_float_16byte", FloatStruct16](s)
    print("float_16byte:", result.f1, result.f2, result.f3, result.f4)


# CHECK: float_16byte: 11.5 21.5 31.5 41.5


# ============================================================================
# 16-byte double struct (two Float64)
# ============================================================================
@fieldwise_init
struct DoubleStruct16(TrivialRegisterPassable):
    var d1: Float64
    var d2: Float64


def test_double_16byte():
    var s = DoubleStruct16(100.25, 200.25)
    var result = external_call["c_func_double_16byte", DoubleStruct16](s)
    print("double_16byte:", result.d1, result.d2)


# CHECK: double_16byte: 101.25 201.25


# ============================================================================
# 17-byte float struct (four Float32 + UInt8) - MEMORY class
# ============================================================================
@fieldwise_init
struct FloatStruct17(TrivialRegisterPassable):
    var f1: Float32
    var f2: Float32
    var f3: Float32
    var f4: Float32
    var extra: UInt8


def test_float_17byte():
    var s = FloatStruct17(10.5, 20.5, 30.5, 40.5, 100)
    var result = external_call["c_func_float_17byte", FloatStruct17](s)
    print(
        "float_17byte:",
        result.f1,
        result.f2,
        result.f3,
        result.f4,
        Int(result.extra),
    )


# CHECK: float_17byte: 11.5 21.5 31.5 41.5 101


# ============================================================================
# 33-byte float struct (eight Float32 + UInt8) - MEMORY class
# ============================================================================
@fieldwise_init
struct FloatStruct33(TrivialRegisterPassable):
    var f1: Float32
    var f2: Float32
    var f3: Float32
    var f4: Float32
    var f5: Float32
    var f6: Float32
    var f7: Float32
    var f8: Float32
    var extra: UInt8


def test_float_33byte():
    var s = FloatStruct33(1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 10)
    var result = external_call["c_func_float_33byte", FloatStruct33](s)
    print(
        "float_33byte:",
        result.f1,
        result.f2,
        result.f3,
        result.f4,
        result.f5,
        result.f6,
        result.f7,
        result.f8,
        Int(result.extra),
    )


# CHECK: float_33byte: 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5 11


# ============================================================================
# 12-byte heterogeneous float struct (Float32 + Float64) - Non-HFA
# ARM64 AAPCS: Mixed float types (f32 ≠ f64) violate HFA homogeneity
# requirement, so this is NOT an HFA and must use GPR (integer) coercion.
# C layout: {float f; /* 4 bytes padding */ double d;} = 16 bytes
# On ARM64: IntegerPair (two i64 in X0, X1)
# ============================================================================
@fieldwise_init
struct MixedFloat32Float64(TrivialRegisterPassable):
    var f: Float32
    var d: Float64


def test_mixed_f32_f64():
    var s = MixedFloat32Float64(10.5, 100.25)
    var result = external_call["c_func_mixed_f32_f64", MixedFloat32Float64](s)
    print("mixed_f32_f64:", result.f, result.d)


# CHECK: mixed_f32_f64: 11.5 101.25


# ============================================================================
# 12-byte heterogeneous float struct (Float64 + Float32) - Non-HFA
# ARM64 AAPCS: Like above, but reversed field order to test that
# heterogeneity is detected regardless of field ordering.
# C layout: {double d; float f; /* 4 bytes padding */} = 16 bytes
# On ARM64: IntegerPair (two i64 in X0, X1)
# ============================================================================
@fieldwise_init
struct MixedFloat64Float32(TrivialRegisterPassable):
    var d: Float64
    var f: Float32


def test_mixed_f64_f32():
    var s = MixedFloat64Float32(100.25, 10.5)
    var result = external_call["c_func_mixed_f64_f32", MixedFloat64Float32](s)
    print("mixed_f64_f32:", result.d, result.f)


# CHECK: mixed_f64_f32: 101.25 11.5


# ============================================================================
# 20-byte homogeneous float struct (5x Float32) - Non-HFA (exceeds field limit)
# ARM64 AAPCS: HFAs are limited to at most 4 fields. With 5 fields, this
# violates the field count limit and must be treated as a regular aggregate.
# On ARM64: >16 bytes → passed by pointer (MEMORY class)
# ============================================================================
@fieldwise_init
struct FiveFloats(TrivialRegisterPassable):
    var f1: Float32
    var f2: Float32
    var f3: Float32
    var f4: Float32
    var f5: Float32


def test_five_floats():
    var s = FiveFloats(1.5, 2.5, 3.5, 4.5, 5.5)
    var result = external_call["c_func_five_floats", FiveFloats](s)
    print("five_floats:", result.f1, result.f2, result.f3, result.f4, result.f5)


# CHECK: five_floats: 2.5 3.5 4.5 5.5 6.5


# ============================================================================
# Main - run all tests
# ============================================================================
def main():
    # Homogeneous float structs (valid HFAs or HFA-like)
    test_float_4byte()
    test_float_8byte()
    test_double_8byte()
    test_float_12byte()
    test_float_16byte()
    test_double_16byte()

    # Large structs (>16 bytes, passed by pointer)
    test_float_17byte()
    test_float_33byte()

    # Edge cases: heterogeneous and field-count violations
    test_mixed_f32_f64()
    test_mixed_f64_f32()
    test_five_floats()
