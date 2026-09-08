// C ABI reference implementation for FLOAT struct tests
// Mojo tests: test_struct_argument_float_*.mojo,
// test_struct_argument_double_*.mojo, test_struct_argument_mixed_*.mojo These
// test SSE register classification vs INTEGER classification

#include <stdint.h>

// ============================================================================
// Pure Float Structs - SSE Register Class
// ============================================================================

// 4-byte: single float (SSE class)
struct FloatStruct4 {
  float a;
};

struct FloatStruct4 c_func_float_4byte(struct FloatStruct4 s) {
  s.a += 1.0f;
  return s;
}

// 8-byte: two floats (SSE class)
struct FloatStruct8 {
  float a;
  float b;
};

struct FloatStruct8 c_func_float_8byte(struct FloatStruct8 s) {
  s.a += 1.0f;
  s.b += 1.0f;
  return s;
}

// 8-byte: single double (SSE class)
struct DoubleStruct8 {
  double a;
};

struct DoubleStruct8 c_func_double_8byte(struct DoubleStruct8 s) {
  s.a += 1.0;
  return s;
}

// 12-byte: three floats (SSE class)
struct FloatStruct12 {
  float a;
  float b;
  float c;
};

struct FloatStruct12 c_func_float_12byte(struct FloatStruct12 s) {
  s.a += 1.0f;
  s.b += 1.0f;
  s.c += 1.0f;
  return s;
}

// 16-byte: four floats (SSE class)
struct FloatStruct16 {
  float a;
  float b;
  float c;
  float d;
};

struct FloatStruct16 c_func_float_16byte(struct FloatStruct16 s) {
  s.a += 1.0f;
  s.b += 1.0f;
  s.c += 1.0f;
  s.d += 1.0f;
  return s;
}

// 16-byte: two doubles (SSE class)
struct DoubleStruct16 {
  double a;
  double b;
};

struct DoubleStruct16 c_func_double_16byte(struct DoubleStruct16 s) {
  s.a += 1.0;
  s.b += 1.0;
  return s;
}

// 17-byte: four floats + one byte (MEMORY class - too large)
struct FloatStruct17 {
  float a;
  float b;
  float c;
  float d;
  uint8_t e;
};

struct FloatStruct17 c_func_float_17byte(struct FloatStruct17 s) {
  s.a += 1.0f;
  s.b += 1.0f;
  s.c += 1.0f;
  s.d += 1.0f;
  s.e += 1;
  return s;
}

// 33-byte: large float struct (eight Float32 + UInt8, MEMORY class)
struct FloatStruct33 {
  float f1;
  float f2;
  float f3;
  float f4;
  float f5;
  float f6;
  float f7;
  float f8;
  uint8_t extra;
};

struct FloatStruct33 c_func_float_33byte(struct FloatStruct33 s) {
  s.f1 += 1.0f;
  s.f2 += 1.0f;
  s.f3 += 1.0f;
  s.f4 += 1.0f;
  s.f5 += 1.0f;
  s.f6 += 1.0f;
  s.f7 += 1.0f;
  s.f8 += 1.0f;
  s.extra += 1;
  return s;
}

// ============================================================================
// Mixed Int/Float Structs - Complex Classification
// ============================================================================

// 8-byte: int32 + float (first eightbyte has both INTEGER and SSE -> INTEGER
// wins)
struct MixedIntFloat8 {
  int32_t i;
  float f;
};

struct MixedIntFloat8 c_func_mixed_if_8byte(struct MixedIntFloat8 s) {
  s.i += 1;
  s.f += 1.0f;
  return s;
}

// 12-byte: int32 + double (split across eightbytes)
struct MixedIntDouble12 {
  int32_t i;
  double d;
};

struct MixedIntDouble12 c_func_mixed_id_12byte(struct MixedIntDouble12 s) {
  s.i += 1;
  s.d += 1.0;
  return s;
}

// 16-byte: double + int64 (SSE + INTEGER)
struct MixedDoubleInt16 {
  double d;
  int64_t i;
};

struct MixedDoubleInt16 c_func_mixed_di_16byte(struct MixedDoubleInt16 s) {
  s.d += 1.0;
  s.i += 1;
  return s;
}

// 16-byte: int64 + double (INTEGER + SSE, order matters)
struct MixedIntDouble16 {
  int64_t i;
  double d;
};

struct MixedIntDouble16 c_func_mixed_id_16byte(struct MixedIntDouble16 s) {
  s.i += 1;
  s.d += 1.0;
  return s;
}

// 24-byte: complex mixed (int32, float, double, int32)
struct MixedComplex24 {
  int32_t i1;
  float f;
  double d;
  int32_t i2;
};

struct MixedComplex24 c_func_mixed_complex_24byte(struct MixedComplex24 s) {
  s.i1 += 1;
  s.f += 1.0f;
  s.d += 1.0;
  s.i2 += 1;
  return s;
}

// ============================================================================
// Heterogeneous Float Structs - Non-HFA (violate homogeneity requirement)
// ============================================================================

// 12-byte: float + double (mixed float types, NOT an HFA on ARM64)
// ARM64 AAPCS: Heterogeneous → use GPR coercion (IntegerPair)
struct MixedFloat32Float64 {
  float f;
  double d;
};

struct MixedFloat32Float64 c_func_mixed_f32_f64(struct MixedFloat32Float64 s) {
  s.f += 1.0f;
  s.d += 1.0;
  return s;
}

// 12-byte: double + float (reversed field order, still heterogeneous)
// ARM64 AAPCS: Heterogeneous → use GPR coercion (IntegerPair)
struct MixedFloat64Float32 {
  double d;
  float f;
};

struct MixedFloat64Float32 c_func_mixed_f64_f32(struct MixedFloat64Float32 s) {
  s.d += 1.0;
  s.f += 1.0f;
  return s;
}

// ============================================================================
// Excessive Field Count - Non-HFA (violate ≤4 field limit)
// ============================================================================

// 20-byte: five floats (homogeneous but >4 fields, NOT an HFA on ARM64)
// ARM64 AAPCS: >16 bytes → passed by pointer (MEMORY class)
struct FiveFloats {
  float f1;
  float f2;
  float f3;
  float f4;
  float f5;
};

struct FiveFloats c_func_five_floats(struct FiveFloats s) {
  s.f1 += 1.0f;
  s.f2 += 1.0f;
  s.f3 += 1.0f;
  s.f4 += 1.0f;
  s.f5 += 1.0f;
  return s;
}
