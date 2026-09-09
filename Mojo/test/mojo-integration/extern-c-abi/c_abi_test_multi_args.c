// C reference for cross-argument register allocation.
// Mojo test: test_struct_arguments_multi.mojo
//
// Every other C reference in this directory takes exactly one struct
// argument, which exercises classification but never allocation: which
// register or stack slot a struct lands in once earlier arguments have
// consumed part of the register file. The functions here pair aggregates with
// scalars, and with enough leading floats to exhaust the SIMD registers.
//
// Each argument carries a distinct weight so that a dropped, duplicated, or
// misplaced one produces a wrong answer rather than an accidental pass.

#include <stdint.h>

// ============================================================================
// Scalar preceding a MEMORY-class aggregate
// ============================================================================

// 36 bytes: MEMORY class on both SysV and AAPCS.
struct MultiInt36 {
  int32_t a, b, c, d, e, f, g, h, i;
};

int64_t c_func_scalar_then_memory_struct(int32_t k, struct MultiInt36 s) {
  return (int64_t)k * 1000000000 + (int64_t)s.a * 1 + (int64_t)s.b * 10 +
         (int64_t)s.c * 100 + (int64_t)s.d * 1000 + (int64_t)s.e * 10000 +
         (int64_t)s.f * 100000 + (int64_t)s.g * 1000000 +
         (int64_t)s.h * 10000000 + (int64_t)s.i * 100000000;
}

// ============================================================================
// Scalar preceding a sub-eightbyte aggregate with tail padding
// ============================================================================

// 4 bytes: two fields plus one byte of tail padding, which forwarding must
// not read.
struct MultiPadShort {
  int16_t a;
  int8_t b;
};

int64_t c_func_scalar_then_padded_small(int32_t k, struct MultiPadShort s) {
  return (int64_t)k * 10000 + (int64_t)s.a * 100 + (int64_t)s.b;
}

// ============================================================================
// Aggregates arriving with the SIMD registers partly or fully consumed
// ============================================================================

struct MultiHfa2D {
  double a, b;
};

// TODO(MOCO-4611): Add the seven-leading-double case, where one SIMD register
// is left — one fewer than the HFA needs — so AAPCS must burn it and pass the
// aggregate whole on the stack. Mojo splits the aggregate there instead, and
// the case cannot ship yet because it is red on aarch64 and green on x86-64,
// with no target-arch lit feature to hang a conditional XFAIL on.

// Eight leading doubles leave no SIMD register at all, so the aggregate
// reaches plain stack passing.
double c_func_eight_doubles_then_hfa2d(double a, double b, double c, double d,
                                       double e, double f, double g, double h,
                                       struct MultiHfa2D s) {
  return a + b * 2.0 + c * 3.0 + d * 4.0 + e * 5.0 + f * 6.0 + g * 7.0 +
         h * 8.0 + s.a * 100.0 + s.b * 1000.0;
}

// ============================================================================
// Trailing scalar after an HFA
// ============================================================================

struct MultiHfa4F {
  float a, b, c, d;
};

// The scalar must take the SIMD register after the four the HFA occupies,
// rather than overwriting one of its members.
struct MultiHfa4F c_func_hfa4f_then_scalar(struct MultiHfa4F s, float k) {
  s.a *= k;
  s.b *= k;
  s.c *= k;
  s.d *= k;
  return s;
}

// ============================================================================
// 24-byte homogeneous double aggregate, where the two ABIs disagree
// ============================================================================

// AAPCS keeps this in D0-D2, since an HFA may hold up to four members. SysV
// classifies anything over 16 bytes as MEMORY.
struct MultiHfa3D {
  double a, b, c;
};

double c_func_hfa3d_sum(struct MultiHfa3D s) {
  return s.a + s.b * 1000.0 + s.c * 1000000.0;
}

// Two scalar arguments and an aggregate return in one signature: the return
// slot must not disturb the argument registers.
struct MultiHfa3D c_func_hfa3d_make(double base, double step) {
  struct MultiHfa3D s;
  s.a = base + step;
  s.b = base + step * 2.0;
  s.c = base + step * 3.0;
  return s;
}
