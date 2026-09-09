// C ABI variadic function tests for FLOAT arguments
// Mojo tests: test_variadic_floats_*.mojo, test_variadic_float_*.mojo,
// test_variadic_mixed_*.mojo Tests variadic float/double handling and struct
// classification in variadic context

#include <stdarg.h>
#include <stdint.h>

// ============================================================================
// Variadic Float/Double Scalars
// ============================================================================

// Variadic floats: sum multiple float arguments
// In variadic context, float is promoted to double
// Variadic inputs: float values (promoted to double, count determined by first
// argument)
double c_func_variadic_floats(int count, ...) {
  va_list args;
  va_start(args, count);

  double sum = 0.0;
  for (int i = 0; i < count; i++) {
    // Note: float is promoted to double in variadic args
    double val = va_arg(args, double);
    sum += val + 1.0;
  }

  va_end(args);
  return sum;
}

// Variadic doubles: explicit double arguments
// Variadic inputs: double values (count determined by first argument)
double c_func_variadic_doubles(int count, ...) {
  va_list args;
  va_start(args, count);

  double sum = 0.0;
  for (int i = 0; i < count; i++) {
    double val = va_arg(args, double);
    sum += val + 1.0;
  }

  va_end(args);
  return sum;
}

// Variadic mixed int and float
// Variadic inputs: int64_t, then double
double c_func_variadic_int_float(int marker, ...) {
  va_list args;
  va_start(args, marker);

  int64_t i = va_arg(args, int64_t);
  double f = va_arg(args, double); // float promoted to double

  va_end(args);
  return (double)(i + 1) + (f + 1.0);
}

// ============================================================================
// Variadic Pure Float Structs
// ============================================================================

// 4-byte float struct in variadic context
struct FloatStruct4 {
  float a;
};

// Variadic inputs: single FloatStruct4 (4 bytes)
struct FloatStruct4 c_func_variadic_float_4byte(int marker, ...) {
  va_list args;
  va_start(args, marker);

  struct FloatStruct4 s = va_arg(args, struct FloatStruct4);
  s.a += 1.0f;

  va_end(args);
  return s;
}

// 8-byte: two floats in variadic context
struct FloatStruct8 {
  float a;
  float b;
};

// Variadic inputs: single FloatStruct8 (8 bytes, two floats)
struct FloatStruct8 c_func_variadic_float_8byte(int marker, ...) {
  va_list args;
  va_start(args, marker);

  struct FloatStruct8 s = va_arg(args, struct FloatStruct8);
  s.a += 1.0f;
  s.b += 1.0f;

  va_end(args);
  return s;
}

// 8-byte: single double in variadic context
struct DoubleStruct8 {
  double a;
};

// Variadic inputs: single DoubleStruct8 (8 bytes, one double)
struct DoubleStruct8 c_func_variadic_double_8byte(int marker, ...) {
  va_list args;
  va_start(args, marker);

  struct DoubleStruct8 s = va_arg(args, struct DoubleStruct8);
  s.a += 1.0;

  va_end(args);
  return s;
}

// 16-byte: four floats in variadic context
struct FloatStruct16 {
  float a;
  float b;
  float c;
  float d;
};

// Variadic inputs: single FloatStruct16 (16 bytes, four floats)
struct FloatStruct16 c_func_variadic_float_16byte(int marker, ...) {
  va_list args;
  va_start(args, marker);

  struct FloatStruct16 s = va_arg(args, struct FloatStruct16);
  s.a += 1.0f;
  s.b += 1.0f;
  s.c += 1.0f;
  s.d += 1.0f;

  va_end(args);
  return s;
}

// 16-byte: two doubles in variadic context
struct DoubleStruct16 {
  double a;
  double b;
};

// Variadic inputs: single DoubleStruct16 (16 bytes, two doubles)
struct DoubleStruct16 c_func_variadic_double_16byte(int marker, ...) {
  va_list args;
  va_start(args, marker);

  struct DoubleStruct16 s = va_arg(args, struct DoubleStruct16);
  s.a += 1.0;
  s.b += 1.0;

  va_end(args);
  return s;
}

// 17-byte: large float struct (must be passed by pointer in variadic)
struct FloatStruct17 {
  float a;
  float b;
  float c;
  float d;
  uint8_t e;
};

// Variadic inputs: single FloatStruct17 (17 bytes, passed by pointer)
struct FloatStruct17 c_func_variadic_float_17byte(int marker, ...) {
  va_list args;
  va_start(args, marker);

  struct FloatStruct17 s = va_arg(args, struct FloatStruct17);
  s.a += 1.0f;
  s.b += 1.0f;
  s.c += 1.0f;
  s.d += 1.0f;
  s.e += 1;

  va_end(args);
  return s;
}

// ============================================================================
// Variadic Mixed Int/Float Structs
// ============================================================================

// 8-byte: int32 + float in variadic context
struct MixedIntFloat8 {
  int32_t i;
  float f;
};

// Variadic inputs: single MixedIntFloat8 (8 bytes, int32 + float)
struct MixedIntFloat8 c_func_variadic_mixed_if_8byte(int marker, ...) {
  va_list args;
  va_start(args, marker);

  struct MixedIntFloat8 s = va_arg(args, struct MixedIntFloat8);
  s.i += 1;
  s.f += 1.0f;

  va_end(args);
  return s;
}

// 16-byte: double + int64 in variadic context
struct MixedDoubleInt16 {
  double d;
  int64_t i;
};

// Variadic inputs: single MixedDoubleInt16 (16 bytes, double + int64)
struct MixedDoubleInt16 c_func_variadic_mixed_di_16byte(int marker, ...) {
  va_list args;
  va_start(args, marker);

  struct MixedDoubleInt16 s = va_arg(args, struct MixedDoubleInt16);
  s.d += 1.0;
  s.i += 1;

  va_end(args);
  return s;
}

// Mixed variadic: int struct + float struct (where the float struct has a
// single field and gets flattened to a bare scalar at the POP level).
// Tests that a flattened float arg gets correct ABI treatment even when
// another struct arg forces the ABI coercion path.
struct IntStruct8_mixed {
  int32_t a;
  int32_t b;
};

// Variadic inputs: IntStruct8 (8 bytes) + FloatStruct4 (4 bytes)
// Returns: sum as FloatStruct4
struct FloatStruct4 c_func_variadic_mixed_int_struct_and_float(int marker,
                                                               ...) {
  va_list args;
  va_start(args, marker);

  struct IntStruct8_mixed s = va_arg(args, struct IntStruct8_mixed);
  struct FloatStruct4 f = va_arg(args, struct FloatStruct4);
  struct FloatStruct4 result;
  result.a = (float)(s.a + s.b) + f.a + 1.0f;

  va_end(args);
  return result;
}

// Many float arguments (stress test for xmm register exhaustion)
// Variadic inputs: up to 12 double values (tests SSE register exhaustion)
double c_func_variadic_many_floats(int count, ...) {
  va_list args;
  va_start(args, count);

  double sum = 0.0;
  // x86_64: First 8 floats go in xmm0-xmm7, rest on stack
  for (int i = 0; i < count && i < 12; i++) {
    double val = va_arg(args, double);
    sum += val;
  }

  va_end(args);
  return sum + count; // Add count as verification
}
