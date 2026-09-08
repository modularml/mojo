// Prototype C variadic functions for testing ABI
// Mojo test: test_variadic_prototype.mojo

#include <stdarg.h>
#include <stdint.h>

// Test 1: Simple variadic integers
// Takes a count, then that many int64_t values, returns their sum + count
// Variadic inputs: int64_t values (count determined by first argument)
int64_t c_func_variadic_sum_ints(int count, ...) {
  va_list args;
  va_start(args, count);

  int64_t sum = 0;
  for (int i = 0; i < count; i++) {
    int64_t val = va_arg(args, int64_t);
    sum += val + 1; // Add 1 to each value
  }

  va_end(args);
  return sum;
}

// Test 2: Variadic with 4-byte struct
struct Struct4 {
  uint8_t a, b, c, d;
};

// Variadic inputs: single Struct4 (4 bytes)
struct Struct4 c_func_variadic_struct4(int marker, ...) {
  va_list args;
  va_start(args, marker);

  // CRITICAL: In variadic context, small structs may be promoted
  struct Struct4 s = va_arg(args, struct Struct4);
  s.a += 1;
  s.b += 1;
  s.c += 1;
  s.d += 1;

  va_end(args);
  return s;
}

// Test 3: Variadic with 16-byte struct
struct Struct16 {
  uint64_t a;
  uint64_t b;
};

// Variadic inputs: single Struct16 (16 bytes)
struct Struct16 c_func_variadic_struct16(int marker, ...) {
  va_list args;
  va_start(args, marker);

  struct Struct16 s = va_arg(args, struct Struct16);
  s.a += 1;
  s.b += 1;

  va_end(args);
  return s;
}

// Test 4: Variadic with 17-byte struct (must be passed by pointer)
struct Struct17 {
  uint64_t a;
  uint64_t b;
  uint8_t c;
};

// Variadic inputs: single Struct17 (17 bytes, passed by pointer)
struct Struct17 c_func_variadic_struct17(int marker, ...) {
  va_list args;
  va_start(args, marker);

  // Large structs in variadic context are passed by pointer
  struct Struct17 s = va_arg(args, struct Struct17);
  s.a += 1;
  s.b += 1;
  s.c += 1;

  va_end(args);
  return s;
}

// Test 5: Variadic with mixed float and int
// Variadic inputs: int64_t, then double
double c_func_variadic_mixed(int count, ...) {
  va_list args;
  va_start(args, count);

  // First arg is int, second is double
  if (count >= 2) {
    int64_t i = va_arg(args, int64_t);
    double d = va_arg(args, double);
    va_end(args);
    return (double)(i + 1) + (d + 1.0);
  }

  va_end(args);
  return 0.0;
}
