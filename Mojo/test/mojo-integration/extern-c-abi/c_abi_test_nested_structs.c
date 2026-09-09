// C reference for nested struct ABI tests.
// Mojo test: test_struct_arguments_nested.mojo
//
// Tests that Outer { Inner {f32, f32}, f32 } (12 bytes) is passed and returned
// correctly via the System V x86-64 ABI: SSEPair(f64, f32) — eightbyte 0
// holds the two f32s of Inner, eightbyte 1 holds z.

#include <stdint.h>

struct Inner {
  float x;
  float y;
};

// 12-byte struct: Inner (8 bytes, eightbyte 0) + z (4 bytes, eightbyte 1).
struct Outer {
  struct Inner inner;
  float z;
};

struct Outer c_func_nested_float_12byte(struct Outer s) {
  s.inner.x += 1.0f;
  s.inner.y += 1.0f;
  s.z += 1.0f;
  return s;
}
