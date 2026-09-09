// C reference implementations for extern_c effect integration tests.

#include <stdint.h>

// ============================================================================
// FloatPair: {float, float} (8 bytes).
// x86-64 SysV: SSE class, returned/passed packed in XMM0 as f64.
// ARM64 AAPCS: HFA (2 floats), passed in V0/V1, returned in V0.
// ============================================================================

struct FloatPair {
  float x;
  float y;
};

typedef struct FloatPair (*FloatPairFn)(struct FloatPair);

struct FloatPair c_add_one(struct FloatPair p) {
  p.x += 1.0f;
  p.y += 1.0f;
  return p;
}

struct FloatPair c_add_ten(struct FloatPair p) {
  p.x += 10.0f;
  p.y += 10.0f;
  return p;
}

// Return function pointers so Mojo can get them via external_call and then
// exercise the call_indirect path (rather than a direct named-symbol call).
FloatPairFn c_get_add_one(void) { return c_add_one; }
FloatPairFn c_get_add_ten(void) { return c_add_ten; }

// ============================================================================
// Mojo-defined abi("C") function test: C calls a Mojo function.
// C accepts a FloatPairFn callback and invokes it, verifying that a
// Mojo-defined abi("C") function has a correct C ABI at its definition site.
// ============================================================================

struct FloatPair c_apply_float_pair_fn(FloatPairFn fp, struct FloatPair p) {
  return fp(p);
}

// ============================================================================
// BigStruct: {int64_t, int64_t, int64_t} (24 bytes).
// x86-64 SysV: size > 16 -> Memory class: sret for return, indirect for arg.
// ARM64 AAPCS: size > 16 -> sret for return, indirect for arg.
// ============================================================================

struct BigStruct {
  int64_t a;
  int64_t b;
  int64_t c;
};

typedef struct BigStruct (*BigStructFn)(struct BigStruct);

struct BigStruct c_big_struct_add_one(struct BigStruct p) {
  p.a++;
  p.b++;
  p.c++;
  return p;
}

BigStructFn c_get_big_struct_add_one(void) { return c_big_struct_add_one; }

// C calls a Mojo-defined abi("C") callback that takes a >16 B struct by-value.
// Exercises the callee side of the C ABI byval / sret lowering — the path
// missed by MOCO-3939 prior to its fix.
struct BigStruct c_apply_big_struct_fn(BigStructFn fp, struct BigStruct p) {
  return fp(p);
}
