// C ABI integer struct tests
// Mojo tests: test_struct_argument_{1,2,4,8,9,12,16,17,int_31,33}byte.mojo
// Tests integer-only struct classification and register passing

#include <stdint.h>

// ============================================================================
// Small Structs (1-4 bytes) - Should be coerced to integer type
// ============================================================================

// 1-byte struct
struct Struct1 {
  uint8_t a;
};

struct Struct1 c_func_1byte(struct Struct1 s) {
  s.a += 1;
  return s;
}

// 2-byte struct
struct Struct2 {
  uint8_t a;
  uint8_t b;
};

struct Struct2 c_func_2byte(struct Struct2 s) {
  s.a += 1;
  s.b += 1;
  return s;
}

// 4-byte struct (MOCO-3220 bug case)
struct Struct4 {
  uint8_t a;
  uint8_t b;
  uint8_t c;
  uint8_t d;
};

struct Struct4 c_func_4byte(struct Struct4 s) {
  s.a += 1;
  s.b += 1;
  s.c += 1;
  s.d += 1;
  return s;
}

// ============================================================================
// Medium Structs (8-16 bytes) - Register passing (two registers max)
// ============================================================================

// 8-byte struct (single eightbyte)
struct Struct8 {
  uint64_t value;
};

struct Struct8 c_func_8byte(struct Struct8 s) {
  s.value += 1;
  return s;
}

// 9-byte struct (two eightbytes)
struct Struct9 {
  uint64_t a;
  uint8_t b;
};

struct Struct9 c_func_9byte(struct Struct9 s) {
  s.a += 1;
  s.b += 1;
  return s;
}

// 12-byte struct (two eightbytes)
struct Struct12 {
  uint32_t a;
  uint32_t b;
  uint32_t c;
};

struct Struct12 c_func_12byte(struct Struct12 s) {
  s.a += 1;
  s.b += 1;
  s.c += 1;
  return s;
}

// 16-byte struct (two eightbytes)
struct Struct16 {
  uint64_t a;
  uint64_t b;
};

struct Struct16 c_func_16byte(struct Struct16 s) {
  s.a += 1;
  s.b += 1;
  return s;
}

// ============================================================================
// Large Structs (>16 bytes) - Memory class (passed by pointer)
// ============================================================================

// 17-byte struct (memory class)
struct Struct17 {
  uint64_t a;
  uint64_t b;
  uint8_t c;
};

struct Struct17 c_func_17byte(struct Struct17 s) {
  s.a += 1;
  s.b += 1;
  s.c += 1;
  return s;
}

// 31-byte struct (memory class)
struct Struct31 {
  uint64_t a;
  uint64_t b;
  uint64_t c;
  uint32_t d;
  uint16_t e;
  uint8_t f;
};

struct Struct31 c_func_31byte(struct Struct31 s) {
  s.a += 1;
  s.b += 1;
  s.c += 1;
  s.d += 1;
  s.e += 1;
  s.f += 1;
  return s;
}

// 33-byte struct (memory class)
struct Struct33 {
  uint64_t a;
  uint64_t b;
  uint64_t c;
  uint64_t d;
  uint8_t e;
};

struct Struct33 c_func_33byte(struct Struct33 s) {
  s.a += 1;
  s.b += 1;
  s.c += 1;
  s.d += 1;
  s.e += 1;
  return s;
}
