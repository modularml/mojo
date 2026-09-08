// C ABI reference for struct-with-pointer tests
// Mojo test: test_struct_arguments_pointers.mojo
// Tests struct passing where fields include pointers

#include <stdint.h>

// ============================================================================
// 16-byte struct: pointer + int32 (+ 4 bytes padding)
// ARM64 AAPCS: IntegerPair class (two registers X0, X1)
// x86-64 SysV: Two eightbytes, both INTEGER class
// ============================================================================
struct PtrInt32 {
  void *p;
  int32_t i;
};

struct PtrInt32 c_func_ptr_int32(struct PtrInt32 s) {
  s.p = (char *)s.p + 1;
  s.i += 1;
  return s;
}

// ============================================================================
// 24-byte struct: three pointers
// ARM64 AAPCS: MEMORY class (>16 bytes, passed by pointer)
// x86-64 SysV: MEMORY class (>16 bytes, passed by pointer)
// ============================================================================
struct ThreePtr {
  void *a;
  void *b;
  void *c;
};

struct ThreePtr c_func_three_ptr(struct ThreePtr s) {
  s.a = (char *)s.a + 1;
  s.b = (char *)s.b + 1;
  s.c = (char *)s.c + 1;
  return s;
}
