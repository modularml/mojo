// C ABI reference for rollback-to-stack tests
// Mojo test: test_struct_arguments_rollback.mojo
// A small by-value struct passed after enough integer-class args to exhaust
// the argument registers must still arrive intact: x86-64 SysV passes it
// entirely on the stack rather than splitting it across a register and memory.

#include <stdint.h>

typedef struct int3_t {
  int32_t x;
  int32_t y;
  int32_t z;
} int3_t;

// Returns the count of fields that do not match the expected (11, 22, 33),
// so 0 means the struct was received intact.
static int check_values(int3_t v) {
  int status = 0;
  if (v.x != 11)
    status += 1;
  if (v.y != 22)
    status += 1;
  if (v.z != 33)
    status += 1;
  return status;
}

// Struct passed first: both eightbytes fit in registers.
int check_struct_early(int3_t value) { return check_values(value); }

// Two pointers + three ints consume five integer registers; the by-value
// int3_t needs two more and only one remains, so it rolls back to the stack.
int check_struct_after_five(void *p0, void *p1, int a, int b, int c,
                            int3_t value) {
  (void)p0;
  (void)p1;
  (void)a;
  (void)b;
  (void)c;
  return check_values(value);
}
