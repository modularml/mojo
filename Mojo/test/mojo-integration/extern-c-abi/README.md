# C ABI Integration Tests

End-to-end tests for C ABI struct handling in Mojo.

## Purpose

Tests Mojo implementation of System V AMD64 and ARM64 AAPCS calling
conventions when calling C functions that:

- Take structs by value as arguments
- Return structs by value
- Use variadic parameters (`...`)

## Test Coverage

### Integer Structs (`test_struct_arguments_integers.mojo`)

Tests struct coercion for integer-only fields across different sizes:

- Small structs (1-8 bytes) → single register (10 tests, all passing)
- Medium structs (9-16 bytes) → two registers
- Large structs (>16 bytes) → pointer passing (MEMORY class)

### Float Structs (`test_struct_arguments_floats.mojo`)

Tests SSE register classification for pure float/double structs (8 tests,
all passing):

- ≤8 bytes → SSE registers
- 9-16 bytes → SSE pair or mixed classification
- >16 bytes → MEMORY class

### Mixed Int/Float Structs (`test_struct_arguments_mixed.mojo`)

Tests multi-eightbyte classification when struct contains both integer and
floating point fields (5 tests, all passing).

### Pointer Structs (`test_struct_arguments_pointers.mojo`)

Tests struct coercion for structs containing pointer fields (3 tests, all
passing).

### Multiple Arguments (`test_struct_arguments_multi.mojo`)

Tests register allocation *across* arguments rather than classification of a
single one — every other file here passes exactly one struct (6 tests):

- A scalar preceding a MEMORY-class struct, and preceding a sub-eightbyte
  struct with tail padding
- An HFA arriving after 8 leading doubles, with no SIMD register left, so it
  reaches plain stack passing. The 7-double case, where one register is left
  and AAPCS must burn it rather than split the aggregate, is MOCO-4611
- A trailing scalar after an HFA, which must take the next free SIMD register
- 3x `double` (24 bytes), where AAPCS keeps the HFA in D0-D2 and SysV calls
  it MEMORY, in both argument and return position

### Variadic Functions

Tests C variadic functions (`printf`-style) which have different ABI rules
than regular functions:

- `test_variadic_floats_basic.mojo` - Basic float/double variadic (3 tests)
- `test_variadic_floats_many.mojo` - Many float arguments (1 test)
- `test_variadic_float_structs.mojo` - Float struct variadic (6 tests)
- `test_variadic_mixed_structs.mojo` - Mixed int/float structs (2 tests)
- `test_variadic_mixed_int_struct_and_float.mojo` - Regression test for
  flattened float struct when another arg forces ABI coercion (1 test)
- `test_variadic_prototype.mojo` - Integer variadic functions (5 tests)

## Test Design

**C reference functions:**

- Take struct by value, modify fields (add 1), return modified struct

**Mojo tests:**

- Create struct with known values
- Call C function via `external_call`
- Print results
- FileCheck validates expected output

**Pattern:**

```mojo
var s = Struct(10, 20, 30)
var result = external_call["c_func", Struct](s)
print(result.a, result.b, result.c)
# CHECK: 11 21 31
```

This validates both argument passing AND return value ABI.

## File Organization

**C Reference Implementations:**

- `c_abi_test_int_structs.c` - Integer struct functions (10 functions)
- `c_abi_test_float_structs.c` - Float and mixed int/float functions (13
  functions)
- `c_abi_test_ptr_structs.c` - Pointer struct functions (2 functions)
- `c_abi_test_multi_args.c` - Cross-argument allocation (6 functions)
- `c_abi_variadic_prototype.c` - Integer variadic functions (5 functions)
- `c_abi_variadic_floats.c` - Float variadic functions (13 functions)

**Mojo Tests (12 files):**

Struct argument tests (consolidated):

- `test_struct_arguments_integers.mojo` - Integer-only structs (10 tests)
- `test_struct_arguments_floats.mojo` - Float/double structs (8 tests)
- `test_struct_arguments_mixed.mojo` - Mixed int/float structs (5 tests)
- `test_struct_arguments_pointers.mojo` - Pointer structs (3 tests)
- `test_struct_arguments_multi.mojo` - Cross-argument allocation (6 tests)

Variadic function tests:

- `test_variadic_floats_basic.mojo` - Basic float/double variadic (3 tests)
- `test_variadic_floats_many.mojo` - Many float arguments (1 test)
- `test_variadic_float_structs.mojo` - Float struct variadic (6 tests)
- `test_variadic_mixed_structs.mojo` - Mixed int/float struct variadic (2 tests)
- `test_variadic_mixed_int_struct_and_float.mojo` - Flattened float
  regression (1 test)
- `test_variadic_prototype.mojo` - Integer variadic (5 tests)

Error tests (compile-failure):

- `test_conflicting_signatures_error.mojo` - Two `external_call` sites to the
  same C function with incompatible argument types must produce a clear error
  pointing to both call sites (expected compile failure)

Each test file includes a comment indicating which C file(s) it uses.

## Running Tests

```bash
# Run all ABI tests (12 test files, all passing)
./bazelw test //Mojo/test/mojo-integration/extern-c-abi:test

# Run specific subset
./bazelw test //Mojo/test/mojo-integration/extern-c-abi:test \
  --test_filter="*integers*"
./bazelw test //Mojo/test/mojo-integration/extern-c-abi:test \
  --test_filter="*float*"
./bazelw test //Mojo/test/mojo-integration/extern-c-abi:test \
  --test_filter="*variadic*"
```

**Test Status Summary:**

- **Total test files:** 12 (all passing via Bazel)
- **All tests passing** on both ARM64 and x86-64

## How to Reproduce / Run Standalone

To run tests manually without Bazel (useful for debugging or quick iteration):

### Step 1: Build the C Library

Compile all C source files into a static library:

```bash
# Navigate to this directory
cd KGEN/test/mojo-integration/extern-c-abi/

# Compile all C files to object files
clang -c -O0 -g c_abi_test_int_structs.c \
                 c_abi_test_float_structs.c \
                 c_abi_test_ptr_structs.c \
                 c_abi_test_multi_args.c \
                 c_abi_variadic_prototype.c \
                 c_abi_variadic_floats.c

# Create static library archive
ar rcs libc_abi_reference.a *.o

# Clean up intermediate files
rm *.o
```

**Note:** This only needs to be done once, or when C files are modified.

### Step 2: Build and Run a Test

```bash
# Build a Mojo test (example: integer struct tests)
mojo build -Xlinker libc_abi_reference.a \
    test_struct_arguments_integers.mojo \
    -o test_integers

# Run the test
./test_integers
```

### Step 3: Validate with FileCheck (Optional)

```bash
# Verify output matches CHECK annotations
./test_integers | FileCheck test_struct_arguments_integers.mojo

# FileCheck succeeds silently if output matches
```
