// RUN: kgen-opt -split-input-file -lower-global-pop-to-llvm %s | FileCheck %s

// Unit tests for C ABI lowering on x86-64 (System V AMD64 ABI).
//
// PURPOSE: These are UNIT tests that lock the current behavior of the lowering
// transform. They exist to catch regressions during refactoring. They may need
// updating if we discover ABI bugs or add new features. The system tests are
// the authoritative check for correct end-to-end C ABI behavior:
//   KGEN/test/mojo-integration/extern-c-abi/
//
// TRANSFORM UNDER TEST:
//   Mojo/lib/KGENToLLVM/LowerPOPToLLVMExternalCalls.cpp — ConvertPOPExternalCall
//   Mojo/lib/KGENToLLVM/CABISystemV.cpp                 — SystemVABIInfo classifier
//   Mojo/lib/KGENToLLVM/CABILowering.cpp                — shared utilities
//
// RELATED TEST FILES:
//   extern-c-abi.mlir         — platform-independent tests (DefaultCABIInfo,
//                               empty triple): struct not field-exploded,
//                               multi-value return, identity pass-through
//   extern-c-abi-aarch64.mlir — ARM64 AAPCS tests
//
// INPUT FORMAT NOTE: Each test uses llvm.func (not kgen.func) as the enclosing
// function.
// The -lower-global-pop-to-llvm pass processes pop.external_call ops
// after kgen.func has already been lowered to llvm.func by an earlier pass.
// For coercion to work, createEntryBlockAlloca requires the enclosing
// function to be an LLVM::LLVMFuncOp. These tests exercise that final
// lowering step in isolation.
//
// TEST TABLE:
// Group A — Small struct args (1–8 bytes): Integer coercion
//   A1  arg_1byte_int_struct      {i8}        → i8
//   A2  arg_2byte_int_struct      {i8,i8}     → i16
//   A3  arg_3byte_int_struct      {i8,i8,i8}  → i32  (rounds up)
//   A4  arg_8byte_int_struct      {i32,i32}   → i64
//   A5  arg_padded_struct         {i32,i8}    → i64  (LLVM pads to 8 bytes)
//
// Group E — Small all-float struct args (1–8 bytes): SSE coercion
//   E1  arg_float_struct          {f32}       → f32
//   E2  arg_double_struct         {f64}       → f64
//   E3  arg_two_floats            {f32,f32}   → f64  (fits one SSE register)
//   E4  arg_int_wins_float        {i32,f32}   → i64  (INTEGER overrides SSE)
//
// Group B — 9–16 byte struct args: two-register classification
//   B1  arg_int_pair_12          {i32,i32,i32}        → (i64, i32)  IntegerPair
//   B2  arg_int_pair_16          {i64,i64}            → (i64, i64)  IntegerPair
//   B3  arg_sse_pair_12          {f32,f32,f32}        → (f64, f32)  SSEPair
//   B4  arg_sse_pair_16          {f64,f64}            → (f64, f64)  SSEPair
//   B5  arg_mixed_16             {i32,i32,f64}        → (i64, f64)  Mixed
//   B6  arg_dense_int_16         {i8,i8,i16,i32,i64}  → (i64, i64)  IntegerPair
//         (many integer fields per eightbyte, all classify as INTEGER)
//
// Group C — Large struct args (>16 bytes): indirect pass + byval
//   C1  arg_large_indirect        {i64,i64,i64}       → ptr with byval on func
//   C2  arg_padded_indirect       {i8,f32,i8,f64}     → ptr with byval on func
//         (alignment padding expands to 24 bytes → MEMORY class)
//
// Group D — Return value coercion
//   D1  ret_small_int_struct      returns {i32}       → call returns i32
//   D2  ret_int_pair              returns {i32,i32,i32} → call returns {i64,i32}
//   D3  ret_sse_pair              returns {f64,f64}   → call returns {f64,f64}
//   D4  ret_sret                  returns {i64,i64,i64} → sret hidden ptr
//
// Group G — Variadic functions
//   G1  variadic_scalars          variadic call, scalar args  → func has ...
//   G2  variadic_large_struct     variadic large struct arg   → byval on call
//
// Group H — Attribute remapping
//   H1  func_attrs_passthrough    funcAttrs = ["noinline"]    → passthrough on llvm.func
//   H2  arg_attrs_two_reg         argAttrs on two-reg struct  → attr on first register
//   H3  arg_attrs_sret            argAttrs with sret return   → attr shifted past sret ptr
//
// Group F — Vector type fields: SSE classification (regression guards)
//   F1  arg_vec2xf32_struct       {vector<2xf32>}       correct: f64 (SSE)
//   F2  arg_vec2xf32_and_f64      {vector<2xf32>,f64}   correct: SSEPair(f64,f64)
//
// Group N — Nested struct fields (LLVMStructType)
//   N1  arg_nested_float_4        {struct<(f32)>}          correct: f32  (SSE)
//   N2  arg_nested_float_8        {struct<(f32,f32)>}      correct: f64  (SSE)
//   N3  arg_nested_float_and_f64  {struct<(f32,f32)>,f64}  correct: SSEPair(f64,f64)
//   N4  arg_nested_int_8          {struct<(i32,i32)>}      correct: i64  (regression guard)
//   N5  arg_nested_int_and_i64    {struct<(i32,i32)>,i64}  correct: IntegerPair(i64,i64)
//
// MODULE HEADER NOTE: Each test uses the standard x86-64 Linux data layout
// string to drive struct size and alignment computation in getStructSize(),
// which in turn determines which ABI classification rule applies. The triple
// selects SystemVABIInfo.

// Reused module header (copy-pasted per test because -split-input-file
// requires a full module per section):
//   triple="x86_64-unknown-linux-gnu"
//   data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

//===----------------------------------------------------------------------===//
// Group A: Small integer struct arguments (1–8 bytes)
//
// Rule: non-HFA structs ≤8 bytes are coerced to the smallest integer type that
// holds their byte count. The struct is bitcast via a store/load pair through
// an entry-block alloca. The LLVM call and function declaration use the
// coerced integer type, not the struct type.
//===----------------------------------------------------------------------===//

// A1: 1-byte struct {i8} → passes in a single byte register (i8).
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_1byte_int_struct
llvm.func @arg_1byte_int_struct(%arg0: !llvm.struct<(i8)>) {
  // CHECK: llvm.call @c_a1(%{{.*}}) : (i8) -> ()
  pop.external_call @c_a1(%arg0) : (!llvm.struct<(i8)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_a1(i8)
}

// -----

// A2: 2-byte struct {i8, i8} → coerces to i16.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_2byte_int_struct
llvm.func @arg_2byte_int_struct(%arg0: !llvm.struct<(i8, i8)>) {
  // CHECK: llvm.call @c_a2(%{{.*}}) : (i16) -> ()
  pop.external_call @c_a2(%arg0) : (!llvm.struct<(i8, i8)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_a2(i16)
}

// -----

// A3: 3-byte struct {i8, i8, i8} → coerces to i32 (3 bytes rounds up to i32).
// The alloca is sized to i32 (4 bytes) to ensure the load reads valid memory.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_3byte_int_struct
llvm.func @arg_3byte_int_struct(%arg0: !llvm.struct<(i8, i8, i8)>) {
  // CHECK: llvm.call @c_a3(%{{.*}}) : (i32) -> ()
  pop.external_call @c_a3(%arg0) : (!llvm.struct<(i8, i8, i8)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_a3(i32)
}

// -----

// A4: 8-byte struct {i32, i32} → coerces to i64 (fits in one register).
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_8byte_int_struct
llvm.func @arg_8byte_int_struct(%arg0: !llvm.struct<(i32, i32)>) {
  // CHECK: llvm.call @c_a4(%{{.*}}) : (i64) -> ()
  pop.external_call @c_a4(%arg0) : (!llvm.struct<(i32, i32)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_a4(i64)
}

// -----

// A5: struct {i32, i8} — LLVM pads to 8 bytes (i32 align=4, i8 at offset 4,
// 3 bytes padding). getStructSize returns 8, so coercion yields i64.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_padded_struct
llvm.func @arg_padded_struct(%arg0: !llvm.struct<(i32, i8)>) {
  // CHECK: llvm.call @c_a5(%{{.*}}) : (i64) -> ()
  pop.external_call @c_a5(%arg0) : (!llvm.struct<(i32, i8)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_a5(i64)
}

//===----------------------------------------------------------------------===//
// Group E: Small all-float struct arguments (SSE classification)
//
// Rule: on x86-64, an all-float struct ≤8 bytes is classified SSE and coerced
// to f32 (≤4 bytes) or f64 (≤8 bytes). This keeps the value in an XMM
// register. If any field is an integer, the whole eightbyte is classified
// INTEGER (Integer overrides SSE) and the struct coerces to iN instead.
//===----------------------------------------------------------------------===//

// -----

// E1: {f32} (4 bytes, all-float) → SSE → f32.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_float_struct
llvm.func @arg_float_struct(%arg0: !llvm.struct<(f32)>) {
  // CHECK: llvm.call @c_e1(%{{.*}}) : (f32) -> ()
  pop.external_call @c_e1(%arg0) : (!llvm.struct<(f32)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_e1(f32)
}

// -----

// E2: {f64} (8 bytes, all-float) → SSE → f64.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_double_struct
llvm.func @arg_double_struct(%arg0: !llvm.struct<(f64)>) {
  // CHECK: llvm.call @c_e2(%{{.*}}) : (f64) -> ()
  pop.external_call @c_e2(%arg0) : (!llvm.struct<(f64)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_e2(f64)
}

// -----

// E3: {f32, f32} (8 bytes, all-float) → SSE → f64.
// Two floats in one eightbyte: the eightbyte is classified SSE, and because
// the eightbyte is 8 bytes, getEightbyteType returns f64 (not <2 x f32>).
// The store/load bitcast pattern reinterprets the two f32 bits as one f64.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_two_floats
llvm.func @arg_two_floats(%arg0: !llvm.struct<(f32, f32)>) {
  // CHECK: llvm.call @c_e3(%{{.*}}) : (f64) -> ()
  pop.external_call @c_e3(%arg0) : (!llvm.struct<(f32, f32)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_e3(f64)
}

// -----

// E4: {i32, f32} (8 bytes, mixed int + float) → INTEGER wins over SSE → i64.
// The System V ABI rule: if any field in an eightbyte is INTEGER class, the
// whole eightbyte is INTEGER. An i32 field makes this eightbyte INTEGER even
// though f32 would be SSE. Key test: must NOT coerce to f64.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_int_wins_float
llvm.func @arg_int_wins_float(%arg0: !llvm.struct<(i32, f32)>) {
  // CHECK: llvm.call @c_e4(%{{.*}}) : (i64) -> ()
  // CHECK-NOT: llvm.call @c_e4(%{{.*}}) : (f64) -> ()
  pop.external_call @c_e4(%arg0) : (!llvm.struct<(i32, f32)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_e4(i64)
}

//===----------------------------------------------------------------------===//
// Group B: Two-register struct arguments (9–16 bytes)
//
// Rule: structs 9–16 bytes are split into two 8-byte eightbytes. Each
// eightbyte is classified independently (INTEGER or SSE), then merged into
// one of: IntegerPair (i64, i64/i32), SSEPair (f64, f64/f32), or Mixed
// (i64, f64). The call passes two arguments; the function signature has two
// parameters.
//===----------------------------------------------------------------------===//

// -----

// B1: {i32, i32, i32} (12 bytes) → IntegerPair → (i64, i32).
// Eightbyte 0 (bytes 0-7): two i32s → INTEGER.
// Eightbyte 1 (bytes 8-11): one i32 → INTEGER.
// Both INTEGER → IntegerPair. Types: i64 for the 8-byte half, i32 for the 4-byte half.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_int_pair_12
llvm.func @arg_int_pair_12(%arg0: !llvm.struct<(i32, i32, i32)>) {
  // CHECK: llvm.call @c_b1(%{{.*}}, %{{.*}}) : (i64, i32) -> ()
  pop.external_call @c_b1(%arg0) : (!llvm.struct<(i32, i32, i32)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_b1(i64, i32)
}

// -----

// B2: {i64, i64} (16 bytes) → IntegerPair → (i64, i64).
// Each eightbyte holds exactly one i64 → INTEGER.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_int_pair_16
llvm.func @arg_int_pair_16(%arg0: !llvm.struct<(i64, i64)>) {
  // CHECK: llvm.call @c_b2(%{{.*}}, %{{.*}}) : (i64, i64) -> ()
  pop.external_call @c_b2(%arg0) : (!llvm.struct<(i64, i64)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_b2(i64, i64)
}

// -----

// B3: {f32, f32, f32} (12 bytes) → SSEPair → (f64, f32).
// Eightbyte 0 (bytes 0-7): two f32s → SSE; 8-byte region → f64.
// Eightbyte 1 (bytes 8-11): one f32 → SSE; 4-byte region → f32.
// Both SSE → SSEPair.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_sse_pair_12
llvm.func @arg_sse_pair_12(%arg0: !llvm.struct<(f32, f32, f32)>) {
  // CHECK: llvm.call @c_b3(%{{.*}}, %{{.*}}) : (f64, f32) -> ()
  pop.external_call @c_b3(%arg0) : (!llvm.struct<(f32, f32, f32)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_b3(f64, f32)
}

// -----

// B4: {f64, f64} (16 bytes) → SSEPair → (f64, f64).
// Each eightbyte holds exactly one f64 → SSE.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_sse_pair_16
llvm.func @arg_sse_pair_16(%arg0: !llvm.struct<(f64, f64)>) {
  // CHECK: llvm.call @c_b4(%{{.*}}, %{{.*}}) : (f64, f64) -> ()
  pop.external_call @c_b4(%arg0) : (!llvm.struct<(f64, f64)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_b4(f64, f64)
}

// -----

// B5: {i32, i32, f64} (16 bytes) → Mixed → (i64, f64).
// Eightbyte 0 (bytes 0-7): two i32s → INTEGER → i64.
// Eightbyte 1 (bytes 8-15): one f64 → SSE → f64.
// INTEGER + SSE → Mixed.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_mixed_16
llvm.func @arg_mixed_16(%arg0: !llvm.struct<(i32, i32, f64)>) {
  // CHECK: llvm.call @c_b5(%{{.*}}, %{{.*}}) : (i64, f64) -> ()
  pop.external_call @c_b5(%arg0) : (!llvm.struct<(i32, i32, f64)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_b5(i64, f64)
}

//===----------------------------------------------------------------------===//
// Group C: Large struct arguments (>16 bytes) — indirect passing
//
// Rule: structs >16 bytes are passed indirectly. The caller allocates stack
// space, stores the struct there, and passes a pointer. On x86-64 System V,
// the byval attribute is set on the function parameter to tell LLVM the callee
// receives a copy on the stack. ARM64 does NOT use byval (see aarch64 file).
//===----------------------------------------------------------------------===//

// -----

// C1: {i64, i64, i64} (24 bytes) → indirect via pointer, byval on func param.
// The function declaration takes !llvm.ptr with byval. The alloca+store is
// done by prepareCoercedArgument. addByvalAttrsToFunc sets the byval attr.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_large_indirect
llvm.func @arg_large_indirect(%arg0: !llvm.struct<(i64, i64, i64)>) {
  // CHECK: llvm.call @c_c1(%{{.*}}) : (!llvm.ptr {llvm.byval = !llvm.struct<(i64, i64, i64)>}) -> ()
  pop.external_call @c_c1(%arg0) : (!llvm.struct<(i64, i64, i64)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_c1(!llvm.ptr {llvm.byval = !llvm.struct<(i64, i64, i64)>})
}

//===----------------------------------------------------------------------===//
// Group D: Return value coercion
//
// Same classification rules as arguments, but applied to the return value.
// The LLVM call returns the coerced type; the transform reconstructs the
// original struct via a store/load bitcast pair (single-register) or via
// extractvalue + store + load (two-register). For >16 bytes, sret convention
// is used: a hidden first pointer parameter is allocated by the caller;
// the callee writes the result there; the caller loads it afterward.
//===----------------------------------------------------------------------===//

// -----

// D1: return {i32} (4 bytes) → call returns i32, bitcast back to struct.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @ret_small_int_struct
llvm.func @ret_small_int_struct() {
  // CHECK: [[R:%.*]] = llvm.call @c_d1() : () -> i32
  // CHECK: llvm.store [[R]], %{{.*}} : i32, !llvm.ptr
  // CHECK: llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(i32)>
  %r = pop.external_call @c_d1() : () -> !llvm.struct<(i32)>
  llvm.return
}
// CHECK: llvm.func @c_d1() -> i32
}

// -----

// D2: return {i32, i32, i32} (12 bytes) → IntegerPair → call returns
// !llvm.struct<(i64, i32)>. Two-register reconstruction: extract both values,
// store at the correct offsets into a typed alloca, load as original struct.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @ret_int_pair
llvm.func @ret_int_pair() {
  // CHECK: [[R:%.*]] = llvm.call @c_d2() : () -> !llvm.struct<(i64, i32)>
  // CHECK: llvm.extractvalue [[R]][0]
  // CHECK: llvm.extractvalue [[R]][1]
  // CHECK: llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(i32, i32, i32)>
  %r = pop.external_call @c_d2() : () -> !llvm.struct<(i32, i32, i32)>
  llvm.return
}
// CHECK: llvm.func @c_d2() -> !llvm.struct<(i64, i32)>
}

// -----

// D3: return {f64, f64} (16 bytes) → SSEPair → call returns
// !llvm.struct<(f64, f64)>. extractvalue + reconstruct.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @ret_sse_pair
llvm.func @ret_sse_pair() {
  // CHECK: [[R:%.*]] = llvm.call @c_d3() : () -> !llvm.struct<(f64, f64)>
  // CHECK: llvm.extractvalue [[R]][0]
  // CHECK: llvm.extractvalue [[R]][1]
  // CHECK: llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(f64, f64)>
  %r = pop.external_call @c_d3() : () -> !llvm.struct<(f64, f64)>
  llvm.return
}
// CHECK: llvm.func @c_d3() -> !llvm.struct<(f64, f64)>
}

// -----

// D4: return {i64, i64, i64} (24 bytes) → sret.
// The function gets a hidden first parameter (!llvm.ptr {llvm.sret = ...}).
// The LLVM call is void. The caller loads the result from the sret pointer.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @ret_sret
llvm.func @ret_sret() {
  // CHECK: llvm.call @c_d4(%{{.*}}) : (!llvm.ptr) -> ()
  // CHECK: llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(i64, i64, i64)>
  %r = pop.external_call @c_d4() : () -> !llvm.struct<(i64, i64, i64)>
  llvm.return
}
// CHECK: llvm.func @c_d4(!llvm.ptr {llvm.sret = !llvm.struct<(i64, i64, i64)>})
}

//===----------------------------------------------------------------------===//
// Group G: Variadic functions
//
// Variadic functions on x86-64: fixed args follow the same classification
// rules as non-variadic. The function type marks the function as variadic
// (isVarArg=true / '...' in the type). For large struct variadic args, byval
// is set on the *call* instruction (not the function declaration) because
// variadic args are not part of the function signature.
//===----------------------------------------------------------------------===//

// -----

// G1: variadic call with scalar args → function type includes '...'.
// The scalar args pass through as-is (identity classification). The key
// observable is that the LLVM function type is variadic.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @variadic_scalars
llvm.func @variadic_scalars(%fixed: i32, %extra: i32) {
  // CHECK: llvm.call @c_g1(%{{.*}}, %{{.*}}) : (i32, i32) -> ()
  pop.external_call @c_g1(%fixed, %extra) attributes {numFixedArgs = 1 : index}
    : (i32, i32) -> ()
  llvm.return
}
// CHECK: llvm.func @c_g1(i32, ...)
}

// -----

// G2: variadic call with a large struct variadic arg → byval on the call.
// The fixed arg (i32) is declared in the function signature normally. The
// variadic large struct arg (24 bytes, >16 → indirect) is passed as a pointer,
// but byval cannot appear on the function declaration (variadic args are not
// in the signature). Instead, addByvalAttrsToCall sets byval on the call
// instruction itself.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @variadic_large_struct
llvm.func @variadic_large_struct(%fixed: i32,
                                  %s: !llvm.struct<(i64, i64, i64)>) {
  // CHECK: llvm.call @c_g2
  // CHECK-SAME: llvm.byval = !llvm.struct<(i64, i64, i64)>
  pop.external_call @c_g2(%fixed, %s) attributes {numFixedArgs = 1 : index}
    : (i32, !llvm.struct<(i64, i64, i64)>) -> ()
  llvm.return
}
// The function declaration does NOT have byval — only the call does.
// CHECK: llvm.func @c_g2(i32, ...)
// CHECK-NOT: llvm.byval
}

//===----------------------------------------------------------------------===//
// Group B (continued): Dense integer struct (B6)
//===----------------------------------------------------------------------===//

// -----

// B6: {i8, i8, i16, i32, i64} (16 bytes) → IntegerPair → (i64, i64).
// Eightbyte 0 (bytes 0-7): i8 + i8 + i16 + i32 — four integer fields, all
// INTEGER class. Size=8 → i64.
// Eightbyte 1 (bytes 8-15): i64 — one integer field, INTEGER class. Size=8 → i64.
// Exercises the case where an eightbyte contains multiple fields of different
// integer widths; they should all classify as INTEGER and coerce to i64.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_dense_int_16
llvm.func @arg_dense_int_16(%arg0: !llvm.struct<(i8, i8, i16, i32, i64)>) {
  // CHECK: llvm.call @c_b6(%{{.*}}, %{{.*}}) : (i64, i64) -> ()
  pop.external_call @c_b6(%arg0) : (!llvm.struct<(i8, i8, i16, i32, i64)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_b6(i64, i64)
}

//===----------------------------------------------------------------------===//
// Group C (continued): Alignment-padded large struct (C2)
//===----------------------------------------------------------------------===//

// -----

// C2: {i8, f32, i8, f64} → indirect via pointer, byval on func param.
// LLVM layout (x86-64): i8@0, pad@1-3, f32@4, i8@8, pad@9-15, f64@16.
// Total size = 24 bytes (> 16) → MEMORY class → indirect pass with byval.
// This tests that structs whose logical field sizes sum to ≤16 bytes can still
// exceed 16 bytes due to alignment padding, triggering the MEMORY path.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_padded_indirect
llvm.func @arg_padded_indirect(%arg0: !llvm.struct<(i8, f32, i8, f64)>) {
  // CHECK: llvm.call @c_c2(%{{.*}}) : (!llvm.ptr {llvm.byval = !llvm.struct<(i8, f32, i8, f64)>}) -> ()
  pop.external_call @c_c2(%arg0) : (!llvm.struct<(i8, f32, i8, f64)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_c2(!llvm.ptr {llvm.byval = !llvm.struct<(i8, f32, i8, f64)>})
}

//===----------------------------------------------------------------------===//
// Group H: Attribute remapping through ABI coercion
//
// These tests verify that funcAttrs, argAttrs, and resAttrs on
// pop.external_call are correctly forwarded to the lowered llvm.func, and
// that argAttrs are remapped when ABI coercion changes the number or order of
// parameters (sret insertion, two-register expansion).
//===----------------------------------------------------------------------===//

// -----

// H1: funcAttrs passthrough → LLVM function has passthrough attribute.
// Verifies that funcAttrs = ["noinline"] on the call site are propagated to
// the LLVM function declaration as passthrough = ["noinline"].
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @func_attrs_passthrough
llvm.func @func_attrs_passthrough(%arg0: i32) {
  // CHECK: llvm.call @c_h1
  pop.external_call @c_h1(%arg0) attributes {funcAttrs = ["noinline"]} : (i32) -> ()
  llvm.return
}
// CHECK: llvm.func @c_h1
// CHECK-SAME: passthrough = {{.*}}"noinline"
}

// -----

// H2: argAttrs remapped through two-register coercion.
// The POP call has one struct arg with {llvm.noundef}. After IntegerPair
// coercion (12-byte struct → (i64, i32)), the argAttr is placed on the first
// LLVM parameter (i64) and the second (i32) gets empty attrs.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_attrs_two_reg
llvm.func @arg_attrs_two_reg(%arg0: !llvm.struct<(i32, i32, i32)>) {
  // CHECK: llvm.call @c_h2
  pop.external_call @c_h2(%arg0) attributes {argAttrs = [{llvm.noundef}]}
    : (!llvm.struct<(i32, i32, i32)>) -> ()
  llvm.return
}
// POP arg 0 → LLVM params 0 (i64) and 1 (i32).  argAttr goes to param 0.
// CHECK: llvm.func @c_h2(i64 {llvm.noundef}, i32)
}

// -----

// H3: argAttrs remapped with sret insertion.
// The POP call returns a 24-byte struct (→ sret hidden pointer at param 0)
// and takes one i32 arg with {llvm.noundef}. remapArgAttrs shifts the user
// attr to param 1 (the original arg), while param 0 (sret ptr) gets only the
// sret attribute set by the lowering.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_attrs_sret
llvm.func @arg_attrs_sret(%arg0: i32) {
  // CHECK: llvm.call @c_h3
  %r = pop.external_call @c_h3(%arg0) attributes {argAttrs = [{llvm.noundef}]}
    : (i32) -> !llvm.struct<(i64, i64, i64)>
  llvm.return
}
// sret ptr at param 0, original arg with user attr at param 1.
// CHECK: llvm.func @c_h3(!llvm.ptr {llvm.sret = !llvm.struct<(i64, i64, i64)>}, i32 {llvm.noundef})
}

//===----------------------------------------------------------------------===//
// Group F: Vector type fields — SSE classification (regression guards)
//
// In the MLIR assembly context, vector<N x f32> inside !llvm.struct<(...)> is
// parsed as mlir::VectorType (MLIR built-in), which IS handled by the existing
// dyn_cast<mlir::VectorType> check in isAllFloatStruct and classifyEightbyte.
// These tests verify that vector fields remain correctly classified as SSE.
//===----------------------------------------------------------------------===//

// -----

// F1: {vector<2 x f32>} (8 bytes, all-float via VectorType) → SSE → f64.
// The vector field is recognized by isAllFloatStruct (VectorType with float
// elements); 8-byte all-float struct → f64 (SSE). Regression guard.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_vec2xf32_struct
llvm.func @arg_vec2xf32_struct(%arg0: !llvm.struct<(vector<2 x f32>)>) {
  // CHECK: llvm.call @c_f1(%{{.*}}) : (f64) -> ()
  pop.external_call @c_f1(%arg0) : (!llvm.struct<(vector<2 x f32>)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_f1(f64)
}

// -----

// F2: {vector<2 x f32>, f64} (16 bytes) → SSEPair → (f64, f64).
// Eightbyte 0 (bytes 0-7): vector<2 x f32> (VectorType) → SSE → f64.
// Eightbyte 1 (bytes 8-15): f64 → SSE → f64.
// Both SSE → SSEPair. Regression guard.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_vec2xf32_and_f64
llvm.func @arg_vec2xf32_and_f64(%arg0: !llvm.struct<(vector<2 x f32>, f64)>) {
  // CHECK: llvm.call @c_f2(%{{.*}}, %{{.*}}) : (f64, f64) -> ()
  pop.external_call @c_f2(%arg0) : (!llvm.struct<(vector<2 x f32>, f64)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_f2(f64, f64)
}

//===----------------------------------------------------------------------===//
// Group N: Nested struct fields (LLVMStructType)
//
// isAllFloatStruct and classifyEightbyte now recurse into nested LLVMStructType
// fields, so float nested structs receive correct SSE classification.
// N1-N3: nested float structs → SSE. N4-N5: integer regression guards.
//===----------------------------------------------------------------------===//

// -----

// N1: {struct<(f32)>} (4 bytes) — CORRECT: isAllFloatStruct now recurses into
// nested LLVMStructType fields. Inner struct<(f32)> is all-float → outer is
// all-float → 4-byte SSE struct → f32.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_nested_float_4
llvm.func @arg_nested_float_4(%arg0: !llvm.struct<(!llvm.struct<(f32)>)>) {
  // CHECK: llvm.call @c_n1(%{{.*}}) : (f32) -> ()
  pop.external_call @c_n1(%arg0) : (!llvm.struct<(!llvm.struct<(f32)>)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_n1(f32)
}

// -----

// N2: {struct<(f32,f32)>} (8 bytes) — CORRECT: isAllFloatStruct recurses into
// nested struct<(f32,f32)> → all-float → 8-byte SSE struct → f64.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_nested_float_8
llvm.func @arg_nested_float_8(%arg0: !llvm.struct<(!llvm.struct<(f32, f32)>)>) {
  // CHECK: llvm.call @c_n2(%{{.*}}) : (f64) -> ()
  pop.external_call @c_n2(%arg0) : (!llvm.struct<(!llvm.struct<(f32, f32)>)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_n2(f64)
}

// -----

// N3: {struct<(f32,f32)>, f64} (16 bytes) — CORRECT: classifyEightbyte now
// recurses into nested LLVMStructType fields.
// Eightbyte 0 (bytes 0-7): struct<(f32,f32)> → recursive classify → SSE → f64.
// Eightbyte 1 (bytes 8-15): f64 → SSE → f64.
// Both SSE → SSEPair (f64, f64).
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_nested_float_and_f64
llvm.func @arg_nested_float_and_f64(%arg0: !llvm.struct<(!llvm.struct<(f32, f32)>, f64)>) {
  // CHECK: llvm.call @c_n3(%{{.*}}, %{{.*}}) : (f64, f64) -> ()
  pop.external_call @c_n3(%arg0) : (!llvm.struct<(!llvm.struct<(f32, f32)>, f64)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_n3(f64, f64)
}

// -----

// N4: {struct<(i32,i32)>} (8 bytes) — CORRECT: nested struct with integer fields.
// isAllFloatStruct correctly returns false (not all float). classifyEightbyte:
// the LLVMStructType field defaults to INTEGER, which is correct here.
// Regression guard: integer nested structs must remain i64.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_nested_int_8
llvm.func @arg_nested_int_8(%arg0: !llvm.struct<(!llvm.struct<(i32, i32)>)>) {
  // CHECK: llvm.call @c_n4(%{{.*}}) : (i64) -> ()
  pop.external_call @c_n4(%arg0) : (!llvm.struct<(!llvm.struct<(i32, i32)>)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_n4(i64)
}

// -----

// N5: {struct<(i32,i32)>, i64} (16 bytes) — CORRECT: eightbyte 0 has
// struct<(i32,i32)> → INTEGER (correct); eightbyte 1 has i64 → INTEGER.
// IntegerPair (i64, i64). Regression guard.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_nested_int_and_i64
llvm.func @arg_nested_int_and_i64(%arg0: !llvm.struct<(!llvm.struct<(i32, i32)>, i64)>) {
  // CHECK: llvm.call @c_n5(%{{.*}}, %{{.*}}) : (i64, i64) -> ()
  pop.external_call @c_n5(%arg0) : (!llvm.struct<(!llvm.struct<(i32, i32)>, i64)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_n5(i64, i64)
}

//===----------------------------------------------------------------------===//
// Group R: Rollback-to-stack (register exhaustion)
//
// A two-eightbyte struct needs two integer registers. When preceding integer
// args leave fewer than two free, the System V ABI passes the whole struct in
// memory (byval) rather than splitting it across a register and the stack.
//===----------------------------------------------------------------------===//

// -----

// R1: two ptrs + three i32 consume all but one integer register, so the
// trailing {i32,i32,i32} (would be IntegerPair) rolls back to memory (byval).
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_rollback_after_five
llvm.func @arg_rollback_after_five(%p0: !llvm.ptr, %p1: !llvm.ptr, %a: i32, %b: i32, %c: i32, %s: !llvm.struct<(i32, i32, i32)>) {
  // CHECK: llvm.call @c_r1(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, !llvm.ptr {llvm.byval = !llvm.struct<(i32, i32, i32)>}) -> ()
  pop.external_call @c_r1(%p0, %p1, %a, %b, %c, %s) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, !llvm.struct<(i32, i32, i32)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_r1(!llvm.ptr, !llvm.ptr, i32, i32, i32, !llvm.ptr {llvm.byval = !llvm.struct<(i32, i32, i32)>})
}

// -----

// R2: control - one ptr + three i32 leave two integer registers free, so the
// same struct fits and stays a two-register IntegerPair (i64, i32).
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_no_rollback_after_four
llvm.func @arg_no_rollback_after_four(%p0: !llvm.ptr, %a: i32, %b: i32, %c: i32, %s: !llvm.struct<(i32, i32, i32)>) {
  // CHECK: llvm.call @c_r2(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, i32, i32, i32, i64, i32) -> ()
  pop.external_call @c_r2(%p0, %a, %b, %c, %s) : (!llvm.ptr, i32, i32, i32, !llvm.struct<(i32, i32, i32)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_r2(!llvm.ptr, i32, i32, i32, i64, i32)
}

// -----

// R3: sret control - the hidden result pointer takes one integer register;
// three i32 leave two free, so the struct still fits as IntegerPair (i64, i32).
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_sret_no_rollback
llvm.func @arg_sret_no_rollback(%a: i32, %b: i32, %c: i32, %s: !llvm.struct<(i32, i32, i32)>) {
  %r = pop.external_call @c_r3(%a, %b, %c, %s) : (i32, i32, i32, !llvm.struct<(i32, i32, i32)>) -> !llvm.struct<(i64, i64, i64)>
  llvm.return
}
// CHECK: llvm.func @c_r3(!llvm.ptr {llvm.sret = !llvm.struct<(i64, i64, i64)>}, i32, i32, i32, i64, i32)
}

// -----

// R4: one extra i32 over R3 tips it over the edge - sret + four i32 leave only
// one integer register, so the struct rolls back to memory (byval).
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_sret_rollback
llvm.func @arg_sret_rollback(%a: i32, %b: i32, %c: i32, %d: i32, %s: !llvm.struct<(i32, i32, i32)>) {
  %r = pop.external_call @c_r4(%a, %b, %c, %d, %s) : (i32, i32, i32, i32, !llvm.struct<(i32, i32, i32)>) -> !llvm.struct<(i64, i64, i64)>
  llvm.return
}
// CHECK: llvm.func @c_r4(!llvm.ptr {llvm.sret = !llvm.struct<(i64, i64, i64)>}, i32, i32, i32, i32, !llvm.ptr {llvm.byval = !llvm.struct<(i32, i32, i32)>})
}

// -----

// R5: SSE control - six f64 leave two SSE registers free, so the {f64,f64}
// struct still fits and stays a two-register SSEPair (f64, f64).
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_sse_no_rollback
llvm.func @arg_sse_no_rollback(%a: f64, %b: f64, %c: f64, %d: f64, %e: f64, %f: f64, %s: !llvm.struct<(f64, f64)>) {
  // CHECK: llvm.call @c_r5(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (f64, f64, f64, f64, f64, f64, f64, f64) -> ()
  pop.external_call @c_r5(%a, %b, %c, %d, %e, %f, %s) : (f64, f64, f64, f64, f64, f64, !llvm.struct<(f64, f64)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_r5(f64, f64, f64, f64, f64, f64, f64, f64)
}

// -----

// R6: one extra f64 over R5 exhausts the SSE registers - seven f64 leave only
// one, so the {f64,f64} struct rolls back to memory (byval).
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_sse_rollback
llvm.func @arg_sse_rollback(%a: f64, %b: f64, %c: f64, %d: f64, %e: f64, %f: f64, %g: f64, %s: !llvm.struct<(f64, f64)>) {
  // CHECK: llvm.call @c_r6(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (f64, f64, f64, f64, f64, f64, f64, !llvm.ptr {llvm.byval = !llvm.struct<(f64, f64)>}) -> ()
  pop.external_call @c_r6(%a, %b, %c, %d, %e, %f, %g, %s) : (f64, f64, f64, f64, f64, f64, f64, !llvm.struct<(f64, f64)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_r6(f64, f64, f64, f64, f64, f64, f64, !llvm.ptr {llvm.byval = !llvm.struct<(f64, f64)>})
}
