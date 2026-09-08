// RUN: kgen-opt -split-input-file -lower-global-pop-to-llvm %s | FileCheck %s

// Unit tests for C ABI lowering on ARM64 (AAPCS — Procedure Call Standard
// for the Arm 64-bit Architecture).
//
// PURPOSE: These are UNIT tests that lock the current behavior of the lowering
// transform. They exist to catch regressions during refactoring. They may need
// updating if we discover ABI bugs or add new features. The system tests are
// the authoritative check for correct end-to-end C ABI behavior:
//   KGEN/test/mojo-integration/extern-c-abi/
//
// TRANSFORM UNDER TEST:
//   Mojo/lib/KGENToLLVM/LowerPOPToLLVM.cpp  — ConvertPOPExternalCall
//   Mojo/lib/KGENToLLVM/CABIAAPCS.cpp        — AAPCSABIInfo classifier
//   Mojo/lib/KGENToLLVM/CABILowering.cpp     — shared utilities
//
// RELATED TEST FILES:
//   extern-c-abi.mlir          — platform-independent tests (DefaultCABIInfo,
//                                empty triple): struct not field-exploded,
//                                multi-value return, identity pass-through
//   extern-c-abi-x86-64.mlir   — x86-64 System V AMD64 tests
//
// INPUT FORMAT NOTE: Each test uses llvm.func (not kgen.func) as the enclosing function.
// The -lower-global-pop-to-llvm pass processes pop.external_call ops
// after kgen.func has already been lowered to llvm.func by an earlier pass.
// For coercion to work, createEntryBlockAlloca requires the enclosing
// function to be an LLVM::LLVMFuncOp. These tests exercise that final
// lowering step in isolation.
//
// KEY DIFFERENCES FROM x86-64:
//   1. HFA (Homogeneous Float Aggregate): all-float structs with ≤4 fields
//      pass via SIMD registers (V0–V3) as identity — no coercion.
//   2. Two-register structs always use IntegerPair (i64, iN) regardless of
//      field types. There is no SSEPair or Mixed class on ARM64.
//   3. Large struct args (>16 bytes) are passed indirectly WITHOUT byval.
//      x86-64 sets byval; ARM64 does not.
//   4. Variadic HFA args — Darwin vs Linux split:
//      Darwin (macOS/iOS): HFA identity bypassed; struct coerced to integer
//        so va_arg reads from the GP save area (flat va_list).
//      Linux AAPCS64: HFA identity preserved; struct stays as-is so LLVM
//        places it in SIMD registers and va_arg reads from the VR save area.
//   5. Variadic float args < 64 bits — Darwin vs Linux split:
//      Darwin: bitcast to integer to prevent LLVM float→double promotion
//        (Darwin va_list reads from GP only, so bits must be in GPRs).
//      Linux: no bitcast; LLVM places floats in SIMD registers and va_arg
//        reads from the VR save area correctly.
//
// TEST TABLE:
// Group A — Small struct args (1–8 bytes): Integer coercion (same as x86-64)
//   A1  arg_1byte_int_struct      {i8}       → i8
//   A2  arg_2byte_int_struct      {i8,i8}    → i16
//   A3  arg_3byte_int_struct      {i8,i8,i8} → i32
//   A4  arg_8byte_int_struct      {i32,i32}  → i64
//
// Group F — HFA (Homogeneous Float Aggregate): identity for fixed args
//   F1  arg_hfa_2xf32             {f32,f32}             → identity (SIMD)
//   F2  arg_hfa_4xf32             {f32,f32,f32,f32}     → identity (max HFA)
//   F3  arg_hfa_2xf64             {f64,f64}             → identity (SIMD)
//   F4  arg_not_hfa_5xf32         {f32,f32,f32,f32,f32} → indirect (>16 AND >4 fields)
//   F5  arg_not_hfa_mixed         {f32,i32}             → integer coercion (not all-float)
//   F6  ret_hfa                   return {f32,f32}       → identity (HFA return)
//
// Group B — 9–16 byte struct args: IntegerPair (always i64+iN on ARM64)
//   B1  arg_int_pair_12           {i32,i32,i32}          → (i64, i32)
//   B2  arg_int_pair_16           {i64,i64}              → (i64, i64)
//   B3  arg_mixed_12_arm64        {f32,f32,i32}          → (i64, i32)  [no SSEPair/Mixed]
//
// Group C — Large struct args (>16 bytes): indirect, NO byval
//   C1  arg_large_indirect_no_byval {i64,i64,i64} → ptr, no byval attribute
//
// Group D — Return value coercion
//   D1  ret_small_int_struct      returns {i32}          → call returns i32
//   D2  ret_int_pair              returns {i32,i32,i32}  → call returns {i64,i32}
//   D3  ret_sret                  returns {i64,i64,i64}  → sret hidden ptr
//
// Group G — Variadic functions: ARM64 Darwin vs Linux split
//   G1  variadic_hfa_linux        Linux:  variadic HFA arg  → identity (VR save area)
//   G2  variadic_float_linux      Linux:  variadic f32 arg  → no bitcast (SIMD reg)
//   G3  variadic_hfa_darwin       Darwin: variadic HFA arg  → integer coercion (GP)
//   G4  variadic_float_darwin     Darwin: variadic f32 arg  → bitcast to i32 (GP)
//
// MODULE HEADER NOTE: Each test uses the standard ARM64 Linux data layout. The data_layout string
// drives struct size and alignment computation. The triple selects
// AAPCSABIInfo.

//===----------------------------------------------------------------------===//
// Group A: Small integer struct arguments (1–8 bytes)
//
// ARM64 AAPCS and x86-64 System V share the same integer coercion rules for
// small non-HFA structs. These tests confirm ARM64 applies them correctly.
//===----------------------------------------------------------------------===//

// A1: 1-byte struct {i8} → i8.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_1byte_int_struct
llvm.func @arg_1byte_int_struct(%arg0: !llvm.struct<(i8)>) {
  // CHECK: llvm.call @c_a1(%{{.*}}) : (i8) -> ()
  pop.external_call @c_a1(%arg0) : (!llvm.struct<(i8)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_a1(i8)
}

// -----

// A2: 2-byte struct {i8, i8} → i16.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_2byte_int_struct
llvm.func @arg_2byte_int_struct(%arg0: !llvm.struct<(i8, i8)>) {
  // CHECK: llvm.call @c_a2(%{{.*}}) : (i16) -> ()
  pop.external_call @c_a2(%arg0) : (!llvm.struct<(i8, i8)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_a2(i16)
}

// -----

// A3: 3-byte struct {i8, i8, i8} → i32 (rounds up).
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_3byte_int_struct
llvm.func @arg_3byte_int_struct(%arg0: !llvm.struct<(i8, i8, i8)>) {
  // CHECK: llvm.call @c_a3(%{{.*}}) : (i32) -> ()
  pop.external_call @c_a3(%arg0) : (!llvm.struct<(i8, i8, i8)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_a3(i32)
}

// -----

// A4: 8-byte struct {i32, i32} → i64.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_8byte_int_struct
llvm.func @arg_8byte_int_struct(%arg0: !llvm.struct<(i32, i32)>) {
  // CHECK: llvm.call @c_a4(%{{.*}}) : (i64) -> ()
  pop.external_call @c_a4(%arg0) : (!llvm.struct<(i32, i32)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_a4(i64)
}

//===----------------------------------------------------------------------===//
// Group F: HFA (Homogeneous Float Aggregate)
//
// AAPCS rule: an all-float struct with ≤4 fields of the same float type is an
// HFA and uses SIMD registers (V0–V3). The lowering treats HFA args as
// identity (no coercion), deferring register assignment to LLVM's backend.
//
// The identity classification means the LLVM call passes the struct type
// directly, identical to the DefaultCABIInfo (no-triple) behavior.
//
// HFA detection (isAllFloatStruct):
//   - All fields must be the same float type (f16, f32, or f64)
//   - At most 4 fields (5+ is not HFA)
//   - Mixed int+float or heterogeneous float types → not HFA
//
// Variadic HFA args are explicitly NOT treated as HFA (see Group G).
//===----------------------------------------------------------------------===//

// -----

// F1: {f32, f32} (8 bytes, all-float, 2 fields) → HFA → identity.
// The LLVM struct type is passed as-is. No store/load bitcast.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_hfa_2xf32
llvm.func @arg_hfa_2xf32(%arg0: !llvm.struct<(f32, f32)>) {
  // CHECK: llvm.call @c_f1(%{{.*}}) : (!llvm.struct<(f32, f32)>) -> ()
  pop.external_call @c_f1(%arg0) : (!llvm.struct<(f32, f32)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_f1(!llvm.struct<(f32, f32)>)
}

// -----

// F2: {f32, f32, f32, f32} (16 bytes, all-float, 4 fields) → HFA → identity.
// 4 is the maximum field count for HFA.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_hfa_4xf32
llvm.func @arg_hfa_4xf32(%arg0: !llvm.struct<(f32, f32, f32, f32)>) {
  // CHECK: llvm.call @c_f2(%{{.*}}) : (!llvm.struct<(f32, f32, f32, f32)>) -> ()
  pop.external_call @c_f2(%arg0) : (!llvm.struct<(f32, f32, f32, f32)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_f2(!llvm.struct<(f32, f32, f32, f32)>)
}

// -----

// F3: {f64, f64} (16 bytes, all-float doubles, 2 fields) → HFA → identity.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_hfa_2xf64
llvm.func @arg_hfa_2xf64(%arg0: !llvm.struct<(f64, f64)>) {
  // CHECK: llvm.call @c_f3(%{{.*}}) : (!llvm.struct<(f64, f64)>) -> ()
  pop.external_call @c_f3(%arg0) : (!llvm.struct<(f64, f64)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_f3(!llvm.struct<(f64, f64)>)
}

// -----

// F4: {f32 × 5} (20 bytes) → NOT HFA (5 fields exceeds the 4-field limit) AND
// size >16 bytes → indirect. The isAllFloatStruct check rejects 5-field
// structs, so the struct falls through to size classification (>16 → indirect).
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_not_hfa_5xf32
llvm.func @arg_not_hfa_5xf32(%arg0: !llvm.struct<(f32, f32, f32, f32, f32)>) {
  // CHECK: llvm.call @c_f4(%{{.*}}) : (!llvm.ptr) -> ()
  pop.external_call @c_f4(%arg0) : (!llvm.struct<(f32, f32, f32, f32, f32)>) -> ()
  llvm.return
}
}

// -----

// F5: {f32, i32} (8 bytes) → NOT HFA (mixed int+float) → integer coercion → i64.
// isAllFloatStruct returns false for mixed types, so the struct is classified
// as a non-HFA integer struct.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_not_hfa_mixed
llvm.func @arg_not_hfa_mixed(%arg0: !llvm.struct<(f32, i32)>) {
  // CHECK: llvm.call @c_f5(%{{.*}}) : (i64) -> ()
  // CHECK-NOT: llvm.call @c_f5(%{{.*}}) : (!llvm.struct
  pop.external_call @c_f5(%arg0) : (!llvm.struct<(f32, i32)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_f5(i64)
}

// -----

// F6: return {f32, f32} → HFA → identity return (no coercion).
// The call returns the struct type directly, and the return value is used
// as-is. No extractvalue or store/load needed.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @ret_hfa
llvm.func @ret_hfa() {
  // CHECK: llvm.call @c_f6() : () -> !llvm.struct<(f32, f32)>
  %r = pop.external_call @c_f6() : () -> !llvm.struct<(f32, f32)>
  llvm.return
}
// CHECK: llvm.func @c_f6() -> !llvm.struct<(f32, f32)>
}

//===----------------------------------------------------------------------===//
// Group B: Two-register struct arguments (9–16 bytes)
//
// ARM64 AAPCS always uses IntegerPair for non-HFA structs in this range.
// There is no SSEPair or Mixed class — both registers are always integer
// (i64 for the first 8 bytes, iN for the remaining bytes). This contrasts
// with x86-64 which classifies each eightbyte independently.
//===----------------------------------------------------------------------===//

// -----

// B1: {i32, i32, i32} (12 bytes) → IntegerPair → (i64, i32).
// ARM64: first 8 bytes → i64, remaining 4 bytes → i32.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
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
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_int_pair_16
llvm.func @arg_int_pair_16(%arg0: !llvm.struct<(i64, i64)>) {
  // CHECK: llvm.call @c_b2(%{{.*}}, %{{.*}}) : (i64, i64) -> ()
  pop.external_call @c_b2(%arg0) : (!llvm.struct<(i64, i64)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_b2(i64, i64)
}

// -----

// B3: {f32, f32, i32} (12 bytes) → IntegerPair → (i64, i32).
// This struct has float fields but it is NOT HFA (mixed types). ARM64 always
// uses IntegerPair for non-HFA 9–16 byte structs, regardless of field types.
// Compare with x86-64 which classifies each eightbyte and could produce Mixed.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_mixed_12_arm64
llvm.func @arg_mixed_12_arm64(%arg0: !llvm.struct<(f32, f32, i32)>) {
  // CHECK: llvm.call @c_b3(%{{.*}}, %{{.*}}) : (i64, i32) -> ()
  // CHECK-NOT: llvm.call @c_b3(%{{.*}}) : (f64
  pop.external_call @c_b3(%arg0) : (!llvm.struct<(f32, f32, i32)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_b3(i64, i32)
}

//===----------------------------------------------------------------------===//
// Group C: Large struct arguments (>16 bytes) — indirect, NO byval
//
// ARM64 AAPCS: structs >16 bytes are passed by pointer (X8 register).
// Unlike x86-64, ARM64 does NOT use the byval attribute. The pointer is
// passed directly without any stack copy annotation. addByvalAttrsToFunc
// is only called when triple.isX86().
//===----------------------------------------------------------------------===//

// -----

// C1: {i64, i64, i64} (24 bytes) → indirect via pointer, no byval.
// The function declaration takes !llvm.ptr WITHOUT byval. This is the key
// behavioral difference from x86-64 (see extern-c-abi-x86-64.mlir test C1).
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_large_indirect_no_byval
llvm.func @arg_large_indirect_no_byval(%arg0: !llvm.struct<(i64, i64, i64)>) {
  // CHECK: llvm.call @c_c1(%{{.*}}) : (!llvm.ptr) -> ()
  pop.external_call @c_c1(%arg0) : (!llvm.struct<(i64, i64, i64)>) -> ()
  llvm.return
}
// The function takes a plain ptr — no byval attribute.
// CHECK: llvm.func @c_c1(!llvm.ptr)
// CHECK-NOT: llvm.byval
}

//===----------------------------------------------------------------------===//
// Group D: Return value coercion
//===----------------------------------------------------------------------===//

// -----

// D1: return {i32} (4 bytes) → call returns i32, bitcast back to struct.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
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
// !llvm.struct<(i64, i32)>. Two-register reconstruction.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
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

// D3: return {i64, i64, i64} (24 bytes) → sret.
// ARM64 uses X8 for the sret pointer (same calling convention shape as x86-64
// for large returns, though the register differs at the hardware level).
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @ret_sret
llvm.func @ret_sret() {
  // CHECK: llvm.call @c_d3(%{{.*}}) : (!llvm.ptr) -> ()
  // CHECK: llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(i64, i64, i64)>
  %r = pop.external_call @c_d3() : () -> !llvm.struct<(i64, i64, i64)>
  llvm.return
}
// CHECK: llvm.func @c_d3(!llvm.ptr {llvm.sret = !llvm.struct<(i64, i64, i64)>})
}

//===----------------------------------------------------------------------===//
// Group G: Variadic functions — Darwin vs Linux ARM64 split
//
// ARM64 has two different va_list layouts depending on the OS:
//
// Darwin (macOS/iOS): va_list is a flat pointer into a contiguous GP/stack
//   area. All variadic args — including HFA structs and floats — must land in
//   GPRs. The lowering coerces HFA structs to integer and bitcasts f32→i32 to
//   prevent LLVM float→double promotion from putting them in SIMD registers.
//
// Linux AAPCS64: va_list has separate GP (__gr_top) and VR (__vr_top) save
//   areas. HFA variadic args and float args land in SIMD/VR registers, and
//   va_arg for HFA structs reads from the VR save area. The lowering preserves
//   HFA identity and does NOT bitcast floats — LLVM places them in SIMD regs.
//
// References:
//   AAPCS64 spec IHI0055 §B.4 — va_list structure definition
//   clang/lib/CodeGen/Targets/AArch64.cpp — EmitDarwinVAArg vs EmitAAPCSVAArg
//===----------------------------------------------------------------------===//

// -----

// G1: Linux ARM64 — variadic HFA arg stays identity.
// {f32, f32} is HFA. On Linux, even variadic HFA args use identity: the struct
// is passed as-is so LLVM places it in SIMD registers where Linux va_arg for
// HFA structs reads from the VR save area.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @variadic_hfa_linux
llvm.func @variadic_hfa_linux(%fixed: i32, %s: !llvm.struct<(f32, f32)>) {
  // HFA struct remains identity on Linux — passed as struct, not coerced to i64.
  // CHECK: llvm.call @c_g1(%{{.*}}, %{{.*}}) : (i32, !llvm.struct<(f32, f32)>) -> ()
  // CHECK-NOT: llvm.call @c_g1(%{{.*}}, %{{.*}}) : (i32, i64)
  pop.external_call @c_g1(%fixed, %s) attributes {numFixedArgs = 1 : index}
    : (i32, !llvm.struct<(f32, f32)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_g1(i32, ...)
}

// -----

// G2: Linux ARM64 — variadic f32 arg, no bitcast.
// On Linux, LLVM places f32 in SIMD registers for variadic calls. Linux va_arg
// for float structs reads from the VR save area, so the bits arrive correctly
// without any bitcast. No applyARM64VariadicFloatBitcast on Linux.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @variadic_float_linux
llvm.func @variadic_float_linux(%fixed: i32, %f: f32) {
  // No bitcast on Linux — f32 is passed as f32 directly.
  // CHECK-NOT: llvm.bitcast %{{.*}} : f32 to i32
  // CHECK: llvm.call @c_g2(%{{.*}}, %{{.*}}) : (i32, f32) -> ()
  pop.external_call @c_g2(%fixed, %f) attributes {numFixedArgs = 1 : index}
    : (i32, f32) -> ()
  llvm.return
}
// CHECK: llvm.func @c_g2(i32, ...)
}

// -----

// G3: Darwin ARM64 — variadic HFA arg coerced to integer.
// {f32, f32} is HFA. On Darwin, variadic HFA args must use GPRs because
// Darwin's flat va_list has no separate VR save area. The isVariadicArg +
// isDarwin flag suppresses the HFA identity return, and the struct falls
// through to size classification: 8 bytes → i64.
module attributes {M.target_info = #M.target<triple="aarch64-apple-macosx12.0.0", arch="", features="", data_layout="e-m:o-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @variadic_hfa_darwin
llvm.func @variadic_hfa_darwin(%fixed: i32, %s: !llvm.struct<(f32, f32)>) {
  // HFA struct coerced to i64 on Darwin so va_arg reads from GP save area.
  // CHECK: llvm.call @c_g3(%{{.*}}, %{{.*}}) : (i32, i64) -> ()
  // CHECK-NOT: llvm.call @c_g3(%{{.*}}, %{{.*}}) : (i32, !llvm.struct
  pop.external_call @c_g3(%fixed, %s) attributes {numFixedArgs = 1 : index}
    : (i32, !llvm.struct<(f32, f32)>) -> ()
  llvm.return
}
// CHECK: llvm.func @c_g3(i32, ...)
}

// -----

// G4: Darwin ARM64 — variadic f32 arg bitcast to i32.
// On Darwin, LLVM would otherwise promote f32 to f64 for variadic calls
// (standard C float→double promotion), placing the value in a SIMD register.
// Darwin's flat va_list reads from GPRs only, so bits would be lost. The
// applyARM64VariadicFloatBitcast helper bitcasts f32→i32 to force bits into GPR.
module attributes {M.target_info = #M.target<triple="aarch64-apple-macosx12.0.0", arch="", features="", data_layout="e-m:o-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @variadic_float_darwin
llvm.func @variadic_float_darwin(%fixed: i32, %f: f32) {
  // CHECK: [[BITS:%.*]] = llvm.bitcast %{{.*}} : f32 to i32
  // CHECK: llvm.call @c_g4(%{{.*}}, [[BITS]])
  pop.external_call @c_g4(%fixed, %f) attributes {numFixedArgs = 1 : index}
    : (i32, f32) -> ()
  llvm.return
}
// CHECK: llvm.func @c_g4(i32, ...)
}
