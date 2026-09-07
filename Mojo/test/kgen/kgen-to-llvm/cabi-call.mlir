// RUN: kgen-opt -lower-kgen-to-llvm -lower-runtime-closures -split-input-file %s | FileCheck %s
//
// Tests for kgen.call / kgen.call_indirect → llvm.call lowering for abi("C")
// callees.
//
// TRANSFORM UNDER TEST:
//   Mojo/lib/KGENToLLVM/LowerKGENToLLVM.cpp  — ConvertKGENCall
//   Mojo/lib/KGENToLLVM/LowerKGENClosuresToLLVM.cpp  — CallIndirectOpConversion
//   Mojo/lib/KGENToLLVM/CABICallHelpers.cpp  — CABICallHelper
//
// Focus: interaction between TailKind and the sret return convention.
//
// On aarch64 AAPCS:
//   {i64,i64,i64} (24 bytes) → sret (hidden pointer in x8)
//   {i32}          (4 bytes)  → integer coercion → i32 register return
//
// A kgen.call marked tail to an abi("C") callee that returns via sret must NOT
// produce a `tail call` in the lowered LLVM IR.  The sret pointer is a local
// alloca on the caller's frame; a tail call would free that frame before the
// callee writes through the pointer, corrupting the return value (MOCO-3841).

//===----------------------------------------------------------------------===//
// T1: sret return — tail call must be suppressed.
//
// {i64,i64,i64} (24 bytes) is sret on aarch64.  A kgen.call with TailKind::Tail
// to such a callee must lower to a regular llvm.call (no tail keyword), because
// the sret alloca lives in the caller's frame.
//===----------------------------------------------------------------------===//

module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {

kgen.func @callee_sret(%s: !llvm.struct<(i64, i64, i64)>) cabi -> !llvm.struct<(i64, i64, i64)> {
  kgen.return %s : !llvm.struct<(i64, i64, i64)>
}

// CHECK-LABEL: llvm.func internal @caller_sret
kgen.func @caller_sret(%s: !llvm.struct<(i64, i64, i64)>) -> !llvm.struct<(i64, i64, i64)> {
  // Regular call (no tail): the sret alloca is on the caller's frame.
  // CHECK-NOT: llvm.call tail @callee_sret
  // CHECK:     llvm.call @callee_sret
  %r = kgen.call tail @callee_sret(%s) : (!llvm.struct<(i64, i64, i64)>) -> !llvm.struct<(i64, i64, i64)>
  kgen.return %r : !llvm.struct<(i64, i64, i64)>
}

} // module

//===----------------------------------------------------------------------===//
// T2: register return — tail call must be preserved.
//
// {i32} (4 bytes) is returned in a register on aarch64 (coerced to i32).
// No sret alloca is created, so TailKind::Tail must flow through to the
// lowered llvm.call unchanged.
//===----------------------------------------------------------------------===//

// -----

module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {

kgen.func @callee_reg(%s: !llvm.struct<(i32)>) cabi -> !llvm.struct<(i32)> {
  kgen.return %s : !llvm.struct<(i32)>
}

// CHECK-LABEL: llvm.func internal @caller_reg
kgen.func @caller_reg(%s: !llvm.struct<(i32)>) -> !llvm.struct<(i32)> {
  // Tail call preserved: register return has no sret alloca.
  // CHECK: llvm.call tail @callee_reg
  %r = kgen.call tail @callee_reg(%s) : (!llvm.struct<(i32)>) -> !llvm.struct<(i32)>
  kgen.return %r : !llvm.struct<(i32)>
}

} // module

//===----------------------------------------------------------------------===//
// T3: indirect sret return — tail call must be suppressed.
//
// This mirrors T1 for a bare abi("C") function pointer. The indirect call path
// builds the same sret alloca, so it must also drop TailKind::Tail.
//===----------------------------------------------------------------------===//

// -----

module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {

// CHECK-LABEL: llvm.func internal @caller_indirect_sret
kgen.func @caller_indirect_sret(
    %callee: !kgen.generator<(!llvm.struct<(i64, i64, i64)>) cabi -> !llvm.struct<(i64, i64, i64)>>,
    %s: !llvm.struct<(i64, i64, i64)>
) -> !llvm.struct<(i64, i64, i64)> {
  // Regular call (no tail): the sret alloca is on the caller's frame.
  // CHECK-NOT: llvm.call tail
  // CHECK:     llvm.call %arg0
  %r = kgen.call_indirect tail %callee(%s) : (!llvm.struct<(i64, i64, i64)>) cabi -> !llvm.struct<(i64, i64, i64)>
  kgen.return %r : !llvm.struct<(i64, i64, i64)>
}

} // module

//===----------------------------------------------------------------------===//
// T4: indirect register return — tail call must be preserved.
//
// This mirrors T2 for a bare abi("C") function pointer. No sret alloca is
// created, so TailKind::Tail must flow through to the lowered llvm.call.
//===----------------------------------------------------------------------===//

// -----

module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {

// CHECK-LABEL: llvm.func internal @caller_indirect_reg
kgen.func @caller_indirect_reg(
    %callee: !kgen.generator<(!llvm.struct<(i32)>) cabi -> !llvm.struct<(i32)>>,
    %s: !llvm.struct<(i32)>
) -> !llvm.struct<(i32)> {
  // Tail call preserved: register return has no sret alloca.
  // CHECK: llvm.call tail %arg0
  %r = kgen.call_indirect tail %callee(%s) : (!llvm.struct<(i32)>) cabi -> !llvm.struct<(i32)>
  kgen.return %r : !llvm.struct<(i32)>
}

} // module
