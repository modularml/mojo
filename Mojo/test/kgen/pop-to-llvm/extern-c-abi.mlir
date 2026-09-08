// RUN: kgen-opt -split-input-file -lower-global-pop-to-llvm %s | FileCheck %s

// Unit tests for C ABI lowering with an empty target triple (DefaultCABIInfo).
//
// PURPOSE: These are UNIT tests that lock platform-independent behavior of the
// lowering transform. They exist to catch regressions during refactoring. They
// may need updating if we discover ABI bugs or add new features. The system
// tests are the authoritative check for correct end-to-end C ABI behavior:
//   KGEN/test/mojo-integration/extern-c-abi/
//
// TRANSFORM UNDER TEST:
//   Mojo/lib/KGENToLLVM/LowerPOPToLLVM.cpp  — ConvertPOPExternalCall
//   Mojo/lib/KGENToLLVM/CABILowering.cpp     — DefaultCABIInfo (empty triple)
//
// RELATED TEST FILES:
//   extern-c-abi-x86-64.mlir  — x86-64 System V AMD64 ABI tests (integer/SSE
//                               coercion, two-register pairs, byval, sret)
//   extern-c-abi-aarch64.mlir — ARM64 AAPCS tests (HFA identity, integer pair,
//                               indirect without byval, variadic)
//
// INPUT FORMAT NOTE: These tests use kgen.func (not llvm.func) as the enclosing
// function. DefaultCABIInfo always returns identity for all arguments and return
// values — no struct coercion occurs, so createEntryBlockAlloca is never called.
// The inputs use POP/kgen types (kgen.struct, scalar<...>); the outputs use
// LLVM types (!llvm.struct, iN).
//
// TEST TABLE:
//   I1  extern_c_struct      struct arg          → single !llvm.struct (not N scalars)
//   I2  ret_struct_identity  single struct return → !llvm.struct unchanged (no coerce)
//
// MODULE HEADER NOTE: Empty triple ("") selects DefaultCABIInfo, which applies
// identity to all args and return values — structs pass through unchanged.

//===----------------------------------------------------------------------===//
// I1: Struct argument — identity (no coercion)
//
// Invariant: a POP struct argument must be passed as a single LLVM struct
// value, not exploded into its fields. The LLVM call signature must match
// exactly what the C side declares — one struct parameter, not N scalars.
//===----------------------------------------------------------------------===//

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @extern_c_struct
kgen.func @extern_c_struct() {
  %s = kgen.param.constant: struct<(scalar<si8>, scalar<si8>, scalar<si8>, scalar<si8>)> = <{ 0, 1, 2, 3 }>

  // CHECK: llvm.call @c_func(%{{.*}}) : (!llvm.struct<(i8, i8, i8, i8)>) -> ()
  pop.external_call @c_func(%s)
    : (!kgen.struct<(scalar<si8>, scalar<si8>, scalar<si8>, scalar<si8>)>) -> ()
  kgen.return
}
// CHECK: llvm.func @c_func(!llvm.struct<(i8, i8, i8, i8)>)
}

// -----

//===----------------------------------------------------------------------===//
// I2: Struct return — identity (no coercion)
//
// A pop.external_call returning a single struct with DefaultCABIInfo (empty
// triple) must lower to an llvm.call returning the equivalent !llvm.struct
// type unchanged. No integer coercion, no sret, no extractvalue — the struct
// is the call's direct return value.
//===----------------------------------------------------------------------===//

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
// CHECK-LABEL: @ret_struct_identity
kgen.func @ret_struct_identity() {
  // CHECK: llvm.call @c_ret() : () -> !llvm.struct<(i32, i32)>
  %0 = pop.external_call @c_ret() : () -> !kgen.struct<(scalar<si32>, scalar<si32>)>
  kgen.return
}
// CHECK: llvm.func @c_ret() -> !llvm.struct<(i32, i32)>
}
