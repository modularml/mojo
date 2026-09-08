// RUN: kgen-opt -split-input-file -pass-pipeline='builtin.module(lower-global-pop-to-llvm,kgen.func(lower-pop-to-llvm))'  %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: extern_ptr_symbols
  kgen.func @extern_ptr_symbols() {
    // CHECK: %0 = llvm.mlir.addressof @hello : !llvm.ptr<3>
    // CHECK: %1 = llvm.mlir.addressof @hello_0 : !llvm.ptr<1>
    %0 = pop.extern_ptr_symbol "hello" alignment <1> : !kgen.pointer<scalar<f32>, 3>
    %1 = pop.extern_ptr_symbol "hello" alignment <2> : !kgen.pointer<scalar<f32>, 1>

    // CHECK: %2 = llvm.mlir.addressof @world : !llvm.ptr<1>
    %2 = pop.extern_ptr_symbol "world" alignment <2> : !kgen.pointer<scalar<f32>, 1>

    kgen.return
  }
  // CHECK: llvm.mlir.global external @hello() {addr_space = 3 : i32, alignment = 1 : i64, dso_local} : f32
  // CHECK: llvm.mlir.global external @hello_0() {addr_space = 1 : i32, alignment = 2 : i64, dso_local} : f32
  // CHECK: llvm.mlir.global external @world() {addr_space = 1 : i32, alignment = 2 : i64, dso_local} : f32
}
