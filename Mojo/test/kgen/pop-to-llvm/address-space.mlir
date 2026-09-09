// RUN: kgen-opt -split-input-file -pass-pipeline='builtin.module(lower-global-pop-to-llvm,kgen.func(lower-pop-to-llvm))'  %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: kgen.func @global_load
  kgen.func @global_load() -> !kgen.scalar<f32> {
    // CHECK-NEXT: %0 = llvm.mlir.addressof @my_global : !llvm.ptr<3>
    // CHECK-NEXT: %1 = llvm.bitcast %0 : !llvm.ptr<3> to !llvm.ptr<3>
    // CHECK-NEXT: %2 = builtin.unrealized_conversion_cast %1 : !llvm.ptr<3> to !kgen.pointer<scalar<f32>, 3>
    // CHECK-NEXT: %3 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
    // CHECK-NEXT: %4 = builtin.unrealized_conversion_cast %3 : f32 to !kgen.scalar<f32>
    %0 = pop.global_alloc "my_global" 2 x !kgen.scalar<f32> address_space 3 align 4
    %1 = pop.load %0 :!kgen.pointer<scalar<f32>, 3>
    kgen.return %1 : !kgen.scalar<f32>
  }

  // CHECK-LABEL: llvm.mlir.global internal @my_global() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<2 x f32>
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: kgen.func @global_store
  kgen.func @global_store(%arg0: !kgen.scalar<f32>) {
    // CHECK-NEXT: %0 = builtin.unrealized_conversion_cast %arg0 : !kgen.scalar<f32> to f32
    // CHECK-NEXT: %1 = llvm.mlir.addressof @my_global : !llvm.ptr<3>
    // CHECK-NEXT: %2 = llvm.bitcast %1 : !llvm.ptr<3> to !llvm.ptr<3>
    // CHECK-NEXT: %3 = builtin.unrealized_conversion_cast %2 : !llvm.ptr<3> to !kgen.pointer<scalar<f32>, 3>
    // CHECK-NEXT: llvm.store %0, %2 {alignment = 4 : i64} : f32, !llvm.ptr
    %0 = pop.global_alloc "my_global" 2 x !kgen.scalar<f32> address_space 3 align 4
    pop.store %arg0, %0 :!kgen.pointer<scalar<f32>, 3>
    kgen.return
  }

  // CHECK-LABEL: llvm.mlir.global internal @my_global() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<2 x f32>
}

// -----

#target = #kgen.target<triple="", arch="", features="", data_layout="", simd_bit_width=128> : !kgen.target

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @memcpy
kgen.func @memcpy(%p: !kgen.pointer<scalar<f32>>, %q: !kgen.pointer<scalar<f32>, 1>) {
  // CHECK:      [[SRC:%.*]] = builtin.unrealized_conversion_cast %arg1 : !kgen.pointer<scalar<f32>, 1> to !llvm.ptr<1>
  // CHECK-NEXT: [[DST:%.*]] = builtin.unrealized_conversion_cast %arg0 : !kgen.pointer<scalar<f32>> to !llvm.ptr
  // CHECK-NEXT: [[IDX4:%.*]] = kgen.param.constant = <4>
  // CHECK-NEXT: [[IDX:%.*]] = builtin.unrealized_conversion_cast [[IDX4]] : index to i64
  // CHECK-NEXT: "llvm.intr.memcpy"([[DST]], [[SRC]], [[IDX]])
  %len = kgen.param.constant: index = <get_sizeof(scalar<f32>, #target)>
  pop.memcpy %p, %q, %len : !kgen.pointer<scalar<f32>, 1> -> !kgen.pointer<scalar<f32>>
  kgen.return
}

}
