// RUN: kgen-opt -split-input-file -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @masked_load_with_alignment
kgen.func @masked_load_with_alignment(
    %ptr: !kgen.pointer<scalar<f32>>,
    %mask: !kgen.simd<4, bool>,
    %passthrough: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK: llvm.intr.masked.load {{.*}} {alignment = 16 : i32}
  %align_builtin = llvm.mlir.constant(16 : i32) : i32
  %align = pop.cast_from_builtin %align_builtin : i32 to !kgen.scalar<si32>
  %0 = pop.call_llvm_intrinsic "llvm.masked.load", (%ptr, %align, %mask, %passthrough) :
    (!kgen.pointer<scalar<f32>>, !kgen.scalar<si32>, !kgen.simd<4, bool>, !kgen.simd<4, f32>) -> !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: @masked_store_with_alignment
kgen.func @masked_store_with_alignment(
    %value: !kgen.simd<4, f32>,
    %ptr: !kgen.pointer<scalar<f32>>,
    %mask: !kgen.simd<4, bool>) {
  // CHECK: llvm.intr.masked.store {{.*}} {alignment = 16 : i32}
  %align_builtin = llvm.mlir.constant(16 : i32) : i32
  %align = pop.cast_from_builtin %align_builtin : i32 to !kgen.scalar<si32>
  pop.call_llvm_intrinsic "llvm.masked.store", (%value, %ptr, %align, %mask) :
    (!kgen.simd<4, f32>, !kgen.pointer<scalar<f32>>, !kgen.scalar<si32>, !kgen.simd<4, bool>) -> ()
  kgen.return
}

// CHECK-LABEL: @masked_gather_with_alignment
kgen.func @masked_gather_with_alignment(
    %ptrs: !kgen.simd<4, index>,
    %mask: !kgen.simd<4, bool>,
    %passthrough: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK: llvm.call_intrinsic "llvm.masked.gather"({{.*}}) : (vector<4xi64> {align = 8 : i32}, vector<4xi1>, vector<4xf32>)
  %align_builtin = llvm.mlir.constant(8 : i32) : i32
  %align = pop.cast_from_builtin %align_builtin : i32 to !kgen.scalar<si32>
  %0 = pop.call_llvm_intrinsic "llvm.masked.gather", (%ptrs, %align, %mask, %passthrough) :
    (!kgen.simd<4, index>, !kgen.scalar<si32>, !kgen.simd<4, bool>, !kgen.simd<4, f32>) -> !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: @masked_scatter_with_alignment
kgen.func @masked_scatter_with_alignment(
    %value: !kgen.simd<4, f32>,
    %ptrs: !kgen.simd<4, index>,
    %mask: !kgen.simd<4, bool>) {
  // CHECK: llvm.call_intrinsic "llvm.masked.scatter"({{.*}}) : (vector<4xf32>, vector<4xi64> {align = 8 : i32}, vector<4xi1>)
  %align_builtin = llvm.mlir.constant(8 : i32) : i32
  %align = pop.cast_from_builtin %align_builtin : i32 to !kgen.scalar<si32>
  pop.call_llvm_intrinsic "llvm.masked.scatter", (%value, %ptrs, %align, %mask) :
    (!kgen.simd<4, f32>, !kgen.simd<4, index>, !kgen.scalar<si32>, !kgen.simd<4, bool>) -> ()
  kgen.return
}

}

