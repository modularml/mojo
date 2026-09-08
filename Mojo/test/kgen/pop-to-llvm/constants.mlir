// RUN: kgen-opt %s -lower-kgen-to-llvm | kgen-translate -mlir-to-llvmir | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// COM: Checking the LLVMIR is easier since the constants are collapsed.

// CHECK-LABEL: @array_constant
kgen.func @array_constant() -> !pop.array<2, i32> {
  // CHECK-NEXT: [2 x i32] [i32 1, i32 2]
  %0 = kgen.param.constant: array<2, i32> = <[1, 2]>
  kgen.return %0 : !pop.array<2, i32>
}

// CHECK-LABEL: @simd_constant
kgen.func @simd_constant() -> (!kgen.simd<2, bool>, !kgen.simd<2, si8>, !kgen.scalar<bf16>) {
  // CHECK-NEXT: { <2 x i1>, <2 x i8>, bfloat }
  // CHECK-SAME: { <2 x i1> <i1 true, i1 false>, <2 x i8> <i8 -3, i8 3>, bfloat 1.250000e+00 }
  %0 = kgen.param.constant: simd<2, bool> = <<true, false>>
  %1 = kgen.param.constant: simd<2, si8> = <<-3, 3>>
  %2 = kgen.param.constant: scalar<bf16> = <<"1.25">>
  kgen.return %0, %1, %2 : !kgen.simd<2, bool>, !kgen.simd<2, si8>, !kgen.scalar<bf16>
}

// CHECK-LABEL: @scalar_index_addr_constants
kgen.func @scalar_index_addr_constants() -> (!kgen.scalar<index>, !kgen.scalar<address>) {
  // CHECK-NEXT: { i64, ptr } { i64 1, ptr inttoptr (i64 2 to ptr) }
  %0 = kgen.param.constant: scalar<index> = <<1>>
  %1 = kgen.param.constant: scalar<address> = <<2>>
  kgen.return %0, %1 : !kgen.scalar<index>, !kgen.scalar<address>
}

// CHECK-LABEL: @simd_index_addr_constants
kgen.func @simd_index_addr_constants() -> (!kgen.simd<2, index>, !kgen.simd<2, address>) {
  // CHECK-NEXT: { <2 x i64>, <2 x ptr> }
  // CHECK-SAME: { <2 x i64> <i64 1, i64 11>, <2 x ptr> <ptr inttoptr (i64 2 to ptr), ptr inttoptr (i64 22 to ptr)> }
  %0 = kgen.param.constant: simd<2, index> = <<1, 11>>
  %1 = kgen.param.constant: simd<2, address> = <<2, 22>>
  kgen.return %0, %1 : !kgen.simd<2, index>, !kgen.simd<2, address>
}

// CHECK-LABEL: @store_to_mem
kgen.func @store_to_mem() -> !kgen.pointer<index> {
  // CHECK-NEXT: %[[ALLOCA:.*]] = alloca i64, i64 1, align 4
  // CHECK-NEXT: store i64 42, ptr %1, align 4
  // CHECK-NEXT: ret ptr %[[ALLOCA]]
  %0 = kgen.param.constant: pointer<index> = <store_to_mem(42)>
  kgen.return %0 : !kgen.pointer<index>
}

}
