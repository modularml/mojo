// RUN: kgen-opt -split-input-file -allow-unregistered-dialect -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @stack_allocation
kgen.func @stack_allocation(%cond: !kgen.scalar<bool>) {
  // CHECK-NEXT: %[[C16:.*]] = llvm.mlir.constant(16 : i64) : i64
  // CHECK-NEXT: %[[PTR0:.*]] = llvm.alloca %[[C16]] x f32 {alignment = 4 : i64}
  // CHECK-NEXT: llvm.intr.lifetime.start %[[PTR0]]
  %0 = pop.stack_allocation 16 x !kgen.simd<1, f32>
  // CHECK: hlcf.if
  hlcf.if %cond {
    // CHECK-NEXT: %[[C4:.*]] = llvm.mlir.constant(4 : i64) : i64
    // CHECK-NEXT: %[[PTR1:.*]] = llvm.alloca %[[C4]] x vector<4xf32> {alignment = 16 : i64}
    // CHECK-NEXT: llvm.intr.lifetime.start %[[PTR1]]
    // CHECK-NEXT: llvm.intr.lifetime.end %[[PTR1]]
    %1 = pop.stack_allocation 4 x !kgen.simd<4, f32>
    // CHECK: }
    hlcf.yield
  } else {
    hlcf.yield
  }
  // CHECK: llvm.intr.lifetime.end %[[PTR0]]
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @stack_allocation_with_alignment
kgen.func @stack_allocation_with_alignment() {
  // CHECK-DAG: %[[C16:.*]] = llvm.mlir.constant(16 : i64) : i64
  // CHECK-DAG: %[[PTR0:.*]] = llvm.alloca %[[C16]] x f32 {alignment = 8 : i64}
  // CHECK: llvm.intr.lifetime.start %[[PTR0]]
  %0 = pop.stack_allocation 16 x !kgen.simd<1, f32> align 8
  // CHECK-NEXT: llvm.intr.lifetime.end %[[PTR0]]
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @stack_allocation_with_addressspace
kgen.func @stack_allocation_with_addressspace() {
  // CHECK-DAG: %[[C16:.*]] = llvm.mlir.constant(16 : i64) : i64
  // CHECK-DAG: %[[PTR0:.*]] = llvm.alloca %[[C16]] x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr<5>
  // CHECK: llvm.intr.lifetime.start %[[PTR0]]
  %0 = pop.stack_allocation 16 x !kgen.simd<1, si32> address_space 5
  // CHECK-NEXT: llvm.intr.lifetime.end %[[PTR0]]
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @stack_allocation_with_align_and_addressspace
kgen.func @stack_allocation_with_align_and_addressspace() {
  // CHECK-DAG: %[[C16:.*]] = llvm.mlir.constant(16 : i64) : i64
  // CHECK-DAG: %[[PTR0:.*]] = llvm.alloca %[[C16]] x f32 {alignment = 8 : i64} : (i64) -> !llvm.ptr<3>
  // CHECK: llvm.intr.lifetime.start %[[PTR0]]
  %0 = pop.stack_allocation 16 x !kgen.simd<1, f32> address_space 3 align 8
  // CHECK-NEXT: llvm.intr.lifetime.end %[[PTR0]]
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @stack_allocation_insertion
kgen.func @stack_allocation_insertion(%v: !kgen.simd<1, si32>) {
  // CHECK: hlcf.loop
  hlcf.loop {
    // CHECK: llvm.alloca
    // CHECK: llvm.intr.lifetime.start
    %2 = pop.stack_allocation 1 x !kgen.simd<1, si32>
    // CHECK: llvm.intr.lifetime.end
    // CHECK: hlcf.break
    hlcf.break
  }
  kgen.return
}

}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="p:64:64", simd_bit_width=128>} {
  // CHECK-LABEL @allocate_64_bit
  kgen.func @allocate_64_bit() {
    // CHECK: lifetime.start {{.*}}
    %0 = pop.stack_allocation 1 x index
    kgen.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="p:32:32", simd_bit_width=128>} {
  // CHECK-LABEL @allocate_32_bit
  kgen.func @allocate_32_bit() {
    // CHECK: lifetime.start {{.*}}
    %0 = pop.stack_allocation 1 x index
    kgen.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="A5", simd_bit_width=128>} {
  // CHECK-LABEL @allocate_with_default_addrspace
  kgen.func @allocate_with_default_addrspace(%arg0: i64) {
    // CHECK: %[[ALLOC:.*]] = llvm.alloca {{.*}} -> !llvm.ptr<5>
    // CHECK: lifetime.start %[[ALLOC]]
    // CHECK: %[[ALLOC_CAST:.*]] = llvm.addrspacecast %[[ALLOC]] : !llvm.ptr<5> to !llvm.ptr
    // CHECK: llvm.store %arg0, %[[ALLOC_CAST]]
    // CHECK: lifetime.end %[[ALLOC]]
    %0 = pop.stack_allocation 1 x i64
    pop.store %arg0, %0 : !kgen.pointer<i64>
    kgen.return
  }
}

// -----

// COM: Ensure the single store load optimization is only applied if the store dominates the load.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  kgen.func @TEST(%arg0: i32) -> i32 {
    %0 = pop.stack_allocation 1 x struct<(pointer<none>, index)>
    lit.try {
      %1 = "somehow.produce_something"() : () -> !kgen.struct<(pointer<none>, index)>
      pop.store %1, %0 : !kgen.pointer<struct<(pointer<none>, index)>>
      lit.try.yield
    } except {
      // CHECK: llvm.load
      %1 = pop.load %0 : !kgen.pointer<struct<(pointer<none>, index)>>
      %2 = builtin.unrealized_conversion_cast %1 : !kgen.struct<(pointer<none>, index)> to !llvm.struct<(ptr, i64)>
      kgen.return %arg0 : i32
    } else {
      lit.try.yield
    }
    kgen.return %arg0 : i32
  }
}
