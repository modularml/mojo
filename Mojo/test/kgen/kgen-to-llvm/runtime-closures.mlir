// RUN: kgen-opt %s --lower-runtime-closures -allow-unregistered-dialect -split-input-file | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @take_closure_no_args
  llvm.func @take_closure_no_args(%arg0: !llvm.struct<(ptr, ptr)>) {
    // CHECK: %[[V0:.+]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr, ptr)>
    // CHECK: %[[V1:.+]] = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr, ptr)>
    // CHECK: llvm.call %[[V0]](%[[V1]])
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(ptr, ptr)> to !kgen.generator<() capturing -> index>
    %1 = kgen.call_indirect %0() : () capturing -> index
    llvm.return
  }
  llvm.func @h(%arg0: i64) -> i64 {
    llvm.return %arg0 : i64
  }
  // CHECK-LABEL: @main_closure_arg
  llvm.func internal @main_closure_arg() {
    // CHECK: [[ARG:%.*]] = builtin.unrealized_conversion_cast %idx98 : index to i64
    // CHECK: [[UNDEF:%.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr)>
    // CHECK: [[ADDR:%.*]] = llvm.mlir.addressof @closure_wrapper_fn_0 : !llvm.ptr
    // CHECK: [[S0:%.*]] = llvm.insertvalue [[ADDR]], [[UNDEF]][0] : !llvm.struct<(ptr, ptr)>
    // CHECK: [[C1:%.*]] = llvm.mlir.constant(1 : i8) : i8
    // CHECK: [[STATE:%.*]] = llvm.alloca [[C1]] x !llvm.struct<(i64)> : (i8) -> !llvm.ptr
    // CHECK: llvm.intr.lifetime.start [[STATE]] : !llvm.ptr
    // CHECK: [[ARGPTR:%.*]] = llvm.getelementptr [[STATE]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64)>
    // CHECK: llvm.store [[ARG]], [[ARGPTR]] : i64, !llvm.ptr
    // CHECK: [[CLOSURE:%.*]] = llvm.insertvalue [[STATE]], [[S0]][1] : !llvm.struct<(ptr, ptr)>
    // CHECK-NEXT: unrealized_conversion_cast [[CLOSURE]]
    %idx98 = index.constant 98
    %0 = kgen.create_closure[(index) -> index: @h](%idx98)
    "use.closure"(%0) : (!kgen.generator<() capturing -> index>) -> ()
    llvm.return
  }
  // CHECK-LABEL: llvm.func internal @closure_wrapper_fn_0(%arg0: !llvm.ptr) -> i64
  // CHECK: %0 = llvm.getelementptr %arg0[0, 0]  : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64)>
  // CHECK: %1 = llvm.load %0 : !llvm.ptr -> i64
  // CHECK: %2 = llvm.call @h(%1)
  // CHECK: llvm.return %2 : i64
}

// -----

// COM: Check TailCallKind is passed to LLVM.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  llvm.func @h(%arg0: i64) -> i64 {
    llvm.return %arg0 : i64
  }
  // CHECK-LABEL: llvm.func @tailcall
  llvm.func @tailcall(%arg0: i64) -> i64 {
    %0 = kgen.create_closure[(i64) -> i64: @h](%arg0)
    // CHECK: llvm.call musttail
    %1 = kgen.call_indirect musttail %0() : !kgen.generator<() capturing -> i64>
    llvm.return %1 : i64
  }
}
