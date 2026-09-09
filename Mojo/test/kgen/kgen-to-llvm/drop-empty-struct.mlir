// RUN: kgen-opt -lower-kgen-to-llvm %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=64>} {

// CHECK-LABEL: llvm.func internal @none_type_func
//  CHECK-SAME: (%{{.*}}: i64 {llvm.noundef}) -> !llvm.struct<()>
kgen.func @none_type_func(%a : !kgen.none, %b: index) -> !kgen.none {
  // CHECK: %[[RVAL:.+]] = llvm.mlir.undef : !llvm.struct<()>
  // CHECK: llvm.return %[[RVAL]]
  kgen.return %a: !kgen.none
}

// CHECK-LABEL: llvm.func internal @none_type_func_caller
kgen.func @none_type_func_caller(){
  %0 = kgen.param.constant = <2>
  %none = kgen.param.constant: none = <#kgen.none>
  // CHECK: llvm.call @none_type_func
  // CHECK-SAME: (i64) -> !llvm.struct<()>
  %1 = kgen.call @none_type_func(%none, %0) : (!kgen.none, index) -> !kgen.none
  kgen.return
}

// CHECK-LABEL: llvm.func internal @complex_empty_types
//  CHECK-SAME: (%arg0: !llvm.struct<(i64)> {llvm.noundef})
kgen.func @complex_empty_types(
  %a0: !kgen.struct<(none, none)>,
  %a1: !pop.array<1, struct<(struct<()>)>>,
  %a2: !pop.array<0, struct<(index)>>,
  %a3: !kgen.struct<(index)>) {
    kgen.return
}

// CHECK-LABEL: llvm.func internal @has_ptr
// CHECK-SAME: (%arg0: !llvm.ptr {llvm.nonnull, llvm.noundef}, %arg1: !llvm.struct<(i64, i64)> {llvm.noundef})
kgen.func @has_ptr(%arg0: !kgen.none, %arg1: !kgen.pointer<none> imm_mem, %arg3: !kgen.struct<(index,index)>) {
  kgen.return
}

}
