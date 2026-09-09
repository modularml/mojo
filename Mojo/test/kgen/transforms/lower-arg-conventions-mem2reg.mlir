// RUN: kgen-opt -lower-arg-conventions -verify-parameters -mem-2-reg %s | FileCheck %s

// CHECK-LABEL: kgen.func @lower_args_mem_2_reg(%arg0: index owned, %arg1: !kgen.struct<(index, index)>) {
kgen.func @lower_args_mem_2_reg(
  %arg0: !kgen.pointer<index> owned_in_mem,
  %arg1: !kgen.pointer<struct<(index, index)>> imm_mem
) {
  // CHECK-NEXT: kgen.call @lower_args_mem_2_reg(%arg0, %arg1) : (index owned, !kgen.struct<(index, index)>) -> ()
  kgen.call @lower_args_mem_2_reg(%arg0, %arg1) : (
    !kgen.pointer<index> owned_in_mem,
    !kgen.pointer<struct<(index, index)>> imm_mem
  ) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @baz() -> index
kgen.func @baz(%__result__: !kgen.pointer<index> byref_result) -> !kgen.none {
  // CHECK-NEXT: %[[RES:.*]] = kgen.call @baz() : () -> index
  // CHECK-NEXT: kgen.return %[[RES]] : index
  %none = kgen.call @baz(%__result__) : (
    !kgen.pointer<index> byref_result) -> !kgen.none
  kgen.return %none : !kgen.none
}
