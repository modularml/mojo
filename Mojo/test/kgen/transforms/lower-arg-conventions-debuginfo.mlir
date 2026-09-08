// RUN: kgen-opt -lower-arg-conventions -mlir-print-debuginfo %s | FileCheck %s

#sp = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#loc = loc(fused<#sp>["a":0:0])

// CHECK-LABEL: kgen.func @rewrite_me
kgen.func @rewrite_me(%arg1: !kgen.pointer<index> loc("a":0:0) imm_mem, %arg0: !kgen.pointer<index> loc("a":0:0) byref_result) -> !kgen.none {
  // CHECK: stack_allocation {{.*}} loc([[LOCSP:#.*]])
  // CHECK-NEXT: store {{.*}} loc([[LOCSP]])
  // CHECK-NEXT: stack_allocation {{.*}} loc([[LOCSP]])
  // CHECK: load {{.*}} loc([[LOCSP]])
  %none = kgen.param.constant: none = <#kgen.none> loc(#loc)
  kgen.return %none : !kgen.none loc(#loc)
} loc(#loc)

// CHECK: [[LOCSP]] = loc(fused<#subprogram>
