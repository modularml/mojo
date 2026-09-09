// RUN: support-dialect-opt -mlir-print-debuginfo %s | support-dialect-opt -mlir-print-debuginfo | FileCheck %s

#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<(!debuginfo.unresolved<i32>) -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "foo"> : !debuginfo.unresolved<i32>
#trivial_expr = #debuginfo.expr.irvalue : !debuginfo.unresolved<i32>

#local_variable1 = #debuginfo.local_variable<scope = #subprogram, name = "foo"> : !debuginfo.ti.ptr<!debuginfo.unresolved<i32>>
#complex_expr = #debuginfo.expr.refof<#debuginfo.expr.irvalue : !debuginfo.unresolved<i32>> : !debuginfo.ti.ptr<!debuginfo.unresolved<i32>>

#loc1 = loc("foo.mlir":7:8)
#loc2 = loc("bar.mlir":5:6)
#fusedLoc = loc(fused<#subprogram>[#loc2])
#loc3 = loc(callsite(#fusedLoc at #loc1))

// CHECK-LABEL: func @foo
// CHECK-SAME: (%[[ARG:.*]]: i32
func.func @foo(%arg: i32) {
  // A trivial conversion expr should be omitted.
  // CHECK: debuginfo.value #[[VAR:.*]] = %[[ARG]] : i32
  debuginfo.value #local_variable #trivial_expr = %arg : i32 loc(callsite(#loc3 at #loc1))
  debuginfo.value #local_variable1 #complex_expr = %arg : i32 loc(callsite(#loc3 at #loc1))
  return
}
