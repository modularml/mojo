// COM: Since errors involving incorrect locations cannot be handled by
// COM: -verify-diagnostics, we check manually.
// RUN: not support-dialect-opt -split-input-file %s 2>&1 | FileCheck %s

#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<(!debuginfo.unresolved<i32>) -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "foo"> : !debuginfo.unresolved<f32>
#loc = loc("foo.mlir":7:8)

func.func @foo(%arg: i32) {
  // CHECK: foo.mlir:7:8: error: 'debuginfo.value' op conversion expression leaf type 'f32' does not match actual IR Value type 'i32'
  debuginfo.value #local_variable #debuginfo.expr.irvalue: f32 = %arg : i32 loc(fused<#subprogram>[#loc])
  return
}
