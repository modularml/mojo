// RUN: kgen-opt -inline-param=optimization-level=0 -verify-parameters -split-input-file %s -mlir-print-debuginfo | FileCheck %s

#subprogram = #debuginfo.subprogram<sourceName = <"caller">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#locCalleeInner = loc(fused<#subprogram>["foo.mlir":0:0])
#locCallee = loc(fused<#subprogram>["foo.mlir":1:0])
#locCaller = loc(fused<#subprogram>["foo.mlir":2:0])

kgen.generator @callee_inner() -> index always_inline {
  %idx0 = index.constant 0 loc(#locCalleeInner)
  kgen.return %idx0 : index loc(#locCalleeInner)
} loc(#locCalleeInner)

kgen.generator @callee_nodebug() -> index always_inline_no_debug {
  %idx0 = index.constant 1
  kgen.call @callee_inner() : () -> index
  kgen.return %idx0 : index
}

kgen.generator @callee_debug() -> index always_inline {
  %idx0 = index.constant 2 loc(#locCallee)
  kgen.call @callee_inner() : () -> index loc(#locCallee)
  kgen.return %idx0 : index loc(#locCallee)
} loc(#locCallee)

// CHECK-LABEL: kgen.generator @caller
kgen.generator @caller() {
  // CHECK-NEXT: index.constant 1
  // CHECK-NEXT: debuginfo.line_table_loc loc(#[[LOC_CALLEE_INNER_AT_CALLEE:.+]])
  // CHECK-NEXT: index.constant 0
  kgen.call @callee_nodebug() : () -> index loc(#locCaller)

  // CHECK-NEXT: debuginfo.line_table_loc loc(#[[LOC_CALLER:.+]])
  // CHECK-NEXT: index.constant 2
  // CHECK-NEXT: debuginfo.line_table_loc loc(#[[LOC_CALLEE_AT_CALLER:.+]])
  // CHECK-NEXT: index.constant 0
  kgen.call @callee_debug() : () -> index loc(#locCaller)
  kgen.return loc(#locCaller)
} loc(#locCaller)

// CHECK-DAG: #[[LOC_CALLER_RAW:.+]] = loc("foo.mlir":2:0)
// CHECK-DAG: #[[LOC_CALLER]] = loc(fused<{{.*}}>[#[[LOC_CALLER_RAW]]])

// CHECK-DAG: #[[LOC_CALLEE_RAW:.+]] = loc("foo.mlir":1:0)
// CHECK-DAG: #[[LOC_CALLEE:.+]] = loc(fused<{{.*}}>[#[[LOC_CALLEE_RAW]]])
// CHECK-DAG: #[[LOC_CALLEE_AT_CALLER]] = loc(callsite(#[[LOC_CALLEE]] at #[[LOC_CALLER]]))
