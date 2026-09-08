// RUN: kgen-opt -automatic-inline=update-debug-info=deferred -mlir-print-debuginfo -split-input-file %s | FileCheck %s
// RUN: kgen-opt -automatic-inline=update-debug-info=immediate -mlir-print-debuginfo -split-input-file %s | FileCheck %s

// CHECK-DAG: #[[SP0:.*]] = #debuginfo.subprogram<sourceName = <"inline_me0">
#callee0Sp = #debuginfo.subprogram<sourceName = <"inline_me0">> : !debuginfo.subroutine<(!debuginfo.unresolved<index>) -> (!debuginfo.unresolved<index>): DW_CC_normal>

// CHECK-DAG: #[[SP1:.*]] = #debuginfo.subprogram<sourceName = <"inline_me1">
#callee1Sp = #debuginfo.subprogram<sourceName = <"inline_me1">> : !debuginfo.subroutine<(!debuginfo.unresolved<index>) -> (!debuginfo.unresolved<index>): DW_CC_normal>

#callerSp = #debuginfo.subprogram<sourceName = <"caller">> : !debuginfo.subroutine<(!debuginfo.unresolved<index>) -> (!debuginfo.unresolved<index>): DW_CC_normal>

#asyncCallerSp = #debuginfo.subprogram<sourceName = <"call_async">> : !debuginfo.subroutine<() -> (!debuginfo.unresolved<!co.routine>): DW_CC_normal>
// CHECK-DAG: #[[INLINED_VAR_FOO:.*]] = #debuginfo.local_variable<{{.*}}name = "foo"
#local_variable0 = #debuginfo.local_variable<scope = #callee0Sp, name = "foo"> : !debuginfo.unresolved<index>
#local_variable1 = #debuginfo.local_variable<scope = #callee1Sp, name = "bar"> : !debuginfo.unresolved<index>

#locAsyncCaller = loc(fused<#asyncCallerSp>["bar.mlir":18:7])

// COM: Test location handling for two-level of fully-inlined exported function

#loc0 = loc("foo.mlir":13:1)
#loc1 = loc("foo.mlir":13:2)
#locArg0 = loc("foo.mlir":13:12)
#locArg1 = loc("foo.mlir":13:13)
#locCallee0 = loc(fused<#callee0Sp>[#loc0])
#locCallee1 = loc(fused<#callee1Sp>[#loc1])

#locCallsite = loc("bar.mlir":27:8)
#locCaller = loc(fused<#callerSp>[#locCallsite])

// Both of inline_me0 and inline_me1 will be fully-inilined,
// However they will not be erased since both are `exported`.
// We need to update location for both of them.
kgen.func export @inline_me0(%arg0: index) -> index {
  debuginfo.value #local_variable0 = %arg0 : index loc(fused<#callee0Sp>[#locArg0])
  kgen.return %arg0: index loc(#locCallee0)
} loc(#locCallee0)

kgen.func export @inline_me1(%arg0: index) -> index {
  debuginfo.value #local_variable1 = %arg0 : index loc(fused<#callee1Sp>[#locArg1])
  %1 = kgen.call @inline_me0(%arg0) : (index) -> index loc(#locCallee1)
  kgen.return %1: index loc(#locCallee1)
} loc(#locCallee1)

// CHECK-LABEL: kgen.func @call_inline_me
kgen.func @call_inline_me() -> index {
  %0 = index.constant 3 loc(#locCaller)
  // CHECK: %idx3 = index.constant 3
  // CHECK-NEXT: debuginfo.value #local_variable1 = %idx3 : index loc(#[[LOC_VALUE_INLINED:.*]])
  %1 = kgen.call @inline_me1(%0) : (index) -> index loc(#locCaller)
  kgen.return %1 : index loc(#locCaller)
} loc(#locCaller)

// Test location for inlining async call with a nodebug function that inlines
// a function that has debugInfo. :-(

kgen.func @nodebug_inline_me(%arg0: index) -> index {
  %0 = index.add %arg0, %arg0
  %1 = kgen.call @inline_me0(%0) : (index) -> index
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.func @call_async
kgen.func @call_async() -> !co.routine {
  // CHECK-NEXT: [[IDX2:%.*]] = index.constant 2 loc(#[[LOC_SCOPED_CALLER:.*]])
  %idx2 = index.constant 2 loc(#locAsyncCaller)
  // CHECK-NEXT: [[V0:%.*]]co.execute : index {
  // CHECK-NEXT:   [[V1:%.*]] = index.add [[IDX2]], [[IDX2]] loc(#[[LOC_ADD:.*]])
  // CHECK-NEXT:   debuginfo.value #[[INLINED_VAR_FOO]] = [[V1]]
  // CHECK-NEXT:   kgen.return [[V1]] : index loc(#[[LOC_INLINED_RETURN:.*]])
  // CHECK-NEXT: } loc(#[[LOC_ASYNC_EXECUTE:.*]])
  %0 = co.invoke[(index) async -> index: @nodebug_inline_me](%idx2) loc(#locAsyncCaller)
  // CHECK-NEXT: kgen.return
  kgen.return %0: !co.routine loc(#locAsyncCaller)
// CHECK-NEXT: } loc(#[[LOC_SCOPED_CALLER]])
} loc(#locAsyncCaller)


// CHECK-DAG: #[[SP_ASYNC:.*]] = #debuginfo.subprogram<sourceName = <"call_async">
// CHECK-DAG: #[[LOC_ASYNC_CALLER:.*]] = loc("bar.mlir":18:7)
// CHECK-DAG: #[[LOC_SCOPED_CALLER]] = loc(fused<#[[SP_ASYNC]]>[#[[LOC_ASYNC_CALLER]]])
// CHECK-DAG: #[[LOC_ADD]] = loc("{{.*}}":{{[0-9]+}}:{{[0-9]+}})
// CHECK-DAG: #[[LOC_INLINED_RETURN]] = loc("{{.*}}":{{[0-9]+}}:{{[0-9]+}})
// CHECK-DAG: #[[CALL_LOC:.*]] = #debuginfo.call_loc<#[[LOC_SCOPED_CALLER]]>
// CHECK-DAG: #[[LOC_ASYNC_EXECUTE]] = loc(fused<#[[CALL_LOC]]>[#[[LOC_ASYNC_FILE_LOC:.*]]])

// -----

#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#loc = loc(fused<#subprogram>["foo.mlir":0:0])

kgen.func @no_debuginfo() -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @has_debuginfo
kgen.func @has_debuginfo() {
  // CHECK: index.constant 0 loc(#[[LOC:.*]])
  kgen.call @no_debuginfo() : () -> index loc(#loc)
  kgen.return loc(#loc)
} loc(#loc)

// CHECK-DAG: #[[LOC:.+]] = loc("foo.mlir":0:0)
// CHECK-DAG: #[[SP:.+]] = #debuginfo.subprogram<sourceName = <"foo">>
// CHECK-DAG: #[[LOC_CALLER_SP:.+]] = loc(fused<#[[SP]]>[#[[LOC]]])
