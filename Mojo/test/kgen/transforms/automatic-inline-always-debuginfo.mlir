// RUN: kgen-opt -automatic-inline=update-debug-info=deferred -mlir-print-debuginfo -split-input-file %s | FileCheck %s
// RUN: kgen-opt -automatic-inline=update-debug-info=immediate -mlir-print-debuginfo -split-input-file %s | FileCheck %s

// COM: The attributes may be printed before or after the functions under test,
// COM: so we try to keep the attribute close to its corresponding CHECK
// COM: statement, if possible. Moreover, to avoid some issues with FileCheck
// COM: getting confused with CHECK-DAG statements (and to reduce duplicate test
// COM: code), we also don't use -split-file.

// CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<sourceName = <"inline_me">
#calleeSp = #debuginfo.subprogram<sourceName = <"inline_me">> : !debuginfo.subroutine<(!debuginfo.unresolved<index>) -> (!debuginfo.unresolved<index>): DW_CC_normal>
#callerSp = #debuginfo.subprogram<sourceName = <"caller">> : !debuginfo.subroutine<(!debuginfo.unresolved<index>) -> (!debuginfo.unresolved<index>): DW_CC_normal>
#asyncCallerSp = #debuginfo.subprogram<sourceName = <"call_async">> : !debuginfo.subroutine<() -> (!debuginfo.unresolved<!co.routine>): DW_CC_normal>
#local_variable = #debuginfo.local_variable<scope = #calleeSp, name = "foo"> : !debuginfo.unresolved<index>

#locAsyncCaller = loc(fused<#asyncCallerSp>["bar.mlir":18:7])

#locCallsite = loc("bar.mlir":27:8)

#locCaller = loc(fused<#callerSp>[#locCallsite])

// Test nodebug behavior for debuginfo.value ops.

kgen.func @nodebug_inline_me(%arg0: index) -> index always_inline_no_debug {
  %0 = index.add %arg0, %arg0 loc(#locCallsite)
  kgen.return %0: index loc(#locCallsite)
} loc(#locCallsite)

// CHECK-LABEL: kgen.func @call_nodebug_inline_me
kgen.func @call_nodebug_inline_me() -> index {
  %0 = index.constant 3
  // CHECK: index.add %idx3, %idx3
  %1 = kgen.call @nodebug_inline_me(%0) : (index) -> index
  kgen.return %1 : index
}

// Test location handling of inlining.

#loc = loc("foo.mlir":13:1)
#locArg = loc("foo.mlir":13:12)
#locCallee = loc(fused<#calleeSp>[#loc])

kgen.func @inline_me(%arg0: index) -> index always_inline {
  debuginfo.value #local_variable = %arg0 : index loc(fused<#calleeSp>[#locArg])
  kgen.return %arg0: index loc(#locCallee)
} loc(#locCallee)

// CHECK-LABEL: kgen.func @call_inline_me
kgen.func @call_inline_me() -> index {
  %0 = index.constant 3 loc(#locCaller)
  // CHECK: %idx3 = index.constant 3
  // CHECK-NEXT: debuginfo.value #local_variable = %idx3 : index loc(#[[LOC_VALUE_INLINED:.*]])
  %1 = kgen.call @inline_me(%0) : (index) -> index loc(#locCaller)
  kgen.return %1 : index loc(#locCaller)
} loc(#locCaller)

// COM: Test location handling of async closure staging.

// CHECK-LABEL: kgen.func @call_async
kgen.func @call_async() -> !co.routine {
  // CHECK-NEXT: %idx2 = index.constant 2 loc(#[[LOC_SCOPED_CALLER:.*]])
  %idx2 = index.constant 2 loc(#locAsyncCaller)
  // CHECK-NEXT: co.execute : index
  // CHECK-NEXT:   debuginfo.value #local_variable = %idx2 : index loc(#[[LOC_VALUE:.*]])
  // CHECK-NEXT:   kgen.return %idx2 : index loc(#[[LOC_ASYNC_EXECUTE_RET:.*]])
  // CHECK-NEXT: } loc(#[[LOC_ASYNC_EXECUTE1:.*]])
  %coroHdl = co.invoke[(index) async -> index: @inline_me](%idx2) loc(#locAsyncCaller)
  // CHECK-NEXT: kgen.return
  kgen.return %coroHdl : !co.routine loc(#locAsyncCaller)
// CHECK-NEXT: } loc(#[[LOC_SCOPED_CALLER]])
} loc(#locAsyncCaller)

// COM: Test location handling of inlined closure staging.

kgen.func @async_wrapper() -> !co.routine always_inline {
  %idx3 = index.constant 3 loc(#locAsyncCaller)
  %0 = co.invoke[(index) async -> index: @inline_me](%idx3) loc(#locAsyncCaller)
  kgen.return %0 : !co.routine loc(#locAsyncCaller)
} loc(#locAsyncCaller)

// CHECK-LABEL: kgen.func @call_async_indirect
kgen.func @call_async_indirect() -> !co.routine {
  // CHECK-NEXT: %idx3 = index.constant 3 loc(#[[LOC_ASYNC_CALLER:.*]])
  // CHECK-NEXT: co.execute : index
  // CHECK-NEXT:   debuginfo.value #local_variable = %idx3 : index loc(#[[LOC_VALUE]])
  // CHECK-NEXT:   kgen.return %idx3 : index loc(#[[LOC_ASYNC_EXECUTE_RET2:.*]])
  // CHECK-NEXT: } loc(#[[LOC_ASYNC_EXECUTE2:.*]])
  %1 = kgen.call @async_wrapper() : () -> !co.routine loc(#locCaller)
  kgen.return %1 : !co.routine loc(#locCaller)
} loc(#locCaller)

// CHECK-LABEL: @call_async_no_debuginfo
kgen.func @call_async_no_debuginfo() -> !co.routine {
  // CHECK-NEXT: index.constant 2
  // CHECK-NEXT: co.execute
  // CHECK-NOT: debuginfo.value
  %idx2 = index.constant 2 loc(#locAsyncCaller)
  %0 = co.invoke[(index) async -> index: @nodebug_inline_me](%idx2) loc(#locAsyncCaller)
  kgen.return %0 : !co.routine
}

// Test nodebug behavior for func with multiple exists.

kgen.func @nodebug_inline_me_multiple_exits(%arg0: index) -> index always_inline_no_debug {
  %idx1 = index.constant 1 loc(#locCallsite)
  %0 = index.add %arg0, %arg0 loc(#locCallsite)
  %1 = index.cmp sgt (%arg0, %idx1) loc(#locCallsite)
  %c1 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool> loc(#locCallsite)
  hlcf.if %c1 {
    kgen.return %0: index loc(#locCallsite)
  } else  {
    hlcf.yield loc(#locCallsite)
  } loc(#locCallsite)
  kgen.return %arg0: index loc(#locCallsite)
} loc(#locCallsite)

// CHECK-LABEL: kgen.func @call_nodebug_inline_me_multiple_exits
kgen.func @call_nodebug_inline_me_multiple_exits() -> index {
  %0 = index.constant 3
  // hlcf.loop should not be folded away since inlined function has multiple exits.
  // CHECK-DAG: %[[V0:.*]] = hlcf.loop "inlined_cf_scope" () -> index {
  %1 = kgen.call @nodebug_inline_me_multiple_exits(%0) : (index) -> index
  kgen.return %1 : index
}

// CHECK-DAG: #[[LOC_ASYNC_CALLER]] = loc("bar.mlir":18:7)
// CHECK-DAG: #[[LOC_SCOPED_CALLER]] = loc(fused<#[[SP_ASYNC:.*]]>[#[[LOC_ASYNC_CALLER]]])
// CHECK-DAG: #[[LOC_CALLSITE_FILE:.*]] = loc("bar.mlir":27:8)
// CHECK-DAG: #[[SP_CALLER:.*]] = #debuginfo.subprogram<sourceName = <"caller">
// CHECK-DAG: #[[LOC_CALLSITE:.*]] = loc(fused<#[[SP_CALLER]]>[#[[LOC_CALLSITE_FILE]]

// CHECK-DAG: #[[SP_ASYNC]] = #debuginfo.subprogram<sourceName = <"call_async">
// CHECK-DAG: #[[LOC:loc[0-9]+]] = loc("foo.mlir":13:1)
// CHECK-DAG: #[[LOC_ARG:loc[0-9]+]] = loc("foo.mlir":13:12)

// CHECK-DAG: #[[LOC_VALUE_INLINED]] = loc(callsite(#[[LOC_VALUE:loc[0-9]+]] at #[[LOC_CALLSITE]]))
// CHECK-DAG: #[[LOC_VALUE]] = loc(fused<#[[SP]]>[#[[LOC_ARG]]])
// CHECK-DAG: #[[LOC_ASYNC_EXEC_CALL_LOC1:.*]] = #debuginfo.call_loc<#[[LOC_SCOPED_CALLER]]>
// CHECK-DAG: #[[LOC_ASYNC_EXEC_CALL_LOC2_CALLSITE:.*]] = loc(callsite(#[[LOC_SCOPED_CALLER]] at #[[LOC_CALLSITE]]))
// CHECK-DAG: #[[LOC_ASYNC_EXEC_CALL_LOC2:.*]] = #debuginfo.call_loc<#[[LOC_ASYNC_EXEC_CALL_LOC2_CALLSITE]]>
// CHECK-DAG: #[[LOC_ASYNC_EXEC_CALL_ENCODED1:.*]] = loc(fused<#[[LOC_ASYNC_EXEC_CALL_LOC1]]>[#[[LOC]]])
// CHECK-DAG: #[[LOC_ASYNC_EXEC_CALL_ENCODED2:.*]] = loc(fused<#[[LOC_ASYNC_EXEC_CALL_LOC2]]>[#[[LOC]]])
// CHECK-DAG: #[[LOC_ASYNC_EXECUTE1]] = loc(fused<#[[SP]]>[#[[LOC_ASYNC_EXEC_CALL_ENCODED1]]])
// CHECK-DAG: #[[LOC_ASYNC_EXECUTE2]] = loc(fused<#[[SP]]>[#[[LOC_ASYNC_EXEC_CALL_ENCODED2]]])

// -----

#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#loc = loc(fused<#subprogram>["foo.mlir":0:0])

kgen.func @no_debuginfo() -> index always_inline {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @has_debuginfo
kgen.func @has_debuginfo() {
  // CHECK: index.constant 0 loc([[LOC:#.*]])
  kgen.call @no_debuginfo() : () -> index loc(#loc)
  kgen.return loc(#loc)
} loc(#loc)

// CHECK: [[LOC]] = loc("{{.*}}":
