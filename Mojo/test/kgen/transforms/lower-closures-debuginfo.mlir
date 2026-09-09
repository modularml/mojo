// RUN: kgen-opt -mlir-print-debuginfo -lower-closures %s | FileCheck %s

#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#subprogram1 = #debuginfo.subprogram<sourceName = <"SomeClosure">>  : !debuginfo.subroutine<() -> (!pop.array<0, i1>): DW_CC_normal>
#subprogram2 = #debuginfo.subprogram<sourceName = <"OtherClosure">>  : !debuginfo.subroutine<() -> (!pop.array<0, i1>): DW_CC_normal>

#loc1 = loc("foo.mlir":44:1)
#loc2 = loc("foo.mlir":46:8)
#loc3 = loc("bar.mlir":327:17)
#loc4 = loc("bar.mlir":415:15)
#loc5 = loc(fused<#subprogram>[#loc1])
#loc6 = loc(fused<#subprogram>[#loc2])
#loc7 = loc(fused<#subprogram1>[#loc3])
#loc8 = loc(fused<#subprogram2>[#loc4])

#loc9 = loc(fused<#subprogram1>[fused<#debuginfo.call_loc<#loc6>>[#loc3]])
#loc10 = loc(fused<#subprogram2>[fused<#debuginfo.call_loc<#loc6>>[#loc4]])

// CHECK-LABEL: kgen.func @foo_async_closure_0()
// CHECK-NEXT:    %array = kgen.param.constant: array<0, i1> = <[]> loc(#[[FOO_ASYNC_CL_CONST_LOC:.*]])
// CHECK:         kgen.return %array {{.*}} loc(#[[FOO_ASYNC_CL_LOC:.*]])
// CHECK-NEXT:  } loc(#[[FOO_ASYNC_CL_LOC]])

// CHECK-LABEL: kgen.func @foo_closure_1()
// CHECK-NEXT:    %array = kgen.param.constant: array<0, i1> = <[]> loc(#[[FOO_CL_CONST_LOC:.*]])
// CHECK-NEXT:    kgen.param.constant: array<2, i1> = <[1, 1]> loc(#[[FOO_CL_LOC:.*]])
// CHECK-NEXT:    kgen.return %array : !pop.array<0, i1> loc(#[[FOO_CL_LOC]])
// CHECK-NEXT:  } loc(#[[FOO_CL_LOC]])

// CHECK-LABEL: kgen.func @foo()
kgen.func @foo() {
  // CHECK-NEXT: kgen.param.constant: array<0, i1> = <[]> loc(#[[LOC_CALLSITE:.*]])
  %array = kgen.param.constant: array<0, i1> = <[]> loc(#loc6)

  // CHECK-NEXT: co.invoke[{{.*}}: @foo_async_closure_0]() loc(#[[LOC_CALLSITE]])
  %0 = co.execute : !pop.array<0, i1> {
    %array_1 = kgen.param.constant: array<1, i1> = <[1]> loc(#loc7)
    kgen.return %array : !pop.array<0, i1> loc(#loc7)
  } loc(#loc9)

  // CHECK-NEXT: kgen.create_closure[() capturing -> !pop.array<0, i1>: @foo_closure_1]()  loc(#[[LOC_CALLSITE]])
  %1 = kgen.stage_closure = () capturing -> !pop.array<0, i1> {
    %array_1 = kgen.param.constant: array<2, i1> = <[1, 1]> loc(#loc8)
    kgen.return %array : !pop.array<0, i1> loc(#loc8)
  } loc(#loc10)

  // CHECK-NEXT: kgen.return
  kgen.return loc(#loc5)
} loc(#loc5)

// CHECK-DAG: #[[SP_ASYNC_CL:.*]] = #debuginfo.subprogram<sourceName = <"async_closure.0" from <"SomeClosure">>, linkageName = "foo_async_closure_0"
// CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<sourceName = <"foo">
// CHECK-DAG: #[[SP_CL:.*]] = #debuginfo.subprogram<sourceName = <"closure.1" from <"OtherClosure">>, linkageName = "foo_closure_1"

// CHECK-DAG: #[[SOME_CL_LOC:.*]] = loc("bar.mlir":327:17)
// CHECK-DAG: #[[FOO_ASYNC_CL_LOC]] = loc(fused<#[[SP_ASYNC_CL]]>[#[[SOME_CL_LOC]]])
// CHECK-DAG: #[[FOO_ASYNC_CL_CONST_LOC]] = loc(callsite(#[[LOC_CALLSITE]] at #[[FOO_ASYNC_CL_LOC]]))

// CHECK-DAG: #[[OTHER_CL_LOC:.*]] = loc("bar.mlir":415:15)
// CHECK-DAG: #[[FOO_CL_LOC]] = loc(fused<#[[SP_CL]]>[#[[OTHER_CL_LOC]]])
// CHECK-DAG: #[[FOO_CL_CONST_LOC]] = loc(callsite(#[[LOC_CALLSITE]] at #[[FOO_CL_LOC]]))
