// RUN: kgen-opt %s -split-input-file -mlir-print-debuginfo -verify-parameters=simplify=true -verify-parameters | FileCheck %s

// CHECK-LABEL: kgen.generator @foo
kgen.generator @foo() {
  kgen.param.declare N = <2> loc(#locFoo)
  // CHECK: kgen.param.declare.region SomeClosure
  kgen.param.declare.region SomeClosure = () capturing {
    // CHECK-NEXT: kgen.param.constant: array<1, i1> = <[1]> loc(#[[LOC_CL:.*]])
    %array = kgen.param.constant: array<1, i1> = <[1]> loc(#locClosure)
    // CHECK-NEXT: kgen.return loc(#[[LOC_CL]])
    kgen.return loc(#locClosure)
  // CHECK-NEXT: } {isolated} loc(#[[LOC_CL]])
  } loc(#locClosure)

  // CHECK: kgen.param.declare.region OtherClosure
  kgen.param.declare.region OtherClosure = <K>(%arg1: !pop.array<K, index>) {
    // CHECK-NEXT: kgen.return loc(#[[LOC_OTHER:.*]])
    kgen.return loc(#locOther)
  // CHECK-NEXT: } {isolated} loc(#[[LOC_OTHER]])
  } loc(#locOther)

  // CHECK-NEXT: kgen.return loc(#[[LOC_FOO:.*]])
  kgen.return loc(#locFoo)
// CHECK-NEXT: } loc(#[[LOC_FOO]])
} loc(#locFoo)

#file = #debuginfo.file<"foo.mojo" in "/">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #file, producer = "Mojo", isOptimized = true, emissionKind = Full>

// CHECK-DAG: ![[CL_SP_TYPE:.*]] = !debuginfo.subroutine<(!kgen.pointer<scalar<#kgen.struct.extract<2, 1>>>) -> (): DW_CC_normal>
// CHECK-DAG: ![[OTHER_SP_TYPE:.*]] = !debuginfo.subroutine<(!pop.array<K, index>) -> (): DW_CC_normal>
// CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<sourceName = <"foo">
// CHECK-DAG: #[[CL_SP:.*]] = #debuginfo.subprogram<sourceName = <"SomeClosure">
// CHECK-DAG: #[[OTHER_SP:.*]] = #debuginfo.subprogram<sourceName = <"OtherClosure">
#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#subprogram1 = #debuginfo.subprogram<sourceName = <"SomeClosure">> : !debuginfo.subroutine<(!kgen.pointer<scalar<#kgen.struct.extract<N, 1>>>) -> (): DW_CC_normal>
#subprogram2 = #debuginfo.subprogram<sourceName = <"OtherClosure">> : !debuginfo.subroutine<(!pop.array<K, index>) -> (): DW_CC_normal>

// CHECK-DAG: #[[LOC1:.*]] = loc("foo.mojo":25:1)
// CHECK-DAG: #[[LOC2:.*]] = loc("foo.mojo":183:5)
// CHECK-DAG: #[[LOC3:.*]] = loc("foo.mlir":56:5)
// CHECK-DAG: #[[LOC_FOO]] = loc(fused<#[[SP]]>[#[[LOC1]]])
// CHECK-DAG: #[[LOC_CL]] = loc(fused<#[[CL_SP]]>[#[[LOC2]]])
// CHECK-DAG: #[[LOC_OTHER]] = loc(fused<#[[OTHER_SP]]>[#[[LOC3]]])
#locFoo = loc(fused<#subprogram>["foo.mojo":25:1])
#locClosure = loc(fused<#subprogram1>["foo.mojo":183:5])
#locOther = loc(fused<#subprogram2>["foo.mlir":56:5])

// -----

#subprogram = #debuginfo.subprogram<sourceName = <"test_stencil_avg_pool">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#subprogram1 = #debuginfo.subprogram<sourceName = <"map_fn">> : !debuginfo.subroutine<(!kgen.simd<rank, f32>) -> (): DW_CC_normal>
// CHECK: [[SR_TYPE:!.*]] = !debuginfo.subroutine<(!kgen.simd<2, f32>) -> (): DW_CC_normal>
// CHECK: [[SP:#.*]] = #debuginfo.subprogram<{{.*}}> : [[SR_TYPE]]
// CHECK: [[BREAK_LOC:#.*]] = loc(fused<[[SP]]>
#loc3 = loc(fused<#subprogram>["a":0:0])
#loc4 = loc(fused<#subprogram1>["a":0:0])

// CHECK-LABEL: kgen.generator @func
kgen.generator @func() {
  kgen.param.declare rank = <2> loc(#loc3)
  // CHECK: kgen.param.declare.region region = (%arg0: index loc(fused<[[SP]]>
  kgen.param.declare.region region = (%arg : index loc(#loc4)) {
    hlcf.loop {
      // CHECK: hlcf.break loc([[BREAK_LOC]])
      hlcf.break loc(#loc4)
    } loc(#loc4)
    kgen.return loc(#loc4)
  } loc(#loc4)
  kgen.return loc(#loc3)
} loc(#loc3)
