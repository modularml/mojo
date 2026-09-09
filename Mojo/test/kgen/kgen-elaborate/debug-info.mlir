// RUN: kgen-opt %s -verify-parameters -elaborate-generators="elaborate-debuginfo=true use-parametric-interpret=false" -split-input-file -mlir-print-debuginfo | FileCheck %s
// RUN: kgen-opt %s -elaborate-generators="elaborate-debuginfo=true use-parametric-interpret=true" -split-input-file -mlir-print-debuginfo | FileCheck %s

// Check that debug info gets resolved during elaboration.

#file = #debuginfo.file<"test.mlir" in "">
!unresolved = !debuginfo.unresolved<!kgen.param<ty>>

// CHECK-DAG: #takeFnContextualType_name = #debuginfo.source_name<"takeFnContextualType"<"index", "@sillyFn">>
// CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<sourceName = #takeFnContextualType_name, linkageName = "takeFnContextualType,ty=index,fn=sillyFn",
#callerSp = #debuginfo.subprogram<file = #file, sourceName = <"takeFnContextualType">> : !debuginfo.subroutine<() -> (!unresolved): DW_CC_normal>

// CHECK-DAG: #[[VAR:.*]] = #debuginfo.local_variable<scope = #[[SP]], name = "0"
#local_variable = #debuginfo.local_variable<scope = #callerSp, name = "0"> : !unresolved

// CHECK-DAG: #[[LOC_TRY_FILE:.*]] = loc("silly.mlir":17:3)
// CHECK-DAG: #[[LOC_TRY:.*]] = loc(fused<#[[SP]]>[#[[LOC_TRY_FILE]]])
#locTry = loc("silly.mlir":17:3)

// CHECK-LABEL: kgen.func @"takeFnContextualType,ty=index,fn=sillyFn"() -> index
kgen.generator @takeFnContextualType<ty: type, fn: () -> !kgen.param<ty>>() -> !kgen.param<ty> {
  // CHECK: %[[RES:.*]] = kgen.call @sillyFn() : () -> index loc(#[[CALL_LOC:.*]])
  %0 = kgen.call_param[() -> !kgen.param<ty>: fn]() loc(#loc11)
  // CHECK: debuginfo.value #[[VAR]] = %[[RES]] : index loc(#[[CALL_LOC]])
  debuginfo.value #local_variable = %0 : !kgen.param<ty> loc(#loc11)
  // CHECK: kgen.param.constant = <17> loc(#[[FW_LOC:.*]])
  %1 = kgen.param.constant = <17> loc(#locFwParam)
  kgen.param.declare a = <1> loc(#loc11)

  // CHECK: lit.try {
  // CHECK: } except (%arg0: index loc(fused<#[[SP]]>[#[[LOC_TRY_FILE]]])) {
  // CHECK:   lit.try.yield loc(#[[LOC_TRY]])
  lit.try {
    lit.try.yield loc(#loc11)
  } except (%arg0: index loc(fused<#callerSp>[#locTry])) {
    lit.try.yield loc(fused<#callerSp>[#locTry])
  } else {
    lit.try.yield loc(#loc11)
  } loc(#loc11)

  // CHECK: kgen.return %[[RES]] : index loc(#[[SP_LOC:.*]])
  kgen.return %0 : !kgen.param<ty> loc(#loc10)
// CHECK: } loc(#[[SP_LOC:.*]])
} loc(#loc10)

kgen.generator @sillyFn() -> index {
  %0 = kgen.param.constant = <42>
  kgen.return %0 : index
}

kgen.generator @elaborateFnWithContextualType() -> index {
  %0 = kgen.call @takeFnContextualType<:type index, :() -> index @sillyFn>() : () -> index
  kgen.return %0 : index
}

// CHECK-DAG: #[[FILE_LOC1:.*]] = loc("test.mlir":2:3)
// CHECK-DAG: #[[SP_LOC]] = loc(fused<#[[SP]]>[#[[FILE_LOC1]]])

// CHECK-DAG: #[[FILE_LOC3:.*]] = loc("test.mlir":4:3)
// CHECK-DAG: #[[PARAM_REF_LOC:.*]] = loc(fused<1 : index>[#[[FILE_LOC3]]])
// CHECK-DAG: #[[FW_LOC]] = loc(fused<#[[SP]]>[#[[PARAM_REF_LOC]]])

// CHECK-DAG: #[[FILE_LOC2:.*]] = loc("test.mlir":3:10)
// CHECK-DAG: #[[CALL_LOC]] = loc(fused<#[[SP]]>[#[[FILE_LOC2]]])
#loc10 = loc(fused<#callerSp>["test.mlir":2:3])
#loc11 = loc(fused<#callerSp>["test.mlir":3:10])
#paramRefLoc = loc(fused<#kgen.param.decl.ref<"a">>["test.mlir":4:3])
#locFwParam = loc(fused<#callerSp>[#paramRefLoc])

// -----

// CHECK: linkageName = "entry"
#sp = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#loc = loc(fused<#sp>["a.mlir":0:0])

kgen.generator @entry() {
  kgen.return loc(#loc)
} loc(#loc)

// -----

#file = #debuginfo.file<"test.mlir" in "/">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_Mojo,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
#kernelSP = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  file = #file,
  sourceName = <"kernel">,
  subprogramFlags = Definition
> : !debuginfo.subroutine<(!kgen.simd<4, dtype>) -> (!kgen.simd<4, dtype>): DW_CC_normal>
#locKernel = loc(fused<#kernelSP>["test.mlir":0:0])

kgen.generator @kernel<dtype: dtype>(%arg0: !kgen.simd<4, dtype>) -> !kgen.simd<4, dtype> {
  kgen.return %arg0 : !kgen.simd<4, dtype> loc(#locKernel)
} loc(#locKernel)

// CHECK-LABEL: kgen.func export @top
kgen.generator export @top() {
  // CHECK-NEXT: kgen.param.constant{{.*}}!DICompositeType(tag: DW_TAG_array_type, name: \22!kgen.simd<4, ui32>\22
  kgen.param.constant: !kgen.struct<(string, index, (!kgen.pointer<none>) capturing -> !kgen.none)> = <#kgen.compile_assembly<current_target(), =llvm, "", false, :(!kgen.simd<4, ui32>) -> (!kgen.simd<4, ui32>) @kernel<:dtype ui32>>>
  kgen.return
}

// -----

// COM: When a generator has a linkageName and debug info, the elaborator
// COM: renames the function and updates the subprogram accordingly.

// The subprogram's linkageName should be updated to the export name.
// CHECK-LABEL: kgen.func @my_export()
// CHECK-NOT: linkageName = "my_export"
// CHECK: } loc([[LOC:#.*]])
// CHECK-NEXT: } loc([[MODULE_LOC:#.*]])

kgen.generator @"mangled::original_name"() attributes {linkageName = #kgen.linkage_name<"my_export" : !kgen.string, false>} {
  kgen.return loc(#export_loc)
} loc(#export_loc)

// CHECK-DAG: [[EXPORT_SP:#.*]] = #debuginfo.subprogram
// CHECK-SAME:  sourceName = <"original_name">
// CHECK-SAME:  linkageName = "my_export"

// CHECK-DAG: [[LOC]] = loc(fused<[[EXPORT_SP]]>[{{.*}}]
#original_sp = #debuginfo.subprogram<
  sourceName = <"original_name">
> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#export_loc = loc(fused<#original_sp>["export.mlir":1:1])
