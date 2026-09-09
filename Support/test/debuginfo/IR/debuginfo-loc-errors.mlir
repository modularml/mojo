// COM: Since errors involving incorrect locations cannot be handled by
// COM: -verify-diagnostics, we check manually.
// RUN: not support-dialect-opt -split-input-file %s 2>&1 | FileCheck %s

#file = #debuginfo.file<"foo.mlir" in "/mlir/">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_Mojo,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  sourceName = <"foo">,
  linkageName = "foo",
  file = #file,
  line = 10,
  scopeLine = 10,
  subprogramFlags = Definition
> : !debuginfo.subroutine<(!debuginfo.unresolved<i32>) -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo",
  file = #file,
  line = 10,
  arg = 1,
  alignInBits = 32
> : !debuginfo.unresolved<i32>

#loc = loc("foo.mlir":7:8)

func.func @foo(%arg: i32) {
  // CHECK: foo.mlir:7:8: error: 'debuginfo.value' op location scope must be a child scope of the variable scope:
  // CHECK: #debuginfo.file<"foo.mlir" in "/mlir/">
  // CHECK: vs.
  // CHECK: #debuginfo.subprogram
  debuginfo.value #local_variable = %arg : i32 loc(fused<#file>[#loc])
  return
}

// -----

#file = #debuginfo.file<"foo.c" in "">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_Mojo,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  sourceName = <"foo">,
  linkageName = "foo",
  file = #file,
  line = 10,
  scopeLine = 10,
  subprogramFlags = Definition
> : !debuginfo.subroutine<(!debuginfo.unresolved<i32>) -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo",
  file = #file,
  line = 10,
  arg = 1,
  alignInBits = 32
> : !debuginfo.unresolved<i32>

#loc1 = loc("foo.mlir":7:8)
#loc2 = loc("bar.mlir":5:6)
#fusedLoc = loc(fused<#file>[#loc2])

func.func @bar(%arg: i32) {
  // CHECK: bar.mlir:5:6: error: 'debuginfo.value' op location scope must be a child scope of the variable scope:
  // CHECK: #debuginfo.file<"foo.c" in "">
  // CHECK: vs.
  // CHECK: #debuginfo.subprogram
  debuginfo.value #local_variable = %arg : i32 loc(callsite(#fusedLoc at #loc1))
  return
}

// -----

#file = #debuginfo.file<"foo.c" in "">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_Mojo,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  sourceName = <"foo">,
  linkageName = "foo",
  file = #file,
  line = 10,
  scopeLine = 10,
  subprogramFlags = Definition
> : !debuginfo.subroutine<(!debuginfo.unresolved<i32>) -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo",
  file = #file,
  line = 10,
  arg = 1,
  alignInBits = 32
> : !debuginfo.unresolved<i32>

#loc1 = loc("foo.mlir":7:8)
#loc2 = loc("bar.mlir":5:6)
#fusedLoc = loc(fused<#file>[#loc2])
#callsiteLoc = loc(callsite(#fusedLoc at #loc1))

func.func @bar(%arg: i32) {
  // CHECK: bar.mlir:5:6: error: 'debuginfo.value' op location scope must be a child scope of the variable scope:
  // CHECK: #debuginfo.file<"foo.c" in "">
  // CHECK: vs.
  // CHECK: #debuginfo.subprogram
  debuginfo.value #local_variable = %arg : i32 loc(callsite(#callsiteLoc at #loc1))
  return
}

// -----

#file = #debuginfo.file<"foo.c" in "">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_Mojo,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  sourceName = <"foo">,
  linkageName = "foo",
  file = #file,
  line = 10,
  scopeLine = 10,
  subprogramFlags = Definition
> : !debuginfo.subroutine<(!debuginfo.unresolved<i32>) -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo",
  file = #file,
  line = 10,
  arg = 1,
  alignInBits = 32
> : !debuginfo.unresolved<i32>

#loc1 = loc("foo.mlir":7:8)
#loc2 = loc("bar.mlir":5:6)

func.func @foo(%arg: i32) {
  // CHECK: foo.mlir:7:8: error: 'debuginfo.value' op with fused location must reference a single location, got 2
  debuginfo.value #local_variable = %arg : i32 loc(fused<#subprogram>[#loc1, #loc2])
  return
}
