// COM: Since errors involving incorrect locations cannot be handled by
// COM: -verify-diagnostics, we check manually.
// RUN: not kgen-opt -split-input-file %s 2>&1 | FileCheck %s

#file = #debuginfo.file<"test.mlir" in "">
#loc = loc("foo.mlir":7:8)

kgen.func @foo() {
  kgen.return
// CHECK: foo.mlir:7:8: error: 'kgen.func' op must have subprogram scope in location, but got #debuginfo.file<"test.mlir" in "">
} loc(fused<#file>[#loc])

// -----

#file = #debuginfo.file<"test.mlir" in "">
#loc = loc("foo.mlir":7:8)

kgen.generator @foo() {
  kgen.return
// CHECK: foo.mlir:7:8: error: 'kgen.generator' op must have subprogram scope in location, but got #debuginfo.file<"test.mlir" in "">
} loc(fused<#file>[#loc])

// -----

#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#loc = loc("foo.mlir":7:8)

kgen.func @foo() {
  // CHECK: error: 'kgen.return' op missing source location scope
  kgen.return
} loc(fused<#subprogram>[#loc])

// -----

#file = #debuginfo.file<"foo.mlir" in "/">
#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#loc = loc("foo.mlir":7:8)

kgen.func @foo() {
  // CHECK: foo.mlir:7:8: error: 'kgen.return' op location scope does not match scope of parent func location
  kgen.return loc(fused<#file>[#loc])
} loc(fused<#subprogram>[#loc])

// -----

#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#loc = loc("foo.mlir":7:8)
#loc1 = loc("bar.mlir":5:6)
#funcLoc = loc(fused<#subprogram>[#loc])
#callsite = loc(callsite(#loc1 at #loc))

kgen.func @foo() {
  // CHECK: bar.mlir:5:6: error: 'kgen.return' op missing source location scope
  kgen.return loc(callsite(#loc1 at #callsite))
} loc(#funcLoc)

// -----

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

#loc = loc("foo.mlir":7:8)
#loc1 = loc("bar.mlir":5:6)
#funcLoc = loc(fused<#subprogram>[#loc])

kgen.func @foo() {
  // CHECK: foo.mlir:7:8: error: 'kgen.param.constant' op contains inconsistent scopes in fused location
  %index1 = kgen.param.constant = <1> loc(callsite(#loc at fused[#loc1, #funcLoc]))
  kgen.return loc(#funcLoc)
} loc(#funcLoc)

// -----

#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#subprogram1 = #debuginfo.subprogram<sourceName = <"SomeClosure">> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#loc1 = loc("foo.mlir":44:1)
#loc2 = loc("foo.mlir":325:11)
#loc4 = loc(fused<#subprogram>[#loc1])
#loc5 = loc(fused<#subprogram1>[#loc2])

kgen.func @foo() {
  %0 = kgen.stage_closure = () {
    kgen.return loc(#loc5)
  // CHECK: foo.mlir:325:11: error: 'kgen.stage_closure' op must have callsite location
  } loc(#loc5)
  kgen.call_indirect %0() : () -> () loc(#loc4)
  kgen.return loc(#loc4)
} loc(#loc4)

// -----

#loc1 = loc("foo.mlir":44:1)
#loc2 = loc("foo.mlir":325:11)
#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#loc4 = loc(fused<#subprogram>[#loc1])

kgen.func @foo() {
  kgen.return
// CHECK: foo.mlir:44:1: error: 'kgen.func' op without debuginfo scope must contain only file/line/col location
} loc(callsite(#loc4 at #loc2))

// -----

#file = #debuginfo.file<"foo.mlir" in "/">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #file, producer = "Mojo", isOptimized = true, emissionKind = Full>
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  sourceName = <"foo">,
  linkageName = "foo",
  file = #file,
  line = 44,
  scopeLine = 44,
  subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#loc1 = loc("foo.mlir":44:1)
#loc2 = loc("foo.mlir":325:11)

kgen.func @foo() {
  kgen.return
// CHECK: foo.mlir:44:1: error: 'kgen.func' op must contain exactly one location
} loc(fused<#subprogram>[#loc1, #loc2])

// -----

#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#loc1 = loc("foo.mlir":44:1)
#loc2 = loc("foo.mlir":325:11)
#loc3 = loc(callsite(#loc1 at #loc2))

kgen.func @foo() {
  kgen.return
// CHECK: foo.mlir:44:1: error: 'kgen.func' op must contain only file/line/col location
} loc(fused<#subprogram>[#loc3])
