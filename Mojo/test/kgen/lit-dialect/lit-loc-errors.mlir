// COM: Since errors involving incorrect locations cannot be handled by
// COM: -verify-diagnostics, we check manually.
// RUN: not kgen-opt -split-input-file %s 2>&1 | FileCheck %s

#file = #debuginfo.file<"test.mlir" in "">
#loc = loc("foo.mlir":7:8)

lit.fn @foo() {
  lit.return
// CHECK: foo.mlir:7:8: error: 'lit.fn' op must have subprogram scope in location, but got #debuginfo.file<"test.mlir" in "">
} loc(fused<#file>[#loc])

// -----

#file = #debuginfo.file<"bar.mlir" in "">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_Mojo,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
#loc = loc("bar.mlir":10:5)

lit.struct.decl @Foo {
  lit.struct.field value : index
// CHECK: bar.mlir:10:5: error: 'lit.struct.decl' op must have file scope in location, but got #debuginfo.compile_unit
} loc(fused<#compile_unit>[#loc])


// -----

#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#subprogram1 = #debuginfo.subprogram<sourceName = <"SomeClosure">> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#loc1 = loc("foo.mlir":44:1)
#loc2 = loc("foo.mlir":325:11)
#loc4 = loc(fused<#subprogram>[#loc1])
#loc5 = loc(fused<#subprogram1>[#loc2])

lit.fn @foo() {
  %0 = co.execute {
    kgen.return loc(#loc5)
  // CHECK: foo.mlir:325:11: error: 'co.execute' op must have callsite location
  } loc(#loc5)
  lit.return loc(#loc4)
} loc(#loc4)
