// RUN: support-dialect-opt %s -debuginfo-strip -mlir-print-debuginfo -allow-unregistered-dialect | FileCheck %s
// RUN: support-dialect-opt %s -debuginfo-strip=preserveLineTables=1 -mlir-print-debuginfo -allow-unregistered-dialect | FileCheck %s --check-prefix=CHECK-PRESERVE-LT

// Use a subroutine type with argument types so we can verify they are stripped
// in line-tables mode (the empty () -> () form should remain).
!subroutine = !debuginfo.subroutine<(index, i64) -> (i32): DW_CC_normal>
#file = #debuginfo.file<"foo" in "foo">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #file, producer = "MLIR", isOptimized = true, emissionKind = Full>
#subprogram = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, sourceName = <"fn">, linkageName = "fn", file = #file, line = 1, scopeLine = 1, subprogramFlags = "Definition|Optimized"> : !subroutine

#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "buf", file = #file, line = 159, arg = 1> : !debuginfo.unresolved<index>

// CHECK-NOT: #debuginfo.
// CHECK: "unknown_dialect.op"
// CHECK: "test.mlir":10:10

// Check that metadata is preserved but the subroutine argument/result types
// are stripped in line-tables mode.
// CHECK-PRESERVE-LT-NOT: debuginfo.value
// CHECK-PRESERVE-LT: #debuginfo.subprogram
// CHECK-PRESERVE-LT-NOT: (index, i64) -> (i32)

func.func @foo(%arg: index) {
  "unknown_dialect.op"() : () -> ()
  debuginfo.value #local_variable = %arg : index
  return
} loc(fused<#subprogram>["test.mlir":10:10])
