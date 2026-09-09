// RUN: support-dialect-opt %s -convert-debuginfo-to-llvm | FileCheck %s

// CHECK: #[[VAR_TYPE:.*]] = #llvm.di_basic_type

// CHECK: #[[FILE:.*]] = #llvm.di_file<"foo.c" in "/mlir/">
#file = #debuginfo.file<"foo.c" in "/mlir/">

// CHECK: #[[CU:.*]] = #llvm.di_compile_unit<
// CHECK-SAME:   sourceLanguage = DW_LANG_Mojo,
// CHECK-SAME:   file = #[[FILE]],
// CHECK-SAME:   producer = "MLIR",
// CHECK-SAME:   isOptimized = true,
// CHECK-SAME:   emissionKind = Full
// CHECK-SAME: >

// CHECK: #[[SP_TYPE:.*]] = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #di_null_type>
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_Mojo,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>

// CHECK: #[[SP:.*]] = #llvm.di_subprogram<
// CHECK-SAME:   compileUnit = #[[CU]],
// CHECK-SAME:   scope = #[[FILE]],
// CHECK-SAME:   name = "foo",
// CHECK-SAME:   linkageName = "foo",
// CHECK-SAME:   file = #[[FILE]],
// CHECK-SAME:   line = 10,
// CHECK-SAME:   scopeLine = 10,
// CHECK-SAME:   subprogramFlags = Definition
// CHECK-SAME:   type = #[[SP_TYPE]]
// CHECK-SAME: >
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  sourceName = <"foo">,
  linkageName = "foo",
  file = #file,
  line = 10,
  scopeLine = 10,
  subprogramFlags = Definition
> : !debuginfo.subroutine<() -> (): DW_CC_normal>

// CHECK: #[[LEX_BLOCK:.*]] = #llvm.di_lexical_block<
// CHECK-SAME:   scope = #[[SP]],
// CHECK-SAME:   file = #[[FILE]],
// CHECK-SAME:   line = 10,
// CHECK-SAME:   column = 1
// CHECK-SAME: >
#lex_block = #debuginfo.lexical_block<
  scope = #subprogram,
  file = #file,
  line = 10,
  column = 1
>

// CHECK: #[[VAR:.*]] = #llvm.di_local_variable<
// CHECK-SAME:   scope = #[[LEX_BLOCK]],
// CHECK-SAME:   name = "foo",
// CHECK-SAME:   file = #[[FILE]],
// CHECK-SAME:   line = 10,
// CHECK-SAME:   alignInBits = 32,
// CHECK-SAME:   type = #[[VAR_TYPE]]
// CHECK-SAME: >
#local_variable = #debuginfo.local_variable<
  scope = #lex_block,
  name = "foo",
  file = #file,
  line = 10,
  arg = 0,
  alignInBits = 32
> : !debuginfo.unresolved<i32>

func.func @foo(%arg: i32) {
  debuginfo.value #local_variable = %arg : i32
  return
}
