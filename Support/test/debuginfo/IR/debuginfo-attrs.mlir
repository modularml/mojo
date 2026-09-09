// RUN: support-dialect-opt %s | support-dialect-opt | FileCheck %s
// RUN: support-dialect-opt -emit-bytecode %s | support-dialect-opt | FileCheck %s

// CHECK: ![[SUBROUTINE:.*]] = !debuginfo.subroutine<() -> (): DW_CC_normal>
// CHECK-DAG: ![[UNRESOLVED_INDEX:.*]] = !debuginfo.unresolved<index>
!unresolved_index = !debuginfo.unresolved<index>
// The "unresolved" wrapper below will be erased by the custom printer.
// CHECK-DAG: ![[PTR2INDEX:.*]] = !debuginfo.ti.ptr<index>
!ptr2index = !debuginfo.ti.ptr<!debuginfo.unresolved<index>>
// CHECK-DAG: ![[PTR2PTR2INDEX:.*]] = !debuginfo.ti.ptr<![[PTR2INDEX]]>
!ptr2ptr2index = !debuginfo.ti.ptr<!ptr2index>
// CHECK-DAG: ![[STRUCTPAIR:.*]] = !debuginfo.struct
!member0 = !debuginfo.member<first: !unresolved_index>
!member1 = !debuginfo.member<second: !unresolved_index>
!struct_pair = !debuginfo.struct<Pair(!member0, !member1)>
// CHECK-DAG: #[[FILE:.*]] = #debuginfo.file<"foo.c" in "/mlir/">
#file = #debuginfo.file<"foo.c" in "/mlir/">

// CHECK: #[[CU:.*]] = #debuginfo.compile_unit<
// CHECK-SAME:   sourceLanguage = DW_LANG_Mojo,
// CHECK-SAME:   file = #[[FILE]],
// CHECK-SAME:   producer = "MLIR",
// CHECK-SAME:   isOptimized = true,
// CHECK-SAME:   emissionKind = Full
// CHECK-SAME: >
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_Mojo,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>

// CHECK-DAG: #[[IRVALUE:.*]] = #debuginfo.expr.irvalue : ![[PTR2INDEX]]
#irvalue = #debuginfo.expr.irvalue : !ptr2index
// CHECK-DAG: #[[DEREF:.*]] = #debuginfo.expr.deref<#[[IRVALUE]]> : ![[UNRESOLVED_INDEX]]
#deref = #debuginfo.expr.deref<#irvalue> : !unresolved_index
// CHECK-DAG: #[[REF:.*]] = #debuginfo.expr.refof<#[[IRVALUE]]> : ![[PTR2PTR2INDEX]]
#ref = #debuginfo.expr.refof<#irvalue> : !ptr2ptr2index

// CHECK: #[[SP:.*]] = #debuginfo.subprogram<
// CHECK-SAME:   compileUnit = #[[CU]],
// CHECK-SAME:   scope = #[[FILE]],
// CHECK-SAME:   sourceName = <"foo">,
// CHECK-SAME:   linkageName = "foo",
// CHECK-SAME:   file = #[[FILE]],
// CHECK-SAME:   line = 10,
// CHECK-SAME:   scopeLine = 10,
// CHECK-SAME:   subprogramFlags = Definition
// CHECK-SAME: > : ![[SUBROUTINE]]
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

// CHECK: #[[AGG:.*]] = #debuginfo.expr.agg<#[[DEREF]], 1> : ![[STRUCTPAIR]]
#agg = #debuginfo.expr.agg<#deref, 1> : !struct_pair

// CHECK: #[[LEX_BLOCK:.*]] = #debuginfo.lexical_block<
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

// CHECK: #[[VAR:.*]] = #debuginfo.local_variable<
// CHECK-SAME:   scope = #[[LEX_BLOCK]],
// CHECK-SAME:   name = "foo",
// CHECK-SAME:   file = #[[FILE]],
// CHECK-SAME:   line = 10,
// CHECK-SAME:   arg = 1,
// CHECK-SAME:   alignInBits = 32
// CHECK-SAME:   flags = Artificial
// CHECK-SAME: > : ![[UNRESOLVED_INDEX]]
#local_variable = #debuginfo.local_variable<
  scope = #lex_block,
  name = "foo",
  file = #file,
  line = 10,
  arg = 1,
  alignInBits = 32,
  flags = Artificial
> : !unresolved_index

// CHECK: module attributes {test.expr1 = #[[REF]], test.expr2 = #[[DEREF]], test.expr3 = #[[AGG]], test.loc = #[[VAR]]}
module attributes {test.expr1 = #ref, test.expr2 = #deref, test.expr3 = #agg, test.loc = #local_variable} {}
