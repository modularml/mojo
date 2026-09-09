// RUN: support-dialect-opt %s -convert-debuginfo-to-llvm=tradeoff-perf -allow-unregistered-dialect -mlir-print-debuginfo -split-input-file | FileCheck %s --check-prefix=CHECK --check-prefix=PRESERVE
// RUN: support-dialect-opt %s -convert-debuginfo-to-llvm -allow-unregistered-dialect -mlir-print-debuginfo -split-input-file | FileCheck %s --check-prefix=CHECK --check-prefix=OPT

#file = #debuginfo.file<"foo.c" in "/mlir/">
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
// CHECK-DAG: #[[LOCAL_VAR:.*]] = #llvm.di_local_variable<scope = {{.*}}, name = "foo"
#local_variable = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo",
  file = #file,
  line = 10,
  arg = 0,
  alignInBits = 32
> : !debuginfo.unresolved<i32>
// CHECK-DAG: #[[LOCAL_VAR2:.*]] = #llvm.di_local_variable<scope = {{.*}}, name = "foo_2"
#local_variable_2 = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo_2",
  file = #file,
  line = 10,
  arg = 0,
  alignInBits = 64
> : !debuginfo.unresolved<!llvm.ptr>
// CHECK-DAG: #[[LOCAL_VAR3:.*]] = #llvm.di_local_variable<scope = {{.*}}, name = "foo_3"
#local_variable_3 = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo_3",
  file = #file,
  line = 10,
  arg = 0,
  alignInBits = 64
> : !debuginfo.unresolved<!llvm.ptr>
#local_variable4 = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo_4",
  file = #file,
  line = 10,
  arg = 0,
  alignInBits = 32
> : !debuginfo.unresolved<i32>
#deref_expr = #debuginfo.expr.deref<#debuginfo.expr.irvalue : !debuginfo.ptr<!llvm.ptr {sizeInBits = 64, alignInBits = 64}>> : !debuginfo.unresolved<!llvm.ptr>

!struct = !debuginfo.struct<MyStruct(
            !debuginfo.member<first: !debuginfo.unresolved<i32>>,
            !debuginfo.member<second: !debuginfo.unresolved<i32>>
          )>
#irvalue = #debuginfo.expr.irvalue : !debuginfo.unresolved<i32>
#fragment_expr0 = #debuginfo.expr.agg<#irvalue, 0> : !struct
#fragment_expr1 = #debuginfo.expr.agg<#irvalue, 1> : !struct

// CHECK-DAG: #[[LOCAL_VAR_STRUCT:.*]] = #llvm.di_local_variable<scope = {{.*}}, name = "foo_struct"
#local_variable_struct = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo_struct",
  file = #file,
  line = 10,
  arg = 0,
  alignInBits = 64
> : !struct

// CHECK-LABEL: func @simple
func.func @simple() {
  // CHECK: %[[VAL:.*]] = llvm.mlir.constant(0 : i32) : i32
  %value = llvm.mlir.constant(0 : i32) : i32

  // CHECK: llvm.intr.dbg.value #{{.*}} = %[[VAL]] : i32
  debuginfo.value #local_variable = %value : i32
  return
}

// Test translation of dbg.value to dbg.addr.

// CHECK-LABEL: func @value_to_addr_arg
// CHECK-SAME: (%[[ARG:.*]]: i32 loc({{.*}}))
func.func @value_to_addr_arg(%arg: i32) -> i32 {
  // PRESERVE: %[[COUNT:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN:.*]])
  // PRESERVE: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x i32 {alignment = 1 : i64} : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // PRESERVE: llvm.intr.dbg.declare #{{.*}} = %[[ALLOC]] : !llvm.ptr
  // PRESERVE: llvm.store volatile %[[ARG]], %[[ALLOC]] {alignment = 1 : i64} : i32, !llvm.ptr loc(#[[LOC_STORE:.*]])
  // PRESERVE: return %[[ARG]] : i32

  // OPT: llvm.intr.dbg.value #{{.*}}

  debuginfo.value #local_variable = %arg : i32
  return %arg : i32
}
// PRESERVE-NEXT } loc(#[[LOC_STORE]])

// CHECK-LABEL: func @value_to_addr_op
func.func @value_to_addr_op() -> i32 {
  // PRESERVE: %[[COUNT:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x i32 {alignment = 1 : i64} : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[VALUE:.*]] = "test.op"() : () -> i32
  // PRESERVE: llvm.store volatile %[[VALUE]], %[[ALLOC]] {alignment = 1 : i64} : i32, !llvm.ptr
  // PRESERVE: llvm.intr.dbg.declare #{{.*}} = %[[ALLOC]] : !llvm.ptr
  // PRESERVE: return %[[VALUE]] : i32

  // OPT: llvm.intr.dbg.value #{{.*}}

  %value = "test.op"() : () -> i32
  debuginfo.value #local_variable = %value : i32
  return %value : i32
}

// CHECK-LABEL: func @value_with_two_nontrivial_ops
func.func @value_with_two_nontrivial_ops() -> (i32, i32) {
  // PRESERVE: %[[COUNT:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN:.*]])
  // PRESERVE: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x i32 {alignment = 1 : i64} : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[VALUE1:.*]] = "test.op"() : () -> i32
  // PRESERVE: llvm.store volatile %[[VALUE1]], %[[ALLOC]] {alignment = 1 : i64} : i32, !llvm.ptr
  // PRESERVE: llvm.intr.dbg.declare #[[VARIABLE:.*]] = %[[ALLOC]] : !llvm.ptr
  // PRESERVE: %[[VALUE2:.*]] = "test.op2"() : () -> i32
  // PRESERVE: llvm.store volatile %[[VALUE2]], %[[ALLOC]] {alignment = 1 : i64} : i32, !llvm.ptr
  // PRESERVE-NOT: llvm.intr.dbg.value #[[VARIABLE]]
  // PRESERVE: return %[[VALUE1]], %[[VALUE2]] : i32, i32

  %value1 = "test.op"() : () -> i32
  debuginfo.value #local_variable = %value1 : i32
  %value2 = "test.op2"() : () -> i32
  debuginfo.value #local_variable = %value2 : i32
  return %value1, %value2 : i32, i32
}

// CHECK-LABEL: func @value_with_two_nontrivial_ops_and_kill
func.func @value_with_two_nontrivial_ops_and_kill() -> (i32, i32) {
  // PRESERVE: %[[COUNT:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN:.*]])
  // PRESERVE: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x i32 {alignment = 1 : i64} : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[VALUE1:.*]] = "test.op"() : () -> i32
  // PRESERVE: llvm.store volatile %[[VALUE1]], %[[ALLOC]] {alignment = 1 : i64} : i32, !llvm.ptr
  // PRESERVE: llvm.intr.dbg.value #[[VARIABLE:.*]] #llvm.di_expression<[DW_OP_deref]> = %[[ALLOC]] : !llvm.ptr
  // PRESERVE: %[[VALUE2:.*]] = "test.op2"() : () -> i32
  // PRESERVE: llvm.store volatile %[[VALUE2]], %[[ALLOC]] {alignment = 1 : i64} : i32, !llvm.ptr
  // PRESERVE: llvm.intr.dbg.value #[[VARIABLE]] #llvm.di_expression<[DW_OP_deref]> = %[[ALLOC]] : !llvm.ptr
  // PRESERVE: %[[UNDEF:.*]] = llvm.mlir.undef
  // PRESERVE: llvm.intr.dbg.value #[[VARIABLE]] = %[[UNDEF]]
  // PRESERVE: return %[[VALUE1]], %[[VALUE2]] : i32, i32

  %value1 = "test.op"() : () -> i32
  debuginfo.value #local_variable = %value1 : i32
  %value2 = "test.op2"() : () -> i32
  debuginfo.value #local_variable = %value2 : i32
  debuginfo.kill #local_variable
  return %value1, %value2 : i32, i32
}

// CHECK-LABEL: func @value_shared_with_two_nontrivial_ops
func.func @value_shared_with_two_nontrivial_ops() -> (i32, i32, i32) {
  // PRESERVE: %[[COUNT2:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN:.*]])
  // PRESERVE: %[[ALLOC2:.*]] = llvm.alloca %[[COUNT2]] x i32 {alignment = 1 : i64} : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[COUNT1:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN:.*]])
  // PRESERVE: %[[ALLOC1:.*]] = llvm.alloca %[[COUNT1]] x i32 {alignment = 1 : i64} : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[VALUE1:.*]] = "test.op"() : () -> i32
  // PRESERVE: llvm.store volatile %[[VALUE1]], %[[ALLOC1]] {alignment = 1 : i64} : i32, !llvm.ptr
  // PRESERVE-DAG: llvm.intr.dbg.declare #[[VARIABLE1:.*]] = %[[ALLOC1]] : !llvm.ptr
  // PRESERVE: llvm.store volatile %[[VALUE1]], %[[ALLOC2]] {alignment = 1 : i64} : i32, !llvm.ptr
  // PRESERVE-DAG: llvm.intr.dbg.declare #[[VARIABLE2:.*]] = %[[ALLOC2]] : !llvm.ptr
  // PRESERVE: %[[VALUE2:.*]] = "test.op2"() : () -> i32
  // PRESERVE: llvm.store volatile %[[VALUE2]], %[[ALLOC1]] {alignment = 1 : i64} : i32, !llvm.ptr
  // PRESERVE: %[[VALUE3:.*]] = "test.op3"() : () -> i32
  // PRESERVE: llvm.store volatile %[[VALUE3]], %[[ALLOC2]] {alignment = 1 : i64} : i32, !llvm.ptr
  // PRESERVE-NOT: llvm.intr.dbg.value #[[VARIABLE1]]
  // PRESERVE-NOT: llvm.intr.dbg.value #[[VARIABLE2]]
  // PRESERVE: return %[[VALUE1]], %[[VALUE2]], %[[VALUE3]] : i32, i32, i32

  %value1 = "test.op"() : () -> i32
  debuginfo.value #local_variable = %value1 : i32
  debuginfo.value #local_variable4 = %value1 : i32
  %value2 = "test.op2"() : () -> i32
  debuginfo.value #local_variable = %value2 : i32
  %value3 = "test.op3"() : () -> i32
  debuginfo.value #local_variable4 = %value3 : i32
  return %value1, %value2, %value3 : i32, i32, i32
}

// CHECK-LABEL: func @value_with_one_kill
func.func @value_with_one_kill() -> i32 {
  // PRESERVE: %[[COUNT:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x i32 {alignment = 1 : i64} : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[VALUE:.*]] = "test.op"() : () -> i32
  // PRESERVE: llvm.store volatile %[[VALUE]], %[[ALLOC]] {alignment = 1 : i64} : i32, !llvm.ptr
  // PRESERVE: llvm.intr.dbg.value #[[LOCAL_VAR:.*]] #llvm.di_expression<[DW_OP_deref]> = %[[ALLOC]] : !llvm.ptr
  // PRESERVE: %[[UNDEF:.*]] = llvm.mlir.undef
  // PRESERVE: llvm.intr.dbg.value #[[LOCAL_VAR]] = %[[UNDEF]]
  // PRESERVE: return %[[VALUE]] : i32

  %value = "test.op"() : () -> i32
  debuginfo.value #local_variable = %value : i32
  debuginfo.kill #local_variable
  return %value : i32
}

// CHECK-LABEL: func @value_with_two_kills
func.func @value_with_two_kills() -> i32 {
  // PRESERVE: %[[COUNT:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x i32 {alignment = 1 : i64} : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[UNDEF1:.*]] = llvm.mlir.undef
  // PRESERVE: llvm.intr.dbg.value #[[LOCAL_VAR:.*]] = %[[UNDEF1]]
  // PRESERVE: %[[VALUE:.*]] = "test.op"() : () -> i32
  // PRESERVE: llvm.store volatile %[[VALUE]], %[[ALLOC]] {alignment = 1 : i64} : i32, !llvm.ptr
  // PRESERVE: llvm.intr.dbg.value #[[LOCAL_VAR]] #llvm.di_expression<[DW_OP_deref]> = %[[ALLOC]] : !llvm.ptr
  // PRESERVE: %[[UNDEF2:.*]] = llvm.mlir.undef
  // PRESERVE: llvm.intr.dbg.value #[[LOCAL_VAR]] = %[[UNDEF2]]
  // PRESERVE: return %[[VALUE]] : i32

  debuginfo.kill #local_variable
  %value = "test.op"() : () -> i32
  debuginfo.value #local_variable = %value : i32
  debuginfo.kill #local_variable
  return %value : i32
}

// CHECK-LABEL: func @undef_value_and_kill
func.func @undef_value_and_kill() -> i32 {
  // PRESERVE: %[[UNDEF1:.*]] = llvm.mlir.undef : i32
  // PRESERVE: llvm.intr.dbg.value #[[LOCAL_VAR]] = %[[UNDEF1]]
  // PRESERVE: %[[UNDEF2:.*]] = llvm.mlir.undef
  // PRESERVE: llvm.intr.dbg.value #[[LOCAL_VAR]] = %[[UNDEF2]]
  // PRESERVE: return %[[UNDEF1]] : i32

  %undef1 = llvm.mlir.undef : i32
  debuginfo.value #local_variable = %undef1 : i32
  debuginfo.kill #local_variable
  return %undef1 : i32
}

// CHECK-LABEL: func @two_value_to_addr_op
func.func @two_value_to_addr_op() -> !llvm.ptr {
  // PRESERVE: %[[COUNT0:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[ALLOC0:.*]] = llvm.alloca %[[COUNT0]] x !llvm.ptr {alignment = 1 : i64} : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[COUNT1:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[ALLOC1:.*]] = llvm.alloca %[[COUNT1]] x !llvm.ptr {alignment = 1 : i64} : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[VALUE:.*]] = "test.op"() : () -> !llvm.ptr
  // PRESERVE: llvm.store volatile %[[VALUE]], %[[ALLOC1]] {alignment = 1 : i64} : !llvm.ptr, !llvm.ptr
  // PRESERVE: llvm.intr.dbg.declare #{{.*}} = %[[ALLOC1]] : !llvm.ptr
  // PRESERVE: llvm.store volatile %[[VALUE]], %[[ALLOC0]] {alignment = 1 : i64} : !llvm.ptr, !llvm.ptr
  // PRESERVE: llvm.intr.dbg.declare #{{.*}} = %[[ALLOC0]] : !llvm.ptr
  // PRESERVE: return %[[VALUE]] : !llvm.ptr

  %value = "test.op"() : () -> !llvm.ptr
  debuginfo.value #local_variable_2 = %value : !llvm.ptr
  debuginfo.value #local_variable_3 = %value : !llvm.ptr
  return %value : !llvm.ptr
}

// CHECK-LABEL: func @one_value_one_value_and_kill
func.func @one_value_one_value_and_kill() -> !llvm.ptr {
  // PRESERVE: %[[COUNT0:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[ALLOC0:.*]] = llvm.alloca %[[COUNT0]] x !llvm.ptr {alignment = 1 : i64} : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[COUNT1:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[ALLOC1:.*]] = llvm.alloca %[[COUNT1]] x !llvm.ptr {alignment = 1 : i64} : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[VALUE:.*]] = "test.op"() : () -> !llvm.ptr
  // PRESERVE-DAG: llvm.store volatile %[[VALUE]], %[[ALLOC1]] {alignment = 1 : i64} : !llvm.ptr, !llvm.ptr
  // PRESERVE: llvm.intr.dbg.declare #[[LOCAL_VAR2]] = %[[ALLOC1]] : !llvm.ptr
  // PRESERVE-DAG: llvm.store volatile %[[VALUE]], %[[ALLOC0]] {alignment = 1 : i64} : !llvm.ptr, !llvm.ptr
  // PRESERVE: llvm.intr.dbg.value #[[LOCAL_VAR3]] #llvm.di_expression<[DW_OP_deref]> = %[[ALLOC0]] : !llvm.ptr
  // PRESERVE: %[[UNDEF:.*]] = llvm.mlir.undef
  // PRESERVE: llvm.intr.dbg.value #[[LOCAL_VAR3]] = %[[UNDEF]]
  // PRESERVE: return %[[VALUE]] : !llvm.ptr

  %value = "test.op"() : () -> !llvm.ptr
  debuginfo.value #local_variable_2 = %value : !llvm.ptr
  debuginfo.value #local_variable_3 = %value : !llvm.ptr
  debuginfo.kill #local_variable_3
  return %value : !llvm.ptr
}

// CHECK-LABEL: func @one_value_one_deref_to_addr_op
func.func @one_value_one_deref_to_addr_op() -> !llvm.ptr {
  // PRESERVE: %[[COUNT:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x !llvm.ptr {alignment = 1 : i64} : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[VALUE:.*]] = "test.op"() : () -> !llvm.ptr
  // PRESERVE: llvm.store volatile %[[VALUE]], %[[ALLOC]] {alignment = 1 : i64} : !llvm.ptr, !llvm.ptr
  // PRESERVE: llvm.intr.dbg.declare #{{.*}} = %[[ALLOC]] : !llvm.ptr
  // PRESERVE: llvm.intr.dbg.declare #{{.*}} = %[[VALUE]] : !llvm.ptr
  // PRESERVE: return %[[VALUE]] : !llvm.ptr

  %value = "test.op"() : () -> !llvm.ptr
  debuginfo.value #local_variable_2 = %value : !llvm.ptr
  debuginfo.value #local_variable_3 #deref_expr = %value : !llvm.ptr
  return %value : !llvm.ptr
}

// CHECK-LABEL: func @one_value_one_deref_to_addr_op_and_kill
func.func @one_value_one_deref_to_addr_op_and_kill() -> !llvm.ptr {
  // PRESERVE: %[[COUNT:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x !llvm.ptr {alignment = 1 : i64} : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[VALUE:.*]] = "test.op"() : () -> !llvm.ptr
  // PRESERVE: llvm.store volatile %[[VALUE]], %[[ALLOC]] {alignment = 1 : i64} : !llvm.ptr, !llvm.ptr
  // PRESERVE: llvm.intr.dbg.value #[[VARIABLE:.*]] #llvm.di_expression<[DW_OP_deref]> = %[[ALLOC]] : !llvm.ptr
  // PRESERVE: llvm.intr.dbg.value #[[VARIABLE]] #llvm.di_expression<[DW_OP_deref]> = %[[VALUE]] : !llvm.ptr
  // PRESERVE: %[[UNDEF:.*]] = llvm.mlir.undef
  // PRESERVE: llvm.intr.dbg.value #[[VARIABLE]] = %[[UNDEF]]
  // PRESERVE: return %[[VALUE]] : !llvm.ptr

  %value = "test.op"() : () -> !llvm.ptr
  debuginfo.value #local_variable_2 = %value : !llvm.ptr
  debuginfo.value #local_variable_2 #deref_expr = %value : !llvm.ptr
  debuginfo.kill #local_variable_2
  return %value : !llvm.ptr
}

// CHECK-LABEL: func @one_arg_one_deref_to_addr_op_and_kill
// CHECK-SAME: (%[[ARG:.*]]: !llvm.ptr
func.func @one_arg_one_deref_to_addr_op_and_kill(%arg: !llvm.ptr) -> !llvm.ptr {
  // PRESERVE: %[[COUNT:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x !llvm.ptr {alignment = 1 : i64} : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // PRESERVE: llvm.intr.dbg.value #[[VARIABLE:.*]] #llvm.di_expression<[DW_OP_deref, DW_OP_deref]> = %[[ALLOC]] : !llvm.ptr
  // PRESERVE: llvm.store volatile %[[ARG]], %[[ALLOC]] {alignment = 1 : i64} : !llvm.ptr, !llvm.ptr
  // PRESERVE: %[[UNDEF:.*]] = llvm.mlir.undef
  // PRESERVE: llvm.intr.dbg.value #[[VARIABLE]] = %[[UNDEF]]
  // PRESERVE: return %[[ARG]] : !llvm.ptr

  debuginfo.value #local_variable_2 #deref_expr = %arg : !llvm.ptr
  debuginfo.kill #local_variable_2
  return %arg : !llvm.ptr
}

// CHECK-LABEL: func @one_deref_one_value_to_addr_op
func.func @one_deref_one_value_to_addr_op() -> !llvm.ptr {
  // PRESERVE: %[[COUNT:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x !llvm.ptr {alignment = 1 : i64} : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[VALUE:.*]] = "test.op"() : () -> !llvm.ptr
  // PRESERVE: llvm.intr.dbg.declare #{{.*}} = %[[VALUE]] : !llvm.ptr
  // PRESERVE: llvm.store volatile %[[VALUE]], %[[ALLOC]] {alignment = 1 : i64} : !llvm.ptr, !llvm.ptr
  // PRESERVE: llvm.intr.dbg.declare #{{.*}} = %[[ALLOC]] : !llvm.ptr
  // PRESERVE: return %[[VALUE]] : !llvm.ptr

  %value = "test.op"() : () -> !llvm.ptr
  debuginfo.value #local_variable_2 #deref_expr = %value : !llvm.ptr
  debuginfo.value #local_variable_3 = %value : !llvm.ptr
  return %value : !llvm.ptr
}

// CHECK-LABEL: @block_arguments
llvm.func @block_arguments() {
  %0 = llvm.mlir.constant(0 : i32) : i32
  llvm.br ^bb1(%0 : i32)
// CHECK: fused<#di_subprogram>
^bb1(%arg0: i32 loc(fused<#subprogram>["foo.mlir":0:0])):
  llvm.return
}

// CHECK-LABEL: func @value_with_struct_fields
func.func @value_with_struct_fields() -> (i32, i32) {
  // PRESERVE: %[[COUNT:.*]] = llvm.mlir.constant(8 : i32) : i32 loc(#[[LOC_UNKNOWN:.*]])
  // PRESERVE: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x i8 {alignment = 1 : i64} : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // PRESERVE: %[[VALUE1:.*]] = "test.op"() : () -> i32
  // PRESERVE: llvm.store volatile %[[VALUE1]], %[[ALLOC]] {alignment = 1 : i64} : i32, !llvm.ptr
  // PRESERVE: llvm.intr.dbg.declare #[[LOCAL_VAR_STRUCT]] = %[[ALLOC]] : !llvm.ptr
  // PRESERVE: %[[VALUE2:.*]] = "test.op2"() : () -> i32
  // PRESERVE: %[[GEP:.*]] = llvm.getelementptr %[[ALLOC]][4]
  // PRESERVE: llvm.store volatile %[[VALUE2]], %[[GEP]] {alignment = 1 : i64} : i32, !llvm.ptr
  // PRESERVE: return %[[VALUE1]], %[[VALUE2]] : i32, i32

  %value1 = "test.op"() : () -> i32
  debuginfo.value #local_variable_struct #fragment_expr0 = %value1 : i32
  %value2 = "test.op2"() : () -> i32
  debuginfo.value #local_variable_struct #fragment_expr1 = %value2 : i32
  return %value1, %value2 : i32, i32
}

// PRESERVE: #[[LOC_UNKNOWN]] = loc(unknown)

// -----

#file = #debuginfo.file<"foo.c" in "/mlir/">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_Mojo,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
#subprogram = #debuginfo.subprogram<
  compileUnit =#compile_unit,
  file = #file,
  scope = #file,
  sourceName = <"foo">
> : !debuginfo.subroutine<(!debuginfo.unresolved<i32>) -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo"
> : !debuginfo.unresolved<i32>
#loc0 = loc(fused<#subprogram>["foo.mlir":0:0])
#loc1 = loc(fused<#subprogram>["foo.mlir":1:0])
#loc2 = loc(fused<#subprogram>["foo.mlir":2:0])
#loc3 = loc(fused<#subprogram>["foo.mlir":3:0])

// CHECK-LABEL: func @sink_debug_kills
func.func @sink_debug_kills() -> i32 {
  // CHECK: llvm.mlir.undef
  // CHECK: llvm.intr.dbg.value
  // CHECK: "test.op"
  // CHECK: llvm.intr.dbg.value
  // CHECK: "test.op2"
  // CHECK: "test.op3"
  // CHECK: llvm.intr.dbg.value
  // CHECK: "test.op4"
  // CHECK: return

  debuginfo.kill #local_variable loc(#loc0)
  %value = "test.op"() : () -> i32 loc(#loc1)
  debuginfo.value #local_variable = %value : i32 loc(#loc1)
  debuginfo.kill #local_variable loc(#loc2)
  %value2 = "test.op2"() : () -> i32 loc(#loc2)
  %value3 = "test.op3"() : () -> i32 loc(#loc2)
  %value4 = "test.op4"() : () -> i32 loc(#loc3)
  return %value : i32  loc(#loc3)
} loc(#loc0)

// CHECK-LABEL: func @sink_debug_kills_stale_after_value
func.func @sink_debug_kills_stale_after_value() -> i32 {
  // CHECK: "test.op"
  // COM: First debug kill made stale by first debug value.
  // CHECK: llvm.intr.dbg.value
  // CHECK: "test.op2"
  // CHECK: "test.op3"
  // COM: Second debug kill made stale by second debug value.
  // CHECK: llvm.intr.dbg.value
  // COM: Last debug kill remains.
  // CHECK: llvm.mlir.undef
  // CHECK: return

  debuginfo.kill #local_variable loc(#loc0)
  %value = "test.op"() : () -> i32 loc(#loc0)
  debuginfo.value #local_variable = %value : i32 loc(#loc1)
  debuginfo.kill #local_variable loc(#loc2)
  %value2 = "test.op2"() : () -> i32 loc(#loc2)
  %value3 = "test.op3"() : () -> i32 loc(#loc2)
  debuginfo.value #local_variable = %value : i32 loc(#loc2)
  %value4 = "test.op4"() : () -> i32 loc(#loc2)
  debuginfo.kill #local_variable loc(#loc3)
  return %value : i32  loc(#loc3)
} loc(#loc0)

// -----

#file = #debuginfo.file<"foo.c" in "/mlir/">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_Mojo,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
#sp0 = #debuginfo.subprogram<
  compileUnit =#compile_unit,
  file = #file,
  scope = #file,
  sourceName = <"sp0">
> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#sp1 = #debuginfo.subprogram<
  compileUnit =#compile_unit,
  file = #file,
  scope = #file,
  sourceName = <"sp1">
> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#sp2 = #debuginfo.subprogram<
  compileUnit =#compile_unit,
  file = #file,
  scope = #file,
  sourceName = <"sp2">
> : !debuginfo.subroutine<() -> (): DW_CC_normal>
// CHECK-DAG: #[[VAR0:.*]] = #llvm.di_local_variable<{{.*}}name = "foo0"
#var0 = #debuginfo.local_variable<
  scope = #sp0,
  name = "foo0"
> : !debuginfo.unresolved<i32>
// CHECK-DAG: #[[VAR1:.*]] = #llvm.di_local_variable<{{.*}}name = "foo1"
#var1 = #debuginfo.local_variable<
  scope = #sp1,
  name = "foo1"
> : !debuginfo.unresolved<i32>
// CHECK-DAG: #[[VAR2:.*]] = #llvm.di_local_variable<{{.*}}name = "foo2"
#var2 = #debuginfo.local_variable<
  scope = #sp2,
  name = "foo2"
> : !debuginfo.unresolved<i32>

#loc0 = loc(fused<#sp0>["foo.mlir":0:0])
#loc1 = loc(fused<#sp0>["foo.mlir":1:0])
#loc100 = loc(fused<#sp1>["foo.mlir":100:0])
#loc101 = loc(fused<#sp1>["foo.mlir":101:0])
#loc200 = loc(fused<#sp2>["foo.mlir":200:0])
#loc201 = loc(fused<#sp2>["foo.mlir":201:0])

// Inlined call stack: sp0 -> sp1 -> sp2
#loc100at1 = loc(callsite(#loc100 at #loc1))
#loc101at1 = loc(callsite(#loc101 at #loc1))
#loc200at101at1 = loc(callsite(callsite(#loc200 at #loc101) at #loc1))
// Linearized locations canonicalize away associativity of callsites.
#loc201at101at1_ver0 = loc(callsite(callsite(#loc201 at #loc101) at #loc1))
#loc201at101at1_ver1 = loc(callsite(#loc201 at callsite(#loc101 at #loc1)))


// CHECK-LABEL: func @sink_kill_debug_values_inlined
func.func @sink_kill_debug_values_inlined() -> i32 {
  // CHECK: "test.op0"
  // CHECK: llvm.intr.dbg.value #[[VAR0]]
  // CHECK: %[[UNDEF0:.*]] = llvm.mlir.undef
  // CHECK: llvm.intr.dbg.value #[[VAR0]] {{.*}}= %[[UNDEF0]]

  // CHECK: "test.op1"
  // CHECK: llvm.intr.dbg.value #[[VAR1]]

  // CHECK: "test.op2"
  // CHECK: llvm.intr.dbg.value #[[VAR2]]
  // CHECK: "test.op3"
  // CHECK: %[[UNDEF2:.*]] = llvm.mlir.undef
  // CHECK: llvm.intr.dbg.value #[[VAR2]] {{.*}}= %[[UNDEF2]]

  // CHECK: "test.op4"
  // CHECK: %[[UNDEF1:.*]] = llvm.mlir.undef
  // CHECK: llvm.intr.dbg.value #[[VAR1]] {{.*}}= %[[UNDEF1]]

  // CHECK: return


  %v0 = "test.op0"() : () -> i32 loc(#loc0)
  debuginfo.value #var0 = %v0 : i32 loc(#loc0)
  debuginfo.kill #var0 loc(#loc0)

  //   Begin body of sp1 {
  %v1 = "test.op1"() : () -> i32 loc(#loc100at1)
  debuginfo.value #var1 = %v1 : i32 loc(#loc100at1)
  debuginfo.kill #var1 loc(#loc101at1)

  //     Begin body of sp2 {
  %v2 = "test.op2"() : () -> i32 loc(#loc200at101at1)
  debuginfo.value #var2 = %v2 : i32 loc(#loc200at101at1)
  debuginfo.kill #var2 loc(#loc201at101at1_ver0)
  %v3 = "test.op3"() : () -> i32 loc(#loc201at101at1_ver1)
  //     } End body of sp2

  %v4 = "test.op4"() : () -> i32 loc(#loc101at1)
  //   } End body of sp1

  return %v4 : i32  loc(#loc1)
} loc(#loc0)

// CHECK-LABEL: func @sink_kill_debug_values_block_end
func.func @sink_kill_debug_values_block_end() -> i32 {
  // CHECK: "test.op0"
  // CHECK: llvm.intr.dbg.value #[[VAR0]]
  // CHECK: "test.op1"
  // CHECK: %[[UNDEF0:.*]] = llvm.mlir.undef
  // CHECK: llvm.intr.dbg.value #[[VAR0]] {{.*}}= %[[UNDEF0]]
  // CHECK: return

  %v0 = "test.op0"() : () -> i32 loc(#loc0)
  debuginfo.value #var0 = %v0 : i32 loc(#loc0)
  debuginfo.kill #var0 loc(#loc1)
  %v1 = "test.op1"() : () -> i32 loc(#loc1)
  return %v0 : i32  loc(#loc1)
} loc(#loc0)

// -----

#file = #debuginfo.file<"foo.c" in "/mlir/">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_Mojo,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
#sp = #debuginfo.subprogram<
  compileUnit =#compile_unit,
  file = #file,
  scope = #file,
  sourceName = <"foo">
> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#loc0 = loc(fused<#sp>["foo.mlir":0:0])
#loc1 = loc(fused<#sp>["foo.mlir":1:0])
#loc100at1 = loc(callsite("foo.mlir":100:0 at #loc1))

// CHECK-LABEL: func @line_table_loc_lowering
func.func @line_table_loc_lowering() {
  // CHECK-NEXT: "test.op0"
  // CHECK-NEXT: llvm.inline_asm has_side_effects asm_dialect = att "nop", ""  : () -> () loc(#[[LOC1:.+]])
  // CHECK-NEXT: "test.op1"

  "test.op0"() : () -> i32 loc(#loc0)
  debuginfo.line_table_loc loc(#loc1)
  "test.op1"() : () -> i32 loc(#loc100at1)
} loc(#loc0)

// CHECK: #[[LOC1_RAW:.+]] = loc("foo.mlir":1:0)
// CHECK: #[[LOC1]] = loc(fused<{{.*}}>[#[[LOC1_RAW]]])
