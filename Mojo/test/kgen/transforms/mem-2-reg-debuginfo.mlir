// RUN: kgen-opt -split-input-file -mem-2-reg -allow-unregistered-dialect -mlir-print-debuginfo %s | FileCheck %s

// CHECK-DAG: #[[CALLER_SP:.*]] = #debuginfo.subprogram<sourceName = <"mem2reg_valueop">>
// CHECK-DAG: #[[CALLEE_SP:.*]] = #debuginfo.subprogram<sourceName = <"mem2reg_valueop_callee">>
// CHECK-DAG: #[[NESTED_CALLEE_SP:.*]] = #debuginfo.subprogram<sourceName = <"mem2reg_valueop_nested_callee">>
#callerSp = #debuginfo.subprogram<sourceName = <"mem2reg_valueop">> : !debuginfo.subroutine<(index) -> (): DW_CC_normal>
#calleeSp = #debuginfo.subprogram<sourceName = <"mem2reg_valueop_callee">> : !debuginfo.subroutine<(index) -> (): DW_CC_normal>
#nestedCalleeSp = #debuginfo.subprogram<sourceName = <"mem2reg_valueop_nested_callee">> : !debuginfo.subroutine<(index) -> (): DW_CC_normal>

// Caller locs
#loc0 = loc(fused<#callerSp>["foo.mlir":0:0])
#loc1 = loc(fused<#callerSp>["foo.mlir":1:0])
#loc2 = loc(fused<#callerSp>["foo.mlir":2:0])
#loc3 = loc(fused<#callerSp>["foo.mlir":3:0])

// Callee locs
#loc100 = loc(fused<#calleeSp>["foo.mlir":100:0])
#loc101 = loc(fused<#calleeSp>["foo.mlir":101:0])
#loc102 = loc(fused<#calleeSp>["foo.mlir":102:0])
#loc100at2 = loc(callsite(#loc100 at #loc2))
#loc101at2 = loc(callsite(#loc101 at #loc2))
#loc102at2 = loc(callsite(#loc102 at #loc2))

// Nested callee locs
#loc200 = loc(fused<#nestedCalleeSp>["foo.mlir":200:0])
#loc201 = loc(fused<#nestedCalleeSp>["foo.mlir":201:0])
#loc200at102at2 = loc(callsite(#loc200 at #loc102at2))
#loc201at102at2 = loc(callsite(#loc201 at #loc102at2))

!struct_with_single_index_ptr = !debuginfo.struct<MyStruct(
  !debuginfo.member<first: !debuginfo.ti.ptr<index>>
)>

// CHECK-DAG: #[[IRVALUE_EXPR:.*]] = #debuginfo.expr.irvalue : index
// CHECK-DAG: #[[REFOF_EXPR:.*]] = #debuginfo.expr.refof<#[[IRVALUE_EXPR]]> : !kgen.pointer<index>
// CHECK-DAG: #[[AGG_REGOF_EXPR:.*]] = #debuginfo.expr.agg<#[[REFOF_EXPR]], 0>

// CHECK-DAG: #[[VAR:.*]] = #debuginfo.local_variable<scope = #[[CALLER_SP]], name = "0"
// CHECK-DAG: #[[VAR_STRUCT:.*]] = #debuginfo.local_variable<scope = #[[CALLER_SP]], name = "1"
// CHECK-DAG: #[[VAR_CALLEE:.*]] = #debuginfo.local_variable<scope = #[[CALLEE_SP]], name = "2", arg = 1
// CHECK-DAG: #[[VAR_NESTED_CALLEE:.*]] = #debuginfo.local_variable<scope = #[[NESTED_CALLEE_SP]], name = "3", arg = 1
#local_variable = #debuginfo.local_variable<scope = #callerSp, name = "0"> : !debuginfo.ti.ptr<index>
#local_variable_struct = #debuginfo.local_variable<scope = #callerSp, name = "1"> : !struct_with_single_index_ptr
#local_variable_callee = #debuginfo.local_variable<scope = #calleeSp, name = "2", arg = 1> : !debuginfo.ti.ptr<index>
#local_variable_nested_callee = #debuginfo.local_variable<scope = #nestedCalleeSp, name = "3", arg = 1> : !debuginfo.ti.ptr<index>

#agg_expr = #debuginfo.expr.agg<#debuginfo.expr.irvalue : !kgen.pointer<index>, 0> : !kgen.struct<(!kgen.pointer<index>)>

// CHECK-LABEL: @mem2reg_valueop_no_undef
kgen.func @mem2reg_valueop_no_undef(%arg0: index, %arg1: index) {
  // CHECK-NOT: kgen.param.constant = <*?>
  // CHECK: debuginfo.value #[[VAR]] #[[REFOF_EXPR]] = %arg0 : index loc(#[[LOC1:.*]])
  // CHECK: debuginfo.value #[[VAR]] #[[REFOF_EXPR]] = %arg1 : index loc(#[[LOC2:.*]])
  %0 = pop.stack_allocation 1 x index loc(#loc0)
  debuginfo.value #local_variable = %0 : !kgen.pointer<index> loc(#loc0)
  pop.store %arg0, %0 : !kgen.pointer<index> loc(#loc1)
  pop.store %arg1, %0 : !kgen.pointer<index> loc(#loc2)
  kgen.return loc(#loc0)
} loc(#loc0)

// CHECK-LABEL: @mem2reg_valueop_with_initial_undef
kgen.func @mem2reg_valueop_with_initial_undef(%arg0: index, %arg1: index) -> index {
  // CHECK: %[[UNDEF_VAL:.*]] = kgen.param.constant = <#interp.uninitmem> loc(#[[LOC3:.*]])
  // CHECK: debuginfo.value #[[VAR]] #[[REFOF_EXPR]] = %[[UNDEF_VAL]] : index loc(#[[LOC3]])
  // CHECK: debuginfo.value #[[VAR]] #[[REFOF_EXPR]] = %arg0 : index loc(#[[LOC1]])
  // CHECK: debuginfo.value #[[VAR]] #[[REFOF_EXPR]] = %arg1 : index loc(#[[LOC2]])
  %0 = pop.stack_allocation 1 x index loc(#loc0)
  debuginfo.value #local_variable = %0 : !kgen.pointer<index> loc(#loc0)
  %1 = pop.load %0 : !kgen.pointer<index> loc(#loc3) // loading undef
  pop.store %arg0, %0 : !kgen.pointer<index> loc(#loc1)
  pop.store %arg1, %0 : !kgen.pointer<index> loc(#loc2)
  kgen.return %1 : index loc(#loc0)
} loc(#loc0)

// CHECK-LABEL: @mem2reg_valueop_with_initial_value
kgen.func @mem2reg_valueop_with_initial_value(%arg0: index, %arg1: index) {
  // CHECK-NOT: kgen.param.constant = <#interp.uninitmem>
  // CHECK: debuginfo.value #[[VAR]] #[[REFOF_EXPR]] = %arg0 : index loc(#[[LOC1:.*]])
  // CHECK: debuginfo.value #[[VAR]] #[[REFOF_EXPR]] = %arg1 : index loc(#[[LOC2:.*]])
  %0 = pop.stack_allocation 1 x index loc(#loc0)
  pop.store %arg0, %0 : !kgen.pointer<index> loc(#loc0)
  debuginfo.value #local_variable = %0 : !kgen.pointer<index> loc(#loc1)
  pop.store %arg1, %0 : !kgen.pointer<index> loc(#loc2)
  kgen.return loc(#loc0)
} loc(#loc0)

// CHECK-LABEL: @mem2reg_inlined_aliases
kgen.func @mem2reg_inlined_aliases(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
  // CHECK: debuginfo.value #[[VAR_STRUCT]] #[[AGG_REGOF_EXPR]] = %arg0 : index loc(#[[LOC1]])
  // -- Entering callee
  // CHECK: debuginfo.value #[[VAR_STRUCT]] #[[AGG_REGOF_EXPR]] = %arg1 : index loc(#[[LOC2]])
  // CHECK: debuginfo.value #[[VAR_CALLEE]] #[[REFOF_EXPR]] = %arg1 : index loc(#[[LOC101AT2:.*]])
  // -- Entering nested callee
  // CHECK: debuginfo.value #[[VAR_STRUCT]] #[[AGG_REGOF_EXPR]] = %arg2 : index loc(#[[LOC2]])
  // CHECK: debuginfo.value #[[VAR_CALLEE]] #[[REFOF_EXPR]] = %arg2 : index loc(#[[LOC102AT2:.*]])
  // CHECK: debuginfo.value #[[VAR_NESTED_CALLEE]] #[[REFOF_EXPR]] = %arg2 : index loc(#[[LOC201AT102AT2:.*]])
  // -- Returned
  // CHECK: debuginfo.value #[[VAR_STRUCT]] #[[AGG_REGOF_EXPR]] = %arg3 : index loc(#[[LOC3]])
  %0 = pop.stack_allocation 1 x index loc(#loc0)
  debuginfo.value #local_variable_struct #agg_expr = %0 : !kgen.pointer<index> loc(#loc0)
  pop.store %arg0, %0 : !kgen.pointer<index> loc(#loc1)
  debuginfo.value #local_variable_callee = %0 : !kgen.pointer<index> loc(#loc100at2)
  pop.store %arg1, %0 : !kgen.pointer<index> loc(#loc101at2)
  debuginfo.value #local_variable_nested_callee = %0 : !kgen.pointer<index> loc(#loc200at102at2)
  pop.store %arg2, %0 : !kgen.pointer<index> loc(#loc201at102at2)
  debuginfo.value #local_variable_struct #agg_expr = %0 : !kgen.pointer<index> loc(#loc3)
  pop.store %arg3, %0 : !kgen.pointer<index> loc(#loc3)
  kgen.return loc(#loc0)
} loc(#loc0)

// CHECK-LABEL: @mem2reg_inlined_return_value
kgen.func @mem2reg_inlined_return_value(%arg0: index, %arg1: index) {
  // CHECK: debuginfo.value #[[VAR_CALLEE]] #[[REFOF_EXPR]] = %arg1 : index loc(#[[LOC101AT2]])
  // CHECK: debuginfo.value #[[VAR]] #[[REFOF_EXPR]] = %arg0 : index loc(#[[LOC3]])
  %0 = pop.stack_allocation 1 x index loc(#loc100at2)
  debuginfo.value #local_variable_callee = %0 : !kgen.pointer<index> loc(#loc100at2)
  pop.store %arg1, %0 : !kgen.pointer<index> loc(#loc101at2)
  debuginfo.value #local_variable = %0 : !kgen.pointer<index> loc(#loc3)
  pop.store %arg0, %0 : !kgen.pointer<index> loc(#loc3)
  kgen.return loc(#loc3)
} loc(#loc0)

// CHECK: #[[LOC1_RAW:.*]] = loc("foo.mlir":1:0)
// CHECK: #[[LOC2_RAW:.*]] = loc("foo.mlir":2:0)
// CHECK: #[[LOC3_RAW:.*]] = loc("foo.mlir":3:0)
// CHECK: #[[LOC101_RAW:.*]] = loc("foo.mlir":101:0)
// CHECK: #[[LOC102_RAW:.*]] = loc("foo.mlir":102:0)
// CHECK: #[[LOC201_RAW:.*]] = loc("foo.mlir":201:0)
// CHECK: #[[LOC1]] = loc(fused<#[[CALLER_SP]]>[#[[LOC1_RAW]]])
// CHECK: #[[LOC2]] = loc(fused<#[[CALLER_SP]]>[#[[LOC2_RAW]]])
// CHECK: #[[LOC3]] = loc(fused<#[[CALLER_SP]]>[#[[LOC3_RAW]]])
// CHECK: #[[LOC101:.*]] = loc(fused<#[[CALLEE_SP]]>[#[[LOC101_RAW]]])
// CHECK: #[[LOC102:.*]] = loc(fused<#[[CALLEE_SP]]>[#[[LOC102_RAW]]])
// CHECK: #[[LOC201:.*]] = loc(fused<#[[NESTED_CALLEE_SP]]>[#[[LOC201_RAW]]])
// CHECK: #[[LOC101AT2]] = loc(callsite(#[[LOC101]] at #[[LOC2]]))
// CHECK: #[[LOC102AT2]] = loc(callsite(#[[LOC102]] at #[[LOC2]]))
// CHECK: #[[LOC201AT102AT2]] = loc(callsite(#[[LOC201]] at #[[LOC102AT2]]))
