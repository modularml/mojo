// RUN: kgen-opt -allow-unregistered-dialect -lower-arg-conventions=mem-to-reg-size-limit=256 -verify-parameters %s | FileCheck %s

// CHECK-LABEL: kgen.func @reg_passable(%arg0: si32 owned, %arg1: si32)
kgen.func @reg_passable(%arg0: si32 owned, %arg1: si32) -> si32 {
  // CHECK: kgen.call @reg_passable(%arg0, %arg1) : (si32 owned, si32)
  %1 = kgen.call @reg_passable(%arg0, %arg1) : (si32 owned, si32) -> si32
  kgen.return %1 : si32
}

// CHECK-LABEL: kgen.func @lower_args(
kgen.func @lower_args(
  // CHECK-SAME: %arg0: index owned,
  // CHECK-SAME: %arg1: !kgen.struct<(index, index)>,
  // CHECK-SAME: %arg2: !kgen.pointer<struct<(index, index) memoryOnly>> imm_mem,
  // CHECK-SAME: %arg3: !kgen.pointer<index> owned
  %arg0: !kgen.pointer<index> owned_in_mem,
  %arg1: !kgen.pointer<struct<(index, index)>> imm_mem,
  %arg2: !kgen.pointer<struct<(index, index) memoryOnly>> imm_mem,
  %arg3: !kgen.pointer<index> owned
) {
  // CHECK: %[[P1:.*]] = pop.stack_allocation 1 x struct<(index, index)>
  // CHECK: pop.store %arg1, %[[P1]] : !kgen.pointer<struct<(index, index)>>
  // CHECK: %[[P0:.*]] = pop.stack_allocation 1 x index
  // CHECK: pop.store %arg0, %[[P0]] : !kgen.pointer<index>
  // CHECK: "some.use"(%[[P0]], %[[P1]])
  "some.use"(%arg0, %arg1) : (!kgen.pointer<index>, !kgen.pointer<struct<(index, index)>>) -> ()
  kgen.return
}

// COM: Ensure that for non GPU targets a register passable struct that exceeds the inline size
// is not promoted.
module attributes {M.target_info = #M.target<triple = "x86_64-unknown-linux-gnu", arch="">} {
  // CHECK-LABEL: kgen.func @size_gated_large_targeted(%arg0: !kgen.pointer<simd<300, ui8>> imm_mem)
  // CHECK: "some.use"(%arg0) : (!kgen.pointer<simd<300, ui8>>) -> ()
  kgen.func @size_gated_large_targeted(%arg0: !kgen.pointer<simd<300, ui8>> imm_mem) {
    "some.use"(%arg0) : (!kgen.pointer<simd<300, ui8>>) -> ()
    kgen.return
  }

  // CHECK-LABEL: kgen.func @byref_res_large_targeted(%arg0: index owned, %arg1: !kgen.pointer<simd<300, ui8>> byref_result) -> !kgen.none
  kgen.func @byref_res_large_targeted(%arg0: index owned, %__result__: !kgen.pointer<simd<300, ui8>> byref_result) -> !kgen.none {
    "somehow.populate"(%__result__) : (!kgen.pointer<simd<300, ui8>>) -> ()
    %none = kgen.param.constant: !kgen.none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}

!lower_args_sig = !kgen.generator<(
  !kgen.pointer<index> owned_in_mem,
  !kgen.pointer<struct<(index, index)>> imm_mem,
  !kgen.pointer<struct<(index, index) memoryOnly>> imm_mem,
  !kgen.pointer<index> owned
) -> ()>

// CHECK: kgen.func @test_lower_args
kgen.func @test_lower_args(%arg0: !lower_args_sig) {
  // CHECK-DAG: %[[P0:.*]] = pop.stack_allocation 1 x index
  %0 = pop.stack_allocation 1 x index
  // CHECK-DAG: %[[P1:.*]] = pop.stack_allocation 1 x struct<(index, index)>
  %1 = pop.stack_allocation 1 x struct<(index, index)>
  // CHECK-DAG: %[[P2:.*]] = pop.stack_allocation 1 x struct<(index, index) memoryOnly>
  %2 = pop.stack_allocation 1 x struct<(index, index) memoryOnly>

  // CHECK-DAG: %[[VAL0:.*]] = pop.load %[[P0]] : !kgen.pointer<index>
  // CHECK-DAG: %[[VAL1:.*]] = pop.load %[[P1]] : !kgen.pointer<struct<(index, index)>>
  // CHECK: kgen.call @lower_args(%[[VAL0]], %[[VAL1]], %[[P2]], %[[P0]]) : (
  // CHECK-SAME: index owned,
  // CHECK-SAME: !kgen.struct<(index, index)>,
  // CHECK-SAME: !kgen.pointer<struct<(index, index) memoryOnly>> imm_mem,
  // CHECK-SAME: !kgen.pointer<index> owned) -> ()
  kgen.call @lower_args(%0, %1, %2, %0) : !lower_args_sig

  // CHECK-DAG: %[[VAL0:.*]] = pop.load %[[P0]] : !kgen.pointer<index>
  // CHECK-DAG: %[[VAL1:.*]] = pop.load %[[P1]] : !kgen.pointer<struct<(index, index)>>
  // CHECK: kgen.call_indirect %arg0(%[[VAL0]], %[[VAL1]], %[[P2]], %[[P0]]) : (
  // CHECK-SAME: index owned,
  // CHECK-SAME: !kgen.struct<(index, index)>,
  // CHECK-SAME: !kgen.pointer<struct<(index, index) memoryOnly>> imm_mem,
  // CHECK-SAME: !kgen.pointer<index> owned) -> ()
  kgen.call_indirect %arg0(%0, %1, %2, %0) : !lower_args_sig
  kgen.return
}

// CHECK-LABEL: kgen.func @byref_res(%arg0: index owned) -> index {
kgen.func @byref_res(%arg0: index owned, %__result__: !kgen.pointer<index> byref_result) -> !kgen.none {
  // CHECK-NEXT: %[[P0:.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: "somehow.populate"(%[[P0]]) : (!kgen.pointer<index>) -> ()
  "somehow.populate"(%__result__) : (!kgen.pointer<index>) -> ()
  %none = kgen.param.constant: !kgen.none = <#kgen.none>
  // CHECK: %[[RES:.*]] = pop.load %[[P0]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.return %[[RES]] : index
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.func @byref_res_reg_passable(%arg0: index owned) -> !kgen.struct<(index, index)> {
kgen.func @byref_res_reg_passable(%arg0: index owned, %__result__: !kgen.pointer<struct<(index, index)>> byref_result) -> !kgen.none {
  // CHECK-NEXT: %[[P0:.*]] = pop.stack_allocation 1 x struct<(index, index)>
  // CHECK-NEXT: "somehow.populate"(%[[P0]]) : (!kgen.pointer<struct<(index, index)>>) -> ()
  "somehow.populate"(%__result__) : (!kgen.pointer<struct<(index, index)>>) -> ()
  %none = kgen.param.constant: !kgen.none = <#kgen.none>
  // CHECK: %[[RES:.*]] = pop.load %[[P0]] : !kgen.pointer<struct<(index, index)>>
  // CHECK-NEXT: kgen.return %[[RES]] : !kgen.struct<(index, index)>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.func @byref_res_mem_only
// CHECK-SAME: -> !kgen.none {
kgen.func @byref_res_mem_only(%arg0: index owned, %__result__: !kgen.pointer<struct<(index, index) memoryOnly>> byref_result) -> !kgen.none {
  // CHECK-NEXT: "somehow.populate"
  "somehow.populate"(%__result__) : (!kgen.pointer<struct<(index, index) memoryOnly>>) -> ()
  %none = kgen.param.constant: !kgen.none = <#kgen.none>
  // CHECK: kgen.return %{{.*}} : !kgen.none
  kgen.return %none : !kgen.none
}

!byref_res_sig = !kgen.generator<(index owned, !kgen.pointer<index> byref_result) -> !kgen.none>
!byref_res_reg_passable_sig = !kgen.generator<(index owned, !kgen.pointer<struct<(index, index)>> byref_result) -> !kgen.none>
!byref_res_mem_only_sig = !kgen.generator<(index owned, !kgen.pointer<struct<(index, index) memoryOnly>> byref_result) -> !kgen.none>

// CHECK-LABEL: kgen.func @test_lower_res
kgen.func @test_lower_res(%arg0: !byref_res_sig, %arg1: !byref_res_reg_passable_sig, %arg2: !byref_res_mem_only_sig, %arg3: index) {
  // CHECK-DAG: %[[P0:.*]] = pop.stack_allocation 1 x index
  %0 = pop.stack_allocation 1 x index
  // CHECK-DAG: %[[P1:.*]] = pop.stack_allocation 1 x struct<(index, index)>
  %1 = pop.stack_allocation 1 x struct<(index, index)>
  // CHECK-DAG: %[[P2:.*]] = pop.stack_allocation 1 x struct<(index, index) memoryOnly>
  %2 = pop.stack_allocation 1 x struct<(index, index) memoryOnly>

  // CHECK: %[[RES0:.*]] = kgen.call @byref_res(%arg3) : (index owned) -> index
  // CHECK-NEXT: pop.store %[[RES0]], %[[P0]] : !kgen.pointer<index>
  kgen.call @byref_res(%arg3, %0) : !byref_res_sig
  // CHECK: %[[RES1:.*]] = kgen.call @byref_res_reg_passable(%arg3) : (index owned) -> !kgen.struct<(index, index)>
  // CHECK-NEXT: pop.store %[[RES1]], %[[P1]] : !kgen.pointer<struct<(index, index)>>
  kgen.call @byref_res_reg_passable(%arg3, %1) : !byref_res_reg_passable_sig
  // CHECK: kgen.call @byref_res_mem_only(%arg3, %2) : (index owned, !kgen.pointer<struct<(index, index) memoryOnly>> byref_result) -> !kgen.none
  // CHECK-NOT: pop.store
  kgen.call @byref_res_mem_only(%arg3, %2) : !byref_res_mem_only_sig

  // CHECK: %[[RES0:.*]] = kgen.call_indirect %arg0(%arg3) : (index owned) -> index
  // CHECK-NEXT: pop.store %[[RES0]], %[[P0]] : !kgen.pointer<index>
  kgen.call_indirect %arg0(%arg3, %0) : !byref_res_sig
  // CHECK: %[[RES1:.*]] = kgen.call_indirect %arg1(%arg3) : (index owned) -> !kgen.struct<(index, index)>
  // CHECK-NEXT: pop.store %[[RES1]], %[[P1]] : !kgen.pointer<struct<(index, index)>>
  kgen.call_indirect %arg1(%arg3, %1) : !byref_res_reg_passable_sig
  // kgen.call_indirect %arg2(%[[P2]], %arg3) : (index owned, !kgen.pointer<struct<(index, index) memoryOnly>> byref_result) -> !kgen.none
  // CHECK-NOT: pop.store
  kgen.call_indirect %arg2(%arg3, %2) : !byref_res_mem_only_sig

  kgen.return
}

!Error = !kgen.struct<(f32)>

// CHECK-LABEL: kgen.func @byref_throws() throws -> !kgen.variant<struct<(f32)>, index>
kgen.func @byref_throws(
  %__error__: !kgen.pointer<!Error> byref_error,
  %__result__: !kgen.pointer<index> byref_result
) throws -> !kgen.scalar<bool> {
  // CHECK: %[[ERROR:.*]] = pop.stack_allocation 1 x struct<(f32)>
  // CHECK: %[[VALUE:.*]] = pop.stack_allocation 1 x index

  // CHECK: %[[FLAG:.*]] = kgen.param.constant: i1 = <?>
  // CHECK: %[[COND:.*]] = pop.cast_from_builtin %[[FLAG]] : i1 to !kgen.scalar<bool>
  %0 = kgen.param.constant: i1 = <?>
  %cflag = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>

  // CHECK: hlcf.if %[[COND]]
  hlcf.if %cflag {
    %1 = kgen.param.constant: scalar<bool> = <true>
    // CHECK: [[ERR:%.*]] = pop.load %[[ERROR]]
    // CHECK-NEXT: [[RESULT:%.*]] = kgen.variant.create [[ERR]], 0
    // CHECK-NEXT: return [[RESULT]]
    kgen.return %1 : !kgen.scalar<bool>
  } else {
    %2 = kgen.param.constant: scalar<bool> = <false>
    // CHECK: [[VAL:%.*]] = pop.load %[[VALUE]]
    // CHECK-NEXT: [[RESULT:%.*]] = kgen.variant.create [[VAL]], 1
    // CHECK-NEXT: return [[RESULT]]
    kgen.return %2 : !kgen.scalar<bool>
  }

  // CHECK:      %[[RES:.*]] = hlcf.if %[[COND]] -> !kgen.variant
  // CHECK-NEXT:   [[ERR:%.*]] = pop.load %[[ERROR]]
  // CHECK-NEXT:   [[RESULT:%.*]] = kgen.variant.create [[ERR]], 0
  // CHECK-NEXT:   hlcf.yield [[RESULT]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[VAL:%.*]] = pop.load %[[VALUE]]
  // CHECK-NEXT:   [[RESULT:%.*]] = kgen.variant.create [[VAL]], 1
  // CHECK-NEXT:   hlcf.yield [[RESULT]]

  // CHECK: kgen.return %[[RES]]
  kgen.return %cflag : !kgen.scalar<bool>
}

!byref_throws_sig = !kgen.generator<(
  !kgen.pointer<!Error> byref_error, !kgen.pointer<index> byref_result
) throws -> !kgen.scalar<bool>>

// CHECK-LABEL: kgen.func @test_byref_throws(
kgen.func @test_byref_throws(%arg0: !byref_throws_sig) {
  // CHECK-NEXT: [[RESULT:%.*]] = pop.stack_allocation 1 x index
  %__result__ = pop.stack_allocation 1 x index
  // CHECK-NEXT: [[ERROR:%.*]] = pop.stack_allocation 1 x struct<(f32)>
  %__error__ = pop.stack_allocation 1 x !Error

  // CHECK: [[RES:%.*]] = kgen.call @byref_throws()
  // CHECK-NEXT: [[IS_ERR:%.*]] = kgen.variant.is [[RES]], 0
  // CHECK-NEXT: hlcf.if [[IS_ERR]] {
  // CHECK-NEXT:   [[ERR:%.*]] = kgen.variant.get [[RES]], 0
  // CHECK-NEXT:   store [[ERR]], [[ERROR]]
  // CHECK-NEXT:   yield
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[VAL:%.*]] = kgen.variant.get [[RES]], 1
  // CHECK-NEXT:   store [[VAL]], [[RESULT]]
  // CHECK-NEXT:   yield
  %res1 = kgen.call @byref_throws(%__error__, %__result__) : (
    !kgen.pointer<!Error> byref_error, !kgen.pointer<index> byref_result
  ) throws -> !kgen.scalar<bool>
  "handle.error"(%res1) : (!kgen.scalar<bool>) -> ()
  "use.result"(%__result__) : (!kgen.pointer<index>) -> ()


  // CHECK: [[RES:%.*]] = kgen.call_indirect %arg0()
  // CHECK-NEXT: [[IS_ERR:%.*]] = kgen.variant.is [[RES]], 0
  // CHECK-NEXT: hlcf.if [[IS_ERR]] {
  // CHECK-NEXT:   [[ERR:%.*]] = kgen.variant.get [[RES]], 0
  // CHECK-NEXT:   store [[ERR]], [[ERROR]]
  // CHECK-NEXT:   yield
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[VAL:%.*]] = kgen.variant.get [[RES]], 1
  // CHECK-NEXT:   store [[VAL]], [[RESULT]]
  // CHECK-NEXT:   yield
  %res2 = kgen.call_indirect %arg0(%__error__, %__result__) : !byref_throws_sig
  "handle.error"(%res2) : (!kgen.scalar<bool>) -> ()
  "use.result"(%__result__) : (!kgen.pointer<index>) -> ()
}

// CHECK-LABEL: @self_result_and_arg
// CHECK-SAME: (%arg0: !kgen.struct<()>, %arg1: i8) -> !kgen.struct<()>
kgen.func @self_result_and_arg(%arg1: !kgen.pointer<struct<()>> imm_mem,
                               %arg2: i8,
                               %arg0: !kgen.pointer<struct<()>> byref_result) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: @call_it_self_result_and_arg
// CHECK-SAME: %arg0: !kgen.struct<()>
kgen.func @call_it_self_result_and_arg(%arg0: !kgen.pointer<struct<()>> imm_mem) -> !kgen.none {
  %0 = pop.stack_allocation 1 x struct<()>
  // CHECK: %[[CST:.*]] = kgen.param.constant: i8 = <4>
  %1 = kgen.param.constant: i8 = <4>
  // CHECK: call @self_result_and_arg(%{{.*}}, %[[CST]]) : (!kgen.struct<()>, i8) -> !kgen.struct<()>
  %2 = kgen.call @self_result_and_arg(%arg0, %1, %0) : (!kgen.pointer<struct<()>> imm_mem, i8, !kgen.pointer<struct<()>> byref_result) -> !kgen.none
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.func @unreachable_byref_result() -> index
kgen.func @unreachable_byref_result(%arg0: !kgen.pointer<index> byref_result) -> !kgen.none {
  // CHECK: loop
  hlcf.loop {
    %none = kgen.param.constant: none = <#kgen.none>
    // CHECK: [[R:%.*]] = pop.load
    // CHECK-NEXT: return [[R]]
    kgen.return %none : !kgen.none
    // CHECK-NEXT: }
  }
  // CHECK-NEXT: unreachable
  kgen.unreachable
}

// CHECK-LABEL: kgen.func @byref_error
// CHECK-SAME: (%arg0: !kgen.generator<(index, !kgen.pointer<struct<(i16) memoryOnly>> byref_result) throws -> (!kgen.scalar<bool>, f16)>
// CHECK-SAME:  %arg1: !kgen.pointer<struct<(i16) memoryOnly>> byref_result) throws -> (!kgen.scalar<bool>, f16)
kgen.func @byref_error(
    %f: !kgen.generator<(index, !kgen.pointer<f16> byref_error, !kgen.pointer<struct<(i16) memoryOnly>> byref_result) throws -> !kgen.scalar<bool>>,
    %err: !kgen.pointer<f16> byref_error,
    %result: !kgen.pointer<struct<(i16) memoryOnly>> byref_result) throws -> !kgen.scalar<bool> {
  // CHECK-NEXT: [[F_ERR:%.*]] = pop.stack_allocation 1 x f16
  // CHECK-NEXT: %idx0
  %idx0 = index.constant 0

  // CHECK-NEXT: [[ERR:%.*]] = pop.stack_allocation 1 x f16
  // CHECK-NEXT: [[VAL:%.*]] = pop.stack_allocation 1 x struct<(i16) memoryOnly>
  %0 = pop.stack_allocation 1 x f16
  %1 = pop.stack_allocation 1 x struct<(i16) memoryOnly>
  // CHECK-NEXT: [[R:%.*]]:2 = kgen.call_indirect %arg0(%idx0, [[VAL]])
  %2 = kgen.call_indirect %f(%idx0, %0, %1) : (index, !kgen.pointer<f16> byref_error, !kgen.pointer<struct<(i16) memoryOnly>> byref_result) throws -> !kgen.scalar<bool>
  // CHECK-NEXT: hlcf.if [[R]]#0 {
  // CHECK-NEXT:   pop.store [[R]]#1, [[ERR]]
  // CHECK-NEXT:   hlcf.yield
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   hlcf.yield
  // CHECK-NEXT: }

  // CHECK-NEXT: [[ERR_RES:%.*]] = pop.load [[F_ERR]]
  // CHECK-NEXT: return [[R]]#0, [[ERR_RES]]
  kgen.return %2 : !kgen.scalar<bool>
}

// CHECK-LABEL: @dont_alter_async_results(%arg0: index, %arg1: !kgen.pointer<index> byref_error, %arg2: !kgen.pointer<index> byref_result) throws|async
kgen.func @dont_alter_async_results(%arg0: !kgen.pointer<index> imm_mem, %arg1: !kgen.pointer<index> byref_error, %arg2: !kgen.pointer<index> byref_result) throws|async {
  kgen.return
}

// CHECK-LABEL: kgen.func @two_call_indirect
kgen.func @two_call_indirect(%arg0: !kgen.generator<(!kgen.pointer<index> byref_result) -> !kgen.none>) {
  %0 = pop.stack_allocation 1 x index
  %1 = pop.stack_allocation 1 x index
  // CHECK: kgen.call_indirect %arg0() : () -> index
  kgen.call_indirect %arg0(%0) : (!kgen.pointer<index> byref_result) -> !kgen.none
  // CHECK: kgen.call_indirect %arg0() : () -> index
  kgen.call_indirect %arg0(%1) : (!kgen.pointer<index> byref_result) -> !kgen.none
  kgen.return
}

#type_value = #kgen.type<typevalue<#kgen.instref<@"IntPackable">>, index> : !kgen.type
#type_value1 = #kgen.type<typevalue<#kgen.instref<@"BoolPackable">>, i1> : !kgen.type

#type_value2 = #kgen.type<pointer<typevalue<#type_value>>, pointer<index>> : !kgen.type
#type_value3 = #kgen.type<pointer<typevalue<#type_value1>>, pointer<i1>> : !kgen.type

// CHECK-LABEL: kgen.func @lower_args1
// CHECK-SAME: (%arg0: index, %arg1: i1, %arg2: index, %arg3: i1)
kgen.func @lower_args1(
 %arg0: !kgen.pointer<struct<(pointer<index>, pointer<i1>, #type_value2, #type_value3) isParamPack>> imm_mem
) -> !kgen.pointer<struct<(pointer<index>, pointer<i1>, #type_value2, #type_value3) isParamPack>> {
    // CHECK: [[V0:%.*]] = pop.stack_allocation 1 x i1
    // CHECK-NEXT: pop.store %arg3, [[V0]] : !kgen.pointer<i1>
    // CHECK-NEXT: [[V1:%.*]] = pop.stack_allocation 1 x index
    // CHECK-NEXT: pop.store %arg2, [[V1]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V2:%.*]] = pop.stack_allocation 1 x i1
    // CHECK-NEXT: pop.store %arg1, [[V2]] : !kgen.pointer<i1>
    // CHECK-NEXT: [[V3:%.*]] = pop.stack_allocation 1 x index
    // CHECK-NEXT: pop.store %arg0, [[V3]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V4:%.*]] = kgen.struct.create([[V3]], [[V2]], [[V1]], [[V0]]) : !kgen.struct<(pointer<index>, pointer<i1>, pointer<index>, pointer<i1>) isParamPack>
    // CHECK-NEXT: [[V5:%.*]] = pop.stack_allocation 1 x struct<(pointer<index>, pointer<i1>, pointer<index>, pointer<i1>) isParamPack>
    // CHECK-NEXT: pop.store [[V4]], [[V5]] : !kgen.pointer<struct<(pointer<index>, pointer<i1>, pointer<index>, pointer<i1>) isParamPack>>
    // CHECK-NEXT: kgen.return [[V5]] : !kgen.pointer<struct<(pointer<index>, pointer<i1>, pointer<index>, pointer<i1>) isParamPack>>
 kgen.return %arg0 : !kgen.pointer<struct<(pointer<index>, pointer<i1>, #type_value2, #type_value3) isParamPack>>
}
// CHECK-LABEL: kgen.func @main
kgen.func @main() {
    // CHECK-NEXT: [[V0:%.*]] = pop.stack_allocation 1 x struct<(pointer<index>, pointer<i1>, pointer<index>, pointer<i1>) isParamPack>
    // CHECK-NEXT: [[V1:%.*]] = pop.load [[V0]] : !kgen.pointer<struct<(pointer<index>, pointer<i1>, pointer<index>, pointer<i1>) isParamPack>>
    // CHECK-NEXT: [[V2:%.*]] = kgen.struct.extract [[V1]][0] : <(pointer<index>, pointer<i1>, pointer<index>, pointer<i1>) isParamPack>
    // CHECK-NEXT: [[V3:%.*]] = kgen.struct.extract [[V1]][1] : <(pointer<index>, pointer<i1>, pointer<index>, pointer<i1>) isParamPack>
    // CHECK-NEXT: [[V4:%.*]] = kgen.struct.extract [[V1]][2] : <(pointer<index>, pointer<i1>, pointer<index>, pointer<i1>) isParamPack>
    // CHECK-NEXT: [[V5:%.*]] = kgen.struct.extract [[V1]][3] : <(pointer<index>, pointer<i1>, pointer<index>, pointer<i1>) isParamPack>
    // CHECK-NEXT: [[V6:%.*]] = pop.load [[V2]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V7:%.*]] = pop.load [[V3]] : !kgen.pointer<i1>
    // CHECK-NEXT: [[V8:%.*]] = pop.load [[V4]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V9:%.*]] = pop.load [[V5]] : !kgen.pointer<i1>
    // CHECK-NEXT: [[V10:%.*]] = kgen.call @lower_args1([[V6]], [[V7]], [[V8]], [[V9]]) : (index, i1, index, i1) -> !kgen.pointer<struct<(pointer<index>, pointer<i1>, pointer<index>, pointer<i1>) isParamPack>>
    %0 = pop.stack_allocation 1 x !kgen.struct<(pointer<index>, pointer<i1>, #type_value2, #type_value3) isParamPack>
    %1 = kgen.call @lower_args1(%0) : (!kgen.pointer<struct<(pointer<index>, pointer<i1>, #type_value2, #type_value3) isParamPack>> imm_mem) -> !kgen.pointer<struct<(pointer<index>, pointer<i1>, #type_value2, #type_value3) isParamPack>>
    kgen.return
}

// CHECK-LABEL: kgen.func @lower_empty
// CHECK-SAME: (%arg0: !kgen.none) -> !kgen.pointer<struct<() isParamPack>>
kgen.func @lower_empty(
 %arg0: !kgen.pointer<struct<() isParamPack>> imm_mem
) -> !kgen.pointer<struct<() isParamPack>> {
 kgen.return %arg0 : !kgen.pointer<struct<() isParamPack>>
}
// CHECK-LABEL: kgen.func @main_empty
kgen.func @main_empty() {
    %0 = pop.stack_allocation 1 x !kgen.struct<() isParamPack>
    // CHECK: kgen.call @lower_empty(%none) : (!kgen.none) -> !kgen.pointer<struct<() isParamPack>>
    %1 = kgen.call @lower_empty(%0) : (!kgen.pointer<struct<() isParamPack>> imm_mem) -> !kgen.pointer<struct<() isParamPack>>
    kgen.return
}

// CHECK-LABEL: kgen.func @none_with_res
// CHECK-SAME: (%arg0: !kgen.none, %arg1: !kgen.pointer<struct<(struct<() isParamPack>) memoryOnly>> byref_result) -> !kgen.none
kgen.func @none_with_res(%arg0: !kgen.pointer<struct<() isParamPack>> owned_in_mem, %arg1: !kgen.pointer<struct<(struct<() isParamPack>) memoryOnly>> byref_result) -> !kgen.none {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
}
// CHECK-LABEL: kgen.func @call_none_with_res
kgen.func @call_none_with_res(%arg0: !kgen.pointer<struct<(variant<struct<() memoryOnly>, index>) memoryOnly>> imm_mem, %arg1: !kgen.pointer<struct<(variant<struct<() memoryOnly>, index>) memoryOnly>> byref_result) -> !kgen.none {
    %2 = pop.stack_allocation 1 x !kgen.struct<() isParamPack>
    %3 = pop.stack_allocation 1 x struct<(struct<() isParamPack>) memoryOnly>
    // CHECK: kgen.call @none_with_res(%none, %{{.*}}) : (!kgen.none, !kgen.pointer<struct<(struct<() isParamPack>) memoryOnly>> byref_result) -> !kgen.none
    %4 = kgen.call @none_with_res(%2, %3) : (!kgen.pointer<struct<() isParamPack>> owned_in_mem, !kgen.pointer<struct<(struct<() isParamPack>) memoryOnly>> byref_result) -> !kgen.none
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.func @recursive_ptr
// CHECK-SAME: (%arg0: !kgen.pointer<none>)
kgen.func @recursive_ptr(%arg0: !kgen.pointer<pointer<none>> imm_mem){
    kgen.return
}
// CHECK-LABEL: kgen.func @call_recursive_ptr
// CHECK: kgen.call @recursive_ptr(%{{.*}}) : (!kgen.pointer<none>) -> ()
kgen.func @call_recursive_ptr(%arg0: !kgen.pointer<pointer<none>> imm_mem){
    kgen.call @recursive_ptr(%arg0) : (!kgen.pointer<pointer<none>> imm_mem) -> ()
    kgen.return
}

// COM: Do not promote types inside a pack not known to be read only.

// CHECK-LABEL: kgen.func @lower_args_no_ptr
// CHECK-SAME: (%arg0: !kgen.pointer<index> mut)
kgen.func @lower_args_no_ptr(
 %arg0: !kgen.struct<(pointer<index>) isParamPack>
) -> !kgen.struct<(pointer<index>) isParamPack> {
    %0 = kgen.struct.extract %arg0[0] : !kgen.struct<(pointer<index>) isParamPack>
    %1 = kgen.param.constant:index = <9>
    pop.store %1, %0 : !kgen.pointer<index>
 kgen.return %arg0 : !kgen.struct<(pointer<index>) isParamPack>
}

// CHECK-LABEL: kgen.func @callIt
kgen.func @callIt() -> index {
    %0 = pop.stack_allocation 1 x index
    %1 = kgen.struct.create(%0) : !kgen.struct<(pointer<index>) isParamPack>
    // CHECK: kgen.call @lower_args_no_ptr(%{{.*}}) : (!kgen.pointer<index> mut) -> !kgen.struct<(pointer<index>) isParamPack>
    %2 = kgen.call @lower_args_no_ptr(%1) : (!kgen.struct<(pointer<index>) isParamPack>) -> !kgen.struct<(pointer<index>) isParamPack>
    %3 = kgen.struct.extract %2[0] : !kgen.struct<(pointer<index>) isParamPack>
    %4 = pop.load %3 : !kgen.pointer<index>
    kgen.return %4 : index
}

//===----------------------------------------------------------------------===//
// `pop.external_call` pack expansion
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @external_call_pack_expand
kgen.func @external_call_pack_expand(%a: !kgen.scalar<si32>, %b: !kgen.scalar<f32>) {
  // CHECK: [[PACK:%.*]] = kgen.struct.create
  %pack = kgen.struct.create(%a, %b) : !kgen.struct<(scalar<si32>, scalar<f32>) isParamPack>
  // CHECK: [[V0:%.*]] = kgen.struct.extract [[PACK]][0]
  // CHECK: [[V1:%.*]] = kgen.struct.extract [[PACK]][1]
  // CHECK: pop.external_call @my_extern([[V0]], [[V1]]) : (!kgen.scalar<si32>, !kgen.scalar<f32>) -> ()
  pop.external_call @my_extern(%pack)
    : (!kgen.struct<(scalar<si32>, scalar<f32>) isParamPack>) -> ()
  kgen.return
}

// Nested packs are recursively flattened. This occurs in practice when Mojo
// variadic args are forwarded to a C variadic function (e.g. fcntl(fd, cmd, *args)).
// CHECK-LABEL: @external_call_nested_pack
kgen.func @external_call_nested_pack(
    %a: !kgen.scalar<si32>, %b: !kgen.scalar<f32>, %c: !kgen.scalar<f64>) {
  %inner = kgen.struct.create(%a, %b) : !kgen.struct<(scalar<si32>, scalar<f32>) isParamPack>
  %outer = kgen.struct.create(%inner, %c)
    : !kgen.struct<(struct<(scalar<si32>, scalar<f32>) isParamPack>, scalar<f64>) isParamPack>
  // The outer pack is expanded, and the inner pack extracted from it is
  // recursively expanded as well, yielding three individual arguments.
  // CHECK: [[INNER:%.*]] = kgen.struct.extract %{{.*}}[0]
  // CHECK: [[A:%.*]] = kgen.struct.extract [[INNER]][0]
  // CHECK: [[B:%.*]] = kgen.struct.extract [[INNER]][1]
  // CHECK: [[C:%.*]] = kgen.struct.extract %{{.*}}[1]
  // CHECK: pop.external_call @my_extern([[A]], [[B]], [[C]])
  // CHECK-SAME: (!kgen.scalar<si32>, !kgen.scalar<f32>, !kgen.scalar<f64>) -> ()
  pop.external_call @my_extern(%outer)
    : (!kgen.struct<(struct<(scalar<si32>, scalar<f32>) isParamPack>, scalar<f64>) isParamPack>) -> ()
  kgen.return
}

//===----------------------------------------------------------------------===//
// 'Never' handling
//===----------------------------------------------------------------------===//

// MOCO-3267: Compiler crash with parametric raise.
// CHECK-LABEL: kgen.func @remove_never_error_slot(%arg0: !kgen.pointer<struct<() memoryOnly>> byref_result) throws
kgen.func @remove_never_error_slot(%err: !kgen.pointer<!kgen.never> byref_error,
                                   %arg0: !kgen.pointer<struct<() memoryOnly>> byref_result) throws -> !kgen.scalar<bool> {
  %1 = kgen.param.constant: scalar<bool> = <false>
  // CHECK: kgen.return{{$}}
  kgen.return %1 : !kgen.scalar<bool>
}

// CHECK: @call_remove_never_error_slot
kgen.func @call_remove_never_error_slot(%err: !kgen.pointer<!kgen.never>,
                                        %arg0: !kgen.pointer<struct<() memoryOnly>>) throws -> !kgen.scalar<bool> {
  // CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: scalar<bool> = <false>
  // CHECK-NEXT: kgen.call @remove_never_error_slot(%arg1)
  %0 = kgen.call @remove_never_error_slot(%err, %arg0) : (!kgen.pointer<!kgen.never> byref_error, !kgen.pointer<struct<() memoryOnly>> byref_result) throws -> !kgen.scalar<bool>
  // CHECK-NEXT: kgen.return [[FALSE]] : !kgen.scalar<bool>
  kgen.return %0 : !kgen.scalar<bool>
}
