// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=false" -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=true" -allow-unregistered-dialect | FileCheck %s

kgen.generator @select_pred<*"cmp`2x": scalar<bool>>() -> !kgen.deferred {
  kgen.param.if <*"cmp`2x"> {
    %0 = kgen.param.constant: !kgen.deferred = <#kgen<deferred #index.cmp_predicate<sle>>>
    kgen.return %0 : !kgen.deferred
  } else {
    %0 = kgen.param.constant: !kgen.deferred = <#kgen<deferred #index.cmp_predicate<sgt>>>
    kgen.return %0 : !kgen.deferred
  } {elseIsolated, thenIsolated}
  kgen.unreachable
}

// CHECK-LABEL: @"test_select_pred,cmp=false"
// CHECK: %[[CMP_RESULT:.*]] = index.cmp sgt(%arg0, %arg1)
// CHECK-NEXT: pop.store %[[CMP_RESULT]], %arg2 : !kgen.pointer<i1>

// CHECK-LABEL: @"test_select_pred,cmp=true"
// CHECK: %[[CMP_RESULT:.*]] = index.cmp sle(%arg0, %arg1)
// CHECK-NEXT: pop.store %[[CMP_RESULT]], %arg2 : !kgen.pointer<i1>
kgen.generator @test_select_pred<cmp: scalar<bool>>(%arg0: index, %arg1: index, %arg2: !kgen.pointer<i1> byref_result) throws -> i1 {
  %0 = kgen.param.constant: i1 = <0>
  kgen.param.declare select_pred: <scalar<bool>>() -> !kgen.deferred = <@select_pred>
  kgen.param.apply *"(lifted)apply_0" = [() -> !kgen.deferred: bind_params(:<scalar<bool>>() -> !kgen.deferred select_pred, cmp)]()
  %1 = kgen.deferred "index.cmp"(%arg0, %arg1 : index, index) {pred = #kgen.param.decl.ref<"(lifted)apply_0"> : !kgen.deferred} : i1
  pop.store %1, %arg2 : !kgen.pointer<i1>
  kgen.return %0 : i1
}

// CHECK-LABEL: @test_elaborate_deferred_op
kgen.generator @test_elaborate_deferred_op(%arg0: index, %arg1: index, %arg2: !kgen.pointer<i1> byref_result) throws -> i1 {
  %0 = kgen.param.constant: i1 = <0>
  // CHECK: %[[CMP_RESULT:.*]] = index.cmp sle(%arg0, %arg1)
  %1 = kgen.deferred "index.cmp"(%arg0, %arg1 : index, index) {pred = #kgen<deferred #index.cmp_predicate<sle>> : !kgen.deferred} : i1
  // CHECK-NEXT: pop.store %[[CMP_RESULT]], %arg2 : !kgen.pointer<i1>
  pop.store %1, %arg2 : !kgen.pointer<i1>
  kgen.return %0 : i1
}

// CHECK-LABEL: @test_elaborate_deferred_op_props_only
kgen.generator @test_elaborate_deferred_op_props_only(%arg0: index, %arg1: index, %arg2: !kgen.pointer<i1> byref_result) throws -> i1 {
  %0 = kgen.param.constant: i1 = <0>
  // CHECK: %[[CMP_RESULT:.*]] = index.cmp eq(%arg0, %arg1)
  %1 = kgen.deferred "index.cmp"(%arg0, %arg1 : index, index) {} : i1 properties {pred = #index.cmp_predicate<eq>}
  // CHECK-NEXT: pop.store %[[CMP_RESULT]], %arg2 : !kgen.pointer<i1>
  pop.store %1, %arg2 : !kgen.pointer<i1>
  kgen.return %0 : i1
}

// CHECK-LABEL: @test_elaborate_deferred_op_props_deferred
kgen.generator @test_elaborate_deferred_op_props_deferred(%arg0: index, %arg1: index, %arg2: !kgen.pointer<i1> byref_result) throws -> i1 {
  %0 = kgen.param.constant: i1 = <0>
  // CHECK: %[[CMP_RESULT:.*]] = index.cmp ne(%arg0, %arg1)
  %1 = kgen.deferred "index.cmp"(%arg0, %arg1 : index, index) {} : i1
       properties {pred = #kgen<deferred #index.cmp_predicate<ne>> : !kgen.deferred}
  // CHECK-NEXT: pop.store %[[CMP_RESULT]], %arg2 : !kgen.pointer<i1>
  pop.store %1, %arg2 : !kgen.pointer<i1>
  kgen.return %0 : i1
}

kgen.generator @select_pred_concat<*"pred`2x": struct<(pointer<none>, index)>>() -> !kgen.deferred {
  %0 = kgen.param.constant: !kgen.deferred = <#kgen<attr_ctor_deferred("#index.cmp_predicate<", #kgen<to_string_deferred(#kgen.param.expr<data_to_str, #kgen.param.decl.ref<"pred`2x"> : !kgen.struct<(pointer<none>, index)>, #kgen.param_list<> : !kgen.param_list<struct<(pointer<none>, index)>>> : !kgen.string) elide_type unit>, ">")>>
  kgen.return %0 : !kgen.deferred
}

// CHECK-LABEL: @"test_elaborate_deferred_op_concat,pred
kgen.generator @test_elaborate_deferred_op_concat<pred: struct<(pointer<none>, index)>>(%arg0: index, %arg1: index) -> i1 {
  kgen.param.declare *"get_pred": <struct<(pointer<none>, index)>>() -> !kgen.deferred = <@select_pred_concat>
  kgen.param.apply *"(lifted)apply_0" = [() -> !kgen.deferred: bind_params(:<struct<(pointer<none>, index)>>() -> !kgen.deferred *"get_pred", pred)]()
  // CHECK: %[[CMP_RESULT:.*]] = index.cmp ne(%arg0, %arg1)
  %0 = kgen.deferred "index.cmp"(%arg0, %arg1 : index, index) {pred = #kgen.param.decl.ref<"(lifted)apply_0"> : !kgen.deferred} : i1
  // CHECK-NEXT: kgen.return %[[CMP_RESULT]]
  kgen.return %0 : i1
}

kgen.generator @to_str<mut: i1, *"value`2x": string>(%arg0: !kgen.struct<()>) -> !kgen.struct<(pointer<none>, index)> always_inline {
  %string = kgen.param.constant: string = <*"value`2x">
  %index = kgen.param.constant = <#pop.string_size<*"value`2x">>
  %0 = pop.string.address %string
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
  %2 = kgen.struct.create(%1, %index) : !kgen.struct<(pointer<none>, index)>
  kgen.return %2 : !kgen.struct<(pointer<none>, index)>
}

kgen.generator export @test(%arg0: index, %arg1: index, %arg2: !kgen.pointer<i1> byref_result) throws {
  %0 = kgen.call @test_select_pred<:scalar<bool> true>(%arg0, %arg1, %arg2) : (index, index, !kgen.pointer<i1> byref_result) throws -> i1
  %1 = kgen.call @test_select_pred<:scalar<bool> false>(%arg0, %arg1, %arg2) : (index, index, !kgen.pointer<i1> byref_result) throws -> i1
  %2 = kgen.call @test_elaborate_deferred_op(%arg0, %arg1, %arg2) : (index, index, !kgen.pointer<i1> byref_result) throws -> i1
  %20 = kgen.call @test_elaborate_deferred_op_props_only(%arg0, %arg1, %arg2) : (index, index, !kgen.pointer<i1> byref_result) throws -> i1
  %21 = kgen.call @test_elaborate_deferred_op_props_deferred(%arg0, %arg1, %arg2) : (index, index, !kgen.pointer<i1> byref_result) throws -> i1

  kgen.param.apply *"(lifted)apply_2" = [(!kgen.struct<()>) -> !kgen.struct<(pointer<none>, index)>: @to_str<:i1 0, :string "ne">]({  })
  %3 = kgen.call @test_elaborate_deferred_op_concat<:struct<(pointer<none>, index)> *"(lifted)apply_2">(%arg0, %arg1) : (index, index) -> i1

  kgen.return
}

