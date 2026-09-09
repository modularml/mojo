// RUN: kgen-opt -verify-parameters -apply-inliner -split-input-file %s | FileCheck %s

kgen.generator @trivial<T: type>(%arg0: !kgen.param<T>) -> !kgen.param<T> {
  kgen.return %arg0 : !kgen.param<T>
}

// CHECK-LABEL: kgen.generator @trivial_exprs
kgen.generator @trivial_exprs() {
  // CHECK-NEXT: constant = <2>
  kgen.param.constant = <apply(:(index) -> index @trivial<:type index>, 2)>
  kgen.return
}

// -----

kgen.generator @fwd_reg<T: type>(%arg0: !kgen.param<T>) -> !kgen.param<T> {
  kgen.return %arg0 : !kgen.param<T>
}

kgen.generator @fwd_reg_byref_result_store_first<T: type>(%arg0: !kgen.param<T>, %arg1: !kgen.pointer<T> byref_result) -> !kgen.none {
  pop.store %arg0, %arg1 : !kgen.pointer<T>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator @fwd_reg_byref_result_store_second<T: type>(%arg0: !kgen.param<T>, %arg1: !kgen.pointer<T> byref_result) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  pop.store %arg0, %arg1 : !kgen.pointer<T>
  kgen.return %none : !kgen.none
}

kgen.generator @reg_constant<T: type, value: !kgen.param<T>>() -> !kgen.param<T> {
  %0 = kgen.param.constant: !kgen.param<T> = <value>
  kgen.return %0 : !kgen.param<T>
}

// CHECK-LABEL: @test_param_inline
kgen.generator @test_param_inline<param>() {
  // CHECK-NEXT: <1>
  kgen.param.constant = <apply(:(index) -> index @fwd_reg<:type index>, 1)>
  // CHECK-NEXT: <3>
  kgen.param.constant = <apply_result_slot(:(index, !kgen.pointer<index> byref_result) -> !kgen.none @fwd_reg_byref_result_store_first<:type index>, 3)>
  // CHECK-NEXT: <4>
  kgen.param.constant = <apply_result_slot(:(index, !kgen.pointer<index> byref_result) -> !kgen.none @fwd_reg_byref_result_store_second<:type index>, 4)>
  // CHECK-NEXT: <5>
  kgen.param.constant = <apply(:() -> index @reg_constant<:type index, 5>)>
  // CHECK-NEXT: <param>
  kgen.param.constant = <apply(:() -> index @reg_constant<:type index, param>)>
  kgen.return
}

// debug versions
#subprogram = #debuginfo.subprogram<sourceName = <"sp">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#var_paramref = #debuginfo.local_variable<scope = #subprogram, name = "paramref"> : !debuginfo.unresolved<!kgen.param<T>>
#var_pointer = #debuginfo.local_variable<scope = #subprogram, name = "pointer"> : !debuginfo.unresolved<!kgen.pointer<T>>
#var_none = #debuginfo.local_variable<scope = #subprogram, name = "none"> : !debuginfo.unresolved<!kgen.none>

kgen.generator @fwd_reg_debug<T: type>(%arg0: !kgen.param<T>) -> !kgen.param<T> {
  debuginfo.value #var_paramref = %arg0 : !kgen.param<T>
  kgen.return %arg0 : !kgen.param<T>
}

kgen.generator @fwd_reg_byref_result_store_first_debug<T: type>(%arg0: !kgen.param<T>, %arg1: !kgen.pointer<T> byref_result) -> !kgen.none {
  debuginfo.value #var_paramref = %arg0 : !kgen.param<T>
  debuginfo.value #var_pointer = %arg1 : !kgen.pointer<T>
  pop.store %arg0, %arg1 : !kgen.pointer<T>
  %none = kgen.param.constant: none = <#kgen.none>
  debuginfo.value #var_none = %none : !kgen.none
  kgen.return %none : !kgen.none
}

kgen.generator @fwd_reg_byref_result_store_second_debug<T: type>(%arg0: !kgen.param<T>, %arg1: !kgen.pointer<T> byref_result) -> !kgen.none {
  debuginfo.value #var_paramref = %arg0 : !kgen.param<T>
  debuginfo.value #var_pointer = %arg1 : !kgen.pointer<T>
  %none = kgen.param.constant: none = <#kgen.none>
  debuginfo.value #var_none = %none : !kgen.none
  pop.store %arg0, %arg1 : !kgen.pointer<T>
  kgen.return %none : !kgen.none
}

kgen.generator @reg_constant_debug<T: type, value: !kgen.param<T>>() -> !kgen.param<T> {
  %0 = kgen.param.constant: !kgen.param<T> = <value>
  debuginfo.value #var_paramref = %0 : !kgen.param<T>
  kgen.return %0 : !kgen.param<T>
}

// CHECK-LABEL: @test_param_inline_debug
kgen.generator @test_param_inline_debug<param>() {
  // CHECK-NEXT: <1>
  kgen.param.constant = <apply(:(index) -> index @fwd_reg_debug<:type index>, 1)>
  // CHECK-NEXT: <3>
  kgen.param.constant = <apply_result_slot(:(index, !kgen.pointer<index> byref_result) -> !kgen.none @fwd_reg_byref_result_store_first_debug<:type index>, 3)>
  // CHECK-NEXT: <4>
  kgen.param.constant = <apply_result_slot(:(index, !kgen.pointer<index> byref_result) -> !kgen.none @fwd_reg_byref_result_store_second_debug<:type index>, 4)>
  // CHECK-NEXT: <5>
  kgen.param.constant = <apply(:() -> index @reg_constant_debug<:type index, 5>)>
  // CHECK-NEXT: <param>
  kgen.param.constant = <apply(:() -> index @reg_constant_debug<:type index, param>)>
  kgen.return
}
