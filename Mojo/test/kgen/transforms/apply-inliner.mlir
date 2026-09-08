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

// -----

// A recursive generator. Ensure we don't inline forever.

// CHECK-LABEL: kgen.generator @self_applying
kgen.generator @self_applying() -> index {
  // CHECK: kgen.param.constant = <apply(:() -> index @self_applying)>
  %0 = kgen.param.constant = <apply(:() -> index @self_applying)>
  kgen.return %0 : index
}

// -----

// Same, with the cycle spanning two mutually-recursive generators. Neither
// takes a parameter, so nothing can ever fold the cycle away and both are left
// exactly as they are. Substituting one level, as this used to, only swaps one
// un-foldable apply for another.

// CHECK-LABEL: kgen.generator @mutual_a
kgen.generator @mutual_a() -> index {
  // CHECK: kgen.param.constant = <apply(:() -> index @mutual_b)>
  %0 = kgen.param.constant = <apply(:() -> index @mutual_b)>
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @mutual_b
kgen.generator @mutual_b() -> index {
  // CHECK: kgen.param.constant = <apply(:() -> index @mutual_a)>
  %0 = kgen.param.constant = <apply(:() -> index @mutual_a)>
  kgen.return %0 : index
}

// -----

// Expansion that grows without ever repeating an apply. Ensure we don't inline
// forever.

// CHECK-LABEL: kgen.generator @grow
kgen.generator @grow<n>() -> index {
  // `n` is symbolic here, so the recursion cannot fold and the body is left
  // alone. Re-running the pass has to reach the same result, or each run would
  // hand the next a deeper expression to expand.
  // CHECK: kgen.param.constant = <apply(:() -> index @grow<to_builtin(:scalar<index> add(from_builtin(n), 1))>)>
  %0 = kgen.param.constant = <apply(:() -> index @grow<add(n, 1)>)>
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @grow_user
kgen.generator @grow_user() {
  // A concrete parameter folds the guard at every step, so this one does
  // expand, and only the depth bound stops it.
  // CHECK: kgen.param.constant = <apply(:() -> index @grow<33>)>
  kgen.param.constant = <apply(:() -> index @grow<0>)>
  kgen.return
}

// -----

// Recursion that terminates through a ternary base case. The guard only folds
// once the parameter is concrete, so the generator's own body is left alone
// while a concrete apply of it collapses all the way. Expanding the symbolic
// form walks past the base case forever, `n - 1`, `n - 2`, ..., without ever
// repeating an apply.

// CHECK-LABEL: kgen.generator @countdown
kgen.generator @countdown<n>() -> index {
  // CHECK: kgen.param.constant = <cond({{.*}}apply(:() -> index @countdown<to_builtin(:scalar<index> add(from_builtin(n), -1))>))>
  %0 = kgen.param.constant = <cond(eq(n, 0), 0, apply(:() -> index @countdown<add(n, -1)>))>
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @countdown_user
kgen.generator @countdown_user() {
  // CHECK: kgen.param.constant = <0>
  kgen.param.constant = <apply(:() -> index @countdown<4>)>
  kgen.return
}

// -----

// Same, with the recursion spanning two generators.

// CHECK-LABEL: kgen.generator @ping
kgen.generator @ping<n>() -> index {
  // CHECK: kgen.param.constant = <cond({{.*}}apply(:() -> index @pong<to_builtin(:scalar<index> add(from_builtin(n), -1))>))>
  %0 = kgen.param.constant = <cond(eq(n, 0), 0, apply(:() -> index @pong<add(n, -1)>))>
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @pong
kgen.generator @pong<n>() -> index {
  // CHECK: kgen.param.constant = <cond({{.*}}apply(:() -> index @ping<to_builtin(:scalar<index> add(from_builtin(n), -1))>))>
  %0 = kgen.param.constant = <cond(eq(n, 0), 1, apply(:() -> index @ping<add(n, -1)>))>
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @mutual_user
kgen.generator @mutual_user() {
  // CHECK: kgen.param.constant = <1>
  kgen.param.constant = <apply(:() -> index @ping<5>)>
  kgen.return
}

// -----

// All four of these sit in one cycle, but `@b`'s only route back to `@r` runs
// through `@a`, which a depth-first walk can already have finished by the time
// it reaches `@b`. Marking back edges alone leaves `@b` looking non-recursive
// and expands it forever, so this needs the components proper.

kgen.generator @r<n>() -> index {
  // CHECK: kgen.param.constant = <cond({{.*}}@a{{.*}}@b{{.*}})>
  %0 = kgen.param.constant = <cond(eq(n, 0), 0, add(apply(:() -> index @a<add(n, -1)>), apply(:() -> index @b<add(n, -1)>)))>
  kgen.return %0 : index
}

kgen.generator @a<n>() -> index {
  // CHECK: kgen.param.constant = <cond({{.*}}@r{{.*}})>
  %0 = kgen.param.constant = <cond(eq(n, 0), 1, apply(:() -> index @r<add(n, -1)>))>
  kgen.return %0 : index
}

kgen.generator @b<n>() -> index {
  // CHECK: kgen.param.constant = <cond({{.*}}@b2{{.*}})>
  %0 = kgen.param.constant = <cond(eq(n, 0), 2, apply(:() -> index @b2<add(n, -1)>))>
  kgen.return %0 : index
}

kgen.generator @b2<n>() -> index {
  // CHECK: kgen.param.constant = <cond({{.*}}@a{{.*}})>
  %0 = kgen.param.constant = <cond(eq(n, 0), 3, apply(:() -> index @a<add(n, -1)>))>
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @scc_user
kgen.generator @scc_user() {
  // CHECK: kgen.param.constant = <4>
  kgen.param.constant = <apply(:() -> index @r<3>)>
  kgen.return
}
