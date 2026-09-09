// RUN: kgen-opt -dead-argument-elimination -split-input-file %s | FileCheck %s

// COM: simple local dead argument.
// CHECK-LABEL: kgen.func @simple(%arg0: index) -> index
kgen.func @simple(%dead_arg: !kgen.scalar<f32> , %live_arg: index) -> index {
  kgen.return %live_arg: index
}

// CHECK-LABEL: kgen.func export @simple_ret_use(%arg0: index) -> index
kgen.func export @simple_ret_use(%arg: index) -> index {
  // CHECK:  %0 = kgen.call @simple(%arg0) : (index) -> index
  %0 = kgen.call @simple(%arg, %arg) : (index, index) -> index
  kgen.return %0: index
}

// -----

// COM: simple dead argument due to callee's argument being dead.
// CHECK-LABEL: kgen.func @f_dead_arg(%arg0: index) -> index
kgen.func @f_dead_arg(%dead_arg: !kgen.scalar<f32>, %live_arg: index) -> index {
  // CHECK: [[V0:%.*]] = kgen.call @g_dead_arg() : () -> index
  %0 = kgen.call @g_dead_arg(%dead_arg): (!kgen.scalar<f32>) -> index
  %1 = index.add %0, %live_arg
  kgen.return %1: index
}

// CHECK-LABEL: kgen.func @g_dead_arg() -> index {
kgen.func @g_dead_arg(%dead_arg: !kgen.scalar<f32>) -> index {
  %0 = index.constant 0
  kgen.return %0: index
}

// CHECK-LABEL: kgen.func export @h_dead_arg(%arg0: index) -> index {
kgen.func export @h_dead_arg(%arg: index) -> index {
  // CHECK: [[V0:%.*]] = kgen.call @g_dead_arg() : () -> index
  %0 = kgen.call @g_dead_arg(%arg) : (index) -> index
  // CHECK: [[V1:%.*]] = kgen.call @f_dead_arg(%arg0) : (index) -> index
  %1 = kgen.call @f_dead_arg(%arg, %arg) : (index, index) -> index
  %2 = index.add %0, %1
  kgen.return %2: index
}

// -----

// COM: f has dead_arg that reaches return/call through operations.
// CHECK-LABEL: kgen.func @f_dead_arg1() -> index {
kgen.func @f_dead_arg1(%dead_arg: index) -> index {
  %0 = index.constant 0
  %1 = index.add %dead_arg, %0
  // CHECK: [[V0:%.*]] = kgen.call @g_dead_arg1() : () -> index
  %2 = kgen.call @g_dead_arg1(%1): (index) -> index
  kgen.return %2: index
}

// CHECK-LABEL: kgen.func @g_dead_arg1() -> index {
kgen.func @g_dead_arg1(%dead_arg: index) -> index {
  %0 = index.constant 0
  kgen.return %0: index
}

// CHECK-LABEL: kgen.func export @h_dead1(%arg0: index) -> index {
kgen.func export @h_dead1(%arg: index) -> index {
  // CHECK: [[V0:%.*]] = kgen.call @g_dead_arg1() : () -> index
  %0 = kgen.call @g_dead_arg1(%arg) : (index) -> index
  kgen.return %0: index
}

// -----

// COM: There is reference to f_maybe_live that is not a kgen.call.
// COM: Mark it as live so that we don't mess up with the API for things like kgen.call_indirect.
// CHECK-LABEL: kgen.func @f_maybe_live(%arg0: index, %arg1: index) -> index {
kgen.func @f_maybe_live(%arg0: index, %arg1: index) -> index {
  kgen.return %arg1: index
}

kgen.func @f_reference() {
  %0 = kgen.param.constant: (index imm, index imm) -> index = <@f_maybe_live>
  kgen.return
}

// -----

// COM: Don't do anything for kgen.create_closure's callee.
// CHECK-LABEL: kgen.func @closure_callee(%arg0: index, %arg1: index) capturing -> index
 kgen.func @closure_callee(%arg0: index, %arg1: index) capturing -> index {
   kgen.return %arg1: index
 }

 kgen.func @closure_caller() -> index {
   %idx0 = index.constant 0
   %0 = kgen.create_closure[(index, index) capturing -> index: @closure_callee](%idx0)
   %1 = kgen.call_indirect %0(%idx0) : (index) capturing -> index
   kgen.return %1: index
 }

// -----

// CHECK-LABEL: kgen.func @test_correctly_handle_pure_ops_callee(%arg0: !pop.array<2, scalar<index>>)
kgen.func @test_correctly_handle_pure_ops_callee(%arg0: !pop.array<2, scalar<index>>, %arg1: !kgen.scalar<index>, %arg2: index) {
  %0 = pop.array.get %arg0[1] : !pop.array<2, scalar<index>>
  %1 = pop.cast_to_builtin %0 : !kgen.scalar<index> to index
  %stack = pop.stack_allocation 1 x index marked
  pop.store %1, %stack : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @test_correctly_handle_pure_ops(%arg0: index, %arg1: !kgen.scalar<index>) {
kgen.func @test_correctly_handle_pure_ops(%arg0: index, %arg1: !kgen.scalar<index>) {
  %0 = index.add %arg0, %arg0
  %1 = index.add %0, %arg0
  %3 = pop.cast_from_builtin %1 : index to !kgen.scalar<index>
  %2 = index.add %1, %0
  %4 = pop.array.create [%3, %arg1] : !pop.array<2, scalar<index>>
  kgen.call @test_correctly_handle_pure_ops_callee(%4, %arg1, %2) : (!pop.array<2, scalar<index>>, !kgen.scalar<index>, index) -> ()
  kgen.return
}

// -----

// COM: An argument passed to an external callee stays live: the callee has no
// COM: call graph edge, so its argument liveness is unknowable.
kgen.func @external_callee(%arg0: index) -> index attributes {external} {
  kgen.unreachable
}

// CHECK-LABEL: kgen.func @calls_external(%arg0: index) -> index {
kgen.func @calls_external(%dead_arg: index, %live_arg: index) -> index {
  // CHECK: kgen.call @external_callee(%arg0) : (index) -> index
  %0 = kgen.call @external_callee(%live_arg) : (index) -> index
  kgen.return %0: index
}
