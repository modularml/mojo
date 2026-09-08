// The --kgen-stack-reuse-promote-to-global-threshold flag is only registered
// in !MODULAR_PRODUCTION builds (see Mojo/include/Mojo/ToolCommon/CLOptions.h).
// UNSUPPORTED: production
// RUN: kgen-opt %s -stack-reuse -split-input-file --kgen-stack-reuse-promote-to-global-threshold=16 | FileCheck %s

// CHECK-LABEL: @two_overlapping
kgen.func @two_overlapping(%arg0: index, %arg1: index) -> (index, index) {
  // CHECK-NEXT: %[[S0:.*]] = pop.stack_allocation
  // CHECK-NEXT: %[[S1:.*]] = pop.stack_allocation
  %s0 = pop.stack_allocation 1 x index
  %s1 = pop.stack_allocation 1 x index
  pop.store %arg0, %s0 : !kgen.pointer<index>
  pop.store %arg1, %s1 : !kgen.pointer<index>

  // CHECK-NOT: pop.stack_allocation
  %s2 = pop.stack_allocation 1 x index
  pop.store %arg0, %s2 : !kgen.pointer<index>
  // CHECK: %[[V0:.*]] = pop.load %[[S0]]
  %v0 = pop.load %s2 : !kgen.pointer<index>
  pop.store %arg1, %s2 : !kgen.pointer<index>
  // CHECK: %[[V1:.*]] = pop.load %[[S1]]
  %v1 = pop.load %s2 : !kgen.pointer<index>
  // CHECK: return %[[V0]], %[[V1]]
  kgen.return %v0, %v1 : index, index
}

// CHECK-LABEL: @control_flow_if
kgen.func @control_flow_if(%arg0: index, %arg1: index, %arg2: !kgen.scalar<bool>) -> index {
  // CHECK-NEXT: %[[S0:.*]] = pop.stack_allocation
  // CHECK-NOT: pop.stack_allocation
  %s0 = pop.stack_allocation 1 x index
  %s1 = pop.stack_allocation 1 x index

  pop.store %arg0, %s0 : !kgen.pointer<index>
  // CHECK: hlcf.if
  %if = hlcf.if %arg2 -> index {
    // CHECK-NEXT: pop.load
    %0 = pop.load %s0 : !kgen.pointer<index>
    pop.store %0, %s1 : !kgen.pointer<index>
    // CHECK-NEXT: %[[V:.*]] = pop.load %[[S0]]
    %1 = pop.load %s1 : !kgen.pointer<index>
    // CHECK-NEXT: yield %[[V]]
    hlcf.yield %1 : index
  // CHECK: else
  } else {
    // CHECK-NEXT: pop.load
    %0 = pop.load %s0 : !kgen.pointer<index>
    pop.store %0, %s1 : !kgen.pointer<index>
    // CHECK-NEXT: %[[V:.*]] = pop.load %[[S0]]
    %1 = pop.load %s1 : !kgen.pointer<index>
    // CHECK-NEXT: yield %[[V]]
    hlcf.yield %1 : index
  }
  // CHECK: pop.load
  %0 = pop.load %s0 : !kgen.pointer<index>
  pop.store %0, %s1 : !kgen.pointer<index>
  // CHECK-NEXT: %[[V:.*]] = pop.load %[[S0]]
  %1 = pop.load %s1 : !kgen.pointer<index>
  // CHECK-NEXT: return %[[V]]
  kgen.return %1 : index
}

// CHECK-LABEL: @control_flow_if_lifetime
kgen.func @control_flow_if_lifetime(%arg0: index, %arg1: index, %arg2: !kgen.scalar<bool>) -> index {
  // CHECK-NEXT: %[[S0:.*]] = pop.stack_allocation
  // CHECK-NEXT: pop.stack_alloc.lifetime.start(%[[S0]])
  // CHECK-NOT: pop.stack_allocation
  %s0 = pop.stack_allocation 1 x index marked
  pop.stack_alloc.lifetime.start(%s0) : !kgen.pointer<index>
  %s1 = pop.stack_allocation 1 x index marked
  pop.stack_alloc.lifetime.start(%s1) : !kgen.pointer<index>

  pop.store %arg0, %s0 : !kgen.pointer<index>
  // CHECK: hlcf.if
  %if = hlcf.if %arg2 -> index {
    // CHECK-NEXT: pop.load
    %0 = pop.load %s0 : !kgen.pointer<index>
    pop.store %0, %s1 : !kgen.pointer<index>
    // CHECK-NEXT: %[[V:.*]] = pop.load %[[S0]]
    %1 = pop.load %s1 : !kgen.pointer<index>
    // CHECK-NEXT: yield %[[V]]
    hlcf.yield %1 : index
  // CHECK: else
  } else {
    // CHECK-NEXT: pop.load
    %0 = pop.load %s0 : !kgen.pointer<index>
    pop.store %0, %s1 : !kgen.pointer<index>
    // CHECK-NEXT: %[[V:.*]] = pop.load %[[S0]]
    %1 = pop.load %s1 : !kgen.pointer<index>
    // CHECK-NEXT: yield %[[V]]
    hlcf.yield %1 : index
  }
  // CHECK: pop.load
  %0 = pop.load %s0 : !kgen.pointer<index>
  pop.store %0, %s1 : !kgen.pointer<index>
  // CHECK-NEXT: %[[V:.*]] = pop.load %[[S0]]
  %1 = pop.load %s1 : !kgen.pointer<index>
  // CHECK-NEXT: pop.stack_alloc.lifetime.end(%[[S0]])
  // CHECK-NOT: pop.stack_alloc.lifetime.end(%[[S1]])
  pop.stack_alloc.lifetime.end(%s0) : !kgen.pointer<index>
  pop.stack_alloc.lifetime.end(%s1) : !kgen.pointer<index>
  // CHECK-NEXT: return %[[V]]
  kgen.return %1 : index
}

// CHECK-LABEL: @loop_and_gep
kgen.func @loop_and_gep(%arg0: !pop.array<2, index>, %arg1: index) {
  // CHECK-NEXT: %[[S0:.*]] = pop.stack_allocation
  %s0 = pop.stack_allocation 1 x !pop.array<2, index>
  pop.store %arg0, %s0 : !kgen.pointer<array<2, index>>
  // CHECK: hlcf.loop
  hlcf.loop {
    %0 = pop.load %s0 : !kgen.pointer<array<2, index>>
    // CHECK-NOT: pop.stack_allocation
    %s1 = pop.stack_allocation 1 x !pop.array<2, index>
    pop.store %0, %s1 : !kgen.pointer<array<2, index>>
    // CHECK: %[[GEP:.*]] = pop.array.gep %[[S0]][%arg1]
    %1 = pop.array.gep %s1[%arg1] : <array<2, index>>
    // CHECK-NEXT: pop.load %[[GEP]]
    %2 = pop.load %1 : !kgen.pointer<index>
    hlcf.continue
  }
  // CHECK: pop.load
  %0 = pop.load %s0 : !kgen.pointer<array<2, index>>
  // CHECK-NOT: pop.stack_allocation
  %s1 = pop.stack_allocation 1 x !pop.array<2, index>
  pop.store %0, %s1 : !kgen.pointer<array<2, index>>
  // CHECK: %[[GEP:.*]] = pop.array.gep %[[S0]][%arg1]
  %1 = pop.array.gep %s1[%arg1] : <array<2, index>>
  // CHECK-NEXT: pop.load %[[GEP]]
  %2 = pop.load %1 : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: @gep_reconstruct
kgen.func @gep_reconstruct(%arg0: !pop.array<2, index>, %arg1: !pop.array<2, index>) -> (index, index) {
  %idx0 = index.constant 0

  // CHECK: %[[S0:.*]] = pop.stack_allocation
  %s0 = pop.stack_allocation 1 x !pop.array<2, index>
  // CHECK-NEXT: %[[S1:.*]] = pop.stack_allocation
  %s1 = pop.stack_allocation 1 x !pop.array<2, index>

  pop.store %arg0, %s0 : !kgen.pointer<array<2, index>>
  pop.store %arg1, %s1 : !kgen.pointer<array<2, index>>

  // CHECK-NOT: pop.stack_allocation
  %s2 = pop.stack_allocation 1 x !pop.array<2, index>
  %gep = pop.array.gep %s2[%idx0] : <array<2, index>>

  pop.store %arg0, %s2 : !kgen.pointer<array<2, index>>
  // CHECK: %[[GEP0:.*]] = pop.array.gep %[[S0]][%idx0]
  // CHECK-NEXT: %[[R0:.*]] = pop.load %[[GEP0]]
  %r0 = pop.load %gep : !kgen.pointer<index>
  pop.store %arg1, %s2 : !kgen.pointer<array<2, index>>
  // CHECK: %[[GEP1:.*]] = pop.array.gep %[[S1]][%idx0]
  // CHECK-NEXT: %[[R1:.*]] = pop.load %[[GEP1]]
  %r1 = pop.load %gep : !kgen.pointer<index>

  // CHECK-NEXT: return %[[R0]], %[[R1]]
  kgen.return %r0, %r1 : index, index
}

// CHECK-LABEL: @gep_reconstruct_lifetime(
kgen.func @gep_reconstruct_lifetime(%arg0: !pop.array<2, index>, %arg1: !pop.array<2, index>) -> (index, index) {
  %idx0 = index.constant 0

  // CHECK: %[[S0:.*]] = pop.stack_allocation
  // CHECK-NEXT: %[[S1:.*]] = pop.stack_allocation
  // CHECK-NEXT: pop.stack_alloc.lifetime.start(%[[S1]])
  // CHECK-NEXT: pop.stack_alloc.lifetime.start(%[[S0]])
  // CHECK-NEXT: pop.store %arg0, %[[S0]]
  // CHECK-NEXT: pop.store %arg1, %[[S1]]
  // CHECK-NEXT: %[[GEP0:.*]] = pop.array.gep %[[S0]][%idx0]
  // CHECK-NEXT: %[[R0:.*]] = pop.load %[[GEP0]]
  // CHECK-NEXT: %[[GEP1:.*]] = pop.array.gep %[[S1]][%idx0]
  // CHECK-NEXT: %[[R1:.*]] = pop.load %[[GEP1]]
  // CHECK-NEXT: pop.stack_alloc.lifetime.end(%[[S1]])
  // CHECK-NEXT: pop.stack_alloc.lifetime.end(%[[S0]])
  // CHECK-NEXT: return %[[R0]], %[[R1]]
  %s0 = pop.stack_allocation 1 x !pop.array<2, index> marked
  %s1 = pop.stack_allocation 1 x !pop.array<2, index> marked
  pop.stack_alloc.lifetime.start(%s1) : !kgen.pointer<array<2, index>>
  pop.stack_alloc.lifetime.start(%s0) : !kgen.pointer<array<2, index>>
  pop.store %arg0, %s0 : !kgen.pointer<array<2, index>>
  pop.store %arg1, %s1 : !kgen.pointer<array<2, index>>
  %s2 = pop.stack_allocation 1 x !pop.array<2, index>
  %gep = pop.array.gep %s2[%idx0] : <array<2, index>>
  pop.store %arg0, %s2 : !kgen.pointer<array<2, index>>
  %r0 = pop.load %gep : !kgen.pointer<index>
  pop.store %arg1, %s2 : !kgen.pointer<array<2, index>>
  %r1 = pop.load %gep : !kgen.pointer<index>
  pop.stack_alloc.lifetime.end(%s1) : !kgen.pointer<array<2, index>>
  pop.stack_alloc.lifetime.end(%s0) : !kgen.pointer<array<2, index>>
  kgen.return %r0, %r1 : index, index
}

// CHECK-LABEL: @gep_reconstruct_lifetime_fullymixed(
kgen.func @gep_reconstruct_lifetime_fullymixed(%arg0: !pop.array<2, index>, %arg1: !pop.array<2, index>) -> (index, index) {
  %idx0 = index.constant 0

  // CHECK: %[[S0:.*]] = pop.stack_allocation
  // CHECK: %[[S1:.*]] = pop.stack_allocation
  // CHECK-NEXT: pop.stack_alloc.lifetime.start(%[[S1]])
  // CHECK-NEXT: pop.stack_alloc.lifetime.start(%[[S0]])
  // CHECK-NEXT: pop.store %arg0, %[[S0]]
  // CHECK-NEXT: pop.store %arg1, %[[S1]]
  // CHECK-NEXT: %[[GEP:.*]] = pop.array.gep %[[S0]][%idx0]
  // CHECK-NEXT: %[[R0:.*]] = pop.load %[[GEP]]
  // CHECK-NEXT: %[[GEP:.*]] = pop.array.gep %[[S1]][%idx0]
  // CHECK-NEXT: %[[R1:.*]] = pop.load %[[GEP]]
  // CHECK-NEXT: pop.stack_alloc.lifetime.end(%[[S1]])
  // CHECK-NEXT: pop.stack_alloc.lifetime.end(%[[S0]])
  // CHECK-NEXT: return %[[R0]], %[[R1]]
  %s0 = pop.stack_allocation 1 x !pop.array<2, index> marked
  %s1 = pop.stack_allocation 1 x !pop.array<2, index> marked
  %s2 = pop.stack_allocation 1 x !pop.array<2, index>
  pop.stack_alloc.lifetime.start(%s1) : !kgen.pointer<array<2, index>>
  pop.stack_alloc.lifetime.start(%s0) : !kgen.pointer<array<2, index>>
  pop.store %arg0, %s0 : !kgen.pointer<array<2, index>>
  pop.store %arg1, %s1 : !kgen.pointer<array<2, index>>
  %gep = pop.array.gep %s2[%idx0] : <array<2, index>>
  pop.store %arg0, %s2 : !kgen.pointer<array<2, index>>
  %r0 = pop.load %gep : !kgen.pointer<index>
  pop.store %arg1, %s2 : !kgen.pointer<array<2, index>>
  %r1 = pop.load %gep : !kgen.pointer<index>
  pop.stack_alloc.lifetime.end(%s1) : !kgen.pointer<array<2, index>>
  pop.stack_alloc.lifetime.end(%s0) : !kgen.pointer<array<2, index>>
  kgen.return %r0, %r1 : index, index
}

// CHECK-LABEL: @no_alloc_users
kgen.func @no_alloc_users() {
  hlcf.loop {
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @use_in_region
kgen.func @use_in_region(%arg0 : index) {
  // CHECK-NEXT: %0 = pop.stack_allocation
  %0 = pop.stack_allocation 1 x index
  // CHECK-NOT: pop.stack_allocation
  %1 = pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.store %arg0, %0
  pop.store %arg0, %0 : !kgen.pointer<index>
  pop.store %arg0, %1 : !kgen.pointer<index>
  // CHECK-NEXT: hlcf.loop
  hlcf.loop {
    // CHECK-NEXT: pop.load %0
    %2 = pop.load %1 : !kgen.pointer<index>
    // CHECK-NEXT: pop.load %0
    %3 = pop.load %0 : !kgen.pointer<index>
    // CHECK-NEXT: hlcf.break
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @use_crosses_region
kgen.func @use_crosses_region(%arg0: index) {
  // CHECK-NEXT: %0 = pop.stack_allocation
  %0 = pop.stack_allocation 1 x index
  // CHECK-NEXT: %1 = pop.stack_allocation
  %1 = pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.store %arg0, %0
  pop.store %arg0, %0 : !kgen.pointer<index>
  // CHECK-NEXT: pop.store %arg0, %1
  pop.store %arg0, %1 : !kgen.pointer<index>
  // CHECK-NEXT: stage_closure
  kgen.stage_closure = () -> () {
    // CHECK-NEXT: pop.load %1
    pop.load %1 : !kgen.pointer<index>
    // CHECK-NEXT: pop.store %arg0, %0
    pop.store %arg0, %0 : !kgen.pointer<index>
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: @copy_elision_alias
// TODO(#22921): The pass should be able to elide this. Just make sure it
// doesn't crash for now.
kgen.func @copy_elision_alias() {
  %0 = pop.stack_allocation 1 x struct<(struct<(index)>)>
  %1 = kgen.struct.gep %0[0] : <struct<(struct<(index)>)>>
  %2 = pop.load %1 : !kgen.pointer<struct<(index)>>
  %3 = pop.stack_allocation 1 x struct<(index)>
  pop.store %2, %3 : !kgen.pointer<struct<(index)>>
  pop.load %3 : !kgen.pointer<struct<(index)>>
  kgen.return
}

// CHECK-LABEL: @function_boundary
kgen.func @function_boundary(%arg0: index) -> index {
  // CHECK: [[S0:%.*]] = pop.stack_allocation
  %0 = pop.stack_allocation 1 x index
  pop.store %arg0, %0 : !kgen.pointer<index>
  %1 = pop.stack_allocation 1 x index
  pop.store %arg0, %1 : !kgen.pointer<index>
  // CHECK: co.execute
  co.execute : index {
    // CHECK: [[S1:%.*]] = pop.stack_allocation
    %3 = pop.stack_allocation 1 x index
    pop.store %arg0, %3 : !kgen.pointer<index>
    %4 = pop.stack_allocation 1 x index
    pop.store %arg0, %4 : !kgen.pointer<index>
    // CHECK: [[R1:%.*]] = pop.load [[S1]]
    %5 = pop.load %4 : !kgen.pointer<index>
    // CHECK-NEXT: return [[R1]]
    kgen.return %5 : index
  }
  // CHECK: [[R0:%.*]] = pop.load [[S0]]
  %2 = pop.load %1 : !kgen.pointer<index>
  // CHECK-NEXT: return [[R0]]
  kgen.return %2 : index
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="">} {
  // CHECK-LABEL: @large_constant_promotion
  kgen.func @large_constant_promotion(%arg0: index) -> () {
      // CHECK-NEXT: %0 = pop.global_constant: array<2, struct<(scalar<ui64>, scalar<ui64>)>> = <[{ 0, 1 }, { 2, 3 }]>
      // CHECK-NEXT: %1 = pop.array.gep %0[%arg0] : <array<2, struct<(scalar<ui64>, scalar<ui64>)>>>
      // CHECK-NEXT: %2 = pop.load %1 : !kgen.pointer<struct<(scalar<ui64>, scalar<ui64>)>>
      // CHECK-NEXT: kgen.return
      %array = kgen.param.constant: array<2, struct<(scalar<ui64>, scalar<ui64>)>> = <[{ 0, 1 }, { 2, 3 }]>
      %21 = pop.stack_allocation 1 x array<2, struct<(scalar<ui64>, scalar<ui64>)>> marked
      pop.stack_alloc.lifetime.start(%21) : !kgen.pointer<array<2, struct<(scalar<ui64>, scalar<ui64>)>>>
      pop.store %array, %21 : !kgen.pointer<array<2, struct<(scalar<ui64>, scalar<ui64>)>>>
      %23 = pop.array.gep %21[%arg0] : <array<2, struct<(scalar<ui64>, scalar<ui64>)>>>
      %24 = pop.load %23 : !kgen.pointer<struct<(scalar<ui64>, scalar<ui64>)>>
      pop.stack_alloc.lifetime.end(%21) : !kgen.pointer<array<2, struct<(scalar<ui64>, scalar<ui64>)>>>
      kgen.return
  }

  // Cannot convert the constant of strings into pop.global_constant
  // CHECK: @negative_large_constant_with_strings
  kgen.func @negative_large_constant_with_strings(%arg0: index) -> index {
    // CHECK-NEXT: kgen.param.constant
    %array = kgen.param.constant: array<8, string> = <["0.0", "0.001953125", "0.00390625", "0.005859375", "0.0078125", "0.009765625", "0.01171875", "0.013671875"]>
    %3 = pop.stack_allocation 1 x array<8, string> marked
    pop.stack_alloc.lifetime.start(%3) : !kgen.pointer<array<8, string>>
    pop.store %array, %3 : !kgen.pointer<array<8, string>>
    %4 = pop.array.gep %3[%arg0] : <array<8, string>>
    %5 = pop.load %4 : !kgen.pointer<string>
    pop.stack_alloc.lifetime.end(%3) : !kgen.pointer<array<8, string>>
    %6 = pop.string.address %5
    %7 = pop.pointer.bitcast %6 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
    %8 = pop.string.size %5
    kgen.return %8 : index
  }
}
