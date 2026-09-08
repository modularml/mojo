// RUN: kgen-opt %s -arg-promotion | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func export @exported
// CHECK-SAME: %arg0: !kgen.pointer<index> imm_mem
kgen.func export @exported(%arg0: !kgen.pointer<index> imm_mem) {
  kgen.return
}

// CHECK-LABEL: kgen.func @inreg_args
// CHECK-SAME: %arg0: index owned
kgen.func @inreg_args(%arg0: index owned) {
  kgen.return
}

// CHECK-LABEL: kgen.func @indirectly_referenced
// CHECK-SAME: %arg0: !kgen.pointer<index> imm_mem
kgen.func @indirectly_referenced(%arg0: !kgen.pointer<index> imm_mem) {
  kgen.create_closure[(!kgen.pointer<index> imm_mem) -> (): @indirectly_referenced]()
  kgen.return
}

// CHECK-LABEL: kgen.func @too_large
// CHECK-SAME: %arg0: !kgen.pointer<array<64, f64>> imm_mem
kgen.func @too_large(%arg0: !kgen.pointer<array<64, f64>> imm_mem) {
  kgen.return
}

// CHECK-LABEL: kgen.func @used_as_store_arg
// CHECK-SAME: %arg0: !kgen.pointer<index> imm_mem
kgen.func @used_as_store_arg(%arg0: !kgen.pointer<index> imm_mem) {
  %0 = pop.stack_allocation 1 x pointer<index>
  pop.store %arg0, %0 : !kgen.pointer<pointer<index>>
  kgen.return
}

// CHECK-LABEL: kgen.func @captured_pointer
// CHECK-SAME: %arg0: !kgen.pointer<index> imm_mem
kgen.func @captured_pointer(%arg0: !kgen.pointer<index> imm_mem) {
  %0 = kgen.struct.create(%arg0) : !kgen.struct<(pointer<index>)>
  kgen.return
}

// CHECK-LABEL: kgen.func @capture_pointer
// CHECK-SAME: %arg0: !kgen.pointer<index>
kgen.func @capture_pointer(%arg0: !kgen.pointer<index>) {
  kgen.return
}

// CHECK-LABEL: kgen.func @captured_by_call
// CHECK-SAME: %arg0: !kgen.pointer<index> imm_mem
kgen.func @captured_by_call(%arg0: !kgen.pointer<index> imm_mem) {
  kgen.call @capture_pointer(%arg0) : (!kgen.pointer<index>) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @projection_access
// CHECK-SAME: %arg0: index
kgen.func @projection_access(%arg0: !kgen.pointer<index> mut) {
  %0 = pop.load %arg0 : !kgen.pointer<index>
  pop.store %0, %arg0 : !kgen.pointer<index>

  %1 = pop.pointer.bitcast %arg0 : !kgen.pointer<index> to !kgen.pointer<struct<(index)>>
  %2 = pop.pointer.bitcast %arg0 : !kgen.pointer<index> to !kgen.pointer<array<2, index>>
  %3 = pop.pointer.bitcast %arg0 : !kgen.pointer<index> to !kgen.pointer<union<index>>

  %4 = pop.load %1 : !kgen.pointer<struct<(index)>>
  pop.store %4, %1 : !kgen.pointer<struct<(index)>>

  %5 = pop.load %2 : !kgen.pointer<array<2, index>>
  pop.store %5, %2 : !kgen.pointer<array<2, index>>

  %6 = pop.load %3 : !kgen.pointer<union<index>>
  pop.store %6, %3 : !kgen.pointer<union<index>>

  %7 = kgen.struct.gep %1[0] : <struct<(index)>>
  %8 = pop.load %7 : !kgen.pointer<index>
  pop.store %8, %7 : !kgen.pointer<index>

  %idx1 = index.constant 1
  %9 = pop.array.gep %2[%idx1] : <array<2, index>>
  %10 = pop.load %9 : !kgen.pointer<index>
  pop.store %10, %9 : !kgen.pointer<index>

  %11 = pop.offset %arg0[%idx1] : !kgen.pointer<index>
  %12 = pop.load %11 : !kgen.pointer<index>
  pop.store %12, %11 : !kgen.pointer<index>

  %13 = pop.union.bitcast %3 : <union<index>> as <index>
  %14 = pop.load %13 : !kgen.pointer<index>
  pop.store %14, %13 : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @projection_capture
// CHECK-SAME: %arg0: !kgen.pointer<i64> mut
kgen.func @projection_capture(%arg0: !kgen.pointer<i64> mut) {
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<i64> to !kgen.pointer<index>
  kgen.call @capture_pointer(%0) : (!kgen.pointer<index>) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @call_use
// CHECK-SAME: %arg0: i64
kgen.func @call_use(%arg0: !kgen.pointer<i64> mut) {
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<i64> to !kgen.pointer<index>
  kgen.call @projection_access(%0) : (!kgen.pointer<index> mut) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @borrowed_in_mem(%arg0: index) {
kgen.func @borrowed_in_mem(%arg0: !kgen.pointer<index> imm_mem) {
  // CHECK-NEXT: %0 = pop.stack_allocation 1 x index
  // CHECK-NEXT: store %arg0, %0

  // CHECK-NEXT: load %0
  %0 = pop.load %arg0 : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @owned_in_mem(%arg0: index owned) {
kgen.func @owned_in_mem(%arg0: !kgen.pointer<index> owned_in_mem) {
  // CHECK-NEXT: %0 = pop.stack_allocation 1 x index
  // CHECK-NEXT: store %arg0, %0

  // CHECK-NEXT: load %0
  %0 = pop.load %arg0 : !kgen.pointer<index>
  kgen.return
}

// CHECK: kgen.func @mut(%arg0: index) -> index {
kgen.func @mut(%arg0: !kgen.pointer<index> mut) {
  // CHECK-NEXT: %0 = pop.stack_allocation 1 x index
  // CHECK-NEXT: store %arg0, %0

  // CHECK-NEXT: %1 = pop.load %0
  // CHECK-NEXT: return %1
  kgen.return
}

// CHECK: kgen.func @ref(%arg0: index) -> index {
kgen.func @ref(%arg0: !kgen.pointer<index> ref) {
  // CHECK-NEXT: %0 = pop.stack_allocation 1 x index
  // CHECK-NEXT: store %arg0, %0

  // CHECK-NEXT: %1 = pop.load %0
  // CHECK-NEXT: return %1
  kgen.return
}

// CHECK-LABEL: kgen.func @byref_result() -> index {
kgen.func @byref_result(%arg0: !kgen.pointer<index> byref_result) {
  // CHECK-NEXT: %0 = pop.stack_allocation 1 x index

  // CHECK-NEXT: %idx0
  %idx0 = index.constant 0
  // CHECK-NEXT: store %idx0, %0
  pop.store %idx0, %arg0 : !kgen.pointer<index>

  // CHECK-NEXT: %1 = pop.load %0
  // CHECK-NEXT: return %1
  kgen.return
}

// CHECK-LABEL: kgen.func @byref_error() throws -> index {
kgen.func @byref_error(%arg0: !kgen.pointer<index> byref_error) throws {
  // CHECK-NEXT: %0 = pop.stack_allocation 1 x index

  // CHECK-NEXT: %idx0
  %idx0 = index.constant 0
  // CHECK-NEXT: store %idx0, %0
  pop.store %idx0, %arg0 : !kgen.pointer<index>

  // CHECK-NEXT: %1 = pop.load %0
  // CHECK-NEXT: return %1
  kgen.return
}

// CHECK-LABEL: kgen.func @all_of_them
// CHECK-SAME: (%arg0: i1, %arg1: i2 owned, %arg2: i3, %arg3: i4) throws -> (i3, i4, i5, i6)
kgen.func @all_of_them(
    %arg0: !kgen.pointer<i1> imm_mem,
    %arg1: !kgen.pointer<i2> owned_in_mem,
    %arg2: !kgen.pointer<i3> mut,
    %arg3: !kgen.pointer<i4> ref,
    %arg4: !kgen.pointer<i5> byref_error,
    %arg5: !kgen.pointer<i6> byref_result) throws {
  // CHECK-NEXT: [[I1:%.*]] = pop.stack_allocation 1 x i1
  // CHECK-NEXT: store %arg0, [[I1]]
  // CHECK-NEXT: [[I2:%.*]] = pop.stack_allocation 1 x i2
  // CHECK-NEXT: store %arg1, [[I2]]
  // CHECK-NEXT: [[I3:%.*]] = pop.stack_allocation 1 x i3
  // CHECK-NEXT: store %arg2, [[I3]]
  // CHECK-NEXT: [[I4:%.*]] = pop.stack_allocation 1 x i4
  // CHECK-NEXT: store %arg3, [[I4]]
  // CHECK-NEXT: [[I5:%.*]] = pop.stack_allocation 1 x i5
  // CHECK-NEXT: [[I6:%.*]] = pop.stack_allocation 1 x i6

  // CHECK-NEXT: [[I3_OUT:%.*]] = pop.load [[I3]]
  // CHECK-NEXT: [[I4_OUT:%.*]] = pop.load [[I4]]
  // CHECK-NEXT: [[I5_OUT:%.*]] = pop.load [[I5]]
  // CHECK-NEXT: [[I6_OUT:%.*]] = pop.load [[I6]]
  // CHECK-NEXT: return [[I3_OUT]], [[I4_OUT]], [[I5_OUT]], [[I6_OUT]]
  kgen.return
}

// CHECK-LABEL: kgen.func @only_one_promoted
// CHECK-SAME: %arg0: index, %arg1: index
kgen.func @only_one_promoted(%arg0: !kgen.pointer<index> imm_mem, %arg1: index) {
  kgen.return
}

// CHECK-LABEL: kgen.func @all_of_them_calls
kgen.func @all_of_them_calls() {
  // CHECK-NEXT: [[I1:%.*]] = pop.stack_allocation 1 x i1
  // CHECK-NEXT: [[I2:%.*]] = pop.stack_allocation 1 x i2
  // CHECK-NEXT: [[I3:%.*]] = pop.stack_allocation 1 x i3
  // CHECK-NEXT: [[I4:%.*]] = pop.stack_allocation 1 x i4
  // CHECK-NEXT: [[I5:%.*]] = pop.stack_allocation 1 x i5
  // CHECK-NEXT: [[I6:%.*]] = pop.stack_allocation 1 x i6
  %0 = pop.stack_allocation 1 x i1
  %1 = pop.stack_allocation 1 x i2
  %2 = pop.stack_allocation 1 x i3
  %3 = pop.stack_allocation 1 x i4
  %4 = pop.stack_allocation 1 x i5
  %5 = pop.stack_allocation 1 x i6

  // CHECK-NEXT: [[I1_IN:%.*]] = pop.load [[I1]]
  // CHECK-NEXT: [[I2_IN:%.*]] = pop.load [[I2]]
  // CHECK-NEXT: [[I3_IN:%.*]] = pop.load [[I3]]
  // CHECK-NEXT: [[I4_IN:%.*]] = pop.load [[I4]]
  // CHECK-NEXT: [[R:%.*]]:4 = kgen.call @all_of_them([[I1_IN]], [[I2_IN]], [[I3_IN]], [[I4_IN]]) : (i1, i2 owned, i3, i4) throws -> (i3, i4, i5, i6)
  kgen.call @all_of_them(%0, %1, %2, %3, %4, %5) : (
    !kgen.pointer<i1> imm_mem,
    !kgen.pointer<i2> owned_in_mem,
    !kgen.pointer<i3> mut,
    !kgen.pointer<i4> ref,
    !kgen.pointer<i5> byref_error,
    !kgen.pointer<i6> byref_result) throws -> ()
  // CHECK-NEXT: store [[R]]#0, [[I3]]
  // CHECK-NEXT: store [[R]]#1, [[I4]]
  // CHECK-NEXT: store [[R]]#2, [[I5]]
  // CHECK-NEXT: store [[R]]#3, [[I6]]
  kgen.return
}

// CHECK-LABEL: kgen.func @recursion
// CHECK-SAME: (%arg0: index) -> index
kgen.func @recursion(%arg0: !kgen.pointer<index> mut) {
  // CHECK-NEXT: %0 = pop.stack_allocation 1 x index
  // CHECK-NEXT: store %arg0, %0
  // CHECK-NEXT: %1 = pop.load %0
  // CHECK-NEXT: %2 = kgen.call @recursion(%1) : (index) -> index
  kgen.call @recursion(%arg0) : (!kgen.pointer<index> mut) -> ()
  // CHECK-NEXT: store %2, %0
  // CHECK-NEXT: %3 = pop.load %0
  // CHECK-NEXT: return %3
  kgen.return
}

}
