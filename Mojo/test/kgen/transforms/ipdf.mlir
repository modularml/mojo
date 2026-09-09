// RUN: kgen-opt %s -ipdf -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: kgen.func @no_arguments
// CHECK-SAME: ipdf = []
kgen.func @no_arguments() {
  kgen.return
}

// CHECK-LABEL: kgen.func @one_argument
// CHECK-SAME: ipdf = [""]
kgen.func @one_argument(%arg0: index) {
  kgen.return
}

// CHECK-LABEL: kgen.func @unused_pointer_argument
// CHECK-SAME: ipdf = ["none"]
kgen.func @unused_pointer_argument(%arg0: !kgen.pointer<index>) {
  kgen.return
}

// CHECK-LABEL: kgen.func @read_pointer_argument
// CHECK-SAME: ipdf = ["read"]
kgen.func @read_pointer_argument(%arg0: !kgen.pointer<index>) {
  %0 = pop.load %arg0 : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @write_pointer_argument
// CHECK-SAME: ipdf = ["", "write"]
kgen.func @write_pointer_argument(%arg0: index, %arg1: !kgen.pointer<index>) {
  pop.store %arg0, %arg1 : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @readwrite_pointer_argument
// CHECK-SAME: ipdf = ["", "read,write"]
kgen.func @readwrite_pointer_argument(%arg0: index, %arg1: !kgen.pointer<index>) {
  %0 = pop.load %arg1 : !kgen.pointer<index>
  pop.store %0, %arg1 : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @roundtrip_stack_alloc
// CHECK-SAME: ipdf = ["none"]
kgen.func @roundtrip_stack_alloc(%arg0: !kgen.pointer<index>) {
  %0 = pop.stack_allocation 1 x pointer<index>
  pop.store %arg0, %0 : !kgen.pointer<pointer<index>>
  %1 = pop.load %0 : !kgen.pointer<pointer<index>>
  kgen.return
}

// CHECK-LABEL: kgen.func @roundtrip_stack_alloc_load
// CHECK-SAME: ipdf = ["read"]
kgen.func @roundtrip_stack_alloc_load(%arg0: !kgen.pointer<index>) {
  %0 = pop.stack_allocation 1 x pointer<index>
  pop.store %arg0, %0 : !kgen.pointer<pointer<index>>
  %1 = pop.load %0 : !kgen.pointer<pointer<index>>
  %2 = pop.load %1 : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @aliasing_none
// CHECK-SAME: ipdf = ["none", "none", "none", "none"]
kgen.func @aliasing_none(%arg0: !kgen.pointer<index>, %arg1: !kgen.pointer<index>,
                         %arg2: !kgen.pointer<struct<(index)>>, %arg3: !kgen.pointer<array<2, index>>) {
  %idx1 = index.constant 1
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<index> to !kgen.pointer<none>
  %1 = pop.offset %arg1[%idx1] : !kgen.pointer<index>
  %2 = kgen.struct.gep %arg2[0] : <struct<(index)>>
  %3 = pop.array.gep %arg3[%idx1] : <array<2, index>>
  kgen.return
}

// CHECK-LABEL: kgen.func @aliasing_effect
// CHECK-SAME: ipdf = ["read", "write", "read", "write"]
kgen.func @aliasing_effect(%arg0: !kgen.pointer<index>, %arg1: !kgen.pointer<index>,
                           %arg2: !kgen.pointer<struct<(index)>>, %arg3: !kgen.pointer<array<2, index>>) {
  %idx1 = index.constant 1
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<index> to !kgen.pointer<i64>
  %1 = pop.offset %arg1[%idx1] : !kgen.pointer<index>
  %2 = kgen.struct.gep %arg2[0] : <struct<(index)>>
  %3 = pop.array.gep %arg3[%idx1] : <array<2, index>>

  %4 = pop.load %0 : !kgen.pointer<i64>
  pop.store %idx1, %1 : !kgen.pointer<index>
  %5 = pop.load %2 : !kgen.pointer<index>
  pop.store %5, %3 : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @multiple_store_load
// CHECK-SAME: ipdf = ["read,write", "write"]
kgen.func @multiple_store_load(%arg0: !kgen.pointer<index>, %arg1: !kgen.pointer<index>) {
  %0 = pop.stack_allocation 1 x pointer<index>
  pop.store %arg0, %0 : !kgen.pointer<pointer<index>>
  // COM: read(%arg0)
  %1 = pop.load %arg0 : !kgen.pointer<index>
  %2 = pop.load %0 : !kgen.pointer<pointer<index>>
  pop.store %arg1, %0 : !kgen.pointer<pointer<index>>
  %3 = pop.stack_allocation 1 x pointer<index>
  pop.store %2, %3 : !kgen.pointer<pointer<index>>
  %4 = pop.load %0 : !kgen.pointer<pointer<index>>
  // COM: write(%arg1)
  pop.store %1, %4 : !kgen.pointer<index>
  %5 = pop.load %3 : !kgen.pointer<pointer<index>>
  // COM: write(%arg0)
  pop.store %1, %5 : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @roundtrip_arg
// CHECK-SAME: ipdf = ["read,write", "read,cap"]
kgen.func @roundtrip_arg(%arg0: !kgen.pointer<pointer<index>>, %arg1: !kgen.pointer<index>) {
  pop.store %arg1, %arg0 : !kgen.pointer<pointer<index>>
  %0 = pop.load %arg0 : !kgen.pointer<pointer<index>>
  %1 = pop.load %0 : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @store_bitcast_drop
// CHECK-SAME: ipdf = ["cap"]
kgen.func @store_bitcast_drop(%arg0: !kgen.pointer<index>) {
  %0 = pop.stack_allocation 1 x pointer<index>
  pop.store %arg0, %0 : !kgen.pointer<pointer<index>>
  // COM: Conservatively, if a pointer is loaded through a view, it must be
  // COM: considered captured because we can't track it anymore.
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<pointer<index>> to !kgen.pointer<index>
  %2 = pop.load %1 : !kgen.pointer<index>
  // COM: E.g. we could index_to_pointer and write through to %arg0.
  kgen.return
}

// CHECK-LABEL: kgen.func @store_bitcast_lost
// CHECK-SAME: ipdf = ["read,cap"]
kgen.func @store_bitcast_lost(%arg0: !kgen.pointer<index>) {
  %0 = pop.stack_allocation 1 x pointer<index>
  pop.store %arg0, %0 : !kgen.pointer<pointer<index>>
  // COM: Conservatively, if we store to a pointer through a view, it must be
  // COM: considered captured because we can't track it.
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<pointer<index>> to !kgen.pointer<i8>
  %2 = kgen.param.constant: i8 = <42>
  pop.store %2, %1 : !kgen.pointer<i8>
  %3 = pop.load %0 : !kgen.pointer<pointer<index>>
  // COM: This is a load from an unknown pointer, which applies a load effect on
  // COM: all pointers.
  %4 = pop.load %3 : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @alias_unknown_array_gep
// CHECK-SAME: ipdf = ["write"]
kgen.func @alias_unknown_array_gep(%arg0: !kgen.pointer<index>) {
  %0 = "unknown.ptr"() : () -> !kgen.pointer<array<2, index>>
  %idx1 = index.constant 1
  %1 = pop.array.gep %0[%idx1] : <array<2, index>>
  pop.store %idx1, %1 : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @alias_unknown_struct_gep
// CHECK-SAME: ipdf = ["read"]
kgen.func @alias_unknown_struct_gep(%arg0: !kgen.pointer<index>) {
  %0 = "unknown.ptr"() : () -> !kgen.pointer<struct<(index)>>
  %1 = kgen.struct.gep %0[0] : <struct<(index)>>
  %2 = pop.load %1 : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @alias_unknown_offset
// CHECK-SAME: ipdf = ["write"]
kgen.func @alias_unknown_offset(%arg0: !kgen.pointer<index>) {
  %0 = "unknown.ptr"() : () -> !kgen.pointer<index>
  %idx1 = index.constant 1
  %1 = pop.offset %0[%idx1] : !kgen.pointer<index>
  pop.store %idx1, %1 : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @alias_unknown_bitcast
// CHECK-SAME: ipdf = ["read"]
kgen.func @alias_unknown_bitcast(%arg0: !kgen.pointer<index>) {
  %0 = "unknown.ptr"() : () -> !kgen.pointer<index>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<index> to !kgen.pointer<i8>
  %2 = pop.load %1 : !kgen.pointer<i8>
  kgen.return
}

// CHECK-LABEL: kgen.func @unknown_op_known_ptr
// CHECK-SAME: ipdf = ["read,write,cap", "none"]
kgen.func @unknown_op_known_ptr(%arg0: !kgen.pointer<index>, %arg1: !kgen.pointer<index>) {
  "unknown.op"(%arg0) : (!kgen.pointer<index>) -> ()
  kgen.return
}

kgen.func @unknown_op_unknown_ptr(%arg0: !kgen.pointer<index>) {
  %0 = "unknown.ptr"() : () -> !kgen.pointer<index>
  "unknown.op"(%0) : (!kgen.pointer<index>) -> ()
  kgen.return
}
