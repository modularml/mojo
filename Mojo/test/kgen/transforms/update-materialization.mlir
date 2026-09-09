// RUN: kgen-opt %s -split-input-file -update-materialization | FileCheck %s

// A comptime value holding a pointer into its own storage. `self = 0`
// marks blob 0 as that storage, and its pointer region `(8, 0, 0)` says
// the pointer at byte 8 targets byte 0. The reference to blob 0 is
// dropped so nothing rebuilds it, and the pointer is written relative to
// the destination instead.

#handle = #interp.memory_handle<8, "0x00000000000000000094357700000000">

// CHECK-LABEL: kgen.func @self_pointer
kgen.func @self_pointer() {
  %0 = pop.stack_allocation 1 x struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>) memoryOnly> align 8
  // COM: The materialized value keeps its bytes but loses the self-reference.

  // CHECK: kgen.param.materialize
  // CHECK-NOT: interp.memref
  %1 = kgen.param.materialize: struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>) memoryOnly> = <{ { [0, 0, 0, 0, 0, 0, 0, 0] }, #interp.memref<{[(#handle, stack, [(8, 0, 0)], [])], [], self = 0}, 0, 0> }>

  // CHECK: pop.store
  pop.store %1, %0 : !kgen.pointer<struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>) memoryOnly>>

  // COM: The pointer at byte 8 is set to the address of byte 0 of the same
  // COM: object.

  // CHECK: %[[BYTES:.*]] = pop.pointer.bitcast %[[DEST:.*]] : {{.*}} to !kgen.pointer<i8>
  // CHECK: %[[T:.*]] = index.constant 0
  // CHECK: %[[TARGET:.*]] = pop.offset %[[BYTES]][%[[T]]]
  // CHECK: %[[O:.*]] = index.constant 8
  // CHECK: %[[SLOT:.*]] = pop.offset %[[BYTES]][%[[O]]]
  // CHECK: %[[SLOTP:.*]] = pop.pointer.bitcast %[[SLOT]] : !kgen.pointer<i8> to !kgen.pointer<pointer<i8>>
  // CHECK: pop.store %[[TARGET]], %[[SLOTP]]
  kgen.return
}

// -----

// A pointer into storage that is NOT the value's own: no `self` marker,
// so the value is left exactly as it was.

#handle = #interp.memory_handle<8, "0x0102030400000000">

// CHECK-LABEL: kgen.func @external_pointer
kgen.func @external_pointer() {
  // CHECK-NEXT: %[[DEST:.*]] = pop.stack_allocation 1 x struct<(pointer<none>, scalar<index>) memoryOnly> align 8
  %0 = pop.stack_allocation 1 x struct<(pointer<none>, scalar<index>) memoryOnly> align 8

  // COM: The reference is to unrelated heap storage, so the value is
  // COM: untouched: the memref survives verbatim and no pointer is written
  // COM: after the store.

  // CHECK-NEXT: %[[VAL:.*]] = kgen.param.materialize: struct<(pointer<none>, scalar<index>) memoryOnly> = <{ #interp.memref<{[(#[[HANDLE:.*]], heap, [], [])], []}, 0, 0>, 4 }>
  %1 = kgen.param.materialize: struct<(pointer<none>, scalar<index>) memoryOnly> = <{ #interp.memref<{[(#handle, heap, [], [])], []}, 0, 0>, 4 }>
  // CHECK-NEXT: pop.store %[[VAL]], %[[DEST]] : !kgen.pointer<struct<(pointer<none>, scalar<index>) memoryOnly>>
  pop.store %1, %0 : !kgen.pointer<struct<(pointer<none>, scalar<index>) memoryOnly>>
  // CHECK-NEXT: kgen.return
  kgen.return
}

// -----

// A self-pointer at a non-zero target: the pointer at byte 8 targets
// byte 4 of the same object, so the written address is the destination
// plus 4.

#handle = #interp.memory_handle<8, "0x00000000000000000094357700000000">

// CHECK-LABEL: kgen.func @interior_target
kgen.func @interior_target() {
  // CHECK-NEXT: %[[DEST:.*]] = pop.stack_allocation
  %0 = pop.stack_allocation 1 x struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>) memoryOnly> align 8
  // CHECK-NEXT: %[[VAL:.*]] = kgen.param.materialize: {{.*}} = <{ { [0, 0, 0, 0, 0, 0, 0, 0] }, 0 }>
  %1 = kgen.param.materialize: struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>) memoryOnly> = <{ { [0, 0, 0, 0, 0, 0, 0, 0] }, #interp.memref<{[(#handle, stack, [(8, 0, 4)], [])], [], self = 0}, 0, 4> }>
  // CHECK-NEXT: pop.store %[[VAL]], %[[DEST]]
  pop.store %1, %0 : !kgen.pointer<struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>) memoryOnly>>
  // COM: Both offsets come from the pointer region `(8, 0, 4)`: the pointer
  // COM: at byte 8 is set to the address of byte 4, not byte 0.
  // CHECK-NEXT: %[[BYTES:.*]] = pop.pointer.bitcast %[[DEST]] : {{.*}} to !kgen.pointer<i8>
  // CHECK-NEXT: %[[I4:.*]] = index.constant 4
  // CHECK-NEXT: %[[TARGET:.*]] = pop.offset %[[BYTES]][%[[I4]]] : !kgen.pointer<i8>
  // CHECK-NEXT: %[[I8:.*]] = index.constant 8
  // CHECK-NEXT: %[[SLOT:.*]] = pop.offset %[[BYTES]][%[[I8]]] : !kgen.pointer<i8>
  // CHECK-NEXT: %[[SLOTP:.*]] = pop.pointer.bitcast %[[SLOT]] : !kgen.pointer<i8> to !kgen.pointer<pointer<i8>>
  // CHECK-NEXT: pop.store %[[TARGET]], %[[SLOTP]] : !kgen.pointer<pointer<i8>>
  // CHECK-NEXT: kgen.return
  kgen.return
}

// -----

// COM: Two self-references in one value. Both pointer regions are resolved,
// COM: each pointer taking the address of the byte its own region names.

#handle = #interp.memory_handle<8, "0x0000000000000000000000000000000000000000000000000000000000000000">

// CHECK-LABEL: kgen.func @two_self_pointers
kgen.func @two_self_pointers() {
  // CHECK-NEXT: %[[DEST:.*]] = pop.stack_allocation
  %0 = pop.stack_allocation 1 x struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>, struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>) memoryOnly> align 8

  // COM: Both memrefs are dropped, so neither blob is rebuilt.
  // CHECK-NEXT: %[[VAL:.*]] = kgen.param.materialize: {{.*}} = <{ { [0, 0, 0, 0, 0, 0, 0, 0] }, 0, { [0, 0, 0, 0, 0, 0, 0, 0] }, 0 }>
  %1 = kgen.param.materialize: struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>, struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>) memoryOnly> = <{ { [0, 0, 0, 0, 0, 0, 0, 0] }, #interp.memref<{[(#handle, stack, [(8, 0, 0), (24, 0, 16)], [])], [], self = 0}, 0, 0>, { [0, 0, 0, 0, 0, 0, 0, 0] }, #interp.memref<{[(#handle, stack, [(8, 0, 0), (24, 0, 16)], [])], [], self = 0}, 0, 16> }>

  // CHECK-NEXT: pop.store %[[VAL]], %[[DEST]]
  pop.store %1, %0 : !kgen.pointer<struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>, struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>) memoryOnly>>

  // CHECK-NEXT: %[[BYTES:.*]] = pop.pointer.bitcast %[[DEST]] : {{.*}} to !kgen.pointer<i8>

  // COM: Region (8, 0, 0): the pointer at byte 8 takes the address of byte 0.
  // CHECK-NEXT: %[[I0:.*]] = index.constant 0
  // CHECK-NEXT: %[[T0:.*]] = pop.offset %[[BYTES]][%[[I0]]] : !kgen.pointer<i8>
  // CHECK-NEXT: %[[I8:.*]] = index.constant 8
  // CHECK-NEXT: %[[S0:.*]] = pop.offset %[[BYTES]][%[[I8]]] : !kgen.pointer<i8>
  // CHECK-NEXT: %[[S0P:.*]] = pop.pointer.bitcast %[[S0]] : !kgen.pointer<i8> to !kgen.pointer<pointer<i8>>
  // CHECK-NEXT: pop.store %[[T0]], %[[S0P]] : !kgen.pointer<pointer<i8>>

  // COM: Region (24, 0, 16): the pointer at byte 24 takes the address of byte 16.
  // CHECK-NEXT: %[[I16:.*]] = index.constant 16
  // CHECK-NEXT: %[[T1:.*]] = pop.offset %[[BYTES]][%[[I16]]] : !kgen.pointer<i8>
  // CHECK-NEXT: %[[I24:.*]] = index.constant 24
  // CHECK-NEXT: %[[S1:.*]] = pop.offset %[[BYTES]][%[[I24]]] : !kgen.pointer<i8>
  // CHECK-NEXT: %[[S1P:.*]] = pop.pointer.bitcast %[[S1]] : !kgen.pointer<i8> to !kgen.pointer<pointer<i8>>
  // CHECK-NEXT: pop.store %[[T1]], %[[S1P]] : !kgen.pointer<pointer<i8>>

  // CHECK-NEXT: kgen.return
  kgen.return
}

// -----

// COM: One self-reference alongside one reference to unrelated storage. Only
// COM: the self-reference is resolved: blob 1's memref survives verbatim so it
// COM: is still materialized as its own object, and just one pointer is written.

#own = #interp.memory_handle<8, "0x000000000000000000000000000000000000000000000000">
#ext = #interp.memory_handle<8, "0x0102030405060708">

// CHECK-LABEL: kgen.func @self_and_external
kgen.func @self_and_external() {
  // CHECK-NEXT: %[[DEST:.*]] = pop.stack_allocation
  %0 = pop.stack_allocation 1 x struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>, pointer<none>) memoryOnly> align 8

  // COM: Field 1 (blob 0, the value's own storage) becomes null; field 2
  // COM: (blob 1) keeps its memref.
  // CHECK-NEXT: %[[VAL:.*]] = kgen.param.materialize: {{.*}} = <{ { [0, 0, 0, 0, 0, 0, 0, 0] }, 0, #interp.memref<{{.*}}, 1, 0> }>
  %1 = kgen.param.materialize: struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>, pointer<none>) memoryOnly> = <{ { [0, 0, 0, 0, 0, 0, 0, 0] }, #interp.memref<{[(#own, stack, [(8, 0, 0), (16, 1, 0)], []), (#ext, heap, [], [])], [], self = 0}, 0, 0>, #interp.memref<{[(#own, stack, [(8, 0, 0), (16, 1, 0)], []), (#ext, heap, [], [])], [], self = 0}, 1, 0> }>

  // CHECK-NEXT: pop.store %[[VAL]], %[[DEST]]
  pop.store %1, %0 : !kgen.pointer<struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>, pointer<none>) memoryOnly>>

  // COM: Only region (8, 0, 0) is resolved; (16, 1, 0) targets another blob
  // COM: and is left alone.
  // CHECK-NEXT: %[[BYTES:.*]] = pop.pointer.bitcast %[[DEST]] : {{.*}} to !kgen.pointer<i8>
  // CHECK-NEXT: %[[I0:.*]] = index.constant 0
  // CHECK-NEXT: %[[TARGET:.*]] = pop.offset %[[BYTES]][%[[I0]]] : !kgen.pointer<i8>
  // CHECK-NEXT: %[[I8:.*]] = index.constant 8
  // CHECK-NEXT: %[[SLOT:.*]] = pop.offset %[[BYTES]][%[[I8]]] : !kgen.pointer<i8>
  // CHECK-NEXT: %[[SLOTP:.*]] = pop.pointer.bitcast %[[SLOT]] : !kgen.pointer<i8> to !kgen.pointer<pointer<i8>>
  // CHECK-NEXT: pop.store %[[TARGET]], %[[SLOTP]] : !kgen.pointer<pointer<i8>>

  // CHECK-NEXT: kgen.return
  kgen.return
}

// -----

// COM: One materialized value stored to two destinations. Each destination
// COM: gets its own pointer written, and the reference is dropped once.

#handle = #interp.memory_handle<8, "0x00000000000000000094357700000000">

// CHECK-LABEL: kgen.func @two_stores
kgen.func @two_stores() {
  // CHECK-NEXT: %[[A:.*]] = pop.stack_allocation
  %0 = pop.stack_allocation 1 x struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>) memoryOnly> align 8
  // CHECK-NEXT: %[[B:.*]] = pop.stack_allocation
  %1 = pop.stack_allocation 1 x struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>) memoryOnly> align 8

  // CHECK-NEXT: %[[VAL:.*]] = kgen.param.materialize: {{.*}} = <{ { [0, 0, 0, 0, 0, 0, 0, 0] }, 0 }>
  %2 = kgen.param.materialize: struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>) memoryOnly> = <{ { [0, 0, 0, 0, 0, 0, 0, 0] }, #interp.memref<{[(#handle, stack, [(8, 0, 0)], [])], [], self = 0}, 0, 0> }>

  // COM: First destination.
  // CHECK-NEXT: pop.store %[[VAL]], %[[A]]
  pop.store %2, %0 : !kgen.pointer<struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>) memoryOnly>>
  // CHECK-NEXT: %[[ABYTES:.*]] = pop.pointer.bitcast %[[A]] : {{.*}} to !kgen.pointer<i8>
  // CHECK-NEXT: %[[A0:.*]] = index.constant 0
  // CHECK-NEXT: %[[ATGT:.*]] = pop.offset %[[ABYTES]][%[[A0]]] : !kgen.pointer<i8>
  // CHECK-NEXT: %[[A8:.*]] = index.constant 8
  // CHECK-NEXT: %[[ASLOT:.*]] = pop.offset %[[ABYTES]][%[[A8]]] : !kgen.pointer<i8>
  // CHECK-NEXT: %[[ASLOTP:.*]] = pop.pointer.bitcast %[[ASLOT]] : !kgen.pointer<i8> to !kgen.pointer<pointer<i8>>
  // CHECK-NEXT: pop.store %[[ATGT]], %[[ASLOTP]] : !kgen.pointer<pointer<i8>>

  // COM: Second destination, patched relative to itself rather than the first.
  // CHECK-NEXT: pop.store %[[VAL]], %[[B]]
  pop.store %2, %1 : !kgen.pointer<struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>) memoryOnly>>
  // CHECK-NEXT: %[[BBYTES:.*]] = pop.pointer.bitcast %[[B]] : {{.*}} to !kgen.pointer<i8>
  // CHECK-NEXT: %[[B0:.*]] = index.constant 0
  // CHECK-NEXT: %[[BTGT:.*]] = pop.offset %[[BBYTES]][%[[B0]]] : !kgen.pointer<i8>
  // CHECK-NEXT: %[[B8:.*]] = index.constant 8
  // CHECK-NEXT: %[[BSLOT:.*]] = pop.offset %[[BBYTES]][%[[B8]]] : !kgen.pointer<i8>
  // CHECK-NEXT: %[[BSLOTP:.*]] = pop.pointer.bitcast %[[BSLOT]] : !kgen.pointer<i8> to !kgen.pointer<pointer<i8>>
  // CHECK-NEXT: pop.store %[[BTGT]], %[[BSLOTP]] : !kgen.pointer<pointer<i8>>

  // CHECK-NEXT: kgen.return
  kgen.return
}

// -----

// COM: A use that is not a store. Dropping the reference would leave that use
// COM: holding a null pointer, so the value is left alone entirely.

#handle = #interp.memory_handle<8, "0x00000000000000000094357700000000">

// CHECK-LABEL: kgen.func @non_store_user
kgen.func @non_store_user() -> !kgen.struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>) memoryOnly> {
  // CHECK-NEXT: %[[DEST:.*]] = pop.stack_allocation
  %0 = pop.stack_allocation 1 x struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>) memoryOnly> align 8

  // COM: The memref survives, so nothing is rewritten.
  // CHECK-NEXT: %[[VAL:.*]] = kgen.param.materialize: {{.*}} = <{ { [0, 0, 0, 0, 0, 0, 0, 0] }, #interp.memref<{{.*}}, 0, 0> }>
  %1 = kgen.param.materialize: struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>) memoryOnly> = <{ { [0, 0, 0, 0, 0, 0, 0, 0] }, #interp.memref<{[(#handle, stack, [(8, 0, 0)], [])], [], self = 0}, 0, 0> }>

  // CHECK-NEXT: pop.store %[[VAL]], %[[DEST]]
  pop.store %1, %0 : !kgen.pointer<struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>) memoryOnly>>
  // CHECK-NEXT: kgen.return %[[VAL]]
  kgen.return %1 : !kgen.struct<(struct<(array<8, scalar<si8>>) memoryOnly>, pointer<none>) memoryOnly>
}
