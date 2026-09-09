// RUN: kgen-opt %s -lower-kgen-to-llvm -split-input-file --verify-diagnostics | FileCheck %s

#mem_heap = #interp.memory_handle<32, "0xEFBE">
#mem_stack = #interp.memory_handle<32, "0xADDE">

#mem_global = #interp.memory_handle<32, "0x01020304">
#mem_string = #interp.memory_handle<16, "hello world" string>
#foo = #interp.memory_handle<16, "0x000000000000000008">
#bar = #interp.memory_handle<16, "0x0000">
#bar0 = #interp.memory_handle<16, "0x0000">
#bar1 = #interp.memory_handle<32, "0x00000000">
#large = #interp.memory_handle<16, "0x000102030405060708090001020304050607080900">
#variadic = #interp.memory_handle<8, "0xDEAD">

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=64>} {

// CHECK-LABEL: llvm.func internal @heap
kgen.func @heap() -> !kgen.pointer<i16> {
  // CHECK: %[[ALLOC:.*]] = pop.aligned_alloc %idx32, %idx2 : <i8>
  // CHECK: %[[ALLOC_LLVM:.*]] = builtin.unrealized_conversion_cast %[[ALLOC]] : !kgen.pointer<i8> to !llvm.ptr
  // CHECK: %[[BASE:.*]] = llvm.bitcast %[[ALLOC_LLVM]] : !llvm.ptr to !llvm.ptr
  // CHECK: %[[P0:.*]] = llvm.getelementptr inbounds %[[BASE]][0]
  // CHECK: %[[CST_EFBE:.*]] = llvm.mlir.constant(#M.dense_array<-17, -66> : vector<2xi8>)
  // CHECK: llvm.store %[[CST_EFBE]], %[[P0]] {alignment = 32 :
  // CHECK: %[[RESULT:.*]] = llvm.getelementptr inbounds %[[BASE]][0]
  // CHECK: %[[RESULT_TYPED:.*]] = llvm.bitcast %[[RESULT]]
  %0 = kgen.param.materialize: !kgen.pointer<i16> = <#interp.memref<{[(#mem_heap, heap, [], [])], []}, 0, 0>>
  // CHECK: llvm.return %[[RESULT_TYPED]] : !llvm.ptr
  kgen.return %0 : !kgen.pointer<i16>
}

// CHECK-LABEL: llvm.func internal @stack
kgen.func @stack() {
  // CHECK: %[[ALLOC:.*]] = pop.stack_allocation 2 x i8 align 32
  // CHECK-NEXT: builtin.unrealized_conversion_cast %[[ALLOC]]
  %0 = kgen.param.materialize: !kgen.pointer<i16> = <#interp.memref<{[(#mem_stack, stack, [], [])], []}, 0, 0>>
  kgen.return
}

// CHECK-LABEL: llvm.func internal @persistent
kgen.func @persistent() {
  // CHECK: pop.stack_allocation 2 x i8 align 8
  %0 = kgen.param.materialize: !kgen.pointer<i16> = <#interp.memref<{[(#variadic, persistent, [], [])], []}, 0, 0>>
  kgen.return
}

// CHECK-LABEL: llvm.func internal @stack_shared
kgen.func @stack_shared() {
  // CHECK: %[[ALLOC:.*]] = pop.stack_allocation 2 x i8 align 32
  %0 = kgen.param.materialize: !kgen.pointer<i16> = <#interp.memref<{[(#mem_stack, stack, [], [])], []}, 0, 0>>
  kgen.return
}

// CHECK-LABEL: llvm.func internal @global
kgen.func @global() -> !kgen.pointer<i8> {
  // CHECK: %[[BASE:.*]] = llvm.mlir.addressof [[MEM_GLOBAL:@.*]] : !llvm.ptr
  // CHECK: %[[BASE_OPAQUE:.*]] = llvm.bitcast %[[BASE]]
  // CHECK: %[[RESULT:.*]] = llvm.getelementptr inbounds %[[BASE_OPAQUE]][2]
  // CHECK: %[[RESULT_TYPED:.*]] = llvm.bitcast %[[RESULT]]
  %0 = kgen.param.materialize: !kgen.pointer<i8> = <#interp.memref<{[(#mem_global, const_global, [], [])], []}, 0, 2>>
  // CHECK: llvm.return %[[RESULT_TYPED]]
  kgen.return %0 : !kgen.pointer<i8>
}

// CHECK-LABEL: llvm.func internal @one_heap_use_only
kgen.func @one_heap_use_only() {
  // COM: only materialize #bar1 here (index 1)
  // CHECK: %[[ALLOC:.*]] = pop.aligned_alloc %idx32, %idx4
  // CHECK: %[[PTR_TYPED:.*]] = builtin.unrealized_conversion_cast %[[ALLOC]]
  // CHECK: %[[PTR:.*]] = llvm.bitcast %[[PTR_TYPED]]

  // CHECK: %[[PTEE:.*]] = llvm.getelementptr inbounds %[[PTR]][0]
  // CHECK: %[[CONST:.*]] = llvm.mlir.constant
  // CHECK: llvm.store %[[CONST]], %[[PTEE]]

  // CHECK-NOT: pop.aligned_alloc

  %0 = kgen.param.materialize: !kgen.pointer<pointer<i16>> = <#interp.memref<{[
    (#bar0, heap, [], []),
    (#bar1, heap, [], []),
    (#foo, stack, [], [])
  ], []}, 1, 0>>
  kgen.return
}

// CHECK-LABEL: llvm.func internal @pointer_to_pointer
kgen.func @pointer_to_pointer() {
  // CHECK: %[[ALLOC1:.*]] = pop.stack_allocation 9 x i8 align 16
  // CHECK: %[[PTR1_TYPED:.*]] = builtin.unrealized_conversion_cast %[[ALLOC1]]
  // CHECK: %[[PTR1:.*]] = llvm.bitcast %[[PTR1_TYPED]]

  // CHECK: %[[ALLOC2:.*]] = pop.aligned_alloc %idx16, %idx2
  // CHECK: %[[PTR2_TYPED:.*]] = builtin.unrealized_conversion_cast %[[ALLOC2]]
  // CHECK: %[[PTR2:.*]] = llvm.bitcast %[[PTR2_TYPED]]

  // CHECK: %[[PTR_REGION:.*]] = llvm.getelementptr inbounds %[[PTR1]][0]
  // CHECK: %[[PTEE:.*]] = llvm.getelementptr inbounds %[[PTR2]][0]
  // CHECK: llvm.store %[[PTEE]], %[[PTR_REGION]]

  // CHECK: %[[TAIL:.*]] = llvm.getelementptr inbounds %[[PTR1]][8]
  // CHECK: %[[C8:.*]] = llvm.mlir.constant(8 :
  // CHECK: llvm.store %[[C8]], %[[TAIL]]
  %0 = kgen.param.materialize: !kgen.pointer<pointer<i16>> = <#interp.memref<{[
    (#foo, stack, [(0, 1, 0)], []),
    (#bar, heap, [], [])
  ], []}, 0, 0>>
  kgen.return
}

// CHECK-LABEL: llvm.func internal @string
kgen.func @string() {
  // CHECK: llvm.mlir.addressof [[MEM_STRING:@.*]] :
  %0 = kgen.param.materialize: !kgen.pointer<i8> = <#interp.memref<{[(#mem_string, const_global, [], [])], []}, 0, 0>>
  // COM: Ensure `const_global` handle gets deduplicated.
  // CHECK: llvm.mlir.addressof [[MEM_STRING]] :
  %1 = kgen.param.materialize: !kgen.pointer<i8> = <#interp.memref<{[(#mem_string, const_global, [], []), (#mem_stack, stack, [], [])], []}, 0, 0>>
  kgen.return
}

// CHECK-LABEL: llvm.func internal @long
kgen.func @long() {
  // CHECK: %[[P0:.*]] = llvm.getelementptr inbounds %[[BASEPTR:.*]][0]
  // CHECK: %[[V0:.*]] = llvm.mlir.constant(#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7> : vector<8xi8>)
  // CHECK: llvm.store %[[V0]], %[[P0]]

  // CHECK: %[[P1:.*]] = llvm.getelementptr inbounds %[[BASEPTR]][8]
  // CHECK: %[[V1:.*]] = llvm.mlir.constant(#M.dense_array<8, 9, 0, 1, 2, 3, 4, 5> : vector<8xi8>)
  // CHECK: llvm.store %[[V1]], %[[P1]]

  // CHECK: %[[P2:.*]] = llvm.getelementptr inbounds %[[BASEPTR]][16]
  // CHECK: %[[V2:.*]] = llvm.mlir.constant(#M.dense_array<6, 7, 8, 9> : vector<4xi8>)
  // CHECK: llvm.store %[[V2]], %[[P2]]
  %0 = kgen.param.materialize: !kgen.pointer<i8> = <#interp.memref<{[(#large, stack, [], [])], []}, 0, 0>>

  // CHECK: %[[P3:.*]] = llvm.getelementptr inbounds %[[BASEPTR]][20]
  // CHECK: %[[V3:.*]] = llvm.mlir.constant(0 : i8)
  // CHECK: llvm.store %[[V3]], %[[P3]]
  kgen.return
}

// CHECK-LABEL: llvm.func internal @ptr_inside_blob
kgen.func @ptr_inside_blob() {
  // CHECK: %[[P0:.*]] = llvm.getelementptr inbounds %[[BASEPTR:.*]][0]
  // CHECK: %[[V0:.*]] = llvm.mlir.constant(#M.dense_array<0, 1, 2, 3> : vector<4xi8>)
  // CHECK: llvm.store %[[V0]], %[[P0]]

  // CHECK: %[[P1:.*]] = llvm.getelementptr inbounds %[[BASEPTR]][4]
  // CHECK: %[[V1:.*]] = llvm.mlir.constant(#M.dense_array<4, 5> : vector<2xi8>)
  // CHECK: llvm.store %[[V1]], %[[P1]]

  // CHECK: %[[P2:.*]] = llvm.getelementptr inbounds %[[BASEPTR]][6]
  // CHECK: %[[V2:.*]] = llvm.getelementptr inbounds %{{.*}}[1]
  // CHECK: llvm.store %[[V2]], %[[P2]]

  // COM: 8-byte pointer occupies [6, 7, 8, 9, 0, 1, 2, 3], so next is 4

  // CHECK: %[[P3:.*]] = llvm.getelementptr inbounds %[[BASEPTR]][14]
  // CHECK: %[[V3:.*]] = llvm.mlir.constant(#M.dense_array<4, 5, 6, 7> : vector<4xi8>)
  // CHECK: llvm.store %[[V3]], %[[P3]]

  // CHECK: %[[P4:.*]] = llvm.getelementptr inbounds %[[BASEPTR]][18]
  // CHECK: %[[V4:.*]] = llvm.mlir.constant(#M.dense_array<8, 9> : vector<2xi8>)
  // CHECK: llvm.store %[[V4]], %[[P4]]

  // CHECK: %[[P5:.*]] = llvm.getelementptr inbounds %[[BASEPTR]][20]
  // CHECK: %[[V5:.*]] = llvm.mlir.constant(0 : i8)
  // CHECK: llvm.store %[[V5]], %[[P5]]

  %0 = kgen.param.materialize: !kgen.pointer<i8> = <#interp.memref<{[
    (#large, stack, [(6, 1, 1)], []),
    (#bar, heap, [], [])
  ], []}, 0, 0>>
  kgen.return
}

// CHECK: llvm.mlir.global internal constant [[MEM_GLOBAL]](#M.dense_array<1, 2, 3, 4> : !M.array<4xi8>) {addr_space = 0 : i32, alignment = 32 : i64} : !llvm.array<4 x i8>
// CHECK: llvm.mlir.global internal constant [[MEM_STRING]]("hello world")

}

// -----

#bar2 = #interp.memory_handle<16, "0x0000">
#large2 = #interp.memory_handle<16, "0x000102030405060708090001020304050607080900">

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=16>} {
kgen.func @indirect_stack_ref() {
  // expected-error@+1 {{indirect access to interpreter stack memory}}
  %0 = kgen.param.materialize: !kgen.pointer<i8> = <#interp.memref<{[
    (#large2, stack, [(6, 1, 1)], []),
    (#bar2, stack, [], [])
  ], []}, 0, 0>>
  kgen.return
}
}

// -----

#compress_me = #interp.memory_handle<16, "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDEDEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF">

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=16>} {
// CHECK-LABEL: llvm.func internal @compress_me
kgen.func @compress_me() {
  // CHECK: %[[P0:.*]] = llvm.getelementptr inbounds %[[BASEPTR:.*]][0]
  // CHECK-DAG: %[[VAL:.*]] = llvm.mlir.constant(-1 : i8)
  // CHECK-DAG: %[[SIZE:.*]] = llvm.mlir.constant(32 : i64)
  // CHECK: "llvm.intr.memset"(%[[P0]], %[[VAL]], %[[SIZE]]) <{isVolatile = false}>

  // CHECK: %[[P1:.*]] = llvm.getelementptr inbounds %[[BASEPTR]][32]
  // CHECK: %[[VAL:.*]] = llvm.mlir.constant(#M.dense_array<-34, -34> : vector<2xi8>)
  // CHECK: llvm.store %[[VAL]], %[[P1]]

  // CHECK: %[[P2:.*]] = llvm.getelementptr inbounds %[[BASEPTR:.*]][34]
  // CHECK-DAG: %[[VAL:.*]] = llvm.mlir.constant(-1 : i8)
  // CHECK-DAG: %[[SIZE:.*]] = llvm.mlir.constant(30 : i64)
  // CHECK: "llvm.intr.memset"(%[[P2]], %[[VAL]], %[[SIZE]]) <{isVolatile = false}>
  %0 = kgen.param.materialize: !kgen.pointer<i8> = <#interp.memref<{[(#compress_me, heap, [], [])], []}, 0, 0>>
  kgen.return
}
}

// -----

#memory_handle = #interp.memory_handle<16, "0x00000000">

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=16>} {

// CHECK-LABEL: llvm.func @persistent_global
kgen.func export @persistent_global() -> !kgen.pointer<i8, 2> {
  // CHECK: llvm.mlir.addressof @memory_blob{{.*}} : !llvm.ptr<2>
  %pointer = kgen.param.constant: pointer<i8, 2> = <#interp.memref<{[(#memory_handle, persistent, [], [], 2)], []}, 0, 0>>
  kgen.return %pointer : !kgen.pointer<i8, 2>
}

// CHECK: llvm.mlir.global internal @memory_blob{{.*}}(#M.dense_array<0, 0, 0, 0> : !M.array<4xi8>) {addr_space = 2 : i32, alignment = 16 : i64} : !llvm.array<4 x i8>

}
