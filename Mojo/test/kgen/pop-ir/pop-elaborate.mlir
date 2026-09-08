// RUN: kgen-opt -elaborate-generators="use-parametric-interpret=false" --kgen-print-inline-type-values %s | FileCheck %s
// RUN: kgen-opt -elaborate-generators="use-parametric-interpret=true" --kgen-print-inline-type-values %s | FileCheck %s

// NOTE: Bytes are encoded backwards in resource blobs.

#mem = #interp.memory_handle<32, "0xADDEEFBE">
#stack = #interp.memory_handle<32, "0xADDE">
#some_ptr = #interp.memory_handle<32, "0xEFBE">
#pointer = #interp.memory_handle<64, "0x000000000000000000F2052A01000000">
#string = #interp.memory_handle<16, "hello world" string>
// CHECK-DAG: [[RETURN_HEAP:#.*]] = #interp.memory_handle<8192, "0xADDEEFBE">
// CHECK-DAG: [[MODIFY_STACK_MEM:#.*]] = #interp.memory_handle<32, "0x0000">
// COM: 0x77359400 -> 2000000000, the base stack address
// CHECK-DAG: [[RETURN_POINTER:#.*]] = #interp.memory_handle<64, "0x00000000000000000094357700000000">
// CHECK-DAG: [[RETURN_POINTER_1:#.*]] = #interp.memory_handle<32, "0xEFBE">
// CHECK-DAG: [[FREED_CONCRETE_MEM:#.*]]:
// CHECK-DAG: [[STRING:#.*]] = #interp.memory_handle<16, "hello world" string>
// CHECK-DAG: [[GLOBAL_ALLOC:#.*]] = #interp.memory_handle<16, "0x00000000">

kgen.generator @store_load_pointer(%arg0: i32) -> i32 {
  %0 = pop.stack_allocation 1 x i32
  pop.store %arg0, %0 : !kgen.pointer<i32>
  %1 = pop.stack_allocation 1 x !kgen.pointer<i32>
  pop.store %0, %1 : !kgen.pointer<pointer<i32>>
  %2 = pop.load %1 : !kgen.pointer<pointer<i32>>
  %3 = pop.load %2 : !kgen.pointer<i32>
  kgen.return %3 : i32
}

kgen.generator @store_load<T: type>(%arg0: !kgen.param<T>) -> !kgen.param<T> {
  %0 = pop.stack_allocation 1 x T
  pop.store %arg0, %0 : !kgen.pointer<T>
  %1 = pop.load %0 : !kgen.pointer<T>
  kgen.return %1 : !kgen.param<T>
}

kgen.generator @i24_pair_bitcast(%arg0: !pop.array<2, i24>) -> i64 {
  %0 = pop.stack_allocation 2 x i24
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<i24> to !kgen.pointer<array<2, i24>>
  pop.store %arg0, %1 : !kgen.pointer<array<2, i24>>
  %2 = pop.pointer.bitcast %0 : !kgen.pointer<i24> to !kgen.pointer<i64>
  %3 = pop.load %2 : !kgen.pointer<i64>
  kgen.return %3 : i64
}

kgen.generator @bitcast<I: type, O: type>(%arg0: !kgen.param<I>) -> !kgen.param<O> {
  %0 = pop.stack_allocation 1 x I
  pop.store %arg0, %0 : !kgen.pointer<I>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<I> to !kgen.pointer<O>
  %2 = pop.load %1 : !kgen.pointer<O>
  kgen.return %2 : !kgen.param<O>
}

// COM: Store the variant and sneakily read its discriminator's raw value.
kgen.generator @variant_bitcast_discr(%arg0: !kgen.variant<i32, i64>) -> i8 {
  %0 = pop.stack_allocation 1 x !kgen.variant<i32, i64>
  pop.store %arg0, %0 : !kgen.pointer<variant<i32, i64>>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<variant<i32, i64>> to !kgen.pointer<struct<(i64, i8)>>
  %2 = kgen.struct.gep %1[1] : <struct<(i64, i8)>>
  %3 = pop.load %2 : !kgen.pointer<i8>
  kgen.return %3 : i8
}

kgen.generator @array_gep_load<I>(%arg0: !pop.array<3, i24>) -> i24 {
  %0 = kgen.param.constant = <I>
  %1 = pop.stack_allocation 1 x !pop.array<3, i24>
  pop.store %arg0, %1 : !kgen.pointer<array<3, i24>>
  %2 = pop.array.gep %1[%0] : <array<3, i24>>
  %3 = pop.load %2 : !kgen.pointer<i24>
  kgen.return %3 : i24
}

kgen.generator @struct_gep_load(%arg0: !kgen.struct<(i8, i16, i32)>) -> i32 {
  %1 = pop.stack_allocation 1 x !kgen.struct<(i8, i16, i32)>
  pop.store %arg0, %1 : !kgen.pointer<struct<(i8, i16, i32)>>
  %2 = kgen.struct.gep %1[2] : <struct<(i8, i16, i32)>>
  %3 = pop.load %2 : !kgen.pointer<i32>
  kgen.return %3 : i32
}

kgen.generator @bitcast_offset() -> !kgen.struct<(scalar<ui8>, scalar<ui8>)>{
  %x = pop.stack_allocation 1 x !kgen.scalar<si64>
  %0 = kgen.param.constant: scalar<si64> = <5>
  pop.store %0, %x : !kgen.pointer<scalar<si64>>
  %1 = pop.pointer.bitcast %x : !kgen.pointer<scalar<si64>> to !kgen.pointer<scalar<ui8>>
  %2 = pop.load %1 : !kgen.pointer<scalar<ui8>>
  %idx1 = index.constant 1
  %3 = pop.offset %1[%idx1] : !kgen.pointer<scalar<ui8>>
  %4 = pop.load %3 : !kgen.pointer<scalar<ui8>>
  %5 = kgen.struct.create(%2, %4) : !kgen.struct<(scalar<ui8>, scalar<ui8>)>
  kgen.return %5 : !kgen.struct<(scalar<ui8>, scalar<ui8>)>
}

kgen.generator @return_heap(%arg0: i16, %arg1: i16) -> !kgen.struct<(pointer<i16>, pointer<i16>)> {
  %idx4 = index.constant 4
  %idx1 = index.constant 1
  %idx32 = index.constant 0x2000
  %0 = pop.aligned_alloc %idx32, %idx4 : <i16>
  pop.store %arg0, %0 : !kgen.pointer<i16>
  %1 = pop.offset %0[%idx1] : !kgen.pointer<i16>
  pop.store %arg1, %1 : !kgen.pointer<i16>
  %2 = kgen.struct.create(%0, %1) : !kgen.struct<(pointer<i16>, pointer<i16>)>
  kgen.return %2 : !kgen.struct<(pointer<i16>, pointer<i16>)>
}

// COM: Check that the pointers alias.
kgen.generator @copy_load(%arg0: !kgen.pointer<i16>, %arg1: !kgen.pointer<i16>) -> i16 {
  %0 = pop.load %arg1: !kgen.pointer<i16>
  %idx1 = index.constant 1
  %1 = pop.offset %arg1[%idx1] : !kgen.pointer<i16>
  pop.store %0, %1 : !kgen.pointer<i16>
  %2 = pop.load %arg0 : !kgen.pointer<i16>
  kgen.return %2 : i16
}

kgen.generator @modify_stack_mem(%arg0: !kgen.pointer<i16>) -> !kgen.pointer<i16> {
  %zero = kgen.param.constant: i16 = <0>
  pop.store %zero, %arg0 : !kgen.pointer<i16>
  kgen.return %arg0 : !kgen.pointer<i16>
}

kgen.generator @return_pointer_to_pointer(%arg0: !kgen.pointer<i16>) -> !kgen.pointer<pointer<i16>> {
  %idx-1 = index.constant -1
  %idx16 = index.constant 16
  %0 = pop.aligned_alloc %idx-1, %idx16 : <pointer<i16>>
  %idx1 = index.constant 1
  %1 = pop.offset %0[%idx1] : !kgen.pointer<pointer<i16>>
  pop.store %arg0, %1 : !kgen.pointer<pointer<i16>>
  kgen.return %1 : !kgen.pointer<pointer<i16>>
}

kgen.generator @load_pointer_to_pointer(%arg0: !kgen.pointer<pointer<i16>>) -> i16 {
  %idx1 = index.constant 1
  %0 = pop.offset %arg0[%idx1] : !kgen.pointer<pointer<i16>>
  %1 = pop.load %0 : !kgen.pointer<pointer<i16>>
  %2 = pop.load %1 : !kgen.pointer<i16>
  kgen.return %2 : i16
}

kgen.generator @free_null(%arg0: !kgen.pointer<i16>) -> index {
  %idx0 = index.constant 0
  pop.aligned_free %arg0 : <i16>
  kgen.return %idx0 : index
}

kgen.generator @freed_memory() -> !kgen.pointer<i16> {
  %idx-1 = index.constant -1
  %idx4 = index.constant 4
  %0 = pop.aligned_alloc %idx-1, %idx4 : <i16>
  %1 = pop.aligned_alloc %idx-1, %idx4 : <i16>
  pop.aligned_free %0 : <i16>
  kgen.return %1 : !kgen.pointer<i16>
}

kgen.generator @const_string(%arg0: !kgen.pointer<i8>) -> !kgen.struct<(i8, pointer<i8>)> {
  %0 = pop.load %arg0 : !kgen.pointer<i8>
  %idx2 = index.constant 2
  %1 = pop.offset %arg0[%idx2] : !kgen.pointer<i8>
  %2 = kgen.struct.create(%0, %1) : !kgen.struct<(i8, pointer<i8>)>
  kgen.return %2 : !kgen.struct<(i8, pointer<i8>)>
}

kgen.generator @parameter_closure() -> index {
  %0 = pop.compiler.global_load "named_global" : index
  kgen.return %0 : index
}

kgen.generator @region_parameter(%arg0: index) -> index {
  pop.compiler.global_store "named_global", %arg0 : index
  %0 = kgen.call @parameter_closure() : () -> index
  kgen.return %0 : index
}

kgen.generator @store_undef() -> index {
  %0 = pop.stack_allocation 1 x index
  %1 = kgen.param.constant = <#interp.uninitmem>
  pop.store %1, %0 : !kgen.pointer<index>
  %2 = pop.load %0 : !kgen.pointer<index>
  kgen.return %2 : index
}

kgen.generator @malloc_and_free(%arg0: i16) -> i16 {
  %idx4 = index.constant 4
  %0 = pop.external_call @malloc(%idx4) : (index) -> (!kgen.pointer<i16>)
  pop.store %arg0, %0 : !kgen.pointer<i16>
  %2 = pop.load %0 : !kgen.pointer<i16>
  pop.external_call @free(%0) : (!kgen.pointer<i16>) -> ()
  kgen.return %2 : i16
}

kgen.generator @variant_discr_gep<Ts: param_list<type>>(%arg0: !kgen.pointer<variant<[Ts]>>) -> !kgen.pointer<scalar<ui8>> {
  %0 = pop.variant.discr_gep %arg0 : <variant<[Ts]>> as <scalar<ui8>>
  kgen.return %0 : !kgen.pointer<scalar<ui8>>
}

kgen.generator @store_union(%arg0: !pop.union<i32>) -> i32 {
  %0 = pop.stack_allocation 1 x union<i32>
  pop.store %arg0, %0 : !kgen.pointer<union<i32>>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<union<i32>> to !kgen.pointer<i32>
  %2 = pop.load %1 : !kgen.pointer<i32>
  kgen.return %2 : i32
}

kgen.generator @global_alloc() -> !kgen.pointer<i8, 2> {
  %0 = pop.global_alloc "alloc" 4 x i8 address_space 2 align 16
  kgen.return %0 : !kgen.pointer<i8, 2>
}

kgen.generator @simd_xor_fold_0() -> !kgen.scalar<uindex> {
  %0 = kgen.param.constant: scalar<uindex> = <0>
  %2 = kgen.param.constant: scalar<index> = <-1>
  %3 = pop.cast %2 : !kgen.scalar<index> to !kgen.scalar<uindex>
  %4 = pop.simd.xor %3, %0 : !kgen.scalar<uindex>
  kgen.return %4: !kgen.scalar<uindex>
}

kgen.generator @simd_xor_fold_1() -> !kgen.scalar<bool> {
  %0 = kgen.param.constant: scalar<bool> = <true>
  %1 = kgen.param.constant: scalar<index> = <1>
  %x = pop.cast %1 : !kgen.scalar<index> to !kgen.scalar<bool>
  %2 = pop.simd.xor %x, %0 : !kgen.scalar<bool>
  %3 = pop.simd.xor %2, %0 : !kgen.scalar<bool>
  kgen.return %3: !kgen.scalar<bool>
}

kgen.generator @simd_reduce_or_fold_0() -> !kgen.scalar<si32> {
  %0 = kgen.param.constant: simd<4, si32> = <<0, 1, 2, 16>>
  %1 = pop.simd.reduce_or %0 : !kgen.simd<4, si32>
  kgen.return %1 : !kgen.scalar<si32>
}

kgen.generator @simd_reduce_or_fold_1() -> !kgen.scalar<bool> {
  %0 = kgen.param.constant: simd<4, bool> = <<true, false, false, true>>
  %1 = pop.simd.reduce_or %0 : !kgen.simd<4, bool>
  kgen.return %1 : !kgen.scalar<bool>
}

kgen.generator @simd_reduce_or_fold_2() -> !kgen.scalar<index> {
  %0 = kgen.param.constant: simd<4, index> = <<64, 8, 2, 1>>
  %1 = pop.simd.reduce_or %0 : !kgen.simd<4, index>
  kgen.return %1 : !kgen.scalar<index>
}

kgen.generator @simd_reduce_and_fold_0() -> !kgen.scalar<si32> {
  %0 = kgen.param.constant: simd<4, si32> = <<1, 3, 5, 19>>
  %1 = pop.simd.reduce_and %0 : !kgen.simd<4, si32>
  kgen.return %1 : !kgen.scalar<si32>
}

kgen.generator @simd_reduce_and_fold_1() -> !kgen.scalar<bool> {
  %0 = kgen.param.constant: simd<4, bool> = <<true, false, false, true>>
  %1 = pop.simd.reduce_and %0 : !kgen.simd<4, bool>
  kgen.return %1 : !kgen.scalar<bool>
}

kgen.generator @simd_reduce_and_fold_2() -> !kgen.scalar<index> {
  %0 = kgen.param.constant: simd<4, index> = <<1, 3, 5, 19>>
  %1 = pop.simd.reduce_and %0 : !kgen.simd<4, index>
  kgen.return %1 : !kgen.scalar<index>
}

kgen.generator @pop_cmp() -> !kgen.scalar<bool> {
  %0 = kgen.param.constant: !kgen.scalar<index> = <3000000000>
  %1 = kgen.param.constant: !kgen.scalar<index> = <0>
  %2 = pop.cmp lt(%0 , %1) : !kgen.scalar<index>
  kgen.return %2 : !kgen.scalar<bool>
}

// COM: Adjacent non-power-of-two SIMD elements stride by the alloc size
// COM: (16 bytes for simd<3, f32>), not the 12-byte store size.
kgen.generator @npot_simd_stride(%arg0: !kgen.simd<3, f32>, %arg1: !kgen.simd<3, f32>) -> f32 {
  %buf = pop.stack_allocation 2 x simd<3, f32>
  pop.store %arg0, %buf : !kgen.pointer<simd<3, f32>>
  %idx1 = index.constant 1
  %elt1 = pop.offset %buf[%idx1] : !kgen.pointer<simd<3, f32>>
  pop.store %arg1, %elt1 : !kgen.pointer<simd<3, f32>>
  %f32s = pop.pointer.bitcast %buf : !kgen.pointer<simd<3, f32>> to !kgen.pointer<f32>
  %idx4 = index.constant 4
  %lane = pop.offset %f32s[%idx4] : !kgen.pointer<f32>
  %0 = pop.load %lane : !kgen.pointer<f32>
  kgen.return %0 : f32
}

// CHECK-LABEL: kgen.func export @do_it
kgen.generator export @do_it() {
  // CHECK-NEXT: <555>
  kgen.param.constant: i32 = <apply(
    :(i32) -> i32 @store_load_pointer, 555)>

  // CHECK-NEXT: [123, 456]
  kgen.param.constant: array<2, i24> = <apply(
    :(!pop.array<2, i24>) -> !pop.array<2, i24> @store_load<:type !pop.array<2, i24>>,
    [123, 456])>
  // CHECK-NEXT: <"1.25", "2.25">
  kgen.param.constant: simd<2, f32> = <apply(
    :(!kgen.simd<2, f32>) -> !kgen.simd<2, f32> @store_load<:type !kgen.simd<2, f32>>,
    <"1.25", "2.25">)>
  // CHECK-NEXT: <-7, 7>
  kgen.param.constant: simd<2, si4> = <apply(
    :(!kgen.simd<2, si4>) -> !kgen.simd<2, si4> @store_load<:type !kgen.simd<2, si4>>,
    <-7, 7>)>
  // CHECK-NEXT: <0, 1, 2, 3, 3, 2>
  kgen.param.constant: simd<6, ui2> = <apply(
    :(!kgen.simd<6, ui2>) -> !kgen.simd<6, ui2> @store_load<:type !kgen.simd<6, ui2>>,
    <0, 1, 2, 3, 3, 2>)>
  // CHECK-NEXT: <-5>
  kgen.param.constant: scalar<index> = <apply(
    :(!kgen.scalar<index>) -> !kgen.scalar<index> @store_load<:type !kgen.scalar<index>>,
    <-5>)>
  // CHECK-NEXT: { 120, 32112, 1.125{{0+}}e+00 }
  kgen.param.constant: struct<(i8, i16, f64)> = <apply(
    :(!kgen.struct<(i8, i16, f64)>) -> !kgen.struct<(i8, i16, f64)> @store_load<:type !kgen.struct<(i8, i16, f64)>>,
    { 120, 32112, 1.125 })>
  // CHECK-NEXT: <{:i32 42, 0}>
  kgen.param.constant: variant<i32, f64> = <apply(
    :(!kgen.variant<i32, f64>) -> !kgen.variant<i32, f64> @store_load<:type !kgen.variant<i32, f64>>,
    {:i32 42, 0})>

  // CHECK-NEXT: <1099511627792>
  kgen.param.constant: i64 = <apply(
    :(!pop.array<2, i24>) -> i64 @i24_pair_bitcast, [16, 256])>

  // CHECK-NEXT: <8590983192>
  kgen.param.constant: i64 = <apply(
    :(!kgen.struct<(i8, i16, i32)>) -> i64 @bitcast<:type !kgen.struct<(i8, i16, i32)>, :type i64>,
    { 24, 16, 2 })>
  // CHECK-NEXT: <1026>
  kgen.param.constant: i16 = <apply(
    :(!kgen.simd<2, si8>) -> i16 @bitcast<:type !kgen.simd<2, si8>, :type i16>,
    <2, 4>)>
  // CHECK-NEXT: <229>
  kgen.param.constant: ui8 = <apply(
    :(!kgen.simd<4, ui2>) -> ui8 @bitcast<:type !kgen.simd<4, ui2>, :type ui8>,
    <1, 1, 2, 3>)>

  // CHECK-NEXT: <0>
  kgen.param.constant: i8 = <apply(
    :(!kgen.variant<i32, i64>) -> i8 @variant_bitcast_discr, #kgen.variant<:i32 1, 0>)>
  // CHECK-NEXT: <1>
  kgen.param.constant: i8 = <apply(
    :(!kgen.variant<i32, i64>) -> i8 @variant_bitcast_discr, #kgen.variant<:i64 1, 1>)>

  // CHECK-NEXT: <12>
  kgen.param.constant: i24 = <apply(
    :(!pop.array<3, i24>) -> i24 @array_gep_load<0>, [12, 34, 56])>
  // CHECK-NEXT: <34>
  kgen.param.constant: i24 = <apply(
    :(!pop.array<3, i24>) -> i24 @array_gep_load<1>, [12, 34, 56])>
  // CHECK-NEXT: <56>
  kgen.param.constant: i24 = <apply(
    :(!pop.array<3, i24>) -> i24 @array_gep_load<2>, [12, 34, 56])>

  // CHECK-NEXT: <56>
  kgen.param.constant: i32 = <apply(
    :(!kgen.struct<(i8, i16, i32)>) -> i32 @struct_gep_load, { 12, 34, 56 })>

  // CHECK-NEXT: <128>
  kgen.param.constant: i32 = <apply(
    :(!kgen.struct<(i8, i16, i32)>) -> i32 @struct_gep_load, { #interp.uninitmem, #interp.uninitmem, 128 })>

  // CHECK-NEXT: <{ 5, 0 }>
  kgen.param.constant: struct<(scalar<ui8>, scalar<ui8>)> = <apply(
    :() -> !kgen.struct<(scalar<ui8>, scalar<ui8>)> @bitcast_offset)>

  // CHECK-NEXT: <{ #interp.memref<{[([[RETURN_HEAP:.*]], heap, [], [])], []}, 0, 0>,
  // CHECK-SAME:    #interp.memref<{[([[RETURN_HEAP]], heap, [], [])], []}, 0, 2> }>
  kgen.param.constant: struct<(pointer<i16>, pointer<i16>)> = <apply(
    :(i16, i16) -> !kgen.struct<(pointer<i16>, pointer<i16>)> @return_heap, 0xDEAD, 0xBEEF)>

  // CHECK-NEXT: <-8531>
  kgen.param.constant: i16 = <apply(
    :(!kgen.pointer<i16>, !kgen.pointer<i16>) -> i16 @copy_load,
    #interp.memref<{[(#mem, heap, [], [])], []}, 0, 2>, #interp.memref<{[(#mem, heap, [], [])], []}, 0, 0>)>

  // CHECK-NEXT: <#interp.memref<{[([[MODIFY_STACK_MEM:.*]], stack, [], [])], []}, 0, 0>>
  kgen.param.constant: pointer<i16> = <apply(
    :(!kgen.pointer<i16>) -> !kgen.pointer<i16> @modify_stack_mem,
    #interp.memref<{[(#stack, stack, [], [])], []}, 0, 0>)>

  // CHECK-NEXT: <#interp.memref<{[([[RETURN_POINTER:.*]], heap, [(8, 1, 0)], []),
  // CHECK-SAME:             ([[RETURN_POINTER_1:.*]], stack, [], [])], []}, 0, 8>>
  kgen.param.constant: !kgen.pointer<pointer<i16>> = <apply(
    :(!kgen.pointer<i16>) -> !kgen.pointer<pointer<i16>> @return_pointer_to_pointer,
    #interp.memref<{[(#some_ptr, stack, [], [])], []}, 0, 0>)>

  // CHECK-NEXT: <-8531>
  kgen.param.constant: i16 = <apply(
    :(!kgen.pointer<pointer<i16>>) -> i16 @load_pointer_to_pointer,
    #interp.memref<{[(#pointer, stack, [(8, 1, 0)], []), (#stack, stack, [], [])], []}, 0, 0>)>

  // CHECK-NEXT: <0>
  kgen.param.constant: index = <apply(
    :(!kgen.pointer<i16>) -> index @free_null, #interp.pointer<0>)>

  // CHECK-NEXT: <#interp.memref<{[([[FREED_CONCRETE_MEM:.*]], heap, [], [])], []}, 0, 0>>
  kgen.param.constant: pointer<i16> = <apply(:() -> !kgen.pointer<i16> @freed_memory)>

  // COM: `ord("hello world"[2]) -> 108`.
  // CHECK-NEXT: <{ 108, #interp.memref<{[([[STRING]], const_global, [], [])], []}, 0, 4> }>
  kgen.param.constant: struct<(i8, pointer<i8>)> = <apply(
    :(!kgen.pointer<i8>) -> !kgen.struct<(i8, pointer<i8>)> @const_string,
    #interp.memref<{[(#string, const_global, [], [])], []}, 0, 2>)>

  // CHECK-NEXT: <1>
  kgen.param.constant = <apply(:(index) -> index @region_parameter, 1)>

  // COM: The value is garbage. Just make sure the function interprets.
  // CHECK-NEXT: <{{.*}}>
  kgen.param.constant = <apply(:() -> index @store_undef)>

  // CHECK-NEXT: <"hello world">
  kgen.param.constant: string = <apply(
    :(!kgen.string) -> !kgen.string @store_load<:type string>, "hello world")>

  // CHECK-NEXT: <"">
  kgen.param.constant: string = <apply(
    :(!kgen.string) -> !kgen.string @store_load<:type string>, "")>

  // CHECK-NEXT: <[3, 4, 5]>
  kgen.param.constant: param_list<i32> = <apply(
    :(!kgen.param_list<i32>) -> !kgen.param_list<i32> @store_load<:type param_list<i32>>, [3, 4, 5])>

  // CHECK-NEXT: <index>
  kgen.param.constant: type = <apply(
    :(!kgen.type) -> !kgen.type @store_load<:type type>, index)>

  // CHECK-NEXT: <7>
  kgen.param.constant: i16 = <apply(:(i16) -> i16 @malloc_and_free, 7)>

  // CHECK-NEXT: <8>
  kgen.param.constant: pointer<scalar<ui8>> = <apply(:(!kgen.pointer<variant<i24, i48>>) -> !kgen.pointer<scalar<ui8>>
    @variant_discr_gep<:param_list<type> [i24, i48]>, 0)>
  // CHECK-NEXT: <16>
  kgen.param.constant: pointer<scalar<ui8>> = <apply(:(!kgen.pointer<variant<i24, i48>>) -> !kgen.pointer<scalar<ui8>>
    @variant_discr_gep<:param_list<type> [simd<4, f32>, i8]>, 0)>

  // CHECK-NEXT: <42>
  kgen.param.constant: i32 = <apply(:(!pop.union<i32>) -> i32 @store_union, {:i32 42})>

  // CHECK-NEXT: constant: pointer<i8, 2> = <#interp.memref<{[([[GLOBAL_ALLOC]], persistent, [], [], 2)], []}, 0, 0>>
  kgen.param.constant: pointer<i8, 2> = <apply(:() -> !kgen.pointer<i8, 2> @global_alloc)>

  // CHECK-NEXT: constant: scalar<uindex> = <18446744073709551615>
  kgen.param.constant: scalar<uindex> = <apply(:() -> !kgen.scalar<uindex> @simd_xor_fold_0)>

  // CHECK-NEXT: constant: scalar<bool> = <true>
  kgen.param.constant: scalar<bool> = <apply(:() -> !kgen.scalar<bool> @simd_xor_fold_1)>

  // CHECK-NEXT: constant: scalar<si32> = <19>
  kgen.param.constant: scalar<si32> = <apply(:() -> !kgen.scalar<si32> @simd_reduce_or_fold_0)>

  // CHECK-NEXT: constant: scalar<bool> = <true>
  kgen.param.constant: scalar<bool> = <apply(:() -> !kgen.scalar<bool> @simd_reduce_or_fold_1)>

  // CHECK-NEXT: constant: scalar<index> = <75>
  kgen.param.constant: scalar<index> = <apply(:() -> !kgen.scalar<index> @simd_reduce_or_fold_2)>

  // CHECK-NEXT: constant: scalar<si32> = <1>
  kgen.param.constant: scalar<si32> = <apply(:() -> !kgen.scalar<si32> @simd_reduce_and_fold_0)>

  // CHECK-NEXT: constant: scalar<bool> = <false>
  kgen.param.constant: scalar<bool> = <apply(:() -> !kgen.scalar<bool> @simd_reduce_and_fold_1)>

  // CHECK-NEXT: constant: scalar<index> = <1>
  kgen.param.constant: scalar<index> = <apply(:() -> !kgen.scalar<index> @simd_reduce_and_fold_2)>

  // CHECK-NEXT: constant: scalar<bool> = <false>
  kgen.param.constant: scalar<bool> = <apply(:() -> !kgen.scalar<bool> @pop_cmp)>

  // COM: Lane 0 of the second simd<3, f32> element lives at byte 16
  // COM: (f32 index 4); a 12-byte stride would place 6.5 there instead.
  // CHECK-NEXT: constant: f32 = <5.500000e+00>
  kgen.param.constant: f32 = <apply(
    :(!kgen.simd<3, f32>, !kgen.simd<3, f32>) -> f32 @npot_simd_stride,
    <"0.5", "1.5", "2.5">, <"5.5", "6.5", "7.5">)>

  kgen.return
}
