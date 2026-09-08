// RUN: kgen-opt %s -verify-diagnostics -split-input-file

kgen.func @pop_select_simd(
    // expected-note @below {{prior use here}}
    %arg0: !kgen.scalar<bool>,
    %arg1: !kgen.simd<4, si32>,
    %arg2: !kgen.simd<4, si32>
  ) -> !kgen.simd<4, si32> {
  // expected-error @below {{use of value '%arg0' expects different type than prior uses: '!kgen.simd<4, bool>' vs '!kgen.scalar<bool>'}}
  %0 = pop.simd.select %arg0, %arg1, %arg2 : !kgen.simd<4, si32>
  kgen.return %0 : !kgen.simd<4, si32>
}

// -----

kgen.func @pop_select_simd(
    // expected-note @below {{prior use here}}
    %arg0: !kgen.simd<8, bool>,
    %arg1: !kgen.simd<4, si32>,
    %arg2: !kgen.simd<4, si32>
  ) -> !kgen.simd<4, si32> {
  // expected-error @below {{use of value '%arg0' expects different type than prior uses: '!kgen.simd<4, bool>' vs '!kgen.simd<8, bool>'}}
  %0 = pop.simd.select %arg0, %arg1, %arg2 : !kgen.simd<4, si32>
  kgen.return %0 : !kgen.simd<4, si32>
}

// -----

kgen.generator @bitcast_scalar(%a: !kgen.scalar<f32>) {
  // expected-error @below {{'pop.bitcast' op input type '!kgen.scalar<f32>' and result type '!kgen.scalar<si8>' are cast incompatible}}
  %0 = pop.bitcast %a : !kgen.scalar<f32> to !kgen.scalar<si8>
  kgen.return
}

// -----

kgen.generator @bitcast_simd(%a: !kgen.simd<4, f32>) {
  // expected-error @below {{'pop.bitcast' op input type '!kgen.simd<4, f32>' and result type '!kgen.simd<8, f32>' are cast incompatible}}
  %0 = pop.bitcast %a : !kgen.simd<4, f32> to !kgen.simd<8, f32>
  kgen.return
}

// -----

kgen.generator @bitcast_simd(%a: !kgen.simd<4, f32>) {
  // expected-error @below {{'pop.bitcast' op input type '!kgen.simd<4, f32>' and result type '!kgen.simd<4, f64>' are cast incompatible}}
  %0 = pop.bitcast %a : !kgen.simd<4, f32> to !kgen.simd<4, f64>
  kgen.return
}

// -----

kgen.generator @cast_simd_size<type: dtype>(%a: !kgen.simd<2, type>) {
  // expected-error @below {{are cast incompatible}}
  %0 = pop.cast %a : !kgen.simd<2, type> to !kgen.simd<4, type>
  kgen.return
}

// -----

kgen.generator @cast_simd_size<size, type: dtype>(%a: !kgen.simd<size, type>) {
  // expected-error @below {{are cast incompatible}}
  %0 = pop.cast %a : !kgen.simd<size, type> to !kgen.simd<add(size, 1), type>
  kgen.return
}

// -----

kgen.generator @simd_shuffle(%a: !kgen.simd<2, f32>) {
  // expected-error @below {{expected result dtype to match operand dtypes}}
  %0 = pop.simd.shuffle <2, f32> %a, %a -> <1, f64> :array<1, index> [1]
  kgen.return
}

// -----

kgen.generator @simd_shuffle<type: dtype>(%a: !kgen.simd<2, f32>) {
  // expected-error @below {{expected result dtype to match operand dtypes}}
  %0 = pop.simd.shuffle <2, f32> %a, %a -> <1, type> :array<1, index> [1]
  kgen.return
}

// -----

kgen.generator @simd_shuffle<size>(%a: !kgen.simd<2, f32>) {
  // expected-error @below {{mask element 4 is out of bounds}}
  %0 = pop.simd.shuffle <2, f32> %a, %a -> <1, f32> :array<1, index> [4]
  kgen.return
}

// -----

kgen.func @cast_from_builtin_type(%arg0: si32) {
  // expected-error @below {{cannot convert to scalar dtype ui32 from 'si32'}}
  %0 = pop.cast_from_builtin %arg0 : si32 to !kgen.scalar<ui32>
  kgen.return
}

// -----

kgen.func @cast_from_simd_to_vector(%arg0: vector<4xsi32>) {
  // expected-error @below {{'pop.cast_from_builtin' op cannot convert to SIMD dtype f32 from vector element 'si32'}}
  %0 = pop.cast_from_builtin %arg0 : vector<4xsi32> to !kgen.simd<4, f32>
  kgen.return
}

// -----

kgen.func @cast_simd_to_vector(%arg0: !kgen.simd<4, f32>) {
  // expected-error @below {{expected a rank 1 non-scalable vector}}
  %0 = pop.cast_to_builtin %arg0 : !kgen.simd<4, f32> to f32
  kgen.return
}

// -----

kgen.func @cast_simd_to_vector_elt_type(%arg0: !kgen.simd<4, f32>) {
  // expected-error @below {{'pop.cast_to_builtin' op cannot convert from SIMD dtype f32 to vector element 'si32'}}
  %0 = pop.cast_to_builtin %arg0 : !kgen.simd<4, f32> to vector<4xsi32>
  kgen.return
}

// -----

kgen.generator @cast_simd_to_vector<size>(%arg0: !kgen.simd<size, f32>) {
  // expected-error @below {{cannot convert   %0 = pop.cast_to_builtin %arg0 : !kgen.simd<size, f32> to vector<4xi32>
  kgen.return
}

// -----

kgen.func @cast_simd_to_vector(%arg0: !kgen.simd<4, f32>) {
  // expected-error @below {{expected vector<4xT>}}
  %0 = pop.cast_to_builtin %arg0 : !kgen.simd<4, f32> to vector<8xf32>
  kgen.return
}

// -----

kgen.func @simd_splat(%arg0: !kgen.scalar<f32>) {
  // expected-error @below {{'pop.simd.splat' op requires a non-negative size}}
  %0 = pop.simd.splat %arg0 : !kgen.simd<-1, f32>
  kgen.return
}

// -----

kgen.func @invalid_array_create(%arg0: i32) {
  // expected-error @below {{expected 2 operands to create array but got 1}}
  %0 = pop.array.create [%arg0] : !pop.array<2, i32>
  kgen.return
}

// -----

kgen.func @array_out_of_bounds(%arg0: !pop.array<1, i32>) {
  // expected-error @below {{'pop.array.get' op array index out of bounds: -1}}
  %0 = pop.array.get %arg0[-1] : !pop.array<1, i32>
  kgen.return
}

// -----

kgen.func @array_out_of_bounds(%arg0: !pop.array<1, i32>) {
  // expected-error @below {{'pop.array.get' op array index out of bounds: 2}}
  %0 = pop.array.get %arg0[2] : !pop.array<1, i32>
  kgen.return
}

// -----

kgen.func @repeat_zero() {
  // expected-error @below {{requires at least one operand to create an array whose size is non-zero}}
  %0 = pop.array.repeat [] : !pop.array<1, i32>
  kgen.return
}

// -----

kgen.func @array_repeat_crash(%arg0: index) {
  // expected-error @below {{'pop.array.repeat' op requires a non-negative size}}
  %0 = "pop.array.repeat"(%arg0) : (index) -> !pop.array<-1, index>
  kgen.return
}

// -----

kgen.func @struct_gep_type(%a: !kgen.pointer<struct<(i32)>>) {
  // expected-error @below {{'kgen.struct.gep' op struct field index 1 is out of bounds for struct with 1 elements}}
  %0 = "kgen.struct.gep"(%a) { index = 1 : index } : (!kgen.pointer<struct<(i32)>>) -> !kgen.pointer<i32>
  kgen.return
}

// -----

kgen.func @struct_gep_type(%a: !kgen.pointer<struct<(i32)>>) {
  // expected-error @below {{'kgen.struct.gep' op result element type 'i64' does not match struct element type 'i32' at index 0}}
  %0 = "kgen.struct.gep"(%a) { index = 0 : index } : (!kgen.pointer<struct<(i32)>>) -> !kgen.pointer<i64>
  kgen.return
}

// -----

kgen.func @load_atomic(%p: !kgen.pointer<scalar<f32>>) {
  // expected-error @below {{invalid combination of volatile or invariant with atomic load}}
  pop.load volatile<1> atomic acquire %p: !kgen.pointer<scalar<f32>>
  kgen.return
}

// -----

kgen.func @load_non_atomic_syncscope(%p: !kgen.pointer<scalar<f32>>) {
  // expected-error @below {{cannot specify syncscope without an atomic load}}
  pop.load atomic syncscope("singlethread") not_atomic %p: !kgen.pointer<scalar<f32>>
  kgen.return
}

// -----

kgen.func @store_atomic_invalid_ordering(%p: !kgen.pointer<scalar<f32>>, %v: !kgen.scalar<f32>) {
  // expected-error @below {{invalid atomic ordering 'acquire' for store operation}}
  pop.store atomic acquire %v, %p : !kgen.pointer<scalar<f32>>
  kgen.return
}

// -----

kgen.func @store_volatile_atomic(%p: !kgen.pointer<scalar<f32>>, %v: !kgen.scalar<f32>) {
  // expected-error @below {{volatile stores cannot be atomic}}
  pop.store volatile<1> atomic release %v, %p : !kgen.pointer<scalar<f32>>
  kgen.return
}

// -----

kgen.func @store_non_atomic_syncscope(%p: !kgen.pointer<scalar<f32>>, %v: !kgen.scalar<f32>) {
  // expected-error @below {{cannot specify syncscope without an atomic store}}
  pop.store atomic syncscope("singlethread") not_atomic %v, %p : !kgen.pointer<scalar<f32>>
  kgen.return
}

// -----

kgen.func @fence() {
  // expected-error @below {{'pop.fence' op can be given only acquire, release, acq_rel, and seq_cst orderings}}
  pop.fence not_atomic
  kgen.return
}

// -----

kgen.func @fence() {
  // expected-error @below {{'pop.fence' op can be given only acquire, release, acq_rel, and seq_cst orderings}}
  pop.fence monotonic
  kgen.return
}

// -----

kgen.func @fence() {
  // expected-error @below {{'pop.fence' op can be given only acquire, release, acq_rel, and seq_cst orderings}}
  pop.fence unordered
  kgen.return
}

// -----

kgen.func @variant_bitcast_oob(%arg0: !kgen.pointer<variant<i32>>) {
  // expected-error @below {{variant index 1 is out of bounds in range [0, 1)}}
  %0 = "pop.variant.bitcast"(%arg0) {index = 1 : index} : (!kgen.pointer<variant<i32>>) -> !kgen.pointer<i32>
  kgen.return
}

// -----

kgen.func @variant_bitcast_oob(%arg0: !kgen.pointer<variant<i32>>) {
  // expected-error @below {{variant element at index 0 expected type 'i32' but result has type 'i64'}}
  %0 = "pop.variant.bitcast"(%arg0) {index = 0 : index} : (!kgen.pointer<variant<i32>>) -> !kgen.pointer<i64>
  kgen.return
}

// -----

kgen.func @variant_discr_gep_type(%arg0: !kgen.pointer<variant<i32, i64>>) {
  // expected-error @below {{variant expected discriminant bitwidth to be 8 but result returns uint with width 16}}
  %0 = pop.variant.discr_gep %arg0 : <variant<i32, i64>> as <scalar<ui16>>
  kgen.return
}

// -----

kgen.func @invalid_union() {
  // expected-error @below {{value type 'i64' is not a union element type of '!pop.union<i32>'}}
  kgen.param.constant: union<i32> = <{:i64 42}>
  kgen.return
}

// -----

kgen.func @invalid_union_bitcast(%arg0: !kgen.pointer<union<i32>>) {
  // expected-error @below {{result pointer element type 'i64' is not an element type of '!pop.union<i32>'}}
  pop.union.bitcast %arg0 : <union<i32>> as <i64>
  kgen.return
}

// -----

kgen.func @invalid_union_wrap(%arg0: i32) {
  // expected-error @below {{operand type 'i32' is not an element type of '!pop.union<i64>'}}
  %0 = pop.union.wrap %arg0 : i32 as <i64>
  kgen.return
}

// -----

kgen.func @invalid_union_unwrap(%arg0: !pop.union<i32>) {
  // expected-error @below {{result type 'i64' is not an element type of '!pop.union<i32>'}}
  %0 = pop.union.unwrap %arg0 : <i32> as i64
  kgen.return
}

// -----

kgen.func @invalid_simd_reduce_or(%arg0: i32) {
  // expected-error @below {{custom op 'pop.simd.reduce_or' 'vector' must be parameterized SIMD vector type, but got 'i32'}}
  %0 = pop.simd.reduce_or %arg0 : i32
  kgen.return
}

// -----

kgen.func @invalid_simd_reduce_and(%arg0: i32) {
  // expected-error @below {{custom op 'pop.simd.reduce_and' 'vector' must be parameterized SIMD vector type, but got 'i32'}}
  %0 = pop.simd.reduce_and %arg0 : i32
  kgen.return
}


// -----

kgen.func @invalid_memcpy(%dst: !kgen.pointer<scalar<f32>, 3>, %src: !kgen.pointer<scalar<i32>>) {
  %0 = kgen.param.constant: index = <4>
  // expected-error @below {{'pop.memcpy' op source and destination must have same element type, got '!kgen.scalar<i32>' and '!kgen.scalar<f32>'}}
  pop.memcpy %dst, %src, %0 : !kgen.pointer<scalar<i32>> -> !kgen.pointer<scalar<f32>, 3>
  kgen.return
}

// -----

kgen.generator @global_alloc_initializer_count_not_1() {
  // expected-error @below {{'pop.global_alloc' op with an initializer requires count to be 1, but got 4}}
  %0 = pop.global_alloc "bad" 4 x !kgen.scalar<si32> = <42>
  kgen.return
}

// -----

// A count above the operand count cannot be a verifier error, because argument
// packs expand into operands after this runs; see pop-to-llvm/globals-error.mlir.
kgen.func @external_call_negative_num_fixed_args(%a: !kgen.simd<1, ui32>) {
  // expected-error @below {{'pop.external_call' op 'numFixedArgs' must be non-negative, found -1}}
  pop.external_call @bar(%a) attributes {numFixedArgs = -1 : index}
    : (!kgen.simd<1, ui32>) -> ()
  kgen.return
}
