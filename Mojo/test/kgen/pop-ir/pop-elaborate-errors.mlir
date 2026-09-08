// RUN: kgen-opt -elaborate-generators="use-parametric-interpret=false" %s -verify-diagnostics -split-input-file
// RUN: kgen-opt -elaborate-generators="use-parametric-interpret=true" %s -split-input-file 2>&1 | FileCheck %s --check-prefix=CHECK-PARAM

// COM: use-parametric-interpret=true has slight difference from =false for error messages.
//      Using FileCheck instead to check those with CHECK-PARAM prefix.

// expected-note @below {{failed to interpret function @out_of_range_read}}
kgen.generator @out_of_range_read() -> i32 {
  %0 = pop.stack_allocation 0 x i32
  // CHECK-PARAM: failed to interpret operation pop.load
  // CHECK-PARAM: memory access size 4 is out-of-bounds
  // expected-note @below {{failed to interpret operation pop.load}}
  // expected-note @below {{memory access size 4 is out-of-bounds}}
  %1 = pop.load %0 : !kgen.pointer<i32>
  kgen.return %1 : i32
}

// expected-error @below {{function instantiation failed}}
kgen.generator @call_it() {
  // CHECK-PARAM: failed to compile-time evaluate function call
  // expected-note @below {{failed to compile-time evaluate function call}}
  %0 = kgen.param.constant: i32 = <apply(:() -> i32 @out_of_range_read)>
  kgen.return
}

// -----

kgen.generator @return_stack_addr() -> !kgen.pointer<index> {
  %0 = pop.stack_allocation 1 x index
  kgen.return %0 : !kgen.pointer<index>
}

// expected-note @below {{failed to interpret function @stack_use_after_free}}
kgen.generator @stack_use_after_free() -> index {
  %0 = kgen.call @return_stack_addr() : () -> !kgen.pointer<index>
  // CHECK-PARAM: failed to interpret operation pop.load{isInvariant: false, isNonTemporal: false, isVolatile: false, ordering: #pop.atomic_ordering<not_atomic>}(#interp.pointer
  // CHECK-PARAM: address is out-of-bounds
  // expected-note @below {{failed to interpret operation pop.load{isInvariant: false, isNonTemporal: false, isVolatile: false, ordering: #pop.atomic_ordering<not_atomic>}(#interp.pointer}}
  // expected-note @below {{address is out-of-bounds}}
  %1 = pop.load %0 : !kgen.pointer<index>
  kgen.return %1 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator @call_it() {
  // CHECK-PARAM: failed to compile-time evaluate function call
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.constant = <apply(:() -> index @stack_use_after_free)>
  kgen.return
}

// -----

// expected-note @below {{failed to interpret function @heap_use_after_free}}
kgen.generator @heap_use_after_free() -> i64 {
  %idx32 = index.constant 32
  %idx8 = index.constant 8
  %0 = pop.aligned_alloc %idx32, %idx8 : <i64>
  pop.aligned_free %0 : <i64>
  // CHECK-PARAM: failed to interpret operation pop.load{isInvariant: false, isNonTemporal: false, isVolatile: false, ordering: #pop.atomic_ordering<not_atomic>}(#interp.pointer
  // CHECK-PARAM: accessing memory that was freed
  // expected-note @below {{failed to interpret operation pop.load{isInvariant: false, isNonTemporal: false, isVolatile: false, ordering: #pop.atomic_ordering<not_atomic>}(#interp.pointer}}
  // expected-note @below {{accessing memory that was freed}}
  %1 = pop.load %0 : !kgen.pointer<i64>
  kgen.return %1 : i64
}

// expected-error @below {{function instantiation failed}}
kgen.generator @call_it() {
  // CHECK-PARAM: failed to compile-time evaluate function call
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.constant: i64 = <apply(:() -> i64 @heap_use_after_free)>
  kgen.return
}

// -----

// expected-note @below {{failed to interpret function @clobber_pointer}}
kgen.generator @clobber_pointer(%arg0: i16) -> i16 {
  %0 = pop.stack_allocation 1 x i64
  %1 = pop.stack_allocation 1 x !kgen.pointer<i64>
  pop.store %0, %1 : !kgen.pointer<pointer<i64>>
  %2 = pop.pointer.bitcast %1 : !kgen.pointer<pointer<i64>> to !kgen.pointer<i16>
  %idx1 = index.constant 1
  %3 = pop.offset %2[%idx1] : !kgen.pointer<i16>
  // CHECK-PARAM: failed to interpret operation pop.store
  // CHECK-PARAM: write clobbers a pointer region
  // expected-note @below {{failed to interpret operation pop.store}}
  // expected-note @below {{write clobbers a pointer region}}
  pop.store %arg0, %3 : !kgen.pointer<i16>
  kgen.return %arg0 : i16
}

// expected-error @below {{function instantiation failed}}
kgen.generator @call_it() {
  // CHECK-PARAM: failed to compile-time evaluate function call
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.constant: i16 = <apply(:(i16) -> i16 @clobber_pointer, 5)>
  kgen.return
}

// -----

module attributes {M.target = #M.target<triple="", arch="", features="", data_layout="p:64:64", simd_bit_width=128>} {

// expected-note @below {{failed to interpret function @clobber_pointer}}
kgen.generator @clobber_pointer(%arg0: i64) -> i64 {
  %0 = pop.stack_allocation 1 x i64
  %1 = pop.stack_allocation 1 x !kgen.struct<(i64, !kgen.pointer<i64>)>
  %2 = kgen.struct.gep %1[1] : <struct<(i64, !kgen.pointer<i64>)>>
  pop.store %0, %2 : !kgen.pointer<pointer<i64>>
  %3 = kgen.struct.gep %1[0] : <struct<(i64, !kgen.pointer<i64>)>>
  %4 = pop.pointer.bitcast %3 : !kgen.pointer<i64> to !kgen.pointer<i32>
  %idx1 = index.constant 1
  %5 = pop.offset %4[%idx1] : !kgen.pointer<i32>
  %6 = pop.pointer.bitcast %5 : !kgen.pointer<i32> to !kgen.pointer<i64>
  // CHECK-PARAM: failed to interpret operation pop.store
  // CHECK-PARAM: write clobbers a pointer region
  // expected-note @below {{failed to interpret operation pop.store}}
  // expected-note @below {{write clobbers a pointer region}}
  pop.store %arg0, %6 : !kgen.pointer<i64>
  kgen.return %arg0 : i64
}

// expected-error @below {{function instantiation failed}}
kgen.generator @call_it() {
  // CHECK-PARAM: failed to compile-time evaluate function call
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.constant: i64 = <apply(:(i64) -> i64 @clobber_pointer, 1)>
  kgen.return
}

}

// -----

module attributes {M.target = #M.target<triple="", arch="", features="", data_layout="p:64:64", simd_bit_width=128>} {

// expected-note @below {{failed to interpret function @parameter_closure}}
kgen.generator @parameter_closure() -> index {
  // CHECK-PARAM: failed to interpret operation pop.compiler.global_load{name: "named_global"}()
  // CHECK-PARAM: cannot evaluate standalone capturing closure at compile time
  // expected-note @below {{failed to interpret operation pop.compiler.global_load{name: "named_global"}()}}
  // expected-note @below {{cannot evaluate standalone capturing closure at compile time}}
  %0 = pop.compiler.global_load "named_global" : index
  kgen.return %0 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @use_it() {
  // CHECK-PARAM: failed to compile-time evaluate function call
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.constant = <apply(:() -> index @parameter_closure)>
  kgen.return
}

}

// -----

module attributes {M.target = #M.target<triple="", arch="", features="", data_layout="p:64:64", simd_bit_width=128>} {

// expected-note @below {{failed to interpret function @load_union}}
kgen.generator @load_union() -> !pop.union<index> {
  %0 = pop.stack_allocation 1 x union<index>
  %1 = kgen.param.constant: union<index> = <{42}>
  pop.store %1, %0 : !kgen.pointer<union<index>>
  // CHECK-PARAM: failed to interpret operation pop.load
  // CHECK-PARAM: cannot read a union-typed value
  // expected-note @below {{failed to interpret operation pop.load}}
  // expected-note @below {{cannot read a union-typed value}}
  %2 = pop.load %0 : !kgen.pointer<union<index>>
  kgen.return %2 : !pop.union<index>
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @use_it() {
  // CHECK-PARAM: failed to compile-time evaluate function call
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.constant: union<index> = <apply(:() -> !pop.union<index> @load_union)>
  kgen.return
}

}

// -----

module attributes {M.target = #M.target<triple="", arch="", features="", data_layout="p:64:64", simd_bit_width=128>} {

// expected-note @below {{failed to interpret function @invalid_simd,dtype=si32,size=5}}
// expected-note @below {{failed to interpret function @invalid_simd,dtype=si32,size=3}}
// expected-note @below {{failed to interpret function @invalid_simd,dtype=invalid,size=4}}
kgen.generator @invalid_simd<dtype: dtype, size>() -> index {
  %0 = kgen.param.constant: index = <42>
  %1 = pop.cast_from_builtin %0 : index to !kgen.scalar<index>
  // expected-note @below {{failed to interpret operation pop.cast}}
  // expected-note @below {{simd type cannot be DType.invalid}}
  %2 = pop.cast %1 : !kgen.scalar<index> to !kgen.scalar<dtype>
  // expected-note @below {{failed to interpret operation pop.simd.splat}}
  // expected-note @below {{simd width must be a power of 2}}
  %3 = pop.simd.splat %2 : !kgen.simd<size, dtype>

  kgen.return %0 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @use_it_simd_5() {
  // CHECK-PARAM: failed to compile-time evaluate function call
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.constant = <apply(:() -> index @invalid_simd<:dtype si32, 5>)>
  kgen.return
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @use_it_simd_3() {
  // CHECK-PARAM: failed to compile-time evaluate function call
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.constant = <apply(:() -> index @invalid_simd<:dtype si32, 3>)>
  kgen.return
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @use_it_simd_invalid() {
  // CHECK-PARAM: failed to compile-time evaluate function call
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.constant = <apply(:() -> index @invalid_simd<:dtype invalid, 4>)>
  kgen.return
}

}

// -----

module attributes {M.target = #M.target<triple="", arch="", features="", data_layout="p:64:64", simd_bit_width=128>} {

// expected-note @below {{failed to interpret function @invalid_simd_reduce_or}}
kgen.generator @invalid_simd_reduce_or() -> !kgen.scalar<f32> {
  %0 = kgen.param.constant: simd<4, f32> = <<"1.0", "0.0", "1.5", "2.0">>
// expected-note @below {{failed to fold operation pop.simd.reduce_or}}
  %1 = pop.simd.reduce_or %0 : !kgen.simd<4, f32>
  kgen.return %1 : !kgen.scalar<f32>
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @use_it_simd_invalid_reduce_or() {
  // CHECK-PARAM: failed to compile-time evaluate function call
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.constant = <apply(:() -> index @invalid_simd_reduce_or)>
  kgen.return
}

}

// -----

module attributes {M.target = #M.target<triple="", arch="", features="", data_layout="p:64:64", simd_bit_width=128>} {

// expected-note @below {{failed to interpret function @invalid_simd_reduce_and}}
kgen.generator @invalid_simd_reduce_and() -> !kgen.scalar<f32> {
  %0 = kgen.param.constant: simd<4, f32> = <<"1.0", "0.0", "1.5", "2.0">>
// expected-note @below {{failed to fold operation pop.simd.reduce_and}}
  %1 = pop.simd.reduce_and %0 : !kgen.simd<4, f32>
  kgen.return %1 : !kgen.scalar<f32>
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @use_it_simd_invalid_reduce_and() {
  // CHECK-PARAM: failed to compile-time evaluate function call
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.constant = <apply(:() -> index @invalid_simd_reduce_and)>
  kgen.return
}

}

// -----

// Test that inline_asm produces a helpful error message explaining that it
// cannot be evaluated at compile time (MOCO-1771).

// expected-note @below {{failed to interpret function @gpu_inline_asm}}
kgen.generator @gpu_inline_asm() -> i64 {
  // CHECK-PARAM: failed to fold operation pop.inline_asm
  // CHECK-PARAM: Note: Inline assembly cannot be evaluated at compile time.
  // The helpful note is part of the same error message, so use expected-note-re
  // to match the beginning and the fact that the note is appended.
  // expected-note-re @below {{failed to fold operation pop.inline_asm{{.*}}Note: Inline assembly cannot be evaluated at compile time. Inline assembly contains hardware-specific instructions that can only execute at runtime on the target hardware.}}
  %0 = pop.inline_asm "createpolicy.fractional.L2::evict_first.b64 $0;", "=l", () : () -> i64
  kgen.return %0 : i64
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @use_inline_asm_in_param() {
  // CHECK-PARAM: failed to compile-time evaluate function call
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.constant: i64 = <apply(:() -> i64 @gpu_inline_asm)>
  kgen.return
}
