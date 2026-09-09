// RUN: kgen-opt %s -elaborate-generators="max-depth=128 use-parametric-interpret=false loop-unrolling-warn-threshold=27" -verify-diagnostics -split-input-file -allow-unregistered-dialect

// Recursive expansions.

// expected-note @below {{function instantiation failed}}
// expected-note-re @below {{elaborator expansion is {{[0-9]+}} levels deep - infinite recursion?}}
// expected-note-re @below {{error recurses {{[0-9]+}} times}}
// expected-note @below {{remaining errors after}}
kgen.generator @genItf3<x>() {
  // expected-note @+1 {{call expansion failed}}
  kgen.call @genItf3<add(x, 1)>() : () -> ()
  kgen.return
}

// expected-error @+1 {{function instantiation failed}}
kgen.generator @use_Itf3two() {
  // expected-note @+1 {{call expansion failed}}
  kgen.call @genItf3<2>() : () -> ()
  kgen.return
}


// -----

#target = #kgen.target<triple="", arch="", features="", data_layout="", simd_bit_width=128> : !kgen.target

// expected-error @below {{function instantiation failed}}
kgen.generator @sizeof_unknown() {
  // expected-note @below {{could not simplify operator get_sizeof}}
  %0 = kgen.param.constant: index = <get_sizeof(!opaque<"type">, #target)>
  kgen.return
}

// -----

// expected-note @below {{failed to interpret function @cant_interpret}}
kgen.generator @cant_interpret(%arg0: index) -> index {
  // expected-note @below {{failed to fold operation some.op(1 : index)}}
  %0 = "some.op"(%arg0) : (index) -> index
  kgen.return %0 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator @interp_func() {
  // expected-note @below {{failed to compile-time evaluate function call}}
  %0 = kgen.param.constant = <apply(:(index) -> index @cant_interpret, 1)>
  kgen.return
}

// -----

// expected-note @below {{failed to interpret function @fails_to_interpret}}
kgen.generator @fails_to_interpret() {
  // expected-note @below {{failed to fold operation some.op()}}
  "some.op"() : () -> ()
  kgen.return
}

// expected-note @below {{failed to interpret function @passthrough}}
kgen.generator @passthrough() -> index {
  // expected-note @below {{failed to evaluate call}}
  kgen.call @fails_to_interpret() : () -> ()
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator @call_it() {
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.constant = <apply(:() -> index @passthrough)>
  kgen.return
}


// -----

// expected-error @below {{function instantiation failed}}
kgen.generator @brokenVLenAssert() {
  kgen.param.declare B : !kgen.string = <"foo">

  // expected-note @+1 {{constraint failed: foo}}
  kgen.param.assert <eq(2, 3)>, B
  kgen.return
}

// -----

// COM: Unused `kgen.param.declare` should not be ignored.

// expected-note @below {{function instantiation failed}}
kgen.generator @fail_if_zero<value>() -> index {
  %0 = index.constant 0
  // expected-note @below {{constraint failed: must not be zero!}}
  kgen.param.assert <ne(value, 0)>, "must not be zero!"
  kgen.return %0 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator @unused_param_declare() {
  kgen.param.declare unused = <apply(:() -> index bind_params(:<index>() -> index @fail_if_zero, 0))>
  kgen.return
}

// -----

// expected-error @below {{function instantiation failed}}
kgen.generator @invalid_rebind(%arg0: !kgen.scalar<si32>) {
  kgen.param.declare dt: dtype = <ui32>
  // expected-note @below {{error: rebind input type '!kgen.scalar<si32>' does not match result type '!kgen.scalar<ui32>'}}
  %0 = kgen.rebind %arg0 : !kgen.scalar<si32> to !kgen.scalar<dt>
  kgen.return
}

// -----

// A rebind is checked even when only its input type is parametric.

// expected-note @below {{function instantiation failed}}
kgen.generator @invalid_rebind_parametric_input<p>(
    %arg0: !kgen.pointer<array<p, i8>>) -> !kgen.pointer<array<2, i8>> {
  // expected-note @below {{error: rebind input type '!kgen.pointer<array<1, i8>>' does not match result type '!kgen.pointer<array<2, i8>>'}}
  %0 = kgen.rebind %arg0 : !kgen.pointer<array<p, i8>> to !kgen.pointer<array<2, i8>>
  kgen.return %0 : !kgen.pointer<array<2, i8>>
}

// expected-error @below {{function instantiation failed}}
kgen.generator @invalid_rebind_parametric_input_caller(
    %arg0: !kgen.pointer<array<1, i8>>) {
  // expected-note @below {{call expansion failed with parameter value(s): ("p": 1)}}
  %0 = kgen.call @invalid_rebind_parametric_input<1>(%arg0) : (!kgen.pointer<array<1, i8>>) -> !kgen.pointer<array<2, i8>>
  kgen.return
}

// -----

// expected-note @below {{failed to interpret function @fails}}
kgen.generator @fails() -> index {
  // expected-note @below {{failed to fold operation kgen.unreachable()}}
  kgen.unreachable
}

// expected-error @below {{function instantiation failed}}
kgen.generator @failed_apply() {
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.apply value = [() -> index: @fails]()
  kgen.param.constant = <value>
  kgen.return
}

// -----

// expected-error @below {{function instantiation failed}}
kgen.generator @failed_param_rebind() {
  // expected-note @below {{rebind input type 'i64' does not match result type 'i32'}}
  kgen.param.declare value: i32 = <rebind(:i64 2)>
  kgen.return
}

// -----

kgen.generator @function<param>() {
  kgen.return
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @invalid_param_ref() {
  // expected-note @below {{cannot reference parametric function}}
  kgen.cost_of[<index>() -> (): @function]
  kgen.return
}

// -----

// expected-error @below {{function instantiation failed}}
kgen.generator export @recursive() -> index {
  // expected-note @below {{function instantiation in parameter domain that recursively requires itself}}
  // expected-note @below {{function recursively calls itself in the parameter domain}}
  kgen.param.apply x = [() -> index: @recursive]()
  %0 = kgen.param.constant = <x>
  kgen.return %0 : index
}

// -----

// expected-error @below {{function instantiation failed}}
kgen.generator export @recursive0() -> index {
  // expected-note @below {{function instantiation in parameter domain that recursively requires itself}}
  // expected-note @below {{back to parameter domain function call here}}
  kgen.param.apply x = [() -> index: @recursive1]()
  %0 = kgen.param.constant = <x>
  kgen.return %0 : index
}

kgen.generator @recursive1() -> index {
  // expected-note @below {{recursively instantiated through here}}
  %0 = kgen.call @recursive0() : () -> index
  kgen.return %0 : index
}


// -----
// COM: MOCO-964 fix.
// expected-error @below {{function instantiation failed}}
kgen.generator @will_fail() {
  kgen.param.declare B : !kgen.string = <"foo">

  // expected-note @+1 {{constraint failed: foo}}
  kgen.param.assert <eq(2, 3)>, B

  kgen.return
}

kgen.generator @will_pass<a, b>() -> (index, index) {
  %0 = kgen.param.constant = <a>
  %1 = kgen.param.constant = <b>
  kgen.return %0, %1 : index, index
}

!capture = !kgen.struct<(string, index, (!kgen.pointer<pointer<none>>) capturing -> !kgen.none)>

// expected-error @below {{function instantiation failed}}
kgen.generator export @main() {
  // expected-note @+1  {{failed to run the pass manager}}
  %0 = kgen.param.constant: !capture = <#kgen.compile_assembly<current_target(), =asm, "", false, :() -> () @will_fail>>
  %1 = kgen.param.constant: !capture = <#kgen.compile_assembly<current_target(), =asm, "", false, :() -> (index, index) @will_pass<3, 4>>>
  kgen.return
}

// -----

// Illegal recursion hidden behind struct type instantiation.

// expected-note @below {{function instantiation failed}}
kgen.generator @recursive() -> index {
  // expected-note @below {{function instantiation in parameter domain that recursively requires itself}}
  // expected-note @below {{function recursively calls itself in the parameter domain}}
  kgen.param.apply x = [() -> index: @recursive]()
  %0 = kgen.param.constant = <x>
  kgen.return %0 : index
}

// expected-note @below {{function instantiation failed}}
kgen.struct.generator @WeirdStruct<T: type> = struct_inst<"WeirdStruct"(data: array<apply(:() -> index @recursive), index>)>

kgen.generator @use_type<T: type>() {
  kgen.return
}

#weird_struct = #kgen.type<typevalue<#kgen.genref<@WeirdStruct<:type index>>>, struct<(array<2, index>)>> : !kgen.type

// expected-error @below {{function instantiation failed}}
kgen.generator export @gen_structs() {
  // expected-note @below {{call expansion failed}}
  kgen.call @use_type<:type #weird_struct>() : () -> ()
  kgen.return
}

// -----

// expected-note @below {{function instantiation failed}}
// expected-note @below {{cannot concretize name in 'llvm_metadata'}}
kgen.generator export @metadata<x>() attributes {LLVMMetadataArray = [
  #pop.array<x> : !pop.array<1, index>,  #pop.array<x> : !pop.array<1, index>
]}{
  kgen.return
}

// expected-error @below {{function instantiation failed}}
kgen.generator @metadata_caller() {
  // expected-note @below {{call expansion failed}}
  kgen.call @metadata<2>() : () -> ()
  kgen.return
}

// -----

// COM: test displaying trivial parameter values with call expansion failures.
// expected-note @below {{function instantiation failed}}
kgen.generator @fn1<a, b>() {
  // expected-note @+1  {{constraint failed: must be equal!}}
  kgen.param.assert <eq(a, b)>, "must be equal!"
  kgen.return
}

// expected-note @below {{function instantiation failed}}
kgen.generator @fn2<a, b>() {
  // expected-note @+1 {{call expansion failed with parameter value(s): ("a": 2, "b": 4)}}
  kgen.call @fn1<a, b>() : () -> ()
  kgen.return
}

// expected-note @below {{function instantiation failed}}
kgen.generator @fn3<a, b>() {
  // expected-note @+1 {{call expansion failed with parameter value(s): ("a": 2, "b": 4)}}
  kgen.call @fn2<a, b>() : () -> ()
  kgen.return
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @main() {
  // expected-note @+1 {{call expansion failed with parameter value(s): ("a": 2, "b": 4)}}
  kgen.call @fn3<2, 4>() : () -> ()
  kgen.return
}

// -----

kgen.generator @g<T: i1>() -> index {
  // expected-note @+1 {{call expansion failed with parameter value(s): ("T": true)}}
  %0 = kgen.call @f<:i1 T>() : () -> index
  kgen.return %0 : index
}

kgen.generator @f<T: i1>() -> index {
  %0 = kgen.param.constant = <42>
  // expected-note @+1 {{codegen unreachable: materializing code that is not codegen reachable is not allowed}}
  kgen.codegen.reachable <not(T)>, "materializing code that is not codegen reachable is not allowed"
  kgen.return %0 : index
}

kgen.generator export @main() {
  // expected-error @+1 {{call expansion failed}}
  %0 = kgen.call @g<:i1 1>() : () -> index
  kgen.return
}

// -----
// COM: MOCO-2892 unbound parameter causing interpret crash fix.
// expected-note @below {{struct not a writeable type, got #kgen.unbound}}
kgen.generator @fn(%arg0: !kgen.pointer<struct<(index) memoryOnly>> imm_mem, %arg1: index) -> index {
  %0 = kgen.struct.gep %arg0[0]: <struct<(index) memoryOnly>>
  %1 = pop.load %0: !kgen.pointer<index>
  %2 = index.add %1, %arg1
  kgen.return %2: index
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @main() -> index {
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.apply x = [(!kgen.pointer<struct<(index) memoryOnly>>, index) -> index: @fn](store_to_mem(?), 1)
  %0 = kgen.param.constant = <x>
  kgen.return %0: index
}

// -----

// expected-error @below {{function instantiation failed}}
kgen.generator export @illegal_type_name() {
  // expected-note @below {{'get_type_name' requires a concrete type}}
  kgen.param.constant: string = <#kgen.get_type_name<:!kgen.struct<()> #kgen.struct<> , #kgen.simd<false>:!kgen.scalar<bool>>>
  kgen.return
}

// -----

// Test struct_field_index_by_name with nonexistent field.

kgen.struct.generator @FieldTestStruct = struct_inst<"FieldTestStruct"(first: index, second: i32)>

#field_test_struct = #kgen.type<typevalue<:!kgen.type #kgen.genref<@FieldTestStruct>>, struct<(index, i32)>> : !kgen.type

// expected-error @below {{function instantiation failed}}
kgen.generator @test_field_not_found() {
  // expected-note @below {{struct 'FieldTestStruct' has no field named 'nonexistent'}}
  kgen.param.constant: index = <#kgen.struct_field_index_by_name<#field_test_struct, "nonexistent">>
  kgen.return
}

// -----

// Test struct_field_type_by_name with nonexistent field.

kgen.struct.generator @FieldTypeTestStruct = struct_inst<"FieldTypeTestStruct"(alpha: f32, beta: f64)>

#field_type_test_struct = #kgen.type<typevalue<:!kgen.type #kgen.genref<@FieldTypeTestStruct>>, struct<(f32, f64)>> : !kgen.type

// expected-error @below {{function instantiation failed}}
kgen.generator @test_field_type_not_found() {
  // expected-note @below {{struct 'FieldTypeTestStruct' has no field named 'missing'}}
  kgen.param.constant: type = <#kgen.struct_field_type_by_name<#field_type_test_struct, "missing">>
  kgen.return
}

// -----

// Test struct_field_types with non-struct type (passing a primitive type i32).

// expected-error @below {{function instantiation failed}}
kgen.generator @test_non_struct_type() {
  // expected-note @+1 {{struct_field_types requires a struct type}}
  kgen.param.constant: param_list<type> = <#kgen.struct_field_types<i32>>
  kgen.return
}

// -----

// Test struct type with illegal alignment (not a power of 2).

// expected-note @below {{function instantiation failed}}
kgen.generator @test_illegal_struct_alignment<my_align>() {
  // expected-note @below {{struct alignment must be a positive power of 2, got 3}}
  %0 = pop.stack_allocation 1 x !kgen.struct<(index) align(my_align)>
  kgen.return
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @main() {
  // expected-note @below {{call expansion failed with parameter value(s): ("my_align": 3)}}
  kgen.call @test_illegal_struct_alignment<3>() : () -> ()
  kgen.return
}

// -----

// Test struct type with alignment exceeding maximum (2^29).

// expected-error @below {{function instantiation failed}}
// expected-note @below {{struct alignment exceeds maximum alignment (2^29), got 1073741824}}
kgen.generator @test_excessive_struct_alignment(%arg0: !kgen.pointer<!kgen.struct<(index) align(1073741824)>>) {
  kgen.return
}

// -----

kgen.generator @sum_from_zero<upper>() -> index {
  %idx0 = index.constant 0
  // expected-warning @below {{comptime for unrolling loop more than 27 times may cause long compilation time and large code size. (use '--loop-unrolling-warn-threshold' to increase the threshold or set to `0` to disable this warning}}
  %0 = kgen.param.for i in upper
    has_next :(index) -> i1 @count_to_zero_has_next
    get_next_iter :(!kgen.pointer<index> imm_mem, !kgen.pointer<index> byref_result) -> !kgen.none @count_to_zero
    (%arg0 = %idx0 : index) -> index {
    kgen.unreachable
  } else {
    kgen.unreachable
  }
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @param_for
kgen.generator export @param_for(%arg0: i1, %arg1: index) {
  kgen.call @sum_from_zero<33>() : () -> index
  kgen.return
}

kgen.generator @count_to_zero(%arg0: !kgen.pointer<index> imm_mem, %arg1: !kgen.pointer<index> byref_result) -> !kgen.none {
  %i0 = pop.load %arg0 : !kgen.pointer<index>
  %idx1 = index.constant 1
  %1 = index.sub %i0, %idx1
  pop.store %1, %arg1 : !kgen.pointer<index>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator @count_to_zero_has_next(%arg0: index) -> !kgen.scalar<bool> {
  %idx0 = index.constant 0
  %0 = index.cmp ne(%idx0, %arg0)
  %1 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
  kgen.return %1 : !kgen.scalar<bool>
}

// -----

// Two generators with the same linkage name should report a clash error.

// expected-remark @below {{existing function here}}
kgen.generator @"clash::a"() attributes {linkageName = #kgen.linkage_name<"foo" : !kgen.string, false>} {
  kgen.return
}

// expected-error @below {{duplicate functions named "foo"}}
kgen.generator @"clash::b"() attributes {linkageName = #kgen.linkage_name<"foo" : !kgen.string, false>} {
  kgen.return
}

// -----

// Test that a negative alloc size is rejected by the interpreter.

// expected-note @below {{failed to interpret function @negative_alloc_size}}
kgen.generator @negative_alloc_size() -> !kgen.pointer<index> {
  %idx8 = index.constant 8
  %idx_neg = index.constant -1
  // expected-note @below {{failed to interpret operation pop.aligned_alloc}}
  // expected-note @below {{alloc has negative size}}
  %0 = pop.aligned_alloc %idx8, %idx_neg : <index>
  kgen.return %0 : !kgen.pointer<index>
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @use_negative_alloc_size() {
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.constant: !kgen.pointer<index> = <apply(:() -> !kgen.pointer<index> @negative_alloc_size)>
  kgen.return
}

// -----

// Test that SIMD-typed parameter values are printed in call expansion failure
// messages (MOCO-3651).

// expected-note @below {{function instantiation failed}}
kgen.generator @simd_param_inner<a: !kgen.scalar<si32>>() {
  // expected-note @below {{constraint failed: always fails}}
  kgen.param.assert <false>, "always fails"
  kgen.return
}

// expected-note @below {{function instantiation failed}}
kgen.generator @simd_param_outer<a: !kgen.scalar<si32>>() {
  // expected-note @below {{call expansion failed with parameter value(s): ("a": 42)}}
  kgen.call @simd_param_inner<a>() : () -> ()
  kgen.return
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @simd_param_main() {
  // expected-note @below {{call expansion failed with parameter value(s): ("a": 42)}}
  kgen.call @simd_param_outer<:!kgen.scalar<si32> #kgen<simd 42>>() : () -> ()
  kgen.return
}

// -----

// Identity against an unknown value stays symbolic, so it cannot be
// concretized during elaboration.

// expected-note @below {{function instantiation failed}}
kgen.generator @identical_to_unknown<T: type, value: !kgen.param<T>>() {
  // expected-note @below {{could not prove whether 1 and *? are the same value}}
  kgen.param.constant: scalar<bool> = <identical(:!kgen.param<T> value, *?)>
  kgen.return
}

// expected-error @below {{function instantiation failed}}
kgen.generator @check_identical() {
  // expected-note @below {{call expansion failed with parameter value(s): (..., "value": 1)}}
  kgen.call @identical_to_unknown<:type i32, :i32 1>() : () -> ()
  kgen.return
}

// -----

// The same for a class of more than two, which names every operand it could not
// relate rather than just the first pair.

// expected-note @below {{function instantiation failed}}
kgen.generator @nary_identical_to_unknown<T: type, value: !kgen.param<T>>() {
  // expected-note @below {{could not prove whether 1, *? and *? are the same value}}
  kgen.param.constant: scalar<bool> = <identical(:!kgen.param<T> value, *?, *?)>
  kgen.return
}

// expected-error @below {{function instantiation failed}}
kgen.generator @check_nary_identical() {
  // expected-note @below {{call expansion failed with parameter value(s): (..., "value": 1)}}
  kgen.call @nary_identical_to_unknown<:type i32, :i32 1>() : () -> ()
  kgen.return
}

// -----

// An unknown does not collapse into the operands around it even once a target
// settles those against each other. 2^32 and 2^33 both truncate to 0 at this
// width, so every operand the target can key agrees; the class would fold
// `true` if the unknown were read as "whatever the others are".

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// expected-error @below {{function instantiation failed}}
kgen.generator export @nary_identical_unknown_with_target() -> !kgen.scalar<bool> {
  // expected-note @below {{could not prove whether 4294967296, 8589934592 and *? are the same value}}
  kgen.param.declare value : !kgen.scalar<bool> = <#kgen.param.identical<#kgen.unknown : !kgen.scalar<index>, #kgen<simd 4294967296> : !kgen.scalar<index>, #kgen<simd 8589934592> : !kgen.scalar<index>>>
  %0 = kgen.param.constant: !kgen.scalar<bool> = <value>
  kgen.return %0 : !kgen.scalar<bool>
}
}
