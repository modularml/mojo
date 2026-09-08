// RUN: kgen-opt %s -elaborate-generators="max-depth=64 use-parametric-interpret=true" -verify-diagnostics -split-input-file -allow-unregistered-dialect


// Recursive expansions.
// TODO: RECURSION ERROR HANDLING
// expected-error @below {{function instantiation failed}}
// expected-note-re @below {{elaborator expansion is {{[0-9]+}} levels deep - infinite recursion?}}
kgen.generator @genItf3<x>() {
  // TODO: should check error here @+1 {{call expansion failed}}
  kgen.call @genItf3<add(x, 1)>() : () -> ()
  kgen.return
}

kgen.generator export @use_Itf3two() {
  // TODO: should check note here @+1 {{call expansion failed}}
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

kgen.generator @interp_func() {
  // expected-error @below {{failed to compile-time evaluate function call}}
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

kgen.generator @call_it() {
  // expected-error @below {{failed to compile-time evaluate function call}}
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
// expected-note @below {{failed to interpret function}}
kgen.generator @fail_if_zero<value>() -> index {
  %0 = index.constant 0
  // expected-note @below {{failed to interpret operation}}
  // expected-note @below {{constraint failed: must not be zero!}}
  kgen.param.assert <ne(value, 0)>, "must not be zero!"
  kgen.return %0 : index
}

kgen.generator @unused_param_declare() {
// expected-error @below {{failed to compile-time evaluate function call}}
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

// expected-error @below {{function instantiation failed}}
kgen.generator @invalid_rebind_parametric_input<p>(
    %arg0: !kgen.pointer<array<p, i8>>) -> !kgen.pointer<array<2, i8>> {
  // expected-note @below {{error: rebind input type '!kgen.pointer<array<1, i8>>' does not match result type '!kgen.pointer<array<2, i8>>'}}
  %0 = kgen.rebind %arg0 : !kgen.pointer<array<p, i8>> to !kgen.pointer<array<2, i8>>
  kgen.return %0 : !kgen.pointer<array<2, i8>>
}

kgen.generator @invalid_rebind_parametric_input_caller(
    %arg0: !kgen.pointer<array<1, i8>>) {
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

// expected-note @below {{failed to interpret function }}
// expected-note-re @below {{error recurses {{[0-9]+}} times}}
// expected-note @below {{remaining errors after}}
// expected-note-re @below {{interpreter is {{[0-9]+}} levels deep - infinite recursion?}}
// expected-error @below {{function instantiation failed}}
kgen.generator export @recursive() -> index {
  // expected-note @below {{failed to compile-time evaluate function call}}
  // expected-note @below {{failed to interpret operation}}
  kgen.param.apply x = [() -> index: @recursive]()
  %0 = kgen.param.constant = <x>
  kgen.return %0 : index
}

// -----


// expected-note @below {{failed to interpret function}}
// expected-note-re @below {{interpreter is {{[0-9]+}} levels deep - infinite recursion?}}
// expected-error @below {{function instantiation failed}}
kgen.generator export @recursive0() -> index {
  // expected-note @below {{failed to compile-time evaluate function call}}
  // expected-note @below {{failed to interpret operation}}
  kgen.param.apply x = [() -> index: @recursive1]()
  %0 = kgen.param.constant = <x>
  kgen.return %0 : index
}

// expected-note-re @below {{error recurses {{[0-9]+}} times}}
// expected-note @below {{failed to interpret function}}
// expected-note @below {{remaining errors after}}
kgen.generator @recursive1() -> index {
  // expected-note @below {{failed to evaluate call}}
  // expected-note @below {{failed to interpret operation}}
  %0 = kgen.call @recursive0() : () -> index
  kgen.return %0 : index
}


// -----
// COM: MOCO-964 fix.
// expected-error @+1 {{function instantiation failed}}
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
  %1 = kgen.param.constant: !capture = <#kgen.compile_assembly<current_target(), =asm, "", false, :() -> (index, index) @will_pass<2, 3>>>
  kgen.return
}

// -----

// Illegal recursion hidden behind struct type instantiation.

// expected-note @below {{failed to interpret function}}
// expected-note-re @below {{error recurses {{[0-9]+}} times}}
// expected-note @below {{remaining errors after}}
// expected-note-re @below {{interpreter is {{[0-9]+}} levels deep - infinite recursion?}}
kgen.generator @recursive() -> index {
  // expected-note @below {{failed to interpret operation}}
  kgen.param.apply x = [() -> index: @recursive]()
  %0 = kgen.param.constant = <x>
  kgen.return %0 : index
}

// expected-error @below {{failed to compile-time evaluate function call}}
kgen.struct.generator @WeirdStruct<T: type> = struct_inst<"WeirdStruct"(data: array<apply(:() -> index @recursive), index>)>

kgen.generator @use_type<T: type>() {
  kgen.return
}

#weird_struct = #kgen.type<typevalue<#kgen.genref<@WeirdStruct<:type index>>>, struct<(array<2, index>)>> : !kgen.type

kgen.generator export @gen_structs() {
  kgen.call @use_type<:type #weird_struct>() : () -> ()
  kgen.return
}

// -----
// expected-error @below {{function instantiation failed}}
// expected-note @below {{cannot concretize name in 'llvm_metadata'}}
kgen.generator export @metadata<x>() attributes {LLVMMetadataArray = [
  #pop.array<x> : !pop.array<1, index>,  #pop.array<x> : !pop.array<1, index>
]}{
  kgen.return
}

kgen.generator @metadata_caller() {
  // TODO: should check error here @below {{call expansion failed}}
  kgen.call @metadata<2>() : () -> ()
  kgen.return
}

// -----

// COM: TODO - this error message currently is missing
//             the complete call graph info.
kgen.generator @g<T: i1>() -> index {
  %0 = kgen.call @f<:i1 T>() : () -> index
  kgen.return %0 : index
}

// expected-error @+1 {{function instantiation failed}}
kgen.generator @f<T: i1>() -> index {
  %0 = kgen.param.constant = <42>
  // expected-note @+1 {{codegen unreachable: materializing code that is not codegen reachable is not allowed}}
  kgen.codegen.reachable <not(T)>, "materializing code that is not codegen reachable is not allowed"
  kgen.return %0 : index
}

kgen.generator export @main() {
  %0 = kgen.call @g<:i1 1>() : () -> index
  kgen.return
}

// -----

// Test struct_field_index_by_name with nonexistent field.

kgen.struct.generator @TestStruct = struct_inst<"TestStruct"(first: index, second: i32)>

#test_struct = #kgen.type<typevalue<:!kgen.type #kgen.genref<@TestStruct>>, struct<(index, i32)>> : !kgen.type

// expected-error @below {{function instantiation failed}}
kgen.generator @test_field_not_found() {
  // expected-note @below {{struct 'TestStruct' has no field named 'nonexistent'}}
  kgen.param.constant: index = <#kgen.struct_field_index_by_name<#test_struct, "nonexistent">>
  kgen.return
}

// -----

// Test struct_field_type_by_name with nonexistent field.

kgen.struct.generator @TestStruct2 = struct_inst<"TestStruct2"(alpha: f32, beta: f64)>

#test_struct2 = #kgen.type<typevalue<:!kgen.type #kgen.genref<@TestStruct2>>, struct<(f32, f64)>> : !kgen.type

// expected-error @below {{function instantiation failed}}
kgen.generator @test_field_type_not_found() {
  // expected-note @below {{struct 'TestStruct2' has no field named 'missing'}}
  kgen.param.constant: type = <#kgen.struct_field_type_by_name<#test_struct2, "missing">>
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
