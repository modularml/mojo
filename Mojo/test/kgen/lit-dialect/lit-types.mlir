// RUN: kgen-opt %s -allow-unregistered-dialect -verify-parameters --kgen-print-inline-type-values | FileCheck %s
// RUN: kgen-opt %s -emit-bytecode -allow-unregistered-dialect | kgen-opt -allow-unregistered-dialect --kgen-print-inline-type-values | FileCheck %s

lit.struct.decl @MyStruct {}
lit.struct.decl @MyStructParams<a, b: dtype, c: type> {}

// CHECK-LABEL: @UseStruct
// CHECK-SAME: !lit.struct<@MyStructParams<a, :dtype b, :type c>>
kgen.generator @UseStruct<a, b: dtype, c: type>(%arg0: !lit.struct<@MyStructParams<a, :dtype b, :type c>>) {
  kgen.return
}

// CHECK-LABEL: @metatype
// CHECK-SAME: !kgen.pointer<!lit.struct<@MyStruct>>
kgen.generator @metatype(%arg0: !kgen.pointer<!lit.struct<@MyStruct>>) {
  kgen.return
}

// CHECK-LABEL: @declref_metatype
// CHECK-SAME: !lit.struct<@MyStruct>
kgen.generator @declref_metatype(%arg0: !lit.struct<@MyStruct>) {
  kgen.return
}

// CHECK-LABEL: lit.trait.decl @TParam<MT: type, T: !kgen.param<MT>>
lit.trait.decl @TParam<MT: type, T: !kgen.param<MT>> {
  // CHECK-NEXT: lit.fn @f(%self: !kgen.param<:!kgen.param<MT> T>) -> !kgen.none
  lit.fn @f(%self: !kgen.param<:!kgen.param<MT> T>) -> !kgen.none {
    kgen.unreachable
  }
}

lit.trait.decl @MyTrait {}

// CHECK-LABEL: @trait_metatype
// CHECK-SAME: !kgen.param<:trait<@MyTrait> T>
lit.fn @trait_metatype<T: trait<@MyTrait>>(%arg0: !kgen.param<:trait<@MyTrait> T>) {
  kgen.return
}

// CHECK: !lit.type_signature
"type.sig"() : () -> !lit.type_signature
// CHECK: !lit.type_signature<index, |>
"type.sig"() : () -> !lit.type_signature<index, |>
// CHECK: !lit.type_signature<"dt": dtype = f32>
"type.sig"() : () -> !lit.type_signature<"dt": dtype = f32>
// CHECK: !lit.type_signature<"i": param_list<index> pos_vararg>
"type.sig"() : () -> !lit<type_signature<"i": param_list<index> pos_vararg>>

// CHECK-LABEL: @type_sig
// CHECK-SAME: !lit.type_signature<index, array<*(0,0), index>>
kgen.generator @type_sig(%arg0: !lit.type_signature<index, array<*(0,0), index>>) {
  kgen.return
}

kgen.generator @nested_index<a>(%arg0: !lit.type_signature<index, index = *(0,0)>) {
  kgen.return
}

kgen.generator @subst_type<T: type>(%arg0: !kgen.param<T>) {
  kgen.return
}

kgen.generator @return_type() -> !lit.type_signature<index, index = *(0,0)> {
  kgen.unreachable
}

kgen.generator @return_sig() -> !lit.generator<<index, index = *(0,0)>() -> ()> {
  kgen.unreachable
}

// CHECK-LABEL: @bind_nested
kgen.generator @bind_nested() {
  // CHECK: bound0: (!lit.type_signature<index, index = *(0,0)>) -> () = <@nested_index<1>>
  kgen.param.declare bound0: (!lit.type_signature<index, index = *(0,0)>) -> () = <@nested_index<1>>
  // CHECK: bound1: (!lit.type_signature<index, index = *(0,0)>) -> () = <@subst_type<:type !lit.type_signature<index, index = *(0,0)>>>
  kgen.param.declare bound1: (!lit.type_signature<index, index = *(0,0)>) -> () = <@subst_type<:type !lit.type_signature<index, index = *(0,0)>>>
  // CHECK: result0: !lit.type_signature<index, index = *(0,0)> = <apply(:() -> !lit.type_signature<index, index = *(0,0)> @return_type)>
  kgen.param.declare result0: !lit.type_signature<index, index = *(0,0)> = <apply(:() -> !lit.type_signature<index, index = *(0,0)> @return_type)>
  // CHECK: result1: !lit.generator<<index, index = *(0,0)>() -> ()> = <apply(:() -> !kgen.generator<!lit.generator<<index, index = *(0,0)>() -> ()>> @return_sig)>
  kgen.param.declare result1: !lit.generator<<index, index = *(0,0)>() -> ()> = <apply(:() -> !lit.generator<<index, index = *(0,0)>() -> ()> @return_sig)>
  kgen.return
}

// CHECK-LABEL: @passing_kinds
kgen.generator @passing_kinds(
    // CHECK-SAME: !lit.generator<<i8, |, i8, *, i8, ?, i8>() -> ()>
    %arg0: !lit.generator<<i8, |, i8, *, i8, ?, i8>() -> ()>,
    // CHECK-SAME: !lit.generator<<i8, i8, *, i8, ?, i8>() -> ()>
    %arg1: !lit.generator<<i8, i8, *, i8, ?, i8>() -> ()>,
    // CHECK-SAME: !lit.generator<<i8, |, i8, i8, ?, i8>() -> ()>
    %arg2: !lit.generator<<i8, |, i8, i8, ?, i8>() -> ()>,
    // CHECK-SAME: !lit.generator<<i8, i8, i8, ?, i8>() -> ()>
    %arg3: !lit.generator<<i8, i8, i8, ?, i8>() -> ()>
) {
  kgen.return
}

// Test that passing kind printing/parsing works correctly for long signatures.
!t = !kgen.generator<!lit.generator<(
    "a": index owned, |, *,
    "b": index owned, "c": index owned, "d": index owned, "e": index owned,
    "f": index owned, "g": index owned, "h": index owned, "i": index owned,
    "j": index owned, "k": index owned, "l": index owned, "m": index owned
) -> !kgen.none>>

// CHECK-LABEL: lit.fn @long_sig(%t: !kgen.generator<!lit.generator<
// CHECK-SAME: "a": index owned, |, *,
// CHECK-SAME: "b": index owned, "c": index owned, "d": index owned, "e": index owned,
// CHECK-SAME: "f": index owned, "g": index owned, "h": index owned, "i": index owned,
// CHECK-SAME: "j": index owned, "k": index owned, "l": index owned, "m": index owned
lit.fn @long_sig(%t: !t) {
    kgen.return
}

lit.trait.decl @Trait {
}

kgen.generator @method() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @trait
kgen.generator @trait() {
  // CHECK-NEXT: trait<@Trait> = <@MyStructParams<1, :dtype f32, :type i32>>
  kgen.param.declare type: trait<@Trait> = <@MyStructParams<1, :dtype f32, :type i32>>
  // CHECK-NEXT: trait<@Trait> = <@MyStructParams<1, :dtype f32, :type i32>>
  kgen.param.declare type_same: trait<@Trait> = <[@MyStructParams<1, :dtype f32, :type i32>, @MyStructParams<1, :dtype f32, :type i32>]>
  // CHECK-NEXT: trait<@Trait> = <[@MyStructParams<1, :dtype f32, :type i32>, @MyStructParams<2, :dtype f64, :type i64>]>
  kgen.param.declare type_diff: trait<@Trait> = <[@MyStructParams<1, :dtype f32, :type i32>, @MyStructParams<2, :dtype f64, :type i64>]>
  kgen.return
}

// CHECK-LABEL: kgen.generator @ref_types
// CHECK: %arg0: !lit.ref<index, imm #lit.any.origin, 4>
kgen.generator @ref_types(%arg0: !lit.ref<index, imm #lit.any.origin, 4>) {
  kgen.unreachable
}

// FIXME: This isn't valid IR!
// CHECK-LABEL: kgen.generator @escape_meta_type_param_names<type: meta<!lit.struct<@Trait>>>()
kgen.generator @escape_meta_type_param_names<type: meta<!lit.struct<@Trait>>>() {
  // CHECK: kgen.param.declare T: meta<!lit.struct<@Trait>> = <type>
  kgen.param.declare T: meta<!lit.struct<@Trait>> = <*"type">
  kgen.return
}

// CHECK-LABEL: kgen.generator @escape_trait_param_names<type: trait<@Trait>>()
kgen.generator @escape_trait_param_names<type: trait<@Trait>>() {
  // CHECK: kgen.param.declare T: trait<@Trait> = <*"type">
  kgen.param.declare T: trait<@Trait> = <*"type">
  kgen.return
}

// CHECK-LABEL: kgen.func @generator_types
kgen.func @generator_types() {
  // CHECK-NEXT: type = <!lit.generator<<>none>>
  kgen.param.declare empty: type = <!lit.generator<<>none>>
  // CHECK-NEXT: type = <!lit.generator<<dtype>index>>
  kgen.param.declare one_arg: type = <!lit.generator<<dtype>index>>
  // CHECK-NEXT: type = <!lit.generator<<"dt": dtype = f32>index>>
  kgen.param.declare one_arg_named: type = <!lit.generator<<"dt": dtype = f32>index>>
  // CHECK-NEXT: type = <!lit.generator<<"dt": dtype, "width": index>index>>
  kgen.param.declare more_args: type = <!lit.generator<<"dt": dtype, "width": index>index>>
  // CHECK-NEXT: type = <!lit.generator<<"dt": dtype, |, "width": index, *, "tag": i1>index>>
  kgen.param.declare arg_kinds: type = <!lit.generator<<"dt": dtype, |, "width": index, *, "tag": i1>index>>
  // CHECK-NEXT: type = <!lit.generator<<dtype, {<true, {{.*}}>, <true, {{.*}}>}>index>>
  kgen.param.declare body_constraint: type = <!lit.generator<<dtype, {<true, loc("body-gen.mlir":1:1)>, <true, loc("body-gen.mlir":1:2)>}>index>>
  // CHECK-NEXT: type = <!lit.generator<<{<true, {{.*}}>, <true, {{.*}}>}>index>>
  kgen.param.declare body_constraint_no_params: type = <!lit.generator<<{<true, loc("body-gen.mlir":1:1)>, <true, loc("body-gen.mlir":1:2)>}>index>>
  kgen.return
}

// CHECK-LABEL: kgen.generator @fn_types
kgen.generator @fn_types<x: index, y: index>() {
  // CHECK-NEXT: type = <!lit.fn<() -> ()>>
  kgen.param.declare type0: type = <!lit.fn<() -> ()>>
  // CHECK-NEXT: type = <!lit.fn<(index, i8) -> ()>>
  kgen.param.declare type1: type = <!lit.fn<(index, i8) -> ()>>
  // CHECK-NEXT: type = <!lit.fn<(index, i8) -> ()>>
  kgen.param.declare type2: type = <!kgen.func<!lit.fn<(index, i8) -> ()>>>
  // CHECK-NEXT: type = <!lit.fn<("a": index, "b": i8 = 2) -> none>>
  kgen.param.declare type3: type = <!lit.fn<("a": index, "b": i8 = 2) -> none>>
  kgen.return
}

// CHECK-LABEL: kgen.generator @meta_type
kgen.generator @meta_type() {
  // CHECK: kgen.param.declare my_struct: meta<!lit.struct<@MyStruct>> = <@MyStruct>
  kgen.param.declare my_struct: meta<!lit.struct<@MyStruct>> = <@MyStruct>
  // CHECK: kgen.param.declare my_struct_params: meta<!lit.struct<@MyStructParams<1, :dtype f32, :type i32>>> = <@MyStructParams<1, :dtype f32, :type i32>>
  kgen.param.declare my_struct_params: meta<!lit.struct<@MyStructParams<1, :dtype f32, :type i32>>> = <@MyStructParams<1, :dtype f32, :type i32>>
  // CHECK: kgen.param.declare my_trait: meta<!lit.trait<@MyTrait>> = <@MyTrait>
  kgen.param.declare my_trait: meta<!lit.trait<@MyTrait>> = <@MyTrait>
  kgen.return
}

// Test TraitType with constraints (conditional trait conformance).
// Tests: unconditional, constrained, true-elision, false-erasure, and type
// identity across a dropped false slot.
// CHECK-LABEL: kgen.generator @trait_constraints
kgen.generator @trait_constraints<x: index>() {
  // Unconditional traits roundtrip without constraints.
  // CHECK: kgen.param.declare unconditional: type = <trait<@MyTrait, @Trait>>
  kgen.param.declare unconditional: type = <!lit.trait<@MyTrait, @Trait>>

  // Trait with non-trivial constraint is printed with `where`.
  // CHECK: kgen.param.declare conditional: type = <trait<@Trait where #kgen.constraint<{{.*}}>
  kgen.param.declare conditional: type = <!lit.trait<@Trait where #kgen.constraint<#kgen.param.expr<lt, 0 : index, #kgen.param.decl.ref<"x"> : index> : !kgen.scalar<bool>, loc("test.mlir":1:1)>>>

  // Literal true constraint is elided.
  // CHECK: kgen.param.declare literal_true: type = <trait<@MyTrait>>
  kgen.param.declare literal_true: type = <!lit.trait<@MyTrait where #kgen.constraint<true, loc("t":1:1)>>>

  // Foldable true constraint (gt(1, 0) = true) is also elided.
  // CHECK: kgen.param.declare foldable_true: type = <trait<@MyTrait>>
  kgen.param.declare foldable_true: type = <!lit.trait<@MyTrait where #kgen.constraint<#kgen.param.expr<lt, 0 : index, 1 : index> : !kgen.scalar<bool>, loc("t":1:2)>>>

  // A trivially-false slot is erased from both the symbol and constraint
  // arrays: the trait can never conform, exactly as if it were omitted. Only
  // the unconditional @Trait survives.
  // CHECK: kgen.param.declare false_removed: type = <trait<@Trait>>
  kgen.param.declare false_removed: type = <!lit.trait<@MyTrait where #kgen.constraint<false, loc("t":2:1)>, @Trait>>

  // Combined: foldable true (elided) + non-trivial (kept) + false (erased).
  // Result: @MyTrait unconstrained, @Trait keeps its constraint, @TParam gone.
  // CHECK: kgen.param.declare combined: type = <trait<@MyTrait, @Trait where #kgen.constraint<{{.*}}>>>
  kgen.param.declare combined: type = <!lit.trait<@MyTrait where #kgen.constraint<#kgen.param.expr<lt, 0 : index, 1 : index> : !kgen.scalar<bool>, loc("t":3:1)>, @Trait where #kgen.constraint<#kgen.param.expr<lt, 0 : index, #kgen.param.decl.ref<"x"> : index> : !kgen.scalar<bool>, loc("t":3:2)>, @TParam where #kgen.constraint<false, loc("t":3:3)>>>

  kgen.return
}
