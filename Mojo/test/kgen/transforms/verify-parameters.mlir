// RUN: kgen-opt %s -split-input-file -verify-parameters=simplify=true -kgen-print-inline-type-values | FileCheck %s

// CHECK-LABEL: no_constrains_deduplication
kgen.generator @no_constrains_deduplication() {
  kgen.param.declare cond = <1>
  kgen.param.if <eq(cond, 1)> {
    kgen.param.declare B0 : !kgen.string = <"foo">
    // CHECK: kgen.param.assert <false>, "foo"
    kgen.param.assert <eq(2, 3)>, B0
    kgen.return
  } else {
    kgen.param.declare B1 : !kgen.string = <"bar">
    // CHECK: kgen.param.assert <false>, "bar"
    kgen.param.assert <eq(2, 3)>, B1
    kgen.param.yield
  }
  kgen.param.declare B2 : !kgen.string = <"baz">
  // CHECK: kgen.param.assert <false>, "baz"
  kgen.param.assert <eq(2, 3)>, B2
  kgen.return
}

// -----

// Test get_witness contextual evaluation under LIT, which is required for
// checking the symbol use of `expect_associated_alias` in the
// `use_associated_alias` function is correct.

!Fooable = !lit.trait<@Fooable>

lit.trait.decl @Fooable<?, SELF: !Fooable> {
  lit.alias.decl MyType: !kgen.type
}

#wrapper_index = #kgen.type<!lit.struct<@Wrapper<:type index>>> : !Fooable

lit.struct.decl @"Wrapper"<T: type> {
  lit.struct.field data: !kgen.param<T>
  lit.alias.decl MyType: !kgen.type = <index>
  kgen.conformance @Fooable {
    kgen.witness "MyType" : !kgen.type = #kgen.type<index>
  }
}

// CHECK-LABEL: lit.fn @expect_associated_alias
// CHECK-SAME:    (%arg: !kgen.param<#kgen.get_witness<:trait<@Fooable> T, @Fooable, "MyType">>)
lit.fn @expect_associated_alias<T: !Fooable>(%arg: !kgen.param<#kgen.get_witness<:!Fooable T, @Fooable, "MyType">>) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}

// CHECK-LABEL: lit.fn @use_associated_alias
lit.fn @use_associated_alias(%arg: index) -> !kgen.none {
  // CHECK-NEXT: lit.call @expect_associated_alias<:trait<@Fooable> @Wrapper<:type index>>(%arg) : !lit.generator<("arg": index) -> !kgen.none>
  %none = lit.call @expect_associated_alias<:!Fooable #wrapper_index>(%arg) : !lit.generator<("arg": index) -> !kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}

// -----

// Test get_witness contextual evaluation under KGEN too.

kgen.struct.generator @Wrapper<T: type> = struct_inst<"Wrapper"[T]<:type T>(data: typevalue<T>)> {
  kgen.conformance @Fooable {
    kgen.witness "MyType" : !kgen.type = #kgen.type<index>
  }
}

#wrapper_index = #kgen.type<typevalue<#kgen.genref<@Wrapper<:type index>>>, struct<(index)>> : !kgen.type

// CHECK-LABEL: kgen.generator @expect_associated_alias
// CHECK-SAME:    (%arg0: !kgen.param<#kgen.get_witness<T, @Fooable, "MyType">>)
kgen.generator @expect_associated_alias<T: !kgen.type>(%arg: !kgen.param<#kgen.get_witness<T, @Fooable, "MyType">>) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator @use_associated_alias
kgen.generator @use_associated_alias(%arg: index) -> !kgen.none {
  // CHECK-NEXT: kgen.call @expect_associated_alias<:type [typevalue<#kgen.genref<@Wrapper<:type index>>>, struct<(index)>]>(%arg0)
  %none = kgen.call @expect_associated_alias<:!kgen.type #wrapper_index>(%arg) : !kgen.generator<(index) -> !kgen.none>
  kgen.return %none : !kgen.none
}

// -----

kgen.struct.generator @MyFoo<T: type> = struct_inst<"MyFoo"[T]<:type T>(data: typevalue<T>)> {
  kgen.conformance @Fooable {
    kgen.witness "foo" : (!kgen.param<:type T>) -> index = @myfoo_foo<:type T>
  }
}

kgen.generator @myfoo_foo<T: !kgen.type>(%arg: !kgen.param<:type T>) -> index {
  %answer = kgen.param.constant: index = <42>
  kgen.return %answer : index
}

#MyFooIndex = #kgen.type<typevalue<#kgen.genref<@MyFoo<:type index>>>, struct<(index)>> : !kgen.type

// CHECK-LABEL: kgen.generator @simplify_call_param
kgen.generator @simplify_call_param(%arg: index) -> index {
  // CHECK-NEXT: kgen.call @myfoo_foo<:type index>(%arg0)
  %result = kgen.call_param[(index) -> index: #kgen.get_witness<#MyFooIndex, @Fooable, "foo">](%arg)
  kgen.return %result : index
}

// -----

// Test struct field reflection attributes contextual evaluation under LIT.

lit.struct.decl @MyPair<T: type, U: type> {
  lit.struct.field first: !kgen.param<T>
  lit.struct.field second: !kgen.param<U>
}

lit.struct.decl @VariadicWrapper<T: type, V: !kgen.param_list<T>> {}

// CHECK-LABEL: lit.fn @expect_field_types_lit
// CHECK-SAME:    (%arg: !lit.struct<@VariadicWrapper<:type type, :param_list<type> #kgen.struct_field_types<T>>>
lit.fn @expect_field_types_lit<T: !kgen.type>(%arg: !lit.struct<@VariadicWrapper<:type type, :param_list<type> #kgen.struct_field_types<T>>>) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}

// CHECK-LABEL: lit.fn @expect_field_names_lit
// CHECK-SAME:    (%arg: !lit.struct<@VariadicWrapper<:type string, :param_list<string> #kgen.struct_field_names<T>>>)
lit.fn @expect_field_names_lit<T: !kgen.type>(%arg: !lit.struct<@VariadicWrapper<:type string, :param_list<string> #kgen.struct_field_names<T>>>) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}

// CHECK-LABEL: lit.fn @expect_field_index_by_name_lit
// CHECK-SAME:    (%arg: !kgen.simd<#kgen.struct_field_index_by_name<T, "second">, si32>)
lit.fn @expect_field_index_by_name_lit<T: !kgen.type>(%arg: !kgen.simd<#kgen.struct_field_index_by_name<T, "second">, si32>) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}

// CHECK-LABEL: lit.fn @expect_field_type_by_name_lit
// CHECK-SAME:    (%arg: !kgen.param<#kgen.struct_field_type_by_name<T, "first">>)
lit.fn @expect_field_type_by_name_lit<T: !kgen.type>(%arg: !kgen.param<#kgen.struct_field_type_by_name<T, "first">>) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}

#mypair_index_i32 = #kgen.type<!lit.struct<@MyPair<:type index, :type i32>>> : !kgen.type

!StructFieldTypesExpected = !lit.struct<@VariadicWrapper<:type !kgen.type, :!kgen.param_list<!kgen.type> #kgen.param_list<index, i32>>>
!StructFieldNamesExpected = !lit.struct<@VariadicWrapper<:type !kgen.string, :!kgen.param_list<!kgen.string> #kgen.param_list<"first", "second">>>
!StructFieldIndexByNameExpected = !kgen.simd<1, si32>
!StructFieldTypeByNameExpected = index

// CHECK-LABEL: lit.fn @use_struct_reflection_lit
lit.fn @use_struct_reflection_lit(%arg0: !StructFieldTypesExpected, %arg1: !StructFieldNamesExpected, %arg2: !StructFieldIndexByNameExpected, %arg3: !StructFieldTypeByNameExpected) -> !kgen.none {
  // CHECK-NEXT: lit.call @expect_field_types_lit<:type !lit.struct<@MyPair<:type index, :type i32>>>(%arg0) : !lit.generator<("arg": !lit.struct<@VariadicWrapper<:type type, :param_list<type> [index, i32]>>) -> !kgen.none>
  %none0 = lit.call @expect_field_types_lit<:!kgen.type #mypair_index_i32>(%arg0) : !lit.generator<("arg": !StructFieldTypesExpected) -> !kgen.none>
  // CHECK-NEXT: lit.call @expect_field_names_lit<:type !lit.struct<@MyPair<:type index, :type i32>>>(%arg1) : !lit.generator<("arg": !lit.struct<@VariadicWrapper<:type string, :param_list<string> ["first", "second"]>>) -> !kgen.none>
  %none1 = lit.call @expect_field_names_lit<:!kgen.type #mypair_index_i32>(%arg1) : !lit.generator<("arg": !StructFieldNamesExpected) -> !kgen.none>
  // CHECK-NEXT: lit.call @expect_field_index_by_name_lit<:type !lit.struct<@MyPair<:type index, :type i32>>>(%arg2) : !lit.generator<("arg": !kgen.scalar<si32>) -> !kgen.none>
  %none2 = lit.call @expect_field_index_by_name_lit<:!kgen.type #mypair_index_i32>(%arg2) : !lit.generator<("arg": !StructFieldIndexByNameExpected) -> !kgen.none>
  // CHECK-NEXT: lit.call @expect_field_type_by_name_lit<:type !lit.struct<@MyPair<:type index, :type i32>>>(%arg3) : !lit.generator<("arg": index) -> !kgen.none>
  %none3 = lit.call @expect_field_type_by_name_lit<:!kgen.type #mypair_index_i32>(%arg3) : !lit.generator<("arg": !StructFieldTypeByNameExpected) -> !kgen.none>
  lit.return %none3 : !kgen.none
  lit.end_fn
}

// -----

// Test struct field reflection attributes contextual evaluation under KGEN.

kgen.struct.generator @MyPair<T: type, U: type> = struct_inst<"MyPair"[T,U]<:type T, :type U>(first: typevalue<T>, second: typevalue<U>)>

kgen.struct.generator @VariadicWrapper<T: type, V: !kgen.param_list<T>> = struct_inst<"VariadicWrapper"[T,V]<:type T, :param_list<T> V>> {
  kgen.conformance @HasType {
    kgen.witness "my_type" : type = T
  }
}

#variadic_wrapper_types = #kgen.type<typevalue<#kgen.genref<@VariadicWrapper<:type type, :param_list<type> #kgen.struct_field_types<T>>>>, struct<()>> : !kgen.type
#variadic_wrapper_names = #kgen.type<typevalue<#kgen.genref<@VariadicWrapper<:type string, :param_list<string> #kgen.struct_field_names<T>>>>, struct<()>> : !kgen.type


// CHECK-LABEL: kgen.generator @expect_field_types_kgen
// CHECK-SAME:    (%arg0: !kgen.param<#kgen.get_witness<[typevalue<#kgen.genref<@VariadicWrapper<:type type, :param_list<type> #kgen.struct_field_types<T>>>>, struct<()>], @HasType, "my_type">>
kgen.generator @expect_field_types_kgen<T: !kgen.type>(%arg: !kgen.param<#kgen.get_witness<#variadic_wrapper_types, @HasType, "my_type">>) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator @expect_field_names_kgen
// CHECK-SAME:    (%arg0: !kgen.param<#kgen.get_witness<[typevalue<#kgen.genref<@VariadicWrapper<:type string, :param_list<string> #kgen.struct_field_names<T>>>>, struct<()>], @HasType, "my_type">>
kgen.generator @expect_field_names_kgen<T: !kgen.type>(%arg: !kgen.param<#kgen.get_witness<#variadic_wrapper_names, @HasType, "my_type">>) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator @expect_field_index_by_name_kgen
// CHECK-SAME:    (%arg0: !kgen.simd<#kgen.struct_field_index_by_name<T, "second">, si32>)
kgen.generator @expect_field_index_by_name_kgen<T: !kgen.type>(%arg: !kgen.simd<#kgen.struct_field_index_by_name<T, "second">, si32>) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator @expect_field_type_by_name_kgen
// CHECK-SAME:    (%arg0: !kgen.param<#kgen.struct_field_type_by_name<T, "first">>)
kgen.generator @expect_field_type_by_name_kgen<T: !kgen.type>(%arg: !kgen.param<#kgen.struct_field_type_by_name<T, "first">>) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

#mypair_index_i32 = #kgen.type<typevalue<#kgen.genref<@MyPair<:type index, :type i32>>>, struct<(index, i32)>> : !kgen.type

!StructFieldTypesExpected = !kgen.type
!StructFieldNamesExpected = !kgen.string
!StructFieldIndexByNameExpected = !kgen.simd<1, si32>
!StructFieldTypeByNameExpected = index

// CHECK-LABEL: kgen.generator @use_struct_reflection_kgen
kgen.generator @use_struct_reflection_kgen(%arg0: !StructFieldTypesExpected, %arg1: !StructFieldNamesExpected, %arg2: !StructFieldIndexByNameExpected, %arg3: !StructFieldTypeByNameExpected) -> !kgen.none {
  // CHECK-NEXT: kgen.call @expect_field_types_kgen<:type [typevalue<#kgen.genref<@MyPair<:type index, :type i32>>>, struct<(index, i32)>]>(%arg0) : (!kgen.type) -> !kgen.none
  %none0 = kgen.call @expect_field_types_kgen<:!kgen.type #mypair_index_i32>(%arg0) : !kgen.generator<(!StructFieldTypesExpected) -> !kgen.none>
  // CHECK-NEXT: kgen.call @expect_field_names_kgen<:type [typevalue<#kgen.genref<@MyPair<:type index, :type i32>>>, struct<(index, i32)>]>(%arg1) : (!kgen.string) -> !kgen.none
  %none1 = kgen.call @expect_field_names_kgen<:!kgen.type #mypair_index_i32>(%arg1) : !kgen.generator<(!StructFieldNamesExpected) -> !kgen.none>
  // CHECK-NEXT: kgen.call @expect_field_index_by_name_kgen<:type [typevalue<#kgen.genref<@MyPair<:type index, :type i32>>>, struct<(index, i32)>]>(%arg2) : (!kgen.scalar<si32>) -> !kgen.none
  %none2 = kgen.call @expect_field_index_by_name_kgen<:!kgen.type #mypair_index_i32>(%arg2) : !kgen.generator<(!StructFieldIndexByNameExpected) -> !kgen.none>
  // CHECK-NEXT: kgen.call @expect_field_type_by_name_kgen<:type [typevalue<#kgen.genref<@MyPair<:type index, :type i32>>>, struct<(index, i32)>]>(%arg3) : (index) -> !kgen.none
  %none3 = kgen.call @expect_field_type_by_name_kgen<:!kgen.type #mypair_index_i32>(%arg3) : !kgen.generator<(!StructFieldTypeByNameExpected) -> !kgen.none>
  kgen.return %none3 : !kgen.none
}

// -----

// Test `#kgen.get_witness` contextual evaluation through a `#kgen.extension`.

kgen.generator @call(%arg0: !kgen.struct<(index)>) -> index {
  %c = kgen.param.constant : index = <1>
  kgen.return %c : index
}

// Source struct carrying the `def() -> T` conformance the extension forwards to.
kgen.struct.generator @make = struct_inst<"make"(data: index)> {
  kgen.conformance @"def() -> T" {
    kgen.witness "T" : type = index
    kgen.witness "__call__" : (!kgen.struct<(index)>) -> index = @call
  }
}

// Stateless extension: supplies `def() -> V` by forwarding to the anchor's own
// `def() -> T` witnesses.
kgen.struct.generator @fwd_ext<A: type> = struct_inst<"fwd_ext"[A]<:type A>> {
  kgen.conformance @"def() -> V" {
    kgen.witness "V" : type = #kgen.get_witness<A, @"def() -> T", "T">
    kgen.witness "__call__" : (!kgen.struct<(index)>) -> index = #kgen.get_witness<A, @"def() -> T", "__call__">
  }
}

// Stateless extension whose `def() -> V` witnesses do not depend on the anchor.
kgen.struct.generator @const_ext<A: type> = struct_inst<"const_ext"[A]<:type A>> {
  kgen.conformance @"def() -> V" {
    kgen.witness "V" : type = index
    kgen.witness "__call__" : (!kgen.struct<(index)>) -> index = @call
  }
}

#make = #kgen.type<typevalue<:!kgen.type #kgen.genref<@make>>, struct<(index)>> : !kgen.type

// Concrete anchor: the extension resolves `__call__` by forwarding to @call.
// CHECK-LABEL: kgen.generator @extension_concrete_anchor
kgen.generator @extension_concrete_anchor(%arg: !kgen.struct<(index)>) -> index {
  // CHECK: kgen.call @call
  %r = kgen.call_param[(!kgen.struct<(index)>) -> index: #kgen.get_witness<#kgen.extension<#make, [#kgen.type<typevalue<:!kgen.type #kgen.genref<@fwd_ext<:type #make>>>, struct<()>> : !kgen.type]>, @"def() -> V", "__call__">](%arg)
  kgen.return %r : index
}

// Parametric anchor, anchor-independent witness: the extension resolves it even
// though `G` is unresolvable.
// CHECK-LABEL: kgen.generator @extension_param_anchor_independent
kgen.generator @extension_param_anchor_independent<G: !kgen.type>(%arg: !kgen.struct<(index)>) -> index {
  // CHECK: kgen.call @call
  %r = kgen.call_param[(!kgen.struct<(index)>) -> index: #kgen.get_witness<#kgen.extension<G, [#kgen.type<typevalue<:!kgen.type #kgen.genref<@const_ext<:type G>>>, struct<()>> : !kgen.type]>, @"def() -> V", "__call__">](%arg)
  kgen.return %r : index
}

// Parametric anchor, forwarding witness: resolution peels the extension down to
// the anchor's own witness and defers there (no error) since `G` is unresolvable.
// CHECK-LABEL: kgen.generator @extension_param_anchor_forwarding
kgen.generator @extension_param_anchor_forwarding<G: !kgen.type>(%arg: !kgen.struct<(index)>) -> index {
  // CHECK: kgen.call_param
  // CHECK-SAME: #kgen.get_witness<G, @"def() -> T", "__call__">
  %r = kgen.call_param[(!kgen.struct<(index)>) -> index: #kgen.get_witness<#kgen.extension<G, [#kgen.type<typevalue<:!kgen.type #kgen.genref<@fwd_ext<:type G>>>, struct<()>> : !kgen.type]>, @"def() -> V", "__call__">](%arg)
  kgen.return %r : index
}
