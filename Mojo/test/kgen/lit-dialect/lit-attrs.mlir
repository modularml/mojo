// RUN: kgen-opt %s -allow-unregistered-dialect -verify-parameters | kgen-opt -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -emit-bytecode -allow-unregistered-dialect | kgen-opt -allow-unregistered-dialect | FileCheck %s

// Pog metadata/list round-trip tests now live in
// `KGEN/test/kgen/kgen-dialect/kgen-pog-attrs.mlir`; their error cases live in
// `kgen-pog-attrs-errors.mlir`.

// CHECK-LABEL: "empty.metadata"
// CHECK-SAME: #lit.fn_meta_origin_data<0>
"empty.metadata"() {metadata = #lit.fn_meta_origin_data<0>} : () -> ()

// CHECK-LABEL: "some.metadata1"
// CHECK-SAME: #lit.fn_meta_origin_data<2, {mut lt}>
"some.metadata1"() {metadata = #lit.fn_meta_origin_data<2, {mut lt}>} : () -> ()

// CHECK-LABEL: "some.metadata2"
// CHECK-SAME: #lit.fn_meta_origin_data<2, {mut lt}, true>
"some.metadata2"() {metadata = #lit.fn_meta_origin_data<2, {mut lt}, true>} : () -> ()

// CHECK-LABEL: "some.metadata3"
// CHECK-SAME: #lit.fn_meta_origin_data<2, true>
"some.metadata3"() {metadata = #lit.fn_meta_origin_data<2, true>} : () -> ()

// CHECK-LABEL: "some.metadata4"
// CHECK-SAME: #lit.fn_meta_origin_data<2, false>
"some.metadata4"() {metadata = #lit.fn_meta_origin_data<2, false>} : () -> ()

// CHECK-LABEL: "some.metadata5"
// CHECK-SAME: #lit.fn_meta_origin_data<2, false, true>
"some.metadata5"() {metadata = #lit.fn_meta_origin_data<2, false, true>} : () -> ()

// CHECK-LABEL: "none.type"
// CHECK-SAME: #kgen.none : !kgen.none
"none.type"() {a = #kgen.none : !kgen.none} : () -> ()

lit.struct.decl @Foo {
  lit.struct.field foo : index
  lit.struct.field bar : !kgen.dtype
}

// CHECK-LABEL: "struct.attr"
// CHECK-SAME: #lit.struct<{foo = 5, bar: dtype = f32}>
"struct.attr"() {a = #lit.struct<{foo = 5, bar: dtype = f32}> : !lit.struct<@Foo>} : () -> ()

// CHECK-LABEL: "origin.attr"
// CHECK: #lit.any.origin : !lit.origin<true>
"origin.attr"() {a = #lit.any.origin : !lit.origin<true>} : () -> ()


kgen.generator @lifetime_lower<p: !lit.origin<true>>(%a: !lit.origin<false>) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @caller
kgen.generator @caller() {
  // CHECK: %origin = kgen.param.constant: origin<false> = <#lit.any.origin>
  %cst = kgen.param.constant: origin<false> = <#lit.any.origin>
  // CHECK: kgen.call @lifetime_lower<:origin<true> #lit.any.origin>(%origin) : (!lit.origin<false>) -> ()
  kgen.call @lifetime_lower<:origin<true> #lit.any.origin>(%cst) : (!lit.origin<false>) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @ref_type<p: origin<false>, q: origin<true>>(
// CHECK-SAME: %arg0: !lit.ref<!lit.struct<@Foo>, imm p>
// CHECK-SAME: %arg1: !lit.ref<!lit.struct<@Foo>, mut q>)
kgen.generator @ref_type<p: !lit.origin<false>, q: !lit.origin<true>>
(%a: !lit.ref<!lit.struct<@Foo>, imm p>, %b: !lit.ref<!lit.struct<@Foo>, mut q>) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @field_attr<life: origin<false>>(
// CHECK-SAME: %arg0: !lit.ref<!lit.struct<@Foo>, imm life->field>
// CHECK-SAME: %arg1: !lit.ref<!lit.struct<@Foo>, imm life->a->b->c>)
kgen.generator @field_attr<life: !lit.origin<false>>
(%arg0: !lit.ref<@Foo, imm life->field>,
 %arg1: !lit.ref<@Foo, imm life->a->b->c>) {

  // CHECK-NEXT: "verbose_attr"() {attr = #lit.origin.field<#kgen.param.decl.ref<"life"> : !lit.origin<false>, "field0"> : !lit.origin<false>} : () -> ()
  "verbose_attr"() {attr = #lit.origin.field<#kgen.param.decl.ref<"life"> : !lit.origin<false>, "field0"> : !lit.origin<false>} : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @interior_origin_attr<orig: origin<false>>(
// CHECK-SAME: %arg0: !lit.ref<!lit.struct<@Foo>, imm orig["interior"]->field>
// CHECK-SAME: %arg1: !lit.ref<!lit.struct<@Foo>, imm orig["interior"]->a["pointee"]>)
kgen.generator @interior_origin_attr<orig: !lit.origin<false>>
(%arg0: !lit.ref<@Foo, imm orig["interior"]->field>,
 %arg1: !lit.ref<@Foo, imm orig["interior"]->a["pointee"]>) {

  // CHECK-NEXT: "verbose_attr"() {attr = #lit.interior.origin<#kgen.param.decl.ref<"orig"> : !lit.origin<false>, "pointee" : !kgen.string> : !lit.origin<false>} : () -> ()
  "verbose_attr"() {attr = #lit.interior.origin<#kgen.param.decl.ref<"orig"> : !lit.origin<false>, "pointee" : !kgen.string> : !lit.origin<false>} : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @subtree_origin_attr
kgen.generator @subtree_origin_attr<root: !lit.origin<false>>() {
  // CHECK-NEXT: "verbose_attr"() {attr = #lit.origin.subtree<#kgen.param.decl.ref<"root"> : !lit.origin<false>> : !lit.origin<false>} : () -> ()
  "verbose_attr"() {attr = #lit.origin.subtree<#kgen.param.decl.ref<"root"> : !lit.origin<false>> : !lit.origin<false>} : () -> ()
  // CHECK-NEXT: "verbose_attr"() {attr = #lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"root"> : !lit.origin<false>, "field0"> : !lit.origin<false>> : !lit.origin<false>} : () -> ()
  "verbose_attr"() {attr = #lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"root"> : !lit.origin<false>, "field0"> : !lit.origin<false>> : !lit.origin<false>} : () -> ()
  kgen.return
}




// CHECK-LABEL: kgen.generator @anystruct
kgen.generator @anystruct() {
  // CHECK: T: meta<!lit.struct<@Foo>> = <@Foo>
  kgen.param.declare T: meta<!lit.struct<@Foo>> = <@Foo>
  kgen.return
}

// CHECK-LABEL: kgen.generator @unpacked
kgen.generator @unpacked<a: param_list<index>>() {
  // CHECK-NEXT: constant = <#lit.unpacked<:param_list<index> a, kw>>
  %c = kgen.param.constant = <#lit.unpacked<:param_list<index> a, kw>>
  kgen.return
}

// CHECK-LABEL: @lifetime_union
kgen.generator @lifetime_union<x: !lit.origin<false>, y: !lit.origin<false>>() {
  // CHECK-NEXT: %a = lit.var.decl
  %a = lit.var.decl "a" imp : !lit.ref<index, mut z>

  // CHECK-NEXT: "empty"() {a = #lit<origin.union > : !lit.origin<false>} : () -> ()
  "empty"() {a = #lit.origin.union<> : !lit.origin<false>} : () -> ()
  // CHECK-NEXT: "a"() {a = #lit.any.origin : !lit.origin<false>} : () -> ()
  "a"() {a = #lit.origin.union<:origin<false> #lit.any.origin> : !lit.origin<false>} : () -> ()
  // CHECK-NEXT: "b"() {a = #kgen.param.decl.ref<"x"> : !lit.origin<false>}
  "b"() {a = #lit.origin.union<:origin<false> #kgen.param.decl.ref<"x">>
        : !lit.origin<false>} : () -> ()
  // CHECK-NEXT: "c"() {a = #lit<origin.union <:origin<false> x, :origin<false> y>> : !lit.origin<false>}
  "c"() {a = #lit.origin.union<:origin<false> #kgen.param.decl.ref<"x">,
                              :origin<false> #kgen.param.decl.ref<"y">>
        : !lit.origin<false>} : () -> ()
  // CHECK-NEXT: "d"() {a = #lit<origin.union <:origin<false> x, :origin<false> x->field0>> : !lit.origin<false>}
  "d"() {a = #lit.origin.union<:origin<false> #kgen.param.decl.ref<"x">,
                               :origin<false> #lit.origin.field<#kgen.param.decl.ref<"x"> : !lit.origin<false>, "field0">,
                               :origin<false> #kgen.param.decl.ref<"x">>
        : !lit.origin<false>} : () -> ()

  // CHECK-NEXT: "e"() {a = #lit<origin.union <:origin<false> *[0,1], :origin<false> *[1,0]>> : !lit.origin<false>}
  "e"() {a = #lit.origin.union<:origin<false> #lit.implicit.origin.ref<1, 0>,
                                :origin<false> #lit.implicit.origin.ref<0, 1>> : !lit.origin<false>} : () -> ()

  kgen.param.declare is_mut: !kgen.scalar<bool> = <#kgen.simd<false>:!kgen.scalar<bool>>
  kgen.param.declare a: origin<true> = <?>
  kgen.param.declare b: origin<false> = <?>
  kgen.param.declare c: origin<is_mut> = <?>

  // CHECK: <{(is_mut) c, mut a, imm b}>
  kgen.param.constant: origin.set = <{imm b, mut a, (is_mut) c, mut a}>
  // CHECK-NEXT: <{mut a}>
  kgen.param.constant: origin.set = <{imm (mutcast mut a)}>
  // CHECK-NEXT: <{mut a, imm b, mut #lit.any.origin}>
  kgen.param.constant: origin.set = <{mut #lit.any.origin, mut a, imm b}>
  // CHECK-NEXT: <{mut a, imm b}>
  kgen.param.constant: origin.set = <{mut {(mutcast imm b), a}}>

  // CHECK-NEXT: <{(mutcast mut a), b}>
  kgen.param.constant: origin<false> = <|{mut a, imm b}|>

  // CHECK-NEXT: kgen.param.declare nothing: origin<false> = <#lit.any.origin>
  kgen.param.declare nothing: !lit.origin<false> = <#lit.any.origin>
  // CHECK-NEXT:  kgen.param.declare nothing_2: origin<false> = <#lit.any.origin>
  kgen.param.declare nothing_2: !lit.origin<false> = <{#lit.any.origin, #lit.any.origin}>
  // CHECK-NEXT: kgen.param.declare x_ref: origin<false> = <x>
  kgen.param.declare x_ref: !lit.origin<false> = <x>
  // CHECK-NEXT: kgen.param.declare x_ref2: origin<false> = <x>
  kgen.param.declare x_ref2: !lit.origin<false> = <*"x">
  // CHECK-NEXT: kgen.param.declare x_or_y_ref: origin<false> = <{x, y}>
  kgen.param.declare x_or_y_ref: !lit.origin<false> = <{x, y, x}>
  // CHECK-NEXT: kgen.param.declare y_ref: origin<false> = <#lit.any.origin>
  kgen.param.declare y_ref: !lit.origin<false> = <{y, #lit.any.origin}>
  // CHECK-NEXT: kgen.param.declare xyz_ref: origin<false> = <{x, y, (mutcast mut z)}>
  kgen.param.declare xyz_ref: !lit.origin<false> = <{{x, y}, {(mutcast mut z), y}}>

  // CHECK-NEXT: kgen.param.declare subtree_collapsed: origin<false> = <#lit.origin.subtree<#kgen.param.decl.ref<"x"> : !lit.origin<false>>>
  kgen.param.declare subtree_collapsed: !lit.origin<false> = <
    {#lit.origin.field<#kgen.param.decl.ref<"x"> : !lit.origin<false>, "field0"> : !lit.origin<false>,
      #lit.origin.subtree<#kgen.param.decl.ref<"x"> : !lit.origin<false>> : !lit.origin<false>}>

  // Field subtree absorbed into a wider parent subtree.
  // CHECK-NEXT: kgen.param.declare field_subtree_in_parent: origin<false> = <#lit.origin.subtree<#kgen.param.decl.ref<"x"> : !lit.origin<false>>>
  kgen.param.declare field_subtree_in_parent: !lit.origin<false> = <
    {#lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"x"> : !lit.origin<false>, "f"> : !lit.origin<false>> : !lit.origin<false>,
      #lit.origin.subtree<#kgen.param.decl.ref<"x"> : !lit.origin<false>> : !lit.origin<false>}>

  // Narrower field subtree must not absorb the wider parent subtree
  // (over-absorption): union(x.f~, x~) collapses to x~, not x.f~.
  // CHECK-NEXT: kgen.param.declare no_over_absorb: origin<false> = <#lit.origin.subtree<#kgen.param.decl.ref<"x"> : !lit.origin<false>>>
  kgen.param.declare no_over_absorb: !lit.origin<false> = <
    {#lit.origin.subtree<#kgen.param.decl.ref<"x"> : !lit.origin<false>> : !lit.origin<false>,
      #lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"x"> : !lit.origin<false>, "f"> : !lit.origin<false>> : !lit.origin<false>}>

  // Sibling field subtrees must stay a union.
  // CHECK-NEXT: kgen.param.declare sibling_subtrees: origin<false> = <{#lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"x"> : !lit.origin<false>, "a"> : !lit.origin<false>>, #lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"x"> : !lit.origin<false>, "b"> : !lit.origin<false>>}>
  kgen.param.declare sibling_subtrees: !lit.origin<false> = <
    {#lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"x"> : !lit.origin<false>, "a"> : !lit.origin<false>> : !lit.origin<false>,
      #lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"x"> : !lit.origin<false>, "b"> : !lit.origin<false>> : !lit.origin<false>}>

  // Multi-level nesting: x.y.z is contained in x~ and in x.y~.
  // CHECK-NEXT: kgen.param.declare nested_in_root: origin<false> = <#lit.origin.subtree<#kgen.param.decl.ref<"x"> : !lit.origin<false>>>
  kgen.param.declare nested_in_root: !lit.origin<false> = <
    {#lit.origin.field<#lit.origin.field<#kgen.param.decl.ref<"x"> : !lit.origin<false>, "y"> : !lit.origin<false>, "z"> : !lit.origin<false>,
      #lit.origin.subtree<#kgen.param.decl.ref<"x"> : !lit.origin<false>> : !lit.origin<false>}>
  // CHECK-NEXT: kgen.param.declare nested_in_mid: origin<false> = <#lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"x"> : !lit.origin<false>, "y"> : !lit.origin<false>>>
  kgen.param.declare nested_in_mid: !lit.origin<false> = <
    {#lit.origin.field<#lit.origin.field<#kgen.param.decl.ref<"x"> : !lit.origin<false>, "y"> : !lit.origin<false>, "z"> : !lit.origin<false>,
      #lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"x"> : !lit.origin<false>, "y"> : !lit.origin<false>> : !lit.origin<false>}>
  // CHECK-NEXT: kgen.param.declare nested_subtree_in_mid: origin<false> = <#lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"x"> : !lit.origin<false>, "y"> : !lit.origin<false>>>
  kgen.param.declare nested_subtree_in_mid: !lit.origin<false> = <
    {#lit.origin.subtree<#lit.origin.field<#lit.origin.field<#kgen.param.decl.ref<"x"> : !lit.origin<false>, "y"> : !lit.origin<false>, "z"> : !lit.origin<false>> : !lit.origin<false>,
      #lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"x"> : !lit.origin<false>, "y"> : !lit.origin<false>> : !lit.origin<false>}>

  // Mixed mutability: {imm x, mut a~, mut a.z} must still absorb a.z into a~
  // even though members are wrapped in mutcasts for the imm result type.
  // CHECK-NEXT: <{x, (mutcast mut #lit.origin.subtree<#kgen.param.decl.ref<"a"> : !lit.origin<true>>)}>
  kgen.param.constant: origin<false> = <|{
    imm x,
    mut #lit.origin.subtree<#kgen.param.decl.ref<"a"> : !lit.origin<true>>,
    mut #lit.origin.field<#kgen.param.decl.ref<"a"> : !lit.origin<true>, "z">
  }|>

  kgen.return
}

// CHECK-LABEL: kgen.generator @test_origin
kgen.generator @test_origin<x: !kgen.scalar<bool>>() {
  // CHECK-NEXT: kgen.param.constant: origin<false> = <rebind(:scalar<bool> x)>
  kgen.param.constant: origin<false> = <rebind(:!kgen.scalar<bool> x)>
  kgen.return
}

// CHECK-LABEL: "deprecation.attrs"
// CHECK-SAME: d0 = #lit.deprecation<"use new_func instead">
// CHECK-SAME: d1 = #lit.deprecation<"old is deprecated", "new_func">
"deprecation.attrs"() {
  d0 = #lit.deprecation<"use new_func instead">,
  d1 = #lit.deprecation<"old is deprecated", "new_func">
} : () -> ()

// CHECK-LABEL: kgen.generator @bytecode_lit_extra_types
kgen.generator @bytecode_lit_extra_types() {
  // CHECK: kgen.param.declare magic_def: type = <!lit.magic.__mlir_deferred_attr>
  kgen.param.declare magic_def: type = <!lit.magic.__mlir_deferred_attr>
  // CHECK: kgen.param.declare nl_wild: type = <!lit.name_lookup_arg_wildcard>
  kgen.param.declare nl_wild: type = <!lit.name_lookup_arg_wildcard>
  kgen.return
}
