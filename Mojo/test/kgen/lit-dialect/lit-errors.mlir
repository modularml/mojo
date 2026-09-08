// RUN: kgen-opt %s -verify-parameters -verify-diagnostics -split-input-file -o /dev/null

lit.struct.decl @SomeStruct {
  // expected-error @+1 {{invalid use of parameter with no declaration "ty"}}
  %size = lit.var.decl "size" var : !lit.ref<simd<1, ty>, mut origin>
}

// -----

// expected-error @below {{expected declaration body to have no arguments}}
"lit.struct.decl"() ({
^bb0(%arg0: i32):
}) {sym_name = "StructArgs",
    decorators = #kgen<decorators[]>,
    signature = !lit.type_signature,
    canonicalTrait = !lit.trait<@Foo>,
    params = #kgen<param.decls[]>
    } : () -> ()

// -----

// expected-error @below {{custom op 'lit.struct.decl' expected no result parameters}}
lit.struct.decl @StructReturns<() -> dtype> {}

// -----

// expected-error @below {{custom op 'lit.fn' expected no result parameters}}
lit.fn @func_param_return<() -> dtype> {}

// -----

lit.struct.decl @StructDuplicate {
  // expected-note @below {{see previous declaration here}}
  lit.struct.field x : i32
  lit.struct.field y : i32
  // expected-error @below {{duplicate struct field "x"}}
  lit.struct.field x : i32
}

// -----

lit.struct.decl @SomeType<v, b> {}

// expected-error @below {{'kgen.generator' op funcTypeGenerator is not self-contained: it references parameter 'c' by name instead of by index}}
kgen.generator @InvalidTypeParamValue<a>(%arg0: !lit.struct<@SomeType<a, c>>) {
  kgen.return
}

// -----

lit.struct.decl @Bar {}

kgen.generator @invalid_field_name(%a: index, %container: !lit.struct<@Bar>) {
  // expected-error @below {{struct @Bar has no field named "a"}}
  %0 = lit.struct.insert %a, %container[a] : index into !lit.struct<@Bar>
  kgen.return
}

// -----

lit.struct.decl @Bar {
  lit.struct.field a : i32
}

kgen.generator @invalid_field_name(%a: index, %container: !lit.struct<@Bar>) {
  // expected-error @below {{cannot insert value of type 'index' into struct field "a" which expected 'i32'}}
  %0 = lit.struct.insert %a, %container[a] : index into !lit.struct<@Bar>
  kgen.return
}

// -----

lit.struct.decl @Bar {}

kgen.generator @invalid_field_name(%a: index, %container: !lit.struct<@Bar>) {
  // expected-error @below {{struct @Bar has no field named "a"}}
  %0 = lit.struct.extract %container[a] : index from !lit.struct<@Bar>
  kgen.return
}

// -----

// expected-error @below {{expected SSA operand}}
lit.fn @no_names(index)

// -----

// expected-error @below {{only one '|' allowed in signature}}
lit.fn @twoSlash(%a: index, |, |, %b: index) {
  kgen.return
}

// -----

// expected-error @below {{only one '*' allowed in signature}}
lit.fn @twoStar(%a: index, *, *, %b: index) {
  kgen.return
}

// -----

// Regression test: when a struct field's declared type contains get_witness,
// the verifier should fold it to a concrete type and still reject a wrong
// result type. This ensures get_witness folding works during verification
// and that genuine type mismatches are not silently accepted.

!HasOutput = !lit.trait<@HasOutput>

lit.trait.decl @HasOutput<?, SELF: !HasOutput> {
  lit.alias.decl Output: type
}

#inner_type = #kgen.type<!lit.struct<@Inner>> : !HasOutput

lit.struct.decl @Inner {
  kgen.conformance @HasOutput {
    kgen.witness "Output" : type = index
  }
}

lit.struct.decl @Wrap<T: !HasOutput> {
  lit.struct.field value : !kgen.param<#kgen.get_witness<:!HasOutput T, @HasOutput, "Output">>
}

kgen.generator @access_wrong_type(%wrap: !lit.struct<@Wrap<:!HasOutput #inner_type>>) {
  // expected-error @below {{cannot extract value of type 'i32' from struct field "value" which has type 'index'}}
  %0 = lit.struct.extract %wrap[value] : i32 from !lit.struct<@Wrap<:!HasOutput #inner_type>>
  kgen.return
}

// -----

// expected-error @below {{'*' cannot precede '|' in signature}}
lit.fn @slashAfterStar(%a: index, *, |, %b: index) {
  kgen.return
}

// -----

// expected-error @+1 {{expected variadic kind, got: stuff}}
lit.fn @incorrect_arg_variadicness(%a: index imm|stuff) {
  kgen.return
}

// -----

// expected-error @+1 {{expected convention|variadicness, got: stuff}}
lit.fn @incorrect_arg_conv_and_variadicness(%a: index stuff) {
  kgen.return
}

// -----

// expected-error @+1 {{expected variadic kind, got: stuff}}
lit.fn @incorrect_param_variadicness<a: dtype stuff>() {
  kgen.return
}

// -----

kgen.generator @not_lit_func() {
  // expected-error @below {{'lit.return' op expected to be nested inside a `lit.fn` operation}}
  lit.return
  kgen.return
}

// -----

lit.fn @mismatched_return_types(%arg0: i64) -> i32 {
  // expected-error @below {{'lit.return' op operand #0 has type 'i64' but expected 'i32'}}
  lit.return %arg0 : i64
  lit.end_fn
}

// -----

lit.fn @does_not_throw() {
  // expected-error @below {{'lit.raise' op must be nested inside the 'try' region of a `lit.try` operation or a throwing function}}
  lit.raise
  lit.end_fn
}

// -----

lit.fn @not_async() {
  // expected-error @below {{'lit.async.call' op callable must be 'async'}}
  %0 = lit.async.call[() -> (): @not_async]()
  lit.end_fn
}

// -----

lit.fn @unbound_region() {
  // expected-error @below {{'lit.unbound_region' op is never valid. Was it not erased by the parser?}}
  "lit.unbound_region"() ({
  ^bb0(%arg0: index):
    hlcf.yield %arg0 : index
  }) : () -> ()
  kgen.return
}

// -----

// expected-error@below {{expected only `lit.file_module`, `lit.package`, `lit.unresolved_import`, or `lit.unresolved_wildcard_import` in its body}}
lit.package @MyPackage {
  // expected-note @below {{see operation defined here}}
  kgen.unreachable
}

// -----

lit.fn @declareWrongType() {
  // expected-error @below {{op declares a parameter with type 'index' but parameter expression has type 'i32'}}
  "lit.alias.decl"() {paramDecl = #kgen<param.decl p1 : index>, value = 1 : i32} : () -> ()
  kgen.return
}

// -----

lit.fn @wrong_error_return1(%arg0: i32) -> i1 {
  %0 = kgen.param.constant = <0>
  // expected-error @below {{'lit.error_return' op operand #0 has type 'index' but expected 'i1'}}
  lit.error_return %0 : index
}

// -----

lit.fn @wrong_error_return2(%arg0: i32) -> !kgen.variant<index> {
  %var = kgen.variant.create %arg0, 0 : <i32, index>
  // expected-error @below {{'lit.error_return' op operand #0 has type '!kgen.variant<i32, index>' but expected '!kgen.variant<index>'}}
  lit.error_return %var : !kgen.variant<i32, index>
}

// -----

// expected-error @below {{specified `declNameLoc` without `declName`}}
lit.unresolved_import <0, ["module"]> as @newModule declNameLoc(loc(unknown))

// -----

// expected-error @below {{import path must be relative or have at least one component}}
lit.unresolved_import <0> as @newModule

// -----

// expected-error @below {{argument #0 with convention 'imm_mem' in signature type should be a `!lit.ref` but got: 'index'}}
!type = !lit.generator<(index imm_mem) -> ()>

// -----

// expected-error @below {{'?' cannot precede '|' in signature}}
!sig = !lit.generator<<?, |>() -> ()>

// -----

// expected-error @below {{'?' cannot precede '*' in signature}}
!sig = !lit.generator<<?, *>() -> ()>

// -----

// expected-error @below {{only one '?' allowed in signature}}
!sig = !lit.generator<<?, ?> -> ()>

// -----

// expected-error @below {{2 origins specified, but signature expected 1}}
lit.call @calls[imm a, mut b]() : !lit.generator<[1]() -> ()>

// -----

// expected-error @+1 {{custom op 'lit.call' implicit origin reference at depth 0 has an out-of-range index: 1 >= 1}}
lit.call @calls[mut a]() : !lit.generator<[1](!lit.ref<index, mut *[0,1]>) -> ()>

// -----

lit.fn @ref_immut<life: origin<false>>(%ref1: !lit.ref<index, imm life>) ->  !lit.ref<index, imm life> {
  %ref2 = lit.ref.immut %ref1: !lit.ref<index, imm life>
  kgen.return %ref2: !lit.ref<index, imm life>
}

// -----

lit.fn @ref_to_kgen_ptr_address_space_mismatch<life: origin<false>>(
    %ref1: !lit.ref<index, imm life, 1>) {
  // expected-error @below {{address space mismatch: ref has address space 1 : index but result has 0 : index}}
  %ptr = lit.ref.to_kgen_ptr %ref1 : !lit.ref<index, imm life, 1>
                                   -> !kgen.pointer<index>
  lit.end_fn
}

// -----

lit.fn @ref_from_kgen_ptr_address_space_mismatch<life: origin<false>>(
    %ptr: !kgen.pointer<index, 2>) {
  // expected-error @below {{address space mismatch: pointer has address space 2 : index but result has 0 : index}}
  %ref = lit.ref.from_kgen_ptr %ptr : !kgen.pointer<index, 2>
                                    -> !lit.ref<index, imm life>
  lit.end_fn
}

// -----

lit.fn @invalid_memcpy<out: origin<true>, in: origin<false>>(%dst: !lit.ref<f32, mut out>, %src: !lit.ref<index, imm in>) {
  // expected-error @below {{'lit.memcpy' op failed to verify that all of {src, dst} have same element types}}
  lit.memcpy %src, %dst : !lit.ref<index, imm in>-> !lit.ref<f32, mut out>
  lit.end_fn
}

// -----

lit.fn @ref_upcast_not_wider<a: origin<true>, b: origin<true>>(
    %ref1: !lit.ref<index, mut a>) {
  // expected-error @below {{result origin is not an upcast of the source origin}}
  %ref2 = lit.ref.upcast %ref1
    : !lit.ref<index, mut a>
    -> !lit.ref<index, mut b>
  lit.end_fn
}

// -----

// Parent subtree is not an upcast of a narrower field subtree destination
// (origin_of(self.f).subtree does not contain origin_of(self).subtree).
lit.fn @ref_upcast_parent_subtree_to_field_subtree<life: origin<true>>(
    %ref1: !lit.ref<index, mut #lit.origin.subtree<#kgen.param.decl.ref<"life"> : !lit.origin<true>>>) {
  // expected-error @below {{result origin is not an upcast of the source origin}}
  %ref2 = lit.ref.upcast %ref1
    : !lit.ref<index, mut #lit.origin.subtree<#kgen.param.decl.ref<"life"> : !lit.origin<true>>>
    -> !lit.ref<index, mut #lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"life"> : !lit.origin<true>, "f"> : !lit.origin<true>>>
  lit.end_fn
}

// -----

// Sibling field subtrees are not upcasts of each other.
lit.fn @ref_upcast_sibling_subtrees<life: origin<true>>(
    %ref1: !lit.ref<index, mut #lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"life"> : !lit.origin<true>, "a"> : !lit.origin<true>>>) {
  // expected-error @below {{result origin is not an upcast of the source origin}}
  %ref2 = lit.ref.upcast %ref1
    : !lit.ref<index, mut #lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"life"> : !lit.origin<true>, "a"> : !lit.origin<true>>>
    -> !lit.ref<index, mut #lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"life"> : !lit.origin<true>, "b"> : !lit.origin<true>>>
  lit.end_fn
}
