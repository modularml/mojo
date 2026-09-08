// RUN: kgen-opt %s -lower-lit -allow-unregistered-dialect -split-input-file | kgen-opt -verify-parameters -split-input-file -kgen-print-inline-type-values  | FileCheck %s

//===----------------------------------------------------------------------===//
// Parametric Structs
//===----------------------------------------------------------------------===//

// CHECK-NOT: lit.struct.decl
lit.struct.decl @SmallVector<N, T: type> register_passable {
  lit.struct.field data: !pop.array<N, T>
}

!size2 = !lit.struct<@SmallVector<2, :type !kgen.simd<4, f32>>>
!size4 = !lit.struct<@SmallVector<4, :type !kgen.simd<1, f64>>>


// CHECK-NOT: lit.struct.decl
lit.struct.decl @Box<T: type> register_passable {
  lit.struct.field value: !kgen.param<T>
}

// CHECK-NOT: lit.struct.decl
lit.struct.decl @Pair<T1: type, T2: type> {
  lit.struct.field first: !kgen.param<T1>
  lit.struct.field second: !kgen.param<T2>
}


!i8Pair = !lit.struct<@Pair<:type i8, :type i8>>


// CHECK-LABEL: @struct_insert
kgen.func @struct_insert(%pair: !i8Pair) -> !i8Pair {
  %c1 = llvm.mlir.constant(2 : i8) : i8
  // CHECK: kgen.struct.replace %{{.*}}, %{{.*}}[1]
  %0 = lit.struct.insert %c1, %pair[second] : i8 into !i8Pair
  kgen.return %0 : !i8Pair
}

// CHECK-LABEL: @struct_extract
kgen.func @struct_extract(%pair: !i8Pair) -> i8 {
  // CHECK: kgen.struct.extract %{{.*}}[1]
  %0 = lit.struct.extract %pair[second] : i8 from !i8Pair
  kgen.return %0 : i8
}

lit.struct.decl @NestedA<T: type> register_passable {
  lit.struct.field v: !kgen.param<T>
}
lit.struct.decl @NestedB<t: dtype> register_passable {
  lit.struct.field a: !lit.struct<@NestedA<:type !kgen.simd<1, t>>>
}
lit.struct.decl @NestedC register_passable {
  lit.struct.field b: !lit.struct<@NestedB<:dtype f32>>
}

// CHECK-LABEL: @use_nested(%arg0: !kgen.scalar<f32>)
kgen.func @use_nested(%a: !lit.struct<@NestedC>) {
  kgen.return
}

// CHECK-LABEL: @struct_element(%arg0: !kgen.pointer<simd<2, f32>>
kgen.func @struct_element(%a: !kgen.pointer<!lit.struct<@NestedA<:type !kgen.simd<2, f32>>>>) {
  kgen.return
}


lit.struct.decl @IndexStruct register_passable {
  lit.struct.field value : index
}

lit.struct.decl @StructInsideStruct register_passable {
  lit.struct.field x : !lit.struct<@IndexStruct>
}

lit.struct.decl @IndexField {
  lit.struct.field first: index
  lit.struct.field second: index
}

// CHECK-LABEL: @structExtract
lit.fn @structExtract<p: !lit.struct<@IndexField>>() {
  kgen.param.constant = <#lit.struct.extract<:!lit.struct<@IndexField> p, "second">>
  kgen.return
}

lit.fn @structExtractInsideStruct<p: @IndexField>(
    %arg0: !lit.struct<@SmallVector<#lit.struct.extract<:@IndexField p, "second">, :type index>>) {
  %0 = lit.struct.extract %arg0[data] : !pop.array<#lit.struct.extract<:@IndexField p, "second">, index> from
    !lit.struct<@SmallVector<#lit.struct.extract<:@IndexField p, "second">, :type index>>
  kgen.return
}

lit.struct.decl @Struct register_passable {}

lit.struct.decl @StructParam<param: @Struct> register_passable {
  lit.struct.field value : !pop.array<apply(:(!lit.struct<@Struct>) -> index @return_one, param), index>
}

lit.fn @return_one(%arg0: !lit.struct<@Struct>) -> index {
  %0 = index.constant 0
  kgen.return %0 : index
}

// CHECK-LABEL: @use_struct_param
// CHECK-SAME: !pop.array<apply(:(!kgen.struct<()>) -> index @return_one, { }), index>
lit.fn @use_struct_param(%arg0: !lit.struct<@StructParam<:@Struct #lit.struct<{}>>>) {
  lit.struct.extract %arg0[value] : !pop.array<apply(:(!lit.struct<@Struct>) -> index @return_one, #lit.struct<{}>), index>
    from !lit.struct<@StructParam<:@Struct #lit.struct<{}>>>
  kgen.return
}

// CHECK-LABEL: kgen.generator @lifetime_lower
// CHECK-SAME: (%arg0: !kgen.struct<()>)
// CHECK: sourceParamList = #kgen.pog_list<[<"p", pos_or_kw, not_vararg>]>
lit.fn @lifetime_lower<p: !lit.origin<false>>(%a: !lit.origin<true>) {

  // CHECK: kgen.param.declare A: struct<()> = <{ }>
  kgen.param.declare A: !lit.origin<true> = <#lit.any.origin>

  // CHECK: kgen.param.declare B: struct<()> = <{ }>
  kgen.param.declare B: origin.set = <{imm p}>
  kgen.return
}

// CHECK-LABEL: kgen.generator @call_lifetime_lower
lit.fn @call_lifetime_lower() {
  // CHECK: %struct = kgen.param.constant: struct<()> = <{ }>
  %cst = kgen.param.constant: origin<true> = <#lit.any.origin>
  // CHECK: kgen.call @lifetime_lower(%struct) : (!kgen.struct<()>) -> ()
  lit.call @lifetime_lower<:origin<false> #lit.any.origin>(%cst) : !lit.generator<(!lit.origin<true>) -> ()>
  kgen.return
}

lit.fn @take_origin<lt: origin<false>>() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @implicit_lifetime_as_param
lit.fn @implicit_lifetime_as_param() {
  // CHECK-NEXT: kgen.call @take_origin() : () -> ()
  kgen.call @take_origin<:origin<false> *[0,0]>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @ref_type(
// CHECK-SAME: %arg0: !kgen.pointer<struct<()>>
// CHECK-SAME: %arg1: !kgen.pointer<struct<()>>)
lit.fn @ref_type<p: !lit.origin<false>, q: !lit.origin<true>>
    (%a: !lit.ref<@Struct, imm p>, %b: !lit.ref<@Struct, mut p>) {
  // Random use of a parameter that goes away should be updated.
  // CHECK: kgen.param.declare A: struct<()> = <{ }>
  kgen.param.declare A : !lit.origin<false> = <p>
  kgen.return
}

// CHECK-LABEL: kgen.generator @call_ref_type
lit.fn @call_ref_type<a: !lit.origin<false>, b: !lit.origin<true>>
    (%a: !lit.ref<@Struct, imm a>, %b: !lit.ref<@Struct, mut b>) {
  // CHECK-NEXT: kgen.call @ref_type(%arg0, %arg1)
  // CHECK-SAME: : (!kgen.pointer<struct<()>>, !kgen.pointer<struct<()>>) -> ()
  kgen.call @ref_type<:origin<false> a, :origin<true> b>(%a, %b): (!lit.ref<@Struct, imm a>, !lit.ref<@Struct, mut b>) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @raw_pointer_from_ref_type
// CHECK-SAME: (%arg0: !kgen.pointer<struct<()>>) -> !kgen.pointer<struct<()>>
lit.fn @raw_pointer_from_ref_type<q: !lit.origin<false>>(%a: !lit.ref<@Struct, imm q>)
  -> !kgen.pointer<@Struct> {
  // CHECK-NEXT: kgen.return %a
  %ptr = lit.ref.to_pointer %a: !lit.ref<@Struct, imm q>
  kgen.return %ptr: !kgen.pointer<@Struct>
}

// CHECK-LABEL: kgen.generator @ref_to_kgen_ptr
// CHECK-SAME: (%arg0: !kgen.pointer<struct<()>>) -> !kgen.pointer<struct<()>>
lit.fn @ref_to_kgen_ptr<q: !lit.origin<false>>(%a: !lit.ref<@Struct, imm q>)
  -> !kgen.pointer<!kgen.struct<()>> {
  // CHECK-NEXT: kgen.return %arg0
  %ptr = lit.ref.to_kgen_ptr %a : !lit.ref<@Struct, imm q>
                                -> !kgen.pointer<!kgen.struct<()>>
  kgen.return %ptr: !kgen.pointer<!kgen.struct<()>>
}

// CHECK-LABEL: kgen.generator @ref_from_kgen_ptr
// CHECK-SAME: (%arg0: !kgen.pointer<struct<()>>) -> !kgen.pointer<struct<()>>
lit.fn @ref_from_kgen_ptr<q: !lit.origin<false>>(%a: !kgen.pointer<!kgen.struct<()>>)
  -> !lit.ref<@Struct, imm q> {
  // CHECK-NEXT: kgen.return %arg0
  %ref = lit.ref.from_kgen_ptr %a : !kgen.pointer<!kgen.struct<()>>
                                  -> !lit.ref<@Struct, imm q>
  kgen.return %ref: !lit.ref<@Struct, imm q>
}

// CHECK-LABEL: kgen.generator @ref_kgen_ptr_roundtrip
// CHECK-SAME: (%arg0: !kgen.pointer<struct<(si32, ui32) memoryOnly>>)
lit.fn @ref_kgen_ptr_roundtrip<q: !lit.origin<false>>
  (%a: !lit.ref<!lit.struct<@PairStruct>, imm q>)
  -> !lit.ref<!lit.struct<@PairStruct>, imm q> {
  // Convert to kgen pointer - uses memoryOnly to match @PairStruct
  // CHECK-NEXT: kgen.return %arg0
  %ptr = lit.ref.to_kgen_ptr %a : !lit.ref<!lit.struct<@PairStruct>, imm q>
                                -> !kgen.pointer<!kgen.struct<(si32, ui32) memoryOnly>>
  // Convert back to ref
  %ref = lit.ref.from_kgen_ptr %ptr : !kgen.pointer<!kgen.struct<(si32, ui32) memoryOnly>>
                                    -> !lit.ref<!lit.struct<@PairStruct>, imm q>
  kgen.return %ref: !lit.ref<!lit.struct<@PairStruct>, imm q>
}

lit.struct.decl @PairStruct {
  lit.struct.field x : si32
  lit.struct.field y : ui32
}

//===----------------------------------------------------------------------===//
// Reference Lowering
//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.generator @gerToGEPFooFromBar
lit.fn @gerToGEPFooFromBar<l: !lit.origin<true>, l2: !lit.origin<true>>
  (%arg0: !lit.ref<@PairStruct, mut l>, %arg1: si32) -> si32 {
  // CHECK-NEXT: %0 = kgen.struct.gep %arg0[0] : <struct<(si32, ui32) memoryOnly>>
  %0 = lit.ref.struct.ger %arg0[x] : <@PairStruct, mut l> -> si32

  // This rebind should be removed entirely by lower types.
  %rb = kgen.rebind %0 : !lit.ref<si32, mut l->x> to !lit.ref<si32, mut l2>

  // CHECK-NEXT: pop.store %arg1, %0
  lit.ref.store %arg1, %rb : <si32, mut l2>

  // CHECK-NEXT: %1 = pop.load %0 : !kgen.pointer<si32>
  %a = lit.ref.load %0 : !lit.ref<si32, mut l->x>
  // CHECK-NEXT: kgen.return %1
  kgen.return %a : si32
}

// CHECK-LABEL: kgen.generator @gerByIndexToKGEN
lit.fn @gerByIndexToKGEN<l: !lit.origin<true>>
  (%arg0: !lit.ref<@PairStruct, mut l>) -> si32 {
  // CHECK-NEXT: %0 = kgen.struct.gep %arg0[0] : <struct<(si32, ui32) memoryOnly>>
  %0 = lit.ref.struct.ger %arg0[idx 0] : <@PairStruct, mut l> -> <si32, mut l>
  // CHECK-NEXT: %1 = pop.load %0 : !kgen.pointer<si32>
  %a = lit.ref.load %0 : !lit.ref<si32, mut l>
  // CHECK-NEXT: kgen.return %1
  kgen.return %a : si32
}

// CHECK-LABEL: kgen.generator @gerASByIndexToKGEN
lit.fn @gerASByIndexToKGEN<l: !lit.origin<true>>
  (%arg0: !lit.ref<@PairStruct, mut l, 3>) -> si32 {
  // CHECK-NEXT: %0 = kgen.struct.gep %arg0[0] : <struct<(si32, ui32) memoryOnly>, 3>
  %0 = lit.ref.struct.ger %arg0[idx 0] : <@PairStruct, mut l, 3> -> <si32, mut l, 3>
  // CHECK-NEXT: %1 = pop.load %0 : !kgen.pointer<si32, 3>
  %a = lit.ref.load %0 : !lit.ref<si32, mut l, 3>
  // CHECK-NEXT: kgen.return %1
  kgen.return %a : si32
}

// Issue #29038 - lower lit can't change positions of parameters.
// CHECK-LABEL: kgen.generator @takes_val_after_origin
// CHECK-SAME: <type: type>(%arg0: !kgen.pointer<type>)
lit.fn @takes_val_after_origin<life: origin<true>, type: type>(%a: !lit.ref<type, mut life>) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @does_memcpy
lit.fn @does_memcpy<l: !lit.origin<true>, l2: !lit.origin<true>>
  (%arg0: !lit.ref<@PairStruct, mut l>, %arg1: !lit.ref<@PairStruct, mut l>, %arg2: !lit.ref<@PairStruct, mut l, 3>) -> si32 {
  // CHECK-NEXT: %0 = kgen.struct.gep %arg0[0] : <struct<(si32, ui32) memoryOnly>>
  %0 = lit.ref.struct.ger %arg0[x] : <@PairStruct, mut l> -> si32
  // CHECK-NEXT: %1 = kgen.struct.gep %arg1[0] : <struct<(si32, ui32) memoryOnly>>
  %1 = lit.ref.struct.ger %arg1[x] : <@PairStruct, mut l> -> si32
  // CHECK-NEXT: %2 = kgen.struct.gep %arg2[0] : <struct<(si32, ui32) memoryOnly>, 3>
  %2 = lit.ref.struct.ger %arg2[x] : <@PairStruct, mut l, 3> -> si32

  // This rebind should be removed entirely by lower types.
  %rb = kgen.rebind %0 : !lit.ref<si32, mut l->x> to !lit.ref<si32, mut l2>

  // CHECK-NEXT: [[LEN:%.*]] = kgen.param.constant = <get_sizeof(si32, current_target())>
  // CHECK-NEXT: pop.memcpy %0, %1, [[LEN]] : !kgen.pointer<si32> -> !kgen.pointer<si32>
  lit.memcpy %1, %rb : !lit.ref<si32, mut l->x> -> !lit.ref<si32, mut l2>

  // CHECK-NEXT: [[LEN:%.*]] = kgen.param.constant = <get_sizeof(si32, current_target())>
  // CHECK-NEXT: pop.memcpy %2, %1, [[LEN]] : !kgen.pointer<si32> -> !kgen.pointer<si32, 3>
  lit.memcpy %1, %2 : !lit.ref<si32, mut l->x> -> !lit.ref<si32, mut l->x, 3>

  // CHECK-NEXT: [[RET:%.*]] = pop.load %0 : !kgen.pointer<si32>
  %a = lit.ref.load %0 : !lit.ref<si32, mut l->x>

  // CHECK-NEXT: kgen.return [[RET]]
  kgen.return %a : si32
}

//===----------------------------------------------------------------------===//
// Reference Pack Lowering
//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.generator @takes_pack<types: param_list<type>>
lit.fn @takes_pack
<life: !lit.origin<true>, types: !kgen.param_list<!kgen.type>>
// CHECK-SAME: (%arg0: !kgen.struct<variadic_ptr_map(:param_list<type> types, 42) isParamPack>)
// CHECK-SAME: sourceParamList = #kgen.pog_list<[<"life", pos_or_kw, not_vararg>, <"types", pos_or_kw, not_vararg>]>
(%args: !lit.ref.pack<:param_list<!kgen.type> types, mut life, 42>) {

  // CHECK-NEXT: [[E:%.*]] = kgen.struct.extract %arg0[0] : <variadic_ptr_map(:param_list<type> types, 42) isParamPack>
  // CHECK-NEXT: kgen.rebind [[E]] : !kgen.param<{{.*}}> to !kgen.pointer
  %v1 = lit.ref.pack.extract %args[0]: !lit.ref.pack<:param_list<!kgen.type> types, mut life, 42>

  // CHECK-NEXT: [[E:%.*]] = kgen.struct.extract %arg0[1] : <variadic_ptr_map(:param_list<type> types, 42) isParamPack>
  // CHECK-NEXT: kgen.rebind [[E]] : !kgen.param<{{.*}}> to !kgen.pointer
  %v2 = lit.ref.pack.extract %args[1]: !lit.ref.pack<:param_list<!kgen.type> types, mut life, 42>

  kgen.return
}

// CHECK-LABEL: kgen.generator @pass_pack
lit.fn @pass_pack<life: !lit.origin<true>>
  (%index: !lit.ref<index, mut life, 42>,
   %float: !lit.ref<f32, mut life, 42>) {

  // CHECK-NEXT: kgen.struct.create(%arg0, %arg1) : !kgen.struct<(pointer<index, 42>, pointer<f32, 42>) isParamPack>
  %pack = lit.ref.pack.create(%index, %float) :
    !lit.ref.pack<:param_list<!kgen.type> [index, f32], mut life, 42>
  // CHECK-NEXT: kgen.call @takes_pack<:param_list<type> [index, f32]>(%0)
  kgen.call @takes_pack<:origin<true> life, :param_list<!kgen.type> [index, f32]>(%pack)
     : (!lit.ref.pack<:param_list<!kgen.type> [index, f32], mut life, 42>) -> ()

  // CHECK-NEXT: kgen.param.constant: struct<(pointer<i8>, pointer<ui4>, pointer<i32>) isParamPack> = <{ store_to_mem(3), store_to_mem(1), store_to_mem(4) }>
  %3 = kgen.param.constant: !lit.ref.pack<:param_list<!kgen.type> [i8, ui4, i32], mut life, 0>
     = <<store_to_mem(3), store_to_mem(1), store_to_mem(4)>>

  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @ref_pack_from_pointer_pack(
// CHECK-SAME: %arg0: !kgen.struct<(pointer<index, 4>, pointer<f32, 4>) isParamPack>) -> !kgen.struct<(pointer<index, 4>, pointer<f32, 4>) isParamPack>
kgen.func @ref_pack_from_pointer_pack(
    %pack: !kgen.struct<(pointer<index, 4>, pointer<f32, 4>) isParamPack>)
    -> !lit.ref.pack<:param_list<!kgen.type> [index, f32], imm #lit.any.origin, 4> {
  // CHECK: kgen.return %arg0 : !kgen.struct<(pointer<index, 4>, pointer<f32, 4>) isParamPack>
  %0 = lit.ref.pack.from_pointer_pack %pack
    : !kgen.struct<(pointer<index, 4>, pointer<f32, 4>) isParamPack>
   -> !lit.ref.pack<:param_list<!kgen.type> [index, f32], imm #lit.any.origin, 4>
  kgen.return %0 : !lit.ref.pack<:param_list<!kgen.type> [index, f32], imm #lit.any.origin, 4>
}

// -----

lit.fn @unbox(%arg: !lit.struct<@Int>) -> index {
  %0 = index.constant 0
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @parameterized_declref_type
lit.fn @parameterized_declref_type() {
  // CHECK-NEXT: array<2, simd<apply(:(!kgen.struct<()>) -> index @unbox, { }), f32>>
  %3 = pop.stack_allocation 1 x @StaticTuple<2,
    :type !lit.struct<@SIMD<:@Int #lit.struct<{}>, :dtype f32>>>
  kgen.return
}

lit.struct.decl @SIMD<size: @Int, type: dtype> register_passable {
  lit.struct.field value : !kgen.simd<apply(:(!lit.struct<@Int>) -> index @unbox, size), type>
}

lit.struct.decl @Int register_passable {}

lit.struct.decl @StaticTuple<size, ty: type> register_passable {
  lit.struct.field array : !pop.array<size, ty>
}

// -----

// CHECK-LABEL: kgen.generator @nested_declref_type
// CHECK-SAME: !kgen.generator<(!kgen.simd<apply(:(index) -> index @pass, 1), si32>
lit.fn @nested_declref_type(
    %arg1: !lit.struct<@UnaryClosure<:type !lit.struct<@SIMD<1>>>>) {
  kgen.return
}

lit.fn @pass(%arg0: index) -> index {
  kgen.return %arg0 : index
}

lit.struct.decl @SIMD<size> register_passable {
  lit.struct.field value : !kgen.simd<apply(:(index) -> index @pass, size), si32>
}

lit.struct.decl @UnaryClosure<input_type: type> register_passable {
  lit.struct.field value : !kgen.generator<(!kgen.param<input_type>) -> ()>
}

//===----------------------------------------------------------------------===//
// Recursive Structs
//===----------------------------------------------------------------------===//

// -----

lit.struct.decl @Bar {
  lit.struct.field x : !lit.struct<@Pointer<:type !lit.struct<@Foo>>>
  lit.struct.field y : ui32
}

lit.struct.decl @Foo {
  lit.struct.field x : !lit.struct<@Bar>
  lit.struct.field y : f32
}

lit.struct.decl @Pointer<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

!bar_ref = !lit.struct<@Bar>
!foo_ref = !lit.struct<@Foo>
!foo_ptr_ref = !lit.struct<@Pointer<:type !foo_ref>>
!null_ptr = !kgen.pointer<none>


// CHECK-LABEL: @makeBar
kgen.func @makeBar(%arg0: !foo_ptr_ref, %arg1: ui32) -> !bar_ref {
  kgen.unreachable
}

// CHECK-LABEL: @structInsertUIntToBar
kgen.func @structInsertUIntToBar(%arg0: ui32, %arg1: !bar_ref) -> !bar_ref {
  // CHECK: %0 = kgen.struct.replace %arg0, %arg1[1] : !kgen.struct<(pointer<none>, ui32) memoryOnly>
  // CHECK: kgen.return %0 : !kgen.struct<(pointer<none>, ui32) memoryOnly>
  %0 = lit.struct.insert %arg0, %arg1[y] : ui32 into !bar_ref
  kgen.return %0 : !bar_ref
}

// CHECK-LABEL: @structInsertFooPtrToBar
kgen.func @structInsertFooPtrToBar(%arg0: !foo_ptr_ref, %arg1: !bar_ref) -> !bar_ref {
  // CHECK: [[V0:%.*]] = kgen.struct.replace %arg0, %arg1[0] : !kgen.struct<(pointer<none>, ui32) memoryOnly>
  // CHECK: kgen.return [[V0]] : !kgen.struct<(pointer<none>, ui32) memoryOnly>
  %0 = lit.struct.insert %arg0, %arg1[x] : !foo_ptr_ref into !bar_ref
  kgen.return %0 : !bar_ref
}

// CHECK-LABEL: @structInsertBarToFoo
kgen.func @structInsertBarToFoo(%arg0: !foo_ptr_ref, %arg1: ui32,  %arg2: !foo_ref) -> !foo_ref {
  // CHECK: [[V0:%.*]] = kgen.call @makeBar(%arg0, %arg1) : (!kgen.pointer<none>, ui32) -> !kgen.struct<(pointer<none>, ui32) memoryOnly>
  // CHECK: [[V1:%.*]] = kgen.struct.replace [[V0]], %arg2[0] : !kgen.struct<(struct<(pointer<none>, ui32) memoryOnly>, f32) memoryOnly>
  // CHECK: kgen.return [[V1]] : !kgen.struct<(struct<(pointer<none>, ui32) memoryOnly>, f32) memoryOnly>

  %0 = kgen.call @makeBar(%arg0, %arg1): (!foo_ptr_ref, ui32) -> !bar_ref
  %1 = lit.struct.insert %0, %arg2[x] : !bar_ref into !foo_ref
  kgen.return %1 : !foo_ref
}

// CHECK-LABEL: @structExtractFooFromBar
kgen.func @structExtractFooFromBar(%arg0: !bar_ref) -> !foo_ptr_ref {
  // CHECK: [[V0:%.*]] = kgen.struct.extract %arg0[0] : <(pointer<none>, ui32) memoryOnly>
  // CHECK: kgen.return [[V0]] : !kgen.pointer<none>
  %0 = lit.struct.extract %arg0[x] : !foo_ptr_ref from !bar_ref
  kgen.return %0 : !foo_ptr_ref
}

// CHECK-LABEL: @structExtractBarFromFoo
kgen.func @structExtractBarFromFoo(%arg0: !foo_ref) -> !bar_ref {
  // CHECK: %0 = kgen.struct.extract %arg0[0] : <(struct<(pointer<none>, ui32) memoryOnly>, f32) memoryOnly>
  // CHECK: kgen.return %0 : !kgen.struct<(pointer<none>, ui32) memoryOnly>
  %0 = lit.struct.extract %arg0[x] : !bar_ref from !foo_ref
  kgen.return %0 : !bar_ref
}

lit.struct.decl @Recursive register_passable {
  lit.struct.field x : !kgen.pointer<@Recursive>
}

// CHECK-LABEL: @thing
// CHECK: -> !kgen.pointer<none>
lit.fn @thing() -> !lit.struct<@Recursive> {
  // CHECK: kgen.unreachable
  kgen.unreachable
}

// CHECK-LABEL: kgen.generator @foo<T: type>()
lit.fn @foo<T: type>() {
  kgen.return
}

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

// CHECK-NOT: lit.trait.decl
lit.trait.decl @Trait {
}

// CHECK: kgen.generator @trait_fn<T: type>()
lit.fn @trait_fn<T: trait<@Trait>>() {
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//
// Erase pointer types in struct
//===----------------------------------------------------------------------===//

!ptr = !lit.struct<@Ptr>

lit.struct.decl @Ptr register_passable {
  lit.struct.field ptr: !kgen.pointer<index>
}

// CHECK-LABEL:  kgen.func @f(%arg0: !kgen.pointer<index>)
kgen.func @f(%x: !kgen.pointer<index>) {
    kgen.return
}

// CHECK-LABEL: kgen.func @pass_it(%arg0: !kgen.pointer<none>)
kgen.func @pass_it(%y: !ptr) {
    // CHECK: [[V0:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<index>
    // CHECK: kgen.call @f([[V0]]) : (!kgen.pointer<index>) -> ()
    %0 = lit.struct.extract %y[ptr]: !kgen.pointer<index> from !ptr
    kgen.call @f(%0): (!kgen.pointer<index>)->()
    kgen.return
}


// -----

//===----------------------------------------------------------------------===//
// More Recursive Structs
//===----------------------------------------------------------------------===//

!bar = !lit.struct<@Bar>
!foo = !lit.struct<@Foo>

lit.struct.decl @Bar register_passable {
  lit.struct.field foo: !foo
}

lit.struct.decl @Foo register_passable {
  lit.struct.field bar_ptr: !kgen.pointer<@Bar>
}

// CHECK-LABEL:  kgen.func @f(%arg0: !kgen.pointer<none>)
kgen.func @f(%bar: !bar) {
    // CHECK: [[V0:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<pointer<none>>
    // CHECK: kgen.call @g([[V0]]) : (!kgen.pointer<pointer<none>>) -> ()

    %foo = lit.struct.extract %bar[foo]: !foo from !bar
    %bar_ptr = lit.struct.extract %foo[bar_ptr]: !kgen.pointer<@Bar> from !foo
    kgen.call @g(%bar_ptr): (!kgen.pointer<@Bar>)->()
    kgen.return
}

// CHECK-LABEL: kgen.func @g(%arg0: !kgen.pointer<pointer<none>>)
kgen.func @g(%arg0: !kgen.pointer<@Bar>) {
    kgen.return
}

// -----

lit.struct.decl @Pointer<T: type, as> register_passable_trivial {
  lit.struct.field value: !kgen.pointer<T, as>
}

lit.fn @make_ptr<T: type>() -> !kgen.pointer<T> {
  kgen.unreachable
}

// CHECK-LABEL: kgen.generator @pointer_const
lit.fn @pointer_const<T: type>() {
  // CHECK-NEXT: constant: pointer<none> = <ptr_bitcast(:pointer<T> apply(:() -> !kgen.pointer<T> @make_ptr<:type T>))>
  kgen.param.constant: @Pointer<:type T, 0, :i1 0> = <{value: pointer<T> = apply(:() -> !kgen.pointer<T> @make_ptr<:type T>)}>
  // CHECK-NEXT: constant: pointer<none, 1> = <0>
  kgen.param.constant: @Pointer<:type T, 1, :i1 1> = <{value: pointer<T, 1> = 0}>
  kgen.return
}

// -----


lit.struct.decl @Thing<T: trait<@Foo>> {
}

lit.fn @x() {
  kgen.return
}

// CHECK: -> !kgen.struct<() memoryOnly>
lit.fn @example<T: trait<@Bar>>() -> !lit.struct<@Thing<:trait<@Foo> [!kgen.param<:trait<@Bar> T>]>> {
  kgen.unreachable
}

// -----

//===----------------------------------------------------------------------===//
// Func & Gen Type value domain lowering
//===----------------------------------------------------------------------===//

lit.struct.decl @Int register_passable {
  lit.struct.field value : index
}

lit.struct.decl @StaticTuple<size, ty: type> register_passable {
  lit.struct.field array : !pop.array<size, ty>
}

kgen.generator @type_values() {
  // CHECK: kgen.param.declare func_type: type = <[
  // CHECK-SAME: (index, !kgen.typevalue<#kgen.genref<@Int>>) -> !kgen.typevalue<#kgen.genref<@StaticTuple<2, :type [typevalue<#kgen.genref<@Int>>, index]>>>
  // CHECK-SAME: (index, index) -> !pop.array<2, index>
  // CHECK-SAME: ]>
  kgen.param.declare func_type: type = <#kgen.type<(index, !lit.struct<@Int>) -> !lit.struct<@StaticTuple<2, :type #kgen.type<@Int>>>>>

  // A generator's parameter decl types stay in the type domain (so index
  // references agree with their parameter declarations); only the body is
  // lowered in the value domain.
  // CHECK-NEXT: kgen.param.declare gen_type: type = <[
  // CHECK-SAME: <index, index>typevalue<#kgen.genref<@StaticTuple<2, :type [typevalue<#kgen.genref<@Int>>, index]>>>
  // CHECK-SAME: <index, index>array<2, index>
  // CHECK-SAME: ]>
  kgen.param.declare gen_type: type = <#kgen.type<<index, !lit.struct<@Int>> !lit.struct<@StaticTuple<2, :type #kgen.type<@Int>>>>>

  kgen.return
}
