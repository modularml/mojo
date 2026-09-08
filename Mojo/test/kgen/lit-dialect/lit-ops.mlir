// RUN: kgen-opt -split-input-file -allow-unregistered-dialect %s | kgen-opt -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: lit.struct.decl @FooStruct
// CHECK-SAME: <size, dtype: dtype, ty: type> {
// CHECK-NEXT: a : index
// CHECK-NEXT: b : !kgen.scalar<dtype>
// CHECK-NEXT: c : !kgen.param<ty>
lit.struct.decl @FooStruct<size, dtype: dtype, ty: type> {
  lit.struct.field a : index
  lit.struct.field b : !kgen.scalar<dtype>
  lit.struct.field c : !kgen.param<ty>
}

// CHECK-LABEL: lit.struct.decl @EmptyStruct
// CHECK-NEXT: }
lit.struct.decl @EmptyStruct {
}

// CHECK-LABEL: lit.struct.decl @ValueType
lit.struct.decl @ValueType
 // CHECK-NEXT: move :() -> () @ValueType::@__moveinit__
 move :() -> () @ValueType::@__moveinit__
 // CHECK-NEXT: copy :() -> () @ValueType::@__copyinit__
 copy :() -> () @ValueType::@__copyinit__ {
}

// CHECK-LABEL: @struct_insert
kgen.generator @struct_insert(%a: index, %struct: !lit.struct<@FooStruct<2, :dtype f32, :type i32>>) {
  // CHECK: lit.struct.insert %{{.*}}, %{{.*}}[a] : index into !lit.struct<@FooStruct
  %0 = lit.struct.insert %a, %struct[a] : index into !lit.struct<@FooStruct<2, :dtype f32, :type i32>>
  kgen.return
}

// CHECK-LABEL: @struct_extract
kgen.generator @struct_extract(%struct: !lit.struct<@FooStruct<2, :dtype f32, :type i32>>) {
  // CHECK: lit.struct.extract %{{.*}}[a] : index from !lit.struct<@FooStruct
  %0 = lit.struct.extract %struct[a] : index from !lit.struct<@FooStruct<2, :dtype f32, :type i32>>
  kgen.return
}

// CHECK-LABEL: lit.fn @calls[imm a, mut b]
lit.fn @calls[imm a, mut b](%arg0: !lit.generator<[2]() -> ()>) {
  // CHECK: lit.call @calls[imm a, mut b]() : !lit.generator<[2]() -> ()>
  lit.call @calls[imm a, mut b]() : !lit.generator<[2]() -> ()>
  // CHECK: lit.call_indirect %arg0[imm a, mut b]() : !lit.generator<[2]() -> ()>
  lit.call_indirect %arg0[imm a, mut b]() : !lit.generator<[2]() -> ()>
  kgen.return
}

// One implementation of dynamic_thing
// CHECK-LABEL: lit.fn @vardecl
lit.fn @vardecl<ty : dtype>(%x : i32) {
  // CHECK-NEXT: %a = lit.var.decl "a" imp : !lit.ref<scalar<ty>, mut life>
  %a = lit.var.decl "a" imp : !lit.ref<scalar<ty>, mut life>

  // CHECK-NEXT: %origin = lit.var.decl "origin" var : !lit.ref<index, mut lt>
  %origin = lit.var.decl "origin" var : !lit.ref<index, mut lt>
  kgen.return
}

// CHECK-LABEL: lit.struct.decl @SomeStruct<ty: dtype, n: scalar<si32> = 7>
lit.struct.decl @SomeStruct<ty: dtype, n: scalar<si32> = 7> {
  // CHECK-NEXT: lit.fn @foo() {
  lit.fn @foo() {
    kgen.return
  }

  // CHECK: %size = lit.var.decl "size" var : !lit.ref<scalar<ty>, mut life>
  %size = lit.var.decl "size" var : !lit.ref<scalar<ty>, mut life>

  // CHECK: lit.fn @getMyType
  // CHECK-NEXT: kgen.param.constant: dtype = <ty>
  lit.fn @getMyType() -> !kgen.dtype {
    %dtype = kgen.param.constant: dtype = <ty>
    kgen.return %dtype : !kgen.dtype
  }
}

// CHECK-LABEL: lit.struct.decl @struct_param_passing_kinds<
// CHECK-SAME: z: dtype, |,
// CHECK-SAME: a: dtype, b: dtype = f32, c: scalar<si32> = 1, *,
// CHECK-SAME: d: dtype, e: dtype = f16, f: scalar<si16> = 2
lit.struct.decl @struct_param_passing_kinds<
  z: dtype, |,
  a: dtype, b: dtype = f32, c: scalar<si32> = 1, *,
  d: dtype, e: dtype = f16, f: scalar<si16> = 2
> {}

// CHECK-LABEL: lit.trait.decl @T {
lit.trait.decl @T {
  // CHECK: lit.fn @f{{.*}}
  // CHECK-NEXT:  kgen.unreachable
  lit.fn @f() -> !kgen.none {
    kgen.unreachable
  }
}

// CHECK-LABEL: lit.trait.decl @RP register_passable
lit.trait.decl @RP register_passable {
}

// CHECK-LABEL: @attributesAndDecorators
lit.fn @attributesAndDecorators()
  // CHECK-NEXT: decorators <{{.*}}> attributes {isParametric} {
  decorators <:() -> () @decorator> attributes {isParametric} {
  lit.end_fn
}

// CHECK-LABEL: @end_fn
lit.fn @end_fn() {
  // CHECK-NEXT: lit.end_fn unresolved
  lit.end_fn unresolved
}


lit.fn @ref_immut<life: origin<true>>(%ref1: !lit.ref<@MyStruct, mut life>)
 -> !lit.ref<@MyStruct, muttoimm life> {
  // CHECK: %0 = lit.ref.immut %ref1 : <!lit.struct<@MyStruct>, mut life>
  %ref2 = lit.ref.immut %ref1: <!lit.struct<@MyStruct>, mut life>
  // CHECK: kgen.return %0 : !lit.ref<!lit.struct<@MyStruct>, muttoimm life>
  kgen.return %ref2: !lit.ref<!lit.struct<@MyStruct>, muttoimm life>
}

lit.fn @ref_upcast<life: origin<true>>(
    %ref1: !lit.ref<@MyStruct, mut life>)
 -> !lit.ref<@MyStruct, mut #lit.origin.subtree<#kgen.param.decl.ref<"life"> : !lit.origin<true>>> {
  // CHECK: %0 = lit.ref.upcast %ref1 : <!lit.struct<@MyStruct>, mut life> -> <!lit.struct<@MyStruct>, mut #lit.origin.subtree<#kgen.param.decl.ref<"life"> : !lit.origin<true>>>
  %ref2 = lit.ref.upcast %ref1
    : !lit.ref<!lit.struct<@MyStruct>, mut life>
    -> !lit.ref<!lit.struct<@MyStruct>, mut #lit.origin.subtree<#kgen.param.decl.ref<"life"> : !lit.origin<true>>>
  // CHECK: kgen.return %0 : !lit.ref<!lit.struct<@MyStruct>, mut #lit.origin.subtree<#kgen.param.decl.ref<"life"> : !lit.origin<true>>>
  kgen.return %ref2
    : !lit.ref<!lit.struct<@MyStruct>, mut #lit.origin.subtree<#kgen.param.decl.ref<"life"> : !lit.origin<true>>>
}

// MOCO-4453: erased accessor style — upcast a member field origin to the
// owner's subtree (Pointer[U, origin_of(self).subtree] from self._memory...).
lit.fn @ref_upcast_field_to_subtree<life: origin<true>>(
    %ref1: !lit.ref<index, mut #lit.origin.field<#kgen.param.decl.ref<"life"> : !lit.origin<true>, "_memory">>)
 -> !lit.ref<index, mut #lit.origin.subtree<#kgen.param.decl.ref<"life"> : !lit.origin<true>>> {
  // CHECK: %0 = lit.ref.upcast %ref1
  %ref2 = lit.ref.upcast %ref1
    : !lit.ref<index, mut #lit.origin.field<#kgen.param.decl.ref<"life"> : !lit.origin<true>, "_memory">>
    -> !lit.ref<index, mut #lit.origin.subtree<#kgen.param.decl.ref<"life"> : !lit.origin<true>>>
  kgen.return %ref2
    : !lit.ref<index, mut #lit.origin.subtree<#kgen.param.decl.ref<"life"> : !lit.origin<true>>>
}

// Field subtree upcasts into a wider parent subtree.
lit.fn @ref_upcast_field_subtree_to_parent<life: origin<true>>(
    %ref1: !lit.ref<index, mut #lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"life"> : !lit.origin<true>, "f"> : !lit.origin<true>>>)
 -> !lit.ref<index, mut #lit.origin.subtree<#kgen.param.decl.ref<"life"> : !lit.origin<true>>> {
  // CHECK: %0 = lit.ref.upcast %ref1
  %ref2 = lit.ref.upcast %ref1
    : !lit.ref<index, mut #lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"life"> : !lit.origin<true>, "f"> : !lit.origin<true>>>
    -> !lit.ref<index, mut #lit.origin.subtree<#kgen.param.decl.ref<"life"> : !lit.origin<true>>>
  kgen.return %ref2
    : !lit.ref<index, mut #lit.origin.subtree<#kgen.param.decl.ref<"life"> : !lit.origin<true>>>
}

// Multi-level: x.y.z upcasts to x.y~ and to x~.
lit.fn @ref_upcast_nested_to_mid_subtree<life: origin<true>>(
    %ref1: !lit.ref<index, mut #lit.origin.field<#lit.origin.field<#kgen.param.decl.ref<"life"> : !lit.origin<true>, "y"> : !lit.origin<true>, "z">>)
 -> !lit.ref<index, mut #lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"life"> : !lit.origin<true>, "y"> : !lit.origin<true>>> {
  // CHECK: %0 = lit.ref.upcast %ref1
  %ref2 = lit.ref.upcast %ref1
    : !lit.ref<index, mut #lit.origin.field<#lit.origin.field<#kgen.param.decl.ref<"life"> : !lit.origin<true>, "y"> : !lit.origin<true>, "z">>
    -> !lit.ref<index, mut #lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"life"> : !lit.origin<true>, "y"> : !lit.origin<true>>>
  kgen.return %ref2
    : !lit.ref<index, mut #lit.origin.subtree<#lit.origin.field<#kgen.param.decl.ref<"life"> : !lit.origin<true>, "y"> : !lit.origin<true>>>
}

lit.fn @ref_upcast_nested_to_root_subtree<life: origin<true>>(
    %ref1: !lit.ref<index, mut #lit.origin.field<#lit.origin.field<#kgen.param.decl.ref<"life"> : !lit.origin<true>, "y"> : !lit.origin<true>, "z">>)
 -> !lit.ref<index, mut #lit.origin.subtree<#kgen.param.decl.ref<"life"> : !lit.origin<true>>> {
  // CHECK: %0 = lit.ref.upcast %ref1
  %ref2 = lit.ref.upcast %ref1
    : !lit.ref<index, mut #lit.origin.field<#lit.origin.field<#kgen.param.decl.ref<"life"> : !lit.origin<true>, "y"> : !lit.origin<true>, "z">>
    -> !lit.ref<index, mut #lit.origin.subtree<#kgen.param.decl.ref<"life"> : !lit.origin<true>>>
  kgen.return %ref2
    : !lit.ref<index, mut #lit.origin.subtree<#kgen.param.decl.ref<"life"> : !lit.origin<true>>>
}

lit.fn @ref_pointer<life: origin<true>, ilife: origin<false>>
     (%ref1: !lit.ref<@MyStruct, mut life>) {
  // CHECK: %0 = lit.ref.to_pointer %ref1 : <!lit.struct<@MyStruct>, mut life>
  %ptr = lit.ref.to_pointer %ref1: <!lit.struct<@MyStruct>, mut life>
  // CHECK: %1 = lit.ref.from_pointer %0 : <!lit.struct<@MyStruct>, imm ilife>
  %ref2 = lit.ref.from_pointer %ptr: !lit.ref<!lit.struct<@MyStruct>, imm ilife>

  // CHECK: %2 = lit.ref.to_pointer %1 : <!lit.struct<@MyStruct>, imm ilife>
  %ptr2 = lit.ref.to_pointer %ref2: !lit.ref<!lit.struct<@MyStruct>, imm ilife>
  lit.end_fn
}

lit.fn @ref_kgen_ptr<life: origin<true>, ilife: origin<false>>
     (%ref1: !lit.ref<@MyStruct, mut life>) {
  // CHECK: %0 = lit.ref.to_kgen_ptr %ref1 : <!lit.struct<@MyStruct>, mut life> -> <struct<(i64)>>
  %ptr = lit.ref.to_kgen_ptr %ref1 : !lit.ref<!lit.struct<@MyStruct>, mut life>
                                   -> !kgen.pointer<!kgen.struct<(i64)>>
  // CHECK: %1 = lit.ref.from_kgen_ptr %0 : <struct<(i64)>> -> <!lit.struct<@MyStruct>, imm ilife>
  %ref2 = lit.ref.from_kgen_ptr %ptr : !kgen.pointer<!kgen.struct<(i64)>>
                                     -> !lit.ref<!lit.struct<@MyStruct>, imm ilife>
  lit.end_fn
}

// CHECK-LABEL: @ref_pack_from_pointer_pack
kgen.generator @ref_pack_from_pointer_pack(
    %pack: !kgen.struct<(pointer<index, 4>, pointer<f32, 4>) isParamPack>) {
  // CHECK: %0 = lit.ref.pack.from_pointer_pack %arg0 : <(pointer<index, 4>, pointer<f32, 4>) isParamPack> -> <:param_list<type> [index, f32], imm #lit.any.origin, 4>
  %0 = lit.ref.pack.from_pointer_pack %pack
    : !kgen.struct<(pointer<index, 4>, pointer<f32, 4>) isParamPack>
   -> !lit.ref.pack<:param_list<!kgen.type> [index, f32], imm #lit.any.origin, 4>
  kgen.return
}

// CHECK-LABEL: lit.fn @nested_function_region
lit.fn @nested_function_region() {
  // CHECK-NEXT: hlcf.loop
  hlcf.loop {
    // CHECK-NEXT: lit.fn nested_fn()
    lit.fn nested_fn() {
      kgen.return
    }
    hlcf.continue
  }
  kgen.return
}

// -----

// CHECK-LABEL: lit.struct.decl @A
lit.struct.decl @A {
  // CHECK-NEXT: lit.fn @foo
  lit.fn @foo(%self: !lit.struct<@A>) {
    kgen.return
  }
}

// CHECK-LABEL: lit.struct.decl @B
lit.struct.decl @B {
  // CHECK-NEXT: lit.fn @foo
  lit.fn @foo(%self: !lit.struct<@B>, %a: !lit.struct<@A>) {
    // CHECK-NEXT: call_param[(!lit.struct<@A>) -> (): @A::@foo]
    kgen.call_param[(!lit.struct<@A>) -> (): @A::@foo](%a)

    kgen.call @A::@foo(%a) : (!lit.struct<@A>) -> ()
    kgen.return
  }
}

// CHECK-LABEL: lit.fn @main
lit.fn @main(%a: !lit.struct<@A>, %b: !lit.struct<@B>) {
  // CHECK-NEXT: call_param[(!lit.struct<@B>, !lit.struct<@A>) -> (): @B::@foo]
  kgen.call_param[(!lit.struct<@B>, !lit.struct<@A>) -> (): @B::@foo](%b, %a)
  // CHECK-NEXT: constant: (!lit.struct<@A>) -> () = <@A::@foo>
  %0 = kgen.param.constant: (!lit.struct<@A>) -> () = <@A::@foo>
  kgen.return
}

// CHECK-LABEL: lit.struct.decl @CrazyParams<*"m`": origin<false>> {
lit.struct.decl @CrazyParams<*"m`": origin<false>> {
}

lit.struct.decl @LifetimeRef<b: origin<false>> {
  lit.struct.field b : !lit.generator<(!lit.ref<@A, imm *(0,1)>) -> ()>
}

// -----

// CHECK-LABEL: lit.struct.decl @A<N>
lit.struct.decl @A<N> {
  // CHECK-NEXT: lit.fn @foo<M>
  lit.fn @foo<M>(%self: !lit.struct<@A<N>>) -> index {
    %0 = kgen.param.constant = <add(N, M)>
    kgen.return %0 : index
  }
}

// CHECK-LABEL: lit.fn @main
lit.fn @main(%a: !lit.struct<@A<1>>) {
  // CHECK-NEXT: call_param[(!lit.struct<@A<1>>) -> index: @A::@foo<1, 2>]
  %- = kgen.call_param[(!lit.struct<@A<1>>) -> index: @A::@foo<1, 2>](%a)
  kgen.return
}

// CHECK-LABEL: lit.struct.decl @NoFields {
// CHECK-NEXT: }
lit.struct.decl @NoFields {}

// COM: Types from the standard library.
lit.struct.decl @Error {}
lit.struct.decl @Int {}

// CHECK-LABEL: @raises_error
lit.fn @raises_error(%raise: !kgen.scalar<bool>, %err: !lit.struct<@Error>, %value: !lit.struct<@Int>) -> !kgen.variant<@Error, @Int> {
  hlcf.if %raise {
    // CHECK: %[[ERR:.*]] = kgen.variant.create %err
    %result = kgen.variant.create %err, 0 : <@Error, @Int>
    // CHECK: kgen.return %[[ERR]]
    kgen.return %result : !kgen.variant<@Error, @Int>
  } else {
    hlcf.yield
  }
  // CHECK: %[[VALUE:.*]] = kgen.variant.create %value
  %result = kgen.variant.create %value, 1 : <@Error, @Int>
  // CHECK: kgen.return %[[VALUE]]
  kgen.return %result : !kgen.variant<@Error, @Int>
}

// CHECK-LABEL: @try_op
lit.fn @try_op(%err: !lit.struct<@Error>, %int: !lit.struct<@Int>) -> !lit.struct<@Int> {
  // CHECK-NEXT: lit.try
  lit.try {
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  // CHECK-NEXT: } except (%{{.*}}: !lit.struct<@Error>) {
  } except (%exception: !lit.struct<@Error>) {
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  // CHECK-NEXT: } finally {
  } finally {
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  }
  kgen.return %int : !lit.struct<@Int>
}

// CHECK-LABEL: @try_in_loop
lit.fn @try_in_loop(%cond: !kgen.scalar<bool>) {
  // CHECK-NEXT: lit.loop
  lit.loop {
    hlcf.if %cond {
      hlcf.yield
    } else {
      lit.loop.break.else
    }

    // CHECK: lit.try
    lit.try {
      // CHECK-NEXT: hlcf.if
      hlcf.if %cond {
        // CHECK-NEXT: hlcf.break
        hlcf.break
      // CHECK-NEXT: else
      } else {
        // CHECK-NEXT: hlcf.yield
        hlcf.yield
      }
      // CHECK: lit.try.yield
      lit.try.yield
    // CHECK-NEXT: except
    } except (%arg0: !lit.struct<@Error>) {
      // CHECK-NEXT: hlcf.break
      hlcf.break
    // CHECK-NEXT: else
    } else {
      // CHECK-NEXT: lit.try.yield
      lit.try.yield
    // CHECK-NEXT: finally
    } finally {
      // CHECK-NEXT: lit.try.yield
      lit.try.yield
    }
    // CHECK: lit.loop.continue
    lit.loop.continue
  } else {
    lit.loop.yield
  }
  // CHECK: kgen.return
  kgen.return
}

// -----

// CHECK-DAG: !B = !lit.struct<@module::@B>
// CHECK-DAG: !A = !lit.struct<@module::@A>

// CHECK-LABEL: lit.file_module @module
lit.file_module @module {
  // CHECK: lit.struct.decl @A
  lit.struct.decl @A {}

  // CHECK: lit.struct.decl @B
  lit.struct.decl @B {
    // CHECK-NEXT: lit.fn @foo(%{{.*}}: !B, %{{.*}}: !kgen.pointer<!A>
    lit.fn @foo(%self: !lit.struct<@module::@B>, %a: !kgen.pointer<@module::@A>) {
      kgen.return
    }
  }
}

// CHECK-LABEL: lit.fn @main
lit.fn @main(%a: !kgen.pointer<@module::@A>, %b: !lit.struct<@module::@B>) {
  // CHECK-NEXT: call_param[(!B, !kgen.pointer<!A>) -> (): @module::@B::@foo]
  kgen.call_param[(!lit.struct<@module::@B>, !kgen.pointer<@module::@A>) -> (): @module::@B::@foo](%b, %a)
  kgen.return
}

lit.struct.decl @Error {}

// CHECK-LABEL: @lexical_terminators
lit.fn @lexical_terminators(%cond: !kgen.scalar<bool>) throws -> !kgen.variant<i32, i64> {
  // CHECK: lit.loop
  lit.loop {
    // CHECK: hlcf.if
    hlcf.if %cond {
      // CHECK-NEXT: lit.break
      lit.break
      hlcf.yield
    // CHECK: else
    } else {
      // CHECK-NEXT: lit.continue
      lit.continue
      hlcf.yield
    }
    lit.loop.continue
  } else {
    lit.loop.yield
  }

  // CHECK: lit.try
  lit.try {
    // CHECK: lit.raise
    lit.raise
    lit.try.yield
  } except {
    lit.try.yield
  } else {
    lit.try.yield
  } finally {
    lit.try.yield
  }
  // CHECK: lit.end_fn
  lit.end_fn
}

// CHECK-LABEL: lit.fn @async_fn() async
lit.fn @async_fn() async {
  lit.end_fn
}

lit.fn @async_fn_byref_result(%res: !lit.ref<index, mut #lit.any.origin> byref_result) async {
  lit.end_fn
}

lit.fn @async_fn_throws(%err: !lit.ref<index, mut #lit.any.origin> byref_error, %res: !lit.ref<index, mut #lit.any.origin> byref_result) async|throws {
  lit.end_fn
}

// CHECK-LABEL: lit.fn @call_async_fn
lit.fn @call_async_fn() {
  // CHECK-NEXT: lit.async.call[!lit.generator<() async -> ()>: @async_fn]()
  lit.async.call[!lit.generator<() async -> ()>: @async_fn]()
  // CHECK-NEXT: lit.async.call[!lit.generator<("res": !lit.ref<index, mut #lit.any.origin> byref_result) async -> ()>: @async_fn_byref_result]()
  lit.async.call[!lit.generator<("res": !lit.ref<index, mut #lit.any.origin> byref_result) async -> ()>: @async_fn_byref_result]()
  // CHECK-NEXT: lit.async.call[!lit.generator<("err": !lit.ref<index, mut #lit.any.origin> byref_error, "res": !lit.ref<index, mut #lit.any.origin> byref_result) throws|async -> ()>: @async_fn_throws]()
  lit.async.call[!lit.generator<("err": !lit.ref<index, mut #lit.any.origin> byref_error, "res": !lit.ref<index, mut #lit.any.origin> byref_result) async|throws -> ()>: @async_fn_throws]()
  lit.end_fn
}

lit.struct.decl @GiveMeDefault {
  lit.struct.field size : !kgen.pointer<scalar<index>>
}

// CHECK-LABEL: lit.fn @default_struct
// CHECK-SAME: !lit.struct<@GiveMeDefault> = {1}
lit.fn @default_struct(%arg0: !lit.struct<@GiveMeDefault> = {1}) {
  kgen.return
}


lit.struct.decl @OuterParams<ty: type, fn: () -> !kgen.param<ty>> {
  lit.fn @some_func() {
    kgen.return
  }
}

// CHECK-LABEL: lit.fn @ref_it
lit.fn @ref_it() {
  // CHECK: F: <type, () -> !kgen.param<*(1,0)>>() -> () = <@OuterParams::@some_func>
  kgen.param.declare F: <type, () -> !kgen.param<*(1,0)>>() -> () = <@OuterParams::@some_func>
  kgen.return
}

// CHECK-LABEL: lit.struct.decl @FuncParamStruct
// CHECK-SAME: <c: !lit.generator<<type>(!kgen.param<*(0,0)>) -> ()>>
lit.struct.decl @FuncParamStruct<c: !lit.generator<<type>(!kgen.param<*(0,0)>) -> ()>>  {
  // CHECK: lit.fn @foo(%x: !kgen.pointer<!lit.struct<@FuncParamStruct<:!lit.generator<<type>(!kgen.param<*(0,0)>) -> ()> c>>>)
  lit.fn @foo(%x: !kgen.pointer<@FuncParamStruct<:!lit.generator<<type>(!kgen.param<*(0,0)>) -> ()> c>>) {
    lit.end_fn
  }
  // CHECK-LABEL: lit.fn @bar
  lit.fn @bar(%x: !kgen.pointer<@FuncParamStruct<:!lit.generator<<type>(!kgen.param<*(0,0)>) -> ()> c>>) {
    // CHECK: call @FuncParamStruct::@foo<:!lit.generator<<type>(!kgen.param<*(0,0)>) -> ()> c>(%x)
    kgen.call @FuncParamStruct::@foo<:!lit.generator<<type>(!kgen.param<*(0,0)>) -> ()> c>(%x)
    // CHECK-SAME: ("x": !kgen.pointer<!lit.struct<@FuncParamStruct<:!lit.generator<<type>(!kgen.param<*(0,0)>) -> ()> c>>>) -> ()
      : !lit.generator<("x": !kgen.pointer<@FuncParamStruct<:!lit.generator<<type>(!kgen.param<*(0,0)>) -> ()> c>>) -> ()>
    lit.end_fn
  }
}

// -----

#file = #debuginfo.file<"foo.mlir" in "">
#loc = loc("foo.mlir":7:8)

// CHECK-LABEL: lit.struct.decl @Foo
lit.struct.decl @Foo {
  lit.struct.field value : index
} loc(fused<#file>[#loc])

// -----

// struct with traits
// CHECK-LABEL: lit.trait.decl @Trait1
lit.trait.decl @Trait1 {}
lit.trait.decl @Trait2 {}
lit.trait.decl @Trait3 {}

// CHECK-LABEL: lit.struct.decl @StructHasTraits
// CHECK-SAME: (trait<@Trait1, @Trait2, @Trait3>)
lit.struct.decl @StructHasTraits(trait<@Trait1, @Trait2, @Trait3>) {}

// CHECK-LABEL: lit.fn @lit_loop
lit.fn @lit_loop() {
  lit.loop {
    %0 = kgen.param.constant: scalar<bool> = <true>
    hlcf.if %0 {
      hlcf.yield
    } else {
      lit.loop.break.else
    }

    // CHECK: lit.loop.continue
    lit.loop.continue
  } else {
    // CHECK: lit.loop.yield
    lit.loop.yield
  } {unrollLevel = #hlcf<unroll_level full>}

  kgen.return
}

// -----

lit.fn @load_consume(%arg0 : !lit.ref<index, mut #lit.any.origin>) -> index {
  %0 = lit.load.consume %arg0 : !lit.ref<index, mut #lit.any.origin>
  kgen.return %0 : index
}

// -----

!FnWithOrigin = !lit.generator<<origin<false>>(!lit.ref<index, imm *(0,0)>) -> ()>
!FnBound = !lit.generator<(!lit.ref<index, imm #lit.any.origin>) -> ()>
!FnWithOriginAndConstraints = !lit.generator<<origin<false>, {<true, loc("bind_params":1:1)>, <true, loc("bind_params":1:2)>}>(!lit.ref<index, imm *(0,0)>) -> ()>
!FnOriginBoundOneConstraint = !lit.generator<<{<true, loc("bind_params":1:2)>}>(!lit.ref<index, imm #lit.any.origin>) -> ()>

lit.fn @origin_callee<lt: !lit.origin<false>>(%x: !lit.ref<index, mut lt>) -> () {
  lit.end_fn
}

// CHECK-LABEL: @bind_params_origin_unbound
kgen.generator @bind_params_origin_unbound(%fn: !FnWithOrigin) {
  // CHECK: lit.bind_params %{{.*}} : !lit.generator<<origin<false>>(!lit.ref<index, imm *(0,0)>) -> ()>, ? to !lit.generator<<origin<false>>
  %partial = lit.bind_params %fn : !FnWithOrigin, ? to !FnWithOrigin
  kgen.return
}

// CHECK-LABEL: @bind_params_origin_bound
kgen.generator @bind_params_origin_bound(%fn: !FnWithOrigin, %x: !lit.ref<index, imm #lit.any.origin>) {
  // CHECK: lit.bind_params %{{.*}} : !lit.generator<<origin<false>>(!lit.ref<index, imm *(0,0)>) -> ()>, :origin<false> #lit.any.origin to !lit.generator<(!lit.ref<index, imm #lit.any.origin>) -> ()>
  %bound = lit.bind_params %fn : !FnWithOrigin, :origin<false> #lit.any.origin to !FnBound
  // CHECK: lit.call_indirect %{{.*}}(%{{.*}}) : !lit.generator<(!lit.ref<index, imm #lit.any.origin>) -> ()>
  lit.call_indirect %bound(%x) : !FnBound
  kgen.return
}

// CHECK-LABEL: @bind_params_origin_discharged
kgen.generator @bind_params_origin_discharged(%fn: !FnWithOriginAndConstraints) {
  // CHECK: lit.bind_params %{{.*}} : !lit.generator<<origin<false>, {{.*}}>(!lit.ref<index, imm *(0,0)>) -> ()>, :origin<false> #lit.any.origin | "10" to !lit.generator<<{<true, {{.*}}>}>(!lit.ref<index, imm #lit.any.origin>) -> ()>
  %bound = lit.bind_params %fn : !FnWithOriginAndConstraints, :origin<false> #lit.any.origin | "10" to !FnOriginBoundOneConstraint
  kgen.return
}

// -----

// COM: Where clauses on parameters

// CHECK-DAG: #[[LOC1:.+]] = loc("test.mlir":1:2)
#loc = loc("test.mlir":1:2)
// CHECK-DAG: #[[LOC2:.+]] = loc("test.mlir":3:4)
#loc1 = loc("test.mlir":3:4)

// CHECK-LABEL: lit.fn @has_body_constraints
// CHECK-SAME: <x, y, {<true, #loc>, <true, #loc1>}>
lit.fn @has_body_constraints<
  x: index,
  y: index,
  {<true, #loc>, <true, #loc1>}
>() -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}
