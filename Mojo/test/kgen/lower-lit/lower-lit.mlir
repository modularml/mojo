// RUN: kgen-opt -verify-parameters -lower-lit -split-input-file %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

lit.fn @callee[imm a, mut b]() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @calls
lit.fn @calls<f: !lit.generator<[2]() -> ()>>[imm a, mut b](%arg0: !lit.generator<[2]() -> ()>) {
  // CHECK: kgen.call @callee() : () -> ()
  lit.call @callee[imm a, mut b]() : !lit.generator<[2]() -> ()>
  // CHECK: kgen.call_param[() -> (): f]()
  lit.call [!lit.generator<[2]() -> ()>: f][imm a, mut b]()
  // CHECK: kgen.call_indirect %arg0() : () -> ()
  lit.call_indirect %arg0[imm a, mut b]() : !lit.generator<[2]() -> ()>

  // CHECK: kgen.call tail @callee() : () -> ()
  lit.call tail @callee[imm a, mut b]() : !lit.generator<[2]() -> ()>
  // CHECK: kgen.call_indirect musttail %arg0() : () -> ()
  lit.call_indirect musttail %arg0[imm a, mut b]() : !lit.generator<[2]() -> ()>
  kgen.return
}

lit.fn @async_fn_throws(%err: !lit.ref<index, mut #lit.any.origin> byref_error, %res: !lit.ref<index, mut #lit.any.origin> byref_result) throws|async {
  kgen.return
}

// CHECK-LABEL: kgen.generator @async_call
lit.fn @async_call[imm a, mut b]() async {
  // CHECK: co.invoke[() async -> (): @async_call]()
  lit.async.call[!lit.generator<[2]() async -> ()>: @async_call][imm a, mut b]()
  // CHECK: co.invoke[(!kgen.pointer<index> byref_error, !kgen.pointer<index> byref_result) throws|async -> (): @async_fn_throws]()
  lit.async.call[!lit.generator<("err": !lit.ref<index, mut #lit.any.origin> byref_error, "res": !lit.ref<index, mut #lit.any.origin> byref_result) throws|async -> ()>: @async_fn_throws]()
  kgen.return
}

// CHECK-LABEL: kgen.generator @trivial_generator
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32
// CHECK-NEXT:    kgen.return %[[ARG0]] : si32
// CHECK-NEXT: }
lit.fn @trivial_generator(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

// CHECK-LABEL: kgen.generator @varDecl
// CHECK-SAME:  (%[[ARG0:.*]]: index) -> index
// CHECK-NEXT:    %[[VAR_A:.*]] = pop.stack_allocation 1 x index
// CHECK-NEXT:    kgen.return %[[ARG0]] : index
// CHECK-NEXT:  }

lit.fn @varDecl(%arg0: index) -> index {
  %a = lit.var.decl "a" var : !lit.ref<index, mut *"life">
  kgen.return %arg0 : index
}

// CHECK-LABEL: kgen.generator @varDecl2
// CHECK-SAME:  (%[[ARG0:.*]]: index)
// CHECK-NEXT: %0 = pop.stack_allocation 1 x index
// CHECK-NEXT: kgen.return
lit.fn @varDecl2(%arg0: index) {
  %a = lit.var.decl "a" var : !lit.ref<index, mut alife>
  kgen.return
}

lit.fn @decorator() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @decorated_fn
lit.fn @decorated_fn()
  // CHECK-NEXT: decorators <:() -> () @decorator>
  decorators<:!lit.generator<() -> ()> @decorator> {
  kgen.return
}

// CHECK-LABEL: kgen.generator @bind_params_discharged_mask_cleared
// CHECK: kgen.param.declare bound = <bind_params(
// CHECK-NOT: | "
// CHECK: kgen.return
kgen.generator @bind_params_discharged_mask_cleared() {
  kgen.param.declare bound: index =
    <#kgen.bind_params<:!lit.generator<<index, {<true, loc("lower-lit":1:1)>}>index> ?, 1 | "1">>
  kgen.return
}

!FnWithOrigin = !lit.generator<<origin<false>>() -> ()>

// CHECK-LABEL: kgen.generator @bind_params_origin_op
lit.fn @bind_params_origin_op(%fn: !FnWithOrigin) {
  // CHECK-NOT: lit.bind_params
  // CHECK: kgen.return
  %bound = lit.bind_params %fn : !FnWithOrigin, :origin<false> #lit.any.origin to !lit.generator<() -> ()>
  kgen.return
}

!FnWithOriginAndConstraints = !lit.generator<<origin<false>, {<true, loc("lower-lit":2:1)>}>() -> ()>

// CHECK-LABEL: kgen.generator @bind_params_origin_discharged
lit.fn @bind_params_origin_discharged(%fn: !FnWithOriginAndConstraints) {
  // CHECK-NOT: lit.bind_params
  // CHECK-NOT: | "
  // CHECK: kgen.return
  %bound = lit.bind_params %fn : !FnWithOriginAndConstraints, :origin<false> #lit.any.origin | "1" to !lit.generator<() -> ()>
  kgen.return
}

// CHECK-LABEL: @generic_types_retain_convention
lit.fn @generic_types_retain_convention<T: type>[imm a](
  // CHECK: %arg0: !kgen.param<T>,
  // CHECK: %arg1: !kgen.pointer<T> mut,
  // CHECK: %arg2: !kgen.param<T> owned,
  // CHECK: %arg3: index,
  // CHECK: %arg4: !kgen.pointer<index> owned
  %p: !kgen.param<T>,
  %q: !lit.ref<T, imm a> mut,
  %r: !kgen.param<T> owned,
  %s1: index,
  %s2: !kgen.pointer<index> owned
){
  kgen.return
}

lit.fn @generic_callee<T: type>(%p: !kgen.param<T>){
  kgen.return
}

// CHECK-LABEL: @call_generic
lit.fn @call_generic(%p: index) {
  // CHECK: kgen.call @generic_callee<:type index>({{.*}}) : (index) -> ()
  kgen.call @generic_callee<:type index>(%p) : !lit.generator<("p": index) -> ()>
  kgen.return
}


//===----------------------------------------------------------------------===//
// Nested Functions
//===----------------------------------------------------------------------===//

lit.struct.decl @StructWithNestedFn<a_param> {
  // CHECK-LABEL: kgen.generator @"StructWithNestedFn::topLevelFunction"<a_param, b_param>() -> index
  lit.fn @topLevelFunction<b_param>() -> index {
    // CHECK: kgen.param.declare.region nestedFunction = () -> index
    lit.fn nestedFunction() -> index {
      kgen.unreachable
    }
    // CHECK: kgen.param.declare b: () -> index = <nestedFunction>
    kgen.param.declare b: !lit.generator<() -> index> = <nestedFunction>

    // CHECK: kgen.param.declare.region paramNestedFunc = <c_param>()
    lit.fn paramNestedFunc<c_param>() {
      kgen.return
    }
    // CHECK: kgen.param.declare c: () -> () = <bind_params(:<index>() -> () paramNestedFunc, 2)>
    kgen.param.declare c: !lit.generator<() -> ()> = <bind_params(:!lit.generator<<"c_param": index>() -> ()> paramNestedFunc, 2)>

    %idx0_0 = index.constant 0
    kgen.return %idx0_0 : index
  }
}

// CHECK-LABEL: kgen.struct.generator @StructWithNestedFn<a_param>

// CHECK-LABEL: kgen.generator @topFunc
lit.fn @topFunc() {
  // CHECK: kgen.param.declare.region midFunc
  lit.fn midFunc() {
    // CHECK: kgen.param.declare.region botFunc
    lit.fn botFunc() {
      kgen.return
    }
    // CHECK: declare bot: () -> () = <botFunc>
    kgen.param.declare bot: !lit.generator<() -> ()> = <botFunc>
    kgen.return
  }
  // CHECK: declare mid: () -> () = <midFunc>
  kgen.param.declare mid: !lit.generator<() -> ()> = <midFunc>
  kgen.return
}

//===----------------------------------------------------------------------===//
// Imports
//===----------------------------------------------------------------------===//

// -----

// CHECK-NOT: lit.unresolved_import
lit.file_module @nested_imports {
  lit.unresolved_import <0, ["foobar"]> as @foo

  lit.fn @func() {
    lit.unresolved_import <0, ["foobar"]> as @foo
    kgen.return
  }
}

//===----------------------------------------------------------------------===//
// Structs
//===----------------------------------------------------------------------===//

// -----

lit.struct.decl @Adder<size> {
  // CHECK-LABEL: kgen.generator @"Adder::__add__"<size>(%arg0: !kgen.struct<() memoryOnly>)
  // CHECK-NEXT:    %[[ONE:.*]] = pop.stack_allocation 1 x index
  // CHECK:       }
  lit.fn @__add__(%self: !lit.struct<@Adder<size>>)  {
    %0 = lit.var.decl "a" var : !lit.ref<index, mut *"life">
    %one = index.constant 1
    lit.ref.store %one, %0 : !lit.ref<index, mut *"life">
    kgen.return
  }
}

// -----

// CHECK-LABEL: kgen.generator @"A::foo"

// CHECK-LABEL: kgen.struct.generator @A
lit.struct.decl @A {
  lit.fn @foo(%self: !lit.struct<@A>) {
    kgen.return
  }
}

// CHECK-LABEL: kgen.generator @"B::foo"
// CHECK-NEXT: call_param[(!kgen.struct<() memoryOnly>) -> (): @"A::foo"]

// CHECK-LABEL: kgen.struct.generator @B
lit.struct.decl @B {
  lit.fn @foo(%self: !lit.struct<@B>, %a: !lit.struct<@A>) {
    kgen.call_param[!lit.generator<("self": !lit.struct<@A>) -> ()>: @A::@foo](%a)
    kgen.return
  }
}

// CHECK-LABEL: kgen.generator @main
lit.fn @main(%a: !lit.struct<@A>, %b: !lit.struct<@B>) {
  // CHECK-NEXT: call_param[(!kgen.struct<() memoryOnly>, !kgen.struct<() memoryOnly>) -> (): @"B::foo"]
  kgen.call_param[!lit.generator<("self": !lit.struct<@B>, "a": !lit.struct<@A>) -> ()>: @B::@foo](%b, %a)
  // CHECK-NEXT: constant: (!kgen.struct<() memoryOnly>) -> () = <@"A::foo">
  %0 = kgen.param.constant: !lit.generator<("self": !lit.struct<@A>) -> ()> = <@A::@foo>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @"A::foo"<N, M>

// CHECK-LABEL: kgen.struct.generator @A<N>
lit.struct.decl @A<N> {
  lit.fn @foo<M>(%self: !lit.struct<@A<N>>) -> index {
    %0 = kgen.param.constant = <add(N, M)>
    kgen.return %0 : index
  }
}

// CHECK-LABEL: kgen.generator @main
lit.fn @main(%a: !lit.struct<@A<1>>) {
  // CHECK-NEXT: call_param[(!kgen.struct<() memoryOnly>) -> index: @"A::foo"<1, 2>]
  %0 = kgen.call_param[!lit.generator<("self": !lit.struct<@A<1>>) -> index>: @A::@foo<1, 2>](%a)
  kgen.return
}

// -----

lit.struct.decl @A {
}

// CHECK: kgen.generator @rhslitdeclref_no_params(%arg0: !kgen.struct<() memoryOnly>)
lit.fn @rhslitdeclref_no_params(%x: !lit.struct<@A>) {
  kgen.return
}

// -----

// Check removing metadata and singleton types from generator types.

lit.struct.decl @EmptyStruct {}

lit.fn @empty_fn<t: !lit.struct<@EmptyStruct>>(%arg0: index) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @removeGenMetadata
lit.fn @removeGenMetadata() {
  // CHECK-NEXT: <index, index>index = <#kgen.gen<to_builtin(:scalar<index> add(from_builtin(*(0,0)), from_builtin(*(0,1))))>>
  kgen.param.declare test: !lit.generator<<"a": index, "b": index> index> = <#kgen.gen<add(*(0,0), *(0,1))>>
  // CHECK-NEXT: struct<() memoryOnly> = <{  }>
  kgen.param.declare test2: !lit.generator<<"t": !lit.struct<@EmptyStruct>> !lit.struct<@EmptyStruct>> = <#kgen.gen<*(0,0)>>
  // CHECK-NEXT: (index) -> () = <@empty_fn>
  kgen.param.declare test3: !lit.generator<<"t": !lit.struct<@EmptyStruct>> ("arg0":index) -> ()> = <@empty_fn>
  // CHECK-NEXT: <index>struct<(struct<() memoryOnly>, index)> = <#kgen.gen<{ { }, *(0,0) }>>
  kgen.param.declare test4: !lit.generator<<"x": !lit.struct<@EmptyStruct>, "y": index = 5> !kgen.struct<(!lit.struct<@EmptyStruct>, index)>> = <#kgen.gen<#kgen.struct<*(0,0), *(0,1)>>>
  kgen.return
}

// -----

lit.struct.decl @A<b, c> {
}

// CHECK: kgen.generator @rhslitdeclref_params(%arg0: !kgen.struct<() memoryOnly>)
lit.fn @rhslitdeclref_params(%x: !lit.struct<@A<10, 11>>) {
  kgen.return
}

// -----

lit.struct.decl @A {
  lit.fn @B() {
    kgen.return
  }
}

// CHECK-LABEL: @callIt
lit.fn @callIt() {
  // CHECK-NEXT: kgen.call @"A::B"
  lit.call @A::@B() : !lit.generator<() -> ()>
  kgen.return
}

// -----

// CHECK-NOT: lit.alias.decl
lit.alias.decl A = <1>
lit.struct.decl @foo {
  // CHECK-NOT: lit.alias.decl
  lit.alias.decl B = <2>
 // CHECK-LABEL:  @"foo::f"() -> index
  lit.fn @f() -> index {
    // CHECK-NOT: kgen.param.declare
    lit.alias.decl C = <3>
    %0 = kgen.param.constant: index = <1>
    kgen.return %0 : index
  }
}

// -----

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

// CHECK: #type_value = #kgen.type<typevalue<#kgen.trait_ref<[@RetZero]>>, type> : !kgen.type
lit.trait.decl @RetZero {
  // CHECK-LABEL: kgen.generator @"RetZero::return_zero"
  lit.fn @return_zero() -> index {
    %idx0_0 = index.constant 0
    kgen.return %idx0_0 : index
  }
}

lit.fn @A<t: !lit.meta<!lit.trait<@RetZero>> = !lit.trait<@RetZero>>() -> index {
    %0 = kgen.param.constant = <0>
    kgen.return %0 : index
}

lit.fn @t() -> index {
  // CHECK: kgen.call @A<:type #type_value>() : () -> index
  %0 = lit.call @A<:!lit.meta<!lit.trait<@RetZero>> !lit.trait<@RetZero>>() : !lit.generator<() -> index>
  kgen.return %0 : index
}

// -----

lit.trait.decl @NestedParams<A> {
  // CHECK-LABEL: kgen.generator @"NestedParams::nested_params"<A, B>
  lit.fn @nested_params<B>() -> index {
    %idx0_0 = index.constant 0
    kgen.return %idx0_0 : index
  }
}

// -----

//===----------------------------------------------------------------------===//
// Error
//===----------------------------------------------------------------------===//
lit.struct.decl @Error {}

lit.fn @throwing_func(%1: !lit.struct<@Error>) throws -> !kgen.variant<@Error, none> {
  %2 = kgen.variant.create %1, 0 : <@Error, none>
  // CHECK: kgen.return %0 : !kgen.variant<struct<() memoryOnly>, none>
  lit.error_return %2 : !kgen.variant<@Error, none>
}

// CHECK-LABEL: kgen.generator @return_raise_or
// CHECK-SAME: -> !kgen.variant<struct<() memoryOnly>, none>
lit.fn @return_raise_or(%cond: !kgen.scalar<bool>, %err: !lit.struct<@Error>) -> !kgen.variant<@Error, none> {
  // CHECK-NEXT: hlcf.if %arg0
  hlcf.elif %cond {
    // CHECK: %[[ERR:.*]] = kgen.variant.create %arg1
    %0 = kgen.variant.create %err, 0 : <@Error, none>
    // CHECK-NEXT: kgen.return %[[ERR]]
    kgen.return %0 : !kgen.variant<@Error, none>
  } else {
    hlcf.yield
  }

  %0 = kgen.param.constant: none = <#kgen.none>
  // CHECK: %[[VAL:.*]] = kgen.variant.create %{{.*}}
  %1 = kgen.variant.create %0, 1 : <@Error, none>
  // CHECK-NEXT: kgen.return %[[VAL]]
  kgen.return %1 : !kgen.variant<@Error, none>
}

// CHECK-LABEL: kgen.generator @removeMetadata
// CHECK-SAME: (%arg0:  !kgen.pointer<index> mut) throws ->
lit.fn @removeMetadata[imm a](%arg0: !lit.ref<index, imm a> mut) throws -> !kgen.variant<@Error, index> {
  %0 = index.constant 0
  %1 = kgen.variant.create %0, 1 : <@Error, index>
  kgen.return %1 : !kgen.variant<@Error, index>
}

// -----

//===----------------------------------------------------------------------===//
// Modules
//===----------------------------------------------------------------------===//

// CHECK-NOT: lit.file_module

lit.file_module @module {
  lit.fn @test()  {
    kgen.return
  }

  lit.struct.decl @Adder<size> {
    // CHECK-LABEL: kgen.generator @"module::Adder::__add__"<size>(%arg0: !kgen.struct<() memoryOnly>)
    // CHECK-NEXT:    kgen.call @"module::test"() : () -> ()
    lit.fn @__add__(%self: !lit.struct<@module::@Adder<size>>)  {
      lit.call @module::@test() : !lit.generator<() -> ()>
      kgen.return
    }
  }

  // CHECK-LABEL: kgen.struct.generator @"module::Adder"<size>
  // CHECK: kgen.generator @"module::test"()
}

// CHECK-LABEL: kgen.generator @caller(%arg0: !kgen.struct<() memoryOnly>)
lit.fn @caller(%ref: !lit.struct<@module::@Adder<10>>)  {
  // CHECK: kgen.call @"module::Adder::__add__"
  kgen.call @module::@Adder::@__add__<10>(%ref) : !lit.generator<("self": !lit.struct<@module::@Adder<10>>) -> ()>
  kgen.return
}

// -----

// CHECK-NOT: lit.package
lit.package @package {
  // CHECK-NOT: lit.file_module
  lit.file_module @module {
    // CHECK: kgen.generator export @"package::module::foo"()
    lit.fn export @foo() {
      kgen.return
    }
  }
}

// -----

lit.file_module @module {
  // CHECK-NOT: lit.alias.decl
  lit.alias.decl A = <42>
}

// CHECK: kgen.generator @metadata
// CHECK-SAME{LITERAL}: LLVMArgMetadataArray = [[], ["llvm.someattr", 2 : index]]
// CHECK-SAME: LLVMMetadataArray = ["llvm.someattr",  3 : index]
lit.fn @metadata(%a: i32, %b: i32) attributes {
  LLVMArgMetadataArray = [[], ["llvm.someattr", 2 : index]],
  LLVMMetadataArray = ["llvm.someattr", 3 : index]
} {
  // CHECK: kgen.param.declare.region metadataNested
  lit.fn metadataNested(%c: i32, %d: i32) attributes {
    LLVMArgMetadataArray = [[], ["llvm.someattr", 4 : index]],
    LLVMMetadataArray = ["llvm.someattr",  5 : index]
  } {
    // CHECK-NEXT: kgen.return
    kgen.return
  // CHECK-NEXT{LITERAL}: LLVMArgMetadataArray = [[], ["llvm.someattr", 4 : index]]
  // CHECK-SAME: LLVMMetadataArray = ["llvm.someattr", 5 : index]
  }
  kgen.return
}

// -----

// COM: Ensure the linkage name is passed through on an exported function.

// CHECK: kgen.generator export @"main::main::main"
// CHECK-SAME: linkageName = #kgen.linkage_name<"main" : !kgen.string, false>
lit.package @main {
  lit.file_module @main {
    lit.fn export @main() attributes {linkageName = #kgen.linkage_name<"main" : !kgen.string, false>} {
      kgen.return
    }
  }
}

// -----

// COM: Ensure that linkageName (static string) is passed through without
// COM: renaming the symbol.

// CHECK: kgen.generator export @"pkg::mod::my_fn"
// CHECK-SAME: linkageName = #kgen.linkage_name<"my_export" : !kgen.string, false>
lit.package @pkg {
  lit.file_module @mod {
    lit.fn export @my_fn() attributes {linkageName = #kgen.linkage_name<"my_export" : !kgen.string, false>} {
      kgen.return
    }
  }
}

// -----

// COM: Ensure that linkageName on a non-export function passes through.

// CHECK: kgen.generator @"pkg3::mod3::orig_name"
// CHECK-SAME: linkageName = #kgen.linkage_name<"my_link_name" : !kgen.string, false>
lit.package @pkg3 {
  lit.file_module @mod3 {
    lit.fn @orig_name() attributes {linkageName = #kgen.linkage_name<"my_link_name" : !kgen.string, false>} {
      kgen.return
    }
  }
}


//===----------------------------------------------------------------------===//
// Implicit lifetimes.
//===----------------------------------------------------------------------===//

// -----

// Verify that the lifetimes get correctly removed and the IR is correct.

!Mem = !lit.struct<@Mem>
lit.struct.decl @Mem   {
  lit.fn @__init__[mut a](%self: !lit.ref<!Mem, mut a> byref_result, |) -> !kgen.none {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}

// CHECK-LABEL: kgen.generator @getThing
// CHECK-SAME:(%arg0: !kgen.pointer<struct<() memoryOnly>> byref_result)
lit.fn @getThing[mut abc](%res: !lit.ref<!Mem, mut abc> byref_result, |) -> !kgen.none {
  // CHECK-NEXT: kgen.param.declare.region localTest = (%arg1: !kgen.pointer<struct<() memoryOnly>> byref_result) capturing
  lit.fn localTest[mut lt](%__result__[__result__]: !lit.ref<!Mem, mut lt> byref_result, |) capturing -> !kgen.none {
    // CHECK-NEXT: call @"Mem::__init__"(%arg1)
    %1 = lit.call @Mem::@__init__[mut lt](%__result__) : !lit.generator<[1]("self": !lit.ref<!Mem, mut *[0,0]> byref_result, |) -> !kgen.none>
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
  // CHECK: }
  // CHECK-NEXT: kgen.call_param[(!kgen.pointer<struct<() memoryOnly>> byref_result) capturing -> !kgen.none: localTest](%arg0)
  %0 = lit.call [!lit.generator<[1]("__result__": !lit.ref<!Mem, mut *[0,0]> byref_result, |) capturing -> !kgen.none>: localTest][mut abc](%res)
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}


// CHECK-LABEL: kgen.generator @callThing
// CHECK-SAME: (%arg0: !kgen.pointer<struct<() memoryOnly>> byref_result)
// CHECK-SAME: sourceName = "callThing"
lit.fn @callThing[mut lt](%__result__: !lit.ref<!Mem, mut lt> byref_result, |) -> !kgen.none attributes {isParametric, sourceName = "callThing", specialFnKind = 0 : i8} {
  // CHECK-NEXT: kgen.call @getThing(%arg0)
  %0 = lit.call @getThing[mut lt](%__result__) : !lit.generator<[1]("res": !lit.ref<!Mem, mut *[0,0]> byref_result, |) -> !kgen.none>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator @testLifetimeOf2
// Verify that we remap the returns as well as the operands.
lit.fn @testLifetimeOf2[imm *"a`"](%a: !lit.ref<!Mem, imm *"a`"> imm_mem) -> !lit.ref<!Mem, imm *"a`">{
  // CHECK-NEXT: kgen.return %arg0
  kgen.return %a : !lit.ref<!Mem, imm *"a`">
}

// CHECK-LABEL: kgen.generator @callLifetimes
// CHECK-SAME: (%arg0: !kgen.pointer<index>) -> !kgen.pointer<index>
lit.fn @callLifetimes[mut lt](%arg0[*""]: !lit.ref<index, mut lt>) -> !lit.ref<index, mut lt> {
  // CHECK: kgen.call @callLifetimes(%arg0) : (!kgen.pointer<index>) -> !kgen.pointer<index>
  %0 = lit.call @callLifetimes[mut lt](%arg0) : !lit.generator<[1](!lit.ref<index, mut *[0,0]>) -> !lit.ref<index, mut *[0,0]>>
  kgen.return %0 : !lit.ref<index, mut lt>
}


// This should drop the explicit origin parameters since they are singletons.

// CHECK-LABEL: kgen.generator @takes_life_explicit<ismut: scalar<bool>, size, val: simd<size, f32>>
// CHECK-SAME: (%arg0: !kgen.pointer<struct<() memoryOnly>> byref_result)
lit.fn @takes_life_explicit<ismut: !kgen.scalar<bool>, life: !lit.origin<ismut>, size: index, val: !kgen.simd<size, f32>>
                    (%ref: !lit.ref<!Mem, mut=ismut, life> byref_result, |) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @call_takes_life_explicit
// CHECK-SAME: <val: simd<4, f32>>(%arg0: !kgen.pointer<struct<() memoryOnly>> byref_result)
lit.fn @call_takes_life_explicit<val: !kgen.simd<4, f32>>[mut lt](%__result__: !lit.ref<!Mem, mut lt> byref_result, |) {
  // CHECK-NEXT: kgen.call @takes_life_explicit<:scalar<bool> true, 4, :simd<4, f32> val>(%arg0)
  // CHECK-SAME: : (!kgen.pointer<struct<() memoryOnly>> byref_result) -> ()
  lit.call @takes_life_explicit<:!kgen.scalar<bool> true, :!lit.origin<true> lt, :index 4, :!kgen.simd<4, f32> val>(%__result__)
      : !lit.generator<("ref": !lit.ref<!Mem, mut lt> byref_result, |) -> ()>
  kgen.return
}

// -----

!Int = !lit.struct<@Int>
#IndexList = #lit<symbol@IndexList>
lit.struct.decl @Int {
  lit.struct.field value : index
}

lit.struct.decl @IndexList<life: !lit.origin<true>, size: !Int> {
  lit.fn @getitem(%self[*""]: !lit.struct<@IndexList<:!lit.origin<true> life, :!Int size>>) -> !Int {
    kgen.unreachable
  }
}

// COM: Origin parameters are singleton values and must be dropped from
// COM: TypeGeneratorRefAttr bindings during LowerLIT.
// CHECK-LABEL: kgen.generator @"IndexList::getitem"<size: struct<(index) memoryOnly>>(
// CHECK-LABEL: kgen.generator @paramReplacement<_1: struct<(index) memoryOnly>, callee: (!kgen.struct<() memoryOnly>) -> ()>()
lit.fn @paramReplacement<
    _1: !Int,
    _2: @IndexList<:!lit.origin<true> lt, :!Int _1>,
    callee: !lit.generator<[1](!lit.struct<#IndexList <:!lit.origin<true> lt, :!Int apply(:!lit.generator<(!lit.struct<#IndexList <:!lit.origin<true> lt, :!Int _1>>) -> !Int> @IndexList::@getitem<:!lit.origin<true> lt, :!Int _1>, _2)>>) -> ()>>[mut lt]() {
  kgen.unreachable
}

// -----

//===----------------------------------------------------------------------===//
// Ownership
//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.generator @ownership_ops
lit.fn @ownership_ops[mut lt](%a: !lit.ref<index, mut lt>) {
  // CHECK-NOT: lit.ownership.
  lit.ownership.mark_initialized %a : !lit.ref<index, mut lt>
  lit.ownership.use %a : !lit.ref<index, mut lt>
  lit.ownership.mark_destroyed %a : !lit.ref<index, mut lt>
  kgen.return
}

//===----------------------------------------------------------------------===//
// Singleton Struct Types.
//===----------------------------------------------------------------------===//

lit.struct.decl @EmptyStruct {}

// CHECK-LABEL: kgen.generator @expect_always_empty_struct()
lit.fn @expect_always_empty_struct<es: !lit.struct<@EmptyStruct>>() {
  kgen.return
}

lit.fn @expect_parametric_empty_struct<t: type, s: !kgen.param<t>>() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @call_using_empty_struct<alwaysFn: <index>() -> (), paramFn: <type, *(0,0)>() -> ()>()
lit.fn @call_using_empty_struct<es: !lit.struct<@EmptyStruct>, alwaysFn: !lit.generator<<index, @EmptyStruct>() -> ()>, paramFn: !lit.generator<<type, !kgen.param<*(0,0)>>() -> ()>>() {
  // CHECK-NEXT: kgen.call @expect_always_empty_struct() : () -> ()
  lit.call @expect_always_empty_struct<:!lit.struct<@EmptyStruct> es>() : !lit.generator<() -> ()>
  // CHECK-NEXT: kgen.call @expect_parametric_empty_struct<:type {{.*}}, :struct<() memoryOnly> {  }>()
  lit.call @expect_parametric_empty_struct<:type #kgen.type<!lit.struct<@EmptyStruct>>, :!lit.struct<@EmptyStruct> es>() : !lit.generator<() -> ()>
  // CHECK-NEXT: <bind_params(:<index>() -> () alwaysFn, 1)>
  kgen.param.declare alwaysFn2: !lit.generator<() -> ()> = <bind_params(:!lit.generator<<index, @EmptyStruct>() -> ()> alwaysFn, :index 1, :!lit.struct<@EmptyStruct> es)>
  // CHECK-NEXT: <bind_params(:<type, *(0,0)>() -> () paramFn, {{.*}}, :struct<() memoryOnly> {  })>
  kgen.param.declare paramFn2: !lit.generator<() -> ()> = <bind_params(:!lit.generator<<type, !kgen.param<*(0,0)>>() -> ()> paramFn, #kgen.type<!lit.struct<@EmptyStruct>>, :!lit.struct<@EmptyStruct> es)>
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//
// Struct alignment lowering
//===----------------------------------------------------------------------===//

!MyInt = !lit.struct<@struct_alignment::@MyInt>
lit.file_module @struct_alignment {
  lit.struct.decl @MyInt {
    lit.struct.field value : index
  }

  lit.fn @calculate_alignment(%my_int: !MyInt) -> index {
    %0 = kgen.param.constant: index = <64>
    kgen.return %0 : index
  }

  // Test concrete alignment on struct propagates to stack_allocation and does not
  // result in a flattened struct.
  lit.struct.decl @AlignedStruct64<my_int: !MyInt> attributes {
      minAlignment = #kgen.param.expr<apply,
                                      #kgen.symbol.constant<@struct_alignment::@calculate_alignment> : !lit.generator<("my_int": !MyInt) -> index>,
                                      #kgen.param.decl.ref<"my_int">: !MyInt> : index
  } {
    lit.struct.field value : index
  }

  // CHECK-LABEL: kgen.generator @"struct_alignment::varDeclAligned64"
  lit.fn @varDeclAligned64() {
    // CHECK-NEXT: pop.stack_allocation 1 x struct<(index) memoryOnly align(apply(:(!kgen.struct<(index) memoryOnly>) -> index @"struct_alignment::calculate_alignment", { 32 }))>
    // CHECK-SAME: align apply(:(!kgen.struct<(index) memoryOnly>) -> index @"struct_alignment::calculate_alignment", { 32 }) marked
    %a = lit.var.decl "a" var : !lit.ref<!lit.struct<@struct_alignment::@AlignedStruct64<:!MyInt {value = 32}>>, mut *"life">
    kgen.return
  }
}

// -----

// Test parametric alignment on struct propagates to stack_allocation.
!Pair = !lit.struct<@Pair>
lit.struct.decl @Pair register_passable {
  lit.struct.field first : index
  lit.struct.field second : index
}

lit.struct.decl @AlignedParam<N: !Pair> attributes {minAlignment = #lit.struct.extract<:!Pair N, "first"> : index} {
  lit.struct.field data : index
}

// CHECK-LABEL: kgen.generator @varDeclAlignedParam
lit.fn @varDeclAlignedParam<pair: !Pair>() {
  // CHECK-NEXT: pop.stack_allocation 1 x struct<(index) memoryOnly align(#kgen.struct.extract<:struct<(index, index)> pair, 0>)>
  // CHECK-SAME: align #kgen.struct.extract<:struct<(index, index)> pair, 0> marked
  %a = lit.var.decl "a" var : !lit.ref<!lit.struct<@AlignedParam<:!Pair pair>>, mut *"life">
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//
// Struct Extensions
//===----------------------------------------------------------------------===//

// Test struct then extension (normal order)
lit.struct.decl @TestStruct1 {
}

// CHECK-LABEL: kgen.struct.generator @TestStruct1

lit.extension.decl @"extension:TestStruct1" attributes {targetStruct = @TestStruct1} {
  // CHECK-LABEL: kgen.generator @"extension:TestStruct1::extension_method"
  lit.fn @extension_method[mut O](%self: !lit.ref<!lit.struct<@TestStruct1>, mut O> imm_mem) -> index {
    %result = kgen.param.constant: index = <42>
    kgen.return %result : index
  }
}

// -----

// Test extension declared before its target struct
// CHECK-LABEL: kgen.generator @"extension:TestStruct2::extension_method"

// CHECK-LABEL: kgen.struct.generator @TestStruct2
lit.extension.decl @"extension:TestStruct2" attributes {targetStruct = @TestStruct2} {
  lit.fn @extension_method[mut O](%self: !lit.ref<!lit.struct<@TestStruct2>, mut O> imm_mem) -> index {
    %result = kgen.param.constant: index = <1>
    kgen.return %result : index
  }
}

lit.struct.decl @TestStruct2 {
}

// TODO(MOCO-522): Add tests for aliases in extensions into LowerLIT
