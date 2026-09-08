// RUN: kgen-opt %s -split-input-file -verify-parameters -elaborate-generators="use-parametric-interpret=false" -allow-unregistered-dialect | FileCheck --check-prefixes=CHECK,CHECK-MAIN %s
// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=true" -allow-unregistered-dialect | FileCheck --check-prefixes=CHECK,CHECK-PARAMINTERP %s

// CHECK-LABEL: kgen.func @parameter_use_chain()
kgen.generator @parameter_use_chain() {
  // Uses r2 and defines r1
  kgen.param.declare r1 = <add(r2, 1)>
  // CHECK-NEXT: %{{.*}} = kgen.param.constant = <3>
  %0 = kgen.param.constant = <r1>

  // Uses 42 and defines r2
  kgen.param.declare r2 = <2>
  // CHECK-NEXT: %{{.*}} = kgen.param.constant = <2>
  %1 = kgen.param.constant = <r2>

  // Uses r1/r2 and defines r3
  kgen.param.declare r3 = <mul(shl(r1, r2), 3)>
  // CHECK-NEXT: %{{.*}} = kgen.param.constant = <36>
  %2 = kgen.param.constant = <r3>

  // Defines a dtype value and uses it.
  kgen.param.declare type1 : !kgen.dtype = <f32>
  // CHECK-NEXT: %{{.*}} = kgen.param.constant: dtype = <f32>
  %3 = kgen.param.constant: !kgen.dtype = <type1>

  // CHECK-NEXT: kgen.return
  kgen.return
}

// CHECK-LABEL: @"unknown_attr,width=4"
kgen.generator @unknown_attr<width>() {
  // CHECK-NEXT: constant: simd<4, f32> = <*?>
  kgen.param.constant: simd<width, f32> = <*?>
  kgen.return
}

// CHECK-LABEL: @"empty_variadic,T=i32"
kgen.generator @empty_variadic<T: type>() {
  // CHECK-NEXT: constant: param_list<i32> = <[]>
  kgen.param.constant: param_list<T> = <[]>
  kgen.return
}

// CHECK-LABEL: @call_unknown_attr
kgen.generator @call_unknown_attr() {
  kgen.call @unknown_attr<4>() : () -> ()
  kgen.call @empty_variadic<:type i32>() : () -> ()
  kgen.return
}

// CHECK-NOT: kgen.generator @trivial_generator
// This gets "specialized" into a kernel.
kgen.generator @trivial_generator(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}
// CHECK-LABEL: kgen.func @trivial_generator
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32 {
// CHECK-NEXT:    kgen.return %[[ARG0]] : si32
// CHECK-NEXT: }

kgen.generator @genA<size, DT: dtype, val: f32>(%arg0: si32) -> si32 {

  %0 = kgen.param.constant = <add(size, 4)>
  %1 = kgen.param.constant: dtype = <DT>
  %2 = kgen.param.constant: f32 = <val>

  // Silly op so we know when something used this.
  "genA.op"() { value = #kgen.param.decl.ref<"size"> : index} : () -> !kgen.scalar<DT>

  kgen.return %arg0 : si32
}

// CHECK-LABEL: kgen.func @"genA,size=19,DT=si8,val=1.50{{.*}}"
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32 {
// CHECK-NEXT:    %[[V0:.*]] = kgen.param.constant  = <23>
// CHECK-NEXT:    %[[V1:.*]] = kgen.param.constant: dtype = <si8>
// CHECK-NEXT:    %[[V2:.*]] = kgen.param.constant: f32 = <1.500000e+00>
// CHECK-NEXT:    %[[V3:.*]] = "genA.op"() {value = 19 : index} : () -> !kgen.scalar<si8>
// CHECK-NEXT:    kgen.return %[[ARG0]] : si32
// CHECK-NEXT:  }

// CHECK-LABEL: kgen.func @"genA,size=42,DT=f32,val=2.00{{.*}}"
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32 {
// CHECK-NEXT:   %[[V0:.*]] = kgen.param.constant  = <46>
// CHECK-NEXT:   %[[V1:.*]] = kgen.param.constant: dtype = <f32>
// CHECK-NEXT:   %[[V2:.*]] = kgen.param.constant: f32 = <2.000000e+00>
// CHECK-NEXT:   %[[V3:.*]] = "genA.op"() {value = 42 : index} : () -> !kgen.scalar<f32>
// CHECK-NEXT:   kgen.return %[[ARG0]] : si32
// CHECK-NEXT: }

// CHECK-LABEL: kgen.func @call_generator_test
// CHECK-SAME: %[[ARG0:.*]]: si32, %[[ARG1:.*]]: si32
kgen.generator @call_generator_test(%arg0: si32, %arg1: si32) -> (si32, si32, si32) {
  // Can invoke the generator directly.
  %0 = kgen.call @trivial_generator(%arg0) : (si32) -> si32
  // CHECK-NEXT: %{{.*}} = kgen.call @trivial_generator(%[[ARG0]])

  // CHECK-NOT: kgen.param.declare
  kgen.param.declare our_size = <42>

  // Can invoke parameterized generators directly.
  // CHECK-NEXT: %{{.*}} = kgen.call @"genA,size=42,DT=f32,val=2.00{{.*}}"(%[[ARG0]]) : (si32) -> si32
  %1 = kgen.call @genA<our_size, :dtype f32, :f32 2.0>(%arg0) : (si32) -> si32

  // CHECK-NEXT: %{{.*}} = kgen.call @"genA,size=19,DT=si8,val=1.50{{.*}}"(%[[ARG1]]) : (si32) -> si32
  %2 = kgen.call @genA<19, :dtype si8, :f32 1.5>(%arg1) : (si32) -> si32

  // CHECK-NEXT: %{{.*}} = kgen.call @"genA,size=19,DT=si8,val=1.50{{.*}}"(%[[ARG1]]) : (si32) -> si32
  %3 = kgen.call @genA<19, :dtype si8, :f32 1.5>(%arg1) : (si32) -> si32

  kgen.return %0, %1, %2 : si32, si32, si32
}

// CHECK-LABEL: kgen.func @test_variadic_ptr_map
kgen.generator @test_variadic_ptr_map() {
  // CHECK-NEXT: %param_list = kgen.param.constant: param_list<type> = <[pointer<i32, 42>, pointer<f32, 42>, pointer<i8, 42>]>
  kgen.param.declare types : param_list<!kgen.type> = <[i32, f32, i8]>
  %param_list = kgen.param.constant: param_list<!kgen.type> = <variadic_ptr_map(:param_list<!kgen.type> types, 42)>
  // CHECK-NEXT: kgen.return
  kgen.return
}

// CHECK-LABEL: kgen.func @test_variadic_ptrremove_map
kgen.generator @test_variadic_ptrremove_map() {
  // CHECK-NEXT: %param_list = kgen.param.constant: param_list<type> = <[i32, f32, i8]>
  kgen.param.declare types : param_list<!kgen.type> = <[pointer<i32>, pointer<f32, 42>, pointer<i8>]>
  %param_list = kgen.param.constant: param_list<!kgen.type> = <variadic_ptrremove_map(:param_list<!kgen.type> types)>
  // CHECK-NEXT: kgen.return
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//

// Test that parameter and result argument types get rewritten and specialized.

// CHECK-LABEL: kgen.func @"float_constant_f32,value=1.50{{.*}},DT=f32"() -> !kgen.scalar<f32> {
// ...
// CHECK:    %[[V1:.*]] = llvm.fptrunc
// CHECK:    %[[V2:.*]] = pop.cast_from_builtin %[[V1]] : f32 to !kgen.scalar<f32>
// CHECK:    kgen.return %[[V2]] : !kgen.scalar<f32>

kgen.generator @float_constant_f32<value: f64, DT: dtype>() -> !kgen.scalar<DT> {
  kgen.param.assert <identical(:dtype DT, f32)>, "float please"
  %0 = kgen.param.constant: f64 = <value>
  %1 = llvm.fptrunc %0 : f64 to f32
  %2 = pop.cast_from_builtin %1: f32 to !kgen.scalar<DT>
  kgen.return %2 : !kgen.scalar<DT>
}

// CHECK-LABEL: kgen.func @test_f32() -> f32 {
// CHECK:    %[[V0:.*]] = kgen.call @"float_constant_f32,value=1.50{{.*}},DT=f32"() : () -> !kgen.scalar<f32>
// CHECK:    %[[V1:.*]] = pop.cast_to_builtin %[[V0]] : !kgen.scalar<f32> to f32
kgen.generator @test_f32() -> f32 {
  kgen.param.declare DT : dtype = <f32>
  %1 = kgen.call @float_constant_f32<:f64 1.5, :dtype DT>() : () -> !kgen.scalar<DT>
  %2 = pop.cast_to_builtin %1 : !kgen.scalar<DT> to f32
  kgen.return %2 : f32
}

// -----

//===----------------------------------------------------------------------===//

// Test that we can do static assertions on computed parameter expressions (e.g.
// those that are the result of a sub-generator invocation.


// CHECK-LABEL: kgen.func @paramAssertExample()
// CHECK-NOT: kgen.param.assert
kgen.generator @paramAssertExample() {
  kgen.param.declare flen = <4>

  // Should succeed.
  kgen.param.assert <eq(flen, 4)>, "vector length should be 4 for floats"
  kgen.return
}

// CHECK-LABEL: kgen.func @"parametricAdd,sz=1,dt=ui64"
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.scalar<ui64>, %[[ARG1:.*]]: !kgen.scalar<ui64>) -> !kgen.scalar<ui64> {
// CHECK-NEXT: %[[V0:.*]] = pop.add %[[ARG0]], %[[ARG1]] : !kgen.scalar<ui64>
// CHECK-NEXT: kgen.return %[[V0]] : !kgen.scalar<ui64>

// CHECK-LABEL: kgen.func @"parametricAdd,sz=2,dt=f32"
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.simd<2, f32>, %[[ARG1:.*]]: !kgen.simd<2, f32>) -> !kgen.simd<2, f32> {
// CHECK-NEXT: %[[V0:.*]] = pop.add %[[ARG0]], %[[ARG1]] : !kgen.simd<2, f32>
// CHECK-NEXT: kgen.return %[[V0]] : !kgen.simd<2, f32>

kgen.generator @parametricAdd<sz, dt: dtype>
  (%a: !kgen.simd<sz,dt>, %b: !kgen.simd<sz,dt>) -> !kgen.simd<sz,dt> {
  %res = pop.add %a, %b : !kgen.simd<sz,dt>
  kgen.return %res : !kgen.simd<sz,dt>
}

// CHECK-LABEL: kgen.func @parametricTypes(
kgen.generator @parametricTypes(%arg0: !kgen.scalar<ui64>, %arg1: !kgen.simd<2, f32>) {
  kgen.param.declare dt: dtype = <ui32>
  kgen.param.declare ty1: type = <!kgen.scalar<dt>>

  // CHECK-NEXT:   "impl.0"() : () -> !kgen.scalar<ui32>
  "impl.0"() : () -> !kgen.param<ty1>

  // CHECK-NEXT: = kgen.call @"parametricAdd,sz=1,dt=ui64"
  // CHECK-SAME: (%[[ARG0:.*]], %[[ARG0:.*]]) : (!kgen.scalar<ui64>, !kgen.scalar<ui64>) -> !kgen.scalar<ui64>
  %0 = kgen.call @parametricAdd<1, :dtype ui64>(%arg0, %arg0) : (!kgen.scalar<ui64>, !kgen.scalar<ui64>) -> !kgen.scalar<ui64>

  // CHECK-NEXT: = kgen.call @"parametricAdd,sz=2,dt=f32"(%[[ARG1]], %[[ARG1]]) : (!kgen.simd<2, f32>, !kgen.simd<2, f32>) -> !kgen.simd<2, f32>
  %1 = kgen.call @parametricAdd<2, :dtype f32>(%arg1, %arg1) : (!kgen.simd<2, f32>, !kgen.simd<2, f32>) -> !kgen.simd<2, f32>

  kgen.return
}

// CHECK-LABEL: kgen.func @"takeUnary,dt=f32,fn=nopExample"() {
// CHECK: %simd = kgen.param.constant
// CHECK: %0 = pop.cast %simd
// CHECK: %1 = kgen.call @"nopExample,dt=f32"(%0) : (!kgen.scalar<f32>) -> !kgen.scalar<f32>
// CHECK: %2 = kgen.call @"nopExample,dt=f32"(%1) : (!kgen.scalar<f32>) -> !kgen.scalar<f32>

// CHECK-LABEL: kgen.func @"takeUnary,dt=si32,fn=doubleExample"()
// CHECK: %simd = kgen.param.constant
// CHECK: %0 = pop.cast %simd
// CHECK: %1 = kgen.call @"doubleExample,dt=si32"(%0) : (!kgen.scalar<si32>) -> !kgen.scalar<si32>
// CHECK: %2 = kgen.call @"doubleExample,dt=si32"(%1) : (!kgen.scalar<si32>) -> !kgen.scalar<si32>

kgen.generator @takeUnary
  <dt: dtype, fn: <dtype>(!kgen.scalar<*(0,0)>) -> !kgen.scalar<*(0,0)>>() {

  %one = kgen.param.constant: scalar<si64> = <1>
  %0 = pop.cast %one : !kgen.scalar<si64> to !kgen.scalar<dt>
  %1 = kgen.call_param[(!kgen.scalar<dt>) -> !kgen.scalar<dt>:
    bind_params(:<dtype>(!kgen.scalar<*(0,0)>) -> !kgen.scalar<*(0,0)> fn, :dtype dt)](%0)
  %2 = kgen.call_param[(!kgen.scalar<dt>) -> !kgen.scalar<dt>:
    bind_params(:<dtype>(!kgen.scalar<*(0,0)>) -> !kgen.scalar<*(0,0)> fn, :dtype  dt)](%1)
  kgen.return
}

kgen.generator @doubleExample<dt:dtype>(%arg0: !kgen.scalar<dt>) -> !kgen.scalar<dt> {
  %0 = pop.add %arg0, %arg0: !kgen.scalar<dt>
  kgen.return %0 : !kgen.scalar<dt>
}

kgen.generator @nopExample<dt:dtype>(%arg0: !kgen.scalar<dt>) -> !kgen.scalar<dt> {
  kgen.return %arg0 : !kgen.scalar<dt>
}

kgen.generator @takeParametricBinary
  <sz,
   dt: dtype,
   fn: <index, dtype>(!kgen.simd<*(0,0),*(0,1)>, !kgen.simd<*(0,0),*(0,1)>) -> !kgen.simd<*(0,0),*(0,1)>
  >() {

  %one = kgen.param.constant: scalar<si64> = <1>
  %0 = pop.cast %one : !kgen.scalar<si64> to !kgen.scalar<dt>

  %1 = kgen.call_param[(!kgen.scalar<dt>, !kgen.scalar<dt>) -> !kgen.scalar<dt>:
    bind_params(:<index, dtype>(!kgen.simd<*(0,0),*(0,1)>, !kgen.simd<*(0,0),*(0,1)>) -> !kgen.simd<*(0,0),*(0,1)> fn, 1, :dtype dt)](%0, %0)
  kgen.return
}

// CHECK-LABEL:  kgen.func @test_symbol() {
kgen.generator @test_symbol() {
  // CHECK: kgen.call @"takeUnary,dt=si32,fn=doubleExample"()
  kgen.call @takeUnary<:dtype si32,
     :<dtype>(!kgen.scalar<*(0,0)>) -> !kgen.scalar<*(0,0)> @doubleExample>() : () -> ()

  // CHECK: kgen.call @"takeUnary,dt=f32,fn=nopExample"()
  kgen.call @takeUnary<:dtype f32,
     :<dtype>(!kgen.scalar<*(0,0)>) -> !kgen.scalar<*(0,0)> @nopExample>() : () -> ()

  // CHECK: kgen.call @"takeParametricBinary,sz=2,dt=f32,fn=parametricAdd"()
  kgen.call @takeParametricBinary
     <
      2,
      :dtype f32,
      :<index, dtype>(!kgen.simd<*(0,0), *(0,1)>, !kgen.simd<*(0,0), *(0,1)>) -> !kgen.simd<*(0,0), *(0,1)> @parametricAdd
     >() : () -> ()

  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @"parametricBinOp,ty=scalar<f32>"
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.scalar<f32>, %[[ARG1:.*]]: !kgen.scalar<f32>) -> !kgen.scalar<f32> {
// CHECK-NEXT: %[[V0:.*]] = "custom.op"(%[[ARG0]], %[[ARG1]]) : (!kgen.scalar<f32>, !kgen.scalar<f32>) -> !kgen.scalar<f32>
// CHECK-NEXT: kgen.return %[[V0]] : !kgen.scalar<f32>
kgen.generator @parametricBinOp<ty: type>
  (%a: !kgen.param<ty>, %b: !kgen.param<ty>) -> !kgen.param<ty> {
  %res = "custom.op" (%a, %b) : (!kgen.param<ty>, !kgen.param<ty>) -> !kgen.param<ty>
  kgen.return %res : !kgen.param<ty>
}

// CHECK-LABEL: kgen.func @"takeParametricBinary,dt=f32,fn=parametricBinOp"() {
kgen.generator @takeParametricBinary
  <dt: dtype,
   fn: <type>(!kgen.param<*(0,0)>, !kgen.param<*(0,0)>) -> !kgen.param<*(0,0)>
  >() {

  %one = kgen.param.constant: scalar<si64> = <1>
  %0 = pop.cast %one : !kgen.scalar<si64> to !kgen.scalar<dt>

  // CHECK: kgen.call @"parametricBinOp,ty=scalar<f32>"
  %1 = kgen.call_param[(!kgen.scalar<dt>, !kgen.scalar<dt>) -> !kgen.scalar<dt>:
    bind_params(:<type>(!kgen.param<*(0,0)>, !kgen.param<*(0,0)>) -> !kgen.param<*(0,0)>
      fn, :type !kgen.scalar<dt>)](%0, %0)
  kgen.return
}

// CHECK-LABEL: kgen.func @test_paramref_type_rewrite() {
kgen.generator @test_paramref_type_rewrite() {
  // CHECK: kgen.call @"takeParametricBinary,dt=f32,fn=parametricBinOp"() : () -> ()
  kgen.call @takeParametricBinary<:dtype f32,
      :<type>(!kgen.param<*(0,0)>, !kgen.param<*(0,0)>) -> !kgen.param<*(0,0)> @parametricBinOp>() : () -> ()

  kgen.return
}


kgen.generator @indirect_callee() -> index {
  %0 = kgen.param.constant = <42>
  kgen.return %0: index
}

kgen.generator @call_indirect(%fp: !kgen.generator<() -> index>) -> index {
  %0 = kgen.call_indirect %fp() : () -> index
  kgen.return %0: index
}

// CHECK-LABEL: kgen.func @test_comptime_call_indirect
kgen.generator @test_comptime_call_indirect() -> index {
  // CHECK-NEXT: %index42 = kgen.param.constant = <42>
  %0 = kgen.param.constant = <apply(:(!kgen.generator<() -> index>) -> index @call_indirect, @indirect_callee)>
  kgen.return %0: index
}

// -----

// This takes a parameter function that uses a contextual type instead of
// to-be-bound types.
// CHECK-LABEL: kgen.func @"takeFnContextualType,ty=index,fn=sillyFn"() -> index {
// CHECK:  %0 = kgen.call @sillyFn() : () -> index
kgen.generator @takeFnContextualType<ty: type, fn: ()->!kgen.param<ty>>() -> !kgen.param<ty> {
  %0 = kgen.call_param[()->!kgen.param<ty>: fn]()
  kgen.return %0: !kgen.param<ty>
}

kgen.generator @sillyFn() -> index {
  %0 = kgen.param.constant = <42>
  kgen.return %0: index
}

// CHECK-LABEL:  kgen.func @elaborateFnWithContextualType() -> index {
// CHECK:   %0 = kgen.call @"takeFnContextualType,ty=index,fn=sillyFn"() : () -> index
kgen.generator @elaborateFnWithContextualType() -> index {
  %0 = kgen.call @takeFnContextualType<:type index, :()->index @sillyFn>() : () -> index
  kgen.return %0 : index
}

// CHECK-LABEL: @elaborateFnWithContextualType2()
kgen.generator @elaborateFnWithContextualType2() -> (index, index) {
  // Show we can bind a generic signature to a concrete one.
  kgen.param.declare boundFn: ()->index =
    <bind_params(:<type, ()->!kgen.param<*(1,0)>>() -> !kgen.param<*(0,0)> @takeFnContextualType,
                    :type index, :()->index @sillyFn)>

  // CHECK-NEXT: %0 = kgen.call @"takeFnContextualType,ty=index,fn=sillyFn"()
  %0 = kgen.call_param[()->index: boundFn]()

  kgen.param.declare fn: <type, ()->!kgen.param<*(1,0)>>() -> !kgen.param<*(0,0)> = <@takeFnContextualType>

  kgen.param.declare boundFn2: ()->index =
    <bind_params(:<type, ()->!kgen.param<*(1,0)>>() -> !kgen.param<*(0,0)> fn,
                    :type index, :()->index @sillyFn)>

  // CHECK-NEXT: %1 = kgen.call @"takeFnContextualType,ty=index,fn=sillyFn"()
  %1 = kgen.call_param[()->index: boundFn2]()

  kgen.return %0, %1 : index, index
}

// -----

// CHECK-LABEL: kgen.func @"takeStringParameter,SomeString=~Qfoo~Q"
kgen.generator @takeStringParameter<SomeString: string>() {
  kgen.param.assert <identical(:string SomeString, "foo")>, "I want foo"
  kgen.return
}

// CHECK-LABEL: kgen.func @giveString
kgen.generator @giveString() {
  // CHECK-NEXT: kgen.call @"takeStringParameter,SomeString=~Qfoo~Q"
  kgen.call @takeStringParameter<:string "foo">() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: @"makeListConst,A=1"
kgen.generator @makeListConst<A>() {
  // CHECK-NEXT: kgen.param.constant: array<2, index> = <[1, 1]>
  %0 = kgen.param.constant: array<2, index> = <[A, A]>
  kgen.return
}

kgen.generator @doIt() {
  kgen.call @makeListConst<1>() : () -> ()
  kgen.return
}

// CHECK-LABEL: @"variableList,N=2,Ts=[1, 2]"
kgen.generator @variableList<N, Ts: array<N, i32>>() {
  // CHECK-NEXT: kgen.param.constant: array<2, i32> = <[1, 2]>
  %0 = kgen.param.constant: array<N, i32> = <Ts>
  kgen.return
}

kgen.generator @passTypeList() {
  kgen.call @variableList<2, :array<2, i32> [1, 2]>() : () -> ()
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//
// Recursion Test
//===----------------------------------------------------------------------===//
//
// This shows that we properly support recursive expansion.
//

kgen.generator @genItf3<x>() {
  kgen.param.if <eq(x, 0)> {
    "impl.0"() {attr=#kgen.param.decl.ref<"x"> : index}: () -> ()
    kgen.param.yield
  } else {
    "impl.1"() {attr=#kgen.param.decl.ref<"x"> : index} : () -> ()
    kgen.call @genItf3<sub(x, 1)>() : () -> ()
    kgen.param.yield
  }
  kgen.return
}

// CHECK-LABEL: kgen.func @"genItf3,x=0"()
// CHECK-NEXT:   "impl.0"() {attr = 0 : index}

// CHECK-LABEL: kgen.func @"genItf3,x=1"()
// CHECK-NEXT:   "impl.1"() {attr = 1 : index}
// CHECK-NEXT:   kgen.call @"genItf3,x=0"()

// CHECK-LABEL: kgen.func @"genItf3,x=2"()
// CHECK-NEXT:   "impl.1"() {attr = 2 : index}
// CHECK-NEXT:   kgen.call @"genItf3,x=1"()

// CHECK-LABEL: kgen.func @"genItf3,x=3"()
// CHECK-NEXT:   "impl.1"() {attr = 3 : index}
// CHECK-NEXT:   kgen.call @"genItf3,x=2"()

// CHECK-LABEL: kgen.func @"genItf3,x=4"()
// CHECK-NEXT:   "impl.1"() {attr = 4 : index}
// CHECK-NEXT:   kgen.call @"genItf3,x=3"()

// CHECK-LABEL:   kgen.func @use_Itf3() {
// CHECK-NEXT:      kgen.call @"genItf3,x=4"() : () -> ()
// CHECK-NEXT:      kgen.call @"genItf3,x=2"() : () -> ()
// CHECK-NEXT:      kgen.return
kgen.generator @use_Itf3() {
  kgen.call @genItf3<4>() : () -> ()
  kgen.call @genItf3<2>() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @"param_add,A=1,B=2"
kgen.generator @param_add<A, B>() -> index {
  // CHECK-NEXT: %index3 = kgen.param.constant = <3>
  %0 = kgen.param.constant = <add(A, B)>
  // CHECK-NEXT: kgen.return %index3
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @partial_bind_params_region
kgen.generator @partial_bind_params_region() -> index {
  kgen.param.declare BoundFn: <index>() -> index = <bind_params(:<index, index>() -> index @param_add, 1, ?)>
  // CHECK-NEXT: %0 = kgen.call @"param_add,A=1,B=2"() : () -> index
  %0 = kgen.call_param[() -> index: bind_params(:<index>() -> index BoundFn, 2)]()
  // CHECK-NEXT: kgen.return %0
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @"param_add2,A=1,B=2,C=3"
kgen.generator @param_add2<A, B, C>() -> index {
  // CHECK-NEXT: %index4 = kgen.param.constant = <4>
  %0 = kgen.param.constant = <add(sub(B, A), C)>
  // CHECK-NEXT: kgen.return %index4
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @partial_bind_params_region_2
kgen.generator @partial_bind_params_region_2() -> index {
  kgen.param.declare BoundFn: <index, index>() -> index = <bind_params(:<index, index, index>() -> index @param_add2, 1, ?, ?)>
  kgen.param.declare BoundFn2: <index>() -> index = <bind_params(:<index, index>() -> index BoundFn, ?, 3)>
  // CHECK-NEXT: %0 = kgen.call @"param_add2,A=1,B=2,C=3"() : () -> index
  %0 = kgen.call_param[() -> index: bind_params(:<index>() -> index BoundFn2, 2)]()
  // CHECK-NEXT: kgen.return %0
  kgen.return %0 : index
}

// -----

// COM: First instantiation of `@fwd` is inside an assert.

kgen.generator @fwd(%a: !kgen.scalar<bool>) -> !kgen.scalar<bool> {
  kgen.return %a : !kgen.scalar<bool>
}

kgen.generator @f() {
  kgen.param.assert <apply(:(!kgen.scalar<bool>) -> !kgen.scalar<bool> @fwd, true)>, "true"
  kgen.return
}

// CHECK-LABEL: kgen.func export @top
kgen.generator export @top() {
  // CHECK-NEXT: call @f
  kgen.call @f() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: @"g1,size=3"
// CHECK-LABEL: @"g1,size=5"
kgen.generator @g1<size>() -> index {
  %0 = kgen.param.constant = <size>
  kgen.return %0 : index
}

// CHECK-LABEL: @"g2,size=3,width=5"
kgen.generator @g2<size, width>() -> index {
  // CHECK-NEXT: call @"g1,size=5"
  %0 = kgen.call @g1<width>() : () -> index
  // CHECK-NEXT: call @"g1,size=3"
  %1 = kgen.call @g1<size>() : () -> index
  kgen.return %0 : index
}

// CHECK-LABEL: @root
kgen.generator @root() {
  kgen.param.declare q = <3>
  kgen.param.declare w = <5>
  // CHECK-NEXT: kgen.call @"g2,size=3,width=5"
  %0 = kgen.call @g2<q, w>() : () -> index
  kgen.return
}

// -----

// COM: Check that `elaborate-generators` attaches the host target info.

// CHECK: module attributes {M.target_info = #M.target<{{.*}}>}

kgen.generator @some_func() {
  kgen.return
}

// -----

// CHECK-LABEL: @constexprIfNoParams()
kgen.generator @constexprIfNoParams() {
  // CHECK-NEXT: "should.appear"
  kgen.param.if<true> {
    "should.appear"() : () -> ()
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  // CHECK-NEXT: kgen.return
  kgen.return
}

// CHECK-LABEL: @constexprIfBasic()
kgen.generator @constexprIfBasic() {
  kgen.param.declare cond_var = <32>

  // CHECK-NEXT: "should.appear"
  %0 = kgen.param.if <lt(cond_var, 10)> -> index {
    %1 = "should.not.appear"() : () -> index
    kgen.param.declare next_lt = <add(cond_var, 10)>
    kgen.param.yield %1 : index
  } else {
    %3 = "should.appear"() : () -> index
    kgen.param.declare next_gt = <add(cond_var, 20)>
    kgen.param.yield %3 : index
  }

  kgen.return
}

// CHECK-LABEL: @nestedConstexprIf()
kgen.generator @nestedConstexprIf() {
  kgen.param.declare cond_var = <32>

  // CHECK-NEXT: "should.appear"
  // CHECK-NOT: "should.not.appear"
  %0 = kgen.param.if <lt(cond_var, 10)> -> index {
    %1 = "should.not.appear"() : () -> index
    kgen.param.declare next_lt = <add(cond_var, 10)>
    kgen.param.yield %1 : index
  } else {
    %3 = kgen.param.if <gt(cond_var, 30)> -> index {
      %4 = "should.appear"() : () -> index
      kgen.param.yield %4 : index
    } else {
      %4 = "should.not.appear"() : () -> index
      kgen.param.yield %4 : index
    }
    kgen.param.yield %3 : index
  }

  kgen.return
}

// CHECK-LABEL: @nestedConstexprIf2()
kgen.generator @nestedConstexprIf2() {
  kgen.param.declare cond_var = <32>

  %0 = kgen.param.if <lt(cond_var, 10)> -> index {
    %1 = "should.not.appear"() : () -> index
    kgen.param.declare next_lt = <add(cond_var, 10)>
    kgen.param.yield %1 : index
  } else {
    // CHECK-NEXT: param.constant: scalar<bool> = <true>
    %condition = kgen.param.constant : scalar<bool> = <gt(cond_var, 30)>
    // CHECK-NEXT: hlcf.if
    %3 = hlcf.if %condition -> index {
      // CHECK-NEXT: "should.appear"
      %4 = "should.appear"() : () -> index
      kgen.param.declare next_inner = <35>
      // CHECK-NEXT: hlcf.yield
      hlcf.yield %4 : index
      // CHECK-NEXT: else
    } else {
      // CHECK-NEXT: "should.also.appear"
      %4 = "should.also.appear"() : () -> index
      // CHECK-NEXT: hlcf.yield
      hlcf.yield %4 : index
    }
    // CHECK-NOT: param.yield
    kgen.param.yield %3 : index
  }

  kgen.return
}

// -----

// CHECK-LABEL: @"constexprIfInputParam,x=11"
kgen.generator @constexprIfInputParam<x>() {
  // CHECK-NEXT: "should.appear"
  %0 = kgen.param.if <gt(x, 10)> -> index {
    %1 = "should.appear"() : () -> index
    kgen.param.declare next_lt = <add(x, 10)>
    kgen.param.yield %1 : index
  } else {
    %3 = "should.not.appear"() : () -> index
    kgen.param.declare next_gt = <add(x, 20)>
    kgen.param.yield %3 : index
  }

  kgen.return
}

kgen.generator @caller() {
  kgen.call @constexprIfInputParam<11>() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: @constexprIfEarlyExit
kgen.generator @constexprIfEarlyExit() -> index {
  kgen.param.declare x = <11>
  // CHECK-NEXT: [[RES:%[0-9]+]] = "should.appear"
  %0 = kgen.param.if <gt(x, 10)> -> index {
    %1 = "should.appear"() : () -> index
    // CHECK-NEXT: kgen.return [[RES]]
    kgen.return %1 : index
  } else {
    %3 = "should.not.appear"() : () -> index
    kgen.param.yield %3 : index
  }
  // CHECK-NOT: param.constant = <32>
  %4 = kgen.param.constant = <32>

  kgen.return %0 : index
}

// COM: This ensures that the blocks after the early exit are correctly
// COM: removed *without* a use-after-free during elaboration.
// CHECK-LABEL: @constexprIfEarlyExit2
kgen.generator @constexprIfEarlyExit2() -> index {
  kgen.param.declare x = <11>
  // CHECK-NEXT: [[RES:%[0-9]+]] = "should.appear"
  %0 = kgen.param.if <gt(x, 10)> -> index {
    %1 = "should.appear"() : () -> index
    // CHECK-NEXT: kgen.return [[RES]]
    kgen.return %1 : index
  } else {
    %3 = "should.not.appear"() : () -> index
    kgen.param.yield %3 : index
  }
  // CHECK-NOT: "should.not.appear"
  kgen.param.if <gt(x, 10)> {
    "should.not.appear"() : () -> ()
    %4 = index.constant 3
    // CHECK-NOT: kgen.return
    kgen.return %4 : index
  } else {
    kgen.param.yield
  }
  // CHECK-NOT: param.constant = <32>
  %4 = kgen.param.constant = <32>

  kgen.return %0 : index
}

// CHECK-LABEL: @constexprIfEarlyExitWithParam
kgen.generator @constexprIfEarlyExitWithParam() -> index {
  // CHECK-NEXT: [[RES:%[0-9]+]] = "should.appear"
  %0 = kgen.param.if <gt(x, 10)> -> index {
    %1 = "should.appear"() : () -> index
    // CHECK-NEXT: kgen.return [[RES]]
    kgen.return %1 : index
  } else {
    %3 = "should.not.appear"() : () -> index
    kgen.param.yield %3 : index
  }
  kgen.param.declare x = <11>
  // CHECK-NOT: param.constant = <32>
  %4 = kgen.param.constant = <32>

  kgen.return %0 : index
}

// CHECK-LABEL: @constexprIfEarlyExitWithParam2
kgen.generator @constexprIfEarlyExitWithParam2() -> index {
  // CHECK-NEXT: param.constant = <11>
  %0 = kgen.param.constant = <x>
  // CHECK-NEXT: [[RES:%[0-9]+]] = "should.appear"
  %1 = kgen.param.if <true> -> index {
    %2 = "should.appear"() : () -> index
    // CHECK-NEXT: kgen.return [[RES]]
    kgen.return %2 : index
  } else {
    %3 = "should.not.appear"() : () -> index
    kgen.param.yield %3 : index
  }
  kgen.param.declare x = <11>
  // CHECK-NOT: param.constant = <32>
  %4 = kgen.param.constant = <32>

  kgen.return %1 : index
}

// -----

kgen.generator @returnTrue() -> !kgen.scalar<bool> {
  %0 = kgen.param.constant: scalar<bool> = <<true>>
  kgen.return %0 : !kgen.scalar<bool>
}

// CHECK-LABEL: @constexprIfFunctionCallCondition
kgen.generator @constexprIfFunctionCallCondition() -> index {
  // CHECK-NEXT: [[RES:%[0-9]+]] = "should.appear"
  %1 = kgen.param.if <apply(:() -> !kgen.scalar<bool> @returnTrue)> -> index {
    %2 = "should.appear"() : () -> index
    // CHECK-NEXT: kgen.return [[RES]]
    kgen.param.yield %2 : index
  } else {
    %3 = "should.not.appear"() : () -> index
    kgen.param.yield %3 : index
  }

  kgen.return %1 : index
}

kgen.generator @returnInputParam(%arg0: !kgen.struct<(scalar<bool>)>) -> !kgen.scalar<bool> {
  %1 = kgen.struct.extract %arg0[0] : !kgen.struct<(scalar<bool>)>
  kgen.return %1 : !kgen.scalar<bool>
}

kgen.generator @returnTrueStruct() -> !kgen.struct<(scalar<bool>)> {
  %0 = kgen.param.constant: scalar<bool> = <<true>>
  %1 = kgen.struct.create(%0) : !kgen.struct<(scalar<bool>)>
  kgen.return %1 : !kgen.struct<(scalar<bool>)>
}

// CHECK-LABEL: @"ifFn
kgen.generator @ifFn<true: !kgen.struct<(scalar<bool>)>>() -> index {
  // CHECK-NEXT: [[RES:%[0-9]+]] = "should.appear"
  %1 = kgen.param.if <apply(:(!kgen.struct<(scalar<bool>)>) -> !kgen.scalar<bool> @returnInputParam, true)> -> index {
    %2 = "should.appear"() : () -> index
    // CHECK-NEXT: kgen.return [[RES]]
    kgen.param.yield %2 : index
  } else {
    %3 = "should.not.appear"() : () -> index
    kgen.param.yield %3 : index
  }

  kgen.return %1 : index
}

// CHECK-LABEL: @constexprIfFunctionCallCondition2
kgen.generator @constexprIfFunctionCallCondition2() {
  kgen.param.declare true: !kgen.struct<(scalar<bool>)> = <apply(:() -> !kgen.struct<(scalar<bool>)> @returnTrueStruct)>
  %0 = kgen.call @ifFn<:!kgen.struct<(scalar<bool>)> true>() : () -> index
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @substitute_current_target
kgen.generator @substitute_current_target() {
  // CHECK-NEXT: constant: target = <#kgen.target<triple = {{.*}}>>
  kgen.param.constant: target = <current_target()>
  kgen.return
}

// -----

// CHECK: module
// CHECK-NOT: kgen.func

kgen.generator @not_a_primary_generator<N>() {
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @rebind_parameter
kgen.generator @rebind_parameter() {
  // CHECK-NEXT: constant: array<2, index> = <[1, 2]>
  kgen.param.declare size = <2>
  kgen.param.declare list_input: array<2, index> = <[1, 2]>
  kgen.param.declare list_output: array<size, index> = <rebind(:array<2, index> list_input)>
  kgen.param.constant: array<size, index> = <list_output>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @recurse
// CHECK-SAME: () {
// CHECK-NEXT:  kgen.call @recurse() : () -> ()
// CHECK-NEXT:  kgen.return
// CHECK-NEXT:  }
kgen.generator @recurse() {
  kgen.call @recurse() : () -> ()
  kgen.return
}

// -----

// COM: Tricky recursion order.

// CHECK-LABEL: kgen.func @err
kgen.generator @err() {
  // CHECK-NEXT: call @call
  kgen.call @call() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() {
  // CHECK-NEXT: call @getattr
  kgen.call @getattr() : () -> ()
  // CHECK-NEXT: call @call
  kgen.call @call() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @getattr
kgen.generator @getattr() {
  // CHECK-NEXT: call @err
  kgen.call @err() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @call
kgen.generator @call() {
  // CHECK-NEXT: call @err
  kgen.call @err() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @unpack_in_type
kgen.generator @unpack_in_type() {
  // CHECK-NEXT: array<1, index>
  %0 = pop.stack_allocation 1 x array<apply(:() -> index @produce_one), index>
  kgen.return
}

kgen.generator @produce_one() -> index {
  %0 = kgen.param.constant: index = <1>
  kgen.return %0 : index
}

// CHECK-LABEL: func @"paramRecurse,in=0"()
// CHECK-NEXT: return

// CHECK-LABEL: func @"paramRecurse,in=1"()
// CHECK-NEXT: call @"paramRecurse,in=0"

// CHECK-LABEL: func @"paramRecurse,in=2"()
// CHECK-NEXT: call @"paramRecurse,in=1"

// CHECK-LABEL:func  @"paramRecurse,in=3"()
// CHECK-NEXT: call @"paramRecurse,in=2"

kgen.generator @paramRecurse<in>() {
  kgen.param.if <eq(in, 0)> {
    kgen.param.yield
  } else {
    kgen.call @paramRecurse<add(in, -1)>() : () -> ()
    kgen.param.yield
  }
  kgen.return
}

kgen.generator @caller() {
  kgen.call @paramRecurse<3>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @pointer_attr_elaborate
kgen.generator @pointer_attr_elaborate() {
  // CHECK-NEXT: kgen.param.constant: pointer<i8> = <0>
  kgen.param.declare type1: type = <i8>
  %0 = kgen.param.constant: pointer<type1> = <0>
  kgen.return
}

// -----

// COM: https://github.com/modularml/modular/issues/9745

// CHECK-LABEL: kgen.func @true_inside_false_param_if
kgen.generator @true_inside_false_param_if() {
  // CHECK-NEXT: should.appear
  // CHECK-NEXT: kgen.return
  kgen.param.if <false> {
    "should.not.appear"() : () -> ()
    kgen.return
  } else {
    kgen.param.if <true> {
      "should.appear"() : () -> ()
      kgen.return
    } else {
      "should.not.appear"() : () -> ()
      kgen.param.yield
    }
    kgen.param.yield
  }
  kgen.return
}

// CHECK-LABEL: kgen.func @"param_if_different,cond=false"
// CHECK-NEXT: constant = <3>

// CHECK-LABEL: kgen.func @"param_if_different,cond=true"
// CHECK-NEXT: constant = <2>
kgen.generator @param_if_different<cond: scalar<bool>>() {
  kgen.param.declare a = <3>
  kgen.param.if <cond> {
    kgen.param.declare b = <2>
    kgen.param.constant = <b>
    kgen.param.yield
  } else {
    kgen.param.constant = <a>
    kgen.param.yield
  }
  kgen.return
}

kgen.generator @instantiate() {
  kgen.call @param_if_different<:scalar<bool> true>() : () -> ()
  kgen.call @param_if_different<:scalar<bool> false>() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @async_function()
kgen.generator @async_function() async {
  kgen.return
}

// CHECK-LABEL: kgen.func @call_it
kgen.generator @call_it() {
  // CHECK: co.invoke[() async -> (): @async_function]
  kgen.param.declare fn: () async -> () = <@async_function>
  co.invoke[() async -> (): fn]()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @async_fn() async
kgen.generator @async_fn() async {
  kgen.return
}

// CHECK-LABEL: kgen.func export @nonparametric_async_call
kgen.generator export @nonparametric_async_call() {
  // CHECK-NEXT: co.invoke[() async -> (): @async_fn]
  co.invoke[() async -> (): @async_fn]()
  kgen.return
}

// -----

// COM: Check conditional parameter expressions.

kgen.generator @add_param<a : index>(%v : index) -> index {
  %0 = kgen.param.constant : index = <a>
  %1 = index.add %v, %0
  kgen.return %1 : index
}

// CHECK-MAIN-LABEL: kgen.func @"add_param,a=1"
// CHECK-PARAMINTERP-NOT: kgen.func @"add_param,a=1"
// CHECK-NOT: kgen.func @"add_param,a=2"
// CHECK-NOT: kgen.func @"add_param,a=3"
// CHECK-MAIN: kgen.func @"add_param,a=4"
// CHECK-PARAMINTERP-NOT: kgen.func @"add_param,a=4"
// CHECK-LABEL: kgen.func @param_cond
kgen.generator @param_cond() -> () {
  kgen.param.declare cond_false : !kgen.scalar<bool> = <false>
  kgen.param.declare cond_true : !kgen.scalar<bool> = <true>

  // COM: This should NOT evaluate @add_param<2> during parameter evaluation
  // CHECK-PARAMINTERP:  kgen.param.constant = <1>
  %5 = kgen.param.constant: index = <cond(cond_true,
        apply(:(index) -> index @add_param<1>, 0), apply(:(index) -> index @add_param<2>, 0))>
  // COM: This should NOT evaluate @add_param<3> during parameter evaluation
  // CHECK-PARAMINTERP:  kgen.param.constant = <4>
  %6 = kgen.param.constant: index = <cond(cond_false,
        apply(:(index) -> index @add_param<3>, 0), apply(:(index) -> index @add_param<4>, 0))>

  kgen.return
}

// -----

// CHECK: kgen.func @"callee,a=1"
// CHECK: kgen.func @"callee,a=2"

kgen.generator @callee<a>(%arg0: index) {
  kgen.return
}

kgen.generator @entry(%arg0: index) {
  // CHECK: create_closure[(index) -> (): @"callee,a=1"]
  kgen.create_closure[(index) -> (): @callee<1>]()
  kgen.param.declare fn: (index) -> () = <@callee<2>>
  // CHECK: create_closure[(index) -> (): @"callee,a=2"]
  kgen.create_closure[(index) -> (): fn](%arg0)
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @"recurse,axis=0"
// CHECK-NEXT:    kgen.return %arg0 : index

// CHECK-LABEL: kgen.func @"recurse,axis=1"
// CHECK-NEXT:    %0 = kgen.call @"recurse,axis=0"(%arg0) : (index) -> index
// CHECK-NEXT:    kgen.return %0 : index

// CHECK-LABEL: kgen.func @"recurse,axis=2"
// CHECK-NEXT:    %0 = kgen.call @"recurse,axis=1"(%arg0) : (index) -> index
// CHECK-NEXT:    kgen.return %0 : index

kgen.generator @recurse<axis>(%arg0: index) -> index {
  kgen.param.if <eq(axis, 0)> {
    kgen.return %arg0 : index
  } else {
    kgen.param.yield
  }
  %0 = kgen.call @recurse<add(axis, -1)>(%arg0) : (index) -> index
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @main
kgen.generator @main() {
  %idx42 = index.constant 42
  // CHECK: kgen.call @"recurse,axis=2"(%idx42) : (index) -> index
  %1 = kgen.call @recurse<2>(%idx42) : (index) -> index
  kgen.return
}

// -----

kgen.generator @take_closure(%arg0: !kgen.generator<(index) capturing -> index>, %arg1: index) {
  %0 = kgen.call_indirect %arg0(%arg1) : (index) capturing -> index
  kgen.return
}

// COM: Ensure that regions lifted by OutlineClosures pass are not erased
// CHECK-LABEL: kgen.func @"foo_k,N=5,M=3"() capturing -> !kgen.scalar<index> {
kgen.generator @foo_k<N, M>() capturing -> !kgen.scalar<index> {
  %0 = kgen.param.constant: scalar<index> = <0>
  kgen.return %0 : !kgen.scalar<index>
}

// CHECK-LABEL: kgen.func @"foo,N=5"(%arg0: !kgen.scalar<index>) {
kgen.generator @foo<N>(%arg0: !kgen.scalar<index>) {
  kgen.param.declare k: <index>() capturing -> !kgen.scalar<index> = <@foo_k<N, ?>>
  // CHECK: kgen.create_closure[() capturing -> !kgen.scalar<index>: @"foo_k,N=5,M=3"]()
  %1 = kgen.create_closure[() capturing -> !kgen.scalar<index>: bind_params(:<index>() capturing -> !kgen.scalar<index> k, 3)]()
  kgen.return
}

// CHECK-LABEL: kgen.func @main
kgen.generator @main() {
  %simd = kgen.param.constant: scalar<index> = <0>
  kgen.param.declare Bound: (!kgen.scalar<index>) -> () = <@foo<5>>
  // CHECK: kgen.call @"foo,N=5"(%simd) : (!kgen.scalar<index>) -> ()
  kgen.call_param[(!kgen.scalar<index>) -> (): Bound](%simd)
  kgen.return
}

// COM: Ensure that staged closures follow the global store
kgen.generator @take_bat(%arg0: !kgen.generator<(index) capturing -> index>) {
	kgen.return
}

kgen.generator @bat(%arg0: index) capturing -> index {
	kgen.return %arg0 : index
}

// CHECK-LABEL: kgen.func @bat_binder
kgen.generator @bat_binder(%arg0: index) {
  // CHECK: kgen.create_closure[(index) capturing -> index: @bat]()
	%2 = kgen.create_closure[(index) capturing -> index: h]()
	kgen.param.declare h: (index) capturing -> index = <@bat>
	kgen.return
}

// -----

kgen.generator @count_ops(%arg0: !kgen.scalar<bool>) -> index {
  %0 = hlcf.if %arg0 -> index {
    %idx0 = index.constant 0
    hlcf.yield %idx0 : index
  } else {
    %idx1 = index.constant 1
    hlcf.yield %idx1 : index
  }
  kgen.return %0 : index
}

kgen.generator @cost_of<fn: (!kgen.scalar<bool>) -> index>() -> index {
  %0:8 = kgen.cost_of[(!kgen.scalar<bool>) -> index: fn]
  // CHECK-MAIN: %loads, %stores, %additions, %comparisons, %divisions, %multiplications, %multiplyAdds, %other = kgen.cost_of[(!kgen.scalar<bool>) -> index: @count_ops]
  kgen.return %0#7  : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() {
  // CHECK-NEXT: <1>
  %0 = kgen.param.constant = <apply(:() -> index @cost_of<:(!kgen.scalar<bool>) -> index @count_ops>)>
  kgen.return
}

// -----

kgen.generator @pop_add(%arg1: !kgen.scalar<si32>) -> !kgen.scalar<si32> {
  %0 = pop.add %arg1, %arg1 : !kgen.scalar<si32>
  kgen.return %0 : !kgen.scalar<si32>
}

kgen.generator
@cost_of_pop_add<fn: (!kgen.scalar<si32>) -> !kgen.scalar<si32>>() -> index {
  // CHECK-MAIN: %loads, %stores, %additions, %comparisons, %divisions, %multiplications, %multiplyAdds, %other = kgen.cost_of[(!kgen.scalar<si32>) -> !kgen.scalar<si32>: @pop_add]
  %0:8 = kgen.cost_of[(!kgen.scalar<si32>) -> !kgen.scalar<si32>: fn]
  kgen.return %0#2  : index
}

kgen.generator @index_add(%arg1: index) -> index {
  // CHECK-MAIN: %loads, %stores, %additions, %comparisons, %divisions, %multiplications, %multiplyAdds, %other = kgen.cost_of[(index) -> index: @index_add]
  %0 = index.add %arg1, %arg1
  kgen.return %0 : index
}

kgen.generator @cost_of_index_add<fn: (index) -> index>() -> index {
  %0:8 = kgen.cost_of[(index) -> index: fn]
  kgen.return %0#2  : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() {
  // CHECK-NEXT: <1>
  %0 = kgen.param.constant = <apply(:() -> index @cost_of_pop_add<:(!kgen.scalar<si32>) -> !kgen.scalar<si32> @pop_add>)>
  // CHECK-NEXT: <1>
  %1 = kgen.param.constant = <apply(:() -> index @cost_of_index_add<:(index) -> index @index_add>)>
 kgen.return
}

// -----

// CHECK-LABEL: kgen.func @preelaborated()
kgen.func @preelaborated() {
  // CHECK-NEXT: kgen.return
  kgen.return
}

// -----

module attributes {kgen.env = #kgen.env<{unit_value, int_value = 42 : index, str_value = "hello" : !kgen.string}>} {
  // CHECK-LABEL: kgen.func @env_test
  kgen.generator @env_test() {
    // CHECK-NEXT: i1 = <0>
    kgen.param.constant: i1 = <get_env("doesnt_exist")>
    // CHECK-NEXT: i1 = <1>
    kgen.param.constant: i1 = <get_env("unit_value")>
    // CHECK-NEXT: <42>
    kgen.param.constant = <get_env("int_value")>
    // CHECK-NEXT: string = <"hello">
    kgen.param.constant: string = <get_env("str_value")>
    kgen.return
  }
}

// -----

kgen.func @already_concrete() -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func export @interpret_concrete
kgen.generator export @interpret_concrete() {
  // CHECK-NEXT: = <0>
  kgen.param.constant = <apply(:() -> index @already_concrete)>
  kgen.return
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="+foobar", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @custom_target
  kgen.generator @custom_target() {
    // CHECK: constant: i1 = <1>
    kgen.param.constant: i1 = <target_has_feature(current_target(), "foobar")>
    kgen.return
  }
}

// -----

// Explicitly disabled feature (leading '-') must return false even if the
// feature name appears in the string.
module attributes {M.target_info = #M.target<triple="", arch="", features="+avx2,-avx512f", data_layout="", simd_bit_width=256>} {
  // CHECK-LABEL: @disabled_feature_returns_false
  kgen.generator @disabled_feature_returns_false() {
    // CHECK: constant: i1 = <1>
    kgen.param.constant: i1 = <target_has_feature(current_target(), "avx2")>
    // CHECK: constant: i1 = <0>
    kgen.param.constant: i1 = <target_has_feature(current_target(), "avx512f")>
    kgen.return
  }
}

// -----

kgen.generator @kernel() {
  kgen.return
}

// CHECK-LABEL: kgen.func export @top
kgen.generator export @top() {
  // COM: Just check that the code compiles. The assembly is target-dependent.
  // CHECK: constant: struct
  kgen.param.constant: struct<(string, index, (!kgen.pointer<none>) capturing -> !kgen.none)> = <#kgen.compile_assembly<current_target(), =asm, "", false, :() -> () @kernel>>
  kgen.return
}

// -----

kgen.generator @func() {
  kgen.return
}

// CHECK-LABEL: kgen.func export @top
kgen.generator export @top() {
  // CHECK: constant: string = <"func">
  kgen.param.constant: string = <#kgen.get_linkage_name<current_target(), #kgen.symbol.constant<@func> : !kgen.generator<() -> ()>>>
  kgen.return
}

// -----

kgen.generator @no_params() {
  kgen.return
}

kgen.generator @params<a, b>() -> (index, index) {
  %0 = kgen.param.constant = <a>
  %1 = kgen.param.constant = <b>
  kgen.return %0, %1 : index, index
}

kgen.generator @func_param<f: <index, index>() -> (index, index)>() -> index {
  kgen.param.declare bind: () -> (index, index) = <bind_params(:<index, index>() -> (index, index) f, 7, 9)>
  %0, %1 = kgen.call_param[() -> (index, index): bind]()
  %2 = index.add %0, %1
  kgen.return %2 : index
}

!capture = !kgen.struct<(string, index, (!kgen.pointer<pointer<none>>) capturing -> !kgen.none)>

// CHECK-LABEL: kgen.func export @get_linkage_name
kgen.generator export @get_linkage_name() {
  // CHECK-NEXT: constant: struct<(string, index, {{.*}})> = <{ "{{.*}}no_params
  %0 = kgen.param.constant: !kgen.struct<(string, index, (!kgen.pointer<none>) capturing -> !kgen.none)> = <#kgen.compile_assembly<current_target(), =asm, "", false, :() -> () @no_params>>
  // CHECK-NEXT: constant: string = <"no_params">
  %1 = kgen.param.constant: string = <#kgen.get_linkage_name<current_target(), #kgen.symbol.constant<@no_params> : !kgen.generator<() -> ()>>>
  // CHECK-NEXT: constant: struct<(string, index, {{.*}})> = <{ "{{.*}}params,a=1,b=2
  %2 = kgen.param.constant: !kgen.struct<(string, index, (!kgen.pointer<none>) capturing -> !kgen.none)> = <#kgen.compile_assembly<current_target(), =asm, "", false, :() -> (index, index) @params<1, 2>>>
  // CHECK-NEXT: constant: string = <"params,a=1,b=2">
  %3 = kgen.param.constant: string = <#kgen.get_linkage_name<current_target(), #kgen.symbol.constant<@params<1, 2>> : !kgen.generator<() -> (index, index)>>>
  // CHECK-NEXT: constant: struct<(string, index, {{.*}})> = <{ "{{.*}}func_param,f=params
  %4 = kgen.param.constant: !kgen.struct<(string, index, (!kgen.pointer<none>) capturing -> !kgen.none)> = <#kgen.compile_assembly<current_target(), =asm, "", false, :() -> index @func_param<:<index, index>() -> (index, index) @params>>>
  // CHECK-NEXT: constant: string = <"func_param,f=params">
  %5 = kgen.param.constant: string = <#kgen.get_linkage_name<current_target(), #kgen.symbol.constant<@func_param<:<index, index>() -> (index, index) @params>> : !kgen.generator<() -> index>>>
  kgen.return
}

// -----

// COM: Function-reflection attributes evaluate against `kgen.generator` decls
// COM: post-elaboration, returning structural metadata: parameter count, names,
// COM: and the raising flag.

kgen.generator @no_parameters() {
  kgen.return
}

kgen.generator @two_parameters<a, b: index>() -> index {
  %0 = kgen.param.constant = <a>
  %1 = kgen.param.constant = <b>
  %2 = index.add %0, %1
  kgen.return %2 : index
}

kgen.generator @raising_func() throws {
  kgen.return
}

// CHECK-LABEL: kgen.func export @get_function_reflection
kgen.generator export @get_function_reflection() {
  // CHECK-NEXT: kgen.param.constant = <0>
  kgen.param.constant: index = <#kgen.get_function_parameter_count<#kgen.symbol.constant<@no_parameters> : !kgen.generator<() -> ()>>>
  // CHECK-NEXT: kgen.param.constant: param_list<string> = <[]>
  kgen.param.constant: !kgen.param_list<!kgen.string> = <#kgen.get_function_parameter_names<#kgen.symbol.constant<@no_parameters> : !kgen.generator<() -> ()>>>
  // CHECK-NEXT: kgen.param.constant: i1 = <0>
  kgen.param.constant: i1 = <#kgen.get_function_is_raising<#kgen.symbol.constant<@no_parameters> : !kgen.generator<() -> ()>>>

  // CHECK-NEXT: kgen.param.constant = <2>
  kgen.param.constant: index = <#kgen.get_function_parameter_count<#kgen.symbol.constant<@two_parameters<1, 2>> : !kgen.generator<() -> index>>>
  // CHECK-NEXT: kgen.param.constant: param_list<string> = <["a", "b"]>
  kgen.param.constant: !kgen.param_list<!kgen.string> = <#kgen.get_function_parameter_names<#kgen.symbol.constant<@two_parameters<1, 2>> : !kgen.generator<() -> index>>>
  // CHECK-NEXT: kgen.param.constant: i1 = <0>
  kgen.param.constant: i1 = <#kgen.get_function_is_raising<#kgen.symbol.constant<@two_parameters<1, 2>> : !kgen.generator<() -> index>>>

  // CHECK-NEXT: kgen.param.constant: i1 = <1>
  kgen.param.constant: i1 = <#kgen.get_function_is_raising<#kgen.symbol.constant<@raising_func> : !kgen.generator<() throws -> ()>>>
  kgen.return
}

// -----

// COM: Reflection prefers the `sourceParamList` snapshot when present. This
// COM: models a generator that's been through `RemoveUnusedParams`: the live
// COM: `inputParams` only retains the surviving param, but the snapshot still
// COM: carries the full source-declared name list.

kgen.generator @snapshot_only<survivor: index>() -> index attributes {
  sourceParamList = #kgen.pog_list<[
    <"dropped_first", pos_or_kw, not_vararg>,
    <"survivor", pos_or_kw, not_vararg>,
    <"dropped_last", pos_or_kw, not_vararg>
  ]>
} {
  %0 = kgen.param.constant = <survivor>
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @get_function_reflection_snapshot
kgen.generator export @get_function_reflection_snapshot() {
  // The count reflects the snapshot's three names, not the single live param.
  // CHECK-NEXT: kgen.param.constant = <3>
  kgen.param.constant: index = <#kgen.get_function_parameter_count<#kgen.symbol.constant<@snapshot_only<7>> : !kgen.generator<() -> index>>>
  // CHECK-NEXT: kgen.param.constant: param_list<string> = <["dropped_first", "survivor", "dropped_last"]>
  kgen.param.constant: !kgen.param_list<!kgen.string> = <#kgen.get_function_parameter_names<#kgen.symbol.constant<@snapshot_only<7>> : !kgen.generator<() -> index>>>
  kgen.return
}

// -----

kgen.struct.generator @NonParametric = struct_inst<
  "NonParametric"(data: index)>

kgen.struct.generator @"LinkedList"<T: type> = struct_inst<
  "LinkedList"[T]<:type T>(data: typevalue<T>)>

#linkedlist = #kgen.type<typevalue<:!kgen.type #kgen.genref<@LinkedList<:type none>>>, struct<(none, !kgen.pointer<none>)>> : !kgen.type

// CHECK-LABEL: kgen.func @"parameter_get_type_name
kgen.generator @parameter_get_type_name<T: type>(%arg: !kgen.param<T>) {
  // CHECK-NEXT: constant: string = <"LinkedList[<unprintable>]">
  kgen.param.constant: string = <#kgen.get_type_name<#kgen.param.decl.ref<"T">, #kgen.simd<false>:!kgen.scalar<bool>>>
  kgen.return
}

// CHECK-LABEL: kgen.func export @get_type_name
kgen.generator export @get_type_name(%arg0: !kgen.struct<(none, !kgen.pointer<none>)>) {
  // CHECK-NEXT: constant: string = <"NonParametric">
  kgen.param.constant: string = <#kgen.get_type_name<#kgen.genref<@NonParametric>, #kgen.simd<false>:!kgen.scalar<bool>>>
  kgen.call @parameter_get_type_name<:type #linkedlist>(%arg0) : (!kgen.struct<(none, !kgen.pointer<none>)>) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func export @get_type_name_of_ref
kgen.generator export @get_type_name_of_ref() {
  // A lowered !lit.ref to a named struct: AsValue is pointer-to-typevalue.
  // CHECK-NEXT: constant: string = <"ref[NonParametric]">
  kgen.param.constant: string = <#kgen.get_type_name<[pointer<typevalue<#kgen.genref<@NonParametric>>>, pointer<index>], #kgen.simd<false>:!kgen.scalar<bool>>>
  kgen.return
}

// CHECK-LABEL: kgen.func export @get_type_name_of_parametric_ref
kgen.generator export @get_type_name_of_parametric_ref() {
  // Same pointer wrapping, parameterized struct.
  // CHECK-NEXT: constant: string = <"ref[LinkedList[<unprintable>]]">
  kgen.param.constant: string = <#kgen.get_type_name<[pointer<typevalue<:!kgen.type #kgen.genref<@LinkedList<:type none>>>>, pointer<index>], #kgen.simd<false>:!kgen.scalar<bool>>>
  kgen.return
}

// -----

!capture = !kgen.struct<(string, index, (!kgen.pointer<none>) capturing -> !kgen.none)>

kgen.generator @lambda() capturing -> index {
  %0 = pop.compiler.global_load "var" : index
  kgen.return %0 : index
}

kgen.generator @captures<f: () capturing -> index>() capturing -> index {
  %0 = kgen.call_param[() capturing -> index: f]()
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() {
  // CHECK-NEXT: struct<(string, index, (!kgen.pointer<none>) capturing -> !kgen.none)> = <{ "{{.*}}", 1, [[POPULATE:@.*]] }>
  %0 = kgen.param.constant: !capture = <#kgen.compile_assembly<current_target(), =asm, "", false, :() capturing -> index @captures<:() capturing -> index @lambda>>>
  kgen.return
}

// CHECK: kgen.func [[POPULATE]](%arg0: !kgen.pointer<none>) capturing -> !kgen.none always_inline
// CHECK: [[VAR:%.*]] = pop.compiler.global_load "var" : index
// CHECK: [[ARG:%.*]] = pop.stack_allocation
// CHECK: pop.store [[VAR]], [[ARG]]
// CHECK: [[ARGCAST:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<pointer<none>>
// CHECK: [[PTR:%.*]] = pop.offset [[ARGCAST]][%index0]
// CHECK: [[RAW:%.*]] = pop.pointer.bitcast [[ARG]]
// CHECK: pop.store [[RAW]], [[PTR]]

// -----

kgen.generator @impl(%arg0: !kgen.pointer<i8, apply(:(index) -> index @fwd, 0)>) {
  kgen.return
}

// CHECK-LABEL: kgen.func export @variadic
kgen.generator export @variadic() {
  // CHECK: constant: param_list<(!kgen.pointer<i8>) -> ()> = <[@impl]>
  kgen.param.constant: param_list<(!kgen.pointer<i8, apply(:(index) -> index @fwd, 0)>) -> ()> = <[@impl]>
  kgen.return
}

kgen.generator @fwd(%arg0: index) -> index {
  kgen.return %arg0 : index
}

// -----

// CHECK-LABEL: kgen.func @"decorators,a=1"
// CHECK-NEXT: decorators <1>

// CHECK-LABEL: kgen.func @"decorators,a=2"
// CHECK-NEXT: decorators <1>

kgen.generator @decorators<a>()
    decorators<1> {
  kgen.return
}

kgen.generator @elaborate() {
  kgen.call @decorators<1>() : () -> ()
  kgen.call @decorators<2>() : () -> ()
  kgen.return
}

// -----

kgen.generator @func<x>() -> !kgen.simd<x, f32> {
  kgen.unreachable
}

kgen.generator @create<T: type>(%arg0: !kgen.param<T>) -> !kgen.variant<T, i1> {
  %0 = kgen.variant.create %arg0, 0 : <T, i1>
  kgen.return %0 : !kgen.variant<T, i1>
}

// CHECK-LABEL: kgen.func export @entry
kgen.generator export @entry() {
  // CHECK: constant: variant<<index>() -> !kgen.simd<*(0,0), f32>, i1> = <{:<index>() -> !kgen.simd<*(0,0), f32> @func, 0}>
  kgen.param.apply value = [(!kgen.generator<<index>() -> !kgen.simd<*(0,0), f32>>) -> !kgen.variant<<index>() -> !kgen.simd<*(0,0), f32>, i1>: @create<:type <index>() -> !kgen.simd<*(0,0), f32>>](@func)
  kgen.param.constant: variant<<index>() -> !kgen.simd<*(0,0), f32>, i1> = <value>
  kgen.return
}

// -----

// CHECK: kgen.func @func
kgen.generator @func() {
  kgen.unreachable
}

// CHECK: kgen.func @"param,a=2"
// CHECK: kgen.func @"param,a=3"
kgen.generator @param<a>() {
  kgen.unreachable
}

// CHECK-LABEL: kgen.func export @entry
kgen.generator export @entry() {
  // CHECK: constant: () -> () = <@func>
  kgen.param.constant: () -> () = <@func>
  // CHECK: constant: () -> () = <@"param,a=2">
  kgen.param.constant: () -> () = <@param<2>>
  // CHECK: constant: struct<(() -> ())> = <{ @"param,a=3" }>
  kgen.param.constant: struct<(() -> ())> = <{ @param<3> }>
  kgen.return
}

// -----

// During elaboration of this example, the type:
//
// <index>(!pop.array<cond(apply(:(index, index) -> !kgen.scalar<bool> @eq, *(0,0), 0), 1, *(0,0)), index>) -> ()
//
// appears in the IR. This type is actually concrete from the perspective of the
// current frame, because it has no parameter expressions. It contains parameter
// operators, but they are part of the signature.
//
// Ensure that this type is valid.

kgen.generator @init<T: type>(%arg0: !kgen.param<T>) -> !kgen.struct<()> {
  %struct = kgen.param.constant: struct<()> = <{  }>
  kgen.return %struct : !kgen.struct<()>
}

kgen.generator @eq(%arg0: index, %arg1: index) -> !kgen.scalar<bool> {
  %cmp = index.cmp eq(%arg0, %arg1)
  %0 = pop.cast_from_builtin %cmp : i1 to !kgen.scalar<bool>
  kgen.return %0 : !kgen.scalar<bool>
}

kgen.generator @make<x>(%arg0: !pop.array<cond(apply(:(index, index) -> !kgen.scalar<bool> @eq, x, 0), 1, x), index>) {
  kgen.return
}

// CHECK-LABEL: kgen.func export @top
kgen.generator export @top() {
  // CHECK-NEXT: constant: struct<()> = <{ }>
  kgen.param.apply lifted = [(!kgen.generator<<index>(!pop.array<cond(apply(:(index, index) -> !kgen.scalar<bool> @eq, *(0,0), 0), 1, *(0,0)), index>) -> ()>) -> !kgen.struct<()>: @init<:type <index>(!pop.array<cond(apply(:(index, index) -> !kgen.scalar<bool> @eq, *(0,0), 0), 1, *(0,0)), index>) -> ()>](@make)
  kgen.param.constant: struct<()> = <lifted>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @"pass_paramref
// CHECK-SAME: () -> !kgen.generator<<index>() -> !kgen.simd<apply(:(index) -> index @some_func, *(0,0)), f32>>
kgen.generator @pass_paramref<T: type>() -> !kgen.param<T> {
  %0 = kgen.param.constant : !kgen.param<T> = <#kgen.unknown : !kgen.param<T>>
  // CHECK: return %0 : !kgen.generator<<index>() -> !kgen.simd<apply(:(index) -> index @some_func, *(0,0)), f32>>
  kgen.return %0 : !kgen.param<T>
}

kgen.generator @some_func(%arg0: index) -> index {
  kgen.return %arg0: index
}

kgen.generator @give_func() -> !kgen.generator<(index) -> index>{
  %0 = kgen.param.constant: (index) -> index = <@some_func>
  kgen.return %0 : !kgen.generator<(index) -> index>
}

// CHECK-LABEL: kgen.func @top
kgen.generator @top() {
  kgen.param.apply func = [() -> !kgen.generator<(index) -> index>: @give_func]()
  // CHECK: () -> !kgen.generator<<index>() -> !kgen.simd<apply(:(index) -> index @some_func, *(0,0)), f32>>
  kgen.call @pass_paramref<:type <index>() -> !kgen.simd<apply(:(index) -> index func, *(0,0)), f32>>() : () -> !kgen.generator<<index>() -> !kgen.simd<apply(:(index) -> index func, *(0,0)), f32>>
  kgen.return
}

// -----

kgen.generator @f() {
  kgen.return
}

kgen.struct.generator @ParamType<p> = !pop.array<p, i8> {}
#paramtype = #kgen.type<typevalue<:!kgen.type #kgen.genref<@ParamType<p>>>, !pop.array<p, i8>> : !kgen.type
#paramtype1 = #kgen.type<typevalue<:!kgen.type #kgen.genref<@ParamType<1>>>, !pop.array<1, i8>> : !kgen.type

kgen.struct.generator @IntType = index {}
#indextype = #kgen.type<typevalue<:!kgen.type #kgen.genref<@IntType>>, index> : !kgen.type

// CHECK-LABEL: kgen.func @"rebind_type,p=1"
kgen.generator @rebind_type<p>(%arg0: !kgen.pointer<array<p, i8>>)
    -> !kgen.pointer<#paramtype> {
  // CHECK-NOT: kgen.rebind %arg0
  %0 = kgen.rebind %arg0 : !kgen.pointer<array<p, i8>> to !kgen.pointer<#paramtype>
  // CHECK-NEXT: constant: pointer<index> = <store_to_mem(1)>
  kgen.param.constant: pointer<#indextype> = <rebind(:pointer<index> store_to_mem(p))>
  // CHECK-NEXT: constant: pointer<array<1, i8>> = <0>
  kgen.param.constant: pointer<#paramtype> = <rebind(:pointer<array<p, i8>> 0)>
  kgen.return %0 : !kgen.pointer<#paramtype>
}

kgen.generator @nonparametric_rebind(%arg0: !kgen.pointer<index>) -> index {
  %0 = kgen.rebind %arg0 : !kgen.pointer<index> to !kgen.pointer<#indextype>
  %1 = pop.load %0 : !kgen.pointer<#indextype>
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.func @try_rebind
kgen.generator @try_rebind(%arg0: !kgen.pointer<array<1, i8>>) {
  kgen.call @rebind_type<1>(%arg0) : (!kgen.pointer<array<1, i8>>) -> !kgen.pointer<#paramtype1>
  kgen.param.apply a = [(!kgen.pointer<index>) -> index: @nonparametric_rebind](store_to_mem(1))
  // CHECK: constant = <1>
  kgen.param.constant = <a>
  kgen.return
}

// -----

kgen.generator @fma(%arg0: index, %arg1: index, %arg2: index) -> index {
  %0 = index.mul %arg1, %arg2
  %1 = index.add %0, %arg0
  kgen.return %1 : index
}

!capture = !kgen.struct<(string, index, (!kgen.pointer<pointer<none>>) capturing -> !kgen.none)>

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() {
  // CHECK: mul i64
  // CHECK: add i64
  %0 = kgen.param.constant: !kgen.struct<(string, index, (!kgen.pointer<none>) capturing -> !kgen.none)> = <#kgen.compile_assembly<current_target(), =llvm, "", false, :(index, index, index) -> (index) @fma>>
  kgen.return
}

// -----

kgen.generator @might_fail<succeed: !kgen.scalar<bool>>() {
  kgen.param.assert <succeed>, "did not succeed"
  kgen.return
}

!capture = !kgen.struct<(string, index, (!kgen.pointer<none>) capturing -> !kgen.none)>

// CHECK-LABEL: @compile_assembly_conditional
kgen.generator export @compile_assembly_conditional() {
  // CHECK-NEXT: <{ "failed {{.*}}", -1, #interp.uninitmem }>
  kgen.param.constant: !capture = <#kgen.compile_assembly<current_target(), =llvm, "", true, :() -> () @might_fail<:!kgen.scalar<bool> false>>>
  // CHECK-NEXT: <{ "{{.*}}", 0, [[CAP_FN:@.*]] }>
  kgen.param.constant: !capture = <#kgen.compile_assembly<current_target(), =llvm, "", true, :() -> () @might_fail<:!kgen.scalar<bool> true>>>
  kgen.return
}

// CHECK: kgen.func [[CAP_FN]]

// -----

// CHECK-NOT: @no_impl
kgen.generator @no_impl() -> index {
  kgen.param.assert <false>, "bad"
  %index0 = kgen.param.constant = <0>
  kgen.return %index0 : index
}

kgen.generator @make_true() -> !kgen.scalar<bool> {
  %0 = kgen.param.constant: scalar<bool> = <true>
  kgen.return %0 : !kgen.scalar<bool>
}

kgen.generator export @conditional_alias() {
  kgen.param.declare value = <cond(apply(:() -> !kgen.scalar<bool> @make_true), 1, apply(:() -> index @no_impl))>
  kgen.return
}

// -----

kgen.generator @call_it(%arg0: !kgen.generator<() -> index>) -> index {
  %0 = kgen.call_indirect %arg0() : () -> index
  kgen.return %0 : index
}

// CHECK-MAIN-LABEL: kgen.func @"give,a=1"
kgen.generator @give<a>() -> index {
  %idx0 = kgen.param.constant = <a>
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func export @apply_expr
kgen.generator export @apply_expr() {
  // CHECK-NEXT: <1>
  kgen.param.constant = <apply(:(!kgen.generator<() -> index>) -> index @call_it, @give<1>)>
  kgen.return
}

// CHECK-LABEL: kgen.func export @apply_op
kgen.generator export @apply_op() {
  // CHECK-NEXT: <2>
  kgen.param.apply x = [(!kgen.generator<() -> index>) -> index: @call_it](@give<2>)
  kgen.param.constant = <x>
  kgen.return
}

// -----

kgen.generator @fwd_type(%arg0: !kgen.type) -> !kgen.type {
  kgen.return %arg0 : !kgen.type
}

kgen.generator @fwd_sig(%arg0: !kgen.generator<<type>(!kgen.param<apply(:(!kgen.type)->!kgen.type @fwd_type, *(0,0))>) -> ()>) -> !kgen.generator<<type>(!kgen.param<apply(:(!kgen.type)->!kgen.type @fwd_type, *(0,0))>) -> ()> {
  kgen.return %arg0 :  !kgen.generator<<type>(!kgen.param<apply(:(!kgen.type)->!kgen.type @fwd_type, *(0,0))>) -> ()>
}

kgen.generator @fn<T: type>(%arg0: !kgen.param<apply(:(!kgen.type) -> !kgen.type @fwd_type, T)>) {
  kgen.return
}

// CHECK-LABEL: kgen.func export @top
kgen.generator export @top() {
  // COM: Just check that the parameter expression can be resolved.
  kgen.param.declare f: <type>(!kgen.param<apply(:(!kgen.type)->!kgen.type @fwd_type, *(0,0))>) -> () = <apply(:(!kgen.generator<<type>(!kgen.param<apply(:(!kgen.type)->!kgen.type @fwd_type, *(0,0))>) -> ()>) -> !kgen.generator<<type>(!kgen.param<apply(:(!kgen.type)->!kgen.type @fwd_type, *(0,0))>) -> ()> @fwd_sig, @fn)>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @conflicting_values
kgen.generator @conflicting_values() {
  // CHECK-NEXT: constant = <1>
  // CHECK-NEXT: constant = <2>
  kgen.param.declare b: scalar<bool> = <true>
  kgen.param.if <b> {
    kgen.param.declare a = <1>
    kgen.param.constant = <a>
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  kgen.param.if <b> {
    kgen.param.declare a = <2>
    kgen.param.constant = <a>
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  kgen.return
}

// -----

kgen.generator @forward_pointer(%arg0: !kgen.pointer<index>) -> !kgen.pointer<index> {
  kgen.return %arg0 : !kgen.pointer<index>
}

// CHECK-LABEL: kgen.func export @load_from_mem
kgen.generator export @load_from_mem() {
  // CHECK-NEXT: <42>
  kgen.param.constant = <load_from_mem(:pointer<index> apply(:(!kgen.pointer<index>) -> !kgen.pointer<index> @forward_pointer, store_to_mem(42)))>
  kgen.return
}

// -----

// Test struct generator elaboration.

// CHECK: #[[TYPEVALUE_INDEX:.+]] = #kgen.type<typevalue<#kgen.instref<@"LinkedList,T=index">>, pointer<none>> : !kgen.type
// CHECK: #[[TYPEVALUE_NONE:.+]] = #kgen.type<typevalue<#kgen.instref<@"LinkedList,T=none">>, pointer<none>> : !kgen.type

// CHECK-LABEL: kgen.struct.instance @"LinkedList,T=index"
// CHECK-SAME:    struct_inst<"LinkedList"[T]<:type index>(data: index, next: #[[TYPEVALUE_INDEX]])>

// CHECK-LABEL: kgen.struct.instance @"LinkedList,T=none"
// CHECK-SAME:    struct_inst<"LinkedList"[T]<:type none>(data: none, next: #[[TYPEVALUE_NONE]])>
kgen.struct.generator @"LinkedList"<T: type> = struct_inst<
  "LinkedList"[T]<:type T>(
    data: [typevalue<T>, !kgen.param<T>],
    next: [typevalue<#kgen.genref<@LinkedList<:type T>>>, pointer<none>]
  )>
{
  kgen.conformance @Boolable {
    kgen.witness "__bool__" : (!kgen.struct<(T, pointer<none>)>) -> i1 = @"LinkedList::__bool__(::LinkedList)"<:type T>
  }
}

kgen.generator @"LinkedList::__bool__(::LinkedList)"<T: type>(%arg0: !kgen.struct<(T, pointer<none>)>) -> i1 {
  %index1 = kgen.param.constant : i1 = <1>
  kgen.return %index1 : i1
}

// CHECK-LABEL: kgen.struct.instance @NonParametric
// CHECK-SAME:    struct_inst<"NonParametric"(data: index)>
kgen.struct.generator @NonParametric = struct_inst<"NonParametric"(data: index)> {
  kgen.conformance @Boolable {
    kgen.witness "__bool__" : (!kgen.struct<(index)>) -> i1 = @"NonParametric::__bool__(::NonParametric)"
  }
}

kgen.generator @"NonParametric::__bool__(::NonParametric)"(%arg0: !kgen.struct<(index)>) -> i1 {
  %index1 = kgen.param.constant : i1 = <1>
  kgen.return %index1 : i1
}

// CHECK-LABEL: kgen.func @"use_boolable,T={{.*}}LinkedList,T=index
// CHECK-NEXT:    kgen.call @"LinkedList::__bool__(::LinkedList),T=index"

// CHECK-LABEL: kgen.func @"use_boolable,T={{.*}}LinkedList,T=none
// CHECK-NEXT:    kgen.call @"LinkedList::__bool__(::LinkedList),T=none"

// CHECK-LABEL: kgen.func @"use_boolable,T={{.*}}NonParametric
// CHECK-NEXT:    kgen.call @"NonParametric::__bool__(::NonParametric)"
kgen.generator @use_boolable<T: type>(%arg: !kgen.param<T>) -> i1 {
  kgen.param.declare traitMethod: (!kgen.param<T>) -> i1  = <#kgen.get_witness<T, @Boolable, "__bool__">>
  %result = kgen.call_param[(!kgen.param<T>) -> i1 : traitMethod](%arg)
  kgen.return %result : i1
}

#linkedlist_index = #kgen.type<typevalue<:!kgen.type #kgen.genref<@LinkedList<:type index>>>, struct<(index, !kgen.pointer<none>)>> : !kgen.type
#linkedlist_none = #kgen.type<typevalue<:!kgen.type #kgen.genref<@LinkedList<:type none>>>, struct<(none, !kgen.pointer<none>)>> : !kgen.type
#nonparametric = #kgen.type<typevalue<:!kgen.type #kgen.genref<@NonParametric>>, struct<(index)>> : !kgen.type

// CHECK-LABEL: kgen.func @gen_structs
kgen.generator @gen_structs(%arg0: !kgen.struct<(index, !kgen.pointer<none>)>, %arg1: !kgen.struct<(none, !kgen.pointer<none>)>, %arg2: !kgen.struct<(index)>) {
  // CHECK-NEXT: kgen.call {{.*}}LinkedList,T=index
  kgen.call @use_boolable<:type #linkedlist_index>(%arg0) : (!kgen.struct<(index, !kgen.pointer<none>)>) -> i1
  // CHECK-NEXT: kgen.call {{.*}}LinkedList,T=none
  kgen.call @use_boolable<:type #linkedlist_none>(%arg1) : (!kgen.struct<(none, !kgen.pointer<none>)>) -> i1
  // CHECK-NEXT: kgen.call {{.*}}NonParametric
  kgen.call @use_boolable<:type #nonparametric>(%arg2) : (!kgen.struct<(index)>) -> i1
  kgen.return
}

// -----

// Non-concrete witness table entries.

// CHECK-LABEL: kgen.struct.instance @"Contrived,n=3"
// CHECK-LABEL: kgen.struct.instance @"Contrived,n=7"
kgen.struct.generator @Contrived<n: index> = struct_inst<"Contrived"(data: index)> {
  kgen.conformance @Fooable {
    kgen.witness "__foo__" : (!kgen.scalar<index>) -> !kgen.simd<apply(:(index) -> index @addOne, n), index> = @"Contrived::__foo__(::Int)"<apply(:(index) -> index @addOne, n)>
  }
}

kgen.struct.generator @AnotherContrived<n: index> = struct_inst<"AnotherContrived"(data: index)> {
  kgen.conformance @Fooable {
    kgen.witness "__foo__" : (!kgen.scalar<index>) -> !kgen.simd<apply(:(index) -> index @addOne, n), index> = @"Contrived::__foo__(::Int)"<apply(:(index) -> index @addOne, n)>
  }
}

kgen.generator @"addOne"(%arg0: index) -> index {
  %index1 = kgen.param.constant : index = <1>
  %index2 = index.add %arg0, %index1
  kgen.return %index2 : index
}

// CHECK-LABEL: kgen.func @"Contrived::__foo__(::Int),n=4"
// CHECK-LABEL: kgen.func @"Contrived::__foo__(::Int),n=8"
kgen.generator @"Contrived::__foo__(::Int)"<n: index>(%arg0: !kgen.scalar<index>) -> !kgen.simd<n, index> {
  %result = pop.simd.splat %arg0 : !kgen.simd<n, index>
  kgen.return %result : !kgen.simd<n, index>
}

#Contrived3 = #kgen.type<typevalue<:!kgen.type #kgen.genref<@Contrived<:index 3>>>, struct<(index)>> : !kgen.type
#Contrived7 = #kgen.type<typevalue<:!kgen.type #kgen.genref<@Contrived<:index 7>>>, struct<(index)>> : !kgen.type
#AnotherContrivedN1 = #kgen.type<typevalue<:!kgen.type #kgen.genref<@AnotherContrived<:index sub(n, 1)>>>, struct<(index)>> : !kgen.type

// CHECK-LABEL: kgen.func @"use_fooable,T={{.*}},n=4"
// CHECK:    kgen.call @"Contrived::__foo__(::Int),n=4"

// CHECK-LABEL: kgen.func @"use_fooable,T={{.*}},n=8"
// CHECK:    kgen.call @"Contrived::__foo__(::Int),n=8"
kgen.generator @use_fooable<T: type, n: index>(%arg: !kgen.param<T>) -> !kgen.simd<n, index> {
  %const = kgen.param.constant : !kgen.scalar<index> = <7>
  kgen.param.declare traitMethod: (!kgen.scalar<index>) -> !kgen.simd<n, index>  = <#kgen.get_witness<T, @Fooable, "__foo__">>
  %result = kgen.call_param[(!kgen.scalar<index>) -> !kgen.simd<n, index> : traitMethod](%const)
  kgen.param.declare traitMethod2: (!kgen.scalar<index>) -> !kgen.simd<n, index>  = <#kgen.get_witness<#AnotherContrivedN1, @Fooable, "__foo__">>
  %result2 = kgen.call_param[(!kgen.scalar<index>) -> !kgen.simd<n, index> : traitMethod2](%const)
  %result3 = pop.add %result, %result2 : !kgen.simd<n, index>
  kgen.return %result3 : !kgen.simd<n, index>
}

// CHECK-LABEL: kgen.func @gen_structs
kgen.generator @gen_structs(%arg0: !kgen.struct<(index)>) {
  // CHECK-NEXT: kgen.call {{.*}}Contrived,n=3
  kgen.call @use_fooable<:type #Contrived3, 4>(%arg0) : (!kgen.struct<(index)>) -> !kgen.simd<4, index>
  // CHECK-NEXT: kgen.call {{.*}}Contrived,n=7
  kgen.call @use_fooable<:type #Contrived7, 8>(%arg0) : (!kgen.struct<(index)>) -> !kgen.simd<8, index>

  // CHECK: kgen.param.constant: string = <";
  // CHECK-SAME: use_fooable,T={{.*}}Contrived,n=3
  // CHECK-SAME: ret <4 x i64> splat (i64 14)
  %0 = kgen.compile_offload<current_target(), 2, "", "",
                            :(!kgen.struct<(index)>) -> !kgen.simd<4, index> @use_fooable<:type #Contrived3, 4>>
                            : !kgen.struct<(string, index)>
  // CHECK: kgen.param.constant: string = <";
  // CHECK-SAME: use_fooable,T={{.*}}Contrived,n=7
  // CHECK-SAME: ret <8 x i64> splat (i64 14)
  %1 = kgen.compile_offload<current_target(), 2, "", "",
                            :(!kgen.struct<(index)>) -> !kgen.simd<8, index> @use_fooable<:type #Contrived7, 8>>
                            : !kgen.struct<(string, index)>
  kgen.return
}

// -----

// Intermixing of function apply and struct instantiation.

// CHECK-MAIN: kgen.func @"get_array_type,T=index"
// CHECK-MAIN-NEXT: kgen.param.constant: type = <array<[[SIZEOF:.+]], index>>
kgen.generator @get_array_type<T: type>() -> !kgen.type {
  %0 = kgen.param.constant: type = <array<get_sizeof(T, current_target()), index>>
  kgen.return %0 : !kgen.type
}

// CHECK-MAIN: kgen.struct.instance @"WeirdStruct,T=index"
// CHECK-MAIN-SAME: struct_inst<"WeirdStruct"(data: array<[[SIZEOF]], index>)>
kgen.struct.generator @WeirdStruct<T: type> = struct_inst<"WeirdStruct"(data: typevalue<apply(:() -> !kgen.type @get_array_type<:type T>)>)>

kgen.generator @get_struct_field_types<T: type>() -> !kgen.param_list<type> {
  %param_list = kgen.param.constant: param_list<type> = <#kgen.struct_field_types<T>>
  kgen.return %param_list : !kgen.param_list<type>
}

kgen.generator @get_struct_field_names<T: type>() -> !kgen.param_list<string> {
  %param_list = kgen.param.constant: param_list<string> = <#kgen.struct_field_names<T>>
  kgen.return %param_list : !kgen.param_list<string>
}

kgen.generator @get_struct_field_index_by_name<T: type>() -> index {
  %index = kgen.param.constant: index = <#kgen.struct_field_index_by_name<T, "data">>
  kgen.return %index : index
}

kgen.generator @get_struct_field_type_by_name<T: type>() -> !kgen.type {
  %type = kgen.param.constant: type = <#kgen.struct_field_type_by_name<T, "data">>
  kgen.return %type : !kgen.type
}

#weird_struct = #kgen.type<typevalue<:type #kgen.genref<@WeirdStruct<:type index>>>, struct<(apply(:() -> !kgen.type @get_array_type<:type index>))>> : !kgen.type

// CHECK: kgen.func @gen_structs
kgen.generator @gen_structs() {
  // CHECK-NEXT: kgen.param.constant: param_list<type> = <[array<[[SIZEOF:.+]], index>]>
  kgen.param.apply test_struct_field_types = [() -> !kgen.param_list<type>: @get_struct_field_types<:type #weird_struct>]()
  kgen.param.constant: param_list<type> = <test_struct_field_types>
  // CHECK-NEXT: kgen.param.constant: param_list<string> = <["data"]>
  kgen.param.apply test_struct_field_names = [() -> !kgen.param_list<string>: @get_struct_field_names<:type #weird_struct>]()
  kgen.param.constant: param_list<string> = <test_struct_field_names>
  // CHECK-NEXT: kgen.param.constant = <0>
  kgen.param.apply test_struct_field_index_by_name = [() -> index: @get_struct_field_index_by_name<:type #weird_struct>]()
  kgen.param.constant: index = <test_struct_field_index_by_name>
  // CHECK-NEXT: kgen.param.constant: type = <array<[[SIZEOF:.+]], index>>
  kgen.param.apply test_struct_field_type_by_name = [() -> !kgen.type: @get_struct_field_type_by_name<:type #weird_struct>]()
  kgen.param.constant: type = <test_struct_field_type_by_name>

  // Test is_struct_type returns true for Mojo struct types
  // CHECK-NEXT: kgen.param.constant: i1 = <1>
  kgen.param.constant: i1 = <#kgen.is_struct_type<#weird_struct>>

  // Test is_struct_type returns false for MLIR primitive types (e.g., index)
  // CHECK-NEXT: kgen.param.constant: i1 = <0>
  kgen.param.constant: i1 = <#kgen.is_struct_type<index>>

  // Test get_base_type_name returns the base type name for parameterized types
  // CHECK-NEXT: kgen.param.constant: string = <"WeirdStruct">
  kgen.param.constant: string = <#kgen.get_base_type_name<#weird_struct>>

  // Test get_base_type_name returns "<unknown>" for MLIR primitive types
  // CHECK-NEXT: kgen.param.constant: string = <"<unknown>">
  kgen.param.constant: string = <#kgen.get_base_type_name<index>>

  kgen.return
}

// CHECK-NOT: @exported_parametric
kgen.generator export @exported_parametric<param>() {
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//
// Additional struct field reflection tests
//===----------------------------------------------------------------------===//

// Test multiple fields: verify correct field ordering and index lookups.
kgen.struct.generator @MultiFieldStruct<T: type> = struct_inst<
  "MultiFieldStruct"[T]<:type T>(
    first: index,
    second: [typevalue<T>, !kgen.param<T>],
    third: i32,
    fourth: f64
  )>

#multi_field_struct = #kgen.type<typevalue<:!kgen.type #kgen.genref<@MultiFieldStruct<:type i64>>>, struct<(index, i64, i32, f64)>> : !kgen.type

// CHECK: kgen.func @multi_field_struct_tests
kgen.generator @multi_field_struct_tests() {
  // Test field types are returned in order
  // CHECK-NEXT: kgen.param.constant: param_list<type> = <[index, i64, i32, f64]>
  kgen.param.constant: param_list<type> = <#kgen.struct_field_types<#multi_field_struct>>

  // Test field names are returned in order
  // CHECK-NEXT: kgen.param.constant: param_list<string> = <["first", "second", "third", "fourth"]>
  kgen.param.constant: param_list<string> = <#kgen.struct_field_names<#multi_field_struct>>

  // Test field indices are correct
  // CHECK-NEXT: kgen.param.constant = <0>
  kgen.param.constant: index = <#kgen.struct_field_index_by_name<#multi_field_struct, "first">>
  // CHECK-NEXT: kgen.param.constant = <1>
  kgen.param.constant: index = <#kgen.struct_field_index_by_name<#multi_field_struct, "second">>
  // CHECK-NEXT: kgen.param.constant = <2>
  kgen.param.constant: index = <#kgen.struct_field_index_by_name<#multi_field_struct, "third">>
  // CHECK-NEXT: kgen.param.constant = <3>
  kgen.param.constant: index = <#kgen.struct_field_index_by_name<#multi_field_struct, "fourth">>

  // Test struct_field_type_by_name returns rebounded type for parametric field
  // CHECK-NEXT: kgen.param.constant: type = <i64>
  kgen.param.constant: type = <#kgen.struct_field_type_by_name<#multi_field_struct, "second">>

  kgen.return
}

// -----

// SourceLoc interpretation

kgen.generator @wrap_source_loc_param<depth: index>() -> !kgen.string always_inline_no_debug {
  %line, %col, %fileName = kgen.source_loc[depth] loc("source":0:0)
  kgen.return %fileName : !kgen.string
}

kgen.generator @level_zero() -> !kgen.string always_inline_no_debug {
  %0 = kgen.call @wrap_source_loc_param<1>() : () -> !kgen.string loc("first_call":1:2)
  kgen.return %0 : !kgen.string
}

kgen.generator @level_one() -> !kgen.string always_inline_no_debug {
  %0 = kgen.call @level_zero() : () -> !kgen.string loc("second_call":3:4)
  kgen.return %0 : !kgen.string
}

#sp = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#loc3 = loc(fused<#sp>["level -3":2:2])
#loc2 = loc(callsite(#loc3 at fused<#sp>["level -2":1:1]))
#loc1 = loc(callsite(#loc2 at fused<#sp>["level -1":0:0]))

kgen.generator @inlined_source_loc_3() -> !kgen.string always_inline_no_debug {
  kgen.param.declare depth = <0>
  %line, %col, %fileName = kgen.source_loc[add(depth, -3)] loc(#loc1)
  kgen.return %fileName : !kgen.string
}

kgen.generator @inlined_source_loc_1() -> !kgen.string always_inline_no_debug {
  kgen.param.declare depth = <0>
  %line, %col, %fileName = kgen.source_loc[add(depth, -1)] loc(#loc1)
  kgen.return %fileName : !kgen.string
}

// CHECK-LABEL: kgen.func @test_wrap_source_loc_param
kgen.generator @test_wrap_source_loc_param() -> !kgen.string always_inline_no_debug {
  // CHECK: kgen.param.constant: string = <"second_call">
  // CHECK-NOT: kgen.call
  kgen.param.apply a = [() -> !kgen.string: @level_one]() loc("param_apply":5:6)
  %0 = kgen.param.constant : !kgen.string = <a>
  // CHECK: kgen.param.constant: string = <"level -3">
  // CHECK-NOT: kgen.call
  kgen.param.apply b = [() -> !kgen.string: @inlined_source_loc_3]() loc("param_apply":7:8)
  %1 = kgen.param.constant : !kgen.string = <b>
  // CHECK: kgen.param.constant: string = <"level -1">
  // CHECK-NOT: kgen.call
  kgen.param.apply c = [() -> !kgen.string: @inlined_source_loc_1]() loc("param_apply":9:10)
  %2 = kgen.param.constant : !kgen.string = <c>
  kgen.return %2 : !kgen.string
}

// -----

kgen.generator @get_hello_address() -> !kgen.pointer<scalar<si8>> {
  %0 = kgen.param.constant: string = <"hello">
  %1 = pop.string.address %0
  kgen.return %1 : !kgen.pointer<scalar<si8>>
}

kgen.generator @use_string_address<a: !kgen.pointer<scalar<si8>>>() {
  kgen.return
}

// CHECK-LABEL: kgen.func export @entry
kgen.generator export @entry() {
  kgen.param.declare from_poc: !kgen.pointer<scalar<si8>> = <string_address("world")>
  kgen.param.apply from_func = [() -> !kgen.pointer<scalar<si8>>: @get_hello_address]()
  // CHECK: kgen.call {{.*}}#interp.memref<{{.*}}world
  kgen.call @use_string_address<:!kgen.pointer<scalar<si8>> from_poc>() : () -> ()
  // COM: Ensure the mem slot for "world" does not appear again. This means the
  // interpreter state was correctly reset after evaluating POC::string_address.
  // CHECK-NOT: world
  kgen.call @use_string_address<:!kgen.pointer<scalar<si8>> from_func>() : () -> ()
  kgen.return
}

// -----

kgen.generator @bind_one_index<f: !kgen.generator<<index> index>>() -> index {
  %0 = kgen.param.constant: index = <#kgen.bind_params<:!kgen.generator<<index> index> f, 4>>
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @"bind_two_index,f=#kgen.gen<to_builtin(:scalar<index> add(from_builtin(*(0,0)), from_builtin(*(0,1))))>"
// CHECK-NEXT:   kgen.param.constant = <20>
// CHECK-NEXT:   kgen.param.constant = <6>

// CHECK-LABEL: kgen.func @"bind_two_index,f=#kgen.gen<to_builtin(:scalar<index> div(from_builtin(*(0,0)), from_builtin(*(0,1))))>"
// CHECK-NEXT:   kgen.param.constant = <4>
// CHECK-NEXT:   kgen.param.constant = <2>

kgen.generator @bind_two_index<f: !kgen.generator<<index, index> index>>() {
  %0 = kgen.param.constant: index = <apply(
    :!kgen.generator<() -> index> @bind_one_index<
        :!kgen.generator<<index> index> #kgen.bind_params<:!kgen.generator<<index, index> index> f, 16>>)>

  %1 = kgen.param.constant: index = <apply(
    :!kgen.generator<() -> index> @bind_one_index<
        :!kgen.generator<<index> index> #kgen.bind_params<:!kgen.generator<<index, index> index> f, ?, 2>>)>
  kgen.return
}

// CHECK-LABEL: kgen.func export @entry
kgen.generator export @entry() {
  kgen.param.declare myAdd: !kgen.generator<<index, index> index> = <#kgen.gen<add(*(0,0), *(0,1))>>
  // CHECK-NEXT: kgen.call @"bind_two_index,f=#kgen.gen<to_builtin(:scalar<index> add(from_builtin(*(0,0)), from_builtin(*(0,1))))>"()
  kgen.call @bind_two_index<:!kgen.generator<<index, index> index> myAdd>() : () -> ()
  kgen.param.declare myDiv: !kgen.generator<<index, index> index> = <#kgen.gen<div(*(0,0), *(0,1))>>
  // CHECK-NEXT: kgen.call @"bind_two_index,f=#kgen.gen<to_builtin(:scalar<index> div(from_builtin(*(0,0)), from_builtin(*(0,1))))>"()
  kgen.call @bind_two_index<:!kgen.generator<<index, index> index> myDiv>() : () -> ()
  kgen.return
}

// -----

kgen.generator @bind_one_index<f: !kgen.generator<<index> index>>() -> index {
  %0 = kgen.param.constant: index = <#kgen.bind_params<:!kgen.generator<<index> index> f, 4>>
  kgen.return %0 : index
}

kgen.generator @bind_one_index_outer<f: !kgen.generator<<index> !kgen.generator<<index> index>>>() -> index {
  %0 = kgen.param.constant: index = <apply(
    :!kgen.generator<() -> index> @bind_one_index<
        :!kgen.generator<<index> index> #kgen.bind_params<:!kgen.generator<<index> !kgen.generator<<index> index>> f, 8>>)>
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @entry
kgen.generator export @entry() {
  kgen.param.declare myCurriedAdd: !kgen.generator<<index> !kgen.generator<<index> index>> = <#kgen.gen<#kgen.gen<add(*(0,0), *(1,0))>>>
  // CHECK-NEXT: kgen.param.constant = <12>
  %0 = kgen.param.constant = <apply(
    :!kgen.generator<() -> index> @bind_one_index_outer<
        :!kgen.generator<<index> !kgen.generator<<index> index>> myCurriedAdd>)>
  kgen.param.declare myCurriedDiv: !kgen.generator<<index> !kgen.generator<<index> index>> = <#kgen.gen<#kgen.gen<div(*(0,0), *(1,0))>>>
  // CHECK-NEXT: kgen.param.constant = <0>
  %1 = kgen.param.constant = <apply(
    :!kgen.generator<() -> index> @bind_one_index_outer<
        :!kgen.generator<<index> !kgen.generator<<index> index>> myCurriedDiv>)>
  kgen.param.declare myReversedCurriedDiv: !kgen.generator<<index> !kgen.generator<<index> index>> = <#kgen.gen<#kgen.gen<div(*(1,0), *(0,0))>>>
  // CHECK-NEXT: kgen.param.constant = <2>
  %2 = kgen.param.constant = <apply(
    :!kgen.generator<() -> index> @bind_one_index_outer<
        :!kgen.generator<<index> !kgen.generator<<index> index>> myReversedCurriedDiv>)>
  kgen.return
}

// -----

// COM: Type conformance check

kgen.struct.generator @S0 = struct_inst<"S0" memoryOnly>{
  kgen.conformance @"A" {}
  kgen.conformance @"B" {}
}
kgen.struct.generator @S1 = struct_inst<"S1" memoryOnly>{}

// CHECK-LABEL: kgen.func @"conformance_check,T=[typevalue<#kgen.instref<{{.*}}S0>>, struct<() memoryOnly>]"() always_inline
// CHECK-NEXT: "sink-conforms-to"() : () -> ()
// CHECK-NEXT: kgen.return

// CHECK-LABEL: kgen.func @"conformance_check,T=[typevalue<#kgen.instref<{{.*}}S1>>, struct<() memoryOnly>]"() always_inline
// CHECK-NEXT: "sink-does-not-conform-to"() : () -> ()
// CHECK-NEXT: kgen.return
kgen.generator @conformance_check<T: type>() always_inline {
  kgen.param.if <#kgen.type_conforms_to_trait<#kgen.param.decl.ref<"T"> : !kgen.type, #kgen.type<typevalue<#kgen.trait_ref<[@"A", @"B"]>>, type> : !kgen.type>> {
    "sink-conforms-to"() : () -> ()
    kgen.param.yield
  } else {
    "sink-does-not-conform-to"() : () -> ()
    kgen.param.yield
  }
  kgen.return
}

kgen.generator export @foo() -> ()  {
  kgen.call @conformance_check<:type #kgen.type<typevalue<#kgen.genref<@S0>>, struct<() memoryOnly>>>() : () -> ()
  kgen.call @conformance_check<:type #kgen.type<typevalue<#kgen.genref<@S1>>, struct<() memoryOnly>>>() : () -> ()
  kgen.return
}

// -----

// COM: data_to_str elaboration (two versions tested during Int->SIMD unification)

// CHECK-LABEL: kgen.func @test_data_to_str
kgen.generator @test_data_to_str() {
  kgen.param.declare s1: struct<(pointer<none>, index)> = <{ string_address("hello"), 5 }>
  // CHECK-NEXT: = kgen.param.constant: string = <"hello">
  %0 = kgen.param.constant: string = <data_to_str(:struct<(pointer<none>, index)> s1, [])>
  // CHECK-NEXT: kgen.return
  kgen.return
}

// CHECK-LABEL: kgen.func @test_data_to_str_concat
kgen.generator @test_data_to_str_concat() {
  kgen.param.declare s1: struct<(pointer<none>, index)> = <{ string_address("hello"), 5 }>
  kgen.param.declare s2: struct<(pointer<none>, index)> = <{ string_address(" world"), 6 }>
  // CHECK-NEXT: = kgen.param.constant: string = <"hello world">
  %0 = kgen.param.constant: string = <data_to_str(:struct<(pointer<none>, index)> s1, [s2])>
  // CHECK-NEXT: kgen.return
  kgen.return
}

// CHECK-LABEL: kgen.func @test_data_to_str_scalar_index
kgen.generator @test_data_to_str_scalar_index() {
  kgen.param.declare s1: struct<(pointer<none>, scalar<index>)> = <{ string_address("hello"), 5 }>
  // CHECK-NEXT: = kgen.param.constant: string = <"hello">
  %0 = kgen.param.constant: string = <data_to_str(:struct<(pointer<none>, scalar<index>)> s1, [])>
  // CHECK-NEXT: kgen.return
  kgen.return
}

// CHECK-LABEL: kgen.func @test_data_to_str_concat_scalar_index
kgen.generator @test_data_to_str_concat_scalar_index() {
  kgen.param.declare s1: struct<(pointer<none>, scalar<index>)> = <{ string_address("hello"), 5 }>
  kgen.param.declare s2: struct<(pointer<none>, scalar<index>)> = <{ string_address(" world"), 6 }>
  // CHECK-NEXT: = kgen.param.constant: string = <"hello world">
  %0 = kgen.param.constant: string = <data_to_str(:struct<(pointer<none>, scalar<index>)> s1, [s2])>
  // CHECK-NEXT: kgen.return
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//
// Export name tests
//===----------------------------------------------------------------------===//

// COM: Static linkage name on a non-parametric generator.  The elaborator
// COM: resolves the linkage name and renames the function.

// CHECK-LABEL: kgen.func export @my_static_export()
// CHECK-NOT: linkageName
kgen.generator export @static_export_test() attributes {linkageName = #kgen.linkage_name<"my_static_export" : !kgen.string, false>} {
  kgen.return
}

// -----

// COM: get_linkage_name on a generator with a static linkageName.

kgen.generator @static_ln_gen() attributes {linkageName = #kgen.linkage_name<"my_export" : !kgen.string, false>} {
  kgen.return
}

// CHECK-LABEL: kgen.func export @test_get_static_linkage_name
kgen.generator export @test_get_static_linkage_name() {
  // CHECK-NEXT: constant: string = <"my_export">
  kgen.param.constant: string = <#kgen.get_linkage_name<current_target(), #kgen.symbol.constant<@static_ln_gen> : !kgen.generator<() -> ()>>>
  kgen.return
}

// -----

// COM: A call reference is updated when the callee generator has a linkage name.

// CHECK-LABEL: kgen.func @callee_linkage()
kgen.generator @"callee::mangled"() attributes {linkageName = #kgen.linkage_name<"callee_linkage" : !kgen.string, false>} {
  kgen.return
}

// CHECK-LABEL: kgen.func @caller_of_linkage
kgen.generator @caller_of_linkage() {
  // CHECK-NEXT: kgen.call @callee_linkage()
  kgen.call @"callee::mangled"() : () -> ()
  kgen.return
}

// -----

// COM: A non-C-exported generator with a non-identifier linkage name is fine
// COM: (no C identifier validation).

// CHECK-LABEL: kgen.func @"has-dashes"()
// CHECK-NOT: linkageName
kgen.generator @"non_c_export::ok"() attributes {linkageName = #kgen.linkage_name<"has-dashes" : !kgen.string, false>} {
  kgen.return
}

// -----

// COM: A C-exported generator with underscores and digits (valid C identifier)
// COM: is accepted.

// CHECK-LABEL: kgen.func export @_leading_underscore_123() cabi
// CHECK-NOT: linkageName
kgen.generator export @"c_export::underscores"() cabi attributes {linkageName = #kgen.linkage_name<"_leading_underscore_123" : !kgen.string, false>} {
  kgen.return
}

// -----

// COM: Breaking `@score`'s self-recursion SCC while the node is suspended on
// COM: a blocker (`@scorer_table`, a separate pending SCC) dropped the
// COM: blocker registration and orphaned the node's dependency count,
// COM: deadlocking elaboration with no diagnosable recursion.

// CHECK-DAG: kgen.func @scorer_table
// CHECK-DAG: kgen.func export @score

kgen.generator @scorer_table() -> index {
  %0 = kgen.call @scorer_table() : () -> index
  kgen.return %0 : index
}

kgen.generator export @score() -> index {
  %0 = kgen.call @score() : () -> index
  %fp = kgen.param.constant: () -> index = <@scorer_table>
  kgen.return %0 : index
}

// -----

// Parameter substitution must re-fold `param.identical`: it is symbolic inside
// the generator, but has to decide once the operands are bound at the call site.
// This works without any per-evaluator handling because the generic
// sub-element rebuild re-runs `ParamIdenticalAttr::get`, which folds.

// CHECK-LABEL: kgen.func @"identical_operands,T=i32,U=i32"
// CHECK: kgen.param.constant: scalar<bool> = <true>
// CHECK-LABEL: kgen.func @"identical_operands,T=i32,U=i64"
// CHECK: kgen.param.constant: scalar<bool> = <false>
kgen.generator @identical_operands<T: type, U: type>() {
  kgen.param.constant: scalar<bool> = <identical(:type T, U)>
  kgen.return
}

// CHECK-LABEL: kgen.func @identical_same
kgen.generator @identical_same() {
  kgen.call @identical_operands<:type i32, :type i32>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @identical_different
kgen.generator @identical_different() {
  kgen.call @identical_operands<:type i32, :type i64>() : () -> ()
  kgen.return
}
