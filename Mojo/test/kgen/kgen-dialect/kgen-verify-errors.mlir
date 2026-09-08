// RUN: kgen-opt -allow-unregistered-dialect %s -verify-parameters -verify-diagnostics -split-input-file

// This tests verification errors which are not enabled in production builds
// UNSUPPORTED: production

// expected-error @below {{type of index reference #kgen.param.index.ref<0, 0> : index does not match parameter type 'ui32'}}
// expected-error @below {{'kgen.generator' op reference to parameter "n" with incorrect type 'index'}}
// expected-note @below {{parameter defined with type 'ui32'}}
kgen.generator @scalar_params_verbose<n : ui32>(%x : !pop.array<n, scalar<invalid>>) {
  kgen.return
}

// -----

kgen.func @entry() {
  // expected-error @below {{invalid symbol use within this operator}}
  // expected-error @below {{@undefined does not reference a KGEN declaration}}
  kgen.call @undefined() : () -> ()
  kgen.return
}

// -----

module @nested_nondecl {
}

kgen.func @entry() {
  // expected-error @below {{invalid symbol use within this operator}}
  // expected-error @below {{@nested_nondecl::@undefined does not reference a KGEN declaration}}
  kgen.call @nested_nondecl::@undefined() : () -> ()
  kgen.return
}

// -----

kgen.generator @g1(%x : i32) {
  // expected-error @below {{invalid symbol use within this operator}}
  // expected-error @below {{symbol use has 1 argument but @g2 expects 0}}
  kgen.call @g2(%x) : (i32) -> ()
  kgen.return
}

// expected-note @below {{@g2 declared here}}
kgen.generator @g2() {
  kgen.return
}

// -----

// expected-note @below {{@only_returns declared here}}
kgen.generator @only_returns<p1>() {
  kgen.return
}

kgen.func @test_only_returns() {
  // expected-error @below {{invalid symbol use within this operator}}
  // expected-error @below {{symbol use has 0 input parameters but @only_returns expects 1}}
  kgen.call @only_returns() : () -> ()
  kgen.return
}

// -----

// expected-note @below {{@fn declared here}}
kgen.generator @fn(%a: i1) -> i1 {
  kgen.return %a : i1
}

kgen.func @result_type(%a: i1) {
  // expected-error @below {{invalid symbol use within this operator}}
  // expected-error @below {{symbol use result #0 has type 'f32' but @fn expected type 'i1'}}
  kgen.call @fn(%a) : (i1) -> f32
  kgen.return
}

// -----

// expected-note @below {{@callee declared here}}
kgen.generator @callee<DT: dtype>(%x: !kgen.scalar<DT>) {
  kgen.return
}

kgen.generator @caller<DT: dtype>(%arg0: !kgen.scalar<DT>) {
  // expected-error @below {{invalid symbol use within this operator}}
  // expected-error @below {{symbol use argument #0 has type '!kgen.scalar<DT>' but @callee expected type '!kgen.scalar<f64>'}}
  kgen.call @callee<:dtype f64>(%arg0) : (!kgen.scalar<DT>) -> ()
  kgen.return
}

// -----

kgen.generator @takeUnary
  <unaryFn: <dtype>(!kgen.scalar<*(0,0)>) -> !kgen.scalar<*(0,0)>>() {
  kgen.return
}

kgen.func @doubleExample(%arg0: !kgen.scalar<si32>) -> !kgen.scalar<si32> {
  %0 = pop.add %arg0, %arg0: !kgen.scalar<si32>
  kgen.return %0 : !kgen.scalar<si32>
}

kgen.generator @test_region() {
  // expected-error @below {{invalid symbol use within this operator}}
  // expected-error @+1 {{caller input parameter #0 has type}}
  kgen.call @takeUnary<:(!kgen.scalar<si32>) -> !kgen.scalar<si32> @doubleExample>() : () -> ()
  kgen.return
}

// -----

kgen.generator @takeFn<fn: () -> ()>() {
  kgen.return
}
kgen.generator @test() {
  // expected-error @below {{invalid symbol use within this operator}}
  // expected-error @+1 {{@missing does not reference a KGEN declaration}}
  kgen.call @takeFn<:()->() @missing>() : () -> ()
  kgen.return
}

// -----

kgen.generator @takeUnary
  <unaryFn: (!kgen.scalar<si32>) -> !kgen.scalar<si32>>() {
  kgen.return
}

// expected-note @below {{@unary declared here}}
kgen.func @unary(%arg0: !kgen.scalar<f32>) -> !kgen.scalar<f32> {
  kgen.return %arg0 : !kgen.scalar<f32>
}

kgen.generator @test1() {
  // expected-error @below {{invalid symbol use within this operator}}
  // expected-error @below {{symbol use argument #0 has type '!kgen.scalar<si32>' but @unary expected type '!kgen.scalar<f32>'}}
  kgen.call @takeUnary<:(!kgen.scalar<si32>) -> !kgen.scalar<si32> @unary>() : () -> ()
  kgen.return
}

// -----

kgen.generator @takeUnary
  <unaryFn: (!kgen.scalar<si32>) -> !kgen.scalar<si32>>() {
  kgen.return
}

// expected-note @below {{@unary2 declared here}}
kgen.generator @unary2<dt: dtype>(%arg0: !kgen.scalar<si32>) -> !kgen.scalar<si32> {
  kgen.return %arg0 : !kgen.scalar<si32>
}

kgen.generator @test2() {
  // expected-error @below {{invalid symbol use within this operator}}
  // expected-error @below {{symbol use has 0 input parameters but @unary2 expects 1}}
  kgen.call @takeUnary<:(!kgen.scalar<si32>) -> !kgen.scalar<si32> @unary2>() : () -> ()
  kgen.return
}

// -----

// expected-note @+1 {{callee declared here}}
kgen.generator @callee(%owned: !kgen.pointer<i32> mut) {
  kgen.return
}

kgen.generator @caller(%arg: !kgen.pointer<i32> owned) {
  // Ok
  kgen.call @callee(%arg) : (!kgen.pointer<i32> mut) -> ()

  // expected-error @below {{invalid symbol use within this operator}}
  // expected-error @+1 {{symbol use argument #0 has convention owned but @callee expected convention mut}}
  kgen.call @callee(%arg) : (!kgen.pointer<i32> owned) -> ()
  kgen.return
}

// -----

kgen.generator @bad_index_ref() {
  // expected-error @below {{index reference has no contextual signature}}
  kgen.param.declare a = <*(0,0)>
  kgen.return
}

// -----

kgen.generator @test_cache() {
  kgen.param.declare p1: !kgen.generator<<index>type> = <#kgen.gen<simd<*(0,0), f32>>>
  // expected-error @below {{index reference has no contextual signature}}
  kgen.param.declare p2: type = <#kgen.type<index, simd<*(0,0), f32>>>
  kgen.return
}

// -----

kgen.generator @bad_index_ref() {
  // expected-error @below {{index reference depth 2 exceeds depth of contextual signatures: 1}}
  kgen.param.declare p1: <index>(!pop.array<*(2,0), i32>) -> () = <*?>
  kgen.return
}

// -----

// expected-error-re @below {{index reference 1 is out of bounds: referenced signature has 1 input parameters}}
kgen.generator @bad_index_ref<fn: <index>(!pop.array<*(0,1), i32>) -> ()>() {
  kgen.return
}

// -----

// expected-error @below {{type of index reference #kgen.param.index.ref<0, 0> : index does not match parameter type 'i32'}}
kgen.generator @bad_index_ref<fn: <i32, !pop.array<*(0,0), i32>>() -> ()>() {
  kgen.return
}

// -----

kgen.generator @two_params<a, b>() {
  // expected-error @below {{invalid symbol use within this operator}}
  // expected-error @below {{generator type expects 2 parameters but got bindings for 1}}
  kgen.param.declare f: <index, index>() -> () = <@two_params<?>>
  kgen.return
}

