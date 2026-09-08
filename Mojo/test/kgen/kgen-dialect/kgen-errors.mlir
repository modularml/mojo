// RUN: kgen-opt -allow-unregistered-dialect %s -verify-parameters -verify-diagnostics -split-input-file

kgen.generator @test() {
  // expected-error @+1 {{invalid use of parameter with no declaration "p"}}
  "someop" () {
    attr = #kgen.param.decl.ref<"p"> : i1
  } : () -> ()
  kgen.return
}

// -----

"someop" () {
  // expected-error @+1 {{shl must have two operands}}
  use1 = #kgen.param.expr<shl, 1 : si32, 2 : si32, 3 : si32>
} : () -> ()

// -----

"someop" () {
  // expected-error @+1 {{operand type mismatch}}
  use1 = #kgen.param.expr<shl, 1 : si32, 2 : ui32>
} : () -> ()

// -----

// expected-error @+1 {{'kgen.param.constant' shl must have two operands}}
%0 = kgen.param.constant = <shl(p1, p2, p3)>

// -----

// expected-error @+1 {{'kgen.param.constant' unknown expression invalid_op}}
%0 = kgen.param.constant = <invalid_op(p1, p2, p3)>

// -----

// expected-error @+1 {{operator requires an index or integer type}}
%0 = kgen.param.constant: f32 = <shl(1., 2.)>

// -----

// expected-error @+1 {{integer literal not valid for specified type}}
kgen.param.constant: !kgen.dtype = <mul(1, 4)>

// -----

kgen.generator @foo() {
  // expected-error @+1 {{invalid use of parameter with no declaration "f32"}}
  kgen.param.constant: i8 = <f32>
  kgen.return
}

// -----

// expected-error @+2 {{attribute type different than expected: expected '!kgen.dtype', but got 'index'}}
kgen.generator @scalar_params_verbose<n>(%x :
           !kgen.scalar<#kgen.param.decl.ref<"n"> : index>) {
  kgen.return
}

// -----

// expected-error @+1 {{funcTypeGenerator is not self-contained: it references parameter 'abc' by name instead of by index}}
kgen.generator @scalar_params_verbose(%x : !kgen.scalar<abc>) {
  kgen.return
}

// -----

kgen.generator @dtype_params() {
  // expected-error @+1 {{invalid use of parameter with no declaration "T"}}
  %y = "someop" () {} : () -> !kgen.scalar<T>
  kgen.return
}

// -----

// expected-error @below {{get_sizeof operator requires two operands}}
"someop"() {a = #kgen.param.expr<get_sizeof, 1>} : () -> ()

// -----

// expected-error @below {{get_sizeof operand 0 should be a type expression}}
"someop"() {a = #kgen.param.expr<get_sizeof, 1, 2> : !kgen.dtype} : () -> ()

// -----

#target = #kgen.target<triple="", arch="", features="", data_layout="", simd_bit_width=128> : !kgen.target

// expected-error @below {{get_sizeof should return index}}
"someop"() {a = #kgen.param.expr<get_sizeof, #kgen.type<i32> : !kgen.type, #target> : !kgen.dtype} : () -> ()

// -----

// expected-error @below {{get_alignof operator requires two operands}}
"someop"() {a = #kgen.param.expr<get_alignof, 1>} : () -> ()

// -----

// expected-error @below {{get_alignof operand 0 should be a type expression}}
"someop"() {a = #kgen.param.expr<get_alignof, 1, 2> : !kgen.dtype} : () -> ()

// -----

#target = #kgen.target<triple="", arch="", features="", data_layout="", simd_bit_width=128> : !kgen.target

// expected-error @below {{get_alignof should return index}}
"someop"() {a = #kgen.param.expr<get_alignof, #kgen.type<i32> : !kgen.type, #target> : !kgen.dtype} : () -> ()

// -----

// expected-error @below {{expected attribute value}}
%0 = kgen.param.constant: i32 = <[:i32]>

// -----

// expected-error @below {{cyclic reference between expressions defining and using parameters}}
kgen.generator @self_cyclic() {
  // Uses r1 and defines r1
  // expected-note @below {{parameter "r1" is defined here, which references itself}}
  kgen.param.declare r1 = <r1>
  kgen.return
}

// -----


// expected-error @below {{cyclic reference between expressions defining and using parameters}}
kgen.generator @mutually_recursive() {
  // Uses r2 and defines r1
  // expected-note @below {{parameter "r1" is defined here, which references the first expression}}
  kgen.param.declare r1 = <r2>

  // Uses r1 and defines r2
  // expected-note @below {{parameter "r2" is defined here, which references the expression:}}
  kgen.param.declare r2 = <r1>

  kgen.return
}

// -----

// expected-error @below {{cyclic reference between expressions defining and using parameters}}
kgen.generator @use_itself() {
  // expected-note @below {{parameter "F" is defined here, which references itself}}
  kgen.param.declare.region F = (){
    kgen.call_param[() -> (): F]()
    kgen.return
  }
  kgen.return
}

// -----

// expected-error @below {{cyclic reference between expressions defining and using parameters}}
kgen.generator @region_cycle() {
  // expected-note @below {{parameter "F" is defined here, which references the first expression}}
  kgen.param.declare.region F = () -> index {
    %0 = kgen.param.constant = <N>
    kgen.return %0 : index
  }
  // expected-note @below {{parameter "N" is defined here, which references the expression:}}
  kgen.param.declare N = <apply(:() -> index F)>
  kgen.return
}

// -----

// expected-error @below {{'kgen.generator' op funcTypeGenerator is not self-contained: it references parameter 'ty2' by name instead of by index}}
kgen.generator @badTypes<ty1 : dtype>(%a : !kgen.scalar<ty2>) {
  kgen.return
}

// -----

// kgen.func isn't allowed to call generators that take parameters,
// but they are allowed to call generators with no parameters.

kgen.generator @hasInputParam<param>() {
  kgen.return
}

kgen.func @test() {  // expected-note {{within 'kgen.func' @test}}
  // expected-error@+1 {{cannot reference generator with input parameters from within a concrete 'kgen.func'}}
  kgen.call @hasInputParam<42>() : () -> ()

  kgen.return
}

// -----

// expected-error @below {{funcTypeGenerator is not self-contained: it references parameter 'dt' by name instead of by index}}
kgen.generator @region_params<r3: () -> !kgen.scalar<dt>>() {
  kgen.return
}

// -----

kgen.generator @call_param() {
  // expected-error @+1 {{'kgen.call_param' callee parameter type must be a func type generator type}}
  %0 = kgen.call_param[si32: 4]()
  kgen.return
}


// -----

kgen.generator @call_param<fn: <type>()->()>() {
  // expected-error @+1 {{cannot name an operation with no results}}
  %0 = kgen.call_param[()->(): bind_params(:<type>()->() fn, f32)]()
  kgen.return
}

// -----

kgen.func @trivial(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

kgen.func @call_param_in_func(%arg0: si32) -> si32 {
  // expected-error @below {{'kgen.call_param' op is only allowed in generators pre-elaboration}}
  %0 = kgen.call_param[(si32) -> si32: @trivial](%arg0)
  kgen.return %0: si32
}

// -----

kgen.generator @bar<F>() {
  kgen.param.declare.region fn = () {
    // expected-error @below {{'kgen.param.constant' op invalid use of parameter with no declaration "Q"}}
    %0 = kgen.param.constant = <Q>
    kgen.return
  }
  kgen.return
}

// -----

kgen.generator @doIt<SomeParam>() {
  kgen.param.declare.region fn = () {
    // expected-error @below {{'kgen.param.constant' op reference to parameter "SomeOtherParam" with incorrect type 'index'}}
    %0 = kgen.param.constant = <SomeOtherParam>
    // expected-note @below {{parameter defined with type '!kgen.dtype'}}
    kgen.param.declare SomeOtherParam : dtype = <f32>
    kgen.return
  }
  kgen.return
}

// -----

kgen.generator @apply_error() {
  // expected-error @below {{custom op 'kgen.param.declare' expected a func type generator type for 'apply'}}
  kgen.param.declare fn = <apply(5, 5)>
}

// -----

kgen.generator @apply_error() {
  // expected-error @below {{custom op 'kgen.param.declare' 'apply' expected a callee operand}}
  kgen.param.declare fn = <apply()>
}

// -----

kgen.generator @apply_error<fn: <index>() -> ()>() {
  // expected-error @below {{'apply' function cannot be parametric: #kgen.param.decl.ref<"fn"> : !kgen.generator<<index>() -> ()>}}
  kgen.param.declare fn = <apply(:<index>() -> () fn)>
}

// -----

kgen.generator @apply_error<fn: () -> ()>() {
  // expected-error @below {{custom op 'kgen.param.declare' 'apply' function must return one result}}
  kgen.param.declare fn = <apply(:() -> () fn)>
}

// -----

kgen.generator @apply_error<fn: () -> ()>() {
  // expected-error @below {{custom op 'kgen.param.declare' 'apply' function result type must be 'index' but got '!kgen.dtype'}}
  kgen.param.declare fn = <apply(:() -> (!kgen.dtype) fn)>
}

// -----

kgen.generator @target_params2<t0: target>() {
 // expected-error @below {{expected '='}}
  kgen.param.assert <identical(:target t0, #kgen.target<triple"triple", "cpu", "features", 3, 4>)>, "must support target!!"
  kgen.return
}

// -----

// COM: Make sure these don't crash and emit an error gracefully.

kgen.generator @no_return() {
  // expected-error @below {{block with no terminator}}
  kgen.param.declare A = <1>
}

// -----

kgen.func @no_return() {
  // expected-error @below {{block with no terminator}}
  kgen.param.declare A = <1>
}

// -----

// expected-error @below {{cyclic reference between expressions}}
kgen.generator @cyclicIf() {
  kgen.param.declare cond_var: i1 = <1>
  // expected-note @below {{parameter "M2" is defined here}}
  kgen.param.declare.region M2 = () -> index {
    kgen.param.constant = <N>
    kgen.unreachable
  }
  // This forwards the output parameter of the if statement back around to N,
  // creating a cycle.
  // expected-note @below {{parameter "N" is defined here}}
  kgen.param.declare N = <apply(:() -> index M2)>
  kgen.return
}

// -----

kgen.generator @declareWrongType() {
  // expected-error @below {{'kgen.param.declare' op declares a parameter with type 'index' but parameter expression has type 'i32'}}
  "kgen.param.declare"() {paramDecl = #kgen<param.decl p1 : index>, value = 1 : i32} : () -> ()
  kgen.return
}

// -----

kgen.generator @duplicate_decl() {
  // expected-note @below {{previous declaration here}}
  kgen.param.declare a = <5>
  // expected-error @below {{redeclaration of parameter "a"}}
  kgen.param.declare a = <6>
  kgen.return
}

// -----

// expected-note @below {{previous declaration here}}
kgen.generator @name_shadowing_1<a>() {
  // expected-error @below {{redeclaration of parameter "a"}}
  kgen.param.declare.region fn = <a>() {
    kgen.unreachable
  }
  kgen.return
}

// -----

kgen.generator @name_shadowing_2<a>() {
  // expected-note @below {{previous declaration here}}
  kgen.param.declare b = <a>
  kgen.param.declare.region fn = () {
    // expected-error @below {{redeclaration of parameter "b"}}
    kgen.param.declare b = <a>
    kgen.unreachable
  }
  kgen.return
}

// -----

kgen.func @stage_closure() {
  // expected-error @below {{staged closures cannot have parameters}}
  %0 = kgen.stage_closure = <n : ui32>() capturing -> index {
  } { name = "k" }
}

// -----

kgen.generator @bad_return() -> index {
  // expected-error @below {{'kgen.return' op expected 1 operands, but given 0}}
  kgen.return
}


// -----

// expected-error @below {{environment value "value" is an integer not of `index` type}}
module attributes {kgen.env = #kgen.env<{value = 1}>} {}


// -----

// expected-error @below {{environment value "str" is a string not of `!kgen.string` type}}
module attributes {kgen.env = #kgen.env<{str = "hello"}>} {}

// -----

// expected-error @below {{environment value "fp" is neither an index, string, or unit attribute}}
module attributes {kgen.env = #kgen.env<{fp = 2.0}>} {}

// -----

kgen.generator @variant_constant<value: i32>() {
  // expected-error @below {{variant attribute value type 'i32' does not match type at index 0 which is 'f32'}}
  %0 = kgen.param.constant: variant<f32, f64> = <#kgen.variant<:i32 value, 0>>
}

// -----

// expected-error @below {{'byref_result' argument must be the last argument}}
kgen.func @invalid(%arg0: !kgen.pointer<index> byref_result, %arg1: index) -> !kgen.none {
  kgen.unreachable
}

// -----

kgen.generator @kernel() {
  kgen.return
}

kgen.generator export @top() {
  // expected-error @below {{the immediate emission kind must be either '=llvm', '=asm', '=llvm-opt', or '=object'}}
  kgen.param.constant: string = <#kgen.compile_assembly<current_target(), =something, false, :() -> () @kernel>>
  kgen.return
}

// -----

kgen.generator @kernel() {
  kgen.return
}

kgen.generator export @top() {
  // expected-error @below {{emissionKind operand should evaluate to either 'asm', 'llvm', 'llvm-opt', 'object', 'llvm-bitcode', or 'llvm-opt-bitcode'}}
  kgen.param.constant: string = <#kgen.compile_assembly<current_target(), 6, "", false, :() -> () @kernel>>
  kgen.return
}

// -----

// expected-error @below {{parameter name and parameter value length mismatch. Expected 1, got 2}}
kgen.func @illegal_applied_struct_param_length(%arg0: !kgen.struct_inst<"Bar"[elemT]<:dtype f32, 16>(data: struct<()>) memoryOnly>) {}

// -----

// expected-error @+2 {{cannot convert to scalar dtype si32 from 'vector<2xsi32>'}}
"some.op"() {
  a = #kgen.cast_from_builtin< #M.dense_array<2, 5> : vector<2xsi32>> : !kgen.scalar<si32>
} : () -> ()

// -----

// expected-error @+2 {{cannot convert to scalar dtype f32 from 'bf16'}}
"some.op"() {
  a = #kgen.cast_from_builtin< 0.0 : bf16> : !kgen.scalar<f32>
} : () -> ()

// -----

// expected-error @+3 {{expected non-function type}}
// expected-error @+2 {{failed to parse KGEN_CastFromBuiltinAttr parameter 'arg' which is to be a `TypedAttr`}}
"some.op"() {
  a = #kgen.cast_from_builtin< 0.0 : woof> : !kgen.scalar<si32>
} : () -> ()

// -----

// expected-error @+2 {{cannot convert from scalar dtype si32 to 'ui32'}}
"some.op"() {
  a = #kgen.cast_to_builtin< #kgen<simd 1> : !kgen.simd<1, si32>> : ui32
} : () -> ()

// -----

// expected-error @+2 {{expected vector<4xT>}}
"some.op"() {
  a = #kgen.cast_to_builtin< #kgen<simd 1> : !kgen.simd<4, si32>> : vector<2xsi32>
} : () -> ()

// -----

// expected-error @+2 {{mismatch between value type 'si32' and splat element type 'f16'}}
"some.op"() {
  a = #kgen.simd_splat< #kgen<simd 1> : !kgen.scalar<si32>> : !kgen.simd<3, f16>
} : () -> ()

// -----

// expected-error @+2 {{operand type mismatch}}
"some.op"() {
  a = #kgen.param.expr<and, #kgen<simd 1> : !kgen.scalar<si32>, #kgen<simd 2> : !kgen.scalar<ui32>> : !kgen.scalar<si32>
} : () -> ()

// -----

// expected-error @+2 {{operand type mismatch}}
"some.op"() {
  a = #kgen.param.expr<xor, #kgen<simd 1> : !kgen.scalar<si32>, #kgen<simd 2> : !kgen.scalar<ui32>> : !kgen.scalar<si32>
} : () -> ()

// -----

// expected-error @+2 {{operand type mismatch}}
"some.op"() {
  b = #kgen.param.expr<eq, #kgen<simd "1.0"> : !kgen.scalar<f64>, #kgen<simd "2.0"> : !kgen.scalar<f32>> : !kgen.scalar<bool>
} : () -> ()

// -----

// expected-error @+2 {{comparisons return simd<bool>}}
"some.op"() {
  b = #kgen.param.expr<eq, #kgen<simd 1> : !kgen.scalar<ui8>, #kgen<simd 2> : !kgen.scalar<ui8>> : !kgen.simd<3, ui8>
} : () -> ()

// -----

// COM: Non-numeric equality belongs on `#kgen.param.identical`, not lane-wise `eq`.
// expected-error @+2 {{eq operands must be numeric; use '#kgen.param.identical' to compare values of type '!kgen.dtype'}}
"some.op"() {
  b = #kgen.param.expr<eq, #kgen.dtype.constant<f32> : !kgen.dtype, #kgen.dtype.constant<bf16> : !kgen.dtype> : !kgen.scalar<bool>
} : () -> ()

// -----

kgen.generator @bind_params_discharged_mask_size() {
  // expected-error @+2 {{bind_params discharged mask has size 1 but the generator has 2 body constraints}}
  kgen.param.declare bad: index =
    <#kgen.bind_params<:!lit.generator<<index, {<true, loc("bind_params":3:1)>, <true, loc("bind_params":3:2)>}>index> ?, 1 | "1">>
  kgen.return
}

// -----

// COM: `param.identical` is n-ary, but a class needs two members to say
// COM: anything; both spellings report the arity through the same verifier.
// expected-error @+2 {{'param.identical' must have at least two operands}}
"some.op"() {
  b = #kgen.param.identical<#kgen.type<i32> : !kgen.type>
} : () -> ()

// -----

// expected-error @+1 {{'kgen.param.constant' 'param.identical' must have at least two operands}}
%0 = kgen.param.constant: scalar<bool> = <identical(:dtype f32)>

// -----

// expected-error @+2 {{operand type mismatch}}
"some.op"() {
  b = #kgen.param.identical<#kgen.type<i32> : !kgen.type,
                            #kgen<simd 1> : !kgen.scalar<index>>
} : () -> ()

// -----

// COM: Every operand is checked against the first, so a mismatch reports from
// COM: any position rather than just the second.
// expected-error @+2 {{operand type mismatch}}
"some.op"() {
  b = #kgen.param.identical<#kgen.type<i32> : !kgen.type,
                            #kgen.type<i64> : !kgen.type,
                            #kgen<simd 1> : !kgen.scalar<index>>
} : () -> ()
