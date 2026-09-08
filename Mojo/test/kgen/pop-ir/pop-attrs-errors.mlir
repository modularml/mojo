// RUN: kgen-opt -verify-diagnostics -split-input-file -allow-unregistered-dialect %s

kgen.func @simd_constant() {
  // expected-error @below {{integer value doesn't fit into 4 bits: 128}}
  %0 = kgen.param.constant: scalar<ui4> = <<128>>
  kgen.return
}

// -----

kgen.func @simd_constant() {
  // expected-error @below {{failed to parse floating point value}}
  %0 = kgen.param.constant: scalar<f16> = <<"e">>
  kgen.return
}

// -----

kgen.func @simd_constant() {
  // expected-error @below {{expected 'true' or 'false' for bool literal}}
  %0 = kgen.param.constant: scalar<bool> = <<e>>
  kgen.return
}

// -----

kgen.generator @simd_constant<size>() {
  // expected-error @below {{SIMD constant requires a concrete type}}
  %0 = kgen.param.constant: simd<size, bool> = <<true>>
  kgen.return
}

// -----

kgen.generator @array_constant<size>() {
  // expected-error @below {{array attribute expected a concrete size}}
  %0 = kgen.param.constant: array<size, index> = <#pop.array<0>>
}

// -----

// expected-error @below {{array attribute type requires 2 elements but value has 1}}
"some.op"() {a = #pop.array<1> : !pop.array<2, index>} : () -> ()
