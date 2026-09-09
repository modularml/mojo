// RUN: kgen %s -emit=header -verify-diagnostics -ignore-failure

// expected-error @below {{unhandled floating point dtype: f16}}
// expected-note @below {{see current operation}}
// expected-error @below {{during header emission for this function}}
kgen.func export @kernel(%a: !kgen.simd<1, f16>) cabi -> !kgen.simd<1, f16> {
  kgen.return %a : !kgen.simd<1, f16>
}
