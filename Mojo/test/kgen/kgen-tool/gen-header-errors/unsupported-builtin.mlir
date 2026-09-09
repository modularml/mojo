// RUN: kgen %s -emit=header -func="kernel" -verify-diagnostics

// expected-error @below {{unhandled elementary type: 'f128'}}
// expected-note @below {{see current operation}}
// expected-error @below {{during header emission for this function}}
kgen.func export @kernel(%a: f128) cabi -> f128 {
  kgen.return %a : f128
}
