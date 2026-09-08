// RUN: kgen %s -emit=header -verify-diagnostics

// expected-error @below {{bitwidth must be a power of 2}}
// expected-note @below {{see current operation}}
// expected-error @below {{during header emission for this function}}
kgen.func export @kernel(%a: i24) cabi -> i24 {
  kgen.return %a : i24
}
