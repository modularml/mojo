// RUN: kgen %s -emit=header -verify-diagnostics

// An 8-byte pair is returned in a single register.

// expected-error @below {{cannot declare a C prototype returning struct}}
// expected-note @below {{see current operation}}
// expected-error @below {{during header emission for this function}}
kgen.func export @twoElemStruct(%arg0: i32) cabi -> !kgen.struct<(i32, i32)> {
  %0 = kgen.param.constant: struct<(i32, i32)> = <{ 0, 0 }>
  kgen.return %0 : !kgen.struct<(i32, i32)>
}
