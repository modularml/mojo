// RUN: kgen %s -emit=header -verify-diagnostics

// Three fields, over 16 bytes: the ABI passes it indirectly.

// expected-error @below {{cannot declare a C prototype taking struct}}
// expected-note @below {{see current operation}}
// expected-error @below {{during header emission for this function}}
kgen.func export @someBufferKernel(%a: !kgen.struct<(pointer<simd<1, invalid>>, index, !kgen.dtype)>) cabi -> index {
  %size = kgen.struct.extract %a[1] : !kgen.struct<(pointer<simd<1, invalid>>, index, !kgen.dtype)>
  kgen.return %size : index
}
