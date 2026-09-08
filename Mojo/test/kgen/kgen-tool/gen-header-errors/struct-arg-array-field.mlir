// RUN: kgen %s -emit=header -verify-diagnostics

// Over 16 bytes. The array field would also flatten to a C array parameter,
// which decays to a pointer.

// expected-error @below {{cannot declare a C prototype taking struct}}
// expected-note @below {{see current operation}}
// expected-error @below {{during header emission for this function}}
kgen.func export @someNDBufferKernel(%a: !kgen.struct<(pointer<simd<1, invalid>>, index, array<5, index>, !kgen.dtype)>) cabi -> index {
  %size = kgen.struct.extract %a[1] : !kgen.struct<(pointer<simd<1, invalid>>, index, array<5, index>, !kgen.dtype)>
  kgen.return %size : index
}
