// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=false" -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=true" -allow-unregistered-dialect | FileCheck %s

// Regression test for MOCO-4259. Reflection's `field_ref` emits a parametric
// `kgen.struct.gep` to field 0 of `Self.T`. When `T` is a single-element
// register-passable struct, it is flattened to its sole field's type, so once
// the concrete type is substituted during elaboration the gep result type
// equals its container type -- an identity that reads "field 0" of a pointee
// that is no longer that struct, which the verifier rejects. The elaborator
// must fold it to its container. Here `T` models the flattened field type (a
// multi-field struct, as in the `Outer { inner: Inner }` reproducer).

// CHECK: kgen.func @"gep_field0
// CHECK-NOT: kgen.struct.gep
// CHECK: kgen.return %arg0
kgen.generator @gep_field0<T: type>(%arg0: !kgen.pointer<T>) -> !kgen.pointer<T> {
  %0 = kgen.struct.gep %arg0[0] : <T> -> <T>
  kgen.return %0 : !kgen.pointer<T>
}

// CHECK-LABEL: kgen.func @caller
kgen.generator @caller(%arg0: !kgen.pointer<struct<(index, index)>>)
    -> !kgen.pointer<struct<(index, index)>> {
  %0 = kgen.call @gep_field0<:type struct<(index, index)>>(%arg0)
      : (!kgen.pointer<struct<(index, index)>>) -> !kgen.pointer<struct<(index, index)>>
  kgen.return %0 : !kgen.pointer<struct<(index, index)>>
}
