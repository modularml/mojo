// RUN: kgen %s -emit=header -verify-diagnostics

// {f32, f64} is ABI-correct flattened, but proving it needs field offsets the
// generator cannot compute, so it refuses rather than guess.
// TODO(MOCO-4513): declaring the struct by name drops this judgement.

lit.struct.decl @Foo<DT:dtype> {
  lit.struct.field value : !kgen.scalar<DT>
}

lit.struct.decl @Bar {
  lit.struct.field a : !lit.struct<@Foo<:dtype f32>>
  lit.struct.field b : !kgen.scalar<f64>
}

// expected-error @below {{cannot declare a C prototype taking struct}}
// expected-note @below {{see current operation}}
// expected-error @below {{during header emission for this function}}
kgen.func export @nestedParametricStruct(%a: !lit.struct<@Bar>) cabi {
  kgen.return
}
