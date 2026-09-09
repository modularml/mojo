// RUN: kgen-opt -verify-parameters -lower-lit -split-input-file -mlir-print-local-scope %s | FileCheck %s

// `LowerLIT` captures a stable snapshot of the source signature in the
// `sourceFuncTypeGenerator` attribute, wrapped in a `TypeParamAttr`
// (`#kgen.type<...>`) so `LowerLITTypes` lowers it in the value domain. Its POG
// metadata is stripped along with the live signature's, but the parameter and
// argument/result types are preserved. `-verify-parameters` running in the
// pipeline confirms the snapshot verifies.

// A non-parametric generator snapshots just the function type.
// CHECK-LABEL: kgen.generator @trivial_generator
// CHECK-SAME:    sourceFuncTypeGenerator = #kgen.type<(si32) -> si32> : !kgen.type
lit.fn @trivial_generator(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

// -----

// A parametric generator snapshots its parameter types and a body func type
// whose `ParamIndexRefAttr`s (e.g. `*(0,0)`) are bound by the generator's own
// scope.
// CHECK-LABEL: kgen.generator @param_gen
// CHECK-SAME:    sourceFuncTypeGenerator = #kgen.type<<index>(!kgen.simd<*(0,0), f32>) -> !kgen.simd<*(0,0), f32>> : !kgen.type
lit.fn @param_gen<w: index>(%arg0: !kgen.simd<w, f32>) -> !kgen.simd<w, f32> {
  kgen.return %arg0 : !kgen.simd<w, f32>
}

// -----

// Lit-level types inside the snapshot are lowered by `LowerLITTypes`: the
// `!lit.ref` argument becomes a `!kgen.pointer` inside the snapshot's body type.
// CHECK-LABEL: kgen.generator @ref_arg
// CHECK-SAME:    sourceFuncTypeGenerator = #kgen.type<(!kgen.pointer<index>) -> ()> : !kgen.type
lit.fn @ref_arg(%arg0: !lit.ref<index, mut #lit.any.origin>) {
  kgen.return
}

// -----

// The snapshot's `typeValue` (the first component of `#kgen.type<value, mlir>`)
// carries the argument types in the value domain: the `@Int` struct argument
// appears as `!kgen.typevalue<#kgen.genref<@Int>>`, while the `mlirType`
// component keeps the type-domain form (`index`).
lit.struct.decl @Int register_passable {
  lit.struct.field value : index
}

// CHECK-LABEL: kgen.generator @takes_int
// CHECK-SAME:    sourceFuncTypeGenerator = #kgen.type<(!kgen.typevalue<#kgen.genref<@Int>>) -> (), (index) -> ()> : !kgen.type
lit.fn @takes_int(%arg0: !lit.struct<@Int>) {
  kgen.return
}
