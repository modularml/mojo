// RUN: kgen-opt -verify-parameters -lower-lit -split-input-file -verify-diagnostics %s

//===----------------------------------------------------------------------===//
// @align on single-element RegisterPassable struct should NOT warn.
// The struct is preserved (not flattened) to maintain alignment metadata.
//===----------------------------------------------------------------------===//

lit.struct.decl @SingleElementAligned register_passable attributes {minAlignment = 64 : index} {
  lit.struct.field value : i64
}

// -----

// Multi-element struct with @align should NOT warn
lit.struct.decl @MultiElementAligned register_passable attributes {minAlignment = 64 : index} {
  lit.struct.field a : i64
  lit.struct.field b : i64
}

// -----

// Non-register-passable struct with @align should NOT warn
lit.struct.decl @NonRegPassSingleElement attributes {minAlignment = 64 : index} {
  lit.struct.field value : i64
}
