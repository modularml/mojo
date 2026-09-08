// RUN: kgen-opt -verify-parameters -lower-lit -split-input-file -verify-diagnostics %s

//===----------------------------------------------------------------------===//
// Recursive type via parameter
//===----------------------------------------------------------------------===//

// -----

lit.struct.decl @foo<T: type> register_passable {
  lit.struct.field field : !kgen.struct<(T)>
}

// expected-error @below {{struct has recursive reference to itself}}
lit.struct.decl @bar register_passable {
  lit.struct.field address : !lit.struct<@foo<:type @bar>>
}

// -----

lit.fn @bind_params_non_singleton(%fn: !lit.generator<<index>() -> ()>) {
  // expected-error @+1 {{may only bind singleton compile-time parameters during lowering}}
  %0 = lit.bind_params %fn : !lit.generator<<index>() -> ()>, 1 to !lit.generator<() -> ()>
  kgen.return
}
