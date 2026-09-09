// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=false" -split-input-file -verify-diagnostics

// The elaborator concretizes a callee as soon as its expansion node exists,
// before that node has elaborated (or failed), so a comptime call can reach
// an instance by its concrete name while the instance is unfinished. Such a
// call must go through the same readiness/error protocol as a call by
// generator name — the interpreter must never be handed an unelaborated body.

// expected-note @below {{function instantiation failed}}
kgen.generator @bad<x: index>() -> index {
  %0 = kgen.param.if <true> -> index {
    // A parameter declaration in the live branch keeps this `param.if` from
    // folding away, so an abandoned body still contains it.
    kgen.param.declare d: dtype = <si32>
    // expected-note @below {{constraint failed: bad instantiation}}
    kgen.param.assert <false>, "bad instantiation"
    %1 = kgen.param.constant = <x>
    kgen.param.yield %1 : index
  } else {
    %2 = kgen.param.constant = <0>
    kgen.param.yield %2 : index
  }
  kgen.return %0 : index
}

// Reaches the instance by the concrete name the elaborator writes for it.
// expected-note @below {{function instantiation failed}}
kgen.generator @by_concrete_name() -> index {
  %0 = kgen.param.constant = <apply(:() -> index @"bad,x=1")>
  kgen.return %0 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @root() -> index {
  // Registers the `@"bad,x=1"` node without waiting for it, so the comptime
  // call below runs while that node is still unfinished.
  %a = kgen.call @bad<:index 1>() : () -> index
  %0 = kgen.param.constant = <apply(:() -> index @by_concrete_name)>
  kgen.return %0 : index
}

// -----

// The readiness check must not reject a callee that did elaborate: an instance
// reached by its concrete name after it completed stays comptime-callable.

kgen.generator @good<x: index>() -> index {
  %0 = kgen.param.if <true> -> index {
    kgen.param.declare d: dtype = <si32>
    %1 = kgen.param.constant = <x>
    kgen.param.yield %1 : index
  } else {
    %2 = kgen.param.constant = <0>
    kgen.param.yield %2 : index
  }
  kgen.return %0 : index
}

kgen.generator export @both_spellings() -> index {
  // The first apply creates and completes the `@"good,x=7"` instance; the
  // second reaches that same instance by its concrete name.
  %0 = kgen.param.constant = <apply(:() -> index @good<:index 7>)>
  %1 = kgen.param.constant = <apply(:() -> index @"good,x=7")>
  kgen.return %1 : index
}
