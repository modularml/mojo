// RUN: kgen-opt %s -elaborate-generators -split-input-file -verify-diagnostics \
// RUN:   | FileCheck %s

// A comptime call spelled with the concrete name the elaborator writes for an
// instance bypasses the readiness protocol a call by generator name goes
// through, so it can reach a body that still holds `kgen.param.*` ops, which
// the interpreter has no semantics for. In each failing case the callee's
// suspension is a causal consequence of the caller's comptime call, so the
// cycle is genuine and must be reported, never interpreted.

// A `kgen.param.apply` on a concrete name. `@bad,x=1` can only elaborate its
// `kgen.param.if` once `@dep` produces the condition, and `@dep` needs
// `@bad,x=1`.
kgen.generator @dep() -> !kgen.scalar<bool> {
  // expected-note @below {{recursively instantiated through here}}
  kgen.param.apply r = [() -> index: @"bad,x=1"]()
  %0 = kgen.param.constant: !kgen.scalar<bool> = <true>
  kgen.return %0 : !kgen.scalar<bool>
}

// expected-note @below {{function instantiation failed}}
kgen.generator @bad<x: index>() -> index {
  // expected-note @below {{function instantiation in parameter domain that recursively requires itself}}
  // expected-note @below {{back to parameter domain function call here}}
  %0 = kgen.param.if <apply(:() -> !kgen.scalar<bool> @dep)> -> index {
    // A parameter declaration keeps this `param.if` from folding away.
    kgen.param.declare d: dtype = <si32>
    %1 = kgen.param.constant = <x>
    kgen.param.yield %1 : index
  } else {
    %2 = kgen.param.constant = <0>
    kgen.param.yield %2 : index
  }
  kgen.return %0 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @root() -> index {
  // A `kgen.call` does not block, so this registers the `@"bad,x=1"` node and
  // moves on, leaving it unfinished when `@dep` reaches it above.
  // expected-note @below {{call expansion failed with parameter value(s): ("x": 1)}}
  %a = kgen.call @bad<:index 1>() : () -> index
  kgen.return %a : index
}

// -----

// The same cycle reached through a parameter expression, which concretizes the
// callee itself rather than going through `kgen.param.apply`.
kgen.generator @dep() -> !kgen.scalar<bool> {
  // expected-note @below {{recursively instantiated through here}}
  %r = kgen.param.constant = <apply(:() -> index @"bad,x=1")>
  %0 = kgen.param.constant: !kgen.scalar<bool> = <true>
  kgen.return %0 : !kgen.scalar<bool>
}

// expected-note @below {{function instantiation failed}}
kgen.generator @bad<x: index>() -> index {
  // expected-note @below {{function instantiation in parameter domain that recursively requires itself}}
  // expected-note @below {{back to parameter domain function call here}}
  %0 = kgen.param.if <apply(:() -> !kgen.scalar<bool> @dep)> -> index {
    kgen.param.declare d: dtype = <si32>
    %1 = kgen.param.constant = <x>
    kgen.param.yield %1 : index
  } else {
    %2 = kgen.param.constant = <0>
    kgen.param.yield %2 : index
  }
  kgen.return %0 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @root() -> index {
  // expected-note @below {{call expansion failed with parameter value(s): ("x": 1)}}
  %a = kgen.call @bad<:index 1>() : () -> index
  kgen.return %a : index
}

// -----

// The concrete name reached through a generator that a comptime call applies.
// Resolution happens while `@helper` is being elaborated, before any
// interpreter frame exists, so `bad,x=1 -> dep -> helper -> bad,x=1` is
// reported from the dependency graph.
kgen.generator @helper() -> index {
  // expected-note @below {{recursively instantiated through here}}
  %c = kgen.call @"bad,x=1"() : () -> index
  kgen.return %c : index
}

kgen.generator @dep() -> !kgen.scalar<bool> {
  // expected-note @below {{recursively instantiated through here}}
  kgen.param.apply h = [() -> index: @helper]()
  %0 = kgen.param.constant: !kgen.scalar<bool> = <true>
  kgen.return %0 : !kgen.scalar<bool>
}

// expected-note @below {{function instantiation failed}}
kgen.generator @bad<x: index>() -> index {
  // expected-note @below {{function instantiation in parameter domain that recursively requires itself}}
  // expected-note @below {{back to parameter domain function call here}}
  %0 = kgen.param.if <apply(:() -> !kgen.scalar<bool> @dep)> -> index {
    kgen.param.declare d: dtype = <si32>
    %1 = kgen.param.constant = <x>
    kgen.param.yield %1 : index
  } else {
    %2 = kgen.param.constant = <0>
    kgen.param.yield %2 : index
  }
  kgen.return %0 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @root() -> index {
  // expected-note @below {{call expansion failed with parameter value(s): ("x": 1)}}
  %a = kgen.call @bad<:index 1>() : () -> index
  kgen.return %a : index
}

// -----

// The early-return shape of the last case, except the taken branch calls
// `@dep` and `@dep` applies this instance by its concrete name.
// `earlyExit,x=5 -> dep -> earlyExit,x=5` is a real dependency - a `kgen.call`
// may have effects, so it cannot be dropped - and must be reported.

// expected-note @below {{function instantiation failed}}
kgen.generator @dep() -> index {
  // expected-note @below {{function instantiation in parameter domain that recursively requires itself}}
  // expected-note @below {{back to parameter domain function call here}}
  %0 = kgen.param.constant = <apply(:() -> index @"earlyExit,x=5")>
  kgen.return %0 : index
}

// expected-note @below {{function instantiation failed}}
kgen.generator @earlyExit<x: index>() -> index {
  %0 = kgen.param.if <true> -> index {
    // expected-note @below {{call expansion failed}}
    // expected-note @below {{recursively instantiated through here}}
    %d = kgen.call @dep() : () -> index
    %1 = kgen.param.constant = <x>
    kgen.return %1 : index
  } else {
    %2 = kgen.param.constant = <0>
    kgen.param.yield %2 : index
  }
  kgen.return %0 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @root() -> index {
  // expected-note @below {{call expansion failed with parameter value(s): ("x": 5)}}
  %a = kgen.call @earlyExit<:index 5>() : () -> index
  kgen.return %a : index
}

// -----

// A comptime `if` whose taken branch ends in a `kgen.return` leaves the ops
// after it in a split-off block, which survives until the instance is
// finalized. Here the taken branch depends on nothing that depends back, so
// that unreachable block is all that is outstanding - and it must not by
// itself keep the instance from being applied by its concrete name.

kgen.generator @dep() -> index {
  %0 = kgen.param.constant = <apply(:() -> index @"earlyExit,x=5")>
  kgen.return %0 : index
}

kgen.generator @earlyExit<x: index>() -> index {
  %0 = kgen.param.if <true> -> index {
    %1 = kgen.param.constant = <x>
    kgen.return %1 : index
  } else {
    %2 = kgen.param.constant = <0>
    kgen.param.yield %2 : index
  }
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @root
// CHECK: kgen.return
kgen.generator export @root() -> index {
  %a = kgen.call @earlyExit<:index 5>() : () -> index
  %b = kgen.call @dep() : () -> index
  kgen.return %a : index
}
