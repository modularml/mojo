// RUN: kgen-opt %s -simplify-cf -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @remove_trivial_loop_0
kgen.func @remove_trivial_loop_0() -> () {
  "foo.op"() : () -> ()
  // CHECK-NOT: hlcf.loop
  hlcf.loop {
    "bar.op"() : () -> ()
    hlcf.break
  }
  // CHECK-NEXT: foo.op
  // CHECK-NEXT: bar.op
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @remove_trivial_loop_1
kgen.func @remove_trivial_loop_1(%arg0: index) -> index {
  // CHECK-NOT: hlcf.loop
  %r = hlcf.loop () -> index {
    hlcf.break %arg0: index
  }
  // CHECK-NEXT: return %arg0
  kgen.return %r: index
}

// CHECK-LABEL: @remove_trivial_loop_2
// This loop shouldn't be removed as it has continue.
kgen.func @remove_trivial_loop_2(%cond: !kgen.scalar<bool>, %arg0: index) -> index {
  // CHECK-NEXT: hlcf.loop
  %r = hlcf.loop () -> index {
    hlcf.if %cond {
      hlcf.continue
    } else {
      hlcf.yield
    }
    hlcf.break %arg0: index
  }
  kgen.return %r: index
}

// CHECK-LABEL: @remove_trivial_loop_3
// This loop shouldn't be removed as it has two breaks.
kgen.func @remove_trivial_loop_3(%cond: !kgen.scalar<bool>, %arg0: index, %arg1: index) -> index {
  // CHECK-NEXT: hlcf.loop
  %r = hlcf.loop () -> index {
    hlcf.if %cond {
      hlcf.break %arg1: index
    } else {
      hlcf.yield
    }
    hlcf.break %arg0: index
  }
  kgen.return %r: index
}

// CHECK-LABEL: @remove_trivial_loop_4
// This loop can be removed as the return doesn't make the transformation incorrect.
kgen.func @remove_trivial_loop_4(%cond: !kgen.scalar<bool>, %arg0: index, %arg1: index) -> index {
  // CHECK-NOT: hlcf.loop
  %r = hlcf.loop () -> index {
    // CHECK-NEXT: hlcf.if
    hlcf.if %cond {
      // CHECK-NEXT: return %arg2
      kgen.return %arg1: index
    } else {
      hlcf.yield
    }
    // CHECK-NOT: break
    hlcf.break %arg0: index
  }
  // CHECK: return %arg1
  kgen.return %r: index
}

// CHECK-LABEL: @remove_trivial_loop_5
// Here we can remove the outer loop despite the presence of break and continue
// in the inner loop (which can't be removed).
kgen.func @remove_trivial_loop_5(%cond: !kgen.scalar<bool>, %arg0: index, %arg1: index) -> index {
  // CHECK-COUNT-1: hlcf.loop
  %r = hlcf.loop () -> index {
    %t = hlcf.loop () -> index {
      // CHECK-NEXT: hlcf.if
      hlcf.if %cond {
        // CHECK-NEXT: continue
        hlcf.continue
      } else {
        hlcf.yield
      }
      // CHECK: break %arg1
      hlcf.break %arg0: index
    }
    // CHECK-NOT: break
    hlcf.break %t: index
  }
  // CHECK: return %0
  kgen.return %r: index
}

// CHECK-LABEL: @remove_trivial_loop_6
// This loop can be removed even though the break is to the outer loop.
kgen.func @remove_trivial_loop_6(%cond: !kgen.scalar<bool>, %arg0: index, %arg1: index) -> index {
  // TODO: We should be able to delete both loops here, but we only manage to
  // delete the inner one now.

  // CHECK-NEXT: hlcf.loop "outer"
  hlcf.loop "outer" {
    // CHECK-NOT: hlcf.loop
    hlcf.loop {
      // CHECK-NEXT: bar.op
      "bar.op"() : () -> ()
      // CHECK-NEXT: hlcf.break "outer"
      hlcf.break "outer"
    }
    // CHECK-NOT: foo.op
    "foo.op"() : () -> ()
    hlcf.break
  }
  kgen.return %arg0: index
}

// CHECK-LABEL: @remove_trivial_loop_7
kgen.func @remove_trivial_loop_7() {
  // CHECK-NEXT: hlcf.loop "outer"
  hlcf.loop "outer" {
    // CHECK-NOT: hlcf.loop
    hlcf.loop {
      // CHECK-NEXT: bar.op
      "bar.op"() : () -> ()
      // CHECK-NEXT: hlcf.continue
      hlcf.continue "outer"
    }
    // CHECK-NOT: foo.op
    "foo.op"() : () -> ()
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @remove_trivial_loop_8
// The inner loop can be removed, the outer cannot.
kgen.func @remove_trivial_loop_8(%cond: !kgen.scalar<bool>) {
  // CHECK-NEXT: hlcf.loop
  hlcf.loop {
    // CHECK-NEXT: hlcf.if
    hlcf.if %cond {
      hlcf.continue
    } else {
      // CHECK-NOT: hlcf.loop
      hlcf.loop {
        hlcf.break
      }
      hlcf.yield
    }
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @remove_trivial_loop_9
// Both loops can be removed.
kgen.func @remove_trivial_loop_9(%cond: !kgen.scalar<bool>) {
  // CHECK-NEXT: return
  hlcf.loop {
    hlcf.loop {
      hlcf.break
    }
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @remove_trivial_loop_10
kgen.func @remove_trivial_loop_10(%cond: !kgen.scalar<bool>) {
  // FIXME: Only one loop can be removed.
  // CHECK-NEXT: hlcf.loop
  hlcf.loop {
    hlcf.loop {
      // CHECK-NEXT: return
      kgen.return
    }
    // CHECK-NOT: foo.op
    "foo.op"() : () -> ()
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @remove_trivial_loop_11
// Only the outer loop can be removed.
kgen.func @remove_trivial_loop_11(%cond: !kgen.scalar<bool>) {
  hlcf.loop () {
    // CHECK-NEXT: hlcf.loop
    hlcf.loop () {
      // CHECK-NEXT: hlcf.if
      hlcf.if %cond {
        hlcf.continue
      } else {
        kgen.return
      }
      hlcf.break
    }
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @remove_trivial_loop_12
// Only the outer loop can be removed.
kgen.func @remove_trivial_loop_12(%cond: !kgen.scalar<bool>) {
  hlcf.loop () {
    // CHECK-NEXT: hlcf.loop
    hlcf.loop () {
      // CHECK-NEXT: hlcf.if
      hlcf.if %cond {
        hlcf.yield
      } else {
        hlcf.break
      }
      kgen.return
    }
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @two_loop_erase
// COM: Ensure this doesn't result in use-after-free.
kgen.func @two_loop_erase() {
  // CHECK-NEXT: return
  hlcf.loop {
    kgen.return
  }
  hlcf.loop {
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: @erase_trivial_try
kgen.func @erase_trivial_try() {
  lit.try {
    // CHECK-NEXT: foo.op
    "foo.op"() : () -> ()
    lit.try.yield
  } except (%e: index) {
    kgen.unreachable
  } else {
    // CHECK-NEXT: bar.op
    "bar.op"() : () -> ()
    lit.try.yield
  }
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @raise_in_try
kgen.func @raise_in_try(%arg0: index) {
  // CHECK-NEXT: lit.try
  lit.try "try0" {
    lit.try.raise "try0" %arg0 : index
  } except (%e: index) {
    lit.try.yield
  } else {
    kgen.unreachable
  }
  kgen.return
}


// CHECK-LABEL: @nested_raise
kgen.func @nested_raise(%arg0: index) {
  // CHECK-NEXT: lit.try
  lit.try "try0" {
    hlcf.loop {
      lit.try.raise "try0" %arg0 : index
    }
    lit.try.yield
  } except (%e: index) {
    lit.try.yield
  } else {
    lit.try.yield
  }
  kgen.return
}

// CHECK-LABEL: @raise_in_else
// COM: Make sure the right contextual try is selected.
kgen.func @raise_in_else(%arg0: index) {
  // CHECK-NEXT: lit.try "try0" {
  lit.try "try0" {
    // CHECK-NEXT: foo.op
    "foo.op"() : () -> ()
    lit.try "try1" {
      // CHECK-NEXT: bar.op
      "bar.op"() : () -> ()
      lit.try.yield
    } except (%arg1: index) {
      kgen.unreachable
    } else {
      // CHECK-NEXT: baz.op
      "baz.op"() : () -> ()
      // CHECK-NEXT: lit.try.raise "try0" %arg0
      lit.try.raise "try0" %arg0 : index
    }
    lit.try.yield
  // CHECK-NEXT: except
  } except (%arg2: index) {
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  } else {
    lit.try.yield
  }
  kgen.return
}

// CHECK-LABEL: @return_in_try
kgen.func @return_in_try() {
  lit.try {
    // CHECK-NEXT: return
    kgen.return
  } except (%e: index) {
    kgen.unreachable
  } else {
    lit.try.yield
  }
  // CHECK-NOT: foo.op
  "foo.op"() : () -> ()
  kgen.return
}

// CHECK-LABEL: @return_in_else
kgen.func @return_in_else() {
  lit.try {
    lit.try.yield
  } except (%e: index) {
    kgen.unreachable
  } else {
    // CHECK-NEXT: return
    kgen.return
  }
  // CHECK-NOT: foo.op
  "foo.op"() : () -> ()
  kgen.return
}

// CHECK-LABEL: @try_arg_passing
kgen.func @try_arg_passing(%arg0: index, %arg1: si32) -> (index, si32) {
  lit.try {
    lit.try.yield %arg0, %arg1 : index, si32
  } except (%e: index) {
    kgen.unreachable
  } else {
  ^bb0(%arg2: index, %arg3: si32):
    // CHECK-NEXT: return %arg0, %arg1
    kgen.return %arg2, %arg3 : index, si32
  }
  kgen.unreachable
}

// CHECK-LABEL: @try_result_passing
kgen.func @try_result_passing(%arg0: index, %arg1: si32) -> (index, si32) {
  %0:2 = lit.try -> index, si32 {
    lit.try.yield %arg0, %arg1 : index, si32
  } except (%e: index) {
    kgen.unreachable
  } else {
  ^bb0(%arg2: index, %arg3: si32):
    lit.try.yield %arg2, %arg3 : index, si32
  }
  // CHECK-NEXT: return %arg0, %arg1
  kgen.return %0#0, %0#1 : index, si32
}

// CHECK-LABEL: @loop_break
kgen.func @loop_break(%arg0: index) -> index {
  // CHECK-NEXT: return %arg0 : index
  %0 = hlcf.loop (%arg1 = %arg0 : index) -> index {
    hlcf.break %arg1 : index
  }
  kgen.return %0 : index
}
