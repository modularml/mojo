// RUN: kgen-opt %s -canonicalize -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @terminators_conditionally_pure
kgen.func @terminators_conditionally_pure(%arg0: !kgen.scalar<bool>) {
  hlcf.loop {
    // CHECK: {b}
    hlcf.if %arg0 {
      kgen.return
    } else {
      hlcf.yield
    } {b}

    // CHECK: {c}
    hlcf.if %arg0 {
      hlcf.continue
    } else {
      hlcf.yield
    } {c}

    // CHECK: {d}
    hlcf.if %arg0 {
      hlcf.continue
    } else {
      hlcf.yield
    } {d}

    // CHECK: {e}
    hlcf.loop {
      "some.operation"() : () -> ()
      hlcf.break
    } {e}

    // CHECK: {f}
    hlcf.loop {
      hlcf.continue
    } {f}

    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @fold_if_return
kgen.func @fold_if_return(%arg0 : index, %arg1: index, %arg2: index) -> index {
  // CHECK-NOT: hlcf.if
  // CHECK-NEXT: kgen.return %arg0
  // CHECK-NOT: kgen.return
  %cond = kgen.param.constant: scalar<bool> = <true>
  hlcf.if %cond {
    kgen.return %arg0: index
  } else {
    kgen.return %arg1: index
  }
  kgen.return %arg2: index
}

// CHECK-LABEL: @fold_if_yield
kgen.func @fold_if_yield(%arg0 : index, %arg1: index) -> index {
  // CHECK-NOT: hlcf.if
  // CHECK-NEXT: %[[TEN:.*]] = index.constant 10
  // CHECK-NEXT: %[[RES:.*]] = index.add %arg1, %[[TEN]]
  // CHECK-NEXT: kgen.return %[[RES]]
  %cond = kgen.param.constant: scalar<bool> = <false>
  %z = hlcf.if %cond -> index {
    hlcf.yield %arg0: index
  } else {
    hlcf.yield %arg1: index
  }
  %ten = index.constant 10
  %r = index.add %z, %ten
  kgen.return %r: index
}

// CHECK-LABEL: @hoist_unconditional_return
kgen.func @hoist_unconditional_return(%arg0: !kgen.scalar<bool>, %arg1: index, %arg2: index, %arg3: index) -> index {
  // CHECK:      %[[IF_RES:.*]] = hlcf.if
  // CHECK-NEXT:   hlcf.yield %arg1
  // CHECK-NEXT: else
  // CHECK-NEXT:   hlcf.yield %arg2
  // CHECK-NOT:  index.add
  // CHECK:      kgen.return %[[IF_RES]]
  %a, %b = hlcf.if %arg0 -> index, index {
    kgen.return %arg1: index
  } else {
    kgen.return %arg2: index
  }
  %r = index.add %a, %arg3
  kgen.return %r: index
}

// CHECK-LABEL: @hoist_cond_return_then
kgen.func @hoist_cond_return_then(%cond: !kgen.scalar<bool>, %arg1: index, %arg2: index, %arg3: index) -> index {
  // CHECK:      %[[IF_RES:.*]] = hlcf.if
  // CHECK-NEXT:   hlcf.yield %arg1
  // CHECK-NEXT: else
  // CHECK-NEXT:   %[[ELSE_VAL:.*]] = index.add
  // CHECK-NEXT:   hlcf.yield %[[ELSE_VAL]]
  // CHECK-NOT:  index.add
  // CHECK:      return %[[IF_RES]]
  %a, %b = hlcf.if %cond -> index, index {
    kgen.return %arg1: index
  } else {
    hlcf.yield %arg2, %arg3: index, index
  }
  %r = index.add %a, %b
  kgen.return %r: index
}

// CHECK-LABEL: @hoist_cond_return_else
kgen.func @hoist_cond_return_else(%cond: !kgen.scalar<bool>, %arg1: index, %arg2: index, %arg3: index) -> index {
  // CHECK:      %[[IF_RES:.*]] = hlcf.if
  // CHECK-NEXT:   %[[THEN_VAL:.*]] = index.add
  // CHECK-NEXT:   hlcf.yield %[[THEN_VAL]]
  // CHECK-NEXT: else
  // CHECK-NEXT:   hlcf.yield %arg1
  // CHECK-NOT:  index.add
  // CHECK:      return %[[IF_RES]]
  %a, %b = hlcf.if %cond -> index, index {
    hlcf.yield %arg2, %arg3: index, index
  } else {
    kgen.return %arg1: index
  }
  %r = index.add %a, %b
  kgen.return %r: index
}

// CHECK-LABEL: @hoist_cond_break
kgen.func @hoist_cond_break(%cond: !kgen.scalar<bool>, %arg1: index, %arg2: index, %arg3: index) -> index {
  // CHECK:      %[[LOOP_RES:.*]] = hlcf.loop
  // CHECK-NEXT:   %[[IF_RES:.*]] = hlcf.if
  // CHECK-NEXT:     hlcf.yield %arg1
  // CHECK-NEXT:   else
  // CHECK-NEXT:     %[[ELSE_VAL:.*]] = index.add
  // CHECK-NEXT:     hlcf.yield %[[ELSE_VAL]]
  // CHECK-NOT:    index.add
  // CHECK:        hlcf.break %[[IF_RES]]
  // CHECK:      kgen.return %[[LOOP_RES]]
  %t = hlcf.loop () -> index {
    %a, %b = hlcf.if %cond -> index, index {
      hlcf.break %arg1: index
    } else {
      hlcf.yield %arg2, %arg3: index, index
    }
    %r = index.add %a, %b
    hlcf.break %r: index
  }
  kgen.return %t: index
}

// CHECK-LABEL: @hoist_cond_break2
kgen.func @hoist_cond_break2(%cond: !kgen.scalar<bool>, %arg1: index, %arg2: index, %arg3: index) -> index{
  // CHECK:      hlcf.loop "outer"
  // CHECK-NEXT:   hlcf.loop "inner"
  // CHECK-NEXT:     hlcf.if
  // CHECK-NEXT:       hlcf.yield
  // CHECK-NEXT:     else
  // CHECK-NEXT:       index.add
  // CHECK-NEXT:       hlcf.yield
  // CHECK:          hlcf.break "outer"
  // CHECK:        hlcf.break "outer"
  // CHECK:      kgen.return
  %outer_res = hlcf.loop "outer" () -> index {
    %inner_res = hlcf.loop "inner" () -> index {
      %a, %b = hlcf.if %cond -> index, index {
        hlcf.break "outer" %arg1: index
      } else {
        hlcf.yield %arg2, %arg3: index, index
      }
      %r = index.add %a, %b
      hlcf.break "outer" %r: index
    }
    hlcf.break "outer" %inner_res: index
  }
  kgen.return %outer_res: index
}

// CHECK-LABEL: @hoist_cond_break3
kgen.func @hoist_cond_break3(%cond: !kgen.scalar<bool>, %arg1: index, %arg2: index, %arg3: index) -> index{
  // CHECK:      hlcf.loop "outer"
  // CHECK-NEXT:   hlcf.loop "inner"
  // CHECK-NEXT:     hlcf.if
  // CHECK-NEXT:       hlcf.break
  // CHECK-NEXT:     else
  // CHECK-NEXT:       hlcf.yield
  // CHECK:          index.add
  // CHECK:          hlcf.break "outer"
  // CHECK:        hlcf.break "outer"
  // CHECK:      kgen.return
  %outer_res = hlcf.loop "outer" () -> index {
    %inner_res = hlcf.loop "inner" () -> index {
      %a, %b = hlcf.if %cond -> index, index {
        hlcf.break %arg1: index
      } else {
        hlcf.yield %arg2, %arg3: index, index
      }
      %r = index.add %a, %b
      hlcf.break "outer" %r: index
    }
    hlcf.break "outer" %inner_res: index
  }
  kgen.return %outer_res: index
}

// CHECK-LABEL: @hoist_cond_break4
kgen.func @hoist_cond_break4(%cond: !kgen.scalar<bool>, %arg1: index, %arg2: index, %arg3: index) -> index{
  // CHECK:      hlcf.loop
  // CHECK-NEXT:   hlcf.loop
  // CHECK-NEXT:     hlcf.if
  // CHECK-NEXT:       kgen.return
  // CHECK-NEXT:     else
  // CHECK-NEXT:       hlcf.yield
  // CHECK:          index.add
  // CHECK:          hlcf.break
  // CHECK:        hlcf.break
  // CHECK:      kgen.return
  %outer_res = hlcf.loop () -> index {
    %inner_res = hlcf.loop () -> index {
      %a, %b = hlcf.if %cond -> index, index {
        kgen.return %arg1: index
      } else {
        hlcf.yield %arg2, %arg3: index, index
      }
      %r = index.add %a, %b
      hlcf.break %r: index
    }
    hlcf.break %inner_res: index
  }
  kgen.return %outer_res: index
}

// CHECK-LABEL: @dont_hoist_cond_return_nested
kgen.func @dont_hoist_cond_return_nested(%cond1: !kgen.scalar<bool>, %cond2: !kgen.scalar<bool>, %arg2: index, %arg3: index) -> index {
  hlcf.if %cond1 {
    hlcf.if %cond2 {
      kgen.return %arg2: index
    } else {
      hlcf.yield
    }
    hlcf.yield
  } else {
    hlcf.yield
  }
  kgen.return %arg3: index
}

// CHECK-LABEL: @several_ifs
// Here we in theory would hoist all returns out. This would happen if we
// visit ifs from bottom to top, but doesn't happen if we go in the usual
// order. We can't control the order in canonicalizer so we should add another
// simplification to deal with that - the function below shows the test that we
// need to simplify.
kgen.func @several_ifs(%cond1: !kgen.scalar<bool>, %cond2: !kgen.scalar<bool>, %cond3: !kgen.scalar<bool>) -> () {
  hlcf.if %cond1 {
    %x = index.constant 1
    kgen.return
  } else {
    hlcf.yield
  }
  hlcf.if %cond2 {
    %x = index.constant 2
    kgen.return
  } else {
    hlcf.yield
  }
  hlcf.if %cond3 {
    %x = index.constant 3
    kgen.return
  } else {
    hlcf.yield
  }
  %y = index.constant 4
  kgen.return
}

// CHECK-LABEL: @cond_return_two_ifs2
// Here we theoretically should be able to hoist return out, but we don't do that now.
// TODO: Implement that.
kgen.func @cond_return_two_ifs2(%cond: !kgen.scalar<bool>, %arg: index) -> index {
  %tt = hlcf.if %cond -> index {
    hlcf.yield %arg: index
  } else {
    %X = index.constant 1
    %Y = index.constant 2
    %c, %d = hlcf.if %cond -> index, index {
      kgen.return %arg: index
    } else {
      %x = index.constant 3
      %y = index.constant 4
      hlcf.yield %x, %y: index, index
    }
    %r = index.add %X, %c
    hlcf.yield %r: index
  }
  kgen.return %tt: index
}

// CHECK-LABEL: @empty_if_1
kgen.func @empty_if_1(%cond: !kgen.scalar<bool>, %arg1: index, %arg2: index) -> index {
  // CHECK-NEXT: kgen.return %arg1
  hlcf.if %cond {
    hlcf.yield
  } else {
    hlcf.yield
  }
  kgen.return %arg1: index
}

// CHECK-LABEL: @empty_if_2
kgen.func @empty_if_2(%cond: !kgen.scalar<bool>, %arg1: index, %arg2: index) -> index {
  // CHECK-NOT:  hlcf.if
  // CHECK-NOT:  hlcf.yield %arg1, %arg2
  // CHECK:      %[[RES:.*]] = index.add %arg1, %arg2
  // CHECK-NEXT: kgen.return %[[RES]]
  %a, %b = hlcf.if %cond -> index, index {
    hlcf.yield %arg1, %arg2: index, index
  } else {
    hlcf.yield %arg1, %arg2: index, index
  }
  %r = index.add %a, %b
  kgen.return %r: index
}

// CHECK-LABEL: @empty_if_partial
kgen.func @empty_if_partial(%cond: !kgen.scalar<bool>, %arg1: index, %arg2: index) -> index {
  // CHECK:      hlcf.if
  // CHECK:      %[[RES:.*]] = index.add %arg1, %{{.*}}
  // CHECK-NEXT: kgen.return %[[RES]]
  %a, %b = hlcf.if %cond -> index, index {
    hlcf.yield %arg1, %arg1: index, index
  } else {
    hlcf.yield %arg1, %arg2: index, index
  }
  %r = index.add %a, %b
  kgen.return %r: index
}

// CHECK-LABEL: @remove_unused_if_results
kgen.func @remove_unused_if_results(%arg0: !kgen.scalar<bool>, %arg1: i32, %arg2: i32) -> i32 {
  // CHECK-NEXT: %0 = hlcf.if %arg0 -> i32 {
  %0:2 = hlcf.if %arg0 -> i32, i32 {
    "some.op"() : () -> ()
    // CHECK: hlcf.yield %arg2 : i32
    hlcf.yield %arg1, %arg2 : i32, i32
  // CHECK-NEXT: else
  } else {
    // CHECK-NEXT: hlcf.yield %arg2 : i32
    hlcf.yield %arg1, %arg2 : i32, i32
  }
  kgen.return %0#1 : i32
}

// CHECK-LABEL: @dead_loop
kgen.func @dead_loop() {
  // CHECK-NOT:  hlcf.loop
  // CHECK-NEXT: kgen.return
  hlcf.loop {
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @dead_loop_inner
kgen.func @dead_loop_inner(%arg0: index, %arg1: index) -> index{
  // CHECK:      hlcf.loop
  // CHECK-NOT:    hlcf.loop
  // CHECK-NEXT:   index.add
  // CHECK-NEXT:   hlcf.break %
  // CHECK:      kgen.return
  %r1 = hlcf.loop () -> index {
    %r2 = hlcf.loop () -> index{
      hlcf.break %arg0: index
    }
    %t = index.add %r2, %arg1
    hlcf.break %t: index
  }
  kgen.return %r1: index
}

// CHECK-LABEL: @dead_loop_inner2
kgen.func @dead_loop_inner2(%arg0: index, %arg1: index, %arg2: index) -> index{
  // CHECK-NOT:  hlcf.loop
  // CHECK-NOT:                index.add %arg0, %arg1
  // CHECK-NEXT: %[[RES:.*]] = index.add %arg0, %arg2
  // CHECK-NEXT: kgen.return %[[RES]]
  %x = hlcf.loop "outer" () -> index {
    hlcf.loop {
      hlcf.break "outer" %arg0: index
    }
    %y = index.add %arg0, %arg1
    hlcf.break %y: index
  }
  %z = index.add %x, %arg2
  kgen.return %z: index
}

// CHECK-LABEL: @dead_loop_label
kgen.func @dead_loop_label() {
  // CHECK-NEXT: hlcf.loop {
  hlcf.loop {
    // CHECK-NEXT: hlcf.continue
    hlcf.loop "inner" {
      hlcf.break
    }
    hlcf.continue
  }
  // CHECK: hlcf.loop "outer"
  hlcf.loop "outer" {
    // CHECK-NEXT: some.op
    "some.op"() : () -> ()
    // CHECK-NEXT: hlcf.break "outer"
    hlcf.loop {
      hlcf.break "outer"
    }
    hlcf.continue
  }
  kgen.return
}

// CHECK-LABEL: @dead_loop_returns_loop_arg
kgen.func @dead_loop_returns_loop_arg() -> index {
  %index0 = index.constant 0
  %0 = hlcf.loop "loop" (%arg0 = %index0 : index) -> index {
    hlcf.break "loop" %arg0 : index
  }
  // CHECK: [[IDX0:%.*]] = index.constant 0
  // CHECK-NEXT: kgen.return [[IDX0]] : index
  kgen.return %0 : index
}

// CHECK-LABEL: @dead_loop_returns_not_loop_arg
kgen.func @dead_loop_returns_not_loop_arg(%arg0: index) -> index {
  %index0 = index.constant 0
  %0 = hlcf.loop "loop" () -> index {
    hlcf.break "loop" %arg0 : index
  }
  // CHECK: kgen.return %arg0 : index
  kgen.return %0 : index
}

// CHECK-LABEL: @unused_loop_results
kgen.func @unused_loop_results(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK-NEXT: %0 = hlcf.loop "outer" () -> i32
  %0:2 = hlcf.loop "outer" () -> (i32, i32) {
    hlcf.loop "inner" {
      "some.op"() : () -> ()
      hlcf.break
    }
    // CHECK: hlcf.loop {
    hlcf.loop {
      "some.op"() : () -> ()
      // CHECK: break "outer" %arg1 : i32
      hlcf.break "outer" %arg0, %arg1 : i32, i32
    }
    // CHECK: break %arg1 : i32
    hlcf.break %arg0, %arg1 : i32, i32
  // CHECK: } {unrollLevel = #hlcf<unroll_level 2>}
  } {unrollLevel = #hlcf<unroll_level 2>}

  kgen.return %0#1 : i32
}

// CHECK-LABEL: @unused_loop_args
kgen.func @unused_loop_args(%arg0: i32) {
  // CHECK-NEXT: hlcf.loop {
  hlcf.loop (%arg1 = %arg0 : i32) -> () {
    // CHECK-NEXT: hlcf.continue
    hlcf.continue %arg0 : i32
  }
  kgen.return
}

// CHECK-LABEL: @break_in_both
kgen.func @break_in_both(%arg0: !kgen.scalar<bool>) {
  hlcf.loop {
    hlcf.if %arg0 {
      hlcf.break
    } else {
      hlcf.break
    }
    hlcf.continue
  }
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @break_outer
kgen.func @break_outer(%arg0: !kgen.scalar<bool>) {
  hlcf.loop "outer" {
    hlcf.loop {
      hlcf.if %arg0 {
        hlcf.break "outer"
      } else {
        hlcf.break "outer"
      }
      hlcf.continue
    }
    hlcf.continue
  }
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @break_different_label
kgen.func @break_different_label(%arg0: !kgen.scalar<bool>) {
  // CHECK-NEXT: hlcf.loop
  hlcf.loop "outer" {
    // CHECK-NEXT: hlcf.loop
    hlcf.loop {
      // CHECK-NEXT: hlcf.if
      hlcf.if %arg0 {
        hlcf.break "outer"
      } else {
        hlcf.break
      }
      hlcf.continue
    }
    hlcf.continue
  }
  kgen.return
}

// CHECK-LABEL: @noop_loop
kgen.func @noop_loop(%arg0: index, %arg1: index, %arg2: index, %arg3: !kgen.pointer<index>) {
  // CHECK-NOT: hlcf.for
  hlcf.for [%arg0 to %arg1 step %arg2 slt add] (%arg4 = %arg0 : index) {
    %0 = index.add %arg4, %arg2
    %1 = pop.load %arg3 : !kgen.pointer<index>
    %2 = index.add %0, %1
    hlcf.for.yield [induction_var (%2 : index)] [retvals ()] [iterargs ()]
  }
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @store_loop
kgen.func @store_loop(%arg0: index, %arg1: index, %arg2: index, %arg3: !kgen.pointer<index>) {
  // CHECK: hlcf.for
  hlcf.for [%arg0 to %arg1 step %arg2 slt add] (%arg4 = %arg0 : index) {
    %0 = index.add %arg4, %arg2
    %1 = pop.load %arg3 : !kgen.pointer<index>
    %2 = index.add %0, %1
    pop.store %2, %arg3 : !kgen.pointer<index>
    hlcf.for.yield [induction_var (%2 : index)] [retvals ()] [iterargs ()]
  }
  kgen.return
}


// CHECK-LABEL: @if_cond_same
kgen.func @if_cond_same(%arg0: !kgen.scalar<bool>) -> !kgen.scalar<bool> {
  %0 = kgen.param.constant: scalar<bool> = <false>
  %1 = kgen.param.constant: scalar<bool> = <true>
  // CHECK-NEXT: return %arg0
  %2 = hlcf.if %arg0 -> !kgen.scalar<bool> {
    hlcf.yield %1 : !kgen.scalar<bool>
  } else {
    hlcf.yield %0 : !kgen.scalar<bool>
  }
  kgen.return %2 : !kgen.scalar<bool>
}


// https://github.com/modular/modular/issues/5137:
// Tail call optimization doesn't happen for tail recursive functions with raises
// CHECK-LABEL: @tail_call_error_fn
// CHECK-NEXT:   %[[C:.*]] = kgen.param.constant: scalar<bool> = <false>
// CHECK-NEXT:    %[[R:.*]] = hlcf.if %arg0 -> !kgen.scalar<bool> {
// CHECK-NEXT:      pop.store %arg2, %arg4 : !kgen.pointer<index>
// CHECK-NEXT:      hlcf.yield %[[C]] : !kgen.scalar<bool>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[REC:.*]] = kgen.call @tail_call_error_fn
// CHECK-NEXT:      hlcf.yield %[[REC]] : !kgen.scalar<bool>
// CHECK-NEXT:    }
// CHECK-NEXT:    kgen.return %[[R]]
kgen.generator export @tail_call_error_fn(%cond: !kgen.scalar<bool>, %arg0: index, %arg1: index, %arg2: !kgen.pointer<struct<() memoryOnly>> byref_error, %arg3: !kgen.pointer<index> byref_result) throws -> !kgen.scalar<bool> attributes {sourceName = "factorial"} {
  %0 = kgen.param.constant: scalar<bool> = <true>
  %1 = kgen.param.constant: scalar<bool> = <false>
  %4 = hlcf.if %cond -> !kgen.scalar<bool> {
    pop.store %arg1, %arg3 : !kgen.pointer<index>
    hlcf.yield %1 : !kgen.scalar<bool>
  } else {
    %6 = kgen.call @tail_call_error_fn(%cond, %arg0, %arg0, %arg2, %arg3) : (!kgen.scalar<bool>, index, index, !kgen.pointer<struct<() memoryOnly>> byref_error, !kgen.pointer<index> byref_result) throws -> !kgen.scalar<bool>
    // This 'if' should be canonicalized away.
    hlcf.if %6 {
      kgen.return %0 : !kgen.scalar<bool>
    } else {
      hlcf.yield
    }
    hlcf.yield %1 : !kgen.scalar<bool>
  }
  kgen.return %4 : !kgen.scalar<bool>
}
