// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func @loop
kgen.func @loop(%arg0: i32, %arg1: i64) {
  // CHECK: hlcf.loop {
  hlcf.loop {
    hlcf.break
  }

  // CHECK: hlcf.loop {
  hlcf.loop () -> () {
    hlcf.break
  }

  // CHECK: hlcf.loop (%{{.*}} = %arg0 : i32) {
  hlcf.loop (%0 = %arg0 : i32) -> () {
    hlcf.break
  }

  // CHECK: %{{.*}} = hlcf.loop () -> index {
  %0 = hlcf.loop () -> index {
    hlcf.continue
  }

  // CHECK: %{{.*}}:2 = hlcf.loop () -> (index, index) {
  %1:2 = hlcf.loop () -> (index, index) {
    hlcf.continue
  }

  kgen.return
}

// CHECK-LABEL: kgen.func @if
kgen.func @if(%arg0: !kgen.scalar<bool>, %arg1: i32, %arg2: i64) {
  // CHECK-NEXT: hlcf.if %arg0 {
  hlcf.if %arg0 {
    // CHECK-NEXT: hlcf.yield
    hlcf.yield
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: hlcf.yield
    hlcf.yield
  // CHECK-NEXT: }
  }

  // CHECK: %{{.*}} = hlcf.if %arg0 -> i32 {
  %0 = hlcf.if %arg0 -> i32 {
    // CHECK-NEXT: hlcf.yield %arg1 : i32
    hlcf.yield %arg1 : i32
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: hlcf.yield %arg1 : i32
    hlcf.yield %arg1 : i32
  }

  // CHECK: %{{.*}} = hlcf.if %arg0 -> i32, i64
  %1:2 = hlcf.if %arg0 -> i32, i64 {
    // CHECK-NEXT: hlcf.yield %arg1, %arg2 : i32, i64
    hlcf.yield %arg1, %arg2 : i32, i64
  } else {
    hlcf.yield %arg1, %arg2 : i32, i64
  }

  kgen.return
}

// CHECK-LABEL: kgen.func @func_loop_if
kgen.func @func_loop_if(%arg0: !kgen.scalar<bool>, %arg1: i32, %arg2: i64) -> i32 {
  // CHECK: %[[V0:.*]] = hlcf.loop (%[[A:.*]] = %arg2 : i64) -> i32
  %2 = hlcf.loop (%0 = %arg2 : i64) -> i32 {
    // CHECK: %[[V1:.*]] = hlcf.if %arg0 -> i64
    %1 = hlcf.if %arg0 -> i64 {
      // CHECK: kgen.return %arg1 : i32
      kgen.return %arg1 : i32
    } else {
      // CHECK: hlcf.yield %[[A]] : i64
      hlcf.yield %0 : i64
    }
    // CHECK: hlcf.if %arg0
    hlcf.if %arg0 {
      // CHECK: hlcf.continue %[[A]] : i64
      hlcf.continue %0 : i64
    } else {
      hlcf.yield
    }
    // CHECK: hlcf.break %arg1 : i32
    hlcf.break %arg1 : i32
  }
  kgen.return %2 : i32
}

// CHECK-LABEL: @labelled_loops
kgen.func @labelled_loops(%cond: !kgen.scalar<bool>, %arg0: index, %arg1: i32) {
  // CHECK-NEXT: hlcf.loop "foo" (%{{.*}} = %{{.*}} : index) -> i32
  %0 = hlcf.loop "foo" (%a0 = %arg0 : index) -> i32 {
    // CHECK-NEXT: hlcf.loop "bar" (%{{.*}} = %{{.*}} : i32) -> index
    %1 = hlcf.loop "bar" (%a1 = %arg1 : i32) -> index {
      hlcf.if %cond {
        // CHECK: break %{{.*}} : index
        hlcf.break %a0 : index
      } else {
        // CHECK: break "bar" %{{.*}} : index
        hlcf.break "bar" %a0 : index
      }
      hlcf.if %cond {
        // CHECK: break "foo" %{{.*}} : i32
        hlcf.break "foo" %a1 : i32
      } else {
        // CHECK: continue %{{.*}} : i32
        hlcf.continue %a1 : i32
      }
      // CHECK: continue "foo" %{{.*}} : index
      hlcf.continue "foo" %a0 : index
    }
    // CHECK: break %{{.*}} : i32
    hlcf.break %arg1 : i32
  }
  kgen.return
}

// CHECK-LABEL: @switch
kgen.func @switch(%arg0: index, %arg1: i32, %arg2: i64) {
  // CHECK-NEXT: hlcf.switch %arg0
  hlcf.switch %arg0
  // CHECK-NEXT: default {
  default {
    // CHECK-NEXT: return
    kgen.return
  }
  // CHECK: case 2 {
  case 2 {
    // CHECK-NEXT: yield
    hlcf.yield
  }
  // CHECK: hlcf.switch %arg0 -> i32, i64
  %0:2 = hlcf.switch %arg0 -> i32, i64
  default {
    hlcf.yield %arg1, %arg2 : i32, i64
  }
  kgen.return
}

// CHECK-LABEL: @elif
kgen.func @elif(%arg0: index, %arg1: index, %arg2: index) {
  %idx0 = index.constant 0
  // CHECK:           [[VAR1:%.*]] = index.cmp eq(%arg0, %idx0)
  // CHECK-NEXT:      [[VAR1B:%.*]] = pop.cast_from_builtin [[VAR1]] : i1 to !kgen.scalar<bool>
  // CHECK-NEXT:      [[VAR0:%.*]] = hlcf.elif [[VAR1B]] -> index {
  // CHECK-NEXT:       hlcf.yield %arg0 : index
  // CHECK-NEXT:      } else {
  // CHECK-NEXT:       %idx1 = index.constant 1
  // CHECK-NEXT:       [[VAR2:%.*]] = index.cmp eq(%arg0, %idx1)
  // CHECK-NEXT:       [[VAR2B:%.*]] = pop.cast_from_builtin [[VAR2]] : i1 to !kgen.scalar<bool>
  // CHECK-NEXT:       hlcf.elif.yield [[VAR2B]]
  // CHECK-NEXT:      } then {
  // CHECK-NEXT:       hlcf.yield %arg0 : index
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       hlcf.yield %arg0 : index
  // CHECK-NEXT:     }
  %c0 = index.cmp eq(%arg0, %idx0)
  %cb0 = pop.cast_from_builtin %c0 : i1 to !kgen.scalar<bool>
  %0 = hlcf.elif %cb0 -> index {
    hlcf.yield %arg0 : index
  } else {
    %idx1 = index.constant 1
    %c1 = index.cmp eq(%arg0, %idx1)
    %cb1 = pop.cast_from_builtin %c1 : i1 to !kgen.scalar<bool>
    hlcf.elif.yield %cb1
  } then {
    hlcf.yield %arg0 : index
  } else {
    hlcf.yield %arg0 : index
  }


  // CHECK:        [[VAR3:%.*]] = index.cmp eq(%arg0, %idx0)
  // CHECK-NEXT:   [[VAR3B:%.*]] = pop.cast_from_builtin [[VAR3]] : i1 to !kgen.scalar<bool>
  // CHECK-NEXT:   hlcf.elif [[VAR3B]] {
  // CHECK-NEXT:    hlcf.yield
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    hlcf.yield
  // CHECK-NEXT:  }
  %c2 = index.cmp eq(%arg0, %idx0)
  %cb2 = pop.cast_from_builtin %c2 : i1 to !kgen.scalar<bool>
  hlcf.elif %cb2 {
    hlcf.yield
  } else {
    hlcf.yield
  }
  kgen.return
}

// CHECK-LABEL: @for_index_bounds
kgen.func @for_index_bounds(%lb: index, %ub: index, %step: index) {
  // CHECK: hlcf.for [%{{.*}} to %{{.*}} step %{{.*}} slt add]
  // CHECK-NEXT: hlcf.for.yield [induction_var (%{{.*}} : index)] [retvals ()] [iterargs ()]
  hlcf.for [%lb to %ub step %step slt add] (%arg0 = %lb : index) {
    hlcf.for.yield [induction_var (%arg0 : index)] [retvals ()] [iterargs ()]
  }
  kgen.return
}

// CHECK-LABEL: @for_si8_bounds
kgen.func @for_si8_bounds(%lb: !kgen.scalar<si8>, %ub: !kgen.scalar<si8>, %step: !kgen.scalar<si8>) {
  // CHECK: hlcf.for [%{{.*}} to %{{.*}} step %{{.*}} : !kgen.scalar<si8> slt add]
  // CHECK-NEXT: hlcf.for.yield [induction_var (%{{.*}} : !kgen.scalar<si8>)] [retvals ()] [iterargs ()]
  hlcf.for [%lb to %ub step %step : !kgen.scalar<si8> slt add] (%arg0 = %lb : !kgen.scalar<si8>) {
    hlcf.for.yield [induction_var (%arg0 : !kgen.scalar<si8>)] [retvals ()] [iterargs ()]
  }
  kgen.return
}

// CHECK-LABEL: @for_si32_bounds
kgen.func @for_si32_bounds(%lb: !kgen.scalar<si32>, %ub: !kgen.scalar<si32>, %step: !kgen.scalar<si32>) {
  // CHECK: hlcf.for [%{{.*}} to %{{.*}} step %{{.*}} : !kgen.scalar<si32> slt add]
  // CHECK-NEXT: hlcf.for.yield [induction_var (%{{.*}} : !kgen.scalar<si32>)] [retvals ()] [iterargs ()]
  hlcf.for [%lb to %ub step %step : !kgen.scalar<si32> slt add] (%arg0 = %lb : !kgen.scalar<si32>) {
    hlcf.for.yield [induction_var (%arg0 : !kgen.scalar<si32>)] [retvals ()] [iterargs ()]
  }
  kgen.return
}

// CHECK-LABEL: @for_si64_bounds
kgen.func @for_si64_bounds(%lb: !kgen.scalar<si64>, %ub: !kgen.scalar<si64>, %step: !kgen.scalar<si64>) {
  // CHECK: hlcf.for [%{{.*}} to %{{.*}} step %{{.*}} : !kgen.scalar<si64> sgt sub]
  // CHECK-NEXT: hlcf.for.yield [induction_var (%{{.*}} : !kgen.scalar<si64>)] [retvals ()] [iterargs ()]
  hlcf.for [%lb to %ub step %step : !kgen.scalar<si64> sgt sub] (%arg0 = %lb : !kgen.scalar<si64>) {
    hlcf.for.yield [induction_var (%arg0 : !kgen.scalar<si64>)] [retvals ()] [iterargs ()]
  }
  kgen.return
}

// CHECK-LABEL: @for_pop_index_bounds
kgen.func @for_pop_index_bounds(%lb: !kgen.scalar<index>, %ub: !kgen.scalar<index>, %step: !kgen.scalar<index>) {
  // CHECK: hlcf.for [%{{.*}} to %{{.*}} step %{{.*}} : !kgen.scalar<index> slt add]
  // CHECK-NEXT: hlcf.for.yield [induction_var (%{{.*}} : !kgen.scalar<index>)] [retvals ()] [iterargs ()]
  hlcf.for [%lb to %ub step %step : !kgen.scalar<index> slt add] (%arg0 = %lb : !kgen.scalar<index>) {
    hlcf.for.yield [induction_var (%arg0 : !kgen.scalar<index>)] [retvals ()] [iterargs ()]
  }
  kgen.return
}

// Loop with a return value.
// CHECK-LABEL: @for_with_retval
kgen.func @for_with_retval(%lb: index, %ub: index, %step: index, %init: index) -> index {
  // CHECK: %{{.*}} = hlcf.for [%{{.*}} to %{{.*}} step %{{.*}} slt add]
  // CHECK-SAME: (%{{.*}} = %{{.*}} : index, %{{.*}} = %{{.*}} : index) -> index
  // CHECK: hlcf.for.yield [induction_var (%{{.*}} : index)] [retvals (%{{.*}} : index)] [iterargs ()]
  %0 = hlcf.for [%lb to %ub step %step slt add]
                (%arg0 = %lb : index, %arg1 = %init : index) -> index {
    hlcf.for.yield [induction_var (%arg0 : index)] [retvals (%arg1 : index)] [iterargs ()]
  }
  kgen.return %0 : index
}

// Loop with an additional iter arg.
// CHECK-LABEL: @for_with_iterarg
kgen.func @for_with_iterarg(%lb: index, %ub: index, %step: index, %init: i32) {
  // CHECK: hlcf.for [%{{.*}} to %{{.*}} step %{{.*}} slt add]
  // CHECK-SAME: (%{{.*}} = %{{.*}} : index, %{{.*}} = %{{.*}} : i32)
  // CHECK: hlcf.for.yield [induction_var (%{{.*}} : index)] [retvals ()] [iterargs (%{{.*}} : i32)]
  hlcf.for [%lb to %ub step %step slt add]
           (%arg0 = %lb : index, %arg1 = %init : i32) {
    hlcf.for.yield [induction_var (%arg0 : index)] [retvals ()] [iterargs (%arg1 : i32)]
  }
  kgen.return
}

// Pop scalar loop with a return value.
// CHECK-LABEL: @for_si64_with_retval
kgen.func @for_si64_with_retval(%lb: !kgen.scalar<si64>, %ub: !kgen.scalar<si64>,
                                 %step: !kgen.scalar<si64>, %init: index) -> index {
  // CHECK: %{{.*}} = hlcf.for [%{{.*}} to %{{.*}} step %{{.*}} : !kgen.scalar<si64> sgt sub]
  // CHECK-SAME: (%{{.*}} = %{{.*}} : !kgen.scalar<si64>, %{{.*}} = %{{.*}} : index) -> index
  // CHECK: hlcf.for.yield [induction_var (%{{.*}} : !kgen.scalar<si64>)] [retvals (%{{.*}} : index)] [iterargs ()]
  %0 = hlcf.for [%lb to %ub step %step : !kgen.scalar<si64> sgt sub]
                (%arg0 = %lb : !kgen.scalar<si64>, %arg1 = %init : index) -> index {
    hlcf.for.yield [induction_var (%arg0 : !kgen.scalar<si64>)] [retvals (%arg1 : index)] [iterargs ()]
  }
  kgen.return %0 : index
}

// CHECK-LABEL:  kgen.func @elifWithArgs
kgen.func @elifWithArgs(%arg0: index) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2
  // The first condition is an operand, so its `then` reads the dominating
  // values directly. Only the additional arm forwards extra values through
  // `hlcf.elif.yield` into its `then` and the `else`.
  // CHECK:      [[V1:%.*]] = index.cmp eq(%arg0, %idx0)
  // CHECK-NEXT: [[V1B:%.*]] = pop.cast_from_builtin [[V1]] : i1 to !kgen.scalar<bool>
  // CHECK-NEXT: [[V0:%.*]]:2 = hlcf.elif [[V1B]] -> index, index {
  // CHECK-NEXT:   hlcf.yield %arg0, %arg0 : index, index
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[V2:%.*]] = index.cmp eq(%arg0, %idx1)
  // CHECK-NEXT:   [[V2B:%.*]] = pop.cast_from_builtin [[V2]] : i1 to !kgen.scalar<bool>
  // CHECK-NEXT:   hlcf.elif.yield [[V2B]], %arg0, %idx2 : index, index
  // CHECK-NEXT: } then (%arg1: index, %arg2: index){
  // CHECK-NEXT:   hlcf.yield %arg1, %arg2 : index, index
  // CHECK-NEXT: } else (%arg1: index, %arg2: index){
  // CHECK-NEXT:   hlcf.yield %idx0, %arg1 : index, index
  // CHECK-NEXT: }
  %3 = index.cmp eq(%arg0, %idx0)
  %c3 = pop.cast_from_builtin %3 : i1 to !kgen.scalar<bool>
  %0:2 = hlcf.elif %c3 -> index, index {
     hlcf.yield %arg0, %arg0 : index, index
  } else {
     %4 = index.cmp eq(%arg0, %idx1)
     %c4 = pop.cast_from_builtin %4 : i1 to !kgen.scalar<bool>
     hlcf.elif.yield %c4, %arg0, %idx2 : index, index
  } then (%arg1: index, %arg2: index) {
     hlcf.yield %arg1, %arg2 : index, index
  } else (%arg3: index, %arg4: index) {
     hlcf.yield %idx0, %arg3 : index, index
  }
  %1 = index.add %0#1, %0#0
  kgen.return %1 : index
}
