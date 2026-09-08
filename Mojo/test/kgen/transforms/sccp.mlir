// RUN: kgen-opt -sccp -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @loop_generates_constant
kgen.func @loop_generates_constant() -> (index, index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2

  %0 = index.add %idx1, %idx2
  %1 = index.mul %0, %0

  // CHECK: [[FALSE:%.*]] = kgen.param.constant: scalar<bool> = <false>
  // CHECK-DAG: [[IDX11:%.*]] = index.constant 11
  // CHECK-DAG: [[IDX9:%.*]] = index.constant 9
  // CHECK-DAG: [[IDX3:%.*]] = index.constant 3
  // CHECK-DAG: [[IDX2:%.*]] = index.constant 2
  // CHECK-DAG: [[IDX1:%.*]] = index.constant 1
  // CHECK-DAG: [[IDX0:%.*]] = index.constant 0

  // The result of this loop will be 2
  %2 = hlcf.loop(%arg0 = %idx0: index) -> index {
    // CHECK: index.cmp
    %3 = index.cmp slt(%arg0, %idx2)
    %b3 = pop.cast_from_builtin %3 : i1 to !kgen.scalar<bool>
    hlcf.if %b3 {
      hlcf.yield
    } else {
      %4 = index.add %arg0, %1
      hlcf.break %4: index
    }
    %5 = index.add %arg0, %idx1
    hlcf.continue %5 : index
  }

  // COM: %2 will be a constant, so this cmp result will be a constant
  // CHECK-NOT: index.cmp
  %6 = index.cmp slt(%2, %idx2)
  %b6 = pop.cast_from_builtin %6 : i1 to !kgen.scalar<bool>

  // CHECK: [[V1:%.*]] = hlcf.if [[FALSE]]
  %7 = hlcf.if %b6 -> index {
    hlcf.yield %idx0: index
  } else {
    // CHECK: hlcf.yield [[IDX9]]
    hlcf.yield %1: index
  }

  // CHECK: kgen.return [[IDX11]], [[IDX9]]
  kgen.return %2, %7 : index, index
}

// CHECK-LABEL: @not_much_can_be_known
kgen.func @not_much_can_be_known(%cond: !kgen.scalar<bool>) -> (index, index) {
  // COM: Not much can be folded except obvious one that has constant operands.
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2

  // CHECK-DAG: [[IDX9:%.*]] = index.constant 9
  // CHECK-DAG: [[IDX3:%.*]] = index.constant 3
  // CHECK-DAG: [[IDX2:%.*]] = index.constant 2
  // CHECK-DAG: [[IDX1:%.*]] = index.constant 1
  // CHECK-DAG: [[IDX0:%.*]] = index.constant 0

  %0 = index.add %idx1, %idx2
  %1 = index.mul %0, %0
  %2 = hlcf.loop(%arg0 = %idx0: index, %arg1 = %cond: !kgen.scalar<bool>) -> index {
    %3 = hlcf.if %arg1 -> index {
      hlcf.yield %idx0: index
    } else {
      hlcf.yield %idx1: index
    }

    %4 = index.cmp slt(%3, %arg0)
    %b4 = pop.cast_from_builtin %4 : i1 to !kgen.scalar<bool>
    hlcf.if %b4 {
      hlcf.yield
    } else {
      %5 = index.add %3, %3
      hlcf.break %5: index
    }

    %6 = index.add %3, %idx1
    %7 = index.cmp slt(%3, %idx2)
    %b7 = pop.cast_from_builtin %7 : i1 to !kgen.scalar<bool>
    hlcf.continue %6, %b7 : index, !kgen.scalar<bool>
  }

  // CHECK: kgen.return [[IDX9]], [[V0:%.*]]
  kgen.return %1, %2 : index, index
}

// CHECK-LABEL: @should_not_crash
kgen.func @should_not_crash() -> (i1) {
  %idx0 = index.constant 0

  %2 = hlcf.loop(%arg0 = %idx0: index) -> index {
    hlcf.loop "_loop_0" {
        hlcf.continue "_loop_0"
    }
    kgen.unreachable
  }

  %6 = index.cmp slt(%2, %idx0)
  kgen.return %6 : i1
}


// CHECK-LABEL: @nested_if_constant_result
kgen.func @nested_if_constant_result(%cond: !kgen.scalar<bool>) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2

  %0 = index.add %idx1, %idx2
  %1 = index.mul %0, %0

  // CHECK: [[TRUE:%.*]] = kgen.param.constant: scalar<bool> = <true>
  // CHECK-DAG: [[IDX9:%.*]] = index.constant 9
  // CHECK-DAG: [[IDX3:%.*]] = index.constant 3
  // CHECK-DAG: [[IDX2:%.*]] = index.constant 2
  // CHECK-DAG: [[IDX1:%.*]] = index.constant 1
  // CHECK-DAG: [[IDX0:%.*]] = index.constant 0

  %2 = hlcf.if %cond -> index {
    %3:2 = hlcf.if %cond -> index, index {
      // CHECK: hlcf.yield [[IDX9]], [[IDX1]]
      hlcf.yield %1, %idx1: index, index
    } else {
      // CHECK: hlcf.yield [[IDX2]], [[IDX1]]
      hlcf.yield %idx2, %idx1: index, index
    }
    kgen.call @foo(%3#0) : (index) -> ()
    // COM: This cmp has constant result.
    %4 = index.cmp slt (%3#1, %idx2)
    %b4 = pop.cast_from_builtin %4 : i1 to !kgen.scalar<bool>

    // CHECK: [[V2:%.*]] = hlcf.if [[TRUE]]
    %5 = hlcf.if %b4 -> index {
      // CHECK: hlcf.yield [[IDX1]]
      hlcf.yield %3#1: index
    } else {
      hlcf.yield %3#0: index
    }

    // CHECK: hlcf.yield [[IDX1]]
    hlcf.yield %5: index
  } else {
    // CHECK: hlcf.yield [[IDX3]]
    hlcf.yield %0: index
  }

  kgen.return %2 : index
}

// CHECK-LABEL: @test_switches
kgen.func @test_switches(%arg0: index) -> (index, index, index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2

  %0 = index.add %idx1, %idx2
  %1 = index.mul %0, %0

  // CHECK-DAG: [[INDEX2:%.*]] = kgen.param.constant = <2>
  // CHECK-DAG: [[IDX9:%.*]] = index.constant 9
  // CHECK-DAG: [[IDX3:%.*]] = index.constant 3
  // CHECK-DAG: [[IDX2:%.*]] = index.constant 2
  // CHECK-DAG: [[IDX1:%.*]] = index.constant 1
  // CHECK-DAG: [[IDX0:%.*]] = index.constant 0

  // COM: switch result is constant.
  %2 = hlcf.switch %arg0 -> index
  default {
    %3 = index.mul %0, %0

    // CHECK: hlcf.yield [[IDX9]]
    hlcf.yield %3: index
  }
  case 2 {
    // CHECK: hlcf.yield [[IDX9]]
    hlcf.yield %1: index
  }

  // COM: switch result is unknown.
  // CHECK: [[V4:%.*]] = hlcf.switch
  %4 = hlcf.switch %arg0 -> index
  default {
    hlcf.yield %arg0: index
  }
  case 1 {
    hlcf.yield %1: index
  }

  // COM: complex switch result is constant.
  %5 = hlcf.switch %arg0 -> index
  default {
    // COM: loop result is constant
    %6 = hlcf.loop(%arg1 = %idx0: index) -> index {
      %7 = index.cmp slt(%arg1, %idx2)
      %b7 = pop.cast_from_builtin %7 : i1 to !kgen.scalar<bool>
      hlcf.if %b7 {
        hlcf.yield
      } else {
        hlcf.break %arg1: index
      }
      %8 = index.add %arg1, %idx1
      hlcf.continue %8 : index
    }
    // CHECK: hlcf.yield [[INDEX2]]
    hlcf.yield %6: index
  }
  case 2 {
    hlcf.yield %idx2: index
  }

  // CHECK: kgen.return [[IDX9]], [[V4]], [[INDEX2]]
  kgen.return %2, %4, %5: index, index, index
}

kgen.func @test_for_loop(%arg0: index) -> (index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2

  // CHECK-DAG: [[IDX3:%.*]] = index.constant 3
  // CHECK-DAG: [[IDX2:%.*]] = index.constant 2
  // CHECK-DAG: [[IDX1:%.*]] = index.constant 1
  // CHECK-DAG: [[IDX0:%.*]] = index.constant 0

  %0 = hlcf.for [%idx0 to %idx2 step %idx1 slt add] (%arg1 = %idx0 : index, %arg2 = %idx1: index) -> index {
    %1 = index.add %arg1, %idx1
    %2 = index.add %arg2, %idx1
    kgen.call @foo(%1, %arg0) : (index, index) -> ()
    hlcf.for.yield [induction_var (%1 : index)] [retvals (%2: index)] [iterargs ()]
  }

  // CHECK: kgen.return [[IDX3]]
  kgen.return %0: index
}

// CHECK-LABEL: @nested_loops
kgen.func @nested_loops() -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2
  %idx4 = index.constant 4

  %0 = index.add %idx1, %idx2
  %1 = index.mul %0, %idx1

  // CHECK-DAG: [[IDX10:%.*]] = index.constant 10
  // CHECK-DAG: [[IDX3:%.*]] = index.constant 3
  // CHECK-DAG: [[IDX4:%.*]] = index.constant 4
  // CHECK-DAG: [[IDX2:%.*]] = index.constant 2
  // CHECK-DAG: [[IDX1:%.*]] = index.constant 1
  // CHECK-DAG: [[IDX0:%.*]] = index.constant 0

  // COM: The result of this loop will be 10
  %2 = hlcf.loop(%arg0 = %idx0: index) -> index {
    %3 = hlcf.loop(%arg1 = %idx0: index) -> index {
      %4 = index.cmp slt(%arg1, %arg0)
      %b4 = pop.cast_from_builtin %4 : i1 to !kgen.scalar<bool>
      hlcf.if %b4 {
        hlcf.yield
      } else {
        %5 = index.add %arg1, %1
        hlcf.break %5: index
      }
      %6 = index.add %arg1, %idx2
      hlcf.continue %6: index
    }

    %7 = index.cmp slt(%3, %idx4)
    %b7 = pop.cast_from_builtin %7 : i1 to !kgen.scalar<bool>
    hlcf.if %b7 {
      hlcf.yield
    } else {
      %8 = index.add %3, %1
      hlcf.break %8: index
    }

    hlcf.continue %3 : index
  }

  // CHECK: kgen.return [[IDX10]]
  kgen.return %2: index
}

// CHECK-LABEL: @loop_generates_constant_but_hits_limit
kgen.func @loop_generates_constant_but_hits_limit() -> (index, index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2
  %idx110 = index.constant 110

  %0 = index.add %idx1, %idx2
  %1 = index.mul %0, %0

  // CHECK-DAG: [[IDX9:%.*]] = index.constant 9
  // CHECK-DAG: [[IDX3:%.*]] = index.constant 3
  // CHECK-DAG: [[IDX110:%.*]] = index.constant 110
  // CHECK-DAG: [[IDX2:%.*]] = index.constant 2
  // CHECK-DAG: [[IDX1:%.*]] = index.constant 1
  // CHECK-DAG: [[IDX0:%.*]] = index.constant 0

  // COM: The result of this loop will be 110, but hits analysis threshold before finishing,
  // COM: so result will be unknown.
  // CHECK: [[V2:%.*]] = hlcf.loop
  %2 = hlcf.loop(%arg0 = %idx0: index) -> index {
    // CHECK: index.cmp
    %3 = index.cmp slt(%arg0, %idx110)
    %b3 = pop.cast_from_builtin %3 : i1 to !kgen.scalar<bool>
    hlcf.if %b3 {
      hlcf.yield
    } else {
      %4 = index.add %arg0, %1
      hlcf.break %4: index
    }
    %5 = index.add %arg0, %idx1
    hlcf.continue %5 : index
  }

  // CHECK: [[V6:%.*]] = index.cmp slt([[V2]], [[IDX2]])
  %6 = index.cmp slt(%2, %idx2)
  %b6 = pop.cast_from_builtin %6 : i1 to !kgen.scalar<bool>

  // CHECK: [[V7:%.*]] = hlcf.if
  %7 = hlcf.if %b6 -> index {
    hlcf.yield %idx0: index
  } else {
    // CHECK: hlcf.yield [[IDX9]]
    hlcf.yield %1: index
  }

  // CHECK: kgen.return [[V2]], [[V7]]
  kgen.return %2, %7 : index, index
}


 // CHECK-LABEL: @nested_if_breaks
 kgen.func @nested_if_breaks(%cond: !kgen.scalar<bool>) -> (index, index) {
   %idx0 = index.constant 0
   %idx1 = index.constant 1
   %idx2 = index.constant 2

   // CHECK-DAG: %idx6 = index.constant 6
   // CHECK-DAG: %idx3 = index.constant 3
   // CHECK-DAG: %idx2 = index.constant 2
   // CHECK-DAG: %idx1 = index.constant 1
   // CHECK-DAG: %idx0 = index.constant 0

   // %0 = 3
   %0 = index.add %idx1, %idx2
   // %1 = 6
   %1 = index.mul %0, %idx2

   // COM: break is in a nested hlcf.if
   // COM: loop generates constant 6.
   %2 = hlcf.loop(%arg0 = %idx0: index) -> index {
     %3 = index.cmp slt(%arg0, %idx2)
     %b3 = pop.cast_from_builtin %3 : i1 to !kgen.scalar<bool>
     hlcf.if %b3 {
       hlcf.yield
     } else {
       %4 = index.add %arg0, %idx2
       %5 = index.cmp slt(%4, %1)
       %b5 = pop.cast_from_builtin %5 : i1 to !kgen.scalar<bool>
       hlcf.if %b5 {
         hlcf.yield
       } else {
         // break and return 6
         hlcf.break %4: index
       }
       hlcf.yield
     }
     %5 = index.add %arg0, %idx1
     hlcf.continue %5 : index
   }

   // COM: loop generates constant 1.
   %6 = hlcf.loop(%arg0 = %cond: !kgen.scalar<bool>) -> index {
     // COM: Both regions of hlcf.if will terminate the current iteration
     // COM: immediately so that %9 and operations after will never be evaluated.
     %8 = hlcf.if %arg0 -> index {
       hlcf.break %idx1: index
     } else {
       hlcf.break %idx1: index
     }
     %9 = index.add %8, %idx1
     %10 = index.cmp slt (%9, %idx2)
     %b10 = pop.cast_from_builtin %10 : i1 to !kgen.scalar<bool>
     hlcf.continue %b10 : !kgen.scalar<bool>
   }
   // CHECK: kgen.return %idx6, %idx1
   kgen.return %2, %6 : index, index
 }

// COM: A break under an unknown condition is only one of the loop's exits, so
// COM: the loop result cannot take its value.
// CHECK-LABEL: @partial_early_exit
kgen.func @partial_early_exit(%cond: !kgen.scalar<bool>) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx3 = index.constant 3

  // CHECK: [[R:%.*]] = hlcf.loop
  %0 = hlcf.loop(%arg0 = %idx0: index) -> index {
    %1 = index.cmp slt(%arg0, %idx3)
    %b1 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
    hlcf.if %b1 {
      hlcf.yield
    } else {
      // COM: the loop leaves here with 3.
      hlcf.break %arg0: index
    }
    hlcf.if %cond {
      %2 = index.cmp eq(%arg0, %idx0)
      %b2 = pop.cast_from_builtin %2 : i1 to !kgen.scalar<bool>
      // COM: this break executes on the first iteration only, and only if
      // COM: %cond holds.
      hlcf.if %b2 {
        hlcf.break %arg0: index
      } else {
        hlcf.yield
      }
      hlcf.yield
    } else {
      hlcf.yield
    }
    %3 = index.add %arg0, %idx1
    hlcf.continue %3 : index
  }

  // COM: the result stays the loop's, not the guarded break's value.
  // CHECK: kgen.return [[R]] : index
  kgen.return %0: index
}

// COM: A break under a condition that folds to false cannot execute, so it does
// COM: not stand in the way of knowing where the loop exits.
// CHECK-LABEL: @break_in_untaken_branch
kgen.func @break_in_untaken_branch() -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx3 = index.constant 3
  %idx5 = index.constant 5
  %false = kgen.param.constant: scalar<bool> = <false>

  %0 = hlcf.loop(%arg0 = %idx0: index) -> index {
    hlcf.if %false {
      hlcf.break %idx1: index
    } else {
      hlcf.yield
    }
    %1 = index.cmp eq(%arg0, %idx3)
    %b1 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
    hlcf.if %b1 {
      hlcf.break %idx5: index
    } else {
      hlcf.yield
    }
    %2 = index.add %arg0, %idx1
    hlcf.continue %2 : index
  }

  // CHECK: kgen.return %idx5
  kgen.return %0: index
}

// COM: A break in a switch case the index does not select cannot execute.
// CHECK-LABEL: @break_in_untaken_switch_case
kgen.func @break_in_untaken_switch_case() -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx3 = index.constant 3
  %idx5 = index.constant 5

  %0 = hlcf.loop(%arg0 = %idx0: index) -> index {
    hlcf.switch %idx0
    default {
      hlcf.yield
    }
    case 1 {
      hlcf.break %idx1: index
    }
    %1 = index.cmp eq(%arg0, %idx3)
    %b1 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
    hlcf.if %b1 {
      hlcf.break %idx5: index
    } else {
      hlcf.yield
    }
    %2 = index.add %arg0, %idx1
    hlcf.continue %2 : index
  }

  // CHECK: kgen.return %idx5
  kgen.return %0: index
}

// COM: A break leaving the outer loop from inside an inner one that stops at
// COM: the unroll threshold may supply results the analysis never saw.
// CHECK-LABEL: @outer_break_from_unconverged_inner
kgen.func @outer_break_from_unconverged_inner() -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx9 = index.constant 9

  // CHECK: [[R:%.*]] = hlcf.loop "outer"
  %0 = hlcf.loop "outer"(%arg0 = %idx0: index) -> index {
    %i = hlcf.loop "inner"(%arg1 = %idx0: index) -> index {
      %1 = index.cmp eq(%arg1, %idx9)
      %b1 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
      hlcf.if %b1 {
        hlcf.break "outer" %idx9 : index
      } else {
        hlcf.yield
      }
      %2 = index.add %arg1, %idx1
      hlcf.continue "inner" %2 : index
    }
    hlcf.break "outer" %arg0 : index
  }

  // CHECK: kgen.return [[R]] : index
  kgen.return %0: index
}

// COM: A continue that re-enters an outer loop is one of its entry edges.
// COM: The inner loop stops at the unroll threshold before reaching it, so the
// COM: outer loop's iterations are not all known and its result is not either.
// CHECK-LABEL: @outer_continue_from_unconverged_inner
kgen.func @outer_continue_from_unconverged_inner() -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx9 = index.constant 9

  // CHECK: [[R:%.*]] = hlcf.loop "outer"
  %0 = hlcf.loop "outer"(%arg0 = %idx0: index) -> index {
    %i = hlcf.loop "inner"(%arg1 = %idx0: index) -> index {
      %1 = index.cmp eq(%arg0, %idx1)
      %b1 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
      hlcf.if %b1 {
        hlcf.break "inner" %arg1 : index
      } else {
        hlcf.yield
      }
      %2 = index.cmp eq(%arg1, %idx9)
      %b2 = pop.cast_from_builtin %2 : i1 to !kgen.scalar<bool>
      // COM: reached only past the threshold, on outer iteration 0.
      hlcf.if %b2 {
        %3 = index.add %arg0, %idx1
        hlcf.continue "outer" %3 : index
      } else {
        hlcf.yield
      }
      %4 = index.add %arg1, %idx1
      hlcf.continue "inner" %4 : index
    }
    hlcf.break "outer" %arg0 : index
  }

  // CHECK: kgen.return [[R]] : index
  kgen.return %0: index
}

// COM: The second break cannot execute: the first one always leaves the loop.
// COM: The result is still known even though the analysis never reaches it.
// CHECK-LABEL: @break_dominated_by_break
kgen.func @break_dominated_by_break(%cond: !kgen.scalar<bool>) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx5 = index.constant 5
  %true = kgen.param.constant: scalar<bool> = <true>

  %0 = hlcf.loop(%arg0 = %idx0: index) -> index {
    hlcf.if %true {
      hlcf.break %idx5: index
    } else {
      hlcf.yield
    }
    hlcf.if %cond {
      hlcf.break %idx1: index
    } else {
      hlcf.yield
    }
    %1 = index.add %arg0, %idx1
    hlcf.continue %1 : index
  }

  // CHECK: kgen.return %idx5
  kgen.return %0: index
}

// CHECK-LABEL: @none_hlcf_controlflownode_donot_crash
kgen.generator @none_hlcf_controlflownode_donot_crash() -> index {
  // COM: Conservatively mark all results as Unknown, but process the subregions.
  kgen.param.declare condition: scalar<bool> = <false>
  %0 = kgen.param.if <condition> -> index {
    %i0 = index.constant 0
    kgen.param.yield %i0: index
  } else {
    %i1 = index.constant 1
    kgen.param.yield %i1: index
  }

  // CHECK: kgen.return [[V0:%.*]]
  kgen.return %0: index
}

// COM: This test should not fail with lattice value assertion due to early exits in the loop.
// CHECK-LABEL: @should_continue
kgen.func @should_continue() -> index {
  // CHECK: [[IDX0:%.*]] = kgen.param.constant = <0>
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %0 = hlcf.loop (%arg0 = %idx0 : index, %arg1 = %idx1 : index) -> index {
    %2 = index.cmp sgt(%arg1, %idx0)
    %b2 = pop.cast_from_builtin %2 : i1 to !kgen.scalar<bool>
    hlcf.if %b2 {
      hlcf.yield
    } else {
      hlcf.break %arg0 : index
    }
    %3 = index.sub %arg1, %idx1
    %4 = index.cmp eq(%idx1, %arg1)
    %b4 = pop.cast_from_builtin %4 : i1 to !kgen.scalar<bool>
    hlcf.if %b4 {
      hlcf.continue %arg0, %3 : index, index
    } else {
      hlcf.yield
    }
    %5 = index.add %arg0, %idx1
    hlcf.continue %5, %3 : index, index
  }
  // CHECK: kgen.call @f([[IDX0]]) : (index) -> index
  %1 = kgen.call @f(%0) : (index) -> index
  kgen.return %1: index
}


// CHECK-LABEL: @indirect_loop_break
kgen.func @indirect_loop_break(%cond: index) -> index {
  // CHECK-DAG: %idx0 = index.constant 0
  // CHECK-DAG: %idx1 = index.constant 1
  // CHECK-DAG: %idx7 = index.constant 7
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx7 = index.constant 7

  // CHECK: [[V0:%.*]] = hlcf.loop
  %0 = hlcf.loop "inlined_cf_scope" () -> index {
    // This loop can't converge and it will lead to the outer loop
    // fail to converge too because one of the break inside breaks
    // the outer loop.
    %2 = hlcf.loop(%arg0 = %idx0: index) -> index {
      // CHECK: index.cmp
      %3 = index.cmp slt(%arg0, %cond)
      %b3 = pop.cast_from_builtin %3 : i1 to !kgen.scalar<bool>
      hlcf.if %b3 {
        // This breaks to the outer loop instead of the inner one.
        hlcf.break "inlined_cf_scope" %arg0 : index
      } else {
        %4 = index.cmp slt(%arg0, %idx7)
        %b4 = pop.cast_from_builtin %4 : i1 to !kgen.scalar<bool>
        hlcf.if %b4 {
          hlcf.yield
        } else {
          %5 = index.add %arg0, %idx1
          hlcf.break %5: index
        }
        hlcf.yield
      }
      %5 = index.add %arg0, %idx1
      hlcf.continue %5 : index
    }
    hlcf.break "inlined_cf_scope" %idx0: index
  }
  // CHECK: kgen.call @f([[V0]])
  %1 = kgen.call @f(%0) : (index) -> index
  kgen.return %1: index
}


// CHECK-LABEL: @single_unreachable_indirect_loop_break
kgen.func @single_unreachable_indirect_loop_break(%cond: index) -> index {
  // CHECK-DAG: %idx0 = index.constant 0
  // CHECK-DAG: %idx1 = index.constant 1
  // CHECK-DAG: %idx100000 = index.constant 100000
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  // MOCO-1318: A large enough trip count such that `else` branch would never be
  // processed by sccp.
  %idx100000 = index.constant 100000

  // CHECK: hlcf.loop
  %0 = hlcf.loop "inlined_cf_scope" () -> index {
    hlcf.loop "_loop_0" (%arg2 = %idx100000 : index, %arg3 = %idx0 : index) {
      %18 = index.cmp sgt(%arg2, %idx0)
      %b18 = pop.cast_from_builtin %18 : i1 to !kgen.scalar<bool>
      hlcf.if %b18 {
        hlcf.yield
      } else {
        hlcf.break "inlined_cf_scope" %idx1 : index
      }
      %19 = index.sub %arg2, %idx1
      hlcf.continue %19, %arg3 : index, index
    }
    kgen.unreachable
  }

  // CHECK: kgen.call @f([[V0]])
  %1 = kgen.call @f(%0) : (index) -> index
  kgen.return %1: index
}
