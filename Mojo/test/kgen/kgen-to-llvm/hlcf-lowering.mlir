// RUN: kgen-opt -lower-kgen-to-llvm -lower-control-flow -allow-unregistered-dialect %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @nested_continue
kgen.func @nested_continue(%arg0: !kgen.scalar<bool>) {
  // CHECK-NEXT: %[[A0SB:.*]] = builtin.unrealized_conversion_cast %arg0 : i1 to !kgen.scalar<bool>
  // CHECK-NEXT: llvm.br ^bb1
  hlcf.loop {
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: %[[C0:.*]] = builtin.unrealized_conversion_cast %[[A0SB]] : !kgen.scalar<bool> to i1
    // CHECK-NEXT: llvm.cond_br %[[C0]], ^bb2, ^bb3
    hlcf.if %arg0 {
      // CHECK-NEXT: ^bb2:
      // CHECK-NEXT: llvm.return
      kgen.return
    } else {
      // CHECK-NEXT: ^bb3:
      // CHECK-NEXT: llvm.br ^bb1
      hlcf.continue
    }
    // CHECK-NOT: ^bb4:
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @nested_break
kgen.func @nested_break(%arg0: !kgen.scalar<bool>) {
  // CHECK-NEXT: %[[A0SB:.*]] = builtin.unrealized_conversion_cast %arg0 : i1 to !kgen.scalar<bool>
  // CHECK-NEXT: llvm.br ^bb1
  hlcf.loop {
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: %[[C0:.*]] = builtin.unrealized_conversion_cast %[[A0SB]] : !kgen.scalar<bool> to i1
    // CHECK-NEXT: llvm.cond_br %[[C0]], ^bb2, ^bb3
    hlcf.if %arg0 {
      // CHECK-NEXT: ^bb2:
      // CHECK-NEXT: llvm.br ^bb4
      hlcf.yield
    } else {
      // CHECK-NEXT: ^bb3:
      // CHECK-NEXT: llvm.br ^bb5
      hlcf.break
    }
    // CHECK-NEXT: ^bb4:
    // CHECK-NEXT: llvm.br ^bb1
    hlcf.continue
  }
  // CHECK-NEXT: ^bb5:
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @deeply_nested
kgen.func @deeply_nested(%arg0: !kgen.scalar<bool>, %arg1: !kgen.scalar<bool>, %arg2: !kgen.scalar<bool>) {
  // CHECK-NEXT: %[[A2SB:.*]] = builtin.unrealized_conversion_cast %arg2 : i1 to !kgen.scalar<bool>
  // CHECK-NEXT: %[[A1SB:.*]] = builtin.unrealized_conversion_cast %arg1 : i1 to !kgen.scalar<bool>
  // CHECK-NEXT: %[[A0SB:.*]] = builtin.unrealized_conversion_cast %arg0 : i1 to !kgen.scalar<bool>
  // CHECK-NEXT: %[[C0:.*]] = builtin.unrealized_conversion_cast %[[A0SB]] : !kgen.scalar<bool> to i1
  // CHECK-NEXT: llvm.cond_br %[[C0]], ^bb1, ^bb10
  hlcf.if %arg0 {
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: llvm.br ^bb2
    hlcf.loop {
      // CHECK-NEXT: ^bb2:
      // CHECK-NEXT: %[[C1:.*]] = builtin.unrealized_conversion_cast %[[A1SB]] : !kgen.scalar<bool> to i1
      // CHECK-NEXT: llvm.cond_br %[[C1]], ^bb3, ^bb7
      hlcf.if %arg1 {
        // CHECK-NEXT: ^bb3:
        // CHECK-NEXT: %[[C2:.*]] = builtin.unrealized_conversion_cast %[[A2SB]] : !kgen.scalar<bool> to i1
        // CHECK-NEXT: llvm.cond_br %[[C2]], ^bb4, ^bb5
        hlcf.if %arg2 {
          // CHECK-NEXT: ^bb4:
          // CHECK-NEXT: llvm.br ^bb9
          hlcf.break
        } else {
          // CHECK-NEXT: ^bb5:
          // CHECK-NEXT: llvm.br ^bb6
          hlcf.yield
        }
        // CHECK-NEXT: ^bb6:
        // CHECK-NEXT: llvm.br ^bb2
        hlcf.continue
      } else {
        // CHECK-NEXT: ^bb7:
        // CHECK-NEXT: llvm.br ^bb8
        hlcf.yield
      }
      // CHECK-NEXT: ^bb8:
      // CHECK-NEXT: llvm.br ^bb2
      hlcf.continue
    }
    // CHECK-NEXT: ^bb9:
    // CHECK-NEXT: llvm.return
    kgen.return
  } else {
    // CHECK-NEXT: ^bb10:
    // CHECK-NEXT: llvm.br ^bb11
    hlcf.yield
  }
  // CHECK-NEXT: ^bb11:
  kgen.return
}

// CHECK-LABEL: @operands_and_results
kgen.func @operands_and_results(%arg0: !kgen.scalar<bool>, %arg1: i32, %arg2: i64) -> (i32, i64) {
  // CHECK-NEXT: %[[A0SB:.*]] = builtin.unrealized_conversion_cast %arg0 : i1 to !kgen.scalar<bool>
  // CHECK-NEXT: %[[INIT0:.*]] = builtin.unrealized_conversion_cast %[[A0SB]] : !kgen.scalar<bool> to i1
  // CHECK-NEXT: %[[INIT1:.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK-NEXT: llvm.br ^bb1(%[[INIT0]], %[[INIT1]] : i1, i32)
  %0:2 = hlcf.loop (%0 = %arg0 : !kgen.scalar<bool>, %1 = %arg1 : i32) -> (i32, i64) {
    // CHECK-NEXT: ^bb1(%[[ARG0:.*]]: i1, %[[ARG1:.*]]: i32):
    // CHECK-NEXT: %[[V0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : i1 to !kgen.scalar<bool>
    // CHECK-NEXT: %[[V1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]]
    // CHECK-NEXT: %[[C0:.*]] = builtin.unrealized_conversion_cast %[[V0]] : !kgen.scalar<bool> to i1
    // CHECK-NEXT: llvm.cond_br %[[C0]], ^bb2, ^bb3
    %2 = hlcf.if %0 -> i32 {
      // CHECK-NEXT: ^bb2:
      // CHECK-NEXT: %[[R0:.*]] = builtin.unrealized_conversion_cast %arg1
      // CHECK-NEXT: %[[R1:.*]] = builtin.unrealized_conversion_cast %arg2
      // CHECK-NEXT: llvm.br ^bb8(%[[R0]], %[[R1]] : i32, i64
      hlcf.break %arg1, %arg2 : i32, i64
    } else {
      // CHECK-NEXT: ^bb3:
      // CHECK-NEXT: %[[R0:.*]] = builtin.unrealized_conversion_cast %arg1
      // CHECK-NEXT: llvm.br ^bb4(%[[R0]] : i32)
      hlcf.yield %arg1 : i32
    }
    // CHECK-NEXT: ^bb4(%[[ARG2:.*]]: i32):
    // CHECK-NEXT: %[[V2:.*]] = builtin.unrealized_conversion_cast %[[ARG2]]
    // CHECK-NEXT: %[[C1:.*]] = builtin.unrealized_conversion_cast %[[V0]] : !kgen.scalar<bool> to i1
    // CHECK-NEXT: llvm.cond_br %[[C1]], ^bb5, ^bb6
    %3 = hlcf.if %0 -> !kgen.scalar<bool> {
      // CHECK-NEXT: ^bb5:
      // CHECK-NEXT: %[[R0:.*]] = builtin.unrealized_conversion_cast %[[A0SB]]
      // CHECK-NEXT: llvm.br ^bb7(%[[R0]] : i1)
      hlcf.yield %arg0 : !kgen.scalar<bool>
    } else {
      // CHECK-NEXT: ^bb6:
      // CHECK-NEXT: %[[R0:.*]] = builtin.unrealized_conversion_cast %[[A0SB]]
      // CHECK-NEXT: %[[R1:.*]] = builtin.unrealized_conversion_cast %arg1
      // CHECK-NEXT: llvm.br ^bb1(%[[R0]], %[[R1]] : i1, i32
      hlcf.continue %arg0, %arg1 : !kgen.scalar<bool>, i32
    }
    // CHECK-NEXT: ^bb7(%[[ARG3:.*]]: i1):
    // CHECK-NEXT: %[[V3:.*]] = builtin.unrealized_conversion_cast %[[ARG3]]
    // CHECK-NEXT: %[[R0:.*]] = builtin.unrealized_conversion_cast %[[V3]]
    // CHECK-NEXT: %[[R1:.*]] = builtin.unrealized_conversion_cast %[[V2]]
    // CHECK-NEXT: llvm.br ^bb1(%[[R0]], %[[R1]] : i1, i32)
    hlcf.continue %3, %2 : !kgen.scalar<bool>, i32
  }
  // CHECK-NEXT: ^bb8(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i64):
  // CHECK-NEXT: %[[R0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK-NEXT: %[[R1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]]
  // CHECK: return %{{.*}}
  kgen.return %0#0, %0#1 : i32, i64
}

// CHECK-LABEL: @multiple_return
kgen.func @multiple_return(%arg0: !kgen.scalar<bool>) -> (i1, i1) {
  // CHECK-NEXT: %[[A0SB:.*]] = builtin.unrealized_conversion_cast %arg0 : i1 to !kgen.scalar<bool>
  // CHECK-NEXT: %[[B:.*]] = builtin.unrealized_conversion_cast %[[A0SB]] : !kgen.scalar<bool> to i1
  %b = builtin.unrealized_conversion_cast %arg0 : !kgen.scalar<bool> to i1
  hlcf.if %arg0 {
    // CHECK: ^bb1:
    // CHECK-NEXT: %[[S0:.*]] = llvm.mlir.undef
    // CHECK-NEXT: %[[S1:.*]] = llvm.insertvalue %[[B]], %[[S0]][0]
    // CHECK-NEXT: %[[S2:.*]] = llvm.insertvalue %[[B]], %[[S1]][1]
    // CHECK-NEXT: llvm.return %[[S2]]
    kgen.return %b, %b : i1, i1
  } else {
    hlcf.yield
  }
  kgen.return %b, %b : i1, i1
}

// CHECK-LABEL: @switch
kgen.func @switch(%arg0: index) {
  // CHECK: llvm.switch %{{.*}} : i64, ^bb1 [
  // CHECK-NEXT: 2: ^bb2
  hlcf.switch %arg0
  // CHECK: ^bb1:
  default {
    // CHECK-NEXT: return
    kgen.return
  }
  // CHECK: ^bb2:
  case 2 {
    // CHECK-NEXT: br
    hlcf.yield
  }
  kgen.return
}

// COM: Ensure raising inside anything other than the try region of a `lit.try`
// COM: will not branch back to the except.

// CHECK-LABEL: @reraise_in_try
kgen.func @reraise_in_try(%err: i32) {
  // CHECK-NEXT: br ^bb1
  lit.try "try0" {
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: br ^bb2
    lit.try "try1" {
      // CHECK-NEXT: ^bb2:
      // CHECK: br ^bb3(%{{.*}} : i32)
      lit.try.raise "try1" %err :i32
    } except (%arg0: i32) {
      // CHECK-NEXT: ^bb3(%{{.*}}: i32):
      // CHECK: br ^bb4(%{{.*}} : i32)
      lit.try.raise "try0" %arg0 :i32
    } else {
      kgen.unreachable
    }
    kgen.unreachable
  } except (%arg0: i32) {
    // CHECK-NEXT: ^bb4(%{{.*}}: i32):
    // CHECK: outer.except
    "outer.except"() : () -> ()
    // CHECK: br ^bb5
    lit.try.yield
  } else {
    kgen.unreachable
  }
  // CHECK-NEXT: ^bb5:
  // CHECK-NEXT: return
  kgen.return
}

}
