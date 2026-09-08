// RUN: kgen-opt -lower-async-functions -split-input-file %s | FileCheck %s

// COM: Verify Ramp + Resume + Async Calls are transformed correctly.
module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_hot_ramp(
kgen.func @coroutine(%arg0: i1, %arg1: index, %__result__: !kgen.pointer<index> byref_result) async -> index {
 // CHECK:      [[CORO:%.*]] = pop.aligned_alloc
 // CHECK:      [[V11:%.*]] = pop.pointer.bitcast [[CORO]]
 // CHECK-NEXT: hlcf.loop "_loop_0" ([[BLOCK_ARG:%.*]] =
 // Check that block arguments are stored in frame.
 // CHECK-NEXT: [[V9:%.*]] = kgen.struct.gep %0[[[#FRAME8:]]]
 // CHECK-NEXT: pop.store [[BLOCK_ARG]], [[V9]] : !kgen.pointer<index>

 // CHECK-NEXT: hlcf.loop "_loop_1" ([[BLOCK_ARG_INNER:%.*]] =
 // CHECK-NEXT: [[V10:%.*]] = kgen.struct.gep %0[[[#FRAME8 - 1]]]
 // CHECK-NEXT: pop.store [[BLOCK_ARG_INNER]], [[V10]] : !kgen.pointer<index>

 // Verify suspension point in nested loops is properly replaced and
 // unreachable is inserted to terminate unreachable blocks.
 // CHECK-NEXT:   [[COND0:%.*]] = pop.cast_from_builtin %arg2 : i1 to !kgen.scalar<bool>
 // CHECK-NEXT:   hlcf.if [[COND0]] {
 // CHECK-NEXT:     hlcf.break "_loop_1"
 // CHECK-NEXT:   } else {
 // CHECK-NEXT:     hlcf.yield
 // CHECK-NEXT:   }
 // CHECK-NEXT:   [[V15:%.*]] = kgen.param.constant: i32 = <1>
 // CHECK-NEXT:   [[V16:%.*]] = kgen.struct.gep [[CORO]][0]
 // CHECK-NEXT:   pop.store [[V15]], [[V16]] : !kgen.pointer<i32>
 // CHECK-NEXT:   kgen.return [[V11]]
 // CHECK-NEXT: }
 // CHECK-NEXT: kgen.call @print
 // CHECK-NEXT: hlcf.continue
 // CHECK-NEXT: }
 // CHECK-NEXT: kgen.unreachable
 hlcf.loop "_loop_0" (%arg3 = %arg1 : index) {
   hlcf.loop "_loop_1" (%arg2 = %arg1 : index) {
     %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
     hlcf.if %arg0_sb {
       hlcf.break "_loop_1"
     } else {
       hlcf.yield
     }
     co.suspend (%hdl) {
       co.suspend.end
     }
     kgen.call @print1(%arg2) : (index) -> ()
     hlcf.continue %arg2 : index
   }
   kgen.call @print(%arg0) : (i1) -> ()
   hlcf.continue %arg3 : index
 }
 co.suspend (%hdl) {
   co.suspend.end
 }
 %final = index.add %arg1, %arg1
 kgen.return %final : index
}

// Check that the operands of parents that are state 0 are replaced with constants. All other ops in state 0 will be erased.
// CHECK-LABEL:  kgen.func @coroutine_resume
// CHECK-NEXT:   [[UNDEF:%.*]] = kgen.param.constant = <#interp.uninitmem>
// CHECK-NEXT:   hlcf.loop "_loop_0" (%arg1 = [[UNDEF]] : index) {
kgen.func @trigger_creation(%arg0: i1, %arg1: index, %__result__: !kgen.pointer<index> byref_result) async {
   %coro = co.hot_invoke[(i1, index, !kgen.pointer<index> byref_result) async -> index: @coroutine](%arg0, %arg1, %__result__)
   kgen.return
}

}

// -----

// COM: Verify Loop With Await In Then Statement Is Correct

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine1_resume
kgen.func @coroutine1(%arg0: i1, %arg1: index, %arg2: index, %arg3: index) async -> index {
  // CHECK-NEXT: %idx3 = index.constant 3
  // CHECK-NEXT: [[V5:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
  // CHECK-NEXT: [[V6:%.*]] = pop.load [[V5]] : !kgen.pointer<index>

  // CHECK-NEXT: [[V7:%.*]] = kgen.call @foo(%idx3, [[V6]]) : (index, index) -> index
  // CHECK-NEXT: [[V8:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
  // CHECK-NEXT: pop.store [[V7]], [[V8]] : !kgen.pointer<index>
  %idx3 = index.constant 3
  %result = kgen.call @foo(%idx3, %arg1) : (index,index) -> index

  // CHECK-NEXT: hlcf.loop
  hlcf.loop "_loop_0" {
    // CHECK-NEXT: [[V19:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
    // CHECK-NEXT: [[V20:%.*]] = pop.load [[V19]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V21:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 + 1]]]
    // CHECK-NEXT: [[V22:%.*]] = pop.load [[V21]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V23:%.*]] = kgen.call @bar([[V20]], [[V22]]) : (index, index) -> index
    %result4 = kgen.call @bar(%result, %arg3): (index,index) -> index


    // CHECK-NEXT: [[V24:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 + 2]]]
    // CHECK-NEXT: [[V25:%.*]] = pop.load [[V24]] : !kgen.pointer<i1>
    // CHECK-NEXT: [[V25SB:%.*]] = pop.cast_from_builtin [[V25]] : i1 to !kgen.scalar<bool>
    // CHECK-NEXT: hlcf.if [[V25SB]] {
    // CHECK-NEXT:   kgen.param.constant
    // CHECK-NEXT:   kgen.struct.gep
    // CHECK-NEXT:   pop.store
    // CHECK-NEXT:   co.suspend {
    // CHECK-NEXT:     co.suspend.end
    // CHECK-NEXT:   }
    // CHECK-NEXT:   hlcf.yield
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   hlcf.break "_loop_0"
    // CHECK-NEXT: }
    %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
    hlcf.if %arg0_sb {
       co.suspend (%hdl) {
         co.suspend.end
       }
       hlcf.yield
    } else {
       hlcf.break "_loop_0"
    }
    // CHECK-NEXT: %idx3_0 = index.constant 3
    // CHECK-NEXT: [[V28:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 + 3]]]
    // CHECK-NEXT: [[V29:%.*]] = pop.load [[V28]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V30:%.*]] = kgen.call @foo(%idx3_0, [[V29]]) : (index, index) -> index
    %result6 = kgen.call @foo(%idx3, %arg2) : (index,index) -> index

    // CHECK-NEXT: hlcf.continue
    hlcf.continue
  }
  // CHECK-NEXT: }
  // CHECK-NEXT: [[V9:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
  // CHECK-NEXT: [[V10:%.*]] = pop.load [[V9]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V11:%.*]] = kgen.struct.gep %arg0[[[#FRAME8]]]
  // CHECK-NEXT: [[V12:%.*]] = pop.load [[V11]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.call @bar([[V10]], [[V12]]) : (index, index) -> index
  %result5 = kgen.call @bar(%result, %arg1): (index,index) -> index

  // CHECK-NEXT: [[V14:%.*]] = kgen.struct.gep %arg0[[[#PROMISE_IDX:]]]
  // CHECK-NEXT: [[PTR:%.*]] = kgen.struct.gep [[V14]][0]
  // CHECK-NEXT: pop.store [[V10]], [[PTR]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.return
  kgen.return %result : index
}

kgen.func @triggerCold(%arg0: i1, %arg1: index, %arg2: index, %arg3: index) {
  %coro = co.invoke[(i1, index, index, index) async -> index:@coroutine1](%arg0, %arg1, %arg2, %arg3)
  kgen.return
}
}

// -----

// COM: Verify Loop With Await In Else Statement Is Correct

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine5_resume
kgen.func @coroutine5(%arg0: i1, %arg1: index, %arg3: index) async -> index {
  // CHECK-NEXT: %idx3 = index.constant 3
  // CHECK-NEXT: [[V4:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
  // CHECK-NEXT: [[V5:%.*]] = pop.load [[V4]] : !kgen.pointer<index>
  // CHECK-NEXT: [[NOT_IN_FRAME:%.*]] = kgen.call @foo(%idx3, [[V5]]) : (index, index) -> index
  %idx3 = index.constant 3
  %result = kgen.call @foo(%idx3, %arg1) : (index,index) -> index
  hlcf.loop "_loop_0" {
     %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
     hlcf.if %arg0_sb {
       hlcf.yield
     } else {
       // CHECK: } else {
       // CHECK-NEXT: [[V17:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 + 2]]]
       // CHECK-NEXT: [[V18:%.*]] = pop.load [[V17]] : !kgen.pointer<index>
       // CHECK-NEXT: [[V19:%.*]] = kgen.call @bar([[NOT_IN_FRAME]], [[V18]]) : (index, index) -> index
       // CHECK-NEXT: kgen.param.constant
       // CHECK-NEXT: kgen.struct.gep
       // CHECK-NEXT: pop.store
       // CHECK-NEXT: co.suspend
       %result4 = kgen.call @bar(%result, %arg3): (index,index) -> index
       co.suspend (%hdl) {
         co.suspend.end
       }
       hlcf.break "_loop_0"
     }
     hlcf.continue
  }
  // CHECK:      [[V8:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
  // CHECK-NEXT: [[V9:%.*]] = pop.load [[V8]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V10:%.*]] = kgen.struct.gep %arg0[[[#PROMISE_IDX:]]]
  // CHECK-NEXT: [[PTR:%.*]] = kgen.struct.gep [[V10]][0]
  // CHECK-NEXT: pop.store [[V9]], [[PTR]]
  // CHECK-NEXT: kgen.return
  kgen.return %result : index
}

kgen.func @triggerCold(%arg0: i1, %arg1: index, %arg3: index) {
  %coro = co.invoke[(i1, index, index) async -> index:@coroutine5](%arg0, %arg1, %arg3)
  kgen.return
}

}

// -----

// COM: Verify Block With Multiple Awaits Is Correct

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine3_resume
kgen.func @coroutine3(%arg0: i1, %arg1: index, %arg3: index) async -> index {
  %idx3 = index.constant 3
  // CHECK: [[NIF:%.*]] = kgen.call @foo(%idx3, %{{.*}}) : (index, index) -> index
  %result = kgen.call @foo(%idx3, %arg1) : (index,index) -> index
  // CHECK: hlcf.loop "_loop_0"
  hlcf.loop "_loop_0" {
    // CHECK-NEXT: [[V13:%.*]] = kgen.struct.gep %arg0[[[#FRAME11:]]]
    // CHECK-NEXT: [[V14:%.*]] = pop.load [[V13]] : !kgen.pointer<i1>
    // CHECK-NEXT: [[V14SB:%.*]] = pop.cast_from_builtin [[V14]] : i1 to !kgen.scalar<bool>
    // CHECK-NEXT: hlcf.if [[V14SB]] {
    // CHECK-NEXT:   hlcf.yield
    // CHECK-NEXT: } else {
    // CHECK-NEXT: [[V15:%.*]] = kgen.struct.gep %arg0[[[#FRAME11 + 1]]]
    // CHECK-NEXT: [[V16:%.*]] = pop.load [[V15]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V17:%.*]] = kgen.call @bar([[NIF]], [[V16]]) : (index, index) -> index
    // CHECK-NEXT: [[V18:%.*]] = kgen.struct.gep %arg0[[[#FRAME11 - 3]]]
    // CHECK-NEXT: pop.store [[V17]], [[V18]] : !kgen.pointer<index>
    // CHECK-NEXT: kgen.param.constant
    // CHECK-NEXT: kgen.struct.gep
    // CHECK-NEXT: pop.store
    // CHECK-NEXT: co.suspend {
    // CHECK-NEXT:   co.suspend.end
    // CHECK-NEXT: }
    // CHECK-NEXT: [[V19:%.*]] = kgen.struct.gep %arg0[[[#FRAME11 - 3]]]
    // CHECK-NEXT: [[V20:%.*]] = pop.load [[V19]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V21:%.*]] = kgen.struct.gep %arg0[[[#FRAME11 + 1]]]
    // CHECK-NEXT: [[V22:%.*]] = pop.load [[V21]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V23:%.*]] = kgen.call @bar([[V20]], [[V22]]) : (index, index) -> index
    // CHECK-NEXT: kgen.param.constant
    // CHECK-NEXT: kgen.struct.gep
    // CHECK-NEXT: pop.store
    // CHECK-NEXT: co.suspend {
    // CHECK-NEXT:   co.suspend.end
    // CHECK-NEXT: }
    // CHECK-NEXT: hlcf.break "_loop_0"
    // CHECK-NEXT: }
     %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
     hlcf.if %arg0_sb {
        hlcf.yield
     } else {
         %result4 = kgen.call @bar(%result, %arg3): (index,index) -> index
         co.suspend (%hdl) {
           co.suspend.end
         }
         %result6 = kgen.call @bar(%result4, %arg3): (index,index) -> index
         co.suspend (%hdl) {
           co.suspend.end
         }
        hlcf.break "_loop_0"
     }
     hlcf.continue
  }
  kgen.return %result : index
}

kgen.func @triggerCold(%arg0: i1, %arg1: index, %arg3: index) {
  %coro = co.invoke[(i1, index, index) async -> index:@coroutine3](%arg0, %arg1, %arg3)
  kgen.return
}
}

// -----

// COM: Verify Nested Control Flow Is Correct

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_nested_resume
kgen.func @coroutine_nested(%arg0: i1, %arg1: index, %arg3: index) async -> index {
  %idx3 = index.constant 3
  %result = kgen.call @foo(%idx3, %arg1) : (index,index) -> index
  hlcf.loop "_loop_0" {
     %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
     hlcf.if %arg0_sb {
        hlcf.yield
     } else {
         %result4 = kgen.call @bar(%result, %arg3): (index,index) -> index
         // CHECK:      hlcf.loop "_loop_1" {
         // CHECK-NEXT: [[V23:%.*]] = kgen.struct.gep %arg0[[[#FRAME10:]]]
         // CHECK-NEXT: [[V24:%.*]] = pop.load [[V23]] : !kgen.pointer<i1>
         // CHECK-NEXT: [[V24SB:%.*]] = pop.cast_from_builtin [[V24]] : i1 to !kgen.scalar<bool>
         // CHECK-NEXT: hlcf.if [[V24SB]] {
         // CHECK-NEXT:   hlcf.yield
         // CHECK-NEXT: } else {
         // CHECK-NEXT:   kgen.param.constant
         // CHECK-NEXT:   kgen.struct.gep
         // CHECK-NEXT:   pop.store
         // CHECK-NEXT:   co.suspend {
         // CHECK-NEXT:     co.suspend.end
         // CHECK-NEXT:   }
         // CHECK-NEXT:   [[V25:%.*]] = kgen.struct.gep %arg0[[[#FRAME10 - 2]]]
         // CHECK-NEXT:   [[V26:%.*]] = pop.load [[V25]] : !kgen.pointer<index>
         // CHECK-NEXT:   [[V27:%.*]] = kgen.struct.gep %arg0[[[#FRAME10 + 1]]]
         // CHECK-NEXT:   [[V28:%.*]] = pop.load [[V27]] : !kgen.pointer<index>
         // CHECK-NEXT:   [[V29:%.*]] = kgen.call @bar([[V26]], [[V28]]) : (index, index) -> index
         // CHECK-NEXT:   hlcf.break "_loop_1"
         // CHECK-NEXT: }
         // CHECK-NEXT: hlcf.continue
         // CHECK-NEXT: }
         // CHECK-NEXT: [[V18:%.*]] = kgen.struct.gep %arg0[[[#FRAME10 - 2]]]
         // CHECK-NEXT: [[V19:%.*]] = pop.load [[V18]] : !kgen.pointer<index>
         // CHECK-NEXT: [[V20:%.*]] = kgen.struct.gep %arg0[[[#FRAME10 + 1]]]
         // CHECK-NEXT: [[V21:%.*]] = pop.load [[V20]] : !kgen.pointer<index>
         // CHECK-NEXT: [[V22:%.*]] = kgen.call @bar([[V19]], [[V21]]) : (index, index) -> index
         // CHECK-NEXT: hlcf.break "_loop_0"
         hlcf.loop "_loop_1" {
           %arg0_sb2 = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
           hlcf.if %arg0_sb2 {
             hlcf.yield
           } else {
             co.suspend (%hdl) {
               co.suspend.end
             }
             %result6 = kgen.call @bar(%result, %arg3): (index,index) -> index
             hlcf.break "_loop_1"
          }
          hlcf.continue
         }
         %result6 = kgen.call @bar(%result, %arg3): (index,index) -> index
         hlcf.break "_loop_0"
     }
     hlcf.continue
  }
  kgen.return %result : index
}

kgen.func @triggerCold(%arg0: i1, %arg1: index, %arg3: index) {
  %coro = co.invoke[(i1, index, index) async -> index:@coroutine_nested](%arg0, %arg1, %arg3)
  kgen.return
}
}

// -----

// COM: Verify that Await Code Loads/Stores From Frame.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine2_resume
kgen.func @coroutine2(%arg0: i1, %arg1: index, %arg3: index) async -> index {
  // CHECK-NEXT: %idx3 = index.constant 3
  // CHECK-NEXT: [[V4:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
  // CHECK-NEXT: [[V5:%.*]] = pop.load [[V4]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V2:%.*]] = kgen.call @foo(%idx3, [[V5]]) : (index, index) -> index
  // CHECK-NEXT: [[V7:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
  // CHECK-NEXT: pop.store [[V2]], [[V7]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.param.constant
  // CHECK-NEXT: kgen.struct.gep
  // CHECK-NEXT: pop.store
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT:   [[V8:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 + 1]]]
  // CHECK-NEXT:   [[V9:%.*]] = pop.load [[V8]]
  // CHECK-NEXT:   kgen.call @bar([[V2]], [[V9]]) : (index, index) -> index
  // CHECK-NEXT:   co.suspend.end
  // CHECK-NEXT:  }
  %idx3 = index.constant 3
  %result = kgen.call @foo(%idx3, %arg1) : (index,index) -> index
  co.suspend (%hdl) {
    %result2 = kgen.call @bar(%result, %arg3): (index,index) -> index
    co.suspend.end
  }
  kgen.return %result : index
}

kgen.func @triggerCold(%arg0: i1, %arg1: index, %arg3: index) {
  %coro = co.invoke[(i1, index, index) async -> index:@coroutine2](%arg0, %arg1, %arg3)
  kgen.return
}
}

// -----

// COM: Verify that Block Arguments Are Added To Frame.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_block_args3_resume
kgen.func @coroutine_block_args3(%arg0: index) async -> index {
  // CHECK-NEXT: [[V4:%.*]] = kgen.struct.gep %arg0[[[#FRAME6:]]]
  // CHECK-NEXT: [[V5:%.*]] = pop.load [[V4]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V6:%.*]] = hlcf.loop (%arg1 = [[V5]] : index) -> index {
  // CHECK-NEXT: [[V10:%.*]] = kgen.struct.gep %arg0[[[#FRAME6 - 1]]]
  // CHECK-NEXT:  pop.store %arg1, [[V10]] : !kgen.pointer<index>
  // CHECK-NEXT:  %idx0 = index.constant 0
  // CHECK-NEXT:  [[V11:%.*]] = index.cmp slt(%arg1, %idx0)
  // CHECK-NEXT:  [[V11SB:%.*]] = pop.cast_from_builtin [[V11]] : i1 to !kgen.scalar<bool>
  // CHECK-NEXT:  hlcf.if [[V11SB]] {
  // CHECK-NEXT:    kgen.param.constant
  // CHECK-NEXT:    kgen.struct.gep
  // CHECK-NEXT:    pop.store
  // CHECK-NEXT:    co.suspend {
  // CHECK-NEXT:      co.suspend.end
  // CHECK-NEXT:    }
  // CHECK-NEXT:    hlcf.yield
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    [[V14:%.*]] = kgen.struct.gep %arg0[[[#FRAME6 - 1]]]
  // CHECK-NEXT:    [[V15:%.*]] = pop.load [[V14]] : !kgen.pointer<index>
  // CHECK-NEXT:    hlcf.break [[V15]] : index
  // CHECK-NEXT:  }
  // CHECK-NEXT:  [[V12:%.*]] = kgen.struct.gep %arg0[[[#FRAME6 - 1]]]
  // CHECK-NEXT:  [[V13:%.*]] = pop.load [[V12]] : !kgen.pointer<index>
  // CHECK-NEXT:  hlcf.continue [[V13]] : index
  // CHECK-NEXT:  }
  %0 = hlcf.loop (%arg1 = %arg0 : index) -> index {
    %idx0 = index.constant 0
    %1 = index.cmp slt(%arg1, %idx0)
    %cb1 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
    hlcf.if %cb1 {
      co.suspend (%hdl) {
        co.suspend.end
      }
      hlcf.yield
    } else {
      hlcf.break %arg1 : index
    }
    hlcf.continue %arg1 : index
  }
  %idx1 = index.constant 1
  kgen.return %idx1 : index
}

kgen.func @triggerCold(%arg1: index) {
  %coro = co.invoke[(index) async -> index:@coroutine_block_args3](%arg1)
  kgen.return
}

}

// -----

// COM: Verify that Block Arguments Are Not Referenced Directly Across Suspension.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_block_args1_resume
kgen.func @coroutine_block_args1(%arg0: index) async -> index {
  // CHECK: [[V6:%.*]] = hlcf.loop (%arg1 = %{{.*}} : index) -> index {
  // CHECK: [[V10:%.*]] = kgen.struct.gep %arg0[[[#FRAME5:]]]
  // CHECK: pop.store %arg1, [[V10]] : !kgen.pointer<index>
  // CHECK: co.suspend {
  // CHECK: co.suspend.end
  // CHECK: }
  // CHECK: %idx0 = index.constant 0
  // CHECK: [[V11:%.*]] = kgen.struct.gep %arg0[[[#FRAME5]]]
  // CHECK: [[V12:%.*]] = pop.load [[V11]] : !kgen.pointer<index>
  // CHECK: [[V13:%.*]] = index.cmp slt([[V12]], %idx0)
  %0 = hlcf.loop (%arg5 = %arg0 : index) -> index {
    co.suspend (%hdl) {
      co.suspend.end
    }
    %idx0 = index.constant 0
    %1 = index.cmp slt(%arg5, %idx0)
    %cb1 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
    hlcf.if %cb1 {
      hlcf.yield
    } else {
      hlcf.break %arg5 : index
    }
    hlcf.continue %arg5 : index
  }
  %idx1 = index.constant 1
  kgen.return %idx1 : index
}

kgen.func @triggerCold(%arg1: index) {
  %coro = co.invoke[(index) async -> index:@coroutine_block_args1](%arg1)
  kgen.return
}

}

// -----

// COM: Verify that Block Arguments Are Not Put In Frame If Not Needed.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_block_args2_resume
kgen.func @coroutine_block_args2(%arg0: index) async -> index {
  // CHECK:      hlcf.loop (%arg1 = %{{.*}} : index) -> index {
  // CHECK-NEXT:   %idx0 = index.constant 0
  // CHECK-NEXT:   [[V10:%.*]] = index.cmp slt(%arg1, %idx0)
  // CHECK-NEXT:   [[V10SB:%.*]] = pop.cast_from_builtin [[V10]] : i1 to !kgen.scalar<bool>
  // CHECK-NEXT:   hlcf.if [[V10SB]] {
  // CHECK-NEXT:     hlcf.yield
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     hlcf.break %arg1 : index
  // CHECK-NEXT:   }
  // CHECK-NEXT:     hlcf.continue %arg1 : index
  // CHECK-NEXT:   }
  %0 = hlcf.loop (%arg5 = %arg0 : index) -> index {
    %idx0 = index.constant 0
    %1 = index.cmp slt(%arg5, %idx0)
    %cb1 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
    hlcf.if %cb1 {
      hlcf.yield
    } else {
      hlcf.break %arg5 : index
    }
    hlcf.continue %arg5 : index
  }
  co.suspend (%hdl) {
    co.suspend.end
  }
  %idx1 = index.constant 1
  kgen.return %idx1 : index
}

kgen.func @triggerCold(%arg1: index) {
  %coro = co.invoke[(index) async -> index:@coroutine_block_args2](%arg1)
  kgen.return
}

}

// -----

// COM: Verify that Unused Arguments Are Not Put In Frame.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @unused_args_resume
// CHECK-SAME: coroutineType = !kgen.struct<(i32, pointer<none>, (!kgen.pointer<none>) -> (), pointer<none>, pointer<none>, pointer<none>, struct<(index)>, index)>
kgen.func @unused_args(%arg0: index, %arg1: index) async -> index {
  co.suspend (%hdl) {
    co.suspend.end
  }
  %result = kgen.call @foo(%arg0) : (index) -> index
  kgen.return %result : index
}

kgen.func @triggerCold(%arg1: index, %arg2:index) {
  %coro = co.invoke[(index, index) async -> index:@unused_args](%arg1, %arg2)
  kgen.return
}
}

// -----

// COM: Test Try Raise

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @tryraise_resume
kgen.func @tryraise(%arg1: index, %arg2 : index) async -> index {
  // CHECK: [[NIF:%.*]] = kgen.call @foo1
  %result3 = kgen.call @foo1(%arg1) : (index) -> index
  lit.try "try0" {
    %result = kgen.call @bar(%arg2) : (index) -> i1
    %result_sb = pop.cast_from_builtin %result : i1 to !kgen.scalar<bool>
    hlcf.elif %result_sb {
      // CHECK: hlcf.elif {{%.*}} {
      // CHECK-NEXT: [[V7:%.*]] = kgen.struct.gep %arg0[[[#FRAME10:]]]
      // CHECK-NEXT: [[V8:%.*]] = pop.load [[V7]] : !kgen.pointer<index>
      // CHECK-NEXT: [[V9:%.*]] = kgen.call @foo([[V8]]) : (index) -> index
      // CHECK-NEXT: [[V10:%.*]] = kgen.struct.gep %arg0[[[#FRAME10 - 3]]]
      // CHECK-NEXT: pop.store [[V9]], [[V10]] : !kgen.pointer<index>
      // CHECK-NEXT: kgen.param.constant
      // CHECK-NEXT: kgen.struct.gep
      // CHECK-NEXT: pop.store
      // CHECK-NEXT: co.suspend {
      // CHECK-NEXT:   co.suspend.end
      // CHECK-NEXT: }
      // CHECK-NEXT: [[V11:%.*]] = kgen.struct.gep %arg0[[[#FRAME10 - 3]]]
      // CHECK-NEXT: [[V12:%.*]] = pop.load [[V11]] : !kgen.pointer<index>
      // CHECK-NEXT: lit.try.raise "try0" [[V12]] : index
      %result2 = kgen.call @foo(%arg2) : (index) -> index
      co.suspend (%hdl) {
        co.suspend.end
      }
      lit.try.raise "try0" %result2 : index
    } else {
      hlcf.yield
    }
    lit.try.yield
  } except (%e: index) {
    // CHECK:     } except (%arg1: index) {
    // CHECK-NEXT: [[V10:%.*]] = kgen.struct.gep %arg0[[[#FRAME10 - 2]]]
    // CHECK-NEXT: pop.store %arg1, [[V10]] : !kgen.pointer<index>
    // CHECK-NEXT: kgen.param.constant
    // CHECK-NEXT: kgen.struct.gep
    // CHECK-NEXT: pop.store
    // CHECK-NEXT: co.suspend {
    // CHECK-NEXT: co.suspend.end
    // CHECK-NEXT: }
    // CHECK-NEXT: [[V11:%.*]] = kgen.struct.gep %arg0[[[#FRAME10 - 2]]]
    // CHECK-NEXT: [[V12:%.*]] = pop.load [[V11]]
    // CHECK:      pop.store [[V12]], {{.*}} : !kgen.pointer<index>
    // CHECK-NEXT: kgen.return
    co.suspend (%hdl) {
      co.suspend.end
    }
    kgen.return %e : index
  } else {
    // CHECK: } else {
    // CHECK:      pop.store [[NIF]], {{.*}}
    // CHECK-NEXT: kgen.return
    kgen.return %result3 : index
  }
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

kgen.func @triggerCold(%arg1: index, %arg2:index) {
  %coro = co.invoke[(index, index) async -> index:@tryraise](%arg1, %arg2)
  kgen.return
}

}

// -----

// COM: Verify Set Error/Results Op is Lowered Correctly

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @throwing_coroutine_resume
kgen.func @throwing_coroutine(%__error__: !kgen.pointer<index> byref_error,
                              %__result__: !kgen.pointer<index> byref_result) throws|async -> i1 {
  // CHECK-NEXT: [[RESSLOT:%.*]] = kgen.struct.gep %arg0[[[#RESULT:]]]
  // CHECK-NEXT: [[RES:%.*]] = pop.load [[RESSLOT]] : !kgen.pointer<pointer<none>>
  // CHECK-NEXT: [[RESTYPED:%.*]] = pop.pointer.bitcast [[RES]] : !kgen.pointer<none> to !kgen.pointer<index>
  // CHECK-NEXT: [[V4:%.*]] = kgen.call @populate([[RESTYPED]]) : (!kgen.pointer<index> byref_result) -> i1
  // CHECK-NEXT: kgen.call @use([[RESTYPED]]) : (!kgen.pointer<index> imm_mem) -> i1
  %0 = kgen.call @populate(%__result__) : (!kgen.pointer<index> byref_result) -> i1
  %2 = kgen.call @use(%__result__) : (!kgen.pointer<index> imm_mem) -> i1
  %cb0 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
  hlcf.if %cb0 {
    // CHECK-NEXT: [[V4SB:%.*]] = pop.cast_from_builtin [[V4]] : i1 to !kgen.scalar<bool>
    // CHECK-NEXT: hlcf.if [[V4SB]] {
    // CHECK-NEXT: [[ERRSLOT:%.*]] = kgen.struct.gep %arg0[[[#ERROR:]]]
    // CHECK-NEXT: [[ERR:%.*]] = pop.load [[ERRSLOT]] : !kgen.pointer<pointer<none>>
    // CHECK-NEXT: [[TYPEDERR:%.*]] = pop.pointer.bitcast [[ERR]] : !kgen.pointer<none> to !kgen.pointer<index>
    // CHECK-NEXT: kgen.call @populate([[TYPEDERR]]) : (!kgen.pointer<index> byref_result) -> i1
    %1 = kgen.call @populate(%__error__) : (!kgen.pointer<index> byref_result) -> i1
    kgen.return %1 : i1
  } else {
    hlcf.yield
  }
  co.suspend (%hdl) {
    co.suspend.end
  }
  %1 = kgen.call @populate(%__result__) : (!kgen.pointer<index> byref_result) -> i1
  kgen.return %1 : i1
}

// CHECK-LABEL: kgen.func @call_throwing_coro
kgen.func @call_throwing_coro() {
  %size = index.constant 1
  %align = index.constant 8
  // CHECK:      [[CONT:%.*]] = kgen.call @throwing_coroutine_ramp()
  // CHECK-NEXT: [[ERR:%.*]] = pop.aligned_alloc %idx8, %idx1 : <index>
  // CHECK-NEXT: [[RES:%.*]] = pop.aligned_alloc %idx8, %idx1 : <index>
  // CHECK-NEXT: [[ERRORSLOT:%.*]] = kgen.struct.gep [[CONT]][[[#ERROR]]]
  // CHECK-NEXT: [[TYPED_ERRORSLOT:%.*]] = pop.pointer.bitcast [[ERRORSLOT]] : !kgen.pointer<pointer<none>> to !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.store [[ERR]], [[TYPED_ERRORSLOT]] : !kgen.pointer<pointer<index>>
  // CHECK-NEXT: [[RESSLOT:%.*]] = kgen.struct.gep [[CONT]][[[#RESULT]]]
  // CHECK-NEXT: [[TYPED_RESSLOT:%.*]] = pop.pointer.bitcast [[RESSLOT]] : !kgen.pointer<pointer<none>> to !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.store [[RES]], [[TYPED_RESSLOT]] : !kgen.pointer<pointer<index>>
  // CHECK-NEXT: kgen.return
  %coro = co.invoke[(!kgen.pointer<index> byref_error, !kgen.pointer<index> byref_result) throws|async -> i1: @throwing_coroutine]()
  %0 = pop.aligned_alloc %align, %size : <index>
  %1 = pop.aligned_alloc %align, %size : <index>
  co.set_byref_error_result %coro(%1, %0) : !co.routine, !kgen.pointer<index>, !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @use2
kgen.func @use2(%a: !co.routine) -> i1 {
  %true = index.bool.constant true
  kgen.return %true : i1
}

// CHECK-LABEL: kgen.func @opaque_coro
kgen.func @opaque_coro(%coro: !co.routine, %arg1: !kgen.pointer<index>, %arg2: !kgen.pointer<index>) {
  // CHECK-NEXT: kgen.call @use2(%arg0)
  %2 = kgen.call @use2(%coro) : (!co.routine) -> i1

  // CHECK-NEXT: [[v3:%.*]] = kgen.struct.gep %arg0[[[#ERROR]]]
  // CHECK-NEXT: [[v4:%.*]] = pop.pointer.bitcast [[v3]] : !kgen.pointer<pointer<none>> to !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.store %arg1, [[v4]] : !kgen.pointer<pointer<index>>
  // CHECK-NEXT: [[v5:%.*]] = kgen.struct.gep %arg0[[[#RESULT]]]
  // CHECK-NEXT: [[v6:%.*]] = pop.pointer.bitcast [[v5]] : !kgen.pointer<pointer<none>> to !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.store %arg2, [[v6]] : !kgen.pointer<pointer<index>>
  co.set_byref_error_result %coro(%arg2, %arg1) : !co.routine, !kgen.pointer<index>, !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @no_error_slot
kgen.func @no_error_slot(%arg0: !co.routine, %arg1: !kgen.pointer<index>) {
  // CHECK-NEXT: [[v5:%.*]] = kgen.struct.gep %arg0[[[#RESULT]]]
  // CHECK-NEXT: [[v6:%.*]] = pop.pointer.bitcast [[v5]] : !kgen.pointer<pointer<none>> to !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.store %arg1, [[v6]] : !kgen.pointer<pointer<index>>
  co.set_byref_error_result %arg0(%arg1) : !co.routine, !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @set_byref_none
kgen.func @set_byref_none(%arg0: !co.routine, %arg1: !kgen.pointer<none>) {
  // CHECK-NOT: kgen.struct.gep %arg0[[[#RESULT]]]
  co.set_byref_error_result %arg0(%arg1) : !co.routine, !kgen.pointer<none>
  co.set_byref_error_result %arg0(%arg1, %arg1) : !co.routine, !kgen.pointer<none>, !kgen.pointer<none>
  kgen.return
}

}

// -----

// COM: Stack Allocations Are Lowered To Frame Allocations

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_resume
kgen.func @coroutine(%arg1: index, %arg2: index) async -> index {
  %0 = pop.stack_allocation 2 x index marked
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
  %idx1 = index.constant 1
  // CHECK-NEXT: %idx1 = index.constant 1
  // CHECK-NEXT: [[V1:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
  // CHECK-NEXT: [[V2:%.*]] = pop.load [[V1]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V3:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
  // CHECK-NEXT: [[V4:%.*]] = pop.pointer.bitcast [[V3]] : !kgen.pointer<array<2, index>> to !kgen.pointer<index>
  // CHECK-NEXT: pop.store [[V2]], [[V4]] : !kgen.pointer<index>
  // CHECK-NEXT: [[V5:%.*]] = pop.offset [[V4]][%idx1] : !kgen.pointer<index>
  // CHECK-NEXT: [[V6:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 + 1]]]
  // CHECK-NEXT: [[V7:%.*]] = pop.load [[V6]]
  // CHECK-NEXT: pop.store [[V7]], [[V5]]
  // CHECK-NEXT: kgen.call @use([[V5]])
  // CHECK-NEXT: kgen.call @use([[V4]])
  pop.store %arg1, %0 : !kgen.pointer<index>
  %1 = pop.offset %0[%idx1] : !kgen.pointer<index>
  pop.store %arg2, %1 : !kgen.pointer<index>
  %3 = kgen.call @use(%1) : (!kgen.pointer<index>) -> index
  %22 = kgen.call @use(%0) : (!kgen.pointer<index>) -> index
  co.suspend (%hdl) {
    co.suspend.end
  }
  // CHECK: [[V8:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
  // CHECK-NEXT: [[V9:%.*]] = pop.pointer.bitcast [[V8]] : !kgen.pointer<array<2, index>> to !kgen.pointer<index>
  // CHECK-NEXT: kgen.call @use([[V9]]) : (!kgen.pointer<index>) -> index
  // CHECK-NOT:  pop.stack_alloc.lifetime
  %2 = kgen.call @use(%0) : (!kgen.pointer<index>) -> index
  pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
  co.suspend (%hdl) {
    co.suspend.end
  }
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

kgen.func @triggerCold(%arg1: index, %arg2:index) {
  %coro = co.invoke[(index, index) async -> index:@coroutine](%arg1, %arg2)
  kgen.return
}

}

// -----

// COM: Stack Allocations Are Not Lowered To Frame Allocations If Lifetime Contained In State

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_resume
kgen.func @coroutine(%arg1: index) async -> index {
  %0 = pop.stack_allocation 1 x index marked
  co.suspend (%hdl) {
    co.suspend.end
  }
  // CHECK:      co.suspend
  // CHECK-NEXT:   co.suspend.end
  // CHECK-NEXT: }
  // CHECK-NEXT: [[V1:%.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_alloc.lifetime.start
  // CHECK-NEXT: [[V2:%.*]] = kgen.struct.gep %arg0[[[#FRAME7:]]]
  // CHECK-NEXT: [[V3:%.*]] = pop.load [[V2]] : !kgen.pointer<index>
  // CHECK-NEXT: pop.store [[V3]], [[V1]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.call @use([[V1]]) : (!kgen.pointer<index>) -> index
  // CHECK:  pop.stack_alloc.lifetime
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
  pop.store %arg1, %0 : !kgen.pointer<index>
  %2 = kgen.call @use(%0) : (!kgen.pointer<index>) -> index
  pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
  kgen.return %2 : index
}

kgen.func @triggerCold(%arg1: index) {
  %coro = co.invoke[(index) async -> index:@coroutine](%arg1)
  kgen.return
}

}

// -----

// COM: Stack Allocations Are Lowered To Frame Allocations When Stack Allocation Not Used Outside Lifetime End

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_resume
kgen.func @coroutine(%arg1: index, %arg2: index) async -> index {
  %0 = pop.stack_allocation 2 x index marked
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
  // CHECK-NEXT: %idx1 = index.constant 1
  %idx1 = index.constant 1
  // Extract pointer to inline frame memory instead of stack allocation.
  // CHECK-NEXT: [[V0:%.*]] = kgen.struct.gep %arg0[[[#FRAME7:]]]
  // CHECK-NEXT: [[V1:%.*]] = pop.pointer.bitcast [[V0]] : !kgen.pointer<array<2, index>> to !kgen.pointer<index>


  // CHECK-NEXT: [[V2:%.*]] = pop.offset [[V1]][%idx1] : !kgen.pointer<index>
  // CHECK-NEXT: [[V3:%.*]] = kgen.struct.gep %arg0[[[#FRAME7 + 1]]]
  // CHECK-NEXT: [[V4:%.*]] = pop.load [[V3]]
  // CHECK-NEXT: pop.store [[V4]], [[V2]]
  %1 = pop.offset %0[%idx1] : !kgen.pointer<index>
  pop.store %arg2, %1 : !kgen.pointer<index>
  co.suspend (%hdl) {
    co.suspend.end
  }
  // CHECK:      %idx1_0 = index.constant 1
  // CHECK-NEXT: [[STACK_MEM:%.*]] = kgen.struct.gep %arg0[[[#FRAME7]]]
  // CHECK-NEXT: [[V6:%.*]] = pop.pointer.bitcast [[STACK_MEM]]
  // CHECK-NEXT: [[V7:%.*]] = pop.offset [[V6]][%idx1_0]
  // CHECK-NEXT: [[V8:%.*]] = kgen.call @use([[V7]]) : (!kgen.pointer<index>) -> index
  // CHECK-NOT:  pop.stack_alloc.lifetime
  %2 = kgen.call @use(%1) : (!kgen.pointer<index>) -> index
  pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
  kgen.return %2 : index
}

kgen.func @triggerCold(%arg1: index, %arg2: index) {
  %coro = co.invoke[(index, index) async -> index:@coroutine](%arg1, %arg2)
  kgen.return
}

}

// -----

// COM: Stack Allocations Of Size 1 Do Not Have Array Types

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_resume
kgen.func @coroutine(%arg1: index) async -> index {
  // CHECK-NEXT: [[STACK_ALLOC:%.*]] = kgen.struct.gep %arg0[[[#FRAME7:]]]
  // CHECK-NEXT: [[SLOT:%.*]] = kgen.struct.gep [[STACK_ALLOC]][1] : <struct<(index, index)>>
  // CHECK-NEXT: [[ARG_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME7 + 1]]]
  // CHECK-NEXT: [[ARG:%.*]] = pop.load [[ARG_SLOT]] : !kgen.pointer<index>
  // CHECK-NEXT: pop.store [[ARG]], [[SLOT]] : !kgen.pointer<index>
  %0 = pop.stack_allocation 1 x !kgen.struct<(index, index)> marked
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<struct<(index, index)>>
  %1 = kgen.struct.gep %0[1] : <struct<(index, index)>>
  pop.store %arg1, %1 : !kgen.pointer<index>
  co.suspend (%hdl) {
    co.suspend.end
  }
  // CHECK:       [[STACK_ALLOC2:%.*]] = kgen.struct.gep %arg0[[[#FRAME7]]]
  // CHECK-NEXT:  kgen.call @use([[STACK_ALLOC2]])
  %2 = kgen.call @use(%0) : (!kgen.pointer<struct<(index, index)>>) -> index
  pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<struct<(index, index)>>
  kgen.return %2 : index
}

kgen.func @triggerCold(%arg1: index) {
  %coro = co.invoke[(index) async -> index:@coroutine](%arg1)
  kgen.return
}

}

// -----


// COM: Do not remove lifetime markers of stack allocations not added to frame.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_resume
kgen.func @coroutine(%arg1: index) async -> index {
  %0 = pop.stack_allocation 1 x index marked
  co.suspend (%hdl) {
      co.suspend.end
  }
  // CHECK: pop.stack_alloc.lifetime.start
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
  pop.store %arg1, %0 : !kgen.pointer<index>
  %2 = kgen.call @use(%0) : (!kgen.pointer<index>) -> index
  // CHECK: pop.stack_alloc.lifetime.end
  pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
  kgen.return %2 : index
}

kgen.func @triggerCold(%arg1: index) {
  %coro = co.invoke[(index) async -> index:@coroutine](%arg1)
  kgen.return
}

}

// -----

// COM: Stack Allocation With Multiple Lifetimes

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @in_frame_resume
kgen.func @in_frame(%arg1: index, %arg2: index) async -> index {
  // CHECK-NEXT: kgen.param.constant: i32 = <1>
  // CHECK-NEXT: kgen.struct.gep
  // CHECK-NEXT: pop.store
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT: co.suspend.end
  // CHECK-NEXT: }
  %0 = pop.stack_allocation 1 x index marked
  co.suspend (%hdl) {
      co.suspend.end
  }
  // CHECK:      [[ARG_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
  // CHECK-NEXT: [[ARG:%.*]] = pop.load [[ARG_SLOT]] : !kgen.pointer<index>
  // CHECK-NEXT: [[FRAME_SA:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
  // CHECK-NEXT: pop.store [[ARG]], [[FRAME_SA]]
  // CHECK-NEXT: kgen.call @use([[FRAME_SA]])
  // CHECK-NEXT: kgen.param.constant: i32 = <2>
  // CHECK-NEXT: kgen.struct.gep
  // CHECK-NEXT: pop.store
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT:   co.suspend.end
  // CHECK-NEXT: }
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
  pop.store %arg1, %0 : !kgen.pointer<index>
  %2 = kgen.call @use(%0) : (!kgen.pointer<index>) -> index
  pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>

  co.suspend (%hdl) {
    co.suspend.end
  }

  // CHECK-NEXT: [[ARG2_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 + 1]]]
  // CHECK-NEXT: [[ARG2:%.*]] = pop.load [[ARG2_SLOT]] : !kgen.pointer<index>
  // CHECK-NEXT: [[FRAME_SA:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
  // CHECK-NEXT: pop.store [[ARG2]], [[FRAME_SA]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.call @use([[FRAME_SA]]) : (!kgen.pointer<index>) -> index
  // CHECK-NEXT: [[ARG1_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME8]]]
  // CHECK-NEXT: [[ARG1:%.*]] = pop.load [[ARG1_SLOT]]
  // CHECK-NEXT: pop.store [[ARG1]], [[FRAME_SA]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.param.constant: i32 = <3>
  // CHECK-NEXT: kgen.struct.gep
  // CHECK-NEXT: pop.store
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT:   co.suspend.end
  // CHECK-NEXT: }
  // CHECK-NEXT: [[FRAME_SA2:%.*]] = kgen.struct.gep %arg0[[[#FRAME8 - 1]]]
  // CHECK-NEXT: kgen.call @use([[FRAME_SA2]]) : (!kgen.pointer<index>) -> index
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
  pop.store %arg2, %0 : !kgen.pointer<index>
  %3 = kgen.call @use(%0) : (!kgen.pointer<index>) -> index
  pop.store %arg1, %0 : !kgen.pointer<index>

  co.suspend (%hdl) {
    co.suspend.end
  }
  %4 = kgen.call @use(%0) : (!kgen.pointer<index>) -> index
  pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

kgen.func @triggerCold(%arg1: index, %arg2: index) {
  %coro = co.invoke[(index, index) async -> index:@in_frame](%arg1, %arg2)
  kgen.return
}

}

// -----

// COM: Stack Allocations With Control Flow In Frame

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @in_frame_cf_resume
kgen.func @in_frame_cf(%arg1: index, %arg2: index, %arg3: i1) async -> index {
  // CHECK-NEXT: [[ARG3_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME9:]]]
  // CHECK-NEXT: [[ARG3:%.*]] = pop.load [[ARG3_SLOT]] : !kgen.pointer<i1>
  // CHECK-NEXT: [[ARG3SB:%.*]] = pop.cast_from_builtin [[ARG3]] : i1 to !kgen.scalar<bool>
  // CHECK-NEXT: hlcf.elif [[ARG3SB]] {
  // CHECK-NEXT: [[ARG2_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME9 + 1]]]
  // CHECK-NEXT: [[ARG2:%.*]] = pop.load [[ARG2_SLOT]] : !kgen.pointer<index>
  // CHECK-NEXT: [[SA:%.*]] = kgen.struct.gep %arg0[[[#FRAME9 - 1]]]
  // CHECK-NEXT: pop.store [[ARG2]], [[SA]] : !kgen.pointer<index>
  // CHECK-NEXT: hlcf.yield
  // CHECK-NEXT: } else {
  // CHECK-NEXT: [[ARG1_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME9 + 2]]]
  // CHECK-NEXT: [[ARG1:%.*]] = pop.load [[ARG1_SLOT]] : !kgen.pointer<index>
  // CHECK-NEXT: [[SA2:%.*]] = kgen.struct.gep %arg0[[[#FRAME9 - 1]]]
  // CHECK-NEXT: pop.store [[ARG1]], [[SA2]] : !kgen.pointer<index>
  // CHECK-NEXT: hlcf.yield
  // CHECK-NEXT: }
  %0 = pop.stack_allocation 1 x index marked
  %arg3_sb = pop.cast_from_builtin %arg3 : i1 to !kgen.scalar<bool>
  hlcf.elif %arg3_sb {
    pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
    pop.store %arg2, %0 : !kgen.pointer<index>
    hlcf.yield
  } else {
    pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
    pop.store %arg1, %0 : !kgen.pointer<index>
    hlcf.yield
  }
  // CHECK-NEXT: kgen.param.constant: i32 = <1>
  // CHECK-NEXT: kgen.struct.gep
  // CHECK-NEXT: pop.store
  // CHECK-NEXT: co.suspend
  // CHECK-NEXT: co.suspend.end
  // CHECK-NEXT: }
  co.suspend (%hdl) {
    co.suspend.end
  }
  %2 = kgen.call @use(%0) : (!kgen.pointer<index>) -> index
  // CHECK-NEXT: [[SA3:%.*]] = kgen.struct.gep %arg0[[[#FRAME9 - 1]]]
  // CHECK-NEXT: kgen.call @use([[SA3]]) : (!kgen.pointer<index>) -> index
  // CHECK-NEXT: [[ARG3_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME9]]]
  // CHECK-NEXT: [[ARG3:%.*]] = pop.load [[ARG3_SLOT]] : !kgen.pointer<i1>
  // CHECK-NEXT: [[ARG3SB:%.*]] = pop.cast_from_builtin [[ARG3]] : i1 to !kgen.scalar<bool>
  // CHECK-NEXT: hlcf.elif [[ARG3SB]] {
  // CHECK-NEXT:   hlcf.yield
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   hlcf.yield
  // CHECK-NEXT: }
  %arg3_sb2 = pop.cast_from_builtin %arg3 : i1 to !kgen.scalar<bool>
  hlcf.elif %arg3_sb2 {
    pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
    hlcf.yield
  } else {
    pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
    hlcf.yield
  }
  kgen.return %2 : index
}

kgen.func @triggerCold(%arg1: index, %arg2: index, %arg3: i1) {
  %coro = co.invoke[(index, index, i1) async -> index: @in_frame_cf](%arg1, %arg2, %arg3)
  kgen.return
}

}

// -----

// COM: Stack Allocations With Control Flow Not In Frame

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @not_in_frame_cf_resume
kgen.func @not_in_frame_cf(%arg1: index, %arg2: index, %arg3: i1) async -> index {
  // CHECK-NEXT: kgen.param.constant: i32 = <1>
  // CHECK-NEXT: kgen.struct.gep
  // CHECK-NEXT: pop.store
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT: co.suspend.end
  // CHECK-NEXT: }
  // CHECK-NEXT: [[ARG3_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
  // CHECK-NEXT: [[ARG3:%.*]] = pop.load [[ARG3_SLOT]] : !kgen.pointer<i1>
  // CHECK-NEXT: [[ARG3SB:%.*]] = pop.cast_from_builtin [[ARG3]] : i1 to !kgen.scalar<bool>
  // CHECK-NEXT: hlcf.if [[ARG3SB]] {
  // CHECK-NEXT: kgen.param.constant: i32 = <2>
  // CHECK-NEXT: kgen.struct.gep
  // CHECK-NEXT: pop.store
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT:  co.suspend.end
  // CHECK-NEXT: }
  // CHECK-NEXT: [[ARG3_SLOT2:%.*]] = kgen.struct.gep %arg0[[[#FRAME8]]]
  // CHECK-NEXT: [[ARG3_2:%.*]] = pop.load [[ARG3_SLOT2]] : !kgen.pointer<i1>
  // CHECK-NEXT: [[ARG3SB2:%.*]] = pop.cast_from_builtin [[ARG3_2]] : i1 to !kgen.scalar<bool>
  // CHECK-NEXT: [[SA:%.*]] = pop.stack_allocation 1 x index

  // CHECK: pop.stack_alloc.lifetime.start([[SA]])
  // CHECK: pop.stack_alloc.lifetime.start([[SA]])
  // CHECK: pop.stack_alloc.lifetime.start([[SA]])
  // CHECK: pop.stack_alloc.lifetime.end([[SA]])
  %0 = pop.stack_allocation 1 x index marked
  co.suspend (%hdl) {
    co.suspend.end
  }
  %arg3_sb1 = pop.cast_from_builtin %arg3 : i1 to !kgen.scalar<bool>
  hlcf.if %arg3_sb1 {
    co.suspend (%hdl) {
      co.suspend.end
    }
    %arg3_sb2 = pop.cast_from_builtin %arg3 : i1 to !kgen.scalar<bool>
    hlcf.elif %arg3_sb2 {
      %arg3_sb3 = pop.cast_from_builtin %arg3 : i1 to !kgen.scalar<bool>
      hlcf.if %arg3_sb3 {
        pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
        pop.store %arg1, %0 : !kgen.pointer<index>
        hlcf.yield
      } else {
        pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
        pop.store %arg2, %0 : !kgen.pointer<index>
        hlcf.yield
      }
      hlcf.yield
    } else {
      pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
      pop.store %arg1, %0 : !kgen.pointer<index>
      hlcf.yield
    }
    %2 = kgen.call @use(%0) : (!kgen.pointer<index>) -> index
    pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
    hlcf.yield
  } else {
    hlcf.yield
  }
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

kgen.func @triggerCold(%arg1: index, %arg2: index, %arg3: i1) {
  %coro = co.invoke[(index, index, i1) async -> index:@not_in_frame_cf](%arg1, %arg2, %arg3)
  kgen.return
}

}

// -----

// COM: LifetimeMarkers With Multiple Operands.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @in_frame_resume
kgen.func @in_frame(%arg1: index) async -> index {
  // CHECK-NOT: pop.stack_alloc.lifetime.start
  // CHECK-NOT: pop.stack_alloc.lifetime.end
  %0 = pop.stack_allocation 1 x index marked
  %1 = pop.stack_allocation 1 x index marked
  pop.stack_alloc.lifetime.start(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>
  pop.store %arg1, %0 : !kgen.pointer<index>
  pop.store %arg1, %1 : !kgen.pointer<index>
  co.suspend (%hdl) {
    co.suspend.end
  }
  %2 = kgen.call @use(%0, %1) : (!kgen.pointer<index>, !kgen.pointer<index>) -> index
  pop.stack_alloc.lifetime.end(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>
  kgen.return %2 : index
}

// CHECK-LABEL: kgen.func @not_in_frame_resume
kgen.func @not_in_frame(%arg1: index) async -> index {
  // CHECK: [[V1:%.*]] = pop.stack_allocation 1 x index
  // CHECK: [[V2:%.*]] = pop.stack_allocation 1 x index
  %0 = pop.stack_allocation 1 x index marked
  %1 = pop.stack_allocation 1 x index marked
  co.suspend (%hdl) {
    co.suspend.end
  }
  // CHECK: pop.stack_alloc.lifetime.start
  pop.stack_alloc.lifetime.start(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>
  pop.store %arg1, %0 : !kgen.pointer<index>
  pop.store %arg1, %1 : !kgen.pointer<index>
  %2 = kgen.call @use(%0, %1) : (!kgen.pointer<index>, !kgen.pointer<index>) -> index
  // CHECK: pop.stack_alloc.lifetime.end
  pop.stack_alloc.lifetime.end(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>

  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @multiple_lifetimes_frame_resume
kgen.func @multiple_lifetimes_frame(%arg1: index, %arg2: i1) async -> index {
  // CHECK-NEXT: kgen.param.constant: i32 = <1>
  // CHECK-NEXT: kgen.struct.gep
  // CHECK-NEXT: pop.store
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT:   co.suspend.end
  // CHECK-NEXT: }
  // CHECK-NEXT: kgen.struct.gep %arg0
  // CHECK-NEXT: pop.load
  // CHECK-NEXT: pop.cast_from_builtin
  // CHECK-NEXT: [[V1:%.*]] = pop.stack_allocation 1 x index
  %0 = pop.stack_allocation 1 x index marked
  %1 = pop.stack_allocation 1 x index marked
  co.suspend (%hdl) {
    co.suspend.end
  }
  // CHECK: hlcf.if {{.*}} {
  // CHECK-NEXT: [[V10:%.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_alloc.lifetime.start([[V1]], [[V10]])
  %arg2_sb = pop.cast_from_builtin %arg2 : i1 to !kgen.scalar<bool>
  hlcf.if %arg2_sb {
    pop.stack_alloc.lifetime.start(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>
    pop.store %arg1, %0 : !kgen.pointer<index>
    pop.store %arg1, %1 : !kgen.pointer<index>
    %2 = kgen.call @use(%0, %1) : (!kgen.pointer<index>, !kgen.pointer<index>) -> index
    pop.stack_alloc.lifetime.end(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>
    hlcf.yield
  } else {
    hlcf.yield
  }
  // CHECK: pop.stack_alloc.lifetime.start([[V1]])
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
  pop.store %arg1, %0 : !kgen.pointer<index>
  %4 = kgen.call @use(%0, %0) : (!kgen.pointer<index>, !kgen.pointer<index>) -> index
  pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
  co.suspend (%hdl) {
    co.suspend.end
  }
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

kgen.func @triggerCold(%arg1: index, %arg2: i1) {
  %coro = co.invoke[(index, i1) async -> index:@multiple_lifetimes_frame](%arg1, %arg2)
  %coro2 = co.invoke[(index) async -> index:@not_in_frame](%arg1)
  %coro3 = co.invoke[(index) async -> index:@in_frame](%arg1)
  kgen.return
}

}

// -----


// COM: Lower GetCallbackPtrOp

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @coroutine1(%arg0: i1) async -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

kgen.func @callback(%arg0: !kgen.pointer<none>) -> !kgen.none {
  %none = kgen.param.constant: !kgen.none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.func @coroutine_resume
kgen.func @coroutine(%arg0: i1) async -> index {
  %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
  hlcf.if %arg0_sb {
    %idx1 = index.constant 1
    kgen.return %idx1 : index
  } else {
    hlcf.yield
  }
  %true = index.bool.constant true
  // CHECK:      [[CORO:%.*]] = kgen.call @coroutine1_ramp(%true)
  // CHECK-NEXT: [[CALLBACK:%.*]] = kgen.create_closure[(!kgen.pointer<none>) -> (): @callback]()
  // CHECK-NEXT: [[SLOT:%.*]] = kgen.struct.gep [[CORO]][[[#FRAME2:]]]
  // CHECK-NEXT: [[CAST:%.*]] = pop.pointer.bitcast [[SLOT]]
  // CHECK-NEXT: [[SLOT2:%.*]] = kgen.struct.gep [[CAST]][0] : <struct<((!kgen.pointer<none>) -> (), pointer<none>)>>
  // CHECK-NEXT: pop.store [[CALLBACK]], [[SLOT2]] : !kgen.pointer<(!kgen.pointer<none>) -> ()>
  // CHECK-NEXT: kgen.param.constant: i32 = <1>
  // CHECK-NEXT: kgen.struct.gep
  // CHECK-NEXT: pop.store
  // CHECK-NEXT: co.suspend {
  %coro = co.invoke[(i1) async -> index: @coroutine1](%true)
  %callback = kgen.create_closure[(!kgen.pointer<none>) -> (): @callback]()
  %ptr = co.get_callback_ptr %coro : <struct<(!kgen.generator<(!kgen.pointer<none>) -> ()>, pointer<none>)>>
  %callbackSlot = kgen.struct.gep %ptr[0] : <struct<(!kgen.generator<(!kgen.pointer<none>) -> ()>, pointer<none>)>>
  pop.store %callback, %callbackSlot : !kgen.pointer<!kgen.generator<(!kgen.pointer<none>) -> ()>>
  co.suspend (%hdl) {
    co.suspend.end
  }
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

kgen.func @triggerCold(%arg0:i1) {
  %coro = co.invoke[(i1) async -> index:@coroutine](%arg0)
  kgen.return
}

}

// -----

// COM: Lower DestroyOp

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @coroutine1(%arg0: i1) async -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @coroutine_resume
kgen.func @coroutine(%arg0: i1) async -> index {
  %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
  hlcf.if %arg0_sb {
    %idx1 = index.constant 1
    kgen.return %idx1 : index
  } else {
    hlcf.yield
  }
  %true = index.bool.constant true
  // CHECK: [[CORO:%.*]] = kgen.call @coroutine1_ramp(%true)
  // CHECK-NEXT: [[CORO_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
  // CHECK-NEXT: pop.store [[CORO]], [[CORO_SLOT]]
  // CHECK-NEXT: kgen.param.constant: i32 = <1>
  // CHECK-NEXT: kgen.struct.gep
  // CHECK-NEXT: pop.store
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT: co.suspend.end
  // CHECK-NEXT: }
  %coro = co.invoke[(i1) async -> index: @coroutine1](%true)
  co.suspend (%hdl) {
    co.suspend.end
  }
  // CHECK-NEXT: [[CORO2_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME8]]]
  // CHECK-NEXT:  [[CORO2:%.*]] = pop.load [[CORO2_SLOT]]
  // CHECK-NEXT: pop.aligned_free [[CORO2]]
  co.destroy %coro
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

kgen.func @triggerCold(%arg0:i1) {
  %coro = co.invoke[(i1) async -> index:@coroutine](%arg0)
  kgen.return
}

}

// -----

// COM: Ensure Non-Result Args Are Mapped to Resume.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_ramp(%arg0: index) -> !kgen.pointer<struct<(i32, pointer<none>, (!kgen.pointer<none>) -> (), pointer<none>, pointer<none>, pointer<none>)>>
kgen.func @coroutine(%arg0: index, %__result__: !kgen.pointer<index> byref_result) async -> index {
  %idx1 = index.constant 1
  %result1 = index.add %arg0, %idx1
  co.suspend (%hdl) {
    co.suspend.end
  }
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

kgen.func @triggerCold(%arg0:index) {
  %coro = co.invoke[(index, !kgen.pointer<index> byref_result) async -> index:@coroutine](%arg0)
  kgen.return
}

}

// -----

// COM: Default Behavior For Stack Allocations Without Markers.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @missing_markers_resume
kgen.func @missing_markers(%arg1: index, %arg2: i1) async -> index {
  // CHECK-NEXT: [[STACK_ALLOC:%.*]] = kgen.struct.gep %arg0[[[#FRAME7:]]]
  // CHECK-NEXT: [[V2:%.*]] = kgen.struct.gep [[STACK_ALLOC]][1] : <struct<(index, index)>>
  %0 = pop.stack_allocation 1 x !kgen.struct<(index, index)>
  %1 = kgen.struct.gep %0[1] : <struct<(index, index)>>
  // CHECK: hlcf.if
  %arg2_sb = pop.cast_from_builtin %arg2 : i1 to !kgen.scalar<bool>
  hlcf.if %arg2_sb {
    // CHECK-NEXT: [[V12:%.*]] = pop.stack_allocation 1 x index
    // CHECK-NEXT: [[V13:%.*]] = kgen.call @use2([[V12]]) : (!kgen.pointer<index>) -> index
    %3 = pop.stack_allocation 1 x index
    %4 = kgen.call @use2(%3) : (!kgen.pointer<index>) -> index
    hlcf.yield
  } else {
    hlcf.yield
  }
  // CHECK:      [[INFRAME:%.*]] = kgen.struct.gep %arg0[[[#FRAME7 + 2]]]
  // CHECK-NEXT: [[V6:%.*]] = pop.load [[INFRAME]] : !kgen.pointer<index>
  // CHECK-NEXT: pop.store [[V6]], [[V2]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.param.constant: i32 = <1>
  // CHECK-NEXT: kgen.struct.gep
  // CHECK-NEXT: pop.store
  // CHECK-NEXT: co.suspend {
  pop.store %arg1, %1 : !kgen.pointer<index>
  co.suspend (%hdl) {
    co.suspend.end
  }
  %2 = kgen.call @use(%0) : (!kgen.pointer<struct<(index, index)>>) -> index
  kgen.return %2 : index
}

kgen.func @triggerCold(%arg0:index, %arg1:i1) {
  %coro = co.invoke[(index, i1) async -> index:@missing_markers](%arg0, %arg1)
  kgen.return
}

}

// -----

// COM: Ensure the Result of Coro Invoke Is Compatible With Other Coro Ops.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @coroutine(%arg0: index) async -> index {
  co.suspend (%hdl) {
    co.suspend.end
  }
  kgen.return %arg0 : index
}

// CHECK-LABEL: kgen.func @call_coroutine
kgen.func @call_coroutine(%arg0: index) -> index {
  // CHECK:      [[CORO:%.*]] = kgen.call @coroutine_ramp
  // CHECK-NEXT: [[RESUMESLOT:%.*]] = kgen.struct.gep [[CORO]][[[#FRAME1:]]]
  // CHECK-NEXT: [[TYPED_RESUMESLOT:%.*]] = pop.pointer.bitcast [[RESUMESLOT]] : !kgen.pointer<pointer<none>> to !kgen.pointer<(!kgen.pointer<struct<(i32, pointer<none>, (!kgen.pointer<none>) -> (), pointer<none>, pointer<none>, pointer<none>)>>) -> ()>
  // CHECK-NEXT: [[RESUME:%.*]] = pop.load [[TYPED_RESUMESLOT]]
  // CHECK-NEXT: kgen.call_indirect [[RESUME]]([[CORO]])
  %coro = co.invoke[(index) async -> index: @coroutine](%arg0)
  %fn = co.resume %coro : <(!co.routine) -> ()>
  kgen.call_indirect %fn(%coro) : (!co.routine) -> ()
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @get_results
kgen.func @get_results(%arg0: !co.routine) {
  // CHECK-NEXT: [[CONT:%.*]] = pop.pointer.bitcast %arg0
  // CHECK-NEXT: [[PROMISE_PTR:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME6:]]]
  // CHECK-NEXT: [[VALUE_PTR:%.*]] = kgen.struct.gep [[PROMISE_PTR]][0]
  // CHECK-NEXT: pop.load [[VALUE_PTR]]
  %0 = co.get_results %arg0 : index
  kgen.return
}

// CHECK-LABEL: kgen.func @multiple_results_resume
kgen.func @multiple_results(%arg0: !co.routine) async -> (i32, i64) {
  // CHECK:      [[PROMISE_PTR:%.*]] = kgen.struct.gep {{.*}}[6]
  // CHECK-NEXT: [[R0_PTR:%.*]] = kgen.struct.gep [[PROMISE_PTR]][0]
  // CHECK-NEXT: [[R0:%.*]] = pop.load [[R0_PTR]]
  // CHECK-NEXT: [[R1_PTR:%.*]] = kgen.struct.gep [[PROMISE_PTR]][1]
  // CHECK-NEXT: [[R1:%.*]] = pop.load [[R1_PTR]]
  %0, %1 = co.get_results %arg0 : i32, i64
  // CHECK-NEXT: [[PROMISE_PTR:%.*]] = kgen.struct.gep {{.*}}[6]
  // CHECK-NEXT: [[R0_PTR:%.*]] = kgen.struct.gep [[PROMISE_PTR]][0]
  // CHECK-NEXT: store [[R0]], [[R0_PTR]]
  // CHECK-NEXT: [[R1_PTR:%.*]] = kgen.struct.gep [[PROMISE_PTR]][1]
  // CHECK-NEXT: store [[R1]], [[R1_PTR]]
  kgen.return %0, %1 : i32, i64
}

// CHECK-LABEL: kgen.func @no_results_ramp
// CHECK: aligned_alloc %idx8, %idx48
kgen.func @no_results() async {
  kgen.return
}

// CHECK-LABEL: kgen.func @use_of_suspend_resume
kgen.func @use_of_suspend() async -> i32 {
  // CHECK: co.suspend {
  co.suspend (%hdl) {
    // CHECK: [[HDL:%.*]] = pop.pointer.bitcast %arg0 : {{.*}} to !kgen.pointer<struct<(i32, pointer<none>, (!kgen.pointer<none>) -> (), pointer<none>, pointer<none>, pointer<none>)>>
    %0 = pop.stack_allocation 1 x !co.routine
    // CHECK: store [[HDL]], %{{.*}}
    pop.store %hdl, %0 : !kgen.pointer<!co.routine>
    co.suspend.end
  }
  kgen.unreachable
}

kgen.func @triggerCold(%arg0:index) {
  %coro1 = co.invoke[() async -> i32:@use_of_suspend]()
  %coro3 = co.invoke[(!co.routine) async -> (i32, i64):@multiple_results](%coro1)
  %coro4 = co.invoke[() async -> ():@no_results]()
  kgen.return
}
}

// -----

// COM: Verify Dry Runs Terminate

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: kgen.func @f_resume
  kgen.func @f(%arg0: index) async {
    // CHECK: hlcf.loop
    hlcf.loop (%arg1 = %arg0 : index) {
      // CHECK-NEXT:      [[V2:%.*]] = kgen.struct.gep %arg0[[[#FRAME7:]]]
      // CHECK-NEXT: pop.store %arg1, [[V2]]
      %0 = index.cmp slt(%arg1, %arg0)
      hlcf.loop (%arg2 = %arg1 : index) {
        %1 = index.cmp slt(%arg2, %arg0)
        %cb1 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
        hlcf.if %cb1 {
          hlcf.yield
        } else {
          hlcf.break
        }
        %2 = index.add %arg2, %arg0
        hlcf.continue %2 : index
      }
      hlcf.loop (%arg2 = %arg1 : index) {
        %1 = index.cmp slt(%arg2, %arg0)
        %cb1 = pop.cast_from_builtin %1 : i1 to !kgen.scalar<bool>
        hlcf.if %cb1 {
          hlcf.yield
        } else {
          co.suspend(%hdl) {
            co.suspend.end
          }
          hlcf.break
        }
        %2 = index.add %arg2, %arg0
        hlcf.continue %2 : index
      }
      // CHECK: [[V6:%.*]] = kgen.struct.gep %arg0[[[#FRAME7]]]
      // CHECK-NEXT: [[V7:%.*]] = pop.load [[V6]] : !kgen.pointer<index>
      // CHECK-NEXT: hlcf.continue [[V7]] : index
      hlcf.continue %arg1 : index
    }
    kgen.return
  }

  kgen.func @triggerCold(%arg0: index) {
   %coro = co.invoke[(index) async -> (): @f](%arg0)
   kgen.return
  }
}

// -----

// COM: Verify that Nested Loops Terminate

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_nested_resume
kgen.func @coroutine_nested(%arg0: i1, %arg1: index, %arg3: index, %arg4: i1) async -> index {
  %idx3 = index.constant 3
  %result = kgen.call @foo(%idx3, %arg1) : (index,index) -> index
  hlcf.loop "_loop_0" {
     hlcf.loop "_loop_1" {
       // CHECK: hlcf.loop "_loop_1" {
       // CHECK-NEXT: [[V6:%.*]] = kgen.struct.gep %arg0[[[#FRAME7:]]]
       // CHECK-NEXT: [[V7:%.*]] = pop.load [[V6]] : !kgen.pointer<index>
       // CHECK-NEXT: [[V8:%.*]] = kgen.struct.gep %arg0[[[#FRAME7+2]]]
       // CHECK-NEXT: [[V9:%.*]] = pop.load [[V8]] : !kgen.pointer<index>
       // CHECK-NEXT: [[V10:%.*]] = kgen.call @bar([[V7]], [[V9]]) : (index, index) -> index
       %isThisDetected = kgen.call @bar(%result, %arg3): (index,index) -> index
       %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
       hlcf.if %arg0_sb {
         hlcf.yield
       } else {
         hlcf.break "_loop_1"
       }
       %arg4_sb = pop.cast_from_builtin %arg4 : i1 to !kgen.scalar<bool>
       hlcf.if %arg4_sb {
         co.suspend (%hdl) {
          co.suspend.end
         }
         hlcf.continue
       } else {
         hlcf.yield
       }
       hlcf.continue
     }
     hlcf.continue
  }

  kgen.return %arg1 : index
}

kgen.func @triggerCold(%arg0: i1, %arg1: index, %arg3: index, %arg4: i1) {
 %coro = co.invoke[(i1, index, index, i1) async -> index:@coroutine_nested](%arg0, %arg1, %arg3, %arg4)
 kgen.return
}
}

// -----

// COM: Co.Suspend Has Correct Predecessors Set

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  kgen.func @foo(%arg0: index imm, %arg1: !kgen.pointer<none> byref_result) async no_inline {
    co.suspend (%hdl) {
      co.suspend.end
    }
    %idx1 = index.constant 1
    %0 = kgen.call @foo(%idx1) : (index) -> index
    // CHECK: kgen.call @foo
    // CHECK-NEXT: [[V0:%.*]] = kgen.struct.gep %arg0[[[#FRAME7:]]]
    co.suspend (%hdl) {
      %12 = index.add %0, %0
      co.suspend.end
    }
    // CHECK: [[V1:%.*]] = kgen.struct.gep %arg0[[[#FRAME7]]]
    %11 = index.add %0, %0
    kgen.return
  }

  kgen.func @triggerCold(%arg0: index) {
   %coro = co.invoke[(index imm, !kgen.pointer<none> byref_result) async -> ():@foo](%arg0)
   kgen.return
  }
}

// -----

// COM: Lower ResumeOp

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @coroutine1(%arg0: i1) async -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.func @coroutine_resume
kgen.func @coroutine(%arg0: i1) async -> index {
  %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
  hlcf.if %arg0_sb {
    %idx1 = index.constant 1
    kgen.return %idx1 : index
  } else {
    hlcf.yield
  }
  %true = index.bool.constant true
  %coro = co.invoke[(i1) async -> index: @coroutine1](%true)
  // CHECK:      [[CORO:%.*]] = kgen.call @coroutine1_ramp(%true)
  // CHECK-NEXT: kgen.param.constant: i32 = <1>
  // CHECK-NEXT: kgen.struct.gep
  // CHECK-NEXT: pop.store
  // CHECK-NEXT: co.suspend {
  // CHECK-NEXT:   [[RESUME_SLOT:%.*]] = kgen.struct.gep [[CORO]][[[#FRAME1:]]]
  // CHECK-NEXT:   [[TYPED_RESUME:%.*]] = pop.pointer.bitcast [[RESUME_SLOT]] : !kgen.pointer<pointer<none>> to !kgen.pointer<(!kgen.pointer<struct<(i32, pointer<none>, (!kgen.pointer<none>) -> (), pointer<none>, pointer<none>, pointer<none>)>>) -> ()>
  // CHECK-NEXT:   [[RESUME:%.*]] = pop.load [[TYPED_RESUME]]
  // CHECK-NEXT:   kgen.call_indirect [[RESUME]]([[CORO]])
  // CHECK-NEXT:   co.suspend.end
  // CHECK-NEXT: }
  co.suspend (%hdl) {
    %fn = co.resume %coro : <(!co.routine) -> ()>
    kgen.call_indirect %fn(%coro) : (!co.routine) -> ()
    co.suspend.end
  }
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

kgen.func @triggerCold(%arg0: i1) {
 %coro = co.invoke[(i1) async -> index:@coroutine](%arg0)
 kgen.return
}

}

// -----

// COM: Verify DryRun Nodes With Multiple Predecessors

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_nested_resume
kgen.func @coroutine_nested(%arg0: i1, %arg1: index, %arg3: index, %arg4: i1) async -> index {
  %idx3 = index.constant 3
  %result = kgen.call @foo(%idx3, %arg1) : (index,index) -> index
  // CHECK: hlcf.loop
  hlcf.loop "_loop_1" {
    // CHECK-NEXT: [[V6:%.*]] = kgen.struct.gep %arg0[[[#FRAME7:]]]
    // CHECK-NEXT: [[V7:%.*]] = pop.load [[V6]]
    // CHECK-NEXT: [[V8:%.*]] = kgen.struct.gep %arg0[[[#FRAME7 + 2]]]
    // CHECK-NEXT: [[V9:%.*]] = pop.load [[V8]]
    // CHECK-NEXT: kgen.call @bar([[V7]], [[V9]])
    %isThisDetected = kgen.call @bar(%result, %arg3): (index,index) -> index
    %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
    hlcf.if %arg0_sb {
      hlcf.continue
    } else {
      hlcf.yield
    }
    co.suspend (%hdl) {
      co.suspend.end
    }
    hlcf.continue
  }
  kgen.return %arg1 : index
}

kgen.func @triggerCold(%arg0: i1, %arg1: index, %arg3: index, %arg4: i1) {
 %coro = co.invoke[(i1, index, index, i1) async -> index:@coroutine_nested](%arg0, %arg1, %arg3, %arg4)
 kgen.return
}
}

// -----

// COM: All Successors Must Be Updated After Dry Run

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
    // CHECK-LABEL: kgen.func @foo_resume
    kgen.func @foo(%arg0: !kgen.pointer<pointer<none>>, %arg1: i1, %arg2: index) async no_inline {
      %idx0 = index.constant 0
      %idx1 = index.constant 1
      hlcf.loop "_loop_0" {
        %0 = pop.stack_allocation 1 x struct<(pointer<none>, pointer<none>) memoryOnly> marked
        %1 = pop.load %arg0 : !kgen.pointer<pointer<none>>
        kgen.call @"CBatch::__init__"(%1, %0) : (!kgen.pointer<none>, !kgen.pointer<struct<(pointer<none>, pointer<none>) memoryOnly>> byref_result) -> ()
        co.suspend (%hdl) {
          co.suspend.end
        }
        hlcf.loop "_loop_1" (%arg3 = %arg2 : index) {
          %3 = index.cmp sgt(%arg3, %idx0)
          %cb3 = pop.cast_from_builtin %3 : i1 to !kgen.scalar<bool>
          hlcf.if %cb3 {
            hlcf.yield
          } else {
            hlcf.break "_loop_1"
          }
          %4 = index.sub %arg2, %idx1
          hlcf.continue %4 : index
        }
        // CHECK:        hlcf.continue %
        // CHECK-NEXT: }
        // CHECK:      [[V8:%.*]] = kgen.struct.gep %arg0[[[#FRAME9:]]]
        // CHECK-NEXT: [[V9:%.*]] = kgen.call @batch_size([[V8]])
        %2 = kgen.call @batch_size(%0) : (!kgen.pointer<struct<(pointer<none>, pointer<none>) memoryOnly>> imm_mem) -> index
        hlcf.continue
      }
      kgen.return
    }

  kgen.func @triggerCold(%arg0: !kgen.pointer<pointer<none>>, %arg1: i1, %arg2: index) {
     %coro = co.invoke[(!kgen.pointer<pointer<none>>, i1, index) async -> ():@foo](%arg0, %arg1, %arg2)
     kgen.return
  }
}

// -----

// COM: Nested Loops With Parent Suspend

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
    // CHECK-LABEL:  kgen.func @foo_resume
    kgen.func @foo(%arg0: index, %arg1: index, %arg2: index, %arg3: i1, %arg4: i1) async {
    // CHECK:      [[V2:%.*]] = kgen.call @foo3
    // CHECK-NEXT: [[V3:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
    // CHECK-NEXT: pop.store [[V2]], [[V3]] : !kgen.pointer<index>
      %0 = kgen.call @foo3(%arg2) : (index) -> index
      %1:3 = hlcf.loop "_loop_2" (%arg5 = %arg0 : index, %arg6 = %arg1 : index, %arg7 = %arg1 : index, %arg8 = %arg1 : index) -> (index, index, index) {
        %2 = hlcf.loop "_loop_0" (%arg9 = %arg2 : index, %arg10 = %arg2 : index) -> index {
          %3 = kgen.call @bar1(%arg9) : (index) -> i1
          // CHECK: hlcf.loop "_loop_1"
          // CHECK-NEXT: [[V25:%.*]] = kgen.struct.gep %arg0[[[#FRAME8]]]
          // CHECK-NEXT: [[V26:%.*]] = pop.load [[V25]] : !kgen.pointer<index>
          // CHECK-NEXT: [[V27:%.*]] kgen.call @bar2([[V26]]) : (index) -> i1
          %4 = hlcf.loop "_loop_1" (%arg11 = %arg1 : index, %arg12 = %arg1 : index) -> index {
            %5 = kgen.call @bar2(%0) : (index) -> i1
            %cb5 = pop.cast_from_builtin %5 : i1 to !kgen.scalar<bool>
            hlcf.if %cb5 {
              hlcf.continue %arg1, %arg1 : index, index
            } else {
              hlcf.break "_loop_1" %arg11 : index
            }
            kgen.unreachable
          }
          %arg3_sb1 = pop.cast_from_builtin %arg3 : i1 to !kgen.scalar<bool>
          hlcf.if %arg3_sb1 {
            hlcf.break "_loop_0" %arg9 : index
          } else {
            hlcf.yield
          }
          hlcf.continue %arg1, %arg1 : index, index
        }
        co.suspend (%hdl) {
          co.suspend.end
        }
        %arg3_sb2 = pop.cast_from_builtin %arg3 : i1 to !kgen.scalar<bool>
        hlcf.if %arg3_sb2 {
          hlcf.break "_loop_2" %arg1, %arg6, %arg7 : index, index, index
        } else {
          hlcf.continue %arg1, %arg6, %arg7, %arg8 : index, index, index, index
        }
        kgen.unreachable
      }
      kgen.return
    }

  kgen.func @triggerCold(%arg0: index, %arg1: index, %arg2: index, %arg3: i1, %arg4: i1) {
     %coro = co.invoke[(index, index, index, i1, i1) async -> ():@foo](%arg0, %arg1, %arg2, %arg3, %arg4)
     kgen.return
  }
}

// -----

// COM: Verify that Constants Are Not Stored In Frame

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: kgen.func @foo_resume
  kgen.func @foo(%arg0: i1 imm, %arg1: !kgen.pointer<none> byref_result) async no_inline {
    %idx1 = index.constant 1
    co.suspend (%hdl) {
      co.suspend.end
    }
    // CHECK: %idx1_0 = index.constant 1
    // CHECK-NEXT: index.mul %idx1_0, %idx1_0
    %14 = index.mul %idx1, %idx1
    kgen.return
  }
  // CHECK-LABEL: kgen.func @needsLift_resume
  kgen.func @needsLift(%arg0: i1 imm, %arg1: !kgen.pointer<none> byref_result) async no_inline {
    %idx1 = index.constant 1
    co.suspend (%hdl) {
      co.suspend.end
    }
    // CHECK:      co.suspend {
    // CHECK-NEXT:   co.suspend.end
    // CHECK-NEXT: }
    // CHECK-NEXT: kgen.struct.gep %arg0
    // CHECK-NEXT: pop.load
    // CHECK-NEXT: pop.cast_from_builtin
    // CHECK-NEXT: %idx1_0 = index.constant 1
    // CHECK-NEXT: hlcf.if
    // CHECK-NEXT: index.sub %idx1_0, %idx1_0
    %arg0_sb1 = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
    hlcf.if %arg0_sb1 {
      %13 = index.sub %idx1, %idx1
      hlcf.yield
    } else {
      hlcf.yield
    }
    // CHECK: index.add %idx1_0, %idx1_0
    %arg0_sb2 = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
    hlcf.if %arg0_sb2 {
      %14 = index.add %idx1, %idx1
      hlcf.yield
    } else {
      hlcf.yield
    }
    kgen.return
  }

  kgen.func @triggerCold(%arg0: i1) {
     %coro = co.invoke[(i1 imm, !kgen.pointer<none> byref_result) async -> ():@foo](%arg0)
     %coro2 = co.invoke[(i1 imm, !kgen.pointer<none> byref_result) async -> ():@needsLift](%arg0)
     kgen.return
  }
}


// -----

// COM: Frame Addresses Are Not Stored In Frame

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  kgen.func @gep(%arg0: i1 imm, %arg1: !kgen.pointer<none> byref_result) async no_inline {
    %0 = pop.stack_allocation 1 x !kgen.struct<(index, index)> marked
    pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<struct<(index, index)>>
    %1 = kgen.call @fillMe(%0) : (!kgen.pointer<struct<(index, index)>> byref_result) -> index
    %2 = kgen.struct.gep %0[1] : <struct<(index, index)>>
    co.suspend (%hdl) {
      co.suspend.end
    }
    // CHECK: co.suspend.end
    // CHECK-NEXT: }
    // CHECK-NEXT: [[V3:%.*]] = kgen.struct.gep %arg0[7]
    // CHECK-SAME: <struct<(i32, pointer<none>, (!kgen.pointer<none>) -> (), pointer<none>, pointer<none>, pointer<none>, struct<()>, struct<(index, index)>)>>
    // CHECK-NEXT: [[V4:%.*]] = kgen.struct.gep [[V3]][1] : <struct<(index, index)>>
    // CHECK-NEXT: [[V5:%.*]] = pop.load [[V4]] : !kgen.pointer<index>
    // CHECK-NEXT:  kgen.call @doSomething([[V5]]) : (index) -> index
    %4 = pop.load %2 : !kgen.pointer<index>
    %3 = kgen.call @doSomething(%4) : (index) -> index
    pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<struct<(index, index)>>
    kgen.return
  }
  kgen.func @offset(%arg0: i1 imm, %arg1: !kgen.pointer<none> byref_result) async no_inline {
    // CHECK:      [[V1:%.*]] = index.constant 1
    // CHECK-NEXT: [[V3:%.*]] = kgen.struct.gep %arg0[[[#FRAME8:]]]
    // CHECK-NEXT: [[V4:%.*]] = pop.pointer.bitcast [[V3]]
    // CHECK-NEXT: [[V5:%.*]] = pop.offset [[V4]][[[V1]]] : !kgen.pointer<index>
    %0 = pop.stack_allocation 2 x index marked
    pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
    %idx1 = index.constant 1
    %1 = pop.offset %0[%idx1] : !kgen.pointer<index>
    hlcf.loop {
      %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
      hlcf.if %arg0_sb {
        hlcf.yield
      } else {
        hlcf.break
      }
      co.suspend (%hdl) {
        co.suspend.end
      }
      // CHECK:      [[V6:%.*]] = index.constant 1
      // CHECK-NEXT: [[V7:%.*]] = kgen.struct.gep %arg0[[[#FRAME8]]]
      // CHECK-NEXT: [[V8:%.*]] = pop.pointer.bitcast [[V7]]
      // CHECK-NEXT: [[V9:%.*]] = pop.offset [[V8]][[[V6]]] : !kgen.pointer<index>
      // CHECK-NEXT: [[V10:%.*]] = pop.load [[V9]]
      // CHECK-NEXT: [[V11:%.*]] = kgen.call @doSomething([[V10]])
      %4 = pop.load %1 : !kgen.pointer<index>
      %3 = kgen.call @doSomething(%4) : (index) -> index
      hlcf.continue
    }
    // CHECK:      [[V12:%.*]] = index.constant 1
    // CHECK-NEXT: [[V13:%.*]] = kgen.struct.gep %arg0[[[#FRAME8]]]
    // CHECK-NEXT: [[V14:%.*]] = pop.pointer.bitcast [[V13]]
    // CHECK-NEXT: [[V15:%.*]] = pop.offset [[V14]][[[V12]]] : !kgen.pointer<index>
    // CHECK-NEXT: [[V16:%.*]] = pop.load [[V15]]
    // CHECK-NEXT: [[V17:%.*]] = kgen.call @doSomething([[V16]])
    %5 = pop.load %1 : !kgen.pointer<index>
    %6 = kgen.call @doSomething(%5) : (index) -> index
    pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
    kgen.return
  }

  kgen.func @triggerCold(%arg0: i1) {
     %coro = co.invoke[(i1 imm, !kgen.pointer<none> byref_result) async -> (): @offset](%arg0)
     %coro2 = co.invoke[(i1 imm, !kgen.pointer<none> byref_result) async -> (): @gep](%arg0)
     kgen.return
  }

}

// -----

// COM: Simple Hot Ramp Function Generation, Hot only.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @conditional_suspoint_hot_ramp
// CHECK-SAME: (%arg0: !kgen.generator<(!kgen.pointer<none>) -> ()>, %arg1: !kgen.pointer<none>, %arg2: i1, %arg3: index, %arg4: index, %arg5: !kgen.pointer<index>)
// CHECK-SAME:  -> !kgen.pointer<struct<(i32, pointer<none>, (!kgen.pointer<none>) -> (), pointer<none>, pointer<none>, pointer<none>)>> {

// FRAME ALLOCATION: Frame space is the same as cold.
// CHECK-NEXT:  %idx72 = index.constant 72
// CHECK-NEXT:  %idx8 = index.constant 8
// CHECK-NEXT:  [[CONT:%.*]] = pop.aligned_alloc %idx8, %idx72 : <struct<(i32, pointer<none>, (!kgen.pointer<none>) -> (), pointer<none>, pointer<none>, pointer<none>, struct<(index)>, index, index)>>

// STATE INIT
// CHECK-NEXT:  [[V1:%.*]] = kgen.param.constant: i32 = <0>
// CHECK-NEXT:  [[V2:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME0:]]]
// CHECK-NEXT:  pop.store [[V1]], [[V2]] : !kgen.pointer<i32>

// RESUME FNC INIT
// CHECK-NEXT:  [[V3:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME1:]]]
// CHECK-NEXT:  [[V4:%.*]] = kgen.create_closure{{.*}}@conditional_suspoint_resume]
// CHECK-NEXT:  [[V5:%.*]] = pop.pointer.bitcast [[V4]]
// CHECK-NEXT:  pop.store [[V5]], [[V3]] : !kgen.pointer<pointer<none>>

// STORE CALLBACK/CLOSURE STATE
// CHECK-NEXT:  [[W6:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME1 + 2]]]
// CHECK-NEXT:  pop.store %arg1, [[W6]] : !kgen.pointer<pointer<none>>
// CHECK-NEXT:  [[W7:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME1 + 1]]]
// CHECK-NEXT:  pop.store %arg0, [[W7]] : !kgen.pointer<(!kgen.pointer<none>) -> ()>

// STORE ARG1
// CHECK-NEXT:  [[V6:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME7:]]]
// CHECK-NEXT:  pop.store %arg3, [[V6:%.*]] : !kgen.pointer<index>

// STORE BY REF RESULT
// CHECK-NEXT:  [[V7:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME5:]]]
// CHECK-NEXT:  [[V8:%.*]] = pop.pointer.bitcast [[V7]]
// CHECK-NEXT:  pop.store %arg5, [[V8]] : !kgen.pointer<pointer<index>>

// Bitcast Cont because ramp returns header type
// CHECK-NEXT: [[V13:%.*]] = pop.pointer.bitcast [[CONT]]

// FIRST STATE
// CHECK-NEXT: [[COND:%.*]] = pop.cast_from_builtin %arg2 : i1 to !kgen.scalar<bool>
// CHECK-NEXT: hlcf.if [[COND]] {
// CHECK-NEXT:   pop.store %arg3, %arg5 : !kgen.pointer<index>
// CHECK-NEXT:   hlcf.yield
// CHECK-NEXT: } else {

// T1 is Used Across suspension points; put it on frame
// CHECK-NEXT:   [[V14:%.*]] = index.add %arg4, %arg4
// CHECK-NEXT:   [[V15:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME9:]]]
// CHECK-NEXT:   pop.store [[V14]], [[V15]] : !kgen.pointer<index>

// Check that the state is updated.
// CHECK-NEXT:   [[V18:%.*]] = kgen.param.constant: i32 = <1>
// CHECK-NEXT:   [[V19:%.*]] = kgen.struct.gep [[CONT]][0]
// CHECK-NEXT:   pop.store [[V18]], [[V19]] : !kgen.pointer<i32>

// CHECK-NEXT:   kgen.return [[V13]]
// CHECK-NEXT: }
// CHECK-NEXT: [[V9:%.*]] = pop.load %arg5 : !kgen.pointer<index>
// CHECK-NEXT: [[V10:%.*]] = index.add [[V9]], %arg3

// Check that callback is invoked.
// This is needed because in this path a suspension point is not hit.
// CHECK-NEXT: kgen.call_indirect musttail %arg0(%arg1) : (!kgen.pointer<none>) -> ()

// Direct Result value is stored in frame
// CHECK-NEXT: [[V11:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME6:]]]
// CHECK-NEXT: [[V12:%.*]] = kgen.struct.gep [[V11]][0] : <struct<(index)>>
// CHECK-NEXT: pop.store [[V10]], [[V12]] : !kgen.pointer<index>

// Return Coroutine
// CHECK-NEXT: kgen.return [[V13]]

// Verify that the hot resume has been stripped of the first state.
// CHECK-LABEL:  kgen.func @conditional_suspoint_resume
// CHECK-NEXT:   [[ARBITRARY_VALUE:%.*]] = kgen.param.constant: scalar<bool> = <#interp.uninitmem>
// CHECK-NEXT:   hlcf.if [[ARBITRARY_VALUE]] {
// CHECK-NEXT:   hlcf.yield
// CHECK-NEXT:   } else {
// CHECK-NEXT:   co.suspend {
// CHECK-NEXT:   co.suspend.end
// CHECK-NEXT:   }
// CHECK-NEXT:   [[V10:%.*]] = kgen.struct.gep %arg0[[[#FRAME9]]]
kgen.func @conditional_suspoint(%arg0: i1,
                     %arg1: index,
                     %arg2: index,
                     %__result__: !kgen.pointer<index> byref_result) async -> index {
 %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
 hlcf.if %arg0_sb {
   pop.store %arg1, %__result__ : !kgen.pointer<index>
   hlcf.yield
 } else {
    %t1 = index.add %arg2, %arg2
    co.suspend (%hdl) {
      co.suspend.end
    }
    %t2 = index.add %t1, %arg1
    pop.store %t2, %__result__ : !kgen.pointer<index>
    hlcf.yield
 }
 %result = pop.load %__result__ : !kgen.pointer<index>
 %final = index.add %result, %arg1
 kgen.return %final : index
}

// CHECK-LABEL: kgen.func @trigger_creation_resume
kgen.func @trigger_creation(%arg0: i1, %arg1: index, %arg2: index, %__result__: !kgen.pointer<index> byref_result) async {
   // CHECK: co.suspend {
   // CHECK: [[V9:%.*]] = kgen.struct.gep %arg0[1]
   // CHECK-NEXT: [[V10:%.*]] = pop.load [[V9]] : !kgen.pointer<pointer<none>>
   // CHECK-NEXT: [[V11:%.*]] = pop.pointer.bitcast [[V10]] : !kgen.pointer<none> to !kgen.generator<(!kgen.pointer<none>) -> ()>
   // CHECK-NEXT: [[V12:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<struct<(i32, pointer<none>, (!kgen.pointer<none>) -> (), pointer<none>, pointer<none>, pointer<none>, struct<()>, pointer<struct<{{.*}}>>, i1, index, index)>> to !kgen.pointer<none>
   // CHECK-NEXT: kgen.call @conditional_suspoint_hot_ramp([[V11]], [[V12]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) :
   // CHECK-SAME: (!kgen.generator<(!kgen.pointer<none>) -> ()>, !kgen.pointer<none>, i1, index, index, !kgen.pointer<index>) -> !kgen.pointer<struct<(i32, pointer<none>, (!kgen.pointer<none>) -> (), pointer<none>, pointer<none>, pointer<none>)>>
   // CHECK-NEXT: kgen.struct.gep
   // CHECK-NEXT: pop.store
   // CHECK-NEXT: co.suspend.end
   // CHECK-NEXT: }
   %coro = co.hot_invoke[(i1, index, index, !kgen.pointer<index> byref_result) async -> index: @conditional_suspoint](%arg0, %arg1, %arg2, %__result__)
   kgen.return
}

kgen.func @trigger_trigger_creation(%arg0: i1, %arg1: index, %arg2: index, %__result__: !kgen.pointer<index> byref_result) {
  %coro = co.invoke[(i1, index, index, !kgen.pointer<index> byref_result) async -> ():@trigger_creation](%arg0, %arg1, %arg2)
  kgen.return
}


// CHECK-LABEL: kgen.func @conditional_suspoint_elif_resume
kgen.func @conditional_suspoint_elif(%arg0: i1,
                     %arg1: index,
                     %arg2: index,
                     %__result__: !kgen.pointer<index> byref_result) async -> index {
  // CHECK-NEXT: [[V9:%.*]] = kgen.param.constant: scalar<bool> = <#interp.uninitmem>
  // CHECK-NEXT: hlcf.elif [[V9]] {
  %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
  hlcf.elif %arg0_sb {
    pop.store %arg1, %__result__ : !kgen.pointer<index>
    hlcf.yield
  } else {
    %t1 = index.add %arg2, %arg2
    co.suspend (%hdl) {
      co.suspend.end
    }
    %t2 = index.add %t1, %arg1
    pop.store %t2, %__result__ : !kgen.pointer<index>
    hlcf.yield
 }
 %result = pop.load %__result__ : !kgen.pointer<index>
 %final = index.add %result, %arg1
 kgen.return %final : index
}

kgen.func @trigger_creation_elif(%arg0: i1, %arg1: index, %arg2: index, %__result__: !kgen.pointer<index> byref_result) async {
   %coro = co.hot_invoke[(i1, index, index, !kgen.pointer<index> byref_result) async -> index: @conditional_suspoint_elif](%arg0, %arg1, %arg2, %__result__)
   kgen.return
}

kgen.func @trigger_trigger_creation_elif(%arg0: i1, %arg1: index, %arg2: index, %__result__: !kgen.pointer<index> byref_result) {
  %coro = co.invoke[(i1, index, index, !kgen.pointer<index> byref_result) async -> ():@trigger_creation_elif](%arg0, %arg1, %arg2)
  kgen.return
}

}

// -----

// COM: Verify Block Storage and Unreachable inserts in Hot Ramp

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @coroutine_hot_ramp(
kgen.func @coroutine(%arg0: i1, %arg1: index, %__result__: !kgen.pointer<index> byref_result) async -> index {
 // CHECK:      [[CORO:%.*]] = pop.aligned_alloc
 // CHECK:      [[V11:%.*]] = pop.pointer.bitcast [[CORO]]
 // CHECK-NEXT: hlcf.loop "_loop_0" ([[BLOCK_ARG:%.*]] =
 // Check that block arguments are stored in frame.
 // CHECK-NEXT: [[V9:%.*]] = kgen.struct.gep %0[[[#FRAME8:]]]
 // CHECK-NEXT: pop.store [[BLOCK_ARG]], [[V9]] : !kgen.pointer<index>

 // CHECK-NEXT: hlcf.loop "_loop_1" ([[BLOCK_ARG_INNER:%.*]] =
 // CHECK-NEXT: [[V10:%.*]] = kgen.struct.gep %0[[[#FRAME8 - 1]]]
 // CHECK-NEXT: pop.store [[BLOCK_ARG_INNER]], [[V10]] : !kgen.pointer<index>

 // Verify suspension point in nested loops is properly replaced and
 // unreachable is inserted to terminate unreachable blocks.
 // CHECK-NEXT:   [[COND0:%.*]] = pop.cast_from_builtin %arg2 : i1 to !kgen.scalar<bool>
 // CHECK-NEXT:   hlcf.if [[COND0]] {
 // CHECK-NEXT:     hlcf.break "_loop_1"
 // CHECK-NEXT:   } else {
 // CHECK-NEXT:     hlcf.yield
 // CHECK-NEXT:   }

 // CHECK-NEXT:   kgen.param.constant: i32 = <1>
 // CHECK-NEXT:   kgen.struct.gep [[CORO]][0]
 // CHECK-NEXT:   pop.store

 // CHECK-NEXT:   kgen.return [[V11]]
 // CHECK-NEXT: }
 // CHECK-NEXT: kgen.call @print
 // CHECK-NEXT: hlcf.continue
 // CHECK-NEXT: }
 // CHECK-NEXT: kgen.unreachable
 hlcf.loop "_loop_0" (%arg3 = %arg1 : index) {
   hlcf.loop "_loop_1" (%arg2 = %arg1 : index) {
     %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
     hlcf.if %arg0_sb {
       hlcf.break "_loop_1"
     } else {
       hlcf.yield
     }
     co.suspend (%hdl) {
       co.suspend.end
     }
     kgen.call @print1(%arg2) : (index) -> ()
     hlcf.continue %arg2 : index
   }
   kgen.call @print(%arg0) : (i1) -> ()
   hlcf.continue %arg3 : index
 }
 co.suspend (%hdl) {
   co.suspend.end
 }
 %final = index.add %arg1, %arg1
 kgen.return %final : index
}

// Check that the operands of parents that are state 0 are replaced with constants. All other ops in state 0 will be erased.
// CHECK-LABEL:  kgen.func @coroutine_resume
// CHECK-NEXT:   [[UNDEF:%.*]] = kgen.param.constant = <#interp.uninitmem>
// CHECK-NEXT:   hlcf.loop "_loop_0" (%arg1 = [[UNDEF]] : index) {
kgen.func @trigger_creation(%arg0: i1, %arg1: index, %__result__: !kgen.pointer<index> byref_result) async {
   %coro = co.hot_invoke[(i1, index, !kgen.pointer<index> byref_result) async -> index: @coroutine](%arg0, %arg1, %__result__)
   kgen.return
}

}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
kgen.func @coroutine(%arg0: i1, %arg1: index, %__result__: !kgen.pointer<index> byref_result) async -> index {
  // CHECK: [[V2:%.*]] = kgen.call @doSomething({{.*}}) : (index) -> index
  // CHECK-NOT: hlcf.loop "_loop_1" (%arg2 = [[V2]] : index) {
  %x = kgen.call @doSomething(%arg1) : (index) -> index
  hlcf.loop "_loop_0" (%arg3 = %arg1 : index) {
    hlcf.loop "_loop_1" (%arg2 = %x : index) {
      %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
      hlcf.if %arg0_sb {
        hlcf.break "_loop_1"
      } else {
        hlcf.yield
      }
      co.suspend (%hdl) {
        co.suspend.end
      }
      hlcf.continue %arg2 : index
    }
    hlcf.continue %arg3 : index
  }
  co.suspend (%hdl) {
    co.suspend.end
  }
  %final = index.add %arg1, %arg1
  kgen.return %final : index
}

kgen.func @triggerCold(%arg0: i1, %arg1: index) {
   %coro = co.invoke[(i1, index, !kgen.pointer<index> byref_result) async -> index: @coroutine](%arg0, %arg1)
   kgen.return
}

}

// -----

// COM: Reusable Resume Function. Verify Only First State Uses Bitcasted Cold Coro Value

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @conditional_suspoint_resume(%arg0: !kgen.pointer<struct<(i32, pointer<none>, (!kgen.pointer<none>) -> (), pointer<none>, pointer<none>, pointer<none>, struct<(index)>, index, index)>>)
kgen.func @conditional_suspoint(%arg0: !kgen.scalar<bool>,
                     %arg1: index,
                     %__result__: !kgen.pointer<index> byref_result) async -> index {
 // CHECK-NEXT: [[COLD_CORO:%.*]] = pop.pointer.bitcast %arg0 :
 // CHECK-SAME: !kgen.pointer<struct<({{.*}}struct<(index)>, index, index)>>
 // CHECK-SAME: to !kgen.pointer<struct<({{.*}}struct<(index)>, index, index, scalar<bool>)>>
 // CHECK-NEXT: [[V1:%.*]] = kgen.struct.gep [[COLD_CORO]][[[#FRAME9:]]]
 // CHECK-NEXT: [[V2:%.*]] = pop.load
 // CHECK-NEXT: hlcf.if [[V2]]
 hlcf.if %arg0 {
   pop.store %arg1, %__result__ : !kgen.pointer<index>
   hlcf.yield
 } else {
    %t1 = index.add %arg1, %arg1
    // Anything after the first suspension point uses the argument not the bitcasted coro
    // CHECK:      co.suspend {
    // CHECK-NEXT:   co.suspend.end
    // CHECK-NEXT: }
    // CHECK-NEXT: [[V16:%.*]] = kgen.struct.gep %arg0[[[#FRAME9 - 2]]]
    co.suspend (%hdl) {
      co.suspend.end
    }
    %t2 = index.add %t1, %arg1
    pop.store %t2, %__result__ : !kgen.pointer<index>
    hlcf.yield
 }
 %result = pop.load %__result__ : !kgen.pointer<index>
 %final = index.add %result, %arg1
 kgen.return %final : index
}

// CHECK-LABEL: kgen.func @trigger_creation_resume
kgen.func @trigger_creation(%arg0: !kgen.scalar<bool>, %arg1: index, %__result__: !kgen.pointer<index> byref_result) async {
   %coro = co.hot_invoke[(!kgen.scalar<bool>, index, !kgen.pointer<index> byref_result) async -> index: @conditional_suspoint](%arg0, %arg1, %__result__)
   kgen.return
}

kgen.func @trigger_trigger_creation(%arg0: !kgen.scalar<bool>, %arg1: index) {
  %coro = co.invoke[(!kgen.scalar<bool>, index, !kgen.pointer<index> byref_result) async -> (): @trigger_creation](%arg0, %arg1)
  %coro2 = co.invoke[(!kgen.scalar<bool>, index, !kgen.pointer<index> byref_result) async -> index: @conditional_suspoint](%arg0, %arg1)
  kgen.return
}
}

// -----

// COM: Verify a coroutine that is both cold and hot started have a hot ramp, cold ramp, and shared resume.


module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// HOT RAMP

// CHECK-LABEL: kgen.func @needsBoth_hot_ramp
// CHECK-SAME: (%arg0: !kgen.generator<(!kgen.pointer<none>) -> ()>, %arg1: !kgen.pointer<none>, %arg2: index) -> !kgen.pointer<struct<(i32, pointer<none>, (!kgen.pointer<none>) -> (), pointer<none>, pointer<none>, pointer<none>)>> {

// Instantiate Continuation and initialize state to 1
// CHECK-NEXT: %idx64 = index.constant 64
// CHECK-NEXT: %idx8 = index.constant 8
// CHECK-NEXT: [[CONT1:%.*]] = pop.aligned_alloc %idx8, %idx64
// CHECK-NEXT: [[V1:%.*]] = kgen.param.constant: i32 = <0>
// CHECK-NEXT: [[V2:%.*]] = kgen.struct.gep [[CONT1]][0]
// CHECK-NEXT: pop.store [[V1]], [[V2]] : !kgen.pointer<i32>

// Configure Resume
// CHECK-NEXT: [[V3:%.*]] = kgen.struct.gep [[CONT1]][1]
// CHECK-NEXT: [[V4:%.*]] = kgen.create_closure[({{.*}}) -> (): @needsBoth_resume]()
// CHECK-NEXT: [[V5:%.*]] = pop.pointer.bitcast [[V4]]
// CHECK-NEXT: pop.store [[V5]], [[V3]] : !kgen.pointer<pointer<none>>

// Configure closure
// CHECK-NEXT: [[V6:%.*]] = kgen.struct.gep [[CONT1]][3]
// CHECK-NEXT: pop.store %arg1, [[V6]] : !kgen.pointer<pointer<none>>

// Configure callback
// CHECK-NEXT: [[V7:%.*]] = kgen.struct.gep [[CONT1]][2]
// CHECK-NEXT: pop.store %arg0, [[V7]] : !kgen.pointer<(!kgen.pointer<none>) -> ()>

// CHECK-NEXT: [[HEADER:%.*]] = pop.pointer.bitcast [[CONT1]]

// STATE 0
// CHECK-NEXT: [[V8:%.*]] = index.add %arg2, %arg2
// CHECK-NEXT: [[V9:%.*]] = index.add [[V8]], [[V8]]
// CHECK-NEXT: [[V10:%.*]] = kgen.struct.gep [[CONT1]][7]
// CHECK-NEXT: pop.store [[V9]], [[V10]] : !kgen.pointer<index>

// UPDATE STATE
// CHECK-NEXT: [[V12:%.*]] = kgen.param.constant: i32 = <1>
// CHECK-NEXT: [[V13:%.*]] = kgen.struct.gep [[CONT1]][0]
// CHECK-NEXT: pop.store [[V12]], [[V13]] : !kgen.pointer<i32>

// CHECK-NEXT: kgen.return [[HEADER]]
// CHECK-NEXT: }

// COLD RAMP

// CHECK-LABEL: kgen.func @needsBoth_ramp(%arg0: index) -> !kgen.pointer<struct<(i32, pointer<none>, (!kgen.pointer<none>) -> (), pointer<none>, pointer<none>, pointer<none>)>> {
// allocate continuation and set it to state 0
// CHECK-NEXT:  %idx72 = index.constant 72
// CHECK-NEXT:  %idx8 = index.constant 8
// CHECK-NEXT:  [[CONT:%.*]] = pop.aligned_alloc %idx8, %idx72
// CHECK-NEXT:  [[V1:%.*]] = kgen.param.constant: i32 = <0>
// CHECK-NEXT:  [[V2:%.*]] = kgen.struct.gep [[CONT]][0]
// CHECK-NEXT:  pop.store [[V1]], [[V2]] : !kgen.pointer<i32>

// Configure resume function.
// CHECK-NEXT:  [[V3:%.*]] = kgen.struct.gep [[CONT]][1]
// CHECK-NEXT:  [[V4:%.*]] = kgen.create_closure[({{.*}}) -> (): @needsBoth_resume]()
// CHECK-NEXT:  [[V5:%.*]] = pop.pointer.bitcast [[V4]]
// CHECK-NEXT:  pop.store [[V5]], [[V3]]

// Since arg0 is used in the resume, store in frame!
// CHECK-NEXT:  [[V6:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8:]]] :
// CHECK-NEXT:  pop.store %arg0, [[V6]] : !kgen.pointer<index>

// Cast continuation to head and return.
// CHECK-NEXT:  [[V6:%.*]] = pop.pointer.bitcast [[CONT]]
// CHECK-NEXT:  kgen.return [[V6]]
// CHECK-NEXT:  }

// SHARED RESUME.

// CHECK-LABEL: kgen.func @needsBoth_resume
// Bit cast the continuation to the larger size
// CHECK-NEXT:  [[CONT:%.*]] = pop.pointer.bitcast %arg0
// CHECK-SAME:  : !kgen.pointer<struct<(i32, pointer<none>, (!kgen.pointer<none>) -> (), pointer<none>, pointer<none>, pointer<none>, struct<(index)>, index)>> to
// CHECK-SAME:    !kgen.pointer<struct<(i32, pointer<none>, (!kgen.pointer<none>) -> (), pointer<none>, pointer<none>, pointer<none>, struct<(index)>, index, index)>>
// CHECK-NEXT:  [[V1:%.*]] = kgen.struct.gep [[CONT]][[[#FRAME8]]] : <struct<(i32, pointer<none>, (!kgen.pointer<none>) -> (), pointer<none>, pointer<none>, pointer<none>, struct<(index)>, index, index)>>
// CHECK-NEXT:  [[V2:%.*]] = pop.load [[V1]] : !kgen.pointer<index>
// CHECK-NEXT:  [[V3:%.*]] = index.add [[V2]], [[V2]]
// CHECK-NEXT:  [[V4:%.*]] = index.add [[V3]], [[V3]]

// Note this reference to the continuation is not replaced because it
// accesses a member that is in both cold and hot frames
// CHECK-NEXT:  [[V5:%.*]] = kgen.struct.gep %arg0[7]
// CHECK-NEXT:  pop.store [[V4]], [[V5]] : !kgen.pointer<index>
// CHECK-NEXT:  kgen.param.constant: i32 = <1>
// CHECK-NEXT:  kgen.struct.gep
// CHECK-NEXT:  pop.store
// CHECK-NEXT:  co.suspend {
// CHECK-NEXT:  co.suspend.end
// CHECK-NEXT:  }

// The rest of the resume uses arg0, not the bitcasted version
// CHECK-NEXT:  [[V6:%.*]] = kgen.struct.gep %arg0[7]
// CHECK-NEXT:  [[V7:%.*]] = pop.load [[V6]] : !kgen.pointer<index>
// CHECK-NEXT:  [[V8:%.*]] = kgen.struct.gep %arg0[6]
// CHECK-NEXT:  [[V9:%.*]] = kgen.struct.gep [[V8]][0] : <struct<(index)>>
// CHECK-NEXT:  pop.store [[V7]], [[V9]] : !kgen.pointer<index>
// CHECK-NEXT:  kgen.return
// CHECK-NEXT:  }
kgen.func @needsBoth(%arg0: index) async -> index {
  %x = index.add %arg0, %arg0
  %y = index.add %x, %x
  co.suspend (%hdl) {
    co.suspend.end
  }
  kgen.return %y : index
}

// An async function with not callers will be erased by this pass.
// CHECK-NOT: kgen.func @triggerHot
kgen.func @triggerHot(%arg0: index, %arg1: index) async {
   %coro = co.hot_invoke[(index) async -> index: @needsBoth](%arg0)
   kgen.return
}

// A non async function can cold start a coro. It will not be erased by this pass.
// CHECK-LABEL: kgen.func @triggerCold
kgen.func @triggerCold(%arg0: index, %arg1: index) {
   %coro = co.invoke[(index) async -> index: @needsBoth](%arg0)
   kgen.return
}

}

// -----

// COM: Stack Allocations in Hot Frame

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
// CHECK-LABEL: kgen.func @hot_stack_alloc_hot_ramp
kgen.func @hot_stack_alloc(%arg1: index) async -> index {
 // CHECK-NEXT:   %idx64 = index.constant 64
 // CHECK-NEXT:   %idx8 = index.constant 8
 // CHECK-NEXT:   [[CORO:%.*]] = pop.aligned_alloc

 // CHECK:        pop.pointer.bitcast [[CORO]]
 // CHECK-NEXT:   [[V9:%.*]] = kgen.struct.gep %0[[[#FRAME7:]]]
 // CHECK-NEXT:   pop.store %arg2, [[V9]] : !kgen.pointer<index>
 %0 = pop.stack_allocation 1 x index marked
 pop.store %arg1, %0 : !kgen.pointer<index>
 co.suspend (%hdl) {
    co.suspend.end
 }
 %2 = pop.load %0 : !kgen.pointer<index>
 kgen.return %2 : index
}

// CHECK-LABEL: kgen.func @hot_stack_alloc_no_sus_hot_ramp
kgen.func @hot_stack_alloc_no_sus(%arg1: index) async -> index {
 // CHECK-NEXT:   %idx64 = index.constant 64
 // CHECK-NEXT:   %idx8 = index.constant 8
 // CHECK-NEXT:   [[CORO2:%.*]] = pop.aligned_alloc

 // CHECK:        pop.pointer.bitcast [[CORO2]]
 // CHECK-NEXT:   [[V9:%.*]] = pop.stack_allocation 1 x index marked
 // CHECK-NEXT:   pop.store %arg2, [[V9]] : !kgen.pointer<index>
 %0 = pop.stack_allocation 1 x index marked
 pop.store %arg1, %0 : !kgen.pointer<index>
 %2 = pop.load %0 : !kgen.pointer<index>
 co.suspend (%hdl) {
    co.suspend.end
 }
 kgen.return %2 : index
}

// CHECK-LABEL: kgen.func @trigger_creation_resume
kgen.func @trigger_creation(%arg1: index) async {
   %coro = co.hot_invoke[(index) async -> index: @hot_stack_alloc](%arg1)
   %coro2 = co.hot_invoke[(index) async -> index: @hot_stack_alloc_no_sus](%arg1)
   kgen.return
}

kgen.func @trigger_trigger_creation(%arg1: index) {
  %coro = co.invoke[(index) async -> ():@trigger_creation](%arg1)
  kgen.return
}

}

// -----

// COM: Hot Invoke With Results

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
kgen.func @foo(%arg1: index) async -> index {
 co.suspend (%hdl) {
    co.suspend.end
 }
 kgen.return %arg1 : index
}

// CHECK-LABEL: kgen.func @trigger_creation_resume
kgen.func @trigger_creation(%arg1: index) async -> index {
   // CHECK: [[CORO:%.*]] = kgen.call @foo_hot_ramp

   // Store coro in frame
   // CHECK-NEXT: [[CORO_SLOT:%.*]] = kgen.struct.gep %arg0[[[#FRAME7:]]]
   // CHECK-NEXT: pop.store [[CORO]], [[CORO_SLOT]]
   %result = co.hot_invoke[(index) async -> index: @foo](%arg1)

   // Access results on callback
   // CHECK-NEXT: co.suspend
   // CHECK-NEXT: }
   // CHECK-NEXT: [[CORO_SLOT2:%.*]] = kgen.struct.gep %arg0[[[#FRAME7]]]
   // CHECK-NEXT: [[CORO2:%.*]] = pop.load [[CORO_SLOT2]]
   // CHECK-NEXT: [[CORO_WITH_PROMISE:%.*]] = pop.pointer.bitcast [[CORO2]]
   // CHECK-NEXT: [[PROMISE_SLOT:%.*]] = kgen.struct.gep [[CORO_WITH_PROMISE]][[[#PROMISE_IDX:]]]
   // CHECK-NEXT: [[PROMISE_PTR:%.*]] = kgen.struct.gep [[PROMISE_SLOT]][0] : <struct<(index)>>
   // CHECK-NEXT: [[PROMISE:%.*]] = pop.load [[PROMISE_PTR]] : !kgen.pointer<index>

   // Store results of child coro in this coro and return.
   // CHECK-NEXT: [[MY_RESULT_SLOT:%.*]] = kgen.struct.gep %arg0[[[#PROMISE_IDX]]]
   // CHECK-NEXT: [[MY_PROMISE_SLOT:%.*]] = kgen.struct.gep [[MY_RESULT_SLOT]][0] : <struct<(index)>>
   // CHECK-NEXT: pop.store [[PROMISE]], [[MY_PROMISE_SLOT]] : !kgen.pointer<index>
   // CHECK-NEXT: kgen.return
   kgen.return %result : index
}

kgen.func @trigger_trigger_creation(%arg1: index) {
  %coro = co.invoke[(index) async -> index:@trigger_creation](%arg1)
  kgen.return
}

}

// -----

// COM: Removal of State 0 Virtual Blocks:
// (1) Users of A are deleted before A
// (2) Control Flow Nodes that are not parents of suspend are deleted.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
// CHECK-LABEL: kgen.func @hot_stack_alloc_resume
// CHECK-NEXT:  co.suspend {
// CHECK-NEXT:  co.suspend.end
// CHECK-NEXT:  }
kgen.func @hot_stack_alloc(%arg0: i1, %arg1: index) async -> index {
 %0 = "kgen.param.constant"() {value = #kgen.none : !kgen.none} : () -> !kgen.none
 %1 = pop.stack_allocation 1 x index marked
 %arg0_sb = pop.cast_from_builtin %arg0 : i1 to !kgen.scalar<bool>
 hlcf.if %arg0_sb {
  pop.store %arg1, %1 : !kgen.pointer<index>
  hlcf.yield
 } else {
  pop.store %arg1, %1 : !kgen.pointer<index>
  hlcf.yield
 }

 co.suspend (%hdl) {
    %2 = pop.load %1 : !kgen.pointer<index>
    pop.store %2, %1 : !kgen.pointer<index>
    co.suspend.end
 }
 %2 = pop.load %1 : !kgen.pointer<index>
 kgen.return %2 : index
}

// CHECK-LABEL: kgen.func @trigger_creation_resume
kgen.func @trigger_creation(%arg0: i1, %arg1: index) async {
   %coro = co.hot_invoke[(i1, index) async -> index: @hot_stack_alloc](%arg0, %arg1)
   kgen.return
}

kgen.func @trigger_trigger_creation(%arg0: i1, %arg1: index) {
  %coro = co.invoke[(i1, index) async -> ():@trigger_creation](%arg0, %arg1)
  kgen.return
}

}
