// RUN: kgen-opt %s -lower-suspension-points -split-input-file | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // COM: STUBS
  llvm.func internal @anotherTask() -> !llvm.ptr {
    %nil = llvm.mlir.constant(0 : i8) : i8
    %nilptr = builtin.unrealized_conversion_cast %nil : i8 to !llvm.ptr
    llvm.return %nilptr : !llvm.ptr
  }

  llvm.func internal @print(%arg0: i1) {
    llvm.return
  }

  llvm.func internal @getElementFromContext(%continuation: !llvm.ptr) -> i1 {
    %cond = llvm.mlir.constant(0 : i1) : i1
    llvm.return %cond : i1
  }
// CHECK-LABEL:  llvm.func @coro
// CHECK-NEXT:    [[STATE_SLOT:%.*]] = llvm.getelementptr %arg0[0, 0]
// CHECK-NEXT:    [[STATE:%.*]] = llvm.load [[STATE_SLOT]] : !llvm.ptr -> i32
// CHECK-NEXT:    llvm.switch [[STATE]] : i32, ^bb1 [
// CHECK-NEXT:      1: ^bb3,
// CHECK-NEXT:      0: ^bb1
// CHECK-NEXT:    ]
// CHECK-NEXT:  ^bb1:  // 2 preds: ^bb0, ^bb0
// CHECK-NEXT:    [[V2:%.*]] = llvm.call @getElementFromContext(%arg0)
// CHECK-NEXT:    llvm.cond_br [[V2]], ^bb2, ^bb4
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    [[V3:%.*]] = llvm.call @anotherTask()
// CHECK-NEXT:    [[V8:%.*]] = llvm.call @getElementFromContext([[V3]])
// CHECK-NEXT:    llvm.call @print([[V8]])
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  ^bb3:  // pred: ^bb0
// CHECK-NEXT:    [[V9:%.*]] = llvm.call @getElementFromContext(%arg0)
// CHECK-NEXT:    llvm.call @print([[V9]])
// CHECK-NEXT:    llvm.br ^bb5
// CHECK-NEXT:  ^bb4:  // pred: ^bb1
// CHECK-NEXT:    llvm.call @print([[V2]])
// CHECK-NEXT:    llvm.br ^bb5
// CHECK-NEXT:  ^bb5:  // 2 preds: ^bb3, ^bb4
// CHECK-NEXT:    [[FNC_SLOT:%.*]] = llvm.getelementptr %arg0[0, 2]
// CHECK-NEXT:    [[FUNC:%.*]] = llvm.load [[FNC_SLOT]]
// CHECK-NEXT:    [[CLSR_SLOT:%.*]] = llvm.getelementptr %arg0[0, 3]
// CHECK-NEXT:    [[CLOSURE_STATE:%.*]] = llvm.load [[CLSR_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    llvm.call musttail [[FUNC]]([[CLOSURE_STATE]]) : !llvm.ptr, (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
  llvm.func @coro(%continuation: !llvm.ptr) attributes { coroutineType = !llvm.struct<(i32, ptr, ptr, ptr, ptr)> } {
    %cond = llvm.call @getElementFromContext(%continuation) : (!llvm.ptr) -> i1
    llvm.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %someContinuation = llvm.call @anotherTask() : () -> !llvm.ptr
    co.suspend {
      %cond1 = llvm.call @getElementFromContext(%someContinuation) : (!llvm.ptr) -> i1
      llvm.call @print(%cond1) : (i1) -> ()
	    co.suspend.end
    }
    %cond1 = llvm.call @getElementFromContext(%continuation) : (!llvm.ptr) -> i1
    llvm.call @print(%cond1) : (i1) -> ()
    llvm.br ^bb3
  ^bb2:
    llvm.call @print(%cond) : (i1) -> ()
    llvm.br ^bb3
  ^bb3:
    llvm.return
  }

// CHECK-LABEL:  llvm.func @coro_multiple_suspends
// CHECK-NEXT:    [[STATE_SLOT:%.*]] = llvm.getelementptr %arg0[0, 0]
// CHECK-NEXT:    [[STATE:%.*]] = llvm.load [[STATE_SLOT]]
// CHECK-NEXT:    llvm.switch [[STATE]] : i32, ^bb1 [
// CHECK-NEXT:      1: ^bb3,
// CHECK-NEXT:      2: ^bb4,
// CHECK-NEXT:      0: ^bb1
// CHECK-NEXT:    ]
// CHECK-NEXT:  ^bb1:  // 2 preds: ^bb0, ^bb0
// CHECK-NEXT:    [[V2:%.*]] = llvm.call @getElementFromContext(%arg0)
// CHECK-NEXT:    llvm.cond_br [[V2]], ^bb2, ^bb5
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    [[V3:%.*]] = llvm.call @anotherTask()
// CHECK-NEXT:    [[V8:%.*]] = llvm.call @getElementFromContext([[V3]])
// CHECK-NEXT:    llvm.call @print([[V8]])
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  ^bb3:  // pred: ^bb0
// CHECK-NEXT:    [[V9:%.*]] = llvm.call @anotherTask() : () -> !llvm.ptr
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  ^bb4:  // pred: ^bb0
// CHECK-NEXT:    llvm.call @getElementFromContext
// CHECK-NEXT:    llvm.call @print
// CHECK-NEXT:    llvm.br ^bb6
// CHECK-NEXT:  ^bb5:  // pred: ^bb1
// CHECK-NEXT:    llvm.call @print([[V2]]) : (i1) -> ()
// CHECK-NEXT:    llvm.br ^bb6
// CHECK-NEXT:  ^bb6:  // 2 preds: ^bb4, ^bb5
// CHECK-NEXT:    [[FNC_SLOT:%.*]] = llvm.getelementptr %arg0[0, 2]
// CHECK-NEXT:    [[FUNC:%.*]] = llvm.load [[FNC_SLOT]]
// CHECK-NEXT:    [[CLSR_SLOT:%.*]] = llvm.getelementptr %arg0[0, 3]
// CHECK-NEXT:    [[CLOSURE_STATE:%.*]] = llvm.load [[CLSR_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    llvm.call musttail [[FUNC]]([[CLOSURE_STATE]]) : !llvm.ptr, (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
  llvm.func @coro_multiple_suspends(%continuation: !llvm.ptr) attributes { coroutineType = !llvm.struct<(i32, ptr, ptr, ptr, ptr)> } {
    %cond = llvm.call @getElementFromContext(%continuation) : (!llvm.ptr) -> i1
    llvm.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %someContinuation = llvm.call @anotherTask() : () -> !llvm.ptr
    co.suspend {
      %cond1 = llvm.call @getElementFromContext(%someContinuation) : (!llvm.ptr) -> i1
      llvm.call @print(%cond1) : (i1) -> ()
	    co.suspend.end
    }
    %someContinuation2 = llvm.call @anotherTask() : () -> !llvm.ptr
    co.suspend {
      co.suspend.end
    }
    %cond1 = llvm.call @getElementFromContext(%continuation) : (!llvm.ptr) -> i1
    llvm.call @print(%cond1) : (i1) -> ()
    llvm.br ^bb3
  ^bb2:
    llvm.call @print(%cond) : (i1) -> ()
    llvm.br ^bb3
  ^bb3:
    llvm.return
 }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
 // STUBS
 llvm.func internal @anotherTask() -> !llvm.ptr {
  %nil = llvm.mlir.constant(0 : i8) : i8
  %nilptr = builtin.unrealized_conversion_cast %nil : i8 to !llvm.ptr
  llvm.return %nilptr : !llvm.ptr
 }

 llvm.func internal @print(%arg0: i1) {
  llvm.return
 }

 llvm.func internal @print2(%arg0: i1) {
  llvm.return
 }

 llvm.func internal @print3(%arg0: i1) {
  llvm.return
 }

 llvm.func internal @getElementFromContext(%continuation: !llvm.ptr) -> i1 {
   %cond = llvm.mlir.constant(0 : i1) : i1
   llvm.return %cond : i1
 }

// CHECK-LABEL:  llvm.func @no_suspoints
// CHECK-NEXT:    [[COND:%.*]] = llvm.call @getElementFromContext(%arg0)
// CHECK-NEXT:    llvm.cond_br [[COND]], ^bb1, ^bb2
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    [[COND1:%.*]] = llvm.call @getElementFromContext(%arg0)
// CHECK-NEXT:    llvm.call @print([[COND1]])
// CHECK-NEXT:    llvm.cond_br [[COND1]], ^bb3, ^bb4
// CHECK-NEXT:  ^bb2:  // pred: ^bb0
// CHECK-NEXT:    llvm.call @print([[COND]])
// CHECK-NEXT:    [[FNC_SLOT:%.*]] = llvm.getelementptr %arg0[0, 2]
// CHECK-NEXT:    [[FUNC:%.*]] = llvm.load [[FNC_SLOT]]
// CHECK-NEXT:    [[CLSR_SLOT:%.*]] = llvm.getelementptr %arg0[0, 3]
// CHECK-NEXT:    [[CLOSURE_STATE:%.*]] = llvm.load [[CLSR_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    llvm.call musttail [[FUNC]]([[CLOSURE_STATE]]) : !llvm.ptr, (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  ^bb3:  // pred: ^bb1
// CHECK-NEXT:    llvm.call @print2(%0) : (i1) -> ()
// CHECK-NEXT:    [[FNC_SLOT:%.*]] = llvm.getelementptr %arg0[0, 2]
// CHECK-NEXT:    [[FUNC:%.*]] = llvm.load [[FNC_SLOT]]
// CHECK-NEXT:    [[CLSR_SLOT:%.*]] = llvm.getelementptr %arg0[0, 3]
// CHECK-NEXT:    [[CLOSURE_STATE:%.*]] = llvm.load [[CLSR_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    llvm.call musttail [[FUNC]]([[CLOSURE_STATE]]) : !llvm.ptr, (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  ^bb4:  // pred: ^bb1
// CHECK-NEXT:    llvm.call @print3(%0) : (i1) -> ()
// CHECK-NEXT:    [[FNC_SLOT:%.*]] = llvm.getelementptr %arg0[0, 2]
// CHECK-NEXT:    [[FUNC:%.*]] = llvm.load [[FNC_SLOT]]
// CHECK-NEXT:    [[CLSR_SLOT:%.*]] = llvm.getelementptr %arg0[0, 3]
// CHECK-NEXT:    [[CLOSURE_STATE:%.*]] = llvm.load [[CLSR_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    llvm.call musttail [[FUNC]]([[CLOSURE_STATE]]) : !llvm.ptr, (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
  llvm.func @no_suspoints(%continuation: !llvm.ptr) attributes { coroutineType = !llvm.struct<(i32, ptr, ptr, ptr, ptr)> } {
    %cond = llvm.call @getElementFromContext(%continuation) : (!llvm.ptr) -> i1
    llvm.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %cond1 = llvm.call @getElementFromContext(%continuation) : (!llvm.ptr) -> i1
    llvm.call @print(%cond1) : (i1) -> ()
    llvm.cond_br %cond1, ^bb3, ^bb5
  ^bb2:
    llvm.call @print(%cond) : (i1) -> ()
    llvm.return
  ^bb3:
    llvm.call @print2(%cond) : (i1) -> ()
    llvm.return
  ^bb5:
   llvm.call @print3(%cond) : (i1) -> ()
   llvm.return
}

// CHECK-LABEL:  llvm.func @multiple_exits
// CHECK:         ^bb4:  // pred: ^bb1
// CHECK-NEXT:    llvm.call @print(%2) : (i1) -> ()
// CHECK-NEXT:    [[FNC_SLOT:%.*]] = llvm.getelementptr %arg0[0, 2]
// CHECK-NEXT:    [[FUNC:%.*]] = llvm.load [[FNC_SLOT]]
// CHECK-NEXT:    [[CLSR_SLOT:%.*]] = llvm.getelementptr %arg0[0, 3]
// CHECK-NEXT:    [[CLOSURE_STATE:%.*]] = llvm.load [[CLSR_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    llvm.call musttail [[FUNC]]([[CLOSURE_STATE]]) : !llvm.ptr, (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:    ^bb5:  // pred: ^bb3
// CHECK-NEXT:    [[FNC_SLOT:%.*]] = llvm.getelementptr %arg0[0, 2]
// CHECK-NEXT:    [[FUNC:%.*]] = llvm.load [[FNC_SLOT]]
// CHECK-NEXT:    [[CLSR_SLOT:%.*]] = llvm.getelementptr %arg0
// CHECK-NEXT:    [[CLOSURE_STATE:%.*]] = llvm.load [[CLSR_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    llvm.call musttail [[FUNC]]([[CLOSURE_STATE]]) : !llvm.ptr, (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:    ^bb6:  // pred: ^bb3
// CHECK-NEXT:    [[FNC_SLOT:%.*]] = llvm.getelementptr %arg0[0, 2]
// CHECK-NEXT:    [[FUNC:%.*]] = llvm.load [[FNC_SLOT]]
// CHECK-NEXT:    [[CLSR_SLOT:%.*]] = llvm.getelementptr %arg0[0, 3]
// CHECK-NEXT:    [[CLOSURE_STATE:%.*]] = llvm.load [[CLSR_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    llvm.call musttail [[FUNC]]([[CLOSURE_STATE]]) : !llvm.ptr, (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return
  llvm.func @multiple_exits(%continuation: !llvm.ptr) attributes { coroutineType = !llvm.struct<(i32, ptr, ptr, ptr, ptr)> } {
    %cond = llvm.call @getElementFromContext(%continuation) : (!llvm.ptr) -> i1
    llvm.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %someContinuation2 = llvm.call @anotherTask() : () -> !llvm.ptr
    co.suspend {
      co.suspend.end
    }
    %cond1 = llvm.call @getElementFromContext(%continuation) : (!llvm.ptr) -> i1
    llvm.call @print(%cond1) : (i1) -> ()
    llvm.cond_br %cond1, ^bb3, ^bb5
  ^bb2:
    llvm.call @print(%cond) : (i1) -> ()
    llvm.return
  ^bb3:
    llvm.return
  ^bb5:
    llvm.return
  }
}

// -----

// COM: Support Control Flow In Suspension Point

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: llvm.func internal @exec_async
// CHECK:     llvm.switch {{.*}} : i32, ^bb1 [
// CHECK-NEXT:  1: ^bb5,
// CHECK-NEXT:  0: ^bb1
// CHECK-NEXT: ]

// First block contains original entry code + update state + first block of suspension body.
// CHECK-NEXT: ^bb1:  // 2 preds: ^bb0, ^bb0
// CHECK-NEXT:   [[V2:%.*]] = llvm.getelementptr %arg0[0, 9]
// CHECK-NEXT:   [[V3:%.*]] = llvm.load [[V2]]

// CHECK-NEXT: llvm.cond_br [[V3]], ^bb2, ^bb3

// Blocks 2 through 4 are lifted from suspension body
// CHECK-NEXT: ^bb2:  // pred: ^bb1
// CHECK-NEXT:   [[V8:%.*]] = llvm.getelementptr %arg0[0, 8]
// CHECK-NEXT:   [[V9:%.*]] = llvm.load [[V8]]
// CHECK-NEXT:   llvm.call @print([[V9]]) : (f32) -> ()
// CHECK-NEXT:   llvm.br ^bb4
// CHECK-NEXT: ^bb3:  // pred: ^bb1
// CHECK-NEXT:   [[V10:%.*]] = llvm.getelementptr %arg0[0, 7]
// CHECK-NEXT:   [[V11:%.*]] = llvm.load [[V10]]
// CHECK-NEXT:   llvm.call @print([[V11]]) : (f32) -> ()
// CHECK-NEXT:   llvm.br ^bb4
// CHECK-NEXT: ^bb4:  // 2 preds: ^bb2, ^bb3
// CHECK-NEXT:   llvm.return

// Final block is the code following the suspension point.
// CHECK-NEXT: ^bb5:  // pred: ^bb0
// CHECK-NEXT:   [[V12:%.*]] = llvm.getelementptr %arg0[0, 2]
// CHECK-NEXT:   [[V13:%.*]] = llvm.load [[V12]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:   [[V14:%.*]] = llvm.getelementptr %arg0[0, 3]
// CHECK-NEXT:   [[V15:%.*]] = llvm.load [[V14]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:   llvm.call musttail [[V13]]([[V15]]) : !llvm.ptr, (!llvm.ptr) -> ()
// CHECK-NEXT:   llvm.return
// CHECK-NEXT: }
llvm.func internal @exec_async_closure_0_resume(%arg0: !llvm.ptr) attributes {coroutineType = !llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, struct<(i1)>, f32, f32, i1)>} {
  %2 = llvm.getelementptr %arg0[0, 9] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, struct<(i1)>, f32, f32, i1)>
  %3 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i1
  co.suspend {
    llvm.cond_br %3, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %21 = llvm.getelementptr %arg0[0, 8] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, struct<(i1)>, f32, f32, i1)>
    %22 = llvm.load %21 {alignment = 8 : i64} : !llvm.ptr -> f32
    llvm.call @print(%22) : (f32) -> ()
    llvm.br ^bb3
  ^bb2:  // pred: ^bb0
    %23 = llvm.getelementptr %arg0[0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, struct<(i1)>, f32, f32, i1)>
    %24 = llvm.load %23 {alignment = 8 : i64} : !llvm.ptr -> f32
    llvm.call @print(%24) : (f32) -> ()
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    co.suspend.end
  }
  llvm.return
}

llvm.func internal @print(%arg0: f32) {
  llvm.return
}

}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  llvm.func @f(%arg0: !llvm.ptr {llvm.noundef}) attributes {coroutineType = !llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, struct<()>, ptr, ptr, struct<(ptr, i64)>, ptr, ptr, i64, i64, i8, i64, i64, i64, struct<(ptr, i64)>, i64, i64, struct<()>)>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(20 : i64) : i64
    // CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb0
    // CHECK:  llvm.br ^bb2({{.*}} : i64)

    // CHECK:      ^bb2([[ARG:%.*]]: i64):  // 2 preds: ^bb1, ^bb5
    // CHECK-NEXT:   [[V10:%.*]] = llvm.icmp "slt" [[ARG]], {{.*}} : i64
    // CHECK-NEXT:   llvm.cond_br [[V10]], ^bb3, ^bb4
    // CHECK-NEXT: ^bb3:  // pred: ^bb2
    // CHECK-NEXT:   llvm.br ^bb5
    // CHECK-NEXT: ^bb4:  // pred: ^bb2
    // CHECK-NEXT:   llvm.br ^bb6
    // CHECK-NEXT: ^bb5:  // pred: ^bb3
    // CHECK-NEXT:   [[V11:%.*]] = llvm.add [[ARG]], {{.*}} : i64
    // CHECK-NEXT:   llvm.br ^bb2([[V11]] : i64)
    // CHECK-NEXT: ^bb6:  // pred: ^bb4
    // CHECK-NEXT:   llvm.return
    co.suspend {
      llvm.br ^bb1(%0 : i64)
    ^bb1(%3: i64):  // 2 preds: ^bb0, ^bb4
      %4 = llvm.icmp "slt" %3, %2 : i64
      llvm.cond_br %4, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      llvm.br ^bb4
    ^bb3:  // pred: ^bb1
      llvm.br ^bb5
    ^bb4:  // pred: ^bb2
      %5 = llvm.add %3, %1 : i64
      llvm.br ^bb1(%5 : i64)
    ^bb5:  // pred: ^bb3
      co.suspend.end
    }
    llvm.return
  }
}
