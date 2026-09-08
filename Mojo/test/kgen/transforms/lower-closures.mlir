// RUN: kgen-opt -lower-closures -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: kgen.func @execute_byref_async_closure_0
// CHECK-SAME:    (%arg0: index, %arg1: !kgen.pointer<index> byref_result) async {
// CHECK-NEXT:    store %arg0, %arg1

// CHECK-LABEL: kgen.func @execute_byref_async_closure_1
// CHECK-SAME:    (%arg0: index, %arg1: !kgen.pointer<index> byref_error, %arg2: !kgen.pointer<index> byref_result) throws|async {
// CHECK-NEXT:    store %arg0, %arg1
// CHECK-NEXT:    store %arg0, %arg2

// CHECK-LABEL: kgen.func @execute_byref
kgen.func @execute_byref(%arg0: index) {
  // CHECK-NEXT: co.invoke[{{.*}}: @execute_byref_async_closure_0](%arg0)
  co.execute (%arg1: !kgen.pointer<index> byref_result) {
    pop.store %arg0, %arg1 : !kgen.pointer<index>
    kgen.return
  }
  // CHECK-NEXT: co.invoke[{{.*}}: @execute_byref_async_closure_1](%arg0)
  co.execute (%arg1: !kgen.pointer<index> byref_error, %arg2: !kgen.pointer<index> byref_result) {
    pop.store %arg0, %arg1 : !kgen.pointer<index>
    pop.store %arg0, %arg2 : !kgen.pointer<index>
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.func @async_execute_async_closure
// CHECK-SAME: (%arg0: index) async -> index {
// CHECK-NEXT: kgen.return %arg0

// CHECK-LABEL: kgen.func @async_execute_async_closure_{{[0-9]}}
// CHECK-SAME: (%arg0: index, %arg1: index) async -> index
// CHECK-NEXT: %idx1 = index.constant 1
// CHECK-NEXT: %0 = index.add %idx1, %arg0
// CHECK-NEXT: %1 = index.add %0, %arg1
// CHECK-NEXT: kgen.return %1

// CHECK-LABEL: kgen.func @async_execute_async_closure_{{[0-9]}}
// CHECK-SAME: (%arg0: index, %arg1: index) async
// CHECK-NEXT: %idx1 = index.constant 1
// CHECK-NEXT: co.invoke[{{.*}}: @async_execute_async_closure_{{[0-9]}}](%arg0, %arg1)
// CHECK-NEXT: kgen.return

// CHECK-LABEL: kgen.func @async_execute
kgen.func @async_execute(%arg0: index) {
  // CHECK: index.add
  // CHECK-NEXT: co.invoke[{{.*}}: @async_execute_async_closure_0](%arg0)
  // CHECK-NEXT: co.invoke[{{.*}}: @async_execute_async_closure_2](%arg0, %0)
  %arg1 = index.add %arg0, %arg0
  %0 = co.execute : index {
    kgen.return %arg0 : index
  }
  %1 = co.execute {
    %idx1 = index.constant 1
    %2 = co.execute : index {
      %3 = index.add %idx1, %arg0
      %4 = index.add %3, %arg1
      kgen.return %4 : index
    }
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.func @co_await
kgen.func @co_await(%arg0: !co.routine, %arg1: !kgen.pointer<index>) {
  // CHECK-NEXT: set_byref_error_result %arg0(%arg1, %arg1)
  // CHECK-NEXT: co.suspend (%hdl)
  // CHECK-NEXT:   [[CALLBACK:%.*]] = co.get_callback_ptr %arg0
  // CHECK-NEXT:   [[FN_PTR:%.*]] = kgen.struct.gep [[CALLBACK]][0]
  // CHECK-NEXT:   [[HDL_PTR:%.*]] = kgen.struct.gep [[CALLBACK]][1]
  // CHECK-NEXT:   [[FN:%.*]] = co.resume %hdl
  // CHECK-NEXT:   store [[FN]], [[FN_PTR]]
  // CHECK-NEXT:   store %hdl, [[HDL_PTR]]
  // CHECK-NEXT:   [[RESUME:%.*]] = co.resume %arg0
  // CHECK-NEXT:   call_indirect [[RESUME]](%arg0)
  // CHECK-NEXT:   co.suspend.end
  // CHECK-NEXT: }
  // CHECK-NEXT: co.destroy %arg0
  // CHECK-NEXT: return
  co.await %arg0, %arg1, %arg1 : (!co.routine, !kgen.pointer<index>, !kgen.pointer<index>) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @co_raises
kgen.func @co_raises(%arg0: !co.routine, %arg1: !kgen.pointer<index>, %arg2: !kgen.pointer<index>) -> i1 {
  // CHECK-NEXT: set_byref_error_result %arg0(%arg1, %arg2)
  // CHECK: [[RES:%.*]] = co.get_results %arg0
  // CHECK-NEXT: co.destroy %arg0
  // CHECK-NEXT: return [[RES]]
  %0 = co.await %arg0, %arg1, %arg2 : (!co.routine, !kgen.pointer<index>, !kgen.pointer<index>) -> i1
  kgen.return %0 : i1
}

// CHECK: kgen.func @some_closure(%arg0: index) capturing -> index

// CHECK-LABEL: kgen.func @main_closure_arg
kgen.func @main_closure_arg(%arg0: index) {
  // CHECK: kgen.create_closure[(index) capturing -> index: @some_closure](%arg0)
  %0 = kgen.stage_closure = () capturing -> index {
    kgen.return %arg0 : index
  } { name = "some_closure" }
  kgen.return
}

// CHECK: kgen.func @two_captures(%arg0: si32, %arg1: si64, %arg2: index) capturing -> index

// CHECK-LABEL: kgen.func @capturing_region
kgen.func @capturing_region(%arg0: si32, %arg1: si64) {
  %idx4 = index.constant 4
  // CHECK: kgen.create_closure[(si32, si64, index) capturing -> index: @two_captures](%arg0, %arg1)
  %0 = kgen.stage_closure = (%arg2: index) capturing -> index {
    "unregistered_op_to_capture"(%arg0, %arg1) : (si32, si64) -> ()
    kgen.return %arg2 : index
  } { name = "two_captures" }
  %1 = kgen.call_indirect %0(%idx4) : (index) capturing -> index
  kgen.return
}

// CHECK: kgen.func @no_name_attr_closure_0(%arg0: index) capturing -> index
// CHECK: kgen.func @no_name_attr_closure_1(%arg0: index) capturing -> index

// CHECK-LABEL: kgen.func @no_name_attr(
kgen.func @no_name_attr(%arg0: index, %arg1: index) {
  // CHECK: kgen.create_closure[(index) capturing -> index: @no_name_attr_closure_0](%arg0)
  %0 = kgen.stage_closure = () capturing -> index {
    kgen.return %arg0 : index
  }
  // CHECK: kgen.create_closure[(index) capturing -> index: @no_name_attr_closure_1](%arg1)
  %1 = kgen.stage_closure = () capturing -> index {
    kgen.return %arg1 : index
  } { name = 6 }
  kgen.return
}

// CHECK: kgen.func @constant_in_closure_0() capturing -> index

// CHECK-LABEL: kgen.func @constant_in(
kgen.func @constant_in(%arg0: index, %arg1: index) {
  %idx4 = index.constant 4
  // CHECK: kgen.create_closure[() capturing -> index: @constant_in_closure_0]()
  %0 = kgen.stage_closure = () capturing -> index {
    kgen.return %idx4 : index
  }
  kgen.return
}

// CHECK-LABEL: kgen.func @create_closure(
kgen.func @create_closure() {
  // CHECK: %0 = kgen.create_closure[() -> (): @create_closure_closure_0]()
  %0 = kgen.stage_closure = () {
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: @transitive_closure_closure_0
// CHECK-NEXT: %string = kgen.param.constant: string = <"123">
// CHECK-NEXT: %0 = pop.string.address %string
// CHECK-NEXT: return %0

// CHECK-LABEL: @transitive_closure
kgen.func @transitive_closure() {
  %0 = kgen.param.constant: string = <"123">
  %1 = pop.string.address %0
  // CHECK: create_closure[() -> !kgen.pointer<scalar<si8>>: @transitive_closure_closure_0]()
  kgen.stage_closure = () -> !kgen.pointer<scalar<si8>> {
    kgen.return %1 : !kgen.pointer<scalar<si8>>
  }
  kgen.return
}

// CHECK-LABEL: @transitive_closure_cloning_closure_0
// CHECK-NEXT: %string = kgen.param.constant: string = <"123">
// CHECK-DAG: [[V0:%.*]] = pop.string.address %string
// CHECK-DAG: %idx2 = index.constant 2
// CHECK-NEXT: [[V1:%.*]] = pop.pointer.bitcast [[V0]] : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
// CHECK: [[V2:%.*]] = pop.offset [[V0]][%idx2] : !kgen.pointer<scalar<si8>>
// CHECK-NEXT: [[V3:%.*]] = pop.pointer.bitcast [[V2]] : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
// CHECK-NEXT: kgen.return [[V1]], [[V3]] : !kgen.pointer<none>, !kgen.pointer<none>

// CHECK-LABEL: @transitive_closure_cloning
kgen.func @transitive_closure_cloning() {
  %str = kgen.param.constant: string = <"123">
  %0 = pop.string.address %str
  %idx2 = index.constant 2
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>
  %2 = pop.offset %0[%idx2] : !kgen.pointer<scalar<si8>>
  %3 = pop.pointer.bitcast %2 : !kgen.pointer<scalar<si8>> to !kgen.pointer<none>

  // CHECK: create_closure[() -> (!kgen.pointer<none>, !kgen.pointer<none>): @transitive_closure_cloning_closure_0]()
  kgen.stage_closure = () -> (!kgen.pointer<none>, !kgen.pointer<none>) {
    kgen.return %1, %3 : !kgen.pointer<none>, !kgen.pointer<none>
  }
  kgen.return
}
