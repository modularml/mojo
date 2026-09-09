// RUN: kgen-opt -pass-pipeline='builtin.module(kgen.func(hoist-trivial-invariants))' -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @basic
kgen.func @basic(%arg0: !kgen.pointer<struct<(struct<(scalar<index>)>, struct<(scalar<index>)>, struct<(scalar<index>)>)>>) {
  %struct = pop.load %arg0 : !kgen.pointer<struct<(struct<(scalar<index>)>, struct<(scalar<index>)>, struct<(scalar<index>)>)>>

  // CHECK: %[[LOAD:.*]] = pop.load
  // CHECK-NEXT: kgen.struct.extract %[[LOAD]][2]
  // CHECK-NEXT: kgen.struct.extract %[[LOAD]][1]
  // CHECK-NEXT: kgen.struct.extract %[[LOAD]][0]

  // Loops should now be empty.
  // CHECK-NOT: kgen.struct.extract

  hlcf.loop {
    %0 = kgen.struct.extract %struct[0] : !kgen.struct<(struct<(scalar<index>)>, struct<(scalar<index>)>, struct<(scalar<index>)>)>
    %1 = kgen.struct.extract %struct[1] : !kgen.struct<(struct<(scalar<index>)>, struct<(scalar<index>)>, struct<(scalar<index>)>)>
    %2 = kgen.struct.extract %struct[2] : !kgen.struct<(struct<(scalar<index>)>, struct<(scalar<index>)>, struct<(scalar<index>)>)>
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @many_nests
kgen.func @many_nests(%arg0: !kgen.pointer<struct<(struct<(scalar<index>)>, struct<(scalar<index>)>, struct<(scalar<index>)>)>>, %cond: !kgen.scalar<bool>) {
  %struct = pop.load %arg0 : !kgen.pointer<struct<(struct<(scalar<index>)>, struct<(scalar<index>)>, struct<(scalar<index>)>)>>

  // CHECK: %[[LOAD:.*]] = pop.load
  // CHECK-NEXT: kgen.struct.extract %[[LOAD]][2]
  // CHECK-NEXT: kgen.struct.extract %[[LOAD]][2]
  // CHECK-NEXT: kgen.struct.extract %[[LOAD]][1]
  // CHECK-NEXT: kgen.struct.extract %[[LOAD]][0]

  // Loops should now be empty.
  // CHECK-NOT: kgen.struct.extract

  hlcf.loop {
    %0 = kgen.struct.extract %struct[0] : !kgen.struct<(struct<(scalar<index>)>, struct<(scalar<index>)>, struct<(scalar<index>)>)>
    hlcf.loop {
      %1 = kgen.struct.extract %struct[1] : !kgen.struct<(struct<(scalar<index>)>, struct<(scalar<index>)>, struct<(scalar<index>)>)>
      hlcf.if %cond {
        %2 = kgen.struct.extract %struct[2] : !kgen.struct<(struct<(scalar<index>)>, struct<(scalar<index>)>, struct<(scalar<index>)>)>
        hlcf.yield
      } else {
        %2 = kgen.struct.extract %struct[2] : !kgen.struct<(struct<(scalar<index>)>, struct<(scalar<index>)>, struct<(scalar<index>)>)>
        hlcf.yield
      }
      hlcf.break
    }
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @memory_ops_untouched
kgen.func @memory_ops_untouched(%input: !kgen.pointer<index>, %output: !kgen.pointer<index>) {
  // CHECK: hlcf.loop
  // CHECK-NEXT: hlcf.loop
  // CHECK-NEXT: pop.load
  // CHECK-NEXT: pop.store
  hlcf.loop {
    hlcf.loop {
      %load = pop.load %input : !kgen.pointer<index>
      pop.store %load, %output  : !kgen.pointer<index>
      hlcf.break
    }
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @hoist_loop_index
kgen.func @hoist_loop_index(%arg0: index, %cond: !kgen.scalar<bool>) {
  // CHECK-NEXT: hlcf.loop
  hlcf.loop (%arg1 = %arg0 : index) {
    // CHECK-NEXT: index.add
    // CHECK-NEXT: hlcf.if
    hlcf.if %cond {
      %0 = index.add %arg1, %arg0
      hlcf.yield
    } else {
      hlcf.yield
    }
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @hoist_nested_funcs
kgen.func @hoist_nested_funcs(%arg0: index) {
  // CHECK-NEXT: index.sub
  // CHECK-NEXT: index.constant 0
  %idx0 = index.constant 0
  // CHECK-NEXT: index.add
  // CHECK-NEXT: hlcf.loop
  hlcf.loop {
    %0 = index.add %arg0, %idx0
    %1 = index.sub %arg0, %arg0
    // CHECK-NEXT: stage_closure
    %2 = kgen.stage_closure = (%arg1: index) -> () {
      // CHECK-NEXT: index.sub
      // CHECK-NEXT: index.constant 1
      %idx1 = index.constant 1
      // CHECK-NEXT: index.mul
      // CHECK-NEXT: index.add
      // CHECK-NEXT: hlcf.loop
      hlcf.loop {
        %3 = index.add %arg1, %idx1
        %4 = index.sub %arg0, %idx0
        %5 = index.mul %idx1, %idx0
        // Note that divs is not pure and should not be hoisted
        // CHECK-NEXT: index.divs
        %6 = index.divs %arg1, %arg0
        hlcf.continue
      }
      kgen.return
    }
    hlcf.continue
  }
  kgen.return
}
