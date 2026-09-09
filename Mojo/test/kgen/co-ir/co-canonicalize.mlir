// RUN: kgen-opt -canonicalize -allow-unregistered-dialect -mlir-print-debuginfo %s | FileCheck %s

// COM: Check that constant are only hoisted from subprogram regions if there is
// COM: no debuginfo scope given.

#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#subprogram1 = #debuginfo.subprogram<sourceName = <"SomeClosure">> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#loc1 = loc("foo.mlir":44:1)
#loc2 = loc("foo.mlir":325:11)
#loc3 = loc("bar.mlir":327:17)
#loc4 = loc(fused<#subprogram>[#loc1])
#loc5 = loc(fused<#subprogram1>[#loc2])
#loc6 = loc(fused<#subprogram1>[#loc3])
#call_loc = #debuginfo.call_loc<#loc4>
#loc7 = loc(fused<#call_loc>[#loc2])
#loc8 = loc(fused<#subprogram1>[#loc7])

// CHECK-LABEL: kgen.func @no_hoist
kgen.func @no_hoist() -> !co.routine {
  // CHECK-NEXT: co.execute {
  %0 = co.execute {
    // CHECK-NEXT: kgen.param.constant: array<1, index> = <[0]>
    %array = kgen.param.constant: array<1, index> = <[0]> loc(#loc6)
    %1 = pop.stack_allocation 1 x !pop.array<1, index>  loc(#loc6)
    pop.store %array, %1 : !kgen.pointer<array<1, index>> loc(#loc6)
    kgen.return loc(#loc5)
  } loc(#loc8)
  kgen.return %0 : !co.routine loc(#loc4)
} loc(#loc4)

// CHECK-LABEL: kgen.func @hoist
kgen.func @hoist() -> !co.routine {
  // CHECK-NEXT: kgen.param.constant: array<1, index> = <[0]>
  // CHECK-NEXT: co.execute {
  %0 = co.execute {
    // CHECK-NOT: kgen.param.constant: array<1, index> = <[0]>
    %array = kgen.param.constant: array<1, index> = <[0]>
    %1 = pop.stack_allocation 1 x !pop.array<1, index>
    pop.store %array, %1 : !kgen.pointer<array<1, index>>
    kgen.return
  }
  kgen.return %0 : !co.routine
}

// CHECK-LABEL: @no_cse_async_execute
kgen.func @no_cse_async_execute() -> (!co.routine, !co.routine) {
  // CHECK-COUNT-2: co.execute
  %0 = co.execute {
    kgen.return
  }
  %1 = co.execute {
    kgen.return
  }
  kgen.return %0, %1 : !co.routine, !co.routine
}

// CHECK-LABEL: kgen.func @await_execute
// CHECK-DAG:     %idx0 =
// CHECK-DAG:     %idx1 =
// CHECK-DAG:     %idx3 =
// CHECK-DAG:     %true =
// CHECK-NEXT:    "op"
// CHECK-NEXT:    store %idx3, %arg1
// CHECK-NEXT:    store %idx0, %arg1
// CHECK-NEXT:    return %true, %idx1, %idx0
kgen.func @await_execute(%arg0: !kgen.pointer<i1> byref_error, %arg1: !kgen.pointer<index> byref_result) throws|async -> (i1, index, index) {
  %0 = co.execute {
    "op"() : () -> ()
    kgen.return
  }
  co.await %0 : (!co.routine) -> ()

  %1 = co.execute : i1 (%arg2: !kgen.pointer<i1> byref_error, %arg3: !kgen.pointer<index> byref_result) {
    %idx3 = index.constant 3
    pop.store %idx3, %arg3 : !kgen.pointer<index>
    %true = index.bool.constant true
    kgen.return %true : i1
  }
  %2 = co.await %1, %arg1, %arg0 : (!co.routine, !kgen.pointer<index>, !kgen.pointer<i1>) -> i1

  %3 = co.execute : index, index (%arg2: !kgen.pointer<index> byref_result) {
    %idx0 = index.constant 0
    %idx1 = index.constant 1
    pop.store %idx0, %arg2 : !kgen.pointer<index>
    %c2 = pop.cast_from_builtin %2 : i1 to !kgen.scalar<bool>
    hlcf.if %c2 {
      kgen.return %idx1, %idx0 : index, index
    } else {
      hlcf.yield
    }
    kgen.return %idx0, %idx1 : index, index
  }
  %4:2 = co.await %3, %arg1 : (!co.routine, !kgen.pointer<index>) -> (index, index)

  kgen.return %2, %4#0, %4#1 : i1, index, index
}

kgen.func @foo(%arg0: !kgen.pointer<i1> byref_error, %arg1: !kgen.pointer<index> byref_result) throws|async -> (i1, index, index) {
  %idx3 = index.constant 3
  %true = index.bool.constant true
  kgen.return %true, %idx3, %idx3 : i1, index, index
}

// CHECK-LABEL: kgen.func @await_invoke
kgen.func @await_invoke(%arg0: !kgen.pointer<i1> byref_error, %arg1: !kgen.pointer<index> byref_result) throws|async -> (i1, index, index) {
  %0 = co.invoke[(!kgen.pointer<i1> byref_error, !kgen.pointer<index> byref_result) throws|async -> (i1, index, index): @foo]()
  // CHECK: %[[#N:]]:3 = co.hot_invoke[(!kgen.pointer<i1> byref_error, !kgen.pointer<index> byref_result) throws|async -> (i1, index, index): @foo](%arg0, %arg1)
  // CHECK-NEXT: kgen.return %[[#N]]#0, %[[#N]]#1, %[[#N]]#2 : i1, index, index
  %1:3 = co.await %0, %arg1, %arg0 : (!co.routine, !kgen.pointer<index>, !kgen.pointer<i1>) -> (i1, index, index)
  kgen.return %1#0, %1#1, %1#2 : i1, index, index
}
