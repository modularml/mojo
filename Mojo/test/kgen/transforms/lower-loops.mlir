// RUN: kgen-opt %s -pass-pipeline='builtin.module(kgen.func(lower-loops, canonicalize))' | FileCheck %s

// CHECK-LABEL: @induction_var_no_retvals_no_iterargs
kgen.func @induction_var_no_retvals_no_iterargs() {
  // CHECK:      [[IDX2:%.*]] = index.constant 2
  // CHECK-NEXT: [[IDX0:%.*]] = index.constant 0
  // CHECK-NEXT: [[IDX1:%.*]] = index.constant 1
  // CHECK-NEXT: hlcf.loop (%arg0 = [[IDX2]] : index) {
  // CHECK-NEXT:   [[V0:%.*]] = index.cmp sgt(%arg0, [[IDX0]])
  // CHECK-NEXT:   [[V0B:%.*]] = pop.cast_from_builtin [[V0]] : i1 to !kgen.scalar<bool>
  // CHECK-NEXT:   hlcf.if [[V0B]] {
  // CHECK-NEXT:     hlcf.yield
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     hlcf.break
  // CHECK-NEXT:   }
  // CHECK-NEXT:   [[V1:%.*]] = index.sub %arg0, [[IDX1]]
  // CHECK-NEXT:   kgen.call @foo([[V1]]) : (index) -> ()
  // CHECK-NEXT:   hlcf.continue [[V1]] : index
  // CHECK-NEXT: }

  %idx2 = index.constant 2
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  hlcf.for [%idx2 to %idx0 step %idx1 sgt sub] (%arg0 = %idx2 : index) {
    %0 = index.sub %arg0, %idx1
    kgen.call @foo(%0) : (index) -> ()
    hlcf.for.yield [induction_var (%0 : index)] [retvals ()] [iterargs ()]
  }
  kgen.return
}

// CHECK-LABEL: @nested_unroll_loops
kgen.func @nested_unroll_loops() {
  // CHECK:      [[IDX2:%.*]] = index.constant 2
  // CHECK-NEXT: [[IDX4:%.*]] = index.constant 4
  // CHECK-NEXT: [[IDX8:%.*]] = index.constant 8
  // CHECK-NEXT: [[IDX0:%.*]] = index.constant 0
  // CHECK-NEXT: [[IDX1:%.*]] = index.constant 1
  // CHECK-NEXT: hlcf.loop (%arg0 = [[IDX2]] : index) {
  // CHECK-NEXT:   [[V0:%.*]] = index.cmp sgt(%arg0, [[IDX0]])
  // CHECK-NEXT:   [[V0B:%.*]] = pop.cast_from_builtin [[V0]] : i1 to !kgen.scalar<bool>
  // CHECK-NEXT:   hlcf.if [[V0B]] {
  // CHECK-NEXT:     hlcf.yield
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     hlcf.break
  // CHECK-NEXT:   }
  // CHECK-NEXT:   [[V1:%.*]] = index.sub %arg0, [[IDX1]]
  // CHECK-NEXT:   kgen.call @foo([[V1]]) : (index) -> ()
  // CHECK-NEXT:   hlcf.loop (%arg1 = [[IDX4]] : index) {
  // CHECK-NEXT:     [[V3:%.*]] = index.cmp slt(%arg1, [[IDX8]])
  // CHECK-NEXT:     [[V3B:%.*]] = pop.cast_from_builtin [[V3]] : i1 to !kgen.scalar<bool>
  // CHECK-NEXT:     hlcf.if [[V3B]] {
  // CHECK-NEXT:       hlcf.yield
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       hlcf.break
  // CHECK-NEXT:     }
  // CHECK-NEXT:     [[V4:%.*]] = index.add %arg1, [[IDX2]]
  // CHECK-NEXT:     kgen.call @foo([[V4]]) : (index) -> ()
  // CHECK-NEXT:     hlcf.continue [[V4]] : index
  // CHECK-NEXT:   }
  // CHECK-NEXT:   hlcf.continue [[V1]] : index
  // CHECK-NEXT: }

  %idx2 = index.constant 2
  %idx4 = index.constant 4
  %idx8 = index.constant 8
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  hlcf.for [%idx2 to %idx0 step %idx1 sgt sub] (%arg0 = %idx2 : index) {
    %0 = index.sub %arg0, %idx1
    kgen.call @foo(%0) : (index) -> ()
    hlcf.for [%idx4 to %idx8 step %idx2 slt add] (%arg1 = %idx4 : index) {
      %3 = index.add %arg1, %idx2
      kgen.call @foo(%3) : (index) -> ()
      hlcf.for.yield [induction_var (%3 : index)] [retvals ()] [iterargs ()]
    }
    hlcf.for.yield [induction_var (%0 : index)] [retvals ()] [iterargs ()]
  }
  kgen.return
}

// CHECK-LABEL: @loop_carried_dependency
kgen.func @loop_carried_dependency() {
  // CHECK:      [[IDX1:%.*]] = index.constant 1
  // CHECK-NEXT: [[IDX9:%.*]] = index.constant 9
  // CHECK-NEXT: [[IDX2:%.*]] = index.constant 2
  // CHECK-NEXT: [[IDX4:%.*]] = index.constant 4
  // CHECK-NEXT: [[IDX8:%.*]] = index.constant 8
  // CHECK-NEXT: [[IDX0:%.*]] = index.constant 0
  // CHECK-NEXT: %0:2 = hlcf.loop (%arg0 = [[IDX1]] : index, %arg1 = [[IDX0]] : index, %arg2 = [[IDX0]] : index) -> (index, index) {
  // CHECK-NEXT:   [[V1:%.*]] = index.cmp slt(%arg0, [[IDX9]])
  // CHECK-NEXT:   [[V1B:%.*]] = pop.cast_from_builtin [[V1]] : i1 to !kgen.scalar<bool>
  // CHECK-NEXT:   hlcf.if [[V1B]] {
  // CHECK-NEXT:     hlcf.yield
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     hlcf.break %arg1, %arg2 : index, index
  // CHECK-NEXT:   }
  // CHECK-NEXT:   [[V2:%.*]] = index.add %arg0, [[IDX2]]
  // CHECK-NEXT:   kgen.call @foo([[V2]], %arg1) : (index, index) -> ()
  // CHECK-NEXT:   [[V3:%.*]] = hlcf.loop (%arg3 = [[IDX4]] : index, %arg4 = %arg2 : index) -> index {
  // CHECK-NEXT:     [[V4:%.*]] = index.cmp slt(%arg3, [[IDX8]])
  // CHECK-NEXT:     [[V4B:%.*]] = pop.cast_from_builtin [[V4]] : i1 to !kgen.scalar<bool>
  // CHECK-NEXT:     hlcf.if [[V4B]] {
  // CHECK-NEXT:       hlcf.yield
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       hlcf.break %arg4 : index
  // CHECK-NEXT:     }
  // CHECK-NEXT:     [[V5:%.*]] = index.add %arg3, [[IDX2]]
  // CHECK-NEXT:     kgen.call @foo([[V5]], %arg4) : (index, index) -> ()
  // CHECK-NEXT:     hlcf.continue [[V5]], [[V5]] : index, index
  // CHECK-NEXT:   }
  // CHECK-NEXT:   hlcf.continue [[V2]], [[V3]], [[V2]] : index, index, index
  // CHECK-NEXT: }

  %idx1 = index.constant 1
  %idx9 = index.constant 9
  %idx2 = index.constant 2
  %idx4 = index.constant 4
  %idx8 = index.constant 8
  %idx0 = index.constant 0
  %0:2 = hlcf.for [%idx1 to %idx9 step %idx2 slt add] (%arg2 = %idx1 : index, %arg0 = %idx0 : index, %arg1 = %idx0 : index) -> (index, index) {
    %3 = index.add %arg2, %idx2
    kgen.call @foo(%3, %arg0) : (index, index) -> ()
    %6 = hlcf.for [%idx4 to %idx8 step %idx2 slt add] (%arg4 = %idx4 : index, %arg3 = %arg1 : index) -> index {
      %7 = index.add %arg4, %idx2
      kgen.call @foo(%7, %arg3) : (index, index) -> ()
      hlcf.for.yield [induction_var (%7 : index)] [retvals (%7: index)] [iterargs ()]
    }
    hlcf.for.yield [induction_var (%3 : index)] [retvals (%6, %3: index, index)] [iterargs ()]
  }
  kgen.call @foo(%0#0) : (index) -> ()
  kgen.call @foo(%0#1) : (index) -> ()
  kgen.return
}

// CHECK-LABEL: @lower_unrolled_inclusive_cmp
 kgen.func @lower_unrolled_inclusive_cmp() -> index {
   // CHECK:      [[IDX2:%.*]] = index.constant 2
   // CHECK-NEXT: [[IDX0:%.*]] = index.constant 0
   // CHECK-NEXT: [[IDX1:%.*]] = index.constant 1
   // CHECK-NEXT: [[IDX3:%.*]] = index.constant 3
   // CHECK-NEXT: [[IDX4:%.*]] = index.constant 4
   // CHECK-NEXT: [[IDX5:%.*]] = index.constant 5
   // CHECK-NEXT: hlcf.loop (%arg0 = [[IDX0]] : index) {
   // CHECK-NEXT:   [[V0:%.*]] = index.cmp sle(%arg0, [[IDX3]])
   // CHECK-NEXT:   [[V0B:%.*]] = pop.cast_from_builtin [[V0]] : i1 to !kgen.scalar<bool>
   // CHECK-NEXT:   hlcf.if [[V0B]] {
   // CHECK-NEXT:     hlcf.yield
   // CHECK-NEXT:   } else {
   // CHECK-NEXT:     hlcf.break
   // CHECK-NEXT:   }
   // CHECK-NEXT:   [[V1:%.*]] = index.add %arg0, %idx1
   // CHECK-NEXT:   kgen.call @foo([[V1]]) : (index) -> ()
   // CHECK-NEXT:   [[V2:%.*]] = index.add %arg0, [[IDX2]]
   // CHECK-NEXT:   kgen.call @foo([[V2]]) : (index) -> ()
   // CHECK-NEXT:   [[V3:%.*]] = index.add %arg0, [[IDX3]]
   // CHECK-NEXT:   kgen.call @foo([[V3]]) : (index) -> ()
   // CHECK-NEXT:   hlcf.continue [[V3]] : index
   // CHECK-NEXT: }
   // CHECK-NEXT: kgen.call @foo([[IDX4]]) : (index) -> ()
   // CHECK-NEXT: kgen.call @foo([[IDX5]]) : (index) -> ()
   // CHECK-NEXT: kgen.return [[IDX5]] : index

   %idx0 = index.constant 0
   %idx1 = index.constant 1
   %idx3 = index.constant 3
   %idx4 = index.constant 4
   %idx5 = index.constant 5
   %0 = hlcf.for [%idx0 to %idx3 step %idx3 sle add]  (%arg0 = %idx0 : index, %arg1 = %idx0 : index) -> index {
     %1 = index.add %arg0, %idx1
     kgen.call @foo(%1) : (index) -> ()
     %2 = index.add %1, %idx1
     kgen.call @foo(%2) : (index) -> ()
     %3 = index.add %2, %idx1
     kgen.call @foo(%3) : (index) -> ()
     hlcf.for.yield [induction_var (%3 : index)] [retvals (%3 : index)] [iterargs ()]
   }
   kgen.call @foo(%idx4) : (index) -> ()
   kgen.call @foo(%idx5) : (index) -> ()
   kgen.return %idx5 : index
 }

// hlcf.for with !kgen.scalar<si64> bounds: casts to index before index.cmp.
// CHECK-LABEL: @pop_scalar_si64_bounds
kgen.func @pop_scalar_si64_bounds(%lb: !kgen.scalar<si64>, %ub: !kgen.scalar<si64>,
                                   %step: !kgen.scalar<si64>) {
  // CHECK:      hlcf.loop ([[ARG:%.*]] = %arg0 : !kgen.scalar<si64>) {
  // CHECK-NEXT:   [[C0:%.*]] = pop.cast [[ARG]] : !kgen.scalar<si64> to !kgen.scalar<index>
  // CHECK-NEXT:   [[I0:%.*]] = pop.cast_to_builtin [[C0]] : !kgen.scalar<index> to index
  // CHECK-NEXT:   [[C1:%.*]] = pop.cast %arg1 : !kgen.scalar<si64> to !kgen.scalar<index>
  // CHECK-NEXT:   [[I1:%.*]] = pop.cast_to_builtin [[C1]] : !kgen.scalar<index> to index
  // CHECK-NEXT:   [[CMP:%.*]] = index.cmp slt([[I0]], [[I1]])
  // CHECK-NEXT:   [[CMPB:%.*]] = pop.cast_from_builtin [[CMP]] : i1 to !kgen.scalar<bool>
  // CHECK-NEXT:   hlcf.if [[CMPB]] {
  // CHECK-NEXT:     hlcf.yield
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     hlcf.break
  // CHECK-NEXT:   }
  // CHECK-NEXT:   [[NEXT:%.*]] = pop.add [[ARG]], %arg2 : !kgen.scalar<si64>
  // CHECK-NEXT:   hlcf.continue [[NEXT]] : !kgen.scalar<si64>
  // CHECK-NEXT: }
  hlcf.for [%lb to %ub step %step : !kgen.scalar<si64> slt add] (%arg0 = %lb : !kgen.scalar<si64>) {
    %next = pop.add %arg0, %step : !kgen.scalar<si64>
    hlcf.for.yield [induction_var (%next : !kgen.scalar<si64>)] [retvals ()] [iterargs ()]
  }
  kgen.return
}
