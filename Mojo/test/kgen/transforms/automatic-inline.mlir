// RUN: kgen-opt -automatic-inline %s | FileCheck %s

// CHECK-LABEL: kgen.func @top
kgen.func @top() -> index {
  // CHECK:      [[IDX0:%.*]] = index.constant 0
  // CHECK-NEXT: [[V0:%.*]] = index.add [[IDX0]], [[IDX0]]
  // CHECK-NEXT: [[V1:%.*]] = index.add [[V0]], [[V0]]
  // CHECK-NEXT: [[V2:%.*]] = kgen.call @foo_over_threshold([[V1]]) : (index) -> index
  // CHECK-NEXT: debuginfo.value
  // CHECK-NEXT: debuginfo.kill
  // CHECK-NOT: kgen.call @foo_debug_over_threshold
  // CHECK: kgen.return [[V2]] : index
  // CHECK-NOT:  kgen.call @bar
  // CHECK-NOT:  kgen.call @foo
  %idx0 = index.constant 0

  // Inline @bar.
  %0 = kgen.call @bar(%idx0) : (index) -> index
  kgen.return %0 : index
}

// CHECK-NOT: kgen.func @bar(%arg0: index) -> index {
kgen.func @bar(%arg0: index) -> index {
  %0 = index.add %arg0, %arg0
  // Inline @foo.
  %1 = kgen.call @foo(%0) : (index) -> index

  // Don't inline @foo_over_threshold.
  // Threshold value is defined by inline heuristics, which is 10 (operations * #calls) for O0 now.
  %2 = kgen.call @foo_over_threshold(%1) : (index) -> index

  // Inline @foo_debug_over_threshold.
  %3 = kgen.call @foo_debug_over_threshold(%1) : (index) -> index
  kgen.return %2 : index
}

// CHECK-NOT: kgen.func @foo(%arg0: index) -> index {
kgen.func @foo(%arg0: index) -> index {
  %0 = index.add %arg0, %arg0
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @foo_over_threshold(%arg0: index) -> index {
kgen.func @foo_over_threshold(%arg0: index) -> index {
  // This function will not be inlined, because
  // it exceeds current inline heuristics which is 10 (operations * #calls) for O0.
  %0 = index.add %arg0, %arg0
  %1 = index.add %arg0, %0
  %2 = index.add %arg0, %1
  %3 = index.add %arg0, %2
  %4 = index.add %arg0, %3
  %5 = index.add %arg0, %4
  %6 = index.add %arg0, %5
  %7 = index.add %arg0, %6
  %8 = index.add %arg0, %7
  %9 = index.add %arg0, %8
  %10 = index.add %arg0, %9
  kgen.return %10 : index
}

#subprogram = #debuginfo.subprogram<sourceName = <"foo_debug_over_threshold">> : !debuginfo.subroutine<(index) -> (index): DW_CC_normal>
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "arg0"> : !debuginfo.unresolved<index>

// CHECK-NOT: kgen.func @foo_debug_over_threshold(%arg0: index) -> index {
kgen.func @foo_debug_over_threshold(%arg0: index) -> index {
  // This function _will_ be inlined, because even though it exceeds current
  // inline heuristics which is 10 (operations * #calls) for O0, it is only
  // due to debug ops.
  debuginfo.value #local_variable = %arg0 : index
  debuginfo.kill #local_variable
  debuginfo.value #local_variable = %arg0 : index
  debuginfo.kill #local_variable
  debuginfo.value #local_variable = %arg0 : index
  debuginfo.kill #local_variable
  debuginfo.value #local_variable = %arg0 : index
  debuginfo.kill #local_variable
  debuginfo.value #local_variable = %arg0 : index
  debuginfo.kill #local_variable
  kgen.return %arg0 : index
}

// CHECK-LABEL: kgen.func @not_inline_bar
kgen.func @not_inline_bar() -> index {
  // CHECK:      [[IDX0:%.*]] = index.constant 0
  // CHECK-NEXT: [[V0:%.*]] = kgen.call @bar_no_inline([[IDX0]]) : (index) -> index
  // CHECK-NEXT: kgen.return [[V0]] : index
  %idx0 = index.constant 0
  %0 = kgen.call @bar_no_inline(%idx0) : (index) -> index
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @bar_no_inline(%arg0: index) -> index
kgen.func @bar_no_inline(%arg0: index) -> index no_inline {
  %0 = index.add %arg0, %arg0
  kgen.return %0 : index
}

// Not inlining any functions in the same SCC
// CHECK-LABEL: kgen.func @scc0_f(%arg0: index) -> index
kgen.func @scc0_f(%arg0: index) -> index {
  // CHECK: [[V0:%.*]] = kgen.call @scc0_g(%arg0) : (index) -> index
  // CHECK: [[V1:%.*]] = kgen.call @scc0_f([[V0]]) : (index) -> index
  // CHECK-NEXT: kgen.return [[V1]] : index
  %0 = kgen.call @scc0_g(%arg0) : (index) -> index
  %1 = kgen.call @scc0_f(%0) : (index) -> index
  kgen.return %1: index
}

// CHECK-LABEL: kgen.func @scc0_g(%arg0: index) -> index
kgen.func @scc0_g(%arg0: index) -> index {
  // CHECK: [[V0:%.*]] = kgen.call @scc0_h(%arg0) : (index) -> index
  // CHECK-NEXT: kgen.return [[V0]] : index
  %0 = kgen.call @scc0_h(%arg0) : (index) -> index
  kgen.return %0: index
}

// CHECK-LABEL: kgen.func @scc0_h(%arg0: index) -> index
kgen.func @scc0_h(%arg0: index) -> index {
  // CHECK: [[V0:%.*]] = kgen.call @scc0_f(%arg0) : (index) -> index
  // CHECK-NEXT: kgen.return [[V0]] : index
  %0 = kgen.call @scc0_f(%arg0) : (index) -> index
  kgen.return %0: index
}

// Inlining function from scc0 to scc1
// CHECK-LABEL: kgen.func @scc1_f(%arg0: index) -> index
kgen.func @scc1_f(%arg0: index) -> index {
  // CHECK: [[V0:%.*]] = kgen.call @scc0_h(%arg0) : (index) -> index
  // CHECK: [[V1:%.*]] = kgen.call @scc0_f([[V0]]) : (index) -> index
  // CHECK-NEXT: kgen.return [[V1]] : index
  %0 = kgen.call @scc0_g(%arg0) : (index) -> index
  %1 = kgen.call @scc0_h(%0) : (index) -> index
  kgen.return %1: index
}

// COM: Not inlining any functions in the same SCC
// COM: All functions in this SCC is not being called by any other functions
// COM: outside of the scc. Pass done should not have a race condition
// COM: and wait for the work item for both of these two functions to be done
// COM: although they are not in the chain starting from the externalNode.

// CHECK-NOT: kgen.func @scc2_f(%arg0: index) -> index
kgen.func @scc2_f(%arg0: index) -> index {
  %0 = kgen.call @scc2_g(%arg0) : (index) -> index
  %1 = kgen.call @scc2_f(%0) : (index) -> index
  kgen.return %1: index
}

// CHECK-NOT: kgen.func @scc2_g(%arg0: index) -> index
kgen.func @scc2_g(%arg0: index) -> index {
  %0 = kgen.call @scc2_f(%arg0) : (index) -> index
  kgen.return %0: index
}

// COM: All functions in this SCC is not being called by any other functions
// COM: outside of the scc. However, one of the function in the scc is exported.
// COM: So neither of them should be erased.
// CHECK: kgen.func export @scc3_f
kgen.func export @scc3_f(%arg0: index) -> index {
  %0 = kgen.call @scc3_g(%arg0) : (index) -> index
  %1 = kgen.call @scc3_f(%0) : (index) -> index
  kgen.return %1: index
}

// CHECK: kgen.func @scc3_g
kgen.func @scc3_g(%arg0: index) -> index {
  %0 = kgen.call @scc3_f(%arg0) : (index) -> index
  kgen.return %0: index
}
