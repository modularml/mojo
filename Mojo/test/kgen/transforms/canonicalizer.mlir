// RUN: kgen-opt -canonicalizer -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @if_to_select
kgen.func @if_to_select(%arg0: !kgen.scalar<bool>, %arg1: f32, %arg2: f32) -> f32 {
  // CHECK-NEXT: pop.select %arg0, %arg1, %arg2 : f32
  %0 = hlcf.if %arg0 -> f32 {
    hlcf.yield %arg1 : f32
  } else {
    hlcf.yield %arg2 : f32
  }
  kgen.return %0 : f32
}


// CHECK-LABEL: @if_to_select
kgen.func @if_to_select_dead_if(%arg0: !kgen.scalar<bool>) {
  // CHECK-NEXT: return
  hlcf.if %arg0 {
    hlcf.yield
  } else {
    hlcf.yield
  }
  kgen.return
}

// CHECK-LABEL: @if_to_select_multiple
kgen.func @if_to_select_multiple(%arg0: !kgen.scalar<bool>, %arg1: f32, %arg2: i32,
                                 %arg3: f32, %arg4: i32) -> (f32, i32) {
  // CHECK-NEXT: pop.select %arg0, %arg1, %arg3 : f32
  // CHECK-NEXT: pop.select %arg0, %arg2, %arg4 : i32
  %0:2 = hlcf.if %arg0 -> f32, i32 {
    hlcf.yield %arg1, %arg2 : f32, i32
  } else {
    hlcf.yield %arg3, %arg4 : f32, i32
  }
  kgen.return %0#0, %0#1 : f32, i32
}

// CHECK-LABEL: @fold_if_single_add_then
kgen.func @fold_if_single_add_then(%cond: !kgen.scalar<bool>, %arg0 : index, %arg1: index) -> index {
  // CHECK-NOT: hlcf.if
  // CHECK:      %[[ADD:.*]] = index.add
  // CHECK-NEXT: %[[IF_RES:.*]] = pop.select {{.*}}, %[[ADD]], {{.*}}
  // CHECK-NEXT: kgen.return %[[IF_RES]]
  %if_res = hlcf.if %cond -> index {
    %add = index.add %arg1, %arg0
    hlcf.yield %add: index
  } else {
    hlcf.yield %arg1: index
  }
  kgen.return %if_res: index
}

// CHECK-LABEL: @negative_fold_if_single_div_then
kgen.func @negative_fold_if_single_div_then(%cond: !kgen.scalar<bool>, %arg0 : index, %arg1: index) -> index {
  // CHECK: %0 = hlcf.if %arg0 -> index {
  // CHECK:   %1 = index.divs %arg2, %arg1
  // CHECK:   hlcf.yield %1 : index
  // CHECK: } else {
  // CHECK:   hlcf.yield %arg2 : index
  // CHECK: }
  %if_res = hlcf.if %cond -> index {
    %add = index.divs %arg1, %arg0
    hlcf.yield %add: index
  } else {
    hlcf.yield %arg1: index
  }
  kgen.return %if_res: index
}

// CHECK-LABEL: @negative_fold_if_single_div_then_pop_type
kgen.func @negative_fold_if_single_div_then_pop_type(%cond: !kgen.scalar<bool>, %arg0 : !kgen.scalar<index>, %arg1: !kgen.scalar<index>) -> !kgen.scalar<index> {
  // CHECK: %0 = hlcf.if %arg0 -> !kgen.scalar<index> {
  // CHECK:   %1 = pop.div %arg2, %arg1 : !kgen.scalar<index>
  // CHECK:   hlcf.yield %1 : !kgen.scalar<index>
  // CHECK: } else {
  // CHECK:   hlcf.yield %arg2 : !kgen.scalar<index>
  // CHECK: }
  %if_res = hlcf.if %cond -> !kgen.scalar<index> {
    %add = pop.div %arg1, %arg0 : !kgen.scalar<index>
    hlcf.yield %add: !kgen.scalar<index>
  } else {
    hlcf.yield %arg1: !kgen.scalar<index>
  }
  kgen.return %if_res: !kgen.scalar<index>
}

// CHECK-LABEL: @fold_if_single_add_else
kgen.func @fold_if_single_add_else(%cond: !kgen.scalar<bool>, %arg0 : index, %arg1: index) -> index {
  // CHECK-NOT: hlcf.if
  // CHECK:      %[[ADD:.*]] = index.add
  // CHECK-NEXT: %[[IF_RES:.*]] = pop.select {{.*}}, {{.*}}, %[[ADD]]
  // CHECK-NEXT: kgen.return %[[IF_RES]]
  %if_res = hlcf.if %cond -> index {
    hlcf.yield %arg0: index
  } else {
    %add = index.add %arg1, %arg0
    hlcf.yield %add: index
  }
  kgen.return %if_res: index
}

// CHECK-LABEL: @fold_if_single_add_then_sub_else
kgen.func @fold_if_single_add_then_sub_else(%cond: !kgen.scalar<bool>, %arg0 : index, %arg1: index) -> index {
  // CHECK-NOT: hlcf.if
  // CHECK:      %[[ADD:.*]] = index.add
  // CHECK-NEXT: %[[SUB:.*]] = index.sub
  // CHECK-NEXT: %[[IF_RES:.*]] = pop.select {{.*}}, %[[ADD]], %[[SUB]]
  // CHECK-NEXT: kgen.return %[[IF_RES]]
  %if_res = hlcf.if %cond -> index {
    %add = index.add %arg1, %arg0
    hlcf.yield %add: index
  } else {
    %sub = index.sub %arg1, %arg0
    hlcf.yield %sub: index
  }
  kgen.return %if_res: index
}

// CHECK-LABEL: @indexify_comparison
kgen.func @indexify_comparison(%arg0: index) -> i1 {
  // CHECK: %0 = index.cmp sgt(%arg0, %idx1)
  %simd = kgen.param.constant: scalar<index> = <1>
  %0 = pop.cast_from_builtin %arg0 : index to !kgen.scalar<index>
  %1 = pop.cmp gt(%0, %simd) : !kgen.scalar<index>
  %2 = pop.cast_to_builtin %1 : !kgen.scalar<bool> to i1
  // CHECK: return %0
  kgen.return %2 : i1
}

// CHECK-LABEL: @canonicalize_loop_range
// FIXME(scalar<bool> migration): the loop-range fold below
// (`index.cmp sgt(pop.select(index.cmp slt(...), b-a, 0), 0)` ->
// `index.cmp slt(...)`) no longer fires because the `pop.select`
// condition is now a `pop.cast_from_builtin` of the `index.cmp`
// rather than the `index.cmp` directly. The canonicalization pattern
// needs to look through the cast. Until then this just checks the
// (unfolded) result is well-formed.
kgen.func @canonicalize_loop_range(%arg0: index, %arg1: index, %arg2: index) -> (i1, i1, i1, i1) {
  // CHECK: index.cmp slt(%arg0, %arg1)
  // CHECK: pop.select
  // CHECK: index.cmp sgt(%arg1, %arg0)
  // CHECK: index.cmp sgt(%arg2, %{{.*}})
  // CHECK: index.cmp slt(%{{.*}}, %arg2)
  // CHECK: return
  %0 = index.cmp slt(%arg0, %arg1)
  %c0 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
  %idx0 = index.constant 0
  %1 = index.sub %arg1, %arg0
  %2 = pop.select %c0, %1, %idx0 : index
  %3 = index.cmp sgt(%2, %idx0)

  %4 = index.cmp sgt(%arg1, %arg0)
  %c4 = pop.cast_from_builtin %4 : i1 to !kgen.scalar<bool>
  %5 = index.sub %arg1, %arg0
  %6 = pop.select %c4, %5, %idx0 : index
  %7 = index.cmp sgt(%6, %idx0)

  %8 = index.cmp sgt(%arg2, %idx0)
  %c8 = pop.cast_from_builtin %8 : i1 to !kgen.scalar<bool>
  %9 = pop.select %c8, %arg2, %idx0 : index
  %10 = index.cmp sgt(%9, %idx0)

  %11 = index.cmp slt(%idx0, %arg2)
  %c11 = pop.cast_from_builtin %11 : i1 to !kgen.scalar<bool>
  %12 = pop.select %c11, %arg2, %idx0 : index
  %13 = index.cmp sgt(%12, %idx0)

  kgen.return %3, %7, %10, %13 : i1, i1, i1, i1
}

// CHECK-LABEL: @condition_propagation
kgen.func @condition_propagation(%cond: !kgen.scalar<bool>) {
  // CHECK: %[[FALSE:.*]] = kgen.param.constant: scalar<bool> = <false>
  // CHECK: %[[TRUE:.*]] = kgen.param.constant: scalar<bool> = <true>
  // CHECK: hlcf.if
  hlcf.if %cond {
    // CHECK-NEXT: "use"(%[[TRUE]])
    "use"(%cond) : (!kgen.scalar<bool>) -> ()
    hlcf.yield
  // CHECK: else
  } else {
    // CHECK-NEXT: "use"(%[[FALSE]])
    "use"(%cond) : (!kgen.scalar<bool>) -> ()
    hlcf.yield
  }
  kgen.return
}

// CHECK-LABEL: @not_cmp
kgen.func @not_cmp(%arg0: index, %arg1: index) -> i1 {
  %simd_0 = kgen.param.constant: scalar<bool> = <true>
  // CHECK-NEXT: %0 = index.cmp ugt(%arg0, %arg1)
  %0 = index.cmp ule(%arg0, %arg1)
  %1 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
  %2 = pop.simd.xor %1, %simd_0 : <1, bool>
  %3 = pop.cast_to_builtin %2 : !kgen.scalar<bool> to i1
  // CHECK-NEXT: return %0
  kgen.return %3 : i1
}

// CHECK-LABEL: kgen.func @select_variant_is_0
kgen.func @select_variant_is_0(%var: !kgen.variant<index, i64>) -> (index, i64) {
  %0 = kgen.param.constant = <#interp.uninitmem : index>
  %1 = kgen.param.constant: i64 = <#interp.uninitmem : i64>
  %2 = kgen.variant.is %var, 0 : !kgen.variant<index, i64>
  // CHECK-NEXT: [[A:%.*]] = kgen.variant.get %arg0, 0
  // CHECK-NEXT: [[B:%.*]] = kgen.variant.get %arg0, 1
  %3 = kgen.variant.get %var, 0 : !kgen.variant<index, i64>
  %4 = kgen.variant.get %var, 1 : !kgen.variant<index, i64>
  %5 = pop.select %2, %3, %0 : index
  %6 = pop.select %2, %1, %4 : i64
  // CHECK-NEXT: return [[A]], [[B]]
  kgen.return %5, %6 : index, i64
}

// CHECK-LABEL: kgen.func @select_variant_is_1
kgen.func @select_variant_is_1(%arg0: index, %c: !kgen.scalar<bool>) -> (index, index) {
  // CHECK-NEXT: return %arg0, %arg0
  %0 = kgen.param.constant = <#interp.uninitmem : index>
  %1 = pop.select %c, %0, %arg0 : index
  %2 = pop.select %c, %arg0, %0 : index
  kgen.return %1, %2 : index, index
}

// CHECK-LABEL: kgen.func @if_hoist_yield
kgen.func @if_hoist_yield(%arg0: !kgen.scalar<bool>, %arg1: index) -> (index, index) {
  %idx0 = index.constant 0
  // CHECK: hlcf.if %arg0 {
  %0 = hlcf.if %arg0 -> index {
    hlcf.yield %idx0 : index
  } else {
    kgen.unreachable
  }
  // CHECK: [[SELECT:%.*]] = pop.select %arg0, %idx0, %arg1
  // CHECK: hlcf.if %arg0 {
  %2:2 = hlcf.if %arg0 -> index, index {
    %1 = "something"() : () -> index
    hlcf.yield %1, %idx0 : index, index
  } else {
    hlcf.yield %idx0, %arg1 : index, index
  }
  // CHECK: return %idx0, [[SELECT]]
  kgen.return %0, %2#1 : index, index
}
