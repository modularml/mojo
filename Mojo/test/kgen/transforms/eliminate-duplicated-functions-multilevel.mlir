// RUN: kgen-opt %s -strip-debuginfo -eliminate-duplicate-functions -eliminate-dead-symbols -allow-unregistered-dialect | FileCheck %s

// This test exercises cascading deduplication across multiple levels of the
// call graph. Each level only becomes duplicated *after* the level below it is
// deduplicated and the callsites are redirected: `mid_*` look identical only
// once `leaf_1`'s callsite is redirected to `leaf_0`, and `upper_*` look
// identical only once `mid_1`'s callsite is redirected to `mid_0`.
//
// It also checks the determinism guarantee: regardless of the parallel
// processing order, every equivalence class collapses to its lexicographically
// smallest symbol (the `_0` variant), so the `_1` variants become dead and are
// stripped by `-eliminate-dead-symbols`.

// Level 0: identical leaves. Lex-min `leaf_0` survives.
// CHECK: @leaf_0
kgen.func @leaf_0(%1 : index, %2 : index) {
  %sum = index.add %1, %2
  "sink"(%sum) : (index) -> ()
  kgen.return
}

// CHECK-NOT: @leaf_1
kgen.func @leaf_1(%1 : index, %2 : index) {
  %sum = index.add %1, %2
  "sink"(%sum) : (index) -> ()
  kgen.return
}

// Level 1: only become duplicates after the leaf callsite is redirected.
// Lex-min `mid_0` survives and calls the surviving leaf `leaf_0`.
// CHECK-LABEL: @mid_0
// CHECK: kgen.call @leaf_0
kgen.func @mid_0() {
  %1 = kgen.param.constant = <0>
  %2 = kgen.param.constant = <1>
  kgen.call @leaf_0(%1, %2) : (index, index) -> ()
  kgen.return
}

// CHECK-NOT: @mid_1
kgen.func @mid_1() {
  %1 = kgen.param.constant = <0>
  %2 = kgen.param.constant = <1>
  kgen.call @leaf_1(%1, %2) : (index, index) -> ()
  kgen.return
}

// Level 2: only become duplicates after the mid callsite is redirected.
// Lex-min `upper_0` survives and calls the surviving mid `mid_0`.
// CHECK-LABEL: @upper_0
// CHECK: kgen.call @mid_0
kgen.func @upper_0() {
  kgen.call @mid_0() : () -> ()
  kgen.return
}

// CHECK-NOT: @upper_1
kgen.func @upper_1() {
  kgen.call @mid_1() : () -> ()
  kgen.return
}

// Exported roots are kept, but their callsites are redirected to the surviving
// lex-min symbol `upper_0`.
// CHECK-LABEL: @root_0
// CHECK-NEXT: kgen.call @upper_0
kgen.func export @root_0() {
  kgen.call @upper_0() : () -> ()
  kgen.return
}

// CHECK-LABEL: @root_1
// CHECK-NEXT: kgen.call @upper_0
kgen.func export @root_1() {
  kgen.call @upper_1() : () -> ()
  kgen.return
}
