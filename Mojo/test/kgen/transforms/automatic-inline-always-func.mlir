// RUN: kgen-opt -automatic-inline %s | FileCheck %s
// RUN: kgen-opt -automatic-inline=func-pipeline='canonicalize,cse' %s | FileCheck %s --check-prefix=CANON
// RUN: not kgen-opt --asyncrt-single-thread --mlir-disable-threading -pass-pipeline='builtin.module(automatic-inline{func-pipeline='canonicalize,cse'}, test-always-fail)' %s --mlir-pass-pipeline-crash-reproducer=- | FileCheck %s --check-prefix=REPRO

// CHECK-LABEL: kgen.func @top
// CANON-LABEL: kgen.func @top
kgen.func @top() -> index {
  %idx0 = index.constant 0
  // CHECK: index.add %idx0, %idx0
  // CANON-NOT: index.add
  %0 = kgen.call @bar(%idx0) : (index) -> index
  // CHECK-NEXT: return %0
  // CANON: return %idx0
  kgen.return %0 : index
}

kgen.func @bar(%arg0: index) -> index always_inline {
  %0 = index.add %arg0, %arg0
  kgen.return %0 : index
}

// CANON-LABEL: kgen.func @no_callers
kgen.func @no_callers() {
  // CANON-NEXT: return
  %unused = kgen.param.constant = <1>
  kgen.return
}

// REPRO: pipeline: "builtin.module(automatic-inline{func-pipeline=canonicalize,cse optimization-level=1 update-debug-info=deferred}, test-always-fail)"
