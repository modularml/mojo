// RUN: kgen-opt -automatic-inline %s | FileCheck %s

// Check that the AutomaticInline pass doesn't consider an inlined function
// dead when it has remaining non-call users. Regression test for MOCO-3737.

// CHECK-LABEL: kgen.func @keep_function_pointer
// CHECK: kgen.param.constant: (index) -> index = <@callee>
kgen.func @keep_function_pointer() -> !kgen.generator<(index) -> index> {
  %0 = kgen.param.constant: (index) -> index = <@callee>
  kgen.return %0 : !kgen.generator<(index) -> index>
}

// CHECK-LABEL: kgen.func @inline_me
// CHECK-NOT: kgen.call @callee
kgen.func @inline_me(%arg0: index) -> index {
  // @callee is small — AutomaticInline inlines this call. After inlining, the
  // call-graph has no remaining callers of @callee, but the
  // `kgen.param.constant` in @keep_function_pointer still references it.
  %0 = kgen.call @callee(%arg0) : (index) -> index
  kgen.return %0 : index
}

// @callee must still exist after the pass because
// @keep_function_pointer references it as a function pointer.
// CHECK-LABEL: kgen.func @callee
kgen.func @callee(%arg0: index) -> index {
  %0 = index.add %arg0, %arg0
  kgen.return %0 : index
}
