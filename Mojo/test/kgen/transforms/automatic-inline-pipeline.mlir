// RUN: kgen-opt %s -automatic-inline='func-pipeline=canonicalizer' | FileCheck %s

// CHECK-LABEL: kgen.func @caller
kgen.func @caller() {
  // CHECK-NEXT: return
  kgen.call @callee() :()->()
  kgen.return
}

// CHECK-NOT: kgen.func @callee
kgen.func @callee() {
  %unused = kgen.param.constant = <1>
  kgen.return
}

// CHECK-LABEL: kgen.func @no_callers
kgen.func @no_callers() {
  // CHECK-NEXT: return
  %unused = kgen.param.constant = <1>
  kgen.return
}

// -----

kgen.func @wrap_source_loc_0() -> !kgen.string always_inline {
  %line, %col, %fileName = kgen.source_loc[0]
  kgen.return %fileName : !kgen.string
}

kgen.func @wrap_source_loc_1() -> !kgen.string always_inline {
  %line, %col, %fileName = kgen.source_loc[1]
  kgen.return %fileName : !kgen.string
}

kgen.func @test_wrap_source_loc_0() -> !kgen.string always_inline {
  %0 = kgen.call @wrap_source_loc_0() : () -> !kgen.string loc("some_file.mojo":4:6)
  kgen.return %0 : !kgen.string
}

kgen.func @call_wrapped_source_loc_1() -> !kgen.string always_inline {
  %0 = kgen.call @wrap_source_loc_1() : () -> !kgen.string
  kgen.return %0 : !kgen.string
}

// CHECK-LABEL: kgen.func @test_wrapped_source_loc_1
kgen.func @test_wrapped_source_loc_1() -> !kgen.string {
  // CHECK-DAG: kgen.param.constant: string = <"other_file.mojo">
  // CHECK-NOT: kgen.call
  %0 = kgen.call @call_wrapped_source_loc_1() : () -> !kgen.string loc("other_file.mojo":10:12)
  kgen.return %0 : !kgen.string
}

kgen.func @test_wrapped_source_loc_1_inlined() -> !kgen.string always_inline {
  %0 = kgen.call @call_wrapped_source_loc_1() : () -> !kgen.string loc("another_file.mojo":42:13)
  kgen.return %0 : !kgen.string
}

// CHECK-LABEL: kgen.func @test_source_loc
kgen.func @test_source_loc() -> (!kgen.string, !kgen.string) {
  // CHECK-DAG: kgen.param.constant: string = <"some_file.mojo">
  %0 = kgen.call @test_wrap_source_loc_0() : () -> !kgen.string

  // CHECK-DAG: kgen.param.constant: string = <"another_file.mojo">
  %1 = kgen.call @test_wrapped_source_loc_1_inlined() : () -> !kgen.string

  kgen.return %0, %1 : !kgen.string, !kgen.string
}
