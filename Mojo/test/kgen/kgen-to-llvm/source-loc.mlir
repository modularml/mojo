// RUN: kgen-opt -lower-kgen-to-llvm -split-input-file %s | FileCheck %s

// A `kgen.source_loc` that survives inlining with a non-negative count (not
// inlined enough times to reach the requested caller) must degrade to a
// best-effort location instead of failing legalization.

// Degrades to the outermost caller frame in the call-site location history.
// CHECK-LABEL: llvm.func internal @with_callsite
// CHECK-DAG: llvm.mlir.constant(42 : i64)
// CHECK-DAG: llvm.mlir.constant(7 : i64)
// CHECK: llvm.mlir.global internal constant @{{.*}}("caller.mojo\00")
module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  kgen.func @with_callsite() capturing -> !kgen.string {
    %line, %col, %fileName = kgen.source_loc[0] loc(callsite("inner.mojo":1:1 at "caller.mojo":42:7))
    kgen.return %fileName : !kgen.string
  }
}

// -----

// With no call-site history, degrades to the op's own location.
// CHECK-LABEL: llvm.func internal @plain_loc
// CHECK-DAG: llvm.mlir.constant(9 : i64)
// CHECK-DAG: llvm.mlir.constant(3 : i64)
// CHECK: llvm.mlir.global internal constant @{{.*}}("bare.mojo\00")
module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  kgen.func @plain_loc() capturing -> !kgen.string {
    %line, %col, %fileName = kgen.source_loc[0] loc("bare.mojo":9:3)
    kgen.return %fileName : !kgen.string
  }
}

// -----

// With no usable location at all, falls back to line 0 / "<unknown location>".
// CHECK-LABEL: llvm.func internal @unknown_loc
// CHECK: llvm.mlir.global internal constant @{{.*}}("<unknown location>\00")
module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  kgen.func @unknown_loc() capturing -> !kgen.string {
    %line, %col, %fileName = kgen.source_loc[0] loc(unknown)
    kgen.return %fileName : !kgen.string
  }
}
