// RUN: kgen-opt %s -lower-calling-conventions -lower-to-llvm | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
// CHECK: llvm.func internal @return_none() attributes
kgen.func @return_none() -> !kgen.none {
  // CHECK: llvm.return
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}
}
