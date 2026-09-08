// RUN: kgen-opt -lower-kgen-to-llvm %s | FileCheck %s

// CHECK: llvm.data_layout = "p:32:32", llvm.target_triple = "arm64-apple-darwin21.6.0"
module attributes {
  M.target_info = #M.target<triple="arm64-apple-darwin21.6.0", arch="", features="",
                            data_layout="p:32:32", simd_bit_width=128>} {
}
