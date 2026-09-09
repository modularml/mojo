// RUN: kgen-opt %s | FileCheck %s

#target = #kgen.target<triple="", arch="", features="", data_layout="i64:64:64", simd_bit_width=128> : !kgen.target

// CHECK-LABEL: @co_sizeof_alignof
kgen.generator @co_sizeof_alignof() {
  // CHECK-NEXT: <8>
  kgen.param.constant: index = <get_sizeof(!co.routine, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: index = <get_alignof(!co.routine, #target)>

  kgen.return
}
