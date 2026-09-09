// RUN: kgen-opt -lower-kgen-to-llvm -mlir-print-debuginfo %s | FileCheck %s

// CHECK-DAG: ![[COROUTINE:.*]] = !debuginfo.struct<"!co.routine"()>
// CHECK-DAG: ![[PTR:.*]] = !debuginfo.ptr<![[COROUTINE]] {sizeInBits = 64, alignInBits = 64}>

// CHECK-DAG: !debuginfo.subroutine<(![[PTR]]) -> (): DW_CC_normal>

!test = !debuginfo.subroutine<(
  !debuginfo.unresolved<!co.routine>
) -> (): DW_CC_normal>

#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !test

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="i64:64:64", simd_bit_width=128>} {
  kgen.func @foo() {
    kgen.return loc(fused<#subprogram>["foo.mlir":10:10])
  } loc(fused<#subprogram>["foo.mlir":10:10])
}
