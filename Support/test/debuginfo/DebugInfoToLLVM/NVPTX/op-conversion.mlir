// RUN: support-dialect-opt %s -convert-debuginfo-to-llvm -allow-unregistered-dialect -mlir-print-debuginfo -split-input-file | FileCheck %s

#file = #debuginfo.file<"foo.c" in "/mlir/">
#sp = #debuginfo.subprogram<
  file = #file,
  scope = #file,
  sourceName = <"foo">
> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#loc0 = loc(fused<#sp>["foo.mlir":0:0])
#loc1 = loc(fused<#sp>["foo.mlir":1:0])
#loc100at1 = loc(callsite("foo.mlir":100:0 at #loc1))

module attributes {M.target_info = #M.target<triple="nvptx64-nvidia-cuda", arch="nvptx64", features="", data_layout="", simd_bit_width=64>} {

// CHECK-LABEL: func @line_table_loc_lowering
func.func @line_table_loc_lowering() {
  // CHECK-NEXT: "test.op0"
  // CHECK-NEXT: llvm.inline_asm has_side_effects asm_dialect = att "pmevent.mask 0;", ""  : () -> () loc(#[[LOC1:.+]])
  // CHECK-NEXT: "test.op1"

  "test.op0"() : () -> i32 loc(#loc0)
  debuginfo.line_table_loc loc(#loc1)
  "test.op1"() : () -> i32 loc(#loc100at1)
} loc(#loc0)

// CHECK: #[[LOC1_RAW:.+]] = loc("foo.mlir":1:0)
// CHECK: #[[LOC1]] = loc(fused<{{.*}}>[#[[LOC1_RAW]]])

}
