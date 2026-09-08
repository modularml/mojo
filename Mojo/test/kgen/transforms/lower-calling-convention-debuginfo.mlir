// RUN: kgen-opt -lower-calling-conventions %s -mlir-print-debuginfo -split-input-file | FileCheck %s

#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (!kgen.none): DW_CC_normal>
#loc = loc(fused<#subprogram>["foo.mlir":0:0])

kgen.func @main() -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none> loc(#loc)
  kgen.return %none : !kgen.none loc(#loc)
} loc(#loc)

// CHECK: !debuginfo.subroutine<() -> ()
