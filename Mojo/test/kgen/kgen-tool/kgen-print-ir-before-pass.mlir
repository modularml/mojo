// RUN: kgen --emit=llvm --mlir-print-ir-before=cse %s -o /dev/null 2>&1 | FileCheck %s
// CHECK: IR Dump Before CSEPass: cse
kgen.generator export @exp_f32(%arg: f32) -> f32 {
  kgen.return %arg : f32
}
