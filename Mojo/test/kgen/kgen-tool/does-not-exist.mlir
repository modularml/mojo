// RUN: not kgen %s -execute -func="does_not_exist:f32()" 2>&1 >/dev/null | FileCheck -check-prefix=BADKERN %s

// BADKERN: could not find func '@does_not_exist'
kgen.func export @filler() -> f32 {
  %0 = llvm.mlir.constant(1.000000e+00 : f32) : f32
  kgen.return %0 : f32
}
