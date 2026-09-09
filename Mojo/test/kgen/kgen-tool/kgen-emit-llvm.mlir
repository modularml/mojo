// RUN: kgen %s --emit=llvm | FileCheck %s

// Check that we generate the LLVM properly.
// CHECK: define dso_local float @exp_f32(float noundef %0)
// CHECK-NEXT: ret float %0

kgen.generator export @exp_f32(%arg: f32) -> f32 {
  kgen.return %arg : f32
}
