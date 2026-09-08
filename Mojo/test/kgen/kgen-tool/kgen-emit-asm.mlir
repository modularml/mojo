// RUN: kgen %s -emit=asm | FileCheck %s
// RUN: kgen %s -emit=asm-verbose | FileCheck %s --check-prefix=CHECK-VERBOSE

// Check that we generate some ASM properly.
// CHECK: exp_f32
// CHECK-VERBOSE: {{.*}}exp_f32                        {{.*}} -- Begin function {{.*}}exp_f32

kgen.generator export @exp_f32(%arg: f32) -> f32 {
  kgen.return %arg : f32
}
