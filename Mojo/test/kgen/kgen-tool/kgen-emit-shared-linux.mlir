// REQUIRES: system-linux
// RUN: kgen %s -emit=shared-lib -o %t
// RUN: llvm-objdump -t %t | FileCheck %s

// COM: Check that we generate the shared object file properly.
// CHECK: dynamic

kgen.generator export @exp_f32(%arg: f32) -> f32 {
  kgen.return %arg : f32
}
