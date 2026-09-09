// REQUIRES: system-darwin
// RUN: kgen %s -emit=shared-lib -o %t
// RUN: llvm-objdump -t %t | FileCheck %s

// COM: Check that we generate the shared object file properly.
// CHECK: file format mach-o
// CHECK-DAG: g     F __TEXT,__text _exp_f32
// CHECK-NOT: l     F __TEXT,__text l_register_call_dtors.0

kgen.generator export @exp_f32(%arg: f32) -> f32 {
  kgen.return %arg : f32
}
