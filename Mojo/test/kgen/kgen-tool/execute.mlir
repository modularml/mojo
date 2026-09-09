// RUN: kgen %s -execute -func="exec_exp:f32()" -func="void:()" | FileCheck %s

kgen.func export @exec_exp() -> f32 {
  %0 = kgen.param.constant: f32 = <2.71>
  kgen.return %0 : f32
}

kgen.func export @void() {
  kgen.return
}

// COM: exec_exp computes exp(1.0)
// CHECK: --- 'exec_exp' returned 2.7{{[0-9]+}}
// CHECK: --- 'void' finished
