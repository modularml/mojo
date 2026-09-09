// RUN: kgen-opt -split-input-file -pass-pipeline='builtin.module(lower-kgen-to-llvm,lower-control-flow,llvm.func(lower-pop-to-llvm))' %s -verify-diagnostics

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // expected-error @below {{cannot run on operations with CFG regions}}
  // expected-note @below {{try running it before lower-control-flow}}
  kgen.func @stack_allocation(%cond: !kgen.scalar<bool>) {
    hlcf.if %cond {
      %0 = pop.stack_allocation 4 x !kgen.simd<4, f32>
      hlcf.yield
    } else {
      hlcf.yield
    }
    kgen.return
  }
}
