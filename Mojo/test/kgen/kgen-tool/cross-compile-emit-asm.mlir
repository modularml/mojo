// RUN: kgen -emit=asm -march skylake-avx512 %s | FileCheck %s

kgen.func export @return_zero() -> index {
  // CHECK: %eax
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}
