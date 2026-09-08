// RUN: not kgen %s -execute -func="unhandled:i31()" 2>&1 >/dev/null | FileCheck -check-prefix=BADSIG %s

// BADSIG: unhandled signature: i31()
kgen.func export @unhandled() -> i31 {
  %0 = llvm.mlir.constant(1 : i31) : i31
  kgen.return %0 : i31
}
