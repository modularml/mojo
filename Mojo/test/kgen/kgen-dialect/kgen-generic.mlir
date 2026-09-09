// RUN: kgen-opt %s -mlir-print-op-generic | kgen-opt -mlir-print-op-generic | FileCheck %s

// CHECK: "kgen.generator"
// CHECK: funcTypeGenerator = !kgen.generator<() -> ()>
kgen.generator @kernel() {
  kgen.return
}
