// RUN: kgen-opt %s -function-stats 2>&1 | FileCheck %s

// CHECK:      13 | @bar_longer_name | {{.*}}function-stats.mlir
// CHECK-NEXT:  4 | @foo             | {{.*}}function-stats.mlir

kgen.func @foo() {
  index.constant 0
  index.constant 0
  kgen.return
}

kgen.func @bar_longer_name() {
  index.constant 0
  index.constant 0
  index.constant 0
  index.constant 0
  index.constant 0
  index.constant 0
  index.constant 0
  index.constant 0
  index.constant 0
  index.constant 0
  index.constant 0
  kgen.return
}
