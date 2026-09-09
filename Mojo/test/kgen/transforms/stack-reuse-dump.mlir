// RUN: kgen-opt %s -stack-reuse -kgen-debug-only=stack-reuse --mlir-disable-threading 2>&1 | FileCheck %s

// REQUIRES: ASSERTIONS

// CHECK: WARNING: `kgen-debug-only` may work incorrectly with multithreading enabled
// CHECK: stack-reuse: %0 = pop.stack_allocation 1 x index doesn't escape
// CHECK: stack-reuse: %1 = pop.stack_allocation 1 x index doesn't escape
// CHECK: stack-reuse: %2 = pop.stack_allocation 1 x index doesn't escape

kgen.func @two_overlapping(%arg0: index, %arg1: index) -> (index, index) {
  %s0 = pop.stack_allocation 1 x index
  %s1 = pop.stack_allocation 1 x index
  pop.store %arg0, %s0 : !kgen.pointer<index>
  pop.store %arg1, %s1 : !kgen.pointer<index>

  %s2 = pop.stack_allocation 1 x index
  pop.store %arg0, %s2 : !kgen.pointer<index>
  %v0 = pop.load %s2 : !kgen.pointer<index>
  pop.store %arg1, %s2 : !kgen.pointer<index>
  %v1 = pop.load %s2 : !kgen.pointer<index>
  kgen.return %v0, %v1 : index, index
}

