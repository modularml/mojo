// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=false" -o /dev/null | FileCheck %s
// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=true" -o /dev/null | FileCheck %s

// COM: pop.external call for 'write'

// CHECK: A
kgen.generator @print(%arg0: index, %arg1: index) -> index {
  %index1 = kgen.param.constant = <1>
  %index2 = kgen.param.constant = <2>
  %3 = pop.stack_allocation 2 x index
  pop.store %arg0, %3 : !kgen.pointer<index>
  %0 = pop.external_call @write(%index1, %3, %index1) : (index, !kgen.pointer<index>, index) -> (index)

  %4 = pop.stack_allocation 1 x index
  pop.store %arg1, %4 : !kgen.pointer<index>
  %1 = pop.external_call @write(%index1, %4, %index1) : (index, !kgen.pointer<index>, index) -> (index)

  %5 = index.add %0, %1
  kgen.return %5 : index
}

kgen.generator export @main() {
  kgen.param.apply x = [(index, index) -> index: @print](65, 10)
  %0 = kgen.param.constant: index = <x>
  kgen.return
}
