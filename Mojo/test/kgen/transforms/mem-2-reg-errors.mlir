// RUN: kgen-opt -mem-2-reg -verify-diagnostics -split-input-file

// expected-error @below {{'ssa-formation' can only be run on operations with all single block regions}}
"someop"() ({
^bb0:
  "terminator"() : () -> ()

^bb1:
  "terminator"() : () -> ()
}) : () -> ()

// -----

// CHECK-LABEL: @read_uninitialized
kgen.generator @read_uninitialized() -> index {
  // CHECK-NEXT: %0 = pop.stack_allocation
  // expected-note @below {{memory allocated here}}
  %0 = pop.stack_allocation 1 x index
  // expected-warning @below {{load of uninitialized memory}}
  %1 = pop.load %0 : !kgen.pointer<index>
  kgen.return %1 : index
}
