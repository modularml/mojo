// RUN: kgen-opt %s -strip-debuginfo -eliminate-duplicate-functions -eliminate-dead-symbols -allow-unregistered-dialect | FileCheck %s

// CHECK: @leaf_repeative_{{[0|1]}}
kgen.func @leaf_repeative_0(%1 : index, %2 : index) {
  %sum = index.add %1, %2
  "sink"(%sum) : (index) -> ()
  kgen.return
}

// CHECK-NOT: @leaf_repeative_{{[0|1]}}
kgen.func @leaf_repeative_1(%1 : index, %2 : index) {
  %sum = index.add %1, %2
  "sink"(%sum) : (index) -> ()
  kgen.return
}

// CHECK: @call_to_repeative_[[CHAIN_ID:[0|1]]]
kgen.func @call_to_repeative_0() {
  %1 = kgen.param.constant = <0>
  %2 = kgen.param.constant = <1>
  kgen.call @leaf_repeative_0(%1, %2) : (index, index) -> ()
  kgen.return
}

// CHECK-NOT: @call_to_repeative_{{[0|1]}}
kgen.func @call_to_repeative_1() {
  %1 = kgen.param.constant = <0>
  %2 = kgen.param.constant = <1>
  kgen.call @leaf_repeative_1(%1, %2) : (index, index) -> ()
  kgen.return
}

// CHECK-LABEL: @exported_fn_repeative_call_0
kgen.func export @exported_fn_repeative_call_0() {
  // CHECK-NEXT: kgen.call @call_to_repeative_[[CHAIN_ID]]
  kgen.call @call_to_repeative_1() : () -> ()
  kgen.return
}

// Exported functions are kept, but the use of duplicated function are replaced.
//
// CHECK-LABEL: @exported_fn_repeative_call_1
kgen.func export @exported_fn_repeative_call_1() {
  // CHECK-NEXT: kgen.call @call_to_repeative_[[CHAIN_ID]]
  kgen.call @call_to_repeative_1() : () -> ()
  kgen.return
}

// CHECK-LABEL: @root
kgen.func export @root() {
  // CHECK-NEXT: kgen.call @exported_fn_repeative_call_[[EXP_ID:[0|1]]]
  kgen.call @exported_fn_repeative_call_0() : () -> ()
  // CHECK-NEXT: kgen.call @exported_fn_repeative_call_[[EXP_ID]]
  kgen.call @exported_fn_repeative_call_1() : () -> ()
  kgen.return
}
