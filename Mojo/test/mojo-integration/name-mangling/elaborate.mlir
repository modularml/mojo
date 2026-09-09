// RUN: kgen-opt %s -split-input-file -verify-parameters -elaborate-generators="use-parametric-interpret=false" \
// RUN:   -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=true" \
// RUN:   -allow-unregistered-dialect | FileCheck %s

kgen.generator @some_generator() attributes {sourceName = "foo"} {
  kgen.return
}

// CHECK-LABEL: kgen.func export @get_source_name
kgen.generator export @get_source_name() {
  // CHECK-NEXT: constant: string = <"foo">
  %0 = kgen.param.constant: string = <#kgen.get_source_name<#kgen.symbol.constant<@some_generator> : !kgen.generator<() -> ()>>>
  kgen.return
}
