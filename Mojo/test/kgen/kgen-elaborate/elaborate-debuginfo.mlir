// RUN: kgen-opt %s -split-input-file -elaborate-generators="elaborate-debuginfo=true use-parametric-interpret=false" -mlir-print-debuginfo | FileCheck %s
// RUN: kgen-opt %s -split-input-file -elaborate-generators="elaborate-debuginfo=true use-parametric-interpret=true" -mlir-print-debuginfo | FileCheck %s

// CHECK-LABEL: kgen.func @loc_ref
kgen.generator @loc_ref() {
  kgen.param.if <false> {
    kgen.param.yield
  } else {
    kgen.param.declare A = <1>
    // CHECK: constant = <1> loc([[LOC1:#.*]])
    kgen.param.constant = <1> loc(fused<#kgen.param.decl.ref<"A">:index>["a":0:0])
    kgen.param.yield
  }
  kgen.param.if <false> {
    kgen.param.yield
  } else {
    kgen.param.declare A = <2>
    // CHECK: constant = <2> loc([[LOC2:#.*]])
    kgen.param.constant = <2> loc(fused<#kgen.param.decl.ref<"A">:index>["a":0:0])
    kgen.param.yield
  }
  kgen.return
}

// CHECK-DAG: [[LOC1]] = loc(fused<1 : index>
// CHECK-DAG: [[LOC2]] = loc(fused<2 : index>
