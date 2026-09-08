// RUN: kgen-opt -cse -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @dont_cse_none
kgen.func @dont_cse_none(%arg0: i1) -> !kgen.none {
  // CHECK: [[NONE:%.*]] = kgen.param.constant: none = <#kgen.none>
  %none = kgen.param.constant: none = <#kgen.none>

  // CHECK-NEXT: kgen.param.declare.region
  // CHECK-NEXT: [[NONE_AGAIN:%.*]] = kgen.param.constant: none = <#kgen.none>
  // CHECK-NEXT: kgen.return [[NONE_AGAIN]]
  kgen.param.declare.region Fn = () capturing -> !kgen.none {
    %none_again = kgen.param.constant: none = <#kgen.none>
    kgen.return %none_again : !kgen.none
  }

  // CHECK: kgen.return [[NONE]] : !kgen.none
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: @cse_simple
kgen.func @cse_simple(%arg0: !kgen.scalar<bool>) -> index {
  // CHECK: [[INDEX:%.*]] = kgen.param.constant = <1>
  %index = kgen.param.constant: index = <1>

  // CHECK-NEXT: hlcf.if
  // CHECK: hlcf.yield [[INDEX]] : index
  // CHECK: hlcf.yield [[INDEX]] : index
  %0 = hlcf.if %arg0 -> index{
    %index_again = kgen.param.constant: index = <1>
    hlcf.yield %index_again : index
  } else {
    %index_again = kgen.param.constant: index = <1>
    hlcf.yield %index_again : index
  }
  // CHECK: kgen.return [[INDEX]] : index
  kgen.return %index : index
}
