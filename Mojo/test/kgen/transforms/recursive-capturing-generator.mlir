// RUN: kgen-opt %s -pass-pipeline='builtin.module(outline-closures,elaborate-generators{use-parametric-interpret=false},resolve-compiler-promises,automatic-inline,canonicalize)' | FileCheck %s
// RUN: kgen-opt %s -pass-pipeline='builtin.module(outline-closures,elaborate-generators{use-parametric-interpret=true},resolve-compiler-promises,automatic-inline,canonicalize)' | FileCheck %s

// COM: https://github.com/modularml/modular/issues/19175

kgen.generator @use(%a: index) no_inline {
  kgen.return
}

kgen.generator @test<recurse: scalar<bool>, inner: () capturing -> index>(%a: index) always_inline {
  kgen.param.declare.region thing = () capturing -> index always_inline {
    kgen.return %a : index
  }

  kgen.param.if <recurse> {
    %first = kgen.call_param[() capturing -> index: thing]()
    kgen.call @use(%first) : (index) -> ()

    kgen.param.declare.region noop = () capturing -> index {
      %idx42 = index.constant 42
      kgen.return %idx42 : index
    }
    %idx77 = index.constant 77
    kgen.call @test<:scalar<bool> false, :() capturing -> index noop>(%idx77) : (index) -> ()

    %second = kgen.call_param[() capturing -> index: thing]()
    kgen.call @use(%second) : (index) -> ()
    kgen.param.yield
  } else {
    kgen.param.yield
  }

  kgen.return
}

// CHECK-LABEL: kgen.func export @top
kgen.generator export @top() {
  kgen.param.declare.region noop = () capturing -> index {
    %idx22 = index.constant 22
    kgen.return %idx22 : index
  }
  // CHECK-NEXT: index.constant 11
  %idx11 = index.constant 11
  // CHECK-NEXT: call @use(%idx11)
  // CHECK-NEXT: call @use(%idx11)
  kgen.call @test<:scalar<bool> true, :() capturing -> index noop>(%idx11) : (index) -> ()
  // CHECK-NEXT: return
  kgen.return
}
