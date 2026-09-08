// RUN: kgen-opt %s -resolve-compiler-promises | FileCheck %s

// CHECK-LABEL: @useAGlobalForNoReason
kgen.func @useAGlobalForNoReason() -> index {
  // CHECK-NEXT: index.constant
  %idx0 = index.constant 0
  pop.compiler.global_store "aGlobal", %idx0 : index
  %0 = pop.compiler.global_load "aGlobal" : index
  // CHECK-NEXT: kgen.return
  kgen.return %0 : index
}

// CHECK-LABEL: @multiLoad
kgen.func @multiLoad() -> (index, index) {
  // CHECK-NEXT: index.constant
  %idx0 = index.constant 0
  pop.compiler.global_store "aGlobal", %idx0 : index
  %0 = pop.compiler.global_load "aGlobal" : index
  %1 = pop.compiler.global_load "aGlobal" : index
  // CHECK-NEXT: kgen.return
  kgen.return %0, %1 : index, index
}

// CHECK-LABEL: @multiLoadNested
kgen.func @multiLoadNested(%pred : !kgen.scalar<bool>) -> index {
  // CHECK-NEXT: index.constant
  %idx0 = index.constant 0
  pop.compiler.global_store "aGlobal", %idx0 : index
  %0 = pop.compiler.global_load "aGlobal" : index
  // CHECK-NEXT: hlcf.if %arg0 {
  hlcf.if %pred {
    // CHECK-NOT: pop.compiler.global_load
    %1 = pop.compiler.global_load "aGlobal" : index
    // CHECK-NEXT: hlcf.yield
    hlcf.yield
  } else {
    hlcf.yield
  }
  // CHECK: kgen.return
  kgen.return %0: index
}

// CHECK-LABEL: kgen.func @store_twice
kgen.func @store_twice(%arg0: i32, %arg1: i64) -> (i32, i64) {
  // CHECK-NEXT: return %arg0, %arg1
  pop.compiler.global_store "foo", %arg0 : i32
  %0 = pop.compiler.global_load "foo" : i32
  pop.compiler.global_store "foo", %arg1: i64
  %1 = pop.compiler.global_load "foo" : i64
  kgen.return %0, %1 : i32, i64
}
