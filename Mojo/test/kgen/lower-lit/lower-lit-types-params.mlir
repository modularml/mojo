// RUN: kgen-opt %s -verify-parameters -lower-lit -verify-parameters -kgen-print-inline-type-values | FileCheck %s

lit.struct.decl @Coro<T: type> register_passable {
  lit.struct.field coro : !kgen.struct<(T)>
}

// CHECK-LABEL: kgen.generator @get_promise
// CHECK-SAME: %arg0: !kgen.struct<(T)>
kgen.generator @get_promise<T: type>(%arg0: !lit.struct<@Coro<:type T>>) {
  kgen.unreachable
}

// CHECK-LABEL: kgen.generator @get_coro
kgen.generator @get_coro<T: type>(%arg0: !lit.struct<@Coro<:type T>>) {
  // CHECK-NEXT: call @get_promise<:type T>(%arg0) : (!kgen.struct<(T)>)
  kgen.call @get_promise<:type T>(%arg0) : (!lit.struct<@Coro<:type T>>) -> ()
  kgen.unreachable
}
