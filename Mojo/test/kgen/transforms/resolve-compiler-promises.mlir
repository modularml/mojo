// RUN: kgen-opt %s -split-input-file -verify-parameters -resolve-compiler-promises -canonicalize -verify-parameters -o %t
// RUN: cat %t | FileCheck %s --check-prefix=GONE
// RUN: cat %t | FileCheck %s

// GONE: kgen.func @use
// GONE-NONE: pop.compiler

kgen.func @use(%arg0: index) {
  kgen.unreachable
}

kgen.func @use_i32(%arg0: i32) {
  kgen.unreachable
}

// CHECK-LABEL: kgen.func @top(%arg0: index, %arg1: i32)
kgen.func @top(%arg0: index) capturing {
  pop.compiler.global_store "foobar", %arg0 : index
  // CHECK: call @transitive(%arg1, %arg0) : (i32, index) capturing -> ()
  kgen.call @transitive() : () capturing -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @transitive(%arg0: i32, %arg1: index)
kgen.func @transitive() capturing {
  // CHECK: call @inner(%arg1)
  kgen.call @inner() : () capturing -> ()
  %0 = pop.compiler.global_load "baz" : i32
  // CHECK: call @use_i32(%arg0)
  kgen.call @use_i32(%0) : (i32) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @inner(%arg0: index)
kgen.func @inner() capturing {
  %idx0 = index.constant 0
  pop.compiler.global_store "index", %idx0 : index
  %0 = pop.compiler.global_load "foobar" : index
  // CHECK: call @use(%arg0)
  kgen.call @use(%0) : (index) -> ()
  %1 = pop.compiler.global_load "index" : index
  // CHECK: call @use(%idx0)
  kgen.call @use(%1) : (index) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @make_a_closure(%arg0: index)
kgen.func @make_a_closure(%arg0: index) -> !kgen.generator<()capturing -> ()> {
  pop.compiler.global_store "foobar", %arg0 : index
  // CHECK: create_closure[(index) capturing -> (): @capturing](%arg0)
  %0 = kgen.create_closure[() capturing -> (): @capturing]()
  kgen.return %0: !kgen.generator<()capturing -> ()>
}

// CHECK-LABEL: kgen.func @capturing(%arg0: index) capturing
kgen.func @capturing() capturing {
  %0 = pop.compiler.global_load "foobar" : index
  // CHECK: call @use(%arg0)
  kgen.call @use(%0) : (index) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @make_a_coroutine(%arg0: index)
kgen.func @make_a_coroutine(%arg0: index) {
  pop.compiler.global_store "foobar", %arg0 : index
  // CHECK: co.invoke[(index) async|capturing -> (): @async_fn](%arg0)
  co.invoke [() async|capturing -> (): @async_fn]()
  kgen.return
}

// CHECK-LABEL: kgen.func @async_fn(%arg0: index) async
kgen.func @async_fn() async|capturing {
  %0 = pop.compiler.global_load "foobar" : index
  // CHECK: call @use(%arg0)
  kgen.call @use(%0) : (index) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @unused_store
kgen.func @unused_store(%arg0: index) {
  pop.compiler.global_store "foobar", %arg0 : index
  // CHECK-NEXT: kgen.return
  kgen.return
}

// CHECK-LABEL kgen.func @store_does_not_dominate(%arg0: index, %arg1: index)
kgen.func @store_does_not_dominate(%arg0: index) capturing {
  // CHECK: loop
  hlcf.loop {
    pop.compiler.global_store "foobar", %arg0 : index
    // CHECK: call @use(%arg0)
    kgen.call @use(%arg0) : (index) -> ()
    // CHECK: break
    hlcf.break
  }
  %0 = pop.compiler.global_load "foobar" : index
  // CHECK: call @use(%arg1)
  kgen.call @use(%0) : (index) -> ()
  kgen.return
}

// COM: The cyclic node does not form an edge in the call graph.

// CHECK-LABEL: kgen.func @scc_pred(%arg0: index)
kgen.func @scc_pred(%arg0: index) {
  pop.compiler.global_store "var", %arg0 : index
  // CHECK: call @request(%arg0)
  kgen.call @request() : () capturing -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @request(%arg0: index)
kgen.func @request() capturing {
  pop.compiler.global_load "var" : index
  // CHECK: call @cyclic() : () -> ()
  kgen.call @cyclic() : () -> ()
  kgen.return
}

kgen.func @cyclic() {
  kgen.call @cyclic() : () -> ()
  kgen.return
}

// -----

// COM: Test that an empty callgraph does not crash.

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="p:64:64", simd_bit_width=128>} {

kgen.func @closure1() capturing {
  %0 = pop.compiler.global_load "cap1" : i32
  kgen.return
}

kgen.func @closure2() capturing {
  %0 = pop.compiler.global_load "cap2" : i64
  kgen.return
}

// CHECK-LABEL: kgen.func @join(%arg0: i64, %arg1: i32) capturing
kgen.func @join() capturing {
  // CHECK-NEXT: call @closure1(%arg1)
  // CHECK-NEXT: call @closure2(%arg0)
  kgen.call @closure1() : () capturing -> ()
  kgen.call @closure2() : () capturing -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @call_fn(%arg0: i64, %arg1: i32)
kgen.func @call_fn(%arg0: i64, %arg1: i32) {
  // CHECK: kgen.call @join(%{{.*}}, %{{.*}}) : (i64, i32) capturing -> ()
  kgen.call @join() : () capturing -> ()
  kgen.return
}

}

// -----

// CHECK-LABEL: kgen.func @entry
// CHECK-SAME: (%arg0: index, %arg1: i32)
kgen.func @entry(%arg0: index, %arg1: i32) {
  pop.compiler.global_store "x", %arg1 : i32
  // CHECK-NEXT: call @simple_recursion(%arg0, %arg1)
  kgen.call @simple_recursion(%arg0) : (index) capturing -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @simple_recursion(%arg0: index, %arg1: i32)
kgen.func @simple_recursion(%arg1: index) capturing {
  // CHECK-NEXT: call @simple_recursion(%arg0, %arg1)
  kgen.call @simple_recursion(%arg1) : (index) capturing -> ()
  %0 = pop.compiler.global_load "x" : i32
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @entry
// CHECK-SAME: (%arg0: index, %arg1: i32)
kgen.func @entry(%arg0: index, %arg1: i32) {
  // CHECK-NEXT: %0 = pop.stack_allocation
  %0 = pop.stack_allocation 1 x i64
  pop.compiler.global_store "x", %arg1 : i32
  // CHECK-NEXT: call @recursion_with_result(%arg0, %arg1, %0)
  kgen.call @recursion_with_result(%arg0, %0) : (index, !kgen.pointer<i64> byref_result) capturing -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @recursion_with_result(%arg0: index, %arg1: i32, %arg2: !kgen.pointer<i64> byref_result)
kgen.func @recursion_with_result(%arg1: index, %res: !kgen.pointer<i64> byref_result) capturing {
  // CHECK-NEXT: call @recursion_with_result(%arg0, %arg1, %arg2)
  kgen.call @recursion_with_result(%arg1, %res) : (index, !kgen.pointer<i64> byref_result) capturing -> ()
  %0 = pop.compiler.global_load "x" : i32
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @entry
// CHECK-SAME: (%arg0: index, %arg1: i32)
kgen.func @entry(%arg0: index, %arg1: i32) {
  // CHECK-NEXT: %0 = pop.stack_allocation
  // CHECK-NEXT: %1 = pop.stack_allocation
  %0 = pop.stack_allocation 1 x i1
  %1 = pop.stack_allocation 1 x i64
  pop.compiler.global_store "x", %arg1 : i32
  // CHECK-NEXT: call @recursion_with_error_result(%arg0, %arg1, %0, %1)
  kgen.call @recursion_with_error_result(%arg0, %0, %1) : (index, !kgen.pointer<i1> byref_error, !kgen.pointer<i64> byref_result) capturing|throws -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @recursion_with_error_result(%arg0: index, %arg1: i32, %arg2: !kgen.pointer<i1> byref_error, %arg3: !kgen.pointer<i64> byref_result)
kgen.func @recursion_with_error_result(%arg1: index, %err: !kgen.pointer<i1> byref_error, %res: !kgen.pointer<i64> byref_result) capturing|throws {
  // CHECK-NEXT: call @recursion_with_error_result(%arg0, %arg1, %arg2, %arg3)
  kgen.call @recursion_with_error_result(%arg1, %err, %res) : (index, !kgen.pointer<i1> byref_error, !kgen.pointer<i64> byref_result) capturing|throws -> ()
  %0 = pop.compiler.global_load "x" : i32
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @entry
// CHECK-SAME: (%arg0: i32, %arg1: i64)
kgen.func @entry(%arg0: i32, %arg1: i64) {
  pop.compiler.global_store "x", %arg0 : i32
  pop.compiler.global_store "y", %arg1 : i64
  // CHECK-NEXT: call @recursion_foo(%arg1, %arg0)
  kgen.call @recursion_foo() : () capturing -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @recursion_foo(%arg0: i64, %arg1: i32)
kgen.func @recursion_foo() capturing {
  %1 = pop.compiler.global_load "y" : i64
  // CHECK-NEXT: call @recursion_bar(%arg0, %arg1)
  kgen.call @recursion_bar() : () capturing -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @recursion_bar(%arg0: i64, %arg1: i32)
kgen.func @recursion_bar() capturing {
  %0 = pop.compiler.global_load "x" : i32
  // CHECK-NEXT: call @recursion_foo(%arg0, %arg1)
  kgen.call @recursion_foo() : () capturing -> ()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @root(%arg0: i1)
kgen.func @root() capturing {
  // CHECK-NEXT: call @a(%arg0)
  kgen.call @a() : () capturing -> ()
  // CHECK-NEXT: call @b(%arg0)
  kgen.call @b() : () capturing -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @a(%arg0: i1)
kgen.func @a() capturing {
  // CHECK-NEXT: call @b(%arg0)
  kgen.call @b() : () capturing -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @b(%arg0: i1)
kgen.func @b() capturing {
  // CHECK-NEXT: call @a(%arg0)
  kgen.call @a() : () capturing -> ()
  %0 = pop.compiler.global_load "x" : i1
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @root(%arg0: i1)
kgen.func @root() capturing {
  // CHECK-NEXT: call @a()
  kgen.call @a() : () capturing -> ()
  // CHECK-NEXT: call @b(%arg0)
  kgen.call @b() : () capturing -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @a()
kgen.func @a() capturing {
  // CHECK-NEXT: %0 = kgen.param.constant
  %0 = kgen.param.constant: i1 = <0>
  pop.compiler.global_store "x", %0 : i1
  // CHECK-NEXT: call @b(%0)
  kgen.call @b() : () capturing -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @b(%arg0: i1)
kgen.func @b() capturing {
  // CHECK-NEXT: call @a()
  kgen.call @a() : () capturing -> ()
  %0 = pop.compiler.global_load "x" : i1
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @foo(%arg0: index)
kgen.func @foo() capturing {
  // CHECK-NEXT: call @xd(%arg0)
  kgen.call @xd() : () capturing -> ()
  // CHECK-NEXT: call @foo(%arg0)
  kgen.call @foo() : () capturing -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @xd(%arg0: index)
kgen.func @xd() capturing {
  %0 = pop.compiler.global_load "var" : index
  kgen.return
}

// CHECK-LABEL: kgen.func export @top(%arg0: index)
kgen.func export @top(%arg0: index) {
  pop.compiler.global_store "var", %arg0 : index
  // CHECK-NEXT: call @foo(%arg0)
  kgen.call @foo() : () capturing -> ()
  kgen.return
}

// -----

// CHECK-LABEL: @cross_device
// CHECK-SAME: ["y" : i64, "x" : i32]
kgen.func export @cross_device() capturing {
  %0 = pop.compiler.global_load "x" : i32
  %1 = pop.compiler.global_load "y" : i64
  kgen.return
}
