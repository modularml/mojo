// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt -allow-unregistered-dialect -emit-bytecode %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-DAG: [[MY_BLOB:#.*]] = #interp.memory_handle<1, "0xFFFEFDFC">
// CHECK-DAG: [[STRING_BLOB:#.*]] = #interp.memory_handle<16, "hello world" string>
// CHECK-DAG: [[VARIADIC:#.*]] = #interp.memory_handle<8, "0xDEAD">
// CHECK-DAG: [[SYM:#.*]] = #interp.memory_handle<8, "0x0000000000000000">
#my_blob = #interp.memory_handle<1, "0xFFFEFDFC">
#string_blob = #interp.memory_handle<16, "hello world" string>
#variadic = #interp.memory_handle<8, "0xDEAD">
#sym = #interp.memory_handle<8, "0x0000000000000000">

// CHECK: #interp.memref<{[([[MY_BLOB]], heap, [(1, 1, 3)], []), ([[STRING_BLOB]], heap, [], [], 3)], []}, 0, 24> : memref<2xi32>
"some.op"() {a = #interp.memref<{[(#my_blob, heap, [(1, 1, 3)], []), (#string_blob, heap, [], [], 3)], []}, 0, 24> : memref<2xi32>} : () -> ()

// CHECK: #interp.memref<{[([[VARIADIC]], persistent, [], [])], []}, 0, 0> : memref<1xi32>
"some.op"() {a = #interp.memref<{[(#variadic, persistent, [], [])], []}, 0, 0> : memref<1xi32>} : () -> ()

// CHECK: #interp.memref<{[([[SYM]], heap, [], [0])], [#foo.symbol.constant<#interp<coord(0,0)>> : (index) -> index]}, 0, 0> : (index) -> index} : () -> ()
"some.op"() {a = #interp.memref<{[(#sym, heap, [], [0])], [#foo.symbol.constant<#interp<coord(0,0)>> : (index) -> index]}, 0, 0> : (index) -> index} : () -> ()

