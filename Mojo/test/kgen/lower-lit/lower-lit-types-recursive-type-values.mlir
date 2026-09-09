// RUN: kgen-opt %s -lower-lit -allow-unregistered-dialect -split-input-file -verify-parameters -kgen-print-inline-type-values | FileCheck %s

//===----------------------------------------------------------------------===//
// Self-Recursive Structs
//===----------------------------------------------------------------------===//

// CHECK: kgen.struct.generator @ListNode = struct_inst<"ListNode"(next: [pointer<typevalue<#kgen.genref<@ListNode>>>, pointer<struct<(pointer<none>) memoryOnly>>]) memoryOnly>
lit.struct.decl @ListNode {
  lit.struct.field next : !kgen.pointer<:type !lit.struct<@ListNode>>
}

kgen.generator @type_values() {
  // CHECK: kgen.param.declare listnode: type = <[typevalue<#kgen.genref<@ListNode>>, struct<(pointer<none>) memoryOnly>]>
  kgen.param.declare listnode: meta<!lit.struct<@ListNode>> = <[@ListNode]>
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//
// Mutual-Recursive Structs
//===----------------------------------------------------------------------===//

// CHECK: kgen.struct.generator @Bar = struct_inst<"Bar"(foo: [typevalue<#kgen.genref<@Foo>>, pointer<none>])>
lit.struct.decl @Bar register_passable {
  lit.struct.field foo: !lit.struct<@Foo>
}

// CHECK: kgen.struct.generator @Foo = struct_inst<"Foo"(bar_ptr: [pointer<typevalue<#kgen.genref<@Bar>>>, pointer<pointer<none>>])>
lit.struct.decl @Foo register_passable {
  lit.struct.field bar_ptr: !kgen.pointer<@Bar>
}

kgen.generator @type_values() {
  // CHECK: kgen.param.declare bar: type = <[typevalue<#kgen.genref<@Bar>>, pointer<none>]>
  kgen.param.declare bar: meta<!lit.struct<@Bar>> = <[@Bar]>
  // CHECK: kgen.param.declare foo: type = <[typevalue<#kgen.genref<@Foo>>, pointer<none>]>
  kgen.param.declare foo: meta<!lit.struct<@Foo>> = <[@Foo]>
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//
// Parametric Self-Recursive Structs
//===----------------------------------------------------------------------===//

// CHECK: kgen.struct.generator @ListNode = struct_inst<"ListNode"(next: [typevalue<#kgen.genref<@Pointer<:type [typevalue<#kgen.genref<@ListNode>>, struct<(pointer<none>) memoryOnly>]>>>, pointer<none>]) memoryOnly>
lit.struct.decl @ListNode {
  lit.struct.field next : !lit.struct<@Pointer<:type !lit.struct<@ListNode>>>
}

// CHECK: kgen.struct.generator @Pointer<ty: type> = struct_inst<"Pointer"[ty]<:type ty>(address: [pointer<typevalue<ty>>, pointer<ty>])>
lit.struct.decl @Pointer<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

kgen.generator @type_values() {
  // CHECK: kgen.param.declare listnode: type = <[typevalue<#kgen.genref<@ListNode>>, struct<(pointer<none>) memoryOnly>]>
  kgen.param.declare listnode: meta<!lit.struct<@ListNode>> = <[@ListNode]>
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//
// Function-Typed Field Referring To Self
//===----------------------------------------------------------------------===//

// CHECK: kgen.struct.generator @Foo = struct_inst<"Foo"(callback: [(!kgen.pointer<typevalue<#kgen.genref<@Foo>>> owned_in_mem) -> !kgen.none, (!kgen.pointer<struct<((!kgen.pointer<struct<(pointer<none>) memoryOnly>> owned_in_mem) -> !kgen.none) memoryOnly>> owned_in_mem) -> !kgen.none]) memoryOnly>
lit.struct.decl @Foo {
  lit.struct.field callback :
      !kgen.generator<(!lit.ref<@Foo, mut #lit.any.origin> owned_in_mem) -> !kgen.none>
}

kgen.generator @type_values() {
  // CHECK: kgen.param.declare foo: type = <[typevalue<#kgen.genref<@Foo>>, struct<((!kgen.pointer<struct<(pointer<none>) memoryOnly>> owned_in_mem) -> !kgen.none) memoryOnly>]>
  kgen.param.declare foo: meta<!lit.struct<@Foo>> = <[@Foo]>
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//
// Mutual Recursion Through Function-Typed Fields
//===----------------------------------------------------------------------===//

// CHECK: kgen.struct.generator @Ping = struct_inst<"Ping"(cb: [(!kgen.pointer<typevalue<#kgen.genref<@Pong>>> owned_in_mem) -> !kgen.none, {{.*}}]) memoryOnly>
lit.struct.decl @Ping {
  lit.struct.field cb :
      !kgen.generator<(!lit.ref<@Pong, mut #lit.any.origin> owned_in_mem) -> !kgen.none>
}

// CHECK: kgen.struct.generator @Pong = struct_inst<"Pong"(cb: [(!kgen.pointer<typevalue<#kgen.genref<@Ping>>> owned_in_mem) -> !kgen.none, {{.*}}]) memoryOnly>
lit.struct.decl @Pong {
  lit.struct.field cb :
      !kgen.generator<(!lit.ref<@Ping, mut #lit.any.origin> owned_in_mem) -> !kgen.none>
}

kgen.generator @type_values() {
  // CHECK: kgen.param.declare ping: type = <[typevalue<#kgen.genref<@Ping>>, {{.*}}]>
  kgen.param.declare ping: meta<!lit.struct<@Ping>> = <[@Ping]>
  // CHECK: kgen.param.declare pong: type = <[typevalue<#kgen.genref<@Pong>>, {{.*}}]>
  kgen.param.declare pong: meta<!lit.struct<@Pong>> = <[@Pong]>
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//
// Embedded Struct Recursion Through Function-Typed Fields
//===----------------------------------------------------------------------===//

lit.struct.decl @Ping {
  lit.struct.field pong : !lit.struct<@Pong>
}

lit.struct.decl @Pong {
  lit.struct.field cb :
      !kgen.generator<(!lit.ref<@Ping, mut #lit.any.origin> owned_in_mem) -> !kgen.none>
}

kgen.generator @type_values() {
  // CHECK: kgen.param.declare ping: type = <[typevalue<#kgen.genref<@Ping>>, struct<(struct<((!kgen.pointer<struct<(pointer<none>) memoryOnly>> owned_in_mem) -> !kgen.none) memoryOnly>) memoryOnly>]>
  kgen.param.declare ping: meta<!lit.struct<@Ping>> = <[@Ping]>
  // CHECK: kgen.param.declare pong: type = <[typevalue<#kgen.genref<@Pong>>, struct<((!kgen.pointer<struct<(struct<((!kgen.pointer<struct<(pointer<none>) memoryOnly>> owned_in_mem) -> !kgen.none) memoryOnly>) memoryOnly>> owned_in_mem) -> !kgen.none) memoryOnly>]>
  kgen.param.declare pong: meta<!lit.struct<@Pong>> = <[@Pong]>
  kgen.return
}
