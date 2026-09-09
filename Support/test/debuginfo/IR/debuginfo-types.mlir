// RUN: support-dialect-opt %s | support-dialect-opt | FileCheck %s
// RUN: support-dialect-opt -emit-bytecode %s | support-dialect-opt | FileCheck %s

// CHECK-DAG: ![[BASIC:.*]] = !debuginfo.basic<f32 {sizeInBits = 32, alignInBits = 32, encoding = DW_ATE_float}>
!f32Type = !debuginfo.basic<f32 {sizeInBits = 32, alignInBits = 32, encoding = DW_ATE_float }>

// CHECK-DAG: ![[ARRAY:.*]] = !debuginfo.array<10 x ![[BASIC]]>
!arrayType = !debuginfo.array<10 x !f32Type>

// CHECK-DAG: ![[MEMBER:.*]] = !debuginfo.member<x: ![[BASIC]]>
!memberType = !debuginfo.member<x: !f32Type>

// CHECK-DAG: ![[PTR:.*]] = !debuginfo.ptr<![[BASIC]] {sizeInBits = 64, alignInBits = 64}>
!pointerType = !debuginfo.ptr<!f32Type {sizeInBits = 64, alignInBits = 64}>

// CHECK-DAG: ![[PTR2:.*]] = !debuginfo.ptr<![[BASIC]] {sizeInBits = 64, alignInBits = 64, addressSpace = 4}>
!pointerType2 = !debuginfo.ptr<!f32Type {sizeInBits = 64, alignInBits = 64, addressSpace = 4}>

// CHECK-DAG: ![[STRUCT:.*]] = !debuginfo.struct<Foo(![[MEMBER]])>
!structType = !debuginfo.struct<"Foo"(!memberType)>

// CHECK-DAG: ![[SUBROUTINE:.*]] = !debuginfo.subroutine<(![[BASIC]]) -> (![[BASIC]]): DW_CC_normal>
!subroutineType = !debuginfo.subroutine<(!f32Type) -> (!f32Type): DW_CC_normal>

// CHECK-DAG: ![[UNRESOLVED:.*]] = !debuginfo.unresolved<index>
!unresolvedType = !debuginfo.unresolved<index>

// CHECK-DAG: ![[UNSPECIFIED:.*]] = !debuginfo.unspecified<"void">
!unspecifiedType = !debuginfo.unspecified<"void">

// CHECK-DAG: ![[VECTOR:.*]] = !debuginfo.vector<10 x ![[BASIC]]>
!vectorType = !debuginfo.vector<10 x !f32Type>

// CHECK: module attributes {
module attributes {
  // CHECK-SAME: test.type1 = ![[BASIC]]
  test.type1 = !f32Type,

  // CHECK-SAME: test.type2 = ![[ARRAY]]
  test.type2 = !arrayType,

  // CHECK-SAME: test.type3 = ![[PTR]]
  test.type3 = !pointerType,

  // CHECK-SAME: test.type4 = ![[STRUCT]]
  test.type4 = !structType,

  // CHECK-SAME: test.type5 = ![[SUBROUTINE]]
  test.type5 = !subroutineType,

  // CHECK-SAME: test.type6 = ![[UNRESOLVED]]
  test.type6 = !unresolvedType,

  // CHECK-SAME: test.type7 = ![[UNSPECIFIED]]
  test.type7 = !unspecifiedType,

  // CHECK-SAME: test.type8 = ![[VECTOR]]
  test.type8 = !vectorType,

  // CHECK-SAME: test.type9 = ![[PTR2]]
  test.type9 = !pointerType2
} {}
