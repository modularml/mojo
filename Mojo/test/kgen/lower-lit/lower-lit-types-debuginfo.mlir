// RUN: kgen-opt %s -lower-lit -allow-unregistered-dialect -split-input-file | FileCheck %s

// Test proper handling of debug types.

// Single-field structs are flattened in debuginfo (until #23914).
// CHECK-DAG: ![[FIELD:.*]] = !debuginfo.unresolved<!pop.array<2, simd<4, f32>>>
lit.struct.decl @SmallVector<N, T: type> register_passable {
  lit.struct.field data: !pop.array<N, T>
}
!structTest = !lit.struct<@SmallVector<2, :type !kgen.simd<4, f32>>>

// CHECK-DAG: ![[LLDATA:.*]] = !debuginfo.member<lldata: index>
// CHECK-DAG: ![[NONE_PTR_TYPE:.*]] = !debuginfo.ti.ptr<!kgen.none>
// CHECK-DAG: ![[RECURSIVE_FIELD:.*]] = !debuginfo.member<ptr: ![[NONE_PTR_TYPE]]>
// CHECK-DAG: ![[RECURSIVE_STRUCT:.*]] = !debuginfo.struct<"struct RecursiveStruct"(![[LLDATA]], ![[RECURSIVE_FIELD]])>
lit.struct.decl @RecursiveStruct register_passable {
  lit.struct.field lldata: index
  lit.struct.field ptr: !kgen.pointer<!lit.struct<@RecursiveStruct>>
}
!structTestRecursive = !lit.struct<@RecursiveStruct>

// CHECK-DAG: ![[MEMBER_A:.*]] = !debuginfo.member<a: !kgen.param<Int>>
// CHECK-DAG: ![[MEMBER_B:.*]] = !debuginfo.member<b: !kgen.simd<4, f32>>
// CHECK-DAG: ![[COMPLEX_STRUCT:.*]] = !debuginfo.struct<"module test::struct ComplexStruct[type,type]<`:type Int`,`:type simd<4, f32>`>"(![[MEMBER_A]], ![[MEMBER_B]])>
#complexStructSourceName = #debuginfo.source_name<(struct)"ComplexStruct"[<"type">, <"type">] from <(module)"test">>
lit.struct.decl @"$test::ComplexStruct"<A: type, B: type> attributes {sourceName = #complexStructSourceName} {
  lit.struct.field a: !kgen.param<A>
  lit.struct.field b: !kgen.param<B>
}
!structTestComplex = !lit.struct<@"$test::ComplexStruct"<:type Int, :type !kgen.simd<4, f32>>>

// This is only possible in mlir tests. Mojo parser will guarantee all structs have SourceNames.
// CHECK-DAG: ![[COMPLEX_STRUCT_NOSOURCENAME:.*]] = !debuginfo.struct<"struct `$test::ComplexStructNoSourceName`<`:type Int`,`:type simd<4, f32>`>"(![[MEMBER_A]], ![[MEMBER_B]])>
lit.struct.decl @"$test::ComplexStructNoSourceName"<A: type, B: type> {
  lit.struct.field a: !kgen.param<A>
  lit.struct.field b: !kgen.param<B>
}
!structTestComplexNoSourceName = !lit.struct<@"$test::ComplexStructNoSourceName"<:type Int, :type !kgen.simd<4, f32>>>

// CHECK-DAG: ![[COMPLEX_STRUCT_REF:.*]] = !debuginfo.ti.ptr<![[COMPLEX_STRUCT]]>
!structTestComplexRef = !lit.ref<!structTestComplex, imm *"mystruct`">

// CHECK: "test.types"
"test.types"() {
  // CHECK-SAME: structType = ![[FIELD]]
  structType = !debuginfo.unresolved<!structTest>,
  // CHECK-SAME: structTypeComplex = ![[COMPLEX_STRUCT]]
  structTypeComplex = !debuginfo.unresolved<!structTestComplex>,
  // CHECK-SAME: structTypeComplexNoSourceName = ![[COMPLEX_STRUCT_NOSOURCENAME]]
  structTypeComplexNoSourceName = !debuginfo.unresolved<!structTestComplexNoSourceName>,
  // CHECK-SAME: structTypeComplexRef = ![[COMPLEX_STRUCT_REF]]
  structTypeComplexRef = !debuginfo.unresolved<!structTestComplexRef>,
  // CHECK-SAME: structTypeRecursive = ![[RECURSIVE_STRUCT]]
  structTypeRecursive = !debuginfo.unresolved<!structTestRecursive>

} : () -> ()

// -----

// Test proper handling of debuginfo operations.

#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "foo"> : !debuginfo.unresolved<!pop.array<3, i32>>
#local_variable1 = #debuginfo.local_variable<scope = #subprogram, name = "bar"> : !debuginfo.unresolved<!pop.array<0, i32>>

#fileLoc = loc("foo.mlir":0:0)
#loc = loc(fused<#subprogram>[#fileLoc])

kgen.func @foo() {
  // CHECK-DAG: %[[LIST:.*]] = kgen.param.constant: array<3, i32> = <[1, 2, 3]>

  // CHECK: debuginfo.value #local_variable = %[[LIST]]
  %values = kgen.param.constant: array<3, i32> = <[1, 2, 3]> loc(#loc)
  debuginfo.value #local_variable = %values : !pop.array<3, i32> loc(#loc)
  // CHECK-DAG: %[[EMPTY:.*]] = kgen.param.constant: array<0, i32> = <[]>
  // CHECK: debuginfo.value #local_variable1 = %[[EMPTY]]
  %empty = kgen.param.constant: array<0, i32> = <[]> loc(#loc)
  debuginfo.value #local_variable1 = %empty : !pop.array<0, i32> loc(#loc)
  kgen.return loc(#loc)
} loc(#loc)
