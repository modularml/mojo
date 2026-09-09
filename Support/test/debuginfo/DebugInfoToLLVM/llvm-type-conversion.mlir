// RUN: support-dialect-opt %s -convert-debuginfo-to-llvm -mlir-print-debuginfo | FileCheck %s

// Test conversions for building debug information for LLVM types.

// CHECK-DAG: #[[VOID:.+]] = #llvm.di_null_type

// CHECK-DAG: #[[OPAQUE_BASE:.*]] = #llvm.di_basic_type<tag = DW_TAG_unspecified_type, name = "opaque">
// CHECK-DAG: #[[OPAQUE_PTR:.*]] = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #[[OPAQUE_BASE]], sizeInBits = {{.*}}, alignInBits = {{.*}}>
!opaquePointer = !debuginfo.unresolved<!llvm.ptr>

// CHECK-DAG: #[[I64:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "i64", sizeInBits = 64, encoding = DW_ATE_unsigned>
// CHECK-DAG: #[[STRUCT_MEM1:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "field_0", baseType = #[[I32:.*]], sizeInBits = 32, alignInBits = 32>
// CHECK-DAG: #[[STRUCT_MEM2:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "field_1", baseType = #[[I64]], sizeInBits = 64, alignInBits = 64, offsetInBits = 64>
// CHECK-DAG: #[[STRUCT:.*]] = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "", sizeInBits = {{.*}}, alignInBits = {{.*}}, elements = #di_derived_type1, #di_derived_type2>
!struct = !debuginfo.unresolved<!llvm.struct<(i32, i64)>>

// CHECK-DAG: #[[NAMED_STRUCT_MEM1:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "field_0", baseType = #[[OPAQUE_PTR]], sizeInBits = {{.*}}, alignInBits = {{.*}}>
// CHECK-DAG: #[[NAMED_STRUCT:.*]] = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "Buffer", sizeInBits = {{.*}}, alignInBits = {{.*}}, elements = #[[NAMED_STRUCT_MEM1]], #[[STRUCT_MEM2]]>
!namedStruct = !debuginfo.unresolved<!llvm.struct<"Buffer", (ptr, i64)>>

// CHECK-DAG: #[[ARRAY:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, name = "", baseType = #[[I32]], sizeInBits = 320, elements = #llvm.di_subrange<count = 10 : i64>>
!array = !debuginfo.unresolved<!llvm.array<10 x i32>>

// CHECK-DAG: #[[VECTOR:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #[[OPAQUE_PTR]], flags = Vector, sizeInBits = {{.*}}, elements = #llvm.di_subrange<count = 10 : i64>>
!vector = !debuginfo.unresolved<vector<10 x !llvm.ptr>>

// CHECK-DAG: #[[SUBROUTINE:.*]] = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #[[VOID]], #[[OPAQUE_PTR]], #[[STRUCT]], #[[NAMED_STRUCT]], #[[ARRAY]], #[[VECTOR]]>
!subroutineType = !debuginfo.subroutine<(
  !opaquePointer, !struct, !namedStruct,
  !array, !vector
) -> (): DW_CC_normal>

#file = #debuginfo.file<"foo.c" in "/mlir/">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_Mojo,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  sourceName = <"foo">,
  linkageName = "foo",
  file = #file,
  line = 10,
  scopeLine = 10,
  subprogramFlags = Definition
> : !subroutineType

func.func private @foo() loc(fused<#subprogram>["test.mlir":10:10])
