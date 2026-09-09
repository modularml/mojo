// RUN: support-dialect-opt %s -convert-debuginfo-to-llvm -mlir-print-debuginfo | FileCheck %s

// CHECK-DAG: #[[BASIC:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "f32", sizeInBits = 32, encoding = DW_ATE_float>
!f32Type = !debuginfo.basic<f32 { sizeInBits = 32, alignInBits = 32, encoding = DW_ATE_float }>

// CHECK-DAG: #[[ARRAY:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, name = "", baseType = #[[BASIC]], sizeInBits = 320, elements = #llvm.di_subrange<count = 10 : i64>>
!arrayType = !debuginfo.array<10 x !f32Type>

// CHECK-DAG: #[[PTR:.*]] = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #[[BASIC]], sizeInBits = 64, alignInBits = 64, dwarfAddressSpace = 5>
!pointerType = !debuginfo.ptr<!f32Type {sizeInBits = 64, alignInBits = 64, addressSpace = 5}>

// CHECK-DAG: #[[MEMBER1:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "first", baseType = #[[BASIC]], sizeInBits = 32, alignInBits = 32>
!memberType1 = !debuginfo.member<first: !f32Type>

// CHECK-DAG: #[[MEMBER2:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "second", baseType = {{.*}}, sizeInBits = 64, alignInBits = 64, offsetInBits = 64>
!memberType2 = !debuginfo.member<second: !pointerType>

// CHECK-DAG: #[[STRUCT:.*]] = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "Foo", sizeInBits = 128, alignInBits = 64, elements = #[[MEMBER1]], #[[MEMBER2]]>
!structType = !debuginfo.struct<"Foo"(!memberType1, !memberType2)>

// CHECK-DAG: #[[UNRESOLVED:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "i64", sizeInBits = 64, encoding = DW_ATE_unsigned>
!unresolvedType = !debuginfo.unresolved<i64>

// CHECK-DAG: #[[UNSPECIFIED:.*]] =  #llvm.di_basic_type<tag = DW_TAG_unspecified_type, name = "void">
!unspecifiedType = !debuginfo.unspecified<"void">

// TODO: Check for discriminator field once upstream is ready.
!i1Type = !debuginfo.basic<i1 { sizeInBits = 1, alignInBits = 1, encoding = DW_ATE_unsigned }>
!discriminator = !debuginfo.member<discr: !i1Type>
// CHECK-DAG: #[[VARIANT1:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "v0", baseType = #[[STRUCT]], sizeInBits = 128, alignInBits = 64>
!variant1 = !debuginfo.member<v0: !structType>
// CHECK-DAG: #[[VARIANT2:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "v1", baseType = #[[BASIC]], sizeInBits = 32, alignInBits = 32>
!variant2 = !debuginfo.member<v1: !f32Type>
// CHECK-DAG: #[[VARIANT:.*]] = #llvm.di_composite_type<tag = DW_TAG_variant_part, name = "", sizeInBits = 128, alignInBits = 64, elements = #[[VARIANT1]], #[[VARIANT2]]>
!variantType = !debuginfo.variant<"Variant"(!variant1, !variant2), !discriminator { sizeInBits = 128, alignInBits = 64 }>

// CHECK-DAG: #[[VECTOR:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #[[BASIC]], flags = Vector, sizeInBits = 320, elements = #llvm.di_subrange<count = 10 : i64>>
!vectorType = !debuginfo.vector<10 x !f32Type>

// CHECK-DAG: #[[NAMEDVECTOR:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, name = "test.op", baseType = #[[BASIC]], flags = Vector, sizeInBits = 320, elements = #llvm.di_subrange<count = 10 : i64>>
!namedVectorType = !debuginfo.vector<10 x !f32Type {name = "test.op"}>

// CHECK: #[[SUBROUTINE:.*]] = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #[[BASIC]], #[[ARRAY]], #[[PTR]], #[[STRUCT]], #[[UNRESOLVED]], #[[UNSPECIFIED]], #[[VECTOR]], #[[NAMEDVECTOR]], #[[VARIANT]]>
!subroutineType = !debuginfo.subroutine<(
  !arrayType, !pointerType, !structType,
  !unresolvedType, !unspecifiedType, !vectorType, !namedVectorType, !variantType
) -> (!f32Type): DW_CC_normal>

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
