// RUN: kgen-opt -lower-kgen-to-llvm -mlir-print-debuginfo %s | FileCheck %s

// Test proper handling of debug types.
!pointerTest = !kgen.pointer<index>
!voidPointerTest = !kgen.pointer<none>
!structTest = !kgen.struct<(index, struct<(index)>)>
!variantTest = !pop.union<index, index>
!signatureTest = !kgen.generator<(index) -> index>
!typeValuePairTest = !kgen.typevalue<[typevalue<#kgen.instref<@Pair>>, struct<(index, i1)>]>
!typeValueRecursionTest = !kgen.typevalue<[typevalue<#kgen.instref<@ListNode>>, struct<(pointer<none>)>]>

// CHECK-DAG: ![[INDEX:.*]] = !debuginfo.basic<index {sizeInBits = 64, alignInBits = 64, encoding = DW_ATE_signed}>

// CHECK-DAG: ![[MEMBER0:.*]] = !debuginfo.member<m0: ![[INDEX]]>

// CHECK-DAG: ![[PTR:.*]] = !debuginfo.ptr<![[INDEX]] {sizeInBits = 64, alignInBits = 64, addressSpace = 0}>

// CHECK-DAG: ![[NONE:.*]] = !debuginfo.struct<"!kgen.none"()>
// CHECK-DAG: ![[VOID_PTR:.*]] = !debuginfo.ptr<![[NONE]] {sizeInBits = 64, alignInBits = 64, addressSpace = 0}>

// CHECK-DAG: ![[INNER_STRUCT:.*]] = !debuginfo.struct<"!kgen.struct<(index)>"(![[MEMBER0]])>
// CHECK-DAG: ![[STRUCT_MEMBER:.*]] = !debuginfo.member<m1: ![[INNER_STRUCT]]>
// CHECK-DAG: ![[STRUCT:.*]] = !debuginfo.struct<"!kgen.struct<(index, struct<(index)>)>"(![[MEMBER0]], ![[STRUCT_MEMBER]])>

// CHECK-DAG: ![[VARIANT0:.*]] = !debuginfo.member<v0: ![[INDEX]]>
// CHECK-DAG: ![[VARIANT1:.*]] = !debuginfo.member<v1: ![[INDEX]]>
// CHECK-DAG: ![[VARIANT:.*]] = !debuginfo.variant<""(![[VARIANT0]], ![[VARIANT1]]) {sizeInBits = 64, alignInBits = 64}>

// CHECK-DAG: ![[SUBROUTINE:.*]] = !debuginfo.subroutine<(![[INDEX]]) -> (![[INDEX]]): DW_CC_normal>
// CHECK-DAG: ![[SIGNATURE:.*]] = !debuginfo.ptr<![[SUBROUTINE]] {sizeInBits = 64, alignInBits = 64}>

// CHECK-DAG: ![[CHAR:.*]] = !debuginfo.basic<kgen.dtype.si8 {sizeInBits = 8, alignInBits = 8, encoding = DW_ATE_signed}>
// CHECK-DAG: ![[SIZE_MEMBER:.*]] = !debuginfo.member<size: ![[INDEX]]>
// CHECK-DAG: ![[CHAR_PTR:.*]] = !debuginfo.ptr<![[CHAR]] {sizeInBits = 64, alignInBits = 64}>
// CHECK-DAG: ![[DATA_MEMBER:.*]] = !debuginfo.member<data: ![[CHAR_PTR]]>
// CHECK-DAG: ![[STRING:.*]] = !debuginfo.struct<"!kgen.string"(![[DATA_MEMBER]], ![[SIZE_MEMBER]])>

// CHECK-DAG: ![[STRUCT_INT_MEMBER:.*]] = !debuginfo.member<value: ![[INDEX]]>
// CHECK-DAG: ![[STRUCT_INT:.*]] = !debuginfo.struct<Int(![[STRUCT_INT_MEMBER]])>
// CHECK-DAG: ![[I1:.*]] = !debuginfo.basic<i1 {sizeInBits = 8, alignInBits = 8, encoding = DW_ATE_unsigned}>
// CHECK-DAG: ![[PAIR_MEMBER_0:.*]] = !debuginfo.member<first: ![[STRUCT_INT]]>
// CHECK-DAG: ![[PAIR_MEMBER_1:.*]] = !debuginfo.member<second: ![[I1]]>
// CHECK-DAG: ![[STRUCT_PAIR:.*]] = !debuginfo.struct<Pair(![[PAIR_MEMBER_0]], ![[PAIR_MEMBER_1]])>

// CHECK-DAG: ![[STRUCT_LISTNODE_INNER:.*]] = !debuginfo.struct<ListNode()>
// CHECK-DAG: ![[PTR_STRUCT_LISTNODE_INNER:.*]] = !debuginfo.ptr<!struct1 {sizeInBits = 64, alignInBits = 64, addressSpace = 0}>
// CHECK-DAG: ![[POINTER_LISTNODE_MEMBER:.*]] = !debuginfo.member<address: ![[PTR_STRUCT_LISTNODE_INNER]]>
// CHECK-DAG: ![[STRUCT_POINTER_LISTNODE:.*]] = !debuginfo.struct<Pointer(![[POINTER_LISTNODE_MEMBER]])>
// CHECK-DAG: ![[LISTNODE_MEMBER:.*]] = !debuginfo.member<next: ![[STRUCT_POINTER_LISTNODE]]>
// CHECK-DAG: ![[STRUCT_LISTNODE:.*]] = !debuginfo.struct<ListNode(![[LISTNODE_MEMBER]])>

// Test that KGENDType::index emits DW_ATE_signed and KGENDType::uindex emits
// DW_ATE_unsigned (these go through buildDebugTypeFromDType, not the
// IndexType path).
// CHECK-DAG: ![[KGEN_INDEX:.*]] = !debuginfo.basic<kgen.dtype.index {sizeInBits = 64, alignInBits = 64, encoding = DW_ATE_signed}>
// CHECK-DAG: ![[KGEN_UINDEX:.*]] = !debuginfo.basic<kgen.dtype.uindex {sizeInBits = 64, alignInBits = 64, encoding = DW_ATE_unsigned}>
// CHECK-DAG: ![[SCALAR_INDEX:.*]] = !debuginfo.vector<1 x ![[KGEN_INDEX]]
// CHECK-DAG: ![[SCALAR_UINDEX:.*]] = !debuginfo.vector<1 x ![[KGEN_UINDEX]]

// CHECK-DAG: !debuginfo.subroutine<(![[PTR]], ![[VOID_PTR]], ![[STRUCT]], ![[VARIANT]], ![[SIGNATURE]], ![[STRING]], ![[NONE]], ![[STRUCT_PAIR]], ![[STRUCT_LISTNODE]], ![[SCALAR_INDEX]], ![[SCALAR_UINDEX]]) -> (): DW_CC_normal>

!test = !debuginfo.subroutine<(
  !debuginfo.unresolved<!pointerTest>,
  !debuginfo.unresolved<!voidPointerTest>,
  !debuginfo.unresolved<!structTest>,
  !debuginfo.unresolved<!variantTest>,
  !debuginfo.unresolved<!signatureTest>,
  !debuginfo.unresolved<!kgen.string>,
  !debuginfo.unresolved<!kgen.none>,
  !debuginfo.unresolved<!typeValuePairTest>,
  !debuginfo.unresolved<!typeValueRecursionTest>,
  !debuginfo.unresolved<!kgen.scalar<index>>,
  !debuginfo.unresolved<!kgen.scalar<uindex>>
) -> (): DW_CC_normal>

#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !test

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="i64:64:64", simd_bit_width=128>} {
  kgen.struct.instance @Int = struct_inst<"Int"(value: index)>

  kgen.struct.instance @Pair = struct_inst<"Pair"(
    first: [typevalue<#kgen.instref<@Int>>, index],
    second: i1
  )>

  kgen.struct.instance @Pointer_ListNode = struct_inst<"Pointer"[ty]
    <:type [typevalue<#kgen.instref<@ListNode>>, struct<(pointer<none>) memoryOnly>]>(
      address: [pointer<typevalue<[typevalue<#kgen.instref<@ListNode>>, struct<(pointer<none>) memoryOnly>]>>, pointer<none>]
    )>

  kgen.struct.instance @ListNode = struct_inst<"ListNode"(
    next: [typevalue<#kgen.instref<@Pointer_ListNode>>, pointer<none>]
  ) memoryOnly>

  kgen.func @foo() {
    kgen.return loc(fused<#subprogram>["foo.mlir":10:10])
  } loc(fused<#subprogram>["foo.mlir":10:10])
}
