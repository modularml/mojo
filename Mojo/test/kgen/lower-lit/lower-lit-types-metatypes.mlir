// RUN: kgen-opt -lower-lit -verify-parameters --kgen-print-inline-type-values %s --split-input-file | FileCheck %s

// CHECK: kgen.struct.generator @[[STRUCT_CONTAINER:.+]]<T: type> = struct_inst<"Container"[T]<:type T>(x: T) memoryOnly>
lit.struct.decl @Container<T: trait<@Trait>> {
  lit.struct.field x: !kgen.param<:trait<@Trait> T>
}

// CHECK: kgen.struct.generator @[[STRUCT_ELEMENT:.+]] = struct_inst<"Element" memoryOnly>
lit.struct.decl @Element {
}

// CHECK: kgen.generator @func<T: type>
// CHECK-SAME: (%arg0: !kgen.struct<(T) memoryOnly>
kgen.generator @func<T: trait<@Trait>>(%arg0: !lit.struct<@Container<:trait<@Trait> T>>) {
  kgen.return
}

kgen.generator @f() {
  kgen.return
}

// CHECK: kgen.generator @top
kgen.generator @top(%arg0: !lit.struct<@Container<:trait<@Trait> [@Element]>>) {
  // CHECK-NEXT: call @func<:type [typevalue<#kgen.genref<@[[STRUCT_ELEMENT]]>>, struct<() memoryOnly>]>(%arg0) : (!kgen.struct<(struct<() memoryOnly>) memoryOnly>) -> ()
  kgen.call @func<:trait<@Trait> [@Element]>(%arg0) : (!lit.struct<@Container<:trait<@Trait> [@Element]>>) -> ()
  kgen.return
}

// -----

!MyTrait = !lit.trait<@MyTrait>
!MyStruct = !lit.struct<@MyStruct>

lit.trait.decl @MyTrait {
  lit.alias.decl Ty: type
}

lit.struct.decl @MyContainer<T: !MyTrait> {
  lit.struct.field a: !kgen.param<#kgen.get_witness<:!MyTrait T, @test::@MyTrait, "Ty">>

  lit.fn @"__init__(get_witness($0, test::MyTrait, Ty))"(%a: !kgen.param<#kgen.get_witness<:!MyTrait T, @test::@MyTrait, "Ty">>) -> !kgen.none {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}

lit.struct.decl @MyStruct {
  lit.alias.decl Ty: type = <index>

  kgen.conformance @"test::MyTrait" {
    kgen.witness "Ty" : type = index
  }
}

// CHECK-LABEL: @test_my_container
lit.fn @test_my_container() -> !kgen.none {
  // CHECK-NEXT: = <[typevalue<#kgen.genref<@MyContainer<:type [typevalue<#kgen.genref<@MyStruct>>, struct<() memoryOnly>]>>>, struct<(index) memoryOnly>]>
  kgen.param.declare ResultType: meta<!lit.struct<@MyContainer <:!MyTrait !MyStruct>>> = <@MyContainer <:!MyTrait !MyStruct>>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}
