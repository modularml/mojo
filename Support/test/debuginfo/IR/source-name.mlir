// RUN: support-dialect-opt %s | support-dialect-opt | FileCheck %s

// CHECK-DAG: #int_name = #debuginfo.source_name<"int" from <"builtin">>
// CHECK-DAG: #simd_name = #debuginfo.source_name<"simd" from <"builtin">>
// CHECK-DAG: #Int_name = #debuginfo.source_name<"Int" from #int_name>
// CHECK-DAG: #SIMD_name = #debuginfo.source_name<"SIMD"[#Int_name] from #simd_name>
// CHECK-DAG: #decorator0_name = #debuginfo.source_name<"decorator0" from <"builtin">>
// CHECK-DAG: #func_name = #debuginfo.source_name<"func"(#SIMD_name)<"1 : index"> decorators<#decorator0_name, #decorator1_name>>
#builtin_name = #debuginfo.source_name<"builtin">
#int_name = #debuginfo.source_name<"int" from #builtin_name>
#simd_name = #debuginfo.source_name<"simd" from #builtin_name>
#Int_name = #debuginfo.source_name<"Int" from #int_name>
#SIMD_name = #debuginfo.source_name<"SIMD"[#Int_name] from #simd_name>
#decorator0_name = #debuginfo.source_name<"decorator0" from #builtin_name>
#decorator1_name = #debuginfo.source_name<"decorator1" from #builtin_name>
#func_name = #debuginfo.source_name<"func"(#SIMD_name)<"1 : index"> decorators<#decorator0_name, #decorator1_name>>

module attributes {debuginfo.source_name = #func_name} {}
