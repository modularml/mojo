// RUN: kgen-opt %s -allow-unregistered-dialect | kgen-opt -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -emit-bytecode -allow-unregistered-dialect | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: "fn_metadata.empty"
// CHECK-SAME: aux = #kgen.fn_metadata<[], "none"> : !kgen.non_struct_type
"fn_metadata.empty"() {aux = #kgen.fn_metadata<[], "none">} : () -> ()

// CHECK-LABEL: "fn_metadata.conventions"
// CHECK-SAME: aux = #kgen.fn_metadata<[imm, imm_mem, owned, owned_in_mem, deinit_mem, mut, ref, mutref, byref_error, byref_result], "none"> : !kgen.non_struct_type
"fn_metadata.conventions"() {aux = #kgen.fn_metadata<
  [imm, imm_mem, owned, owned_in_mem, deinit_mem, mut, ref, mutref,
   byref_error, byref_result], "none">} : () -> ()

// CHECK-LABEL: "fn_metadata.one_effect"
// CHECK-SAME: aux = #kgen.fn_metadata<[], "throws"> : !kgen.non_struct_type
"fn_metadata.one_effect"() {aux = #kgen.fn_metadata<[], "throws">} : () -> ()

// CHECK-LABEL: "fn_metadata.all_effects"
// CHECK-SAME: aux = #kgen.fn_metadata<[], "throws|async|capturing|refresult|cabi"> : !kgen.non_struct_type
"fn_metadata.all_effects"() {aux = #kgen.fn_metadata<
  [], "throws|async|capturing|refresult|cabi">} : () -> ()

// The metadata is the only optional component, and is dialect-specific.
// CHECK-LABEL: "fn_metadata.metadata"
// CHECK-SAME: aux = #kgen.fn_metadata<[imm, mut], "throws", #lit.fn_meta_origin_data<2>> : !kgen.non_struct_type
"fn_metadata.metadata"() {
  aux = #kgen.fn_metadata<[imm, mut], "throws", #lit.fn_meta_origin_data<2>>
} : () -> ()

// An explicitly typed attribute parses to the same attribute as the elided
// form above, since the type is always `!kgen.non_struct_type`.
// CHECK-LABEL: "fn_metadata.explicit_type"
// CHECK-SAME: aux = #kgen.fn_metadata<[imm], "async"> : !kgen.non_struct_type
"fn_metadata.explicit_type"() {
  aux = #kgen.fn_metadata<[imm], "async"> : !kgen.non_struct_type
} : () -> ()
