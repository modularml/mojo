// RUN: kgen-opt %s -allow-unregistered-dialect -verify-parameters | kgen-opt -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -emit-bytecode -allow-unregistered-dialect | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: "pog.metadata"
// CHECK-SAME: pog1 = #kgen.pog_metadata<"some_keyword_param", pos_or_kw, not_vararg>,
// CHECK-SAME: pog2 = #kgen.pog_metadata<"some_variadic_param", pos_or_kw, pos_vararg>
// CHECK-SAME: pog3 = #kgen.pog_metadata<"some_default_param", pos_or_kw, not_vararg, default ?>
"pog.metadata"() {
  pog1 = #kgen.pog_metadata<"some_keyword_param", pos_or_kw, not_vararg>,
  pog2 = #kgen.pog_metadata<"some_variadic_param", pos_or_kw, pos_vararg>,
  pog3 = #kgen.pog_metadata<"some_default_param", pos_or_kw, not_vararg, default :index ?>
} : () -> ()

// CHECK-LABEL: "pogs.with_defaults"
// CHECK-SAME: {pogs = #kgen.pog_list<
// CHECK-SAME: [<"a", pos, not_vararg>, <"b", pos_or_kw, not_vararg, default :f32 4.200000e+00>,
// CHECK-SAME: <"c", kw, not_vararg>, <"d", kw, not_vararg, default :i64 1>]> : !kgen.non_struct_type}
"pogs.with_defaults"() {pogs = #kgen.pog_list<
  [<"a", pos, not_vararg>, <"b", pos_or_kw, not_vararg, default :f32 4.2>,
   <"c", kw, not_vararg>, <"d", kw, not_vararg, default :i64 1>]
>} : () -> ()

// CHECK-LABEL: "pogs.with_variadics"
// CHECK-SAME: {pogs = #kgen.pog_list<
// CHECK-SAME: [<"a", pos, not_vararg>, <"b", pos_or_kw, not_vararg>, <"c", kw, not_vararg>, <"d", kw, pack_vararg>],
// CHECK-SAME: owned_in_mem> : !kgen.non_struct_type}
"pogs.with_variadics"() {pogs = #kgen.pog_list<
  [<"a", pos, not_vararg>, <"b", pos_or_kw, not_vararg>, <"c", kw, not_vararg>, <"d", kw, pack_vararg>],
  owned_in_mem
>} : () -> ()

// CHECK-LABEL: "pogs.with_body_constraints"
// CHECK-SAME: {pogs = #kgen.pog_list<[<"a", pos, not_vararg>]{{.*}}<true, #{{loc[0-9]*}}>, <true, #{{loc[0-9]*}}>}> : !kgen.non_struct_type}
"pogs.with_body_constraints"() {pogs = #kgen.pog_list<
  [<"a", pos, not_vararg>]
  {<true, loc("body.mojo":1:1)>, <true, loc("body.mojo":1:2)>}
>} : () -> ()

// CHECK-LABEL: "pogs.variadic_and_body"
// CHECK-SAME: {pogs = #kgen.pog_list<[<"a", pos, not_vararg>, <"b", pos_or_kw, pack_vararg>]{{.*}}<true, #{{loc[0-9]*}}>, <true, #{{loc[0-9]*}}>}{{.*}}owned_in_mem> : !kgen.non_struct_type}
"pogs.variadic_and_body"() {pogs = #kgen.pog_list<
  [<"a", pos, not_vararg>, <"b", pos_or_kw, pack_vararg>]
  {<true, loc("body.mojo":2:1)>, <true, loc("body.mojo":2:2)>},
  owned_in_mem
>} : () -> ()

// CHECK-LABEL: "empty.pogs"
// CHECK-SAME: {pogs = #kgen.pog_list<[]> : !kgen.non_struct_type}
"empty.pogs"() {pogs = #kgen.pog_list<[]>} : () -> ()
