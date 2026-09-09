// RUN: kgen-opt %s -allow-unregistered-dialect | kgen-opt -allow-unregistered-dialect | FileCheck %s

// The builder's parameter decls, argument types and result type are all
// "quoted" parameter expressions: the quote freezes the index references so
// they can be pulled out of their enclosing scope, and the fold unquotes them
// into the generator type it describes.
// ===----------------------------------------------------------------------=== //

// A builder whose components are all constant folds into the generator type it
// describes. The single argument carries a frozen implicit-origin reference.
// CHECK-LABEL: "func_gen_type_builder.concrete"
// CHECK-SAME: () -> !kgen.generator<!lit.generator<[1](!lit.ref<index, imm *[0,0]>) throws -> index>>
"func_gen_type_builder.concrete"() : () -> !kgen.func_gen_type_builder<
  #kgen.param_list<> : !kgen.param_list<!kgen.type>,
  #kgen.param_list<#kgen.quote<!lit.ref<index, imm #lit.implicit.origin.ref<0, 0>>>> : !kgen.param_list<!kgen.type>,
  #kgen.quote<#kgen.type<index>>,
  #kgen.fn_metadata<[imm, imm], "throws", #lit.fn_meta_origin_data<1>>,
  #kgen.pog_list<[]>,
  #kgen.pog_list<[]>>

// The argument and result types reference the declared parameters by frozen
// index reference, which the fold rewrites into index references of the
// generator it builds.
// CHECK-LABEL: "func_gen_type_builder.parametric"
// CHECK-SAME: () -> !kgen.generator<<type, type>(!kgen.param<*(0,0)>, !kgen.param<*(0,1)>) -> !kgen.param<*(0,0)>>
"func_gen_type_builder.parametric"() : () -> !kgen.func_gen_type_builder<
  #kgen.param_list<#kgen.quote<#kgen.type<!kgen.type>>, #kgen.quote<#kgen.type<!kgen.type>>> : !kgen.param_list<!kgen.type>,
  #kgen.param_list<#kgen.quote<*(0,0)>, #kgen.quote<*(0,1)>> : !kgen.param_list<!kgen.type>,
  #kgen.quote<*(0,0)>,
  #kgen.fn_metadata<[imm, imm], "none">>

// A parameter may be declared in terms of a parameter declared before it.
// CHECK-LABEL: "func_gen_type_builder.dependent_param"
// CHECK-SAME: () -> !kgen.generator<<type, *(0,0)>(!kgen.param<*(0,0)>) -> !kgen.param<*(0,0)>>
"func_gen_type_builder.dependent_param"() : () -> !kgen.func_gen_type_builder<
  #kgen.param_list<#kgen.quote<#kgen.type<!kgen.type>>, #kgen.quote<*(0,0)>> : !kgen.param_list<!kgen.type>,
  #kgen.param_list<#kgen.quote<*(0,0)>> : !kgen.param_list<!kgen.type>,
  #kgen.quote<*(0,0)>,
  #kgen.fn_metadata<[imm], "none">>

// Every component is a parameter expression, so a builder that is still
// symbolic survives until elaboration.
// CHECK-LABEL: "func_gen_type_builder.symbolic"
// CHECK-SAME: () -> !kgen.func_gen_type_builder
"func_gen_type_builder.symbolic"() : () -> !kgen.func_gen_type_builder<
  #kgen.param.decl.ref<"Ps"> : !kgen.param_list<!kgen.type>,
  #kgen.param.decl.ref<"Ts"> : !kgen.param_list<!kgen.type>,
  #kgen.param.decl.ref<"R"> : !kgen.type,
  #kgen.fn_metadata<[imm, mut], "throws">>
