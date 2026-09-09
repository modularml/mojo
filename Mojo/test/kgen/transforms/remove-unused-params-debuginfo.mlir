// RUN: kgen-opt --split-input-file --remove-unused-params --mlir-print-debuginfo %s  | FileCheck %s

#subprogram = #debuginfo.subprogram<sourceName = <"basic_arg_remove_1">>  : !debuginfo.subroutine<(!kgen.scalar<dt>, !kgen.pointer<T>, !kgen.pointer<scalar<dt>>) -> (): DW_CC_normal>
#loc = loc(fused<#subprogram>["foo.mlir":1:1])

#otherSP = #debuginfo.subprogram<sourceName = <"param_inlined_fn">>  : !debuginfo.subroutine<(!kgen.pointer<T>) -> (): DW_CC_normal>
#otherLoc = loc(fused<#otherSP>["foo.mlir":2:2])
#inlinedLoc = loc(callsite(#otherLoc at #loc))

// CHECK: kgen.generator @basic_arg_remove_1_REMOVED_ARG(%[[ARG0:.+]]: !kgen.pointer<none>
kgen.generator @basic_arg_remove_1<dt: dtype, T: type>(%arg0: !kgen.scalar<dt>, %arg1: !kgen.pointer<none>, %arg2: !kgen.pointer<none>) {
  pop.load %arg1 : !kgen.pointer<none> loc(#loc)
  pop.load %arg2 : !kgen.pointer<none> loc(#loc)
  // CHECK: kgen.return loc(#[[INLINED_LOC:.+]])
  kgen.return loc(#inlinedLoc)
// CHECK: } loc(#[[LOC:.+]])
} loc(#loc)

kgen.generator @user<dt: dtype, T: type>(%arg0: !kgen.scalar<dt>, %arg1: !kgen.pointer<none>, %arg2: !kgen.pointer<none>) {
  kgen.call @basic_arg_remove_1<:dtype dt, :type T>(%arg0, %arg1, %arg2) : (!kgen.scalar<dt>, !kgen.pointer<none>, !kgen.pointer<none>) -> ()
  kgen.return
}

// CHECK-DAG: ![[UNSPECIFIED:.+]] = !debuginfo.unspecified<"optimized out">
// CHECK-DAG: ![[SUBROUTINE:.+]] = !debuginfo.subroutine<(![[UNSPECIFIED]], !kgen.pointer<#interp.uninitmem>, !kgen.pointer<scalar<#interp.uninitmem>>) -> (): DW_CC_normal>
// CHECK-DAG: #[[SP:.+]] = #debuginfo.subprogram<sourceName = <"basic_arg_remove_1">, linkageName = "basic_arg_remove_1_REMOVED_ARG"> : ![[SUBROUTINE]]
// CHECK-DAG: #[[LOC]] = loc(fused<#[[SP]]>


// CHECK-DAG: ![[OTHER_SUBROUTINE:.+]] = !debuginfo.subroutine<(!kgen.pointer<#interp.uninitmem>) -> (): DW_CC_normal>
// CHECK-DAG: #[[OTHER_SP:.+]] = #debuginfo.subprogram<sourceName = <"param_inlined_fn">> : ![[OTHER_SUBROUTINE]]
// CHECK-DAG: #[[OTHER_LOC:.+]] = loc(fused<#[[OTHER_SP]]>
// CHECK-DAG: #[[INLINED_LOC]] = loc(callsite(#[[OTHER_LOC]] at

// -----

#subprogram = #debuginfo.subprogram<sourceName = <"basic_arg_remove_debug_only_user">>  : !debuginfo.subroutine<(!kgen.scalar<dt>, !kgen.pointer<none>) -> (): DW_CC_normal>
#loc = loc(fused<#subprogram>["foo.mlir":1:1])
#di_arg0 = #debuginfo.local_variable<scope = #subprogram, name = "arg0"> : !debuginfo.unresolved<!kgen.scalar<dt>>

// CHECK: #[[SP_REMOVED_ARG:.+]] = #debuginfo.subprogram<sourceName = <"basic_arg_remove_debug_only_user">, linkageName = "basic_arg_remove_debug_only_user_REMOVED_ARG">
// CHECK: #[[DI_ARG:.+]] = #debuginfo.local_variable<scope = #[[SP_REMOVED_ARG]], name = "arg0">

// CHECK: kgen.generator @basic_arg_remove_debug_only_user_REMOVED_ARG
kgen.generator @basic_arg_remove_debug_only_user<dt: dtype>(%arg0: !kgen.scalar<dt>, %arg1: !kgen.pointer<none>) {
  // CHECK-NEXT: debuginfo.kill #[[DI_ARG]]
  // CHECK-NOT: debuginfo.value
  debuginfo.value #di_arg0 = %arg0 : !kgen.scalar<dt> loc(#loc)
  pop.load %arg1 : !kgen.pointer<none> loc(#loc)
  kgen.return loc(#loc)
} loc(#loc)

kgen.generator @user<dt: dtype, T: type>(%arg0: !kgen.scalar<dt>, %arg1: !kgen.pointer<none>, %arg2: !kgen.pointer<none>) {
  kgen.call @basic_arg_remove_debug_only_user<:dtype dt>(%arg0, %arg1) : (!kgen.scalar<dt>, !kgen.pointer<none>) -> ()
  kgen.return
}
