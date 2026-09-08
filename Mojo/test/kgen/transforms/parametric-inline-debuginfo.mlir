// RUN: kgen-opt -mlir-print-debuginfo -inline-param=optimization-level=3 -verify-parameters -split-input-file %s | FileCheck %s

#subprogram = #debuginfo.subprogram<sourceName = <"parent">> : !debuginfo.subroutine<() -> (index): DW_CC_normal>
#subprogram1 = #debuginfo.subprogram<sourceName = <"callee">> : !debuginfo.subroutine<() -> (index): DW_CC_normal>
#loc = loc("foo.mlir":1:1)
#loc1 = loc("foo.mlir":10:10)
#loc2 = loc("foo.mlir":20:20)
#loc3 = loc("foo.mlir":30:30)
#loc4 = loc("foo.mlir":40:40)
#loc5 = loc("foo.mlir":50:50)
#loc6 = loc("foo.mlir":60:60)
#loc7 = loc("foo.mlir":70:70)
#loc8 = loc("foo.mlir":80:80)
#parentLoc = loc(fused<#subprogram>[#loc])
#callOpLoc = loc(fused<#subprogram>[#loc1])
#parentRetLoc = loc(fused<#subprogram>[#loc2])
#calleeLoc = loc(fused<#subprogram1>[#loc3])
#ret0 = loc(fused<#subprogram1>[#loc4])
#ret1 = loc(fused<#subprogram1>[#loc5])
#closure = loc(fused<#subprogram1>[#loc6])
#closureRet = loc(fused<#subprogram1>[#loc7])
#calleeMisc = loc(fused<#subprogram1>[#loc8])

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent() -> index {
  // CHECK: %[[RES:.*]] = hlcf.loop "[[LABEL:.*]]" () -> index
    // CHECK-NEXT: index.constant 0 loc(#[[CONST_LOC:.*]])
    // CHECK: hlcf.if
      // CHECK-NEXT: hlcf.break "[[LABEL]]" %idx0 : index loc(#[[BREAK_LOC0:.*]])
    // CHECK: kgen.param.declare.region SomeClosure = () {
      // CHECK-NEXT: kgen.return loc(#[[CL_RET_LOC:.*]])
    // CHECK-NEXT: } {{.*}} loc(#[[CL_LOC:.*]])
    // CHECK: hlcf.break "[[LABEL]]" %idx0 : index loc(#[[BREAK_LOC1:.*]])
  // CHECK-NEXT: } loc(#[[CALL_LOC:.*]])
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee() : () -> index loc(#callOpLoc)
  // CHECK: return %[[RES]]
  kgen.return %0 : index loc(#parentRetLoc)
} loc(#parentLoc)

// CHECK: kgen.generator @callee
kgen.generator @callee() -> index always_inline {
  // CHECK: index.constant 0 loc(#[[CALLEE_LOC:.*]])
  %0 = index.constant 0 loc(#calleeMisc)
  %false = kgen.param.constant: scalar<bool> = <false> loc(#calleeMisc)
  hlcf.if %false {
    // CHECK: kgen.return %idx0 : index loc(#[[RET_LOC0:.*]])
    kgen.return %0 : index loc(#ret0)
  } else {
    hlcf.yield loc(#calleeMisc)
  } loc(#calleeMisc)
  // CHECK: kgen.param.declare.region SomeClosure = () {
    // CHECK-NEXT: kgen.return loc(#[[CL_RET_LOC]])
  // CHECK-NEXT: } {{.*}} loc(#[[CL_LOC]])
  kgen.param.declare.region SomeClosure = () -> () {
    kgen.return loc(#closureRet)
  } loc(#closure)
  // CHECK: kgen.return %idx0 : index loc(#[[RET_LOC1:.*]])
  kgen.return %0 : index loc(#ret1)
} loc(#calleeLoc)

// CHECK: #[[CONST_LOC]] = loc("foo.mlir":80:80)
// CHECK: #[[BREAK_LOC0]] = loc(callsite(#[[RET_LOC0]] at #[[CALL_LOC]]))
// CHECK: #[[BREAK_LOC1]] = loc(callsite(#[[RET_LOC1]] at #[[CALL_LOC]]))

// -----

#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<(!debuginfo.unresolved<!kgen.param<T>>) -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "foo"> : !debuginfo.unresolved<!kgen.param<T>>

#fileLoc = loc("foo.mlir":0:0)
#loc = loc(fused<#subprogram>[#fileLoc])

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<T: type>(%arg0: index) {
  // CHECK: kgen.param.declare T0: type = <index> loc(#[[CALL_LOC:.*]])
  // CHECK-NEXT: kgen.rebind %arg0 : index to !kgen.param<T0> loc(#[[CALL_LOC]])
  // CHECK-NEXT: kgen.return
  kgen.call @nodebug_inline_me<:type index>(%arg0) : (index) -> () loc(#loc)
  kgen.return loc(#loc)
} loc(#loc)

// CHECK-LABEL: kgen.generator @nodebug_inline_me
kgen.generator @nodebug_inline_me<T: type>(%arg0: !kgen.param<T>) always_inline_no_debug {
  kgen.return loc(#loc)
} loc(#loc)

// -----

// CHECK-LABEL: kgen.generator @foo
kgen.generator @foo() {
  // CHECK: kgen.param.declare.region SomeClosure = <DT: dtype, N>(%[[ARG:.*]]: !kgen.simd<N, DT>
  // CHECK-NEXT: kgen.param.declare A = <1> loc(#[[LOC:.*]])
  // CHECK-NEXT: kgen.return %[[ARG]]
  kgen.call @bar() : () -> ()
  // CHECK: kgen.param.declare.region SomeClosure0[SomeClosure] = <DT0: dtype, N0>(%[[ARG0:.*]]: !kgen.simd<N0, DT0>
  // CHECK-NEXT: kgen.param.declare A0 = <1> loc(#[[LOC0:.*]])
  // CHECK-NEXT: kgen.return %[[ARG0]] : !kgen.simd<N0, DT0> loc(#[[LOC0]])
  kgen.call @bar() : () -> ()
  kgen.return
}
kgen.generator @bar() always_inline {
  kgen.param.declare.region SomeClosure = <DT: dtype, N>(%arg0: !kgen.simd<N, DT>) capturing -> !kgen.simd<N, DT> {
    kgen.param.declare A = <1> loc(#loc)
    kgen.return %arg0 : !kgen.simd<N, DT> loc(#loc)
  } loc(#loc)
  kgen.return
}

// CHECK-DAG: ![[M:.*]] = !debuginfo.member<value: !kgen.simd<N, DT>>
// CHECK-DAG: ![[M0:.*]] = !debuginfo.member<value: !kgen.simd<N0, DT0>>
// CHECK-DAG: ![[STR:.*]] = !debuginfo.struct<"builtin::$simd::SIMD"(![[M]])>
// CHECK-DAG: ![[STR0:.*]] = !debuginfo.struct<"builtin::$simd::SIMD"(![[M0]])>
// CHECK-DAG: ![[SR:.*]] = !debuginfo.subroutine<(![[STR]]) -> (![[STR]]): DW_CC_normal>
// CHECK-DAG: ![[SR0:.*]] = !debuginfo.subroutine<(![[STR0]]) -> (![[STR0]]): DW_CC_normal>
// CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<{{.*}}, sourceName = <"SomeClosure">, linkageName = "SomeClosure", file = #file, line = 1314, scopeLine = 1314, subprogramFlags = "Definition|Optimized"> : ![[SR]]
// CHECK-DAG: #[[SP0:.*]] = #debuginfo.subprogram<{{.*}}, sourceName = <"SomeClosure">, linkageName = "SomeClosure", file = #file, line = 1314, scopeLine = 1314, subprogramFlags = "Definition|Optimized"> : ![[SR0]]

!struct = !debuginfo.struct<"builtin::$simd::SIMD"(!debuginfo.member<value: !kgen.simd<N, DT>>)>
#file = #debuginfo.file<"foo.mlir" in "/">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #file, producer = "Mojo", isOptimized = true, emissionKind = Full>
#subprogram2 = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, sourceName = <"SomeClosure">, linkageName = "SomeClosure", file = #file, line = 1314, scopeLine = 1314, subprogramFlags = "Definition|Optimized"> : !debuginfo.subroutine<(!struct) -> (!struct): DW_CC_normal>

// CHECK-DAG: #[[LOC_ORI:.*]] = loc("foo.mlir":1317:13)
// CHECK-DAG: #[[LOC]] = loc(fused<#[[SP]]>[#[[LOC_ORI]]])
// CHECK-DAG: #[[LOC0]] = loc(fused<#[[SP0]]>[#[[LOC_ORI]]])
#loc = loc(fused<#subprogram2>["foo.mlir":1317:13])

// -----

#file = #debuginfo.file<"foo.c" in "/mlir/">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #file, producer = "MLIR", isOptimized = true, emissionKind = Full>
#subprogram = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, sourceName = <"foo">, linkageName = "foo", file = #file, line = 10, scopeLine = 10, subprogramFlags = Definition> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#loc = loc(fused<#subprogram>["foo.mlir":0:0])

kgen.generator @no_debuginfo() -> index always_inline {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.generator @has_debuginfo
kgen.generator @has_debuginfo() {
  // CHECK: index.constant 0 loc([[LOC:#.*]])
  kgen.call @no_debuginfo() : () -> index loc(#loc)
  kgen.return loc(#loc)
} loc(#loc)

// CHECK: [[LOC]] = loc("{{.*}}":

// -----

// CHECK-DAG: ![[M:.*]] = !debuginfo.member<value: !kgen.simd<N, DT>>
// CHECK-DAG: ![[M0:.*]] = !debuginfo.member<value: !kgen.simd<N, DT0>>
// CHECK-DAG: ![[STR:.*]] = !debuginfo.struct<"builtin::$simd::SIMD"(![[M]])>
// CHECK-DAG: ![[STR0:.*]] = !debuginfo.struct<"builtin::$simd::SIMD"(![[M0]])>
// CHECK-DAG: ![[SR:.*]] = !debuginfo.subroutine<(![[STR]]) -> (![[STR]]): DW_CC_normal>
// CHECK-DAG: ![[SR0:.*]] = !debuginfo.subroutine<(![[STR0]]) -> (![[STR0]]): DW_CC_normal>
// CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<{{.*}}, sourceName = <"SomeClosure">, linkageName = "SomeClosure", file = #file, line = 1314, scopeLine = 1314, subprogramFlags = "Definition|Optimized"> : ![[SR]]
// CHECK-DAG: #[[SP0:.*]] = #debuginfo.subprogram<{{.*}}, sourceName = <"SomeClosure">, linkageName = "SomeClosure", file = #file, line = 1314, scopeLine = 1314, subprogramFlags = "Definition|Optimized"> : ![[SR0]]

!struct = !debuginfo.struct<"builtin::$simd::SIMD"(!debuginfo.member<value: !kgen.simd<N, DT>>)>
#file = #debuginfo.file<"foo.mlir" in "/">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #file, producer = "Mojo", isOptimized = true, emissionKind = Full>
#subprogram2 = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, sourceName = <"SomeClosure">, linkageName = "SomeClosure", file = #file, line = 1314, scopeLine = 1314, subprogramFlags = "Definition|Optimized"> : !debuginfo.subroutine<(!struct) -> (!struct): DW_CC_normal>

// CHECK-DAG: #[[LOC_ORI:.*]] = loc("foo.mlir":1317:13)
// CHECK-DAG: #[[LOC0:.*]] = loc(fused<#[[SP0]]>[#[[LOC_ORI]]])
#loc = loc(fused<#subprogram2>["foo.mlir":1317:13])

// CHECK-LABEL: kgen.generator @foo
kgen.generator @foo<DT>() {
  // CHECK-NEXT: kgen.param.declare.region
  // CHECK-NEXT:   hlcf.loop
  // CHECK-NEXT:     kgen.param.declare A = <1> loc(#[[LOC0]])
  // CHECK:        kgen.param.for
  // CHECK-NEXT: has_next
  // CHECK-NEXT: get_next_iter
  // CHECK-SAME:     (%arg1 loc(fused<#[[SP0]]>[#[[LOC_ORI]]]) = %arg0 : !kgen.simd<N, DT0>) -> !kgen.simd<N, DT0>
  // CHECK:        } else (%arg1: !kgen.simd<N, DT0> loc(fused<#[[SP0]]>[#[[LOC_ORI]]]))
  kgen.call @bar() : () -> ()
  kgen.return
}

kgen.generator @bar() always_inline {
  kgen.param.declare.region SomeClosure = <DT: dtype, N>(%arg0: !kgen.simd<N, DT>) capturing -> !kgen.simd<N, DT> {
    hlcf.loop {
      kgen.param.declare A = <1> loc(#loc)
      hlcf.break loc(#loc)
    } loc(#loc)

    %0 = kgen.param.for I in ?
      has_next :() -> i1 ?
      get_next_iter :() -> () ?
    (%arg1 loc(#loc) = %arg0 : !kgen.simd<N, DT>) -> !kgen.simd<N, DT> {
      kgen.param.yield %arg1 : !kgen.simd<N, DT> loc(#loc)
    } else (%arg1 : !kgen.simd<N, DT> loc(#loc)) {
      kgen.param.yield %arg1 : !kgen.simd<N, DT> loc(#loc)
    } loc(#loc)

    kgen.return %0 : !kgen.simd<N, DT> loc(#loc)
  } loc(#loc)
  kgen.return
}
