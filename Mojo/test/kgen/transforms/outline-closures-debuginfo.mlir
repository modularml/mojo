// RUN: kgen-opt %s -split-input-file -outline-closures=debug-build=true -mlir-print-debuginfo | FileCheck %s


// CHECK-LABEL: kgen.generator @foo_NestedClosure() -> !pop.array<0, i32> attributes {sourceName = "NestedClosure"} {
// CHECK-NEXT:    %array = kgen.param.constant: array<0, i32> = <[]> loc(#[[LOC_NESTED:loc[0-9]*]])
// CHECK-NEXT:    kgen.return %array : !pop.array<0, i32> loc(#[[LOC_NESTED]])
// CHECK-NEXT:  } loc(#[[LOC_NESTED]])

// CHECK-LABEL: kgen.generator @foo_Closure() -> !pop.array<0, i8> attributes {sourceName = "Closure"} {
// CHECK-NEXT:    kgen.param.declare NestedClosure: () -> !pop.array<0, i32> = <@foo_NestedClosure> loc(#[[LOC_NESTED_DEC:loc[0-9]*]])
// CHECK-NEXT:    %array = kgen.param.constant: array<0, i8> = <[]> loc(#[[LOC_CLOSURE:loc[0-9]*]])
// CHECK-NEXT:    kgen.return %array : !pop.array<0, i8> loc(#[[LOC_CLOSURE]])
// CHECK-NEXT:  } loc(#[[LOC_CLOSURE]])

// CHECK-LABEL: kgen.generator @foo_OtherClosure() always_inline_no_debug attributes {sourceName = "OtherClosure"} {
// CHECK-NEXT:    kgen.return loc(#[[LOC1:.*]])
// CHECK-NEXT:  } loc(#[[LOC1]])

// CHECK-LABEL: kgen.generator @foo_NestedCapturing()
// CHECK-NEXT:    %0 = pop.compiler.global_load "foo_context_var_0"

// CHECK-LABEL: kgen.generator @foo_Capturing()
// CHECK-NEXT:    %0 = pop.compiler.global_load "foo_context_var_1"
// CHECK-NEXT:    pop.compiler.global_store "foo_context_var_0", %0

// CHECK-LABEL: kgen.generator @foo(
// CHECK-SAME:      %[[ARG:.*]]: index
// CHECK-NEXT:    kgen.param.declare Closure: () -> !pop.array<0, i8> = <@foo_Closure> loc(#[[LOC_CLOSURE_DEC:.*]])
// CHECK-NEXT:    kgen.param.declare OtherClosure: () -> () = <@foo_OtherClosure> loc(#[[LOC_FOO:.*]])
// CHECK-NEXT:    pop.compiler.global_store "foo_context_var_1", %[[ARG]] : index loc(#[[LOC_CAP:.*]])
// CHECK-NEXT:    kgen.param.declare Capturing: () capturing -> () = <@foo_Capturing> loc(#[[LOC_CAP]])
// CHECK-NEXT:    %array = kgen.param.constant: array<0, i1> = <[]> loc(#[[LOC_FOO]])
// CHECK-NEXT:    kgen.return %array : !pop.array<0, i1> loc(#[[LOC_FOO]])
// CHECK-NEXT:  } loc(#[[LOC_FOO]])

kgen.generator @foo(%arg0: index) -> !pop.array<0, i1> {
  kgen.param.declare.region Closure = () -> !pop.array<0, i8> {
    kgen.param.declare.region NestedClosure = () -> !pop.array<0, i32> {
      %array_3 = kgen.param.constant: array<0, i32> = <[]> loc(#locNested)
      kgen.return %array_3 : !pop.array<0, i32> loc(#locNested)
    } loc(#locNested)

    %array_2 = kgen.param.constant: array<0, i8> = <[]> loc(#locClosure)
    kgen.return %array_2 : !pop.array<0, i8> loc(#locClosure)
  } loc(#locClosure)

  kgen.param.declare.region OtherClosure = () -> () always_inline_no_debug {
    kgen.return loc(#loc1)
  } loc(#loc1)

  kgen.param.declare.region Capturing = () capturing {
    kgen.param.declare.region NestedCapturing = () capturing -> index {
      kgen.return %arg0 : index loc(#locNestedCap)
    } loc(#locNestedCap)
    kgen.return loc(#locCap)
  } loc(#locCap)

  %array = kgen.param.constant: array<0, i1> = <[]> loc(#locFoo)
  kgen.return %array : !pop.array<0, i1> loc(#locFoo)
} loc(#locFoo)

// CHECK-DAG: #[[LOC1]] = loc("foo.mojo":170:1)
// CHECK-DAG: #[[LOC2:.*]] = loc("foo.mojo":239:5)
// CHECK-DAG: #[[LOC3:.*]] = loc("foo.mojo":242:9)
// CHECK-DAG: #[[LOC4:.*]] = loc("foo.mojo":1473:5)
#loc1 = loc("foo.mojo":170:1)
#loc2 = loc("foo.mojo":239:5)
#loc3 = loc("foo.mojo":242:9)
#loc4 = loc("foo.mojo":1473:5)
#loc5 = loc("foo.mojo":1489:9)

// CHECK-DAG: #[[SP_FOO:.*]] = #debuginfo.subprogram<sourceName = <"foo">
// CHECK-DAG: #[[SP_CLOSURE:.*]] = #debuginfo.subprogram<sourceName = <"Closure">
// CHECK-DAG: #[[SP_NESTED:.*]] = #debuginfo.subprogram<sourceName = <"NestedClosure">
#sp = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#spClosure = #debuginfo.subprogram<sourceName = <"Closure">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#spNested = #debuginfo.subprogram<sourceName = <"NestedClosure">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#spCap = #debuginfo.subprogram<sourceName = <"Capturing">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#spNestedCap = #debuginfo.subprogram<sourceName = <"NestedCapturing">> : !debuginfo.subroutine<() -> (): DW_CC_normal>

// CHECK-DAG: #[[LOC_NESTED]] = loc(fused<#[[SP_NESTED]]>[#[[LOC3]]])
// CHECK-DAG: #[[LOC_CLOSURE]] = loc(fused<#[[SP_CLOSURE]]>[#[[LOC2]]])
// CHECK-DAG: #[[LOC_NESTED_DEC]] = loc(fused<#[[SP_CLOSURE]]>[#[[LOC3]]])
// CHECK-DAG: #[[LOC_FOO]] = loc(fused<#[[SP_FOO]]>[#[[LOC1]]])
// CHECK-DAG: #[[LOC_CLOSURE_DEC]] = loc(fused<#[[SP_FOO]]>[#[[LOC2]]])
// CHECK-DAG: #[[LOC_CAP]] = loc(fused<#[[SP_FOO]]>[#[[LOC4]]])
#locFoo = loc(fused<#sp>[#loc1])
#locClosure = loc(fused<#spClosure>[#loc2])
#locNested = loc(fused<#spNested>[#loc3])
#locCap = loc(fused<#spCap>[#loc4])
#locNestedCap = loc(fused<#spNestedCap>[#loc5])

// -----

// COM: Use of 'a' appears only in a location inside the closure.

// CHECK-LABEL: @outline_closures_ref_closure<a>()
// CHECK-NEXT: kgen.return loc([[LOC:#.*]])

// CHECK-LABEL: @outline_closures_ref<a>
kgen.generator @outline_closures_ref<a>() {
  // CHECK-NEXT: declare closure: () -> () = <@outline_closures_ref_closure<a>>
  kgen.param.declare.region closure = () {
    kgen.return loc(fused<#kgen.param.decl.ref<"a"> : index>["a:0:0"])
  }
  kgen.return
}

// CHECK: [[LOC]] = loc(fused<#kgen.param.decl.ref<"a"> : index>[

// -----

// COM: Fix for MOCO-869:
// COM: decl is defined in the nested scope of `kgen.param.for` which is not at or above current
// COM: `param.declare.region`. It is safe to ignore and should not crash the compiler.
// CHECK-LABEL: @ignore_param_defined_in_nested_non_decl_region_scope()
kgen.generator @ignore_param_defined_in_nested_non_decl_region_scope() {
  kgen.param.declare.region closure = () {
    kgen.param.for decl: index in :index 2
      has_next :(index) -> i1 @wrapper2
      get_next_iter :(index) -> index @wrapper {
      kgen.param.for.continue loc(fused<#kgen.param.decl.ref<"decl"> : index>["x:0"])
    } else {
      kgen.param.yield
    }
    kgen.return
  }
  kgen.return
}

// CHECK: #[[LOC_X:.*]] = loc("x:0")
// CHECK: #[[LOC_CONTINUE:.*]] = loc(fused<#kgen.param.decl.ref<"decl"> : index>[#[[LOC_X]]])

// -----

// COM: Ensure inlined scopes are not fused with kgen.param.op's location

module {
  kgen.generator @toplevel() {
    // CHECK: hlcf.loop "inlined_cf_scope" {
    hlcf.loop "inlined_cf_scope" {
      // CHECK-NEXT: kgen.param.declare _float32_dispatch: () capturing -> !kgen.none = <@toplevel__float32_dispatch> loc([[LOC:#.*]])
      kgen.param.declare.region _float32_dispatch = () capturing -> !kgen.none always_inline_no_debug {
        %none = kgen.param.constant: none = <#kgen.none> loc(#loc4)
        kgen.return %none : !kgen.none loc(#loc4)
      } {isolated} loc(#loc4)
      kgen.unreachable loc(#loc8)
    } loc(#loc8)
    kgen.unreachable loc(#loc7)
  } loc(#loc5)
} loc(#loc)
// CHECK-DAG: [[LOC1:#.*]] = loc("delete-me.mojo":28:8)
// CHECK-DAG: #subprogram = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, sourceName = <"CALLER">
// CHECK-DAG: [[LOC]] = loc(fused<#subprogram>[[[LOC1]]])
!subroutine = !debuginfo.subroutine<() -> (): DW_CC_normal>
#file = #debuginfo.file<"delete-me.mojo" in "">
#loc = loc("this.mlir":1:1)
#loc1 = loc("delete-me.mojo":29:5)
#loc2 = loc("delete-me.mojo":26:36)
#loc3 = loc("delete-me.mojo":31:41)
#loc4 = loc("delete-me.mojo":28:8)
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #file, producer = "Mojo", isOptimized = true, emissionKind = Full, nameTableKind = None>
#subprogram = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, sourceName = <"CALLER">, linkageName = "CALLER", file = #file, line = 29, scopeLine = 29, subprogramFlags = "Definition|Optimized"> : !subroutine
#subprogram1 = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, sourceName = <"CALLEE">, linkageName = "CALLEE", file = #file, line = 23, scopeLine = 23, subprogramFlags = "Definition|Optimized"> : !subroutine
#loc5 = loc(fused<#subprogram>[#loc1])
#loc6 = loc(fused<#subprogram1>[#loc2])
#loc7 = loc(fused<#subprogram>[#loc3])
#loc8 = loc(callsite(#loc6 at #loc7))
