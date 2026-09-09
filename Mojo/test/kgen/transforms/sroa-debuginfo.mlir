
// RUN: kgen-opt -sroa -split-input-file %s | FileCheck %s

!subroutine = !debuginfo.subroutine<() -> (): DW_CC_normal>
!member0 = !debuginfo.member<first: index>
!member1 = !debuginfo.member<second: index>
!struct = !debuginfo.struct<Foo(!member0, !member1)>
!ptr = !debuginfo.ti.ptr<!struct>
#subprogram = #debuginfo.subprogram<sourceName = <"__next__">> : !subroutine
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "self"> : !ptr

#fileLoc = loc("foo.mlir":0:0)
#loc = loc(fused<#subprogram>[#fileLoc])

// CHECK-DAG: ![[STRUCT:.*]] = !debuginfo.struct<Foo(!{{.*}}, !{{.*}})>
// CHECK-DAG: ![[STRUCT_PTR:.*]] = !debuginfo.ti.ptr<![[STRUCT]]>

// CHECK-DAG: #[[IRVAL:.*]] = #debuginfo.expr.irvalue : !kgen.pointer<index>
// CHECK-DAG: #[[DEREF:.*]] = #debuginfo.expr.deref<#[[IRVAL]]> : index
// CHECK-DAG: #[[AGG0:.*]] = #debuginfo.expr.agg<#[[DEREF]], 0> : !kgen.struct<(index, index)>
// CHECK-DAG: #[[AGG1:.*]] = #debuginfo.expr.agg<#[[DEREF]], 1> : !kgen.struct<(index, index)>
// CHECK-DAG: #[[REF0:.*]] = #debuginfo.expr.refof<#[[AGG0]]> : !kgen.pointer<struct<(index, index)>>
// CHECK-DAG: #[[REF1:.*]] = #debuginfo.expr.refof<#[[AGG1]]> : !kgen.pointer<struct<(index, index)>>

// CHECK-DAG: #[[VAR:.*]] = #debuginfo.local_variable<{{.*}}, name = "self"> : ![[STRUCT_PTR]]

// CHECK-LABEL: @sroa_valueop
kgen.func @sroa_valueop() {
  // CHECK-NEXT: %0 = pop.stack_allocation 1 x index
  // CHECK-NEXT: %1 = pop.stack_allocation 1 x index
  %0 = pop.stack_allocation 1 x !kgen.struct<(index, index)> loc(#loc)
  // CHECK-NEXT: debuginfo.value #[[VAR]] #[[REF0]] = %0 : !kgen.pointer<index>
  // CHECK-NEXT: debuginfo.value #[[VAR]] #[[REF1]] = %1 : !kgen.pointer<index>
  debuginfo.value #local_variable = %0 : !kgen.pointer<struct<(index, index)>> loc(#loc)
  // CHECK-NEXT: kgen.return
  kgen.return loc(#loc)
} loc(#loc)

// -----

#sp = #debuginfo.subprogram<sourceName = <"max">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
!member0 = !debuginfo.member<first: index>
!member1 = !debuginfo.member<second: index>
!struct = !debuginfo.struct<Foo(!member0, !member1)>
#local_variable = #debuginfo.local_variable<scope = #sp, name = "x"> : !struct

#loc = loc(fused<#sp>["foo.mojo":0:0])

// CHECK-DAG: ![[STRUCT:.*]] = !debuginfo.struct<Foo(!{{.*}}, !{{.*}})>

// CHECK-DAG: #[[IRVAL:.*]] = #debuginfo.expr.irvalue : index
// CHECK-DAG: #[[AGG0:.*]] = #debuginfo.expr.agg<#[[IRVAL]], 0> : !kgen.struct<(index, index)>
// CHECK-DAG: #[[AGG1:.*]] = #debuginfo.expr.agg<#[[IRVAL]], 1> : !kgen.struct<(index, index)>

// CHECK-DAG: #[[VAR:.*]] = #debuginfo.local_variable<{{.*}}, name = "x"> : ![[STRUCT]]

// CHECK-LABEL: @load_debug_var
kgen.func @load_debug_var(%arg0: !kgen.struct<(index, index)>) {
  // CHECK-COUNT-2: pop.stack_allocation 1 x index
  %0 = pop.stack_allocation 1 x struct<(index, index)> loc(#loc)
  pop.store %arg0, %0 : !kgen.pointer<struct<(index, index)>> loc(#loc)
  %1 = pop.load %0 : !kgen.pointer<struct<(index, index)>> loc(#loc)
  // CHECK: [[VALUE0:%.*]] = pop.load
  // CHECK-NEXT: debuginfo.value #[[VAR]] #[[AGG0]] = [[VALUE0]]
  // CHECK: [[VALUE1:%.*]] = pop.load
  // CHECK-NEXT: debuginfo.value #[[VAR]] #[[AGG1]] = [[VALUE1]]
  debuginfo.value #local_variable = %1 : !kgen.struct<(index, index)> loc(#loc)
  kgen.return loc(#loc)
} loc(#loc)

// -----

// SROA of an *array* alloc must likewise split a `debuginfo.value` describing
// the whole array into one per element. This used to be handled only for
// structs, so under `--debug-level full` the array's `debuginfo.value` user
// blocked array SROA entirely (leaving `memoryOnly` tile arrays undecomposed).

!subroutine = !debuginfo.subroutine<() -> (): DW_CC_normal>
!element = !debuginfo.member<e: index>
!array = !debuginfo.array<2 x !element>
!ptr = !debuginfo.ti.ptr<!array>
#subprogram = #debuginfo.subprogram<sourceName = <"kernel">> : !subroutine
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "a"> : !ptr

#fileLoc = loc("arr.mlir":0:0)
#loc = loc(fused<#subprogram>[#fileLoc])

// CHECK-DAG: #[[IRVAL:.*]] = #debuginfo.expr.irvalue : !kgen.pointer<index>
// CHECK-DAG: #[[DEREF:.*]] = #debuginfo.expr.deref<#[[IRVAL]]> : index
// CHECK-DAG: #[[AGG0:.*]] = #debuginfo.expr.agg<#[[DEREF]], 0> : !pop.array<2, index>
// CHECK-DAG: #[[AGG1:.*]] = #debuginfo.expr.agg<#[[DEREF]], 1> : !pop.array<2, index>
// CHECK-DAG: #[[REF0:.*]] = #debuginfo.expr.refof<#[[AGG0]]> : !kgen.pointer<array<2, index>>
// CHECK-DAG: #[[REF1:.*]] = #debuginfo.expr.refof<#[[AGG1]]> : !kgen.pointer<array<2, index>>

// CHECK-DAG: #[[VAR:.*]] = #debuginfo.local_variable<{{.*}}, name = "a"> : !{{.*}}

// CHECK-LABEL: @sroa_array_valueop
kgen.func @sroa_array_valueop() {
  // CHECK-NEXT: %0 = pop.stack_allocation 1 x index
  // CHECK-NEXT: %1 = pop.stack_allocation 1 x index
  %0 = pop.stack_allocation 1 x !pop.array<2, index> loc(#loc)
  // CHECK-NEXT: debuginfo.value #[[VAR]] #[[REF0]] = %0 : !kgen.pointer<index>
  // CHECK-NEXT: debuginfo.value #[[VAR]] #[[REF1]] = %1 : !kgen.pointer<index>
  debuginfo.value #local_variable = %0 : !kgen.pointer<array<2, index>> loc(#loc)
  // CHECK-NEXT: kgen.return
  kgen.return loc(#loc)
} loc(#loc)

// -----

// Nested SROA (array of structs) with an indirect `debuginfo.value`: the array
// splits into structs and the structs into scalars, and the whole-array DI
// value is re-expressed per leaf via nested (array-of-struct) AggregatesInto
// exprs. This is the shape of the tile pools that motivated the fix.

#sp = #debuginfo.subprogram<sourceName = <"k">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
!member = !debuginfo.member<e: index>
!struct = !debuginfo.struct<A(!member, !member)>
!array = !debuginfo.array<2 x !struct>
!ptr = !debuginfo.ti.ptr<!array>
#local_variable = #debuginfo.local_variable<scope = #sp, name = "x"> : !ptr
#loc = loc(fused<#sp>["f.mojo":0:0])

// CHECK-DAG: #[[DEREF:.*]] = #debuginfo.expr.deref<{{.*}}> : index
// The struct field is aggregated first, then wrapped in an array-element agg.
// CHECK-DAG: #[[SAGG0:.*]] = #debuginfo.expr.agg<#[[DEREF]], 0> : !kgen.struct<(index, index)>
// CHECK-DAG: #[[SAGG1:.*]] = #debuginfo.expr.agg<#[[DEREF]], 1> : !kgen.struct<(index, index)>
// CHECK-DAG: #debuginfo.expr.agg<#[[SAGG0]], 0> : !pop.array<2, struct<(index, index)>>
// CHECK-DAG: #debuginfo.expr.agg<#[[SAGG1]], 0> : !pop.array<2, struct<(index, index)>>
// CHECK-DAG: #debuginfo.expr.agg<#[[SAGG0]], 1> : !pop.array<2, struct<(index, index)>>
// CHECK-DAG: #debuginfo.expr.agg<#[[SAGG1]], 1> : !pop.array<2, struct<(index, index)>>

// CHECK-LABEL: @sroa_array_of_struct_valueop
kgen.func @sroa_array_of_struct_valueop() {
  // 2 array elements x 2 struct fields = 4 scalar allocations.
  // CHECK-COUNT-4: pop.stack_allocation 1 x index
  %0 = pop.stack_allocation 1 x !pop.array<2, struct<(index, index)>> loc(#loc)
  // CHECK-COUNT-4: debuginfo.value
  debuginfo.value #local_variable = %0
      : !kgen.pointer<array<2, struct<(index, index)>>> loc(#loc)
  // CHECK: kgen.return
  kgen.return loc(#loc)
} loc(#loc)

// -----

// Direct `debuginfo.value` on a *loaded* array value: the array still splits
// into scalars, and the loaded aggregate is rematerialized (`pop.array.create`)
// with the DI value retargeted onto it.

#sp = #debuginfo.subprogram<sourceName = <"k">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
!member = !debuginfo.member<e: index>
!array = !debuginfo.array<2 x !member>
#local_variable = #debuginfo.local_variable<scope = #sp, name = "x"> : !array
#loc = loc(fused<#sp>["f.mojo":0:0])

// CHECK-LABEL: @sroa_array_load_debug_var
kgen.func @sroa_array_load_debug_var(%arg0: !pop.array<2, index>) {
  // CHECK-COUNT-2: pop.stack_allocation 1 x index
  %0 = pop.stack_allocation 1 x !pop.array<2, index> loc(#loc)
  pop.store %arg0, %0 : !kgen.pointer<array<2, index>> loc(#loc)
  %1 = pop.load %0 : !kgen.pointer<array<2, index>> loc(#loc)
  // CHECK: [[ARR:%.*]] = pop.array.create
  // CHECK-NEXT: debuginfo.value #{{.*}} = [[ARR]] : !pop.array<2, index>
  debuginfo.value #local_variable = %1 : !pop.array<2, index> loc(#loc)
  // CHECK: kgen.return
  kgen.return loc(#loc)
} loc(#loc)

// -----

// A `debuginfo.value` coexisting with real element accesses: the array still
// fully decomposes -- the DI value splits per element and the constant-index
// `pop.array.gep` + load fold onto the matching scalar alloc.

#sp = #debuginfo.subprogram<sourceName = <"k">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
!member = !debuginfo.member<e: index>
!array = !debuginfo.array<2 x !member>
!ptr = !debuginfo.ti.ptr<!array>
#local_variable = #debuginfo.local_variable<scope = #sp, name = "x"> : !ptr
#loc = loc(fused<#sp>["f.mojo":0:0])

// CHECK-LABEL: @sroa_array_di_and_use
kgen.func @sroa_array_di_and_use(%arg0: !pop.array<2, index>) -> index {
  %c1 = kgen.param.constant = <1>
  // CHECK-COUNT-2: pop.stack_allocation 1 x index
  // CHECK-NOT: pop.stack_allocation 1 x !pop.array
  %0 = pop.stack_allocation 1 x !pop.array<2, index> loc(#loc)
  pop.store %arg0, %0 : !kgen.pointer<array<2, index>> loc(#loc)
  // CHECK-COUNT-2: debuginfo.value {{.*}} : !kgen.pointer<index>
  debuginfo.value #local_variable = %0 : !kgen.pointer<array<2, index>> loc(#loc)
  %1 = pop.array.gep %0[%c1] : <array<2, index>> loc(#loc)
  // CHECK: [[V:%.*]] = pop.load {{.*}} : !kgen.pointer<index>
  %2 = pop.load %1 : !kgen.pointer<index> loc(#loc)
  // CHECK: kgen.return [[V]]
  kgen.return %2 : index loc(#loc)
} loc(#loc)

// -----

// Single *element* store/load with a `debuginfo.value` on the loaded element
// (by-value, `: index`). The array fully decomposes, the constant-index
// `pop.array.gep` folds onto the matching scalar alloc, and the element's DI
// record follows the scalar load. Contrast with @sroa_array_load_debug_var,
// which describes the whole loaded array.

#sp = #debuginfo.subprogram<sourceName = <"k">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
!index = !debuginfo.basic<index {sizeInBits = 64, alignInBits = 64, encoding = DW_ATE_signed}>
#local_variable = #debuginfo.local_variable<scope = #sp, name = "e"> : !index
#loc = loc(fused<#sp>["f.mojo":0:0])

// CHECK-LABEL: @sroa_array_element_debug_var
kgen.func @sroa_array_element_debug_var(%arg0: index) -> index {
  %c1 = kgen.param.constant = <1>
  // CHECK-COUNT-2: pop.stack_allocation 1 x index
  // CHECK-NOT: pop.stack_allocation 1 x !pop.array
  %0 = pop.stack_allocation 1 x !pop.array<2, index> loc(#loc)
  %1 = pop.array.gep %0[%c1] : <array<2, index>> loc(#loc)
  // CHECK: pop.store %arg0, [[E:%.*]] : !kgen.pointer<index>
  pop.store %arg0, %1 : !kgen.pointer<index> loc(#loc)
  // CHECK: [[V:%.*]] = pop.load [[E]] : !kgen.pointer<index>
  %2 = pop.load %1 : !kgen.pointer<index> loc(#loc)
  // CHECK: debuginfo.value #{{.*}} = [[V]] : index
  debuginfo.value #local_variable = %2 : index loc(#loc)
  // CHECK: kgen.return [[V]]
  kgen.return %2 : index loc(#loc)
} loc(#loc)
