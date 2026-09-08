// RUN: kgen-opt %s -split-input-file -verify-parameters -elaborate-generators="use-parametric-interpret=false" -allow-unregistered-dialect | FileCheck --check-prefixes=CHECK,CHECK-MAIN %s
// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=true" -allow-unregistered-dialect | FileCheck --check-prefixes=CHECK %s

kgen.generator @recursive(%arg0: index) -> index {
  %idx1 = index.constant 1
  %0 = index.cmp sge(%arg0, %idx1)
  %c0 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
  hlcf.if %c0 {
    %1 = index.sub %arg0, %idx1
    %2 = kgen.call @recursive(%1) : (index) -> index
    kgen.return %2 : index
  } else {
    hlcf.yield
  }
  kgen.return %idx1 : index
}

// CHECK-LABEL: kgen.func export @recursive_return_after_call
kgen.generator export @recursive_return_after_call() {
  kgen.param.apply x = [(index) -> index: @recursive](5)
  // CHECK-NEXT: <1>
  kgen.param.constant = <x>
  kgen.return
}

// -----

kgen.generator @fma(%arg0: index, %arg1: index, %arg2: index) -> index {
  %0 = index.mul %arg1, %arg2
  %1 = index.add %0, %arg0
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.func export @constexpr_fma
kgen.generator export @constexpr_fma() -> index {
  // CHECK-NEXT: kgen.param.constant = <7>
  %0 = kgen.param.constant = <apply(:(index, index, index) -> index @fma, 1, 2, 3)>
  kgen.return %0 : index
}

kgen.generator @byref_result(%arg0: !kgen.pointer<index>, %arg1: !kgen.pointer<index> byref_result) {
  %0 = pop.load %arg0 : !kgen.pointer<index>
  %idx2 = index.constant 2
  %1 = index.mul %idx2, %0
  pop.store %1, %arg1 : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func export @top
kgen.generator export @top() {
  // CHECK-NEXT: kgen.param.constant = <2048>
  kgen.param.declare value = <1024>
  kgen.param.constant = <apply_result_slot(:(!kgen.pointer<index>, !kgen.pointer<index> byref_result) -> () @byref_result, store_to_mem(value))>
  kgen.return
}

// -----

kgen.generator @alloc_load_store(%arg0: index) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2
  %idx3 = index.constant 3

  %p0 = pop.stack_allocation 4 x index
  pop.store %idx0, %p0 : !kgen.pointer<index>
  %p1 = pop.offset %p0[%idx1] : !kgen.pointer<index>
  pop.store %idx1, %p1 : !kgen.pointer<index>
  %p2 = pop.offset %p0[%idx2] : !kgen.pointer<index>
  pop.store %idx2, %p2 : !kgen.pointer<index>
  %p3 = pop.offset %p1[%idx2] : !kgen.pointer<index>
  pop.store %idx3, %p3 : !kgen.pointer<index>

  %ptr = pop.offset %p0[%arg0] : !kgen.pointer<index>
  %result = pop.load %ptr : !kgen.pointer<index>
  kgen.return %result : index
}

// CHECK-LABEL: kgen.func @constexpr_load_store
kgen.generator @constexpr_load_store() {
  // CHECK-NEXT: = <0>
  %0 = kgen.param.constant = <apply(:(index) -> index @alloc_load_store, 0)>
  // CHECK-NEXT: = <1>
  %1 = kgen.param.constant = <apply(:(index) -> index @alloc_load_store, 1)>
  // CHECK-NEXT: = <2>
  %2 = kgen.param.constant = <apply(:(index) -> index @alloc_load_store, 2)>
  // CHECK-NEXT: = <3>
  %3 = kgen.param.constant = <apply(:(index) -> index @alloc_load_store, 3)>
  kgen.return
}

// -----

kgen.generator @return_it<A>() -> index {
  %0 = kgen.param.constant = <A>
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @call_it
kgen.generator @call_it() {
  // CHECK-NEXT: <1>
  kgen.param.constant = <apply(:() -> index bind_params(:<index>() -> index @return_it, :index 1))>
  // CHECK-NEXT: <2>
  kgen.param.constant = <apply(:() -> index bind_params(:<index>() -> index @return_it, :index 2))>
  // CHECK-NEXT: <3>
  kgen.param.constant = <apply(:() -> index bind_params(:<index>() -> index @return_it,
    :index apply(:() -> index bind_params(:<index>() -> index @return_it, 3))))>
  kgen.return
}

// -----

kgen.generator @callee(%arg0: index) -> index {
  %0 = index.add %arg0, %arg0
  kgen.return %0 : index
}

kgen.generator @func(%arg0: index) -> index {
  %0 = kgen.call @callee(%arg0) : (index) -> index
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @call_it
kgen.generator @call_it() -> index {
  // CHECK-NEXT: <14>
  %0 = kgen.param.constant = <apply(:(index) -> index @func, 7)>
  kgen.return %0 : index
}

// -----

kgen.generator @sum(%from: index, %to: index) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %result = hlcf.loop (%acc = %idx0 : index, %i = %from : index) -> index {
    %cond = index.cmp sle(%i, %to)
    %condb = pop.cast_from_builtin %cond : i1 to !kgen.scalar<bool>
    hlcf.if %condb {
      hlcf.yield
    } else {
      hlcf.break %acc : index
    }
    %nextI = index.add %idx1, %i
    %nextAcc = index.add %acc, %i
    hlcf.continue %nextAcc, %nextI : index, index
  }
  kgen.return %result : index
}

// CHECK-LABEL: kgen.func @call_it
kgen.generator @call_it() {
  // CHECK-NEXT: <55>
  kgen.param.constant = <apply(:(index, index) -> index @sum, 0, 10)>
  kgen.return
}

// -----

kgen.generator @early_return(%cond: i1) -> index {
  %idx0 = index.constant 0
  %condb = pop.cast_from_builtin %cond : i1 to !kgen.scalar<bool>
  %result = hlcf.if %condb -> index {
    hlcf.yield %idx0 : index
  } else {
    %idx1 = index.constant 1
    kgen.return %idx1 : index
  }
  kgen.return %result : index
}

// CHECK-LABEL: kgen.func @call_it
kgen.generator @call_it() {
  // CHECK-NEXT: <1>
  kgen.param.constant = <apply(:(i1) -> index @early_return, 0)>
  // CHECK-NEXT: <0>
  kgen.param.constant = <apply(:(i1) -> index @early_return, 1)>
  kgen.return
}

// CHECK-MAIN-LABEL: kgen.func @"rebind_value,dtype=ui8"
kgen.generator @rebind_value<dtype: dtype>(%a: !kgen.scalar<ui8>) -> !kgen.scalar<dtype> {
  // CHECK-MAIN-NEXT: return %arg0 : !kgen.scalar<ui8>
  %result = kgen.rebind %a : !kgen.scalar<ui8> to !kgen.scalar<dtype>
  kgen.return %result : !kgen.scalar<dtype>
}

// CHECK-LABEL: kgen.func @rebind_it
kgen.generator @rebind_it() {
  // CHECK-NEXT: constant: scalar<ui8> = <4>
  kgen.param.declare Fn: (!kgen.scalar<ui8>) -> !kgen.scalar<ui8> =
    <bind_params(:<dtype>(!kgen.scalar<ui8>) -> !kgen.scalar<*(0,0)> @rebind_value, :dtype ui8)>
  kgen.param.constant: scalar<ui8> = <apply(:(!kgen.scalar<ui8>) -> !kgen.scalar<ui8> Fn, <4>)>
  kgen.return
}

// -----

kgen.generator @box(%a: index) -> !kgen.struct<(index)> {
  %0 = kgen.struct.create(%a) : !kgen.struct<(index)>
  kgen.return %0 : !kgen.struct<(index)>
}

kgen.generator @unbox(%a: !kgen.struct<(index)>) -> index {
  %0 = kgen.struct.extract %a[0] : !kgen.struct<(index)>
  kgen.return %0 : index
}

kgen.generator @callee<a: !kgen.struct<(index)>>(
    %a: !pop.array<apply(:(!kgen.struct<(index)>) -> index @unbox, a), index>) {
  kgen.return
}

// CHECK-LABEL: kgen.func @unbox_in_result_sig
kgen.generator @unbox_in_result_sig() {
  // CHECK-NEXT: kgen.create_closure[(!pop.array<2, index>) -> (): @"callee,a={ 2 }"]()
  kgen.param.declare a = <2>
  kgen.param.declare fn: <!kgen.struct<(index)>>(
    !pop.array<apply(:(!kgen.struct<(index)>) -> index @unbox, *(0,0)), index>
  ) -> () = <@callee>
  kgen.create_closure[(
    !pop.array<apply(:(!kgen.struct<(index)>) -> index @unbox,
                     apply(:(index) -> !kgen.struct<(index)> @box, a)),
               index>) -> ():
    bind_params(:<!kgen.struct<(index)>>(
      !pop.array<apply(:(!kgen.struct<(index)>) -> index @unbox, *(0,0)), index>
     ) -> () fn, :!kgen.struct<(index)> apply(:(index) -> !kgen.struct<(index)> @box, a))]()
  kgen.return
}

// -----

kgen.generator @make_one() -> index {
  %0 = index.constant 1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @parametric_const
kgen.generator @parametric_const() {
  // CHECK-NEXT: constant: variant<index, scalar<f32>> = <{1, 0}>
  kgen.param.constant: variant<index, simd<apply(:() -> index @make_one), f32>> = <{1, 0}>
  kgen.return
}

// -----

kgen.generator @pass(%arg0: index) -> index {
  kgen.return %arg0 : index
}

kgen.generator @make_array<size>() -> !pop.array<apply(:(index) -> index @pass, size), i1> {
  %false = index.bool.constant false
  %0 = pop.array.repeat [%false] : !pop.array<apply(:(index) -> index @pass, size), i1>
  kgen.return %0 : !pop.array<apply(:(index) -> index @pass, size), i1>
}

// CHECK-LABEL: kgen.func @caller
kgen.generator @caller() {
  // CHECK-NEXT: array<2, i1> = <[0, 0]>
  kgen.param.constant: array<apply(:(index) -> index @pass, 2), i1> = <
    apply(:() -> !pop.array<apply(:(index) -> index @pass, 2), i1> @make_array<2>)
  >
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @some_function
kgen.generator @some_function() {
  kgen.return
}

kgen.generator @return_closure_formation() -> !kgen.generator<() -> ()> {
  %0 = kgen.create_closure[() -> (): @some_function]()
  kgen.return %0 : !kgen.generator<() -> ()>
}

// CHECK-LABEL: kgen.func export @interpret_create_closure
kgen.generator export @interpret_create_closure() {
  // CHECK-NEXT: constant: () -> () = <@some_function>
  kgen.param.constant: () -> () = <apply(:() -> !kgen.generator<() -> ()> @return_closure_formation)>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @a_function
kgen.generator @a_function() {
  kgen.return
}

kgen.generator @load_store_function(%arg0: !kgen.generator<() -> ()>) -> !kgen.generator<() -> ()> {
  %0 = pop.stack_allocation 1 x !kgen.generator<() -> ()>
  pop.store %arg0, %0 : !kgen.pointer<() -> ()>
  %1 = pop.load %0 : !kgen.pointer<() -> ()>
  kgen.return %1 : !kgen.generator<() -> ()>
}

// CHECK-LABEL: kgen.func export @call_it
kgen.generator export @call_it() {
  // CHECK-NEXT: constant: () -> () = <@a_function>
  kgen.param.constant: () -> () = <apply(:(!kgen.generator<() -> ()>) -> !kgen.generator<() -> ()> @load_store_function, @a_function)>
  kgen.return
}

// -----

kgen.generator @store_variadic(%arg0: !kgen.param_list<index>, %arg1: !kgen.pointer<param_list<index>>) {
  pop.store %arg0, %arg1 : !kgen.pointer<param_list<index>>
  kgen.return
}

kgen.generator @pass_and_read_variadic(%arg0: !kgen.param_list<index>) -> !kgen.param_list<index> {
  %0 = pop.stack_allocation 1 x !kgen.param_list<index>
  kgen.call @store_variadic(%arg0, %0) : (!kgen.param_list<index>, !kgen.pointer<param_list<index>>) -> ()
  %1 = pop.load %0 : !kgen.pointer<param_list<index>>
  kgen.return %1 : !kgen.param_list<index>
}

// CHECK-LABEL: kgen.func export @persistent_variadic
kgen.generator export @persistent_variadic() {
  // CHECK-NEXT: param_list<index> = <[1, 2]>
  kgen.param.constant: param_list<index> = <apply(:(!kgen.param_list<index>) -> !kgen.param_list<index> @pass_and_read_variadic, [1, 2])>
  kgen.return
}

// -----

kgen.generator @size_zero_alloc() -> !kgen.pointer<index> {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %0 = pop.aligned_alloc %idx1, %idx0 : <index>
  kgen.return %0 : !kgen.pointer<index>
}

// CHECK-LABEL: kgen.func export @malloc_nullptr
kgen.generator export @malloc_nullptr() {
  // CHECK-NEXT: pointer<index> = <0>
  %0 = kgen.param.constant: pointer<index> = <apply(:() -> !kgen.pointer<index> @size_zero_alloc)>
  kgen.return
}

// -----

// COM: This test ensures scalar data and a pointer region can co-exist
// COM: correctly in the same memory allocation.

// CHECK: [[BLOB1:#.*]] = #interp.memory_handle<8, "0x0100000000000000000000000000000020CA9A3B000000000000000000000000">
// CHECK: [[BLOB2:#.*]] = #interp.memory_handle<1, "0x00">

!ptr_t = !kgen.pointer<variant<index, pointer<none>>>

kgen.generator @fill_ptr() -> !ptr_t {
  %idx1 = index.constant 1
  %idx8 = index.constant 8
  %idx32 = index.constant 32

  // Allocate an array of 2 elements.
  %0 = pop.aligned_alloc %idx8, %idx32 : !ptr_t

  // Store 1 to the first element.
  %1 = pop.pointer.bitcast %0 : !ptr_t to !kgen.pointer<index>
  pop.store %idx1, %1 : !kgen.pointer<index>

  // Store a pointer to the second element.
  %2 = pop.offset %0[%idx1] : !ptr_t
  %3 = pop.aligned_alloc %idx1, %idx1 : !kgen.pointer<none>
  %4 = pop.pointer.bitcast %2 : !ptr_t to !kgen.pointer<pointer<none>>
  pop.store %3, %4 : !kgen.pointer<pointer<none>>

  kgen.return %0 : !ptr_t
}

// CHECK-LABEL: kgen.func export @pointer_overwrite
kgen.generator export @pointer_overwrite() {
  // CHECK-NEXT: memref<{[([[BLOB1]], heap, [(16, 1, 0)], []), ([[BLOB2]], heap, [], [])], []}, 0, 0>>
  kgen.param.constant: !ptr_t = <apply(:() -> !ptr_t @fill_ptr)>
  kgen.return
}

// -----

kgen.generator @elif(%arg0: index, %arg1: index) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2
  %cond0 = index.cmp eq(%arg0, %idx0)
  %cond0b = pop.cast_from_builtin %cond0 : i1 to !kgen.scalar<bool>
  %0 = hlcf.elif %cond0b -> index {
    hlcf.yield %idx0 : index
  } else {
    %cond1 = index.cmp eq(%arg0, %idx1)
    %cond1b = pop.cast_from_builtin %cond1 : i1 to !kgen.scalar<bool>
    hlcf.elif.yield %cond1b
  } then {
    hlcf.yield %idx1 : index
  } else {
    %cond2 = index.cmp eq(%arg0, %idx2)
    %cond2b = pop.cast_from_builtin %cond2 : i1 to !kgen.scalar<bool>
    hlcf.elif.yield %cond2b
  } then {
    hlcf.yield %idx2 : index
  } else {
    hlcf.yield %arg1 : index
  }

  %2 = index.mul %0, %0
  kgen.return %2 : index
}

// CHECK-LABEL: kgen.func export @constexpr_elif
kgen.generator export @constexpr_elif() -> index {
  // CHECK-NEXT: kgen.param.constant = <4>
  %0 = kgen.param.constant = <apply(:(index, index) -> index @elif, 2, 3)>
  // CHECK-NEXT: kgen.param.constant = <25>
  %1 = kgen.param.constant = <apply(:(index, index) -> index @elif, 3, 5)>
  kgen.return %1 : index
}

// -----

kgen.func @elifWithArgs(%arg0: index) -> index {
  %idx3 = index.constant 3
  %idx1 = index.constant 1
  %2 = index.add %arg0, %idx1
  %3 = index.cmp eq(%2, %idx3)
  %c3 = pop.cast_from_builtin %3 : i1 to !kgen.scalar<bool>
  %0:2 = hlcf.elif %c3 -> index, index {
    hlcf.yield %2, %2 : index, index
  } else {
    hlcf.yield %idx1, %idx1 : index, index
  }
  %1 = index.add %0#1, %0#0
  kgen.return %1 : index
}

kgen.func @elifManyRegionsWithArgs(%arg0: index) -> index {
  %idx3 = index.constant 3
  %idx1 = index.constant 1
  %idx4 = index.constant 4
  %2 = index.add %arg0, %idx1
  %3 = index.cmp eq(%2, %idx3)
  %c3 = pop.cast_from_builtin %3 : i1 to !kgen.scalar<bool>
  %0:2 = hlcf.elif %c3 -> index, index {
    hlcf.yield %2, %2 : index, index
  } else {
    %4 = index.cmp eq(%arg0, %idx4)
    %c4 = pop.cast_from_builtin %4 : i1 to !kgen.scalar<bool>
    %5 = index.add %2, %idx3
    hlcf.elif.yield %c4, %5 : index
  } then (%arg1 : index) {
    hlcf.yield %arg1, %arg1 : index, index
  } else (%arg2 : index) {
    hlcf.yield %idx1, %idx1 : index, index
  }
  %1 = index.add %0#1, %0#0
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.func export @constexpr_elif
kgen.generator export @constexpr_elif() -> index {
  // CHECK: kgen.param.constant = <6>
  %0 = kgen.param.constant = <apply(:(index) -> index @elifWithArgs, 2)>
  // CHECK: kgen.param.constant = <16>
  %1 = kgen.param.constant = <apply(:(index) -> index @elifManyRegionsWithArgs, 4)>
  kgen.return %0 : index
}

// -----

kgen.func @pack_load() -> !kgen.struct<(si4, ui8) isParamPack> {
  %i0 = kgen.param.constant: si4 = <-5>
  %i1 = kgen.param.constant: ui8 = <42>
  %p0 = pop.stack_allocation 1 x si4
  %p1 = pop.stack_allocation 1 x ui8
  pop.store %i0, %p0 : !kgen.pointer<si4>
  pop.store %i1, %p1 : !kgen.pointer<ui8>
  %pack = kgen.struct.create(%p0, %p1) : !kgen.struct<(pointer<si4>, pointer<ui8>) isParamPack>
  %loaded_pack = kgen.struct.load_indirect %pack : !kgen.struct<(pointer<si4>, pointer<ui8>) isParamPack>
  kgen.return %loaded_pack : !kgen.struct<(si4, ui8) isParamPack>
}

// CHECK-LABEL: kgen.func export @interpret_pack_load
kgen.generator export @interpret_pack_load() -> !kgen.struct<(si4, ui8) isParamPack> {
  // CHECK-NEXT: %{{.*}} = kgen.param.constant: struct<(si4, ui8) isParamPack> = <{ -5, 42 }>
  %0 = kgen.param.constant: !kgen.struct<(si4, ui8) isParamPack> = <apply(:() -> !kgen.struct<(si4, ui8) isParamPack> @pack_load)>
  kgen.return %0 : !kgen.struct<(si4, ui8) isParamPack>
}

// -----

#mem = #interp.memref<{[(#interp.memory_handle<64, "0x0700000000000000">, heap, [], [])], []}, 0, 0> : !kgen.pointer<index>

// COM: Ensure 'load_from_mem' externalizes pointer values.

// CHECK-LABEL: kgen.func @xd
kgen.generator @xd() {
  // CHECK-NEXT: %index7 = kgen.param.constant = <7>
  kgen.param.constant: index = <load_from_mem(:pointer<index> #mem)>
  kgen.return
}

// -----

#mem = #interp.memref<{[(#interp.memory_handle<64, "0x0700000000000000">, heap, [], [])], []}, 0, 0> : !kgen.pointer<index>

// COM: Ensure results of kgen.param.constant are internalized.

kgen.generator @testInternalization(%pointer: !kgen.pointer<index>) -> index {
  %3 = pop.load %pointer : !kgen.pointer<index>
  kgen.return %3 : index
}

kgen.generator @makePtrPtrConstant<ptr: !kgen.pointer<index>>() -> !kgen.pointer<pointer<index>> {
  %idx8 = index.constant 8
  %idx-1 = index.constant -1
  %0 = pop.aligned_alloc %idx-1, %idx8 : <index>
  %pointer = kgen.param.constant: !kgen.pointer<index> = <ptr>
  %x = pop.load %pointer : !kgen.pointer<index>
  pop.store %x, %0 : !kgen.pointer<index>
  %1 = pop.aligned_alloc %idx-1, %idx8 : !kgen.pointer<pointer<index>>
  pop.store %0, %1 : !kgen.pointer<pointer<index>>
  kgen.return %1 :  !kgen.pointer<pointer<index>>
}

// CHECK-LABEL: kgen.func export @constant() -> index {
kgen.generator export @constant() -> index {
  kgen.param.apply ptrptr = [() -> !kgen.pointer<pointer<index>>: @makePtrPtrConstant<:pointer<index> #mem>]()
  kgen.param.apply loadIt = [(!kgen.pointer<index>) -> index: @testInternalization](load_from_mem(:!kgen.pointer<pointer<index>> ptrptr))
  // CHECK-NEXT: %index7 = kgen.param.constant = <7>
  %0 = kgen.param.constant: index = <loadIt>
  kgen.return %0 : index
}

// COM: Ensure results of kgen.param.materialize are internalized.

kgen.generator @makePtrPtrMaterialize() -> !kgen.pointer<pointer<index>> {
  %idx8 = index.constant 8
  %idx-1 = index.constant -1
  %0 = pop.aligned_alloc %idx-1, %idx8 : <index>
  %pointer = kgen.param.materialize: !kgen.pointer<index> = <#mem>
  %x = pop.load %pointer : !kgen.pointer<index>
  pop.store %x, %0 : !kgen.pointer<index>
  %1 = pop.aligned_alloc %idx-1, %idx8 : !kgen.pointer<pointer<index>>
  pop.store %0, %1 : !kgen.pointer<pointer<index>>
  kgen.return %1 :  !kgen.pointer<pointer<index>>
}

// CHECK-LABEL: kgen.func export @materialize() -> index {
kgen.generator export @materialize() -> index {
  kgen.param.apply ptrptr = [() -> !kgen.pointer<pointer<index>>: @makePtrPtrMaterialize]()
  kgen.param.apply loadIt = [(!kgen.pointer<index>) -> index: @testInternalization](load_from_mem(:!kgen.pointer<pointer<index>> ptrptr))
  // CHECK-NEXT: %index7 = kgen.param.constant = <7>
  %0 = kgen.param.constant: index = <loadIt>
  kgen.return %0 : index
}

// -----

// CHECK-MAIN-DAG: #memory_handle = #interp.memory_handle<8, "0x0000000000000000">
kgen.generator @target(%arg0 : index) -> index {
  kgen.return %arg0 : index
}

kgen.generator @testExternalization(%arg0: !kgen.pointer<(index) -> index>) -> !kgen.pointer<(index) -> index> {
   kgen.return %arg0 : !kgen.pointer<(index) -> index>
}

// CHECK-MAIN-LABEL: kgen.func @"testInternalization{{.*}}() -> index {
kgen.generator @testInternalization<ptr: !kgen.pointer<(index) -> index>>() -> index {
  %0 = kgen.param.constant = <7>
  // CHECK-MAIN: %pointer = kgen.param.constant: pointer<(index) -> index> = <#interp.memref<{[(#memory_handle, stack, [], [0])], [#kgen.symbol.constant<@target> : !kgen.generator<(index) -> index>]}, 0, 0>>
  %pointer = kgen.param.constant: pointer<(index) -> index> = <ptr>
  %3 = pop.load %pointer : !kgen.pointer<(index) -> index>
  %4 = kgen.call_indirect %3(%0) : (index) -> index
  kgen.return %4 : index
}

// CHECK: kgen.func export @root
// CHECK-NEXT: %index7 = kgen.param.constant = <7>
// CHECK-NEXT: kgen.return %index7 : index
kgen.generator export @root() -> index {
  kgen.param.declare symbol: (index) -> index = <@target>
  kgen.param.apply storeIt = [(!kgen.pointer<(index) -> index>) -> !kgen.pointer<(index) -> index> : @testExternalization](store_to_mem(symbol))
  kgen.param.apply loadIt = [() -> index: @testInternalization<:!kgen.pointer<(index) -> index> storeIt>]()
  %0 = kgen.param.constant: index = <loadIt>
  kgen.return %0 : index
}

// -----

// Test lifting of nested `store_to_mem`s.

// This function takes a pack of two dtype pointers, and returns the first element, loaded.
kgen.generator @use_pack(%arg0: !kgen.pointer<!kgen.struct<(pointer<dtype>, pointer<dtype>) isParamPack>> owned_in_mem) -> !kgen.dtype {
  %pack = pop.load %arg0 : !kgen.pointer<!kgen.struct<(pointer<dtype>, pointer<dtype>) isParamPack>>
  %elem = kgen.struct.extract %pack[0] : <(pointer<dtype>, pointer<dtype>) isParamPack>
  %val = pop.load %elem : !kgen.pointer<dtype>
  kgen.return %val : !kgen.dtype
}

// CHECK-LABEL: kgen.func export @top
kgen.generator export @top() {
  // Call `use_pack` with a pack of two dtypes stored in memory, with the pack stored in memory too.
  // After interpreting the function, we should get back the first dtype, loaded from memory.
  // CHECK-NEXT: kgen.param.constant: dtype = <f8e5m2>
  kgen.param.constant : dtype = <apply(:(!kgen.pointer<!kgen.struct<(pointer<dtype>, pointer<dtype>) isParamPack>> owned_in_mem) -> !kgen.dtype @use_pack, store_to_mem({ store_to_mem(f8e5m2), store_to_mem(f8e5m2fnuz) }))>
  kgen.return
}

// -----

// COM: MOCO-1048 is a bug triggered by Memrefs in Symbol Constants Not Being Properly Internalized/Externalized By Interpreter

#mem = #interp.memref<{[(#interp.memory_handle<64, "0x0300000000000000">, heap, [], [])], []}, 0, 0> : !kgen.pointer<index>

// The target symbol must have the following to trigger error state:
// A bound parameter that contains a MemRef attribute
// An unbound parameter (idx_type) to prevent the symbol from being concretized before its stored to mem.
// CHECK-MAIN-DAG: [[MHVal:#.*]] = #interp.memory_handle<64, "0x0300000000000000">
// CHECK-MAIN-DAG: [[MHSig:#.*]] = #interp.memory_handle<8, "0x0000000000000000">
// CHECK-MAIN-LABEL: kgen.func @"captureIt{{.*}}"(%arg0: index) -> index {
kgen.generator @"captureIt"<dst_layout: pointer<index>, idx_type: dtype>(%arg0: index) -> index {
  // CHECK-MAIN-NEXT: %pointer = kgen.param.constant: pointer<index> = <#interp.memref<{[([[MHVal]], heap, [], []), ([[MHSig]], stack, [], [0])], [#kgen.symbol.constant<@captureIt<:pointer<index> #interp<coord(0, 0)>, :dtype ?>> : !kgen.generator<<dtype>(index) -> index>]}, 0, 0>>
  %pointer = kgen.param.constant: pointer<index> = <dst_layout>
  %3 = pop.load %pointer : !kgen.pointer<index>
  kgen.return %3 : index
}

kgen.generator @embedMemRefInSymbol<dst_layout: pointer<index>>() -> index {
  %0 = kgen.param.constant: index = <0>
  kgen.param.declare symbolWithMemRef: <dtype>(index) -> index = <@"captureIt"<:pointer<index> dst_layout, :dtype ?>>

  // The "store to mem" operation results in an opaque capture of a PointerAttr.
  kgen.param.apply callIt = [(!kgen.pointer<<dtype>(index) -> index>) -> !kgen.generator<<dtype>(index) -> index>: @call_it](store_to_mem(symbolWithMemRef))

  // The call_param of the loaded symbol results in a read of the symbol with the unmapped pointer symbol
  %1 = kgen.call_param[(index) -> index: bind_params(:<dtype>(index) -> index callIt, :dtype index)](%0)
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.func export @main() -> index {
kgen.generator export @main() -> index {
  kgen.param.apply result = [() -> index: @embedMemRefInSymbol<:pointer<index> #mem>]()
  // CHECK-NEXT: %index3 = kgen.param.constant = <3>
  %0 = kgen.param.constant = <result>
  kgen.return %0 : index
}

kgen.generator @call_it(%arg1: !kgen.pointer<<dtype>(index) -> index>) -> !kgen.generator<<dtype>(index) -> index> {
  %1 = pop.load %arg1 : !kgen.pointer<<dtype>(index) -> index>
  kgen.return %1 : !kgen.generator<<dtype>(index) -> index>
}

// -----

// This should only contain a pointer to heap base addr (1'000'000'000 == 0x3B9ACA00) and all zeros afterwards.
// CHECK: #[[MEM_STACK:.+]] = #interp.memory_handle<8, "0x00CA9A3B000000000000000000000000000000000000000000000000000000000000000000000000">
!variant = !kgen.variant<pointer<index>, struct<(!pop.array<4, index>)>>
!ptr_v = !kgen.pointer<!variant>

kgen.generator @get_variant(%arg0: !kgen.pointer<!variant> byref_result) {
  // Create inner pointer to index.
  %size = index.constant 8
  %align = index.constant 8
  %mem = pop.aligned_alloc %align, %size : !kgen.pointer<index>

  // Fill inner pointer.
  %i1 = kgen.param.constant: index = <1>
  pop.store %i1, %mem : !kgen.pointer<index>

  // Create overall variant and store in return slot.
  %v = kgen.variant.create %mem, 0 : !variant
  pop.store %v, %arg0 : !kgen.pointer<!variant>
  kgen.return
}

// CHECK-LABEL: kgen.func export @call_result_slot
kgen.generator export @call_result_slot() {
  // CHECK-NEXT: <{:pointer<index> #interp.memref<{[({{.*}}, heap, [], []), (#[[MEM_STACK]], stack, [(0, 0, 0)], [])], []}, 0, 0>, 0}>
  kgen.param.constant: !variant = <apply_result_slot(:(!ptr_v byref_result) -> () @get_variant)>
  kgen.return
}

// -----

// COM: Load/Store IndexTypes

kgen.generator @writeIndexType(%arg0: index) -> !kgen.pointer<index> {
  %index1 = kgen.param.constant = <1>
  %2 = index.sub %arg0, %index1
  %idx8 = index.constant 8
  %idx-1 = index.constant -1
  %3 = pop.aligned_alloc %idx-1, %idx8 : <index>
  pop.store %2, %3 : !kgen.pointer<index>
  kgen.return %3 : !kgen.pointer<index>
}

// CHECK-LABEL: kgen.func export @readIndexType() -> index
kgen.generator export @readIndexType() -> index {
  kgen.param.apply PTR = [(index) -> !kgen.pointer<index>: @writeIndexType](0)

  // CHECK-NEXT: %index-1 = kgen.param.constant = <-1>
  %0 = kgen.param.constant: index = <load_from_mem(:!kgen.pointer<index> PTR)>
  kgen.return %0 :  index
}

// -----

kgen.generator @union_wrap(%arg0: index) -> !pop.union<index, i64> {
  %0 = pop.union.wrap %arg0 : index as !pop.union<index, i64>
  kgen.return %0 : !pop.union<index, i64>
}

kgen.generator @union_unwrap(%arg0: !pop.union<index, i64>) -> index {
  %0 = pop.union.unwrap %arg0 : !pop.union<index, i64> as index
  kgen.return %0 : index
}

kgen.generator @union_in_memory(%arg0: index) -> index {
  %0 = pop.union.wrap %arg0 : index as !pop.union<index, i64>
  %1 = pop.stack_allocation 1 x !pop.union<index, i64>
  pop.store %0, %1 : !kgen.pointer<!pop.union<index, i64>>
  %2 = pop.union.bitcast %1 : !kgen.pointer<!pop.union<index, i64>> as !kgen.pointer<index>
  %3 = pop.load %2 : !kgen.pointer<index>
  kgen.return %3 : index
}

// CHECK-LABEL: kgen.func export @test_union
kgen.generator export @test_union() {
  kgen.param.apply union = [(index) -> !pop.union<index, i64>: @union_wrap](43)
  kgen.param.apply unwrapped = [(!pop.union<index, i64>) -> index: @union_unwrap](union)
  // CHECK: = <43>
  %0 = kgen.param.constant: index = <unwrapped>
  kgen.param.apply union_in_memory = [(index) -> index: @union_in_memory](56)
  // CHECK: = <56>
  %1 = kgen.param.constant: index = <union_in_memory>
  kgen.return
}

// -----
// COME: test for MOCO-1978
!structTy = !kgen.struct<(!kgen.pointer<index>)>
!structTy2 = !kgen.struct<(!kgen.pointer<!structTy>, !kgen.pointer<!structTy>)>

// CHECK: [[BLOB:#.*]] = #interp.memory_handle<8, "0x0004000000000000">
// CHECK: [[BLOB1:#.*]] = #interp.memory_handle<8, "0x00CA9A3B00000000">
// CHECK: [[BLOB2:#.*]] = #interp.memory_handle<8, "0x08CA9A3B0000000008CA9A3B00000000">

// CHECK-LABEL @f
kgen.generator @f(
    %arg0: index) -> !kgen.pointer<index> no_inline {
  %idx8 = index.constant 8
  %0 = pop.aligned_alloc %idx8, %idx8 : !kgen.pointer<none>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<none> to !kgen.pointer<index>
  pop.store %arg0, %1: !kgen.pointer<index>
  kgen.return %1: !kgen.pointer<index>
}

// CHECK-LABEL @g
kgen.generator @g(
    %arg0: !kgen.pointer<index>) -> !kgen.pointer<!structTy> no_inline {
  %idx8 = index.constant 8
  %0 = pop.aligned_alloc %idx8, %idx8 : !kgen.pointer<none>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<none> to !kgen.pointer<!structTy>
  %2 = kgen.struct.gep %1[0] : <!structTy>
  pop.store %arg0, %2: !kgen.pointer<pointer<index>>
  kgen.return %1: !kgen.pointer<!structTy>
}

// CHECK-LABEL @h
kgen.generator @h(%arg0: !kgen.pointer<!structTy>,%arg1: !kgen.pointer<!structTy>) -> !kgen.pointer<!structTy2> no_inline {
  %idx8 = index.constant 8
  %idx16 = index.constant 16
  %0 = pop.aligned_alloc %idx8, %idx16 : !kgen.pointer<none>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<none> to !kgen.pointer<!structTy2>
  %2 = kgen.struct.gep %1[0] : <!structTy2>
  %3 = kgen.struct.gep %1[1] : <!structTy2>
  pop.store %arg0, %2: !kgen.pointer<pointer<!structTy>>
  pop.store %arg1, %3: !kgen.pointer<pointer<!structTy>>
  kgen.return %1: !kgen.pointer<!structTy2>
}

// CHECK-LABEL @m
kgen.generator @m(%arg0: !kgen.pointer<index>, %arg1: !kgen.pointer<!structTy2>) -> !kgen.pointer<index> no_inline {
  %1 = kgen.struct.gep %arg1[0] : <!structTy2>
  %2 = pop.load %1 : !kgen.pointer<pointer<!structTy>>
  %3 = kgen.struct.gep %2[0] : <!structTy>
  %4 = pop.load %3 : !kgen.pointer<pointer<index>>
  kgen.return %4: !kgen.pointer<index>
}

// CHECK-LABEL @top
kgen.generator export @top() -> () {
  kgen.param.declare v0 = <512>
  kgen.param.declare v1 = <1024>
  kgen.param.declare v2 = <2048>

  // COM: check that BLOB1 and BLOB2 have pointer regions.
  // CHECK: kgen.param.materialize
  // CHECK-SAME: [[BLOB]], heap, [], [])
  // CHECK-SAME: [[BLOB1]], heap, [(0, 0, 0)], [])
  // CHECK-SAME: [[BLOB2]], heap, [(0, 1, 0), (8, 1, 0)], [])
  %3 = kgen.param.materialize: !kgen.pointer<index> = <
    apply(:(!kgen.pointer<index>, !kgen.pointer<!structTy2>) -> !kgen.pointer<index> @m,
      store_to_mem(v0),
      apply(:(!kgen.pointer<!structTy>, !kgen.pointer<!structTy>) -> !kgen.pointer<!structTy2> @h,
        apply(:(!kgen.pointer<index>) -> !kgen.pointer<!structTy> @g,
            apply(:(index) -> !kgen.pointer<index> @f, v1)
        ),
        apply(:(!kgen.pointer<index>) -> !kgen.pointer<!structTy> @g,
            apply(:(index) -> !kgen.pointer<index> @f, v1)
        )
      )
    )
  >
  kgen.return
}

// -----

// COM: Check integer bitcasting when sizes do not match.

kgen.generator @check128to64(%arg0: !kgen.simd<1, ui128>) -> !kgen.simd<2, ui64> {
  %0 = pop.bitcast %arg0 : !kgen.simd<1, ui128> to !kgen.simd<2, ui64>
  kgen.return %0 : !kgen.simd<2, ui64>
}

kgen.generator @check64to128(%arg0: !kgen.simd<2, ui64>) -> !kgen.simd<1, ui128> {
  %0 = pop.bitcast %arg0 : !kgen.simd<2, ui64> to !kgen.simd<1, ui128>
  kgen.return %0 : !kgen.simd<1, ui128>
}

kgen.generator @callIt() -> !kgen.simd<1, ui128> {
  kgen.param.declare X: simd<1, ui128> = <<16622636600618719503991588326398409450>>
  kgen.param.declare Y: simd<2, ui64> = <apply(:(!kgen.simd<1, ui128>) -> !kgen.simd<2, ui64> @check128to64, X)>
  // CHECK: <<4225598797516028650, 901114935741393800>>
  %0 = kgen.param.constant: !kgen.simd<2, ui64> = <Y>
  // CHECK: <16622636600618719503991588326398409450>
  kgen.param.declare Z: simd<1, ui128> = <apply(:(!kgen.simd<2, ui64>) -> !kgen.simd<1, ui128> @check64to128, Y)>
  %1 = kgen.param.constant: !kgen.simd<1, ui128> = <Z>
  kgen.return %1 : !kgen.simd<1, ui128>
}

// -----

// COM: Check float bitcasting when sizes do not match.

kgen.generator @check64to32(%arg0: !kgen.simd<1, f64>) -> !kgen.simd<2, f32> {
  %0 = pop.bitcast %arg0 : !kgen.simd<1, f64> to !kgen.simd<2, f32>
  kgen.return %0 : !kgen.simd<2, f32>
}

kgen.generator @check32to64(%arg0: !kgen.simd<2, f32>) -> !kgen.simd<1, f64> {
  %0 = pop.bitcast %arg0 : !kgen.simd<2, f32> to !kgen.simd<1, f64>
  kgen.return %0 : !kgen.simd<1, f64>
}

kgen.generator @callFloatBitcast() -> !kgen.simd<2, f32> {
  kgen.param.declare X: simd<1, f64> = <<"3.1415926535897931">>
  kgen.param.declare Y: simd<2, f32> = <apply(:(!kgen.simd<1, f64>) -> !kgen.simd<2, f32> @check64to32, X)>
  // CHECK: <<"3.37028055E+12", "2.142699">>
  %0 = kgen.param.constant: !kgen.simd<2, f32> = <Y>
  // CHECK: <"3.1415926535897931">
  kgen.param.declare Z: simd<1, f64> = <apply(:(!kgen.simd<2, f32>) -> !kgen.simd<1, f64> @check32to64, Y)>
  %1 = kgen.param.constant: !kgen.simd<1, f64> = <Z>
  kgen.return %0 : !kgen.simd<2, f32>
}

// -----

// COM: Check integer bitcasting for big Endianness

module attributes {M.target_info = #M.target<triple ="", arch = "", features = "", data_layout = "E-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {

kgen.generator @check128to64(%arg0: !kgen.simd<1, ui128>) -> !kgen.simd<2, ui64> {
  %0 = pop.bitcast %arg0 : !kgen.simd<1, ui128> to !kgen.simd<2, ui64>
  kgen.return %0 : !kgen.simd<2, ui64>
}

kgen.generator @check64to128(%arg0: !kgen.simd<2, ui64>) -> !kgen.simd<1, ui128> {
  %0 = pop.bitcast %arg0 : !kgen.simd<2, ui64> to !kgen.simd<1, ui128>
  kgen.return %0 : !kgen.simd<1, ui128>
}

kgen.generator @callIt() -> !kgen.simd<2, ui64> {
  kgen.param.declare X: simd<1, ui128> = <<16622636600618719503991588326398409450>>
  kgen.param.declare Y: simd<2, ui64> = <apply(:(!kgen.simd<1, ui128>) -> !kgen.simd<2, ui64> @check128to64, X)>
  // CHECK: <<901114935741393800, 4225598797516028650>>
  %0 = kgen.param.constant: !kgen.simd<2, ui64> = <Y>
  // CHECK: <16622636600618719503991588326398409450>
  kgen.param.declare Z: simd<1, ui128> = <apply(:(!kgen.simd<2, ui64>) -> !kgen.simd<1, ui128> @check64to128, Y)>
  %1 = kgen.param.constant: !kgen.simd<1, ui128> = <Z>
  kgen.return %0 : !kgen.simd<2, ui64>
}
}

// -----

// COM: Check pop.global_constant

// CHECK: [[BLOB:#.*]] = #interp.memory_handle<16, "0xFECABEBAEFBEADDEEFCDAB9078563412">
kgen.generator @global_const() -> !kgen.pointer<struct<(array<1, scalar<ui128>>)>> {
  %0 = pop.global_constant: struct<(array<1, scalar<ui128>>)> =
    <{[0x1234567890abcdefdeadbeefbabecafe]}>

  kgen.return %0 : !kgen.pointer<struct<(array<1, scalar<ui128>>)>>
}

kgen.generator @callIt()->!kgen.pointer<struct<(array<1, scalar<ui128>>)>> {
  // CHECK: [[BLOB]], const_global
  kgen.param.declare C: !kgen.pointer<struct<(array<1, scalar<ui128>>)>> =
    <apply(:() -> !kgen.pointer<struct<(array<1, scalar<ui128>>)>> @global_const)>

  %0 = kgen.param.constant: !kgen.pointer<struct<(array<1, scalar<ui128>>)>> = <C>

  kgen.return %0: !kgen.pointer<struct<(array<1, scalar<ui128>>)>>
}


// -----

// COM: Check a pop.global_constant whose value embeds a string: writing the
// COM: aggregate re-enters constant-global mapping for the string while the
// COM: aggregate's blob has no handle yet, and must neither assert nor
// COM: invalidate the blob reference held across the write (MOCO-4493).

kgen.generator @global_const_with_string() -> index {
  %idx1 = index.constant 1
  %0 = pop.global_constant: struct<(string)> = <{"a string nobody mapped yet"}>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<struct<(string)>> to !kgen.pointer<index>
  %2 = pop.offset %1[%idx1] : !kgen.pointer<index>
  %3 = pop.load %2 : !kgen.pointer<index>
  kgen.return %3 : index
}

// CHECK-LABEL: kgen.func @string_global_constant_size
kgen.generator @string_global_constant_size() -> index {
  kgen.param.declare S: index = <apply(:() -> index @global_const_with_string)>
  // COM: The KGEN string is laid out as a {ptr, size}. Here we load the second
  // field of that struct and so return the string's length.
  // CHECK: kgen.param.constant = <26>
  %0 = kgen.param.constant: index = <S>
  kgen.return %0 : index
}

// -----

kgen.generator @memcpy_1(%arg0: index) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2
  %idx3 = index.constant 3

  %p0 = pop.stack_allocation 1 x !kgen.struct<(index, index, index, index)>
  %p1 = pop.stack_allocation 1 x !kgen.struct<(index, index, index, index)>

  %p = pop.pointer.bitcast %p0 : !kgen.pointer<!kgen.struct<(index, index, index, index)>> to !kgen.pointer<index>
  %q = pop.pointer.bitcast %p1 : !kgen.pointer<!kgen.struct<(index, index, index, index)>> to !kgen.pointer<index>

  pop.store %idx0, %p : !kgen.pointer<index>
  %p2 = pop.offset %p[%idx1] : !kgen.pointer<index>
  pop.store %idx1, %p2 : !kgen.pointer<index>
  %p3 = pop.offset %p[%idx2] : !kgen.pointer<index>
  pop.store %idx2, %p3 : !kgen.pointer<index>
  %p4 = pop.offset %p2[%idx2] : !kgen.pointer<index>
  pop.store %idx3, %p4 : !kgen.pointer<index>

  %len = kgen.param.constant: index = <get_sizeof(struct<(index, index, index, index)>, current_target())>
  pop.memcpy %q, %p, %len: !kgen.pointer<index> -> !kgen.pointer<index>

  %ptr = pop.offset %q[%arg0] : !kgen.pointer<index>
  %result = pop.load %ptr : !kgen.pointer<index>
  kgen.return %result : index
}

// CHECK-LABEL: kgen.func @constexpr_memcpy
kgen.generator @constexpr_memcpy() {
  // CHECK-NEXT: = <0>
  %0 = kgen.param.constant = <apply(:(index) -> index @memcpy_1, 0)>
  // CHECK-NEXT: = <1>
  %1 = kgen.param.constant = <apply(:(index) -> index @memcpy_1, 1)>
  // CHECK-NEXT: = <2>
  %2 = kgen.param.constant = <apply(:(index) -> index @memcpy_1, 2)>
  // CHECK-NEXT: = <3>
  %3 = kgen.param.constant = <apply(:(index) -> index @memcpy_1, 3)>
  kgen.return
}

// -----

// COM: Check abs

kgen.generator @abs_f32(%arg0: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  %0 = pop.abs %arg0 : !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

kgen.generator @abs_si32(%arg0: !kgen.simd<4, si32>) -> !kgen.simd<4, si32> {
  %0 = pop.abs %arg0 : !kgen.simd<4, si32>
  kgen.return %0 : !kgen.simd<4, si32>
}

kgen.generator @abs_ui32(%arg0: !kgen.simd<4, ui32>) -> !kgen.simd<4, ui32> {
  %0 = pop.abs %arg0 : !kgen.simd<4, ui32>
  kgen.return %0 : !kgen.simd<4, ui32>
}

kgen.generator @abs_bool(%arg0: !kgen.simd<2, bool>) -> !kgen.simd<2, bool> {
  %0 = pop.abs %arg0 : !kgen.simd<2, bool>
  kgen.return %0 : !kgen.simd<2, bool>
}

kgen.generator @abs_index(%arg0: !kgen.simd<4, index>) -> !kgen.simd<4, index> {
  %0 = pop.abs %arg0 : !kgen.simd<4, index>
  kgen.return %0 : !kgen.simd<4, index>
}

kgen.generator @callAbs() -> !kgen.simd<4, f32> {
  kgen.param.declare F0: simd<4, f32> = <<"7.0", "-7.0", "-NaN", "-inf">>
  kgen.param.declare F1: simd<4, f32> = <apply(:(!kgen.simd<4, f32>) -> !kgen.simd<4, f32> @abs_f32, F0)>
  // CHECK: <<"7", "7", "NaN", "+Inf">>
  %0 = kgen.param.constant: !kgen.simd<4, f32> = <F1>

  // COM: abs(INT_MIN) -> INT_MIN
  kgen.param.declare S0: simd<4, si32> = <<7, 7, 0, -2147483648>>
  kgen.param.declare S1: simd<4, si32> = <apply(:(!kgen.simd<4, si32>) -> !kgen.simd<4, si32> @abs_si32, S0)>
  // CHECK: <<7, 7, 0, -2147483648>>
  %1 = kgen.param.constant: !kgen.simd<4, si32> = <S1>

  kgen.param.declare U0: simd<4, ui32> = <<7, 7, 0, 1>>
  kgen.param.declare U1: simd<4, ui32> = <apply(:(!kgen.simd<4, ui32>) -> !kgen.simd<4, ui32> @abs_ui32, U0)>
  // CHECK: <<7, 7, 0, 1>>
  %2 = kgen.param.constant: !kgen.simd<4, ui32> = <U1>

  kgen.param.declare B0: simd<2, bool> = <<true, false>>
  kgen.param.declare B1: simd<2, bool> = <apply(:(!kgen.simd<2, bool>) -> !kgen.simd<2, bool> @abs_bool, B0)>
  // CHECK: <<true, false>>
  %3 = kgen.param.constant: !kgen.simd<2, bool> = <B1>

  kgen.param.declare I0: simd<4, index> = <<-1, -8, 9223090564025548800, -9223372036854775808>>
  kgen.param.declare I1: simd<4, index> = <apply(:(!kgen.simd<4, index>) -> !kgen.simd<4, index> @abs_index, I0)>
  // COM: Even if we don't know the target index width, we can take the
  // absolute value of something like -1 at it returns the same thing no
  // matter whether it's 32 or 64-bit.
  // CHECK: <<1, 8, 9223090564025548800, -9223372036854775808>>
  %4 = kgen.param.constant: !kgen.simd<4, index> = <I1>

  kgen.return %0 : !kgen.simd<4, f32>
}

// -----

// COM: Check round

kgen.generator @round_f32(%arg0: !kgen.simd<8, f32>) -> !kgen.simd<8, f32> {
  %0 = pop.round %arg0 : !kgen.simd<8, f32>
  kgen.return %0 : !kgen.simd<8, f32>
}

kgen.generator @round_si32(%arg0: !kgen.simd<4, si32>) -> !kgen.simd<4, si32> {
  %0 = pop.round %arg0 : !kgen.simd<4, si32>
  kgen.return %0 : !kgen.simd<4, si32>
}

kgen.generator @round_ui32(%arg0: !kgen.simd<4, ui32>) -> !kgen.simd<4, ui32> {
  %0 = pop.round %arg0 : !kgen.simd<4, ui32>
  kgen.return %0 : !kgen.simd<4, ui32>
}

kgen.generator @round_bool(%arg0: !kgen.simd<2, bool>) -> !kgen.simd<2, bool> {
  %0 = pop.round %arg0 : !kgen.simd<2, bool>
  kgen.return %0 : !kgen.simd<2, bool>
}

kgen.generator @round_index(%arg0: !kgen.simd<4, index>) -> !kgen.simd<4, index> {
  %0 = pop.round %arg0 : !kgen.simd<4, index>
  kgen.return %0 : !kgen.simd<4, index>
}

kgen.generator @callRound() -> !kgen.simd<8, f32> {
  kgen.param.declare F0: simd<8, f32> = <<"7.0", "-7.0", "1.5", "1.1", "1.7", "-1.2", "-1.7", "-inf">>
  kgen.param.declare F1: simd<8, f32> = <apply(:(!kgen.simd<8, f32>) -> !kgen.simd<8, f32> @round_f32, F0)>
  // CHECK: <<"7", "-7", "2", "1", "2", "-1", "-2", "-Inf">>
  %0 = kgen.param.constant: !kgen.simd<8, f32> = <F1>

  kgen.param.declare S0: simd<4, si32> = <<7, 7, 0, -2147483648>>
  kgen.param.declare S1: simd<4, si32> = <apply(:(!kgen.simd<4, si32>) -> !kgen.simd<4, si32> @round_si32, S0)>
  // CHECK: <<7, 7, 0, -2147483648>>
  %1 = kgen.param.constant: !kgen.simd<4, si32> = <S1>

  kgen.param.declare U0: simd<4, ui32> = <<7, 7, 0, 1>>
  kgen.param.declare U1: simd<4, ui32> = <apply(:(!kgen.simd<4, ui32>) -> !kgen.simd<4, ui32> @round_ui32, U0)>
  // CHECK: <<7, 7, 0, 1>>
  %2 = kgen.param.constant: !kgen.simd<4, ui32> = <U1>

  kgen.param.declare B0: simd<2, bool> = <<true, false>>
  kgen.param.declare B1: simd<2, bool> = <apply(:(!kgen.simd<2, bool>) -> !kgen.simd<2, bool> @round_bool, B0)>
  // CHECK: <<true, false>>
  %3 = kgen.param.constant: !kgen.simd<2, bool> = <B1>

  kgen.param.declare I0: simd<4, index> = <<-1, -8, 9223090564025548800, -9223372036854775808>>
  kgen.param.declare I1: simd<4, index> = <apply(:(!kgen.simd<4, index>) -> !kgen.simd<4, index> @round_index, I0)>
  // CHECK: <<-1, -8, 9223090564025548800, -9223372036854775808>>
  %4 = kgen.param.constant: !kgen.simd<4, index> = <I1>

  kgen.return %0 : !kgen.simd<8, f32>
}

// COM: Check floordiv

kgen.generator @floordiv_f32(%arg0: !kgen.simd<2, f32>, %arg1: !kgen.simd<2, f32>) -> !kgen.simd<2, f32> {
  %0 = pop.floordiv %arg0, %arg1 : !kgen.simd<2, f32>
  kgen.return %0 : !kgen.simd<2, f32>
}

kgen.generator @floordiv_si32(%arg0: !kgen.simd<2, si32>, %arg1: !kgen.simd<2, si32>) -> !kgen.simd<2, si32> {
  %0 = pop.floordiv %arg0, %arg1 : !kgen.simd<2, si32>
  kgen.return %0 : !kgen.simd<2, si32>
}

kgen.generator @floordiv_ui32(%arg0: !kgen.simd<2, ui32>, %arg1: !kgen.simd<2, ui32>) -> !kgen.simd<2, ui32> {
  %0 = pop.floordiv %arg0, %arg1 : !kgen.simd<2, ui32>
  kgen.return %0 : !kgen.simd<2, ui32>
}

kgen.generator @callFloordiv() -> !kgen.simd<2, f32> {
  kgen.param.declare F0: simd<2, f32> = <<"7.0", "7.0">>
  kgen.param.declare F1: simd<2, f32> = <<"3.0", "-3.0">>
  kgen.param.declare F2: simd<2, f32> = <apply(:(!kgen.simd<2, f32>, !kgen.simd<2, f32>) -> !kgen.simd<2, f32> @floordiv_f32, F0, F1)>
  // CHECK: <<"2", "-3">>
  %0 = kgen.param.constant: !kgen.simd<2, f32> = <F2>

  kgen.param.declare S0: simd<2, si32> = <<7, 7>>
  kgen.param.declare S1: simd<2, si32> = <<3, -3>>
  kgen.param.declare S2: simd<2, si32> = <apply(:(!kgen.simd<2, si32>, !kgen.simd<2, si32>) -> !kgen.simd<2, si32> @floordiv_si32, S0, S1)>
  // CHECK: <<2, -3>>
  %1 = kgen.param.constant: !kgen.simd<2, si32> = <S2>

  kgen.param.declare U0: simd<2, ui32> = <<7, 7>>
  kgen.param.declare U1: simd<2, ui32> = <<3, 7>>
  kgen.param.declare U2: simd<2, ui32> = <apply(:(!kgen.simd<2, ui32>, !kgen.simd<2, ui32>) -> !kgen.simd<2, ui32> @floordiv_ui32, U0, U1)>
  // CHECK: <<2, 1>>
  %2 = kgen.param.constant: !kgen.simd<2, ui32> = <U2>

  kgen.return %0 : !kgen.simd<2, f32>
}

// -----

// COM: Check that identical pop.global_constant values across separate
// COM: interpreter invocations are uniqued to the same MemoryHandleAttr.
kgen.generator @global_const_for_uniquing() -> !kgen.pointer<struct<(array<1, scalar<ui128>>)>> {
  %0 = pop.global_constant: struct<(array<1, scalar<ui128>>)> =
    <{[0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA]}>
  kgen.return %0 : !kgen.pointer<struct<(array<1, scalar<ui128>>)>>
}

kgen.generator @call_global_const_uniquing() {
  // Two separate apply calls to the same generator with the same constant.
  // Both should produce PointerAttrs with the same MemoryHandleAttr.
  kgen.param.declare A: !kgen.pointer<struct<(array<1, scalar<ui128>>)>> =
    <apply(:() -> !kgen.pointer<struct<(array<1, scalar<ui128>>)>> @global_const_for_uniquing)>
  kgen.param.declare B: !kgen.pointer<struct<(array<1, scalar<ui128>>)>> =
    <apply(:() -> !kgen.pointer<struct<(array<1, scalar<ui128>>)>> @global_const_for_uniquing)>

  // CHECK: [[BLOB_UNIQ:#.*]] = #interp.memory_handle<16, "0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA">
  // Both param.constants should reference the same memory handle (uniqued).
  // CHECK: [[BLOB_UNIQ]], const_global
  // CHECK: [[BLOB_UNIQ]], const_global
  %0 = kgen.param.constant: !kgen.pointer<struct<(array<1, scalar<ui128>>)>> = <A>
  %1 = kgen.param.constant: !kgen.pointer<struct<(array<1, scalar<ui128>>)>> = <B>
  kgen.return
}

// -----

// COM: Recursive function-pointer struct field. A struct whose field type
// COM: refers back to the containing struct is lowered with the recursive
// COM: occurrence erased to `pointer<none>` (the "shallow" signature), but the
// COM: function value stored into it keeps its full ("deep") signature. Reading
// COM: the value back during comptime evaluation wraps it in a
// COM: `#kgen.func_ptr_bitcast` that keeps the symbol at its deep type (so it
// COM: still verifies against its declaration) while presenting the shallow
// COM: type the struct field expects.
module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="p:64:64">} {

kgen.generator @recursive_fnptr_bar(%arg0: !kgen.pointer<struct<(index, (!kgen.pointer<struct<(index, pointer<none>) memoryOnly>> owned_in_mem) -> !kgen.none) memoryOnly>> owned_in_mem) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator @recursive_fnptr_init(
    %arg0: index,
    %arg1: !kgen.generator<(!kgen.pointer<struct<(index, (!kgen.pointer<struct<(index, pointer<none>) memoryOnly>> owned_in_mem) -> !kgen.none) memoryOnly>> owned_in_mem) -> !kgen.none>,
    %arg2: !kgen.pointer<struct<(index, (!kgen.pointer<struct<(index, pointer<none>) memoryOnly>> owned_in_mem) -> !kgen.none) memoryOnly>> byref_result) -> !kgen.none {
  %0 = kgen.struct.gep %arg2[0] : <struct<(index, (!kgen.pointer<struct<(index, pointer<none>) memoryOnly>> owned_in_mem) -> !kgen.none) memoryOnly>>
  pop.store %arg0, %0 : !kgen.pointer<index>
  %1 = kgen.struct.gep %arg2[1] : <struct<(index, (!kgen.pointer<struct<(index, pointer<none>) memoryOnly>> owned_in_mem) -> !kgen.none) memoryOnly>>
  %2 = pop.pointer.bitcast %1 : !kgen.pointer<(!kgen.pointer<struct<(index, pointer<none>) memoryOnly>> owned_in_mem) -> !kgen.none> to !kgen.pointer<(!kgen.pointer<struct<(index, (!kgen.pointer<struct<(index, pointer<none>) memoryOnly>> owned_in_mem) -> !kgen.none) memoryOnly>> owned_in_mem) -> !kgen.none>
  pop.store %arg1, %2 : !kgen.pointer<(!kgen.pointer<struct<(index, (!kgen.pointer<struct<(index, pointer<none>) memoryOnly>> owned_in_mem) -> !kgen.none) memoryOnly>> owned_in_mem) -> !kgen.none>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// Evaluating `@recursive_fnptr_init(3, @recursive_fnptr_bar)` into a result slot
// folds to a struct constant whose function-pointer field is a
// `func_ptr_bitcast` of the symbol.
// CHECK-LABEL: kgen.func export @recursive_fnptr_driver
kgen.generator export @recursive_fnptr_driver() {
  // CHECK: kgen.param.constant: struct<(index, (!kgen.pointer<struct<(index, pointer<none>) memoryOnly>> owned_in_mem) -> !kgen.none) memoryOnly> = <{ 3, #kgen.func_ptr_bitcast<#kgen.symbol.constant<@recursive_fnptr_bar> : !kgen.generator<(!kgen.pointer<struct<(index, (!kgen.pointer<struct<(index, pointer<none>) memoryOnly>> owned_in_mem) -> !kgen.none) memoryOnly>> owned_in_mem) -> !kgen.none>> }>
  kgen.param.constant: struct<(index, (!kgen.pointer<struct<(index, pointer<none>) memoryOnly>> owned_in_mem) -> !kgen.none) memoryOnly> = <apply_result_slot(:(index, !kgen.generator<(!kgen.pointer<struct<(index, (!kgen.pointer<struct<(index, pointer<none>) memoryOnly>> owned_in_mem) -> !kgen.none) memoryOnly>> owned_in_mem) -> !kgen.none>, !kgen.pointer<struct<(index, (!kgen.pointer<struct<(index, pointer<none>) memoryOnly>> owned_in_mem) -> !kgen.none) memoryOnly>> byref_result) -> !kgen.none @recursive_fnptr_init, 3, @recursive_fnptr_bar)>
  kgen.return
}

}
