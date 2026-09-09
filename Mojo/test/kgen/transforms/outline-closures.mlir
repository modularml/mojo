// RUN: kgen-opt %s -verify-parameters -allow-unregistered-dialect -outline-closures | kgen-opt -allow-unregistered-dialect | FileCheck %s

// COM: This shouldn't change at all, save for whatever canonicalizations happen at parse time.
// CHECK-LABEL: @call_region<fn: <index>() capturing -> index>() -> index
kgen.generator @call_region<fn: <index>() capturing ->index>() -> index {
  // CHECK-NEXT: kgen.param.declare BoundFn: () capturing -> index = <bind_params(:<index>() capturing -> index fn, 2)>
  kgen.param.declare BoundFn: () capturing -> index = <bind_params(:<index>() capturing -> index fn, 2)>
  // CHECK-NEXT: %0 = kgen.call_param[() capturing -> index: BoundFn]()
  %0 = kgen.call_param[() capturing -> index: BoundFn]()
  // CHECK-NEXT: kgen.return %0 : index
  kgen.return %0 : index
}

// CHECK-LABEL: @call_region_2_args<fn: <index>(index, index) capturing -> index>() -> index
kgen.generator @call_region_2_args<fn: <index>(index, index) capturing ->index>() -> index {
  // CHECK-NEXT: %[[CST:.*]] = index.constant 0
  %cst = index.constant 0
  // CHECK-NEXT: kgen.param.declare BoundFn: (index, index) capturing -> index = <bind_params(:<index>(index, index) capturing -> index fn, 2)>
  kgen.param.declare BoundFn: (index, index) capturing -> index = <bind_params(:<index>(index, index) capturing -> index fn, 2)>
  // CHECK-NEXT: %[[CALL:.*]] = kgen.call_param[(index, index) capturing -> index: BoundFn](%[[CST]], %[[CST]])
  %0 = kgen.call_param[(index, index) capturing -> index: BoundFn](%cst, %cst)
  // CHECK-NEXT: kgen.return %[[CALL]] : index
  kgen.return %0 : index
}

// COM: This is the region hoisted out into a generator.
// COM: This is the wrapper that loads values from the global variable.
// CHECK-LABEL: kgen.generator @raiseClosure_Fn<Jefffffffffff, C, A, B>(%arg0: index, %arg1: index) capturing -> index
// CHECK-SAME{LITERAL}:   LLVMArgMetadataArray = [[], ["llvm.someattr", 3 : index]]
// CHECK-SAME:            LLVMMetadataArray = ["llvm.someattr", 4 : index]
// CHECK-SAME:            sourceName = "my_nested_func"
// CHECK-NEXT:   [[ARG0:%.*]] = pop.compiler.global_load "raiseClosure_context_var_0" : index
// CHECK-NEXT:   [[ARG1:%.*]] = pop.compiler.global_load "raiseClosure_context_var_1" : index
// CHECK-NEXT:   [[CST:%.*]] = kgen.param.constant = <to_builtin(:scalar<index> add(mul(from_builtin(B), from_builtin(Jefffffffffff), -1), mul(from_builtin(A), from_builtin(Jefffffffffff)), mul(from_builtin(C), from_builtin(Jefffffffffff))))>
// CHECK-NEXT:   [[RESULT:%.*]] = index.add [[ARG0]], [[ARG1]]
// CHECK-NEXT:   kgen.return [[RESULT]]

// CHECK-LABEL: kgen.generator @raiseClosure
kgen.generator @raiseClosure<Jefffffffffff>(%arg0: index) -> index {
  %cst = index.constant 0
  // CHECK: pop.compiler.global_store "raiseClosure_context_var_0", %idx0 : index
  // CHECK-NEXT: pop.compiler.global_store "raiseClosure_context_var_1", %arg0 : index
  kgen.param.declare C = <15>
  kgen.param.declare.region Fn[my_nested_func] = <A, B>(%a: index, %b: index) capturing -> index {
    %0 = kgen.param.constant = <mul(add(sub(A, B), C), Jefffffffffff)>
    %1 = index.add %cst, %arg0
    kgen.return %1 : index
  } {
    LLVMArgMetadataArray = [[], ["llvm.someattr", 3 : index]],
    LLVMMetadataArray = ["llvm.someattr", 4 : index]
  }
  // CHECK: kgen.param.declare Fn: <index, index>(index, index) capturing -> index = <@raiseClosure_Fn<Jefffffffffff, C, ?, ?>>
  // CHECK: kgen.param.declare BoundFn: <index>(index, index) capturing -> index = <bind_params(:<index, index>(index, index) capturing -> index Fn, ?, 1)>
  kgen.param.declare BoundFn: <index>(index, index) capturing -> index = <bind_params(:<index, index>(index, index) capturing -> index Fn, ?, 1)>
  // CHECK: kgen.call @call_region_2_args<:<index>(index, index) capturing -> index BoundFn>() : () -> index
  %0 = kgen.call @call_region_2_args<:<index>(index, index) capturing -> index BoundFn>() : () -> index
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @raise2Closures_Empty()
// CHECK-NEXT:    kgen.return

// CHECK-LABEL: kgen.generator @raise2Closures_Fn<C, A>() capturing -> index
// CHECK-NEXT:    %[[ARG0:.*]] = pop.compiler.global_load "raise2Closures_context_var_0" : index
// CHECK-NEXT:    %[[V0:.*]] = kgen.param.constant = <to_builtin(:scalar<index> add(from_builtin(A), from_builtin(C)))>
// CHECK-NEXT:    kgen.return %[[ARG0]] : index


// CHECK-LABEL: kgen.generator @raise2Closures
kgen.generator @raise2Closures() {
  %cst = index.constant 0
  // CHECK: index.constant 0
  kgen.param.declare C = <15>

  // CHECK: kgen.param.declare Empty: () -> () = <@raise2Closures_Empty>
  kgen.param.declare.region Empty = () -> () {
    kgen.return
  }

  // CHECK-NEXT: pop.compiler.global_store "raise2Closures_context_var_0", %idx0 : index
  // CHECK-NEXT: kgen.param.declare Fn: <index>() capturing -> index = <@raise2Closures_Fn<C, ?>>
  kgen.param.declare.region Fn = <A>() capturing -> index {
    %0 = kgen.param.constant = <add(A, C)>
    kgen.return %cst : index
  }

  // CHECK: kgen.call @call_region<:<index>() capturing -> index Fn>() : ()  -> index
  %0 = kgen.call @call_region<:<index>() capturing -> index Fn>() : () -> index
  // CHECK: kgen.return
  kgen.return
}

// CHECK-LABEL: kgen.generator @parametrizedClosure_Fn<T: type>() capturing -> !kgen.param<T>
// CHECK-NEXT:    %0 = pop.compiler.global_load "parametrizedClosure_context_var_0" : !kgen.param<T>
// CHECK-NEXT:    kgen.return %0 : !kgen.param<T>

// CHECK-LABEL: kgen.generator @parametrizedClosure<T: type>(%arg0: !kgen.param<T>) -> !kgen.param<T>
// CHECK-NEXT:    pop.compiler.global_store "parametrizedClosure_context_var_0", %arg0 : !kgen.param<T>
// CHECK-NEXT:    kgen.param.declare Fn: () capturing -> !kgen.param<T> = <@parametrizedClosure_Fn<:type T>>
// CHECK-NEXT:    %0 = kgen.call_param[() capturing -> !kgen.param<T>: Fn]()
// CHECK-NEXT:    kgen.return %0 : !kgen.param<T>

// CHECK-LABEL: kgen.generator @raiseParamClosure() -> f32
// CHECK-NEXT:    %simd = kgen.param.constant: scalar<f32> = <"0">
// CHECK-NEXT:    %0 = pop.cast_to_builtin %simd : !kgen.scalar<f32> to f32
// CHECK-NEXT:    %1 = kgen.call @parametrizedClosure<:type f32>(%0) : (f32) -> f32
// CHECK-NEXT:    kgen.return %1 : f32


kgen.generator @parametrizedClosure<T: type>(%arg0: !kgen.param<T>) -> !kgen.param<T> {
  kgen.param.declare.region Fn = () capturing -> !kgen.param<T> {
    kgen.return %arg0 : !kgen.param<T>
  }
  %1 = kgen.call_param[() capturing -> !kgen.param<T>: Fn]()
  kgen.return %1 : !kgen.param<T>
}

kgen.generator @raiseParamClosure() -> f32 {
  %0 = kgen.param.constant : !kgen.scalar<f32> = <<"0.000000e+00">>
  %1 = pop.cast_to_builtin %0 : !kgen.scalar<f32> to f32
  %2 = kgen.call @parametrizedClosure<:type f32>(%1) : (f32) -> (f32)
  kgen.return %2 : f32
}

// CHECK-LABEL: @useAfterDef
kgen.generator @useAfterDef() -> index {
  %cst = index.constant 0
  // CHECK: index.constant 0
  kgen.param.declare C = <15>

  // CHECK: kgen.call @call_region<:<index>() capturing -> index Fn>() : () -> index
  %call = kgen.call @call_region<:<index>() capturing -> index Fn>() : () -> index

  // CHECK-NEXT: pop.compiler.global_store "useAfterDef_context_var_0", %idx0 : index
  // CHECK-NEXT: kgen.param.declare Fn: <index>() capturing -> index = <@useAfterDef_Fn<C, ?>>
  kgen.param.declare.region Fn = <A>() capturing -> index {
    %0 = kgen.param.constant = <add(A, C)>
    kgen.return %cst : index
  }

  // CHECK: kgen.return
  kgen.return %call : index
}

// CHECK-LABEL: @nested
kgen.generator @nested(%pred: !kgen.scalar<bool>) -> index {
  kgen.param.declare C = <15>

  // CHECK: hlcf.if
  %if = hlcf.if %pred -> index {
    %cst = index.constant 0
    // CHECK-NEXT: index.constant 0

    // CHECK-NEXT: kgen.call @call_region<:<index>() capturing -> index Fn>() : () -> index
    %call = kgen.call @call_region<:<index>() capturing -> index Fn>() : () -> index

    // CHECK-NEXT: pop.compiler.global_store "nested_context_var_0", %idx0 : index
    // CHECK-NEXT: kgen.param.declare
    kgen.param.declare.region Fn = <A>() capturing -> index {
      %0 = kgen.param.constant = <add(A, C)>
      kgen.return %cst : index
    }
    hlcf.yield %call : index
  } else {
    %cst = index.constant 0
    hlcf.yield %cst : index
  }

  // CHECK: kgen.param.declare Empty: () -> () = <@nested_Empty>
  kgen.param.declare.region Empty = () -> () {
    kgen.return
  }

  // CHECK: kgen.return
  kgen.return %if : index
}

// CHECK-LABEL: @nested2
kgen.generator @nested2() -> index {
  %cst = index.constant 0
  // CHECK: index.constant
  kgen.param.declare C = <15>

  // CHECK: hlcf.loop
  %res = hlcf.loop (%input = %cst: index) -> index {
    // CHECK-NEXT: pop.compiler.global_store "nested2_context_var_0", %arg0 : index
    // CHECK-NEXT: pop.compiler.global_store "nested2_context_var_1", %idx0 : index
    // CHECK-NEXT: kgen.param.declare
    kgen.param.declare.region Fn = <A>() capturing -> index {
      %5 = index.add %input, %cst
      kgen.return %5 : index
    }
    // CHECK-NEXT: kgen.call @call_region<:<index>() capturing -> index Fn>() : () -> index
    %call = kgen.call @call_region<:<index>() capturing -> index Fn>() : () -> index
    hlcf.break %call : index
  }

  // CHECK: kgen.return
  kgen.return %res : index
}

// CHECK-LABEL: kgen.generator @capture_crosses_parameter_domain
kgen.generator @capture_crosses_parameter_domain<T: type>(%arg0: !kgen.param<T>) {
  // CHECK: declare Fn: <index, type>
  kgen.param.declare.region Fn = <A, S: type>() capturing -> !kgen.param<T> {
    kgen.return %arg0: !kgen.param<T>
  }
  kgen.return
}

// COM: We have to parametrize the wrapper on captured SSA values as well, check that this actually happens.
// CHECK-LABEL: @parametrizedSSACapture_fn<T: type>
kgen.generator @parametrizedSSACapture<T: type>(%arg0 : !kgen.param<T>) -> index {
  %0 = kgen.call_param[() capturing -> index: fn]()
  // CHECK: kgen.param.declare fn: () capturing -> index = <@parametrizedSSACapture_fn<:type T>>
  kgen.param.declare.region fn = () capturing -> index {
    "op.use"(%arg0) : (!kgen.param<T>) -> ()
    %1 = kgen.param.constant = <0>
    kgen.return %1 : index
  }
  kgen.return %0 : index
}

// COM: We should not try and capture parameters.
// CHECK-LABEL: @dontBindInputParameters_fn<T: type, N>
kgen.generator @dontBindInputParameters<T: type, I>(%arg0 : !kgen.param<T>) -> index {
  %0 = kgen.call_param[() capturing -> index: bind_params(:<index>() capturing -> index fn, I)]()
  // CHECK: kgen.param.declare fn: <index>() capturing -> index = <@dontBindInputParameters_fn<:type T, ?>>
  kgen.param.declare.region fn = <N>() capturing -> index {
    %1 = kgen.param.constant = <N>
    "use.op"(%arg0) : (!kgen.param<T>) -> ()
    kgen.return %1 : index
  }
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @innermostCapturesThroughMid_Bot<A>
// CHECK-LABEL: kgen.generator @innermostCapturesThroughMid_Mid<A>
// CHECK-LABEL: kgen.generator @innermostCapturesThroughMid<A>
// CHECK-NEXT: @innermostCapturesThroughMid_Mid<A>

kgen.generator @innermostCapturesThroughMid<A>() {
  kgen.param.declare.region Mid = () {
    kgen.param.declare.region Bot = () {
      kgen.param.constant = <A>
      kgen.return
    }
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @paramCaptureNestedInParamRefType_Fn<N, Vs: array<N, i32>>
// CHECK: constant: array<N, i32> = <Vs>
// CHECK: declare Fn: () -> () = <@paramCaptureNestedInParamRefType_Fn<N, :array<N, i32> Vs>>

kgen.generator @paramCaptureNestedInParamRefType<N, Vs: array<N, i32>>() {
  kgen.param.declare.region Fn = () {
    kgen.param.constant: array<N, i32> = <Vs>
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: @left_to_right_dependency_CaptureThemAll
// CHECK-SAME: <A, F: type, G: type, H: type, I: type, J: type,
// CHECK-SAME:  L: array<A, struct<(F, G, H, I, J)>>, B: type,
// CHECK-SAME:  E: array<A, array<A, array<A, B>>>,
// CHECK-SAME:  D: array<A, array<A, B>>, C: array<A, B>
kgen.generator @left_to_right_dependency<
    A, B: type, C: array<A, B>, D: array<A, array<A, B>>,
    E: array<A, array<A, array<A, B>>>,
    F: type, G: type, H: type, I: type, J: type,
    K: struct<(F, G, H, I, J)>, L: array<A, struct<(F, G, H, I, J)>>>() {
  kgen.param.declare.region CaptureThemAll = () {
    "use"() {
      a = #kgen.param.decl.ref<"L"> : !pop.array<A, struct<(F, G, H, I, J)>>,
      b = #kgen.param.decl.ref<"E"> : !pop.array<A, array<A, array<A, B>>>,
      c = #kgen.param.decl.ref<"D"> : !pop.array<A, array<A, B>>,
      d = #kgen.param.decl.ref<"C"> : !pop.array<A, B>
    } : () -> ()
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @dependent_outline<a>
kgen.generator @dependent_outline<a>() {
  // CHECK-NEXT: kgen.param.declare fn: <type>(!pop.array<a, *(0,0)>) -> () =
  // CHECK-SAME: <@dependent_outline_fn<a, :type ?>>
  kgen.param.declare.region fn = <b: type>(%arg0: !pop.array<a, b>) {
    kgen.return
  }
  kgen.return
}


// CHECK-LABEL: kgen.generator @two_nested_closures
kgen.generator @two_nested_closures(%arg0: index, %arg1: index) {
  // CHECK: global_store "two_nested_closures_context_var_0", %arg0 : index
  // CHECK: global_store "two_nested_closures_context_var_1", %arg1 : index
  kgen.param.declare.region f1 = () capturing -> index {
    kgen.return %arg0 : index
  }
  kgen.param.declare.region f2 = () capturing -> index {
    kgen.return %arg1 : index
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @feras_F() -> index
// CHECK-LABEL: kgen.generator @feras_G<F: () -> index>() capturing
// CHECK-LABEL: kgen.generator @feras()
kgen.generator @feras() {
  // CHECK-NEXT: declare F: () -> index = <@feras_F>
  kgen.param.declare.region F = () -> index {
    kgen.unreachable
  }
  // CHECK-NEXT: constant: array<apply(:() -> index F)
  %0 = kgen.param.constant: array<apply(:() -> index F), index> = <?>
  // CHECK-NEXT: global_store
  // CHECK-NEXT: declare G: () capturing -> () = <@feras_G<:() -> index F>>
  kgen.param.declare.region G = () capturing {
    "use"(%0) : (!pop.array<apply(:() -> index F), index>) -> ()
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @lifetime_markers_hack_closure
// CHECK: [[VAR:%.*]] = pop.stack_allocation
// CHECK-NEXT: lifetime.start([[VAR]])
// CHECK-NEXT: lifetime.end([[VAR]])

// CHECK-LABEL: kgen.generator @lifetime_markers_hack
kgen.generator @lifetime_markers_hack() {
  %0 = pop.stack_allocation 1 x index marked
  kgen.param.declare.region closure = () capturing {
    %1 = pop.stack_allocation 1 x index marked
    pop.stack_alloc.lifetime.start(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>
    pop.stack_alloc.lifetime.end(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>
    kgen.return
  }
  kgen.return
}
