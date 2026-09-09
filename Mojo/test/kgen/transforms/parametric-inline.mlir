// RUN: kgen-opt -inline-param=optimization-level=3 -verify-parameters -split-input-file -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent() {
  // CHECK: hlcf.loop
  // CHECK: hlcf.break "[[LABEL:.*]]" %idx1
  // CHECK: hlcf.break "[[LABEL]]" %idx0
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee() : () -> index
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee() -> index always_inline {
  %cond = "some.cond"() : () -> !kgen.scalar<bool>
  hlcf.if %cond {
    %0 = index.constant 1
    kgen.return %0 : index
  } else {
    hlcf.yield
  }
  %0 = index.constant 0
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
  // CHECK: hlcf.loop
  // CHECK: %[[V:.*]] = "some.producer"
  // CHECK: %[[R0:.*]] = kgen.rebind %[[V]] : !kgen.param<T> to index
  // CHECK-NEXT: hlcf.break "[[LABEL:.*]]" %[[R0]]
  // CHECK: %[[R1:.*]] = kgen.rebind %[[V]] : !kgen.param<T> to index
  // CHECK-NEXT: hlcf.break "[[LABEL]]" %[[R1]]
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee<:type index>() : () -> index
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<T: type>() -> !kgen.param<T> always_inline {
  %0 = "some.producer"() : () -> !kgen.param<T>
  %cond = "some.cond"() : () -> !kgen.scalar<bool>
  hlcf.if %cond {
    kgen.return %0 : !kgen.param<T>
  } else {
    hlcf.yield
  }
  kgen.return %0 : !kgen.param<T>
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
  // CHECK-NEXT: declare A0 = <1>
  // CHECK-NEXT: constant = <A0>
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee<1>() : () -> index
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A>() -> index always_inline {
  %0 = kgen.param.constant = <A>
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A: i64>() {
  // CHECK: declare B: i64 = <A>
  kgen.param.declare B: i64 = <A>
  // CHECK-NEXT: declare A0: i32 = <1>
  // CHECK-NEXT: constant: i32 = <A0>
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee<:i32 1>() : () -> i32
  // CHECK: declare C: i64 = <A>
  kgen.param.declare C: i64 = <A>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A: i32 >() -> i32 always_inline {
  %0 = kgen.param.constant: i32 = <A>
  kgen.return %0 : i32
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() -> index {
  // CHECK-NEXT: declare A0 = <A>
  // CHECK-NEXT: constant = <A0>
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee<A>() : () -> index
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A>() -> index always_inline {
  %0 = kgen.param.constant = <A>
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
  // CHECK: kgen.param.declare.region
  kgen.param.declare.region C = <B>() {
    // CHECK-NEXT: declare A0 = <B>
    // CHECK-NEXT: constant = <A0>
    // CHECK-NOT: kgen.call @callee
    kgen.call @callee<B>() : () -> ()
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A>() always_inline {
  kgen.param.constant = <A>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
  // CHECK: kgen.param.declare.region
  kgen.param.declare.region B = <C>() {
    // CHECK-NEXT: declare A0 = <C>
    // CHECK-NEXT: declare C0 = <A>
    // CHECK-NEXT: constant = <A0>
    // CHECK-NEXT: constant = <C0>
    // CHECK-NOT: kgen.call @callee
    kgen.call @callee<C, A>() : () -> ()
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A, C>() always_inline {
  kgen.param.constant = <A>
  kgen.param.constant = <C>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
  // CHECK: kgen.param.declare.region
  kgen.param.declare.region B = <C>() {
    // CHECK-NEXT: declare A0 = <A>
    // CHECK-NEXT: declare C0 = <C>
    // CHECK-NEXT: constant = <A0>
    // CHECK-NEXT: constant = <C0>
    // CHECK-NOT: kgen.call @callee
    kgen.call @callee<A, C>() : () -> ()
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A, C>() always_inline {
  kgen.param.constant = <A>
  kgen.param.constant = <C>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A, B, C>() {
  // CHECK: kgen.param.declare.region D
  kgen.param.declare.region D = <E>() {
    // CHECK-NEXT: kgen.param.declare.region F
    kgen.param.declare.region F = <G>() {
      // CHECK-NEXT: declare A0 = <A>
      // CHECK-NEXT: declare B0 = <B>
      // CHECK-NEXT: declare C0 = <C>
      // CHECK-NEXT: constant = <A0>
      // CHECK-NEXT: constant = <B0>
      // CHECK-NEXT: constant = <C0>
      // CHECK-NOT: kgen.call @callee
      kgen.call @callee<A, B, C>() : () -> ()
      kgen.return
    }
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A, B, C>() always_inline {
  kgen.param.constant = <A>
  kgen.param.constant = <B>
  kgen.param.constant = <C>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
// CHECK-NOT: kgen.call @callee
  kgen.call @callee<1>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<B>() always_inline {
  kgen.param.declare A = <B>
  kgen.param.constant = <A>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
  // CHECK-NEXT: param.declare B = <1>
  // CHECK-NEXT: param.declare.region A0[A] = ()
  // CHECK: call_param[() -> (): A0]()
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<1>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<B>() always_inline {
  kgen.param.declare.region A = () -> () {
    kgen.return
  }
  kgen.call_param[() -> (): A]()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
  // CHECK-NEXT: declare B0 = <A>
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<A>() : () -> ()
  kgen.param.declare B = <0>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<B>() always_inline {
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<B>() {
  // CHECK-NEXT: declare B0 = <B>
  kgen.call @callee<B>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<B>() always_inline {
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<B>() {
  // CHECK-NEXT: declare A0 = <B>
  // CHECK-NEXT: declare.region F = <C, D>
  // CHECK-NEXT: constant = <C>
  // CHECK-NEXT: constant = <D>
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<B>() : () -> ()
  kgen.param.declare A = <0>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A>() always_inline {
  kgen.param.declare.region F = <C, D>() {
    kgen.param.constant = <C>
    kgen.param.constant = <D>
    kgen.return
  }
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<B>() {
  // CHECK-NEXT: declare A0 = <B>
  // CHECK-NEXT: declare.region F = <C>
  // CHECK-NEXT:   constant = <A0>
  // CHECK-NEXT:   constant = <C>
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<B>() : () -> ()
  kgen.param.declare A = <0>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A>() always_inline {
  kgen.param.declare.region F = <C>() {
    kgen.param.constant = <A>
    kgen.param.constant = <C>
    kgen.return
  }
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<B>() {
  // CHECK-NEXT: declare A = <B>
  kgen.param.declare A = <B>
  // CHECK-NEXT: declare A0 = <B>
  // CHECK-NEXT: declare.region F = <B0>
  // CHECK-NEXT:   constant = <A0>
  // CHECK-NEXT:   constant = <B0>
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<B>() : () -> ()
  // CHECK: constant = <A>
  kgen.param.constant = <A>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A>() always_inline {
  kgen.param.declare.region F = <B>() {
    kgen.param.constant = <A>
    kgen.param.constant = <B>
    kgen.return
  }
  kgen.return
}

// -----

// CHECK-LABEL: @parent
kgen.generator @parent<A>() {
  // CHECK: kgen.param.for A0 in
  // CHECK: kgen.param.constant = <A0>
  kgen.call @callee() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee() always_inline {
  kgen.param.for A in ?
     has_next :() -> i1 ?
     get_next_iter :() -> () ? {
    kgen.param.constant = <A>
    kgen.param.for.continue
  } else {
    kgen.param.yield
  }
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent() {
  // CHECK: declare A = <1>
  kgen.param.declare A = <1>
  // CHECK-NEXT: declare A0 = <2>
  // CHECK-NEXT: declare.region F
  // CHECK-NEXT:   constant = <A0>
  // CHECK-NEXT:   declare.region G
  // CHECK-NEXT:     constant = <A0>
  // CHECK-NOT: declare A = <A0>
  kgen.call @callee() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee() always_inline {
  kgen.param.declare A = <2>
  kgen.param.declare.region F = () {
    kgen.param.constant = <A>
    kgen.param.declare.region G = () {
      kgen.param.constant = <A>
      kgen.return
    }
    kgen.return
  }
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent() {
  // CHECK-NEXT: kgen.param.declare.region F
  kgen.param.declare.region F = () {
    // CHECK-NEXT: kgen.param.declare.region G
    kgen.param.declare.region G = () {
      // CHECK: kgen.param.declare A = <0>
      kgen.call @callee() : () -> ()
      kgen.return
    }
    // CHECK: kgen.param.declare A0 = <0>
    kgen.call @callee() : () -> ()
    kgen.return
  }
  // CHECK: kgen.param.declare A1 = <0>
  kgen.call @callee() : () -> ()
  kgen.return
}

kgen.generator @callee() always_inline {
  kgen.param.declare A = <0>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent() {
kgen.generator @parent() {
  // CHECK-NEXT: kgen.param.if <true> {
  // CHECK-NEXT:   kgen.param.declare A0 = <0>

  // CHECK:        kgen.param.if <false> {
  // CHECK-NEXT:     kgen.param.yield
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     kgen.param.declare A0 = <1>
  kgen.call @callee() : () -> ()
  // CHECK-NOT: kgen.call @callee
  // CHECK: kgen.param.declare A = <0>
  kgen.param.declare A = <0>
  kgen.return
}

// CHECK: kgen.generator @callee
kgen.generator @callee() always_inline {
  kgen.param.if <true> {
    kgen.param.declare A = <0>
    kgen.param.yield
  } else {
    kgen.param.if <false> {
      kgen.param.yield
    } else {
      kgen.param.declare A = <1>
      kgen.param.yield
    }
    kgen.param.yield
  }
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent() {
  // CHECK-NEXT: kgen.param.declare A: i1 = <0>
  kgen.param.declare A: i1 = <0>
  // CHECK-NEXT: kgen.param.declare A1: scalar<bool> = <true>
  // CHECK-NEXT: kgen.param.if <A1> {
  // CHECK-NEXT:   kgen.param.declare A0 = <2>
  // CHECK-NEXT:   kgen.param.yield
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   kgen.param.declare A0: scalar<bool> = <A1>
  // CHECK-NEXT:   kgen.param.if <A0> {
  // CHECK-NEXT:     kgen.param.declare B: scalar<bool> = <A1>
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee() : () -> ()
  kgen.return
}

// CHECK: kgen.generator @callee
kgen.generator @callee() always_inline {
  kgen.param.declare A: scalar<bool> = <true>
  kgen.param.if <A> {
    kgen.param.declare A0 = <2>
    kgen.param.yield
  } else {
    kgen.param.declare A0: scalar<bool> = <A>
    kgen.param.if <A0> {
      kgen.param.declare B: scalar<bool> = <A>
      kgen.param.yield
    } else {
      kgen.param.yield
    } {elseIsolated, thenIsolated}
    kgen.param.yield
  } {elseIsolated, thenIsolated}
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent() {
  // CHECK: declare.region F
    // CHECK-NEXT: hlcf.if
      // CHECK-NEXT: kgen.return
  // CHECK: hlcf.if
  // CHECK-NEXT: hlcf.break "[[LABEL:.*]]"
  kgen.call @callee() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee() always_inline {
  %cond = "some.cond"() : () -> !kgen.scalar<bool>
  kgen.param.declare.region F = () {
    hlcf.if %cond {
      kgen.return
    } else {
      hlcf.yield
    }
    kgen.return
  }
  hlcf.if %cond {
    kgen.return
  } else {
    hlcf.yield
  }
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A: dtype>() {
  %0 = kgen.call @callee<1>() : () -> index
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A>() -> index always_inline {
  kgen.param.declare B = <A>
  %0 = kgen.param.constant = <B>
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: kgen.generator @inline_call_in_if
kgen.generator @inline_call_in_if(%cond: !kgen.scalar<bool>) {
  // CHECK-NEXT: hlcf.if
  hlcf.if %cond {
    // CHECK: inlined.a
    kgen.call @callee() : () -> ()
    hlcf.yield
  } else {
    hlcf.yield
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee() always_inline {
  "inlined.a"() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @inline_call_in_param_if
kgen.generator @inline_call_in_param_if<cond: scalar<bool>>() {
  // CHECK-NEXT: kgen.param.if
  kgen.param.if <cond> {
    // CHECK: inlined.a
    kgen.call @callee() : () -> ()
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee() always_inline {
  "inlined.a"() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @rebind_call_operands
kgen.generator @rebind_call_operands(%arg0: !kgen.scalar<f32>) {
  // CHECK: kgen.param.declare DT: dtype = <f32>
  // CHECK-NEXT: %0 = kgen.rebind %arg0 : !kgen.scalar<f32> to !kgen.scalar<DT>
  // CHECK: pop.simd.extractelement %0[%idx0] : !kgen.scalar<DT>
  kgen.call @callee<:dtype f32>(%arg0) : (!kgen.scalar<f32>) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<DT: dtype>(%arg0: !kgen.scalar<DT>) always_inline {
  %idx0 = index.constant 0
  %0 = pop.simd.extractelement %arg0[%idx0] : !kgen.scalar<DT>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @rebind_mangled_types
kgen.generator @rebind_mangled_types<DT: dtype>(%arg0: !kgen.scalar<DT>) {
  // CHECK: kgen.param.declare DT0: dtype = <DT>
  // CHECK-NEXT: %0 = kgen.rebind %arg0 : !kgen.scalar<DT> to !kgen.scalar<DT0>
  // CHECK: %1 = pop.simd.extractelement %0[%idx0] : !kgen.scalar<DT0>
  // CHECK: %2 = kgen.rebind %1 : !kgen.scalar<DT0> to !kgen.scalar<DT>
  %0 = kgen.call @callee<:dtype DT>(%arg0) : (!kgen.scalar<DT>) -> !kgen.scalar<DT>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<DT: dtype>(%arg0: !kgen.scalar<DT>) -> !kgen.scalar<DT> always_inline {
  %idx0 = index.constant 0
  %0 = pop.simd.extractelement %arg0[%idx0] : !kgen.scalar<DT>
  kgen.return %0 : !kgen.scalar<DT>
}

// -----

// CHECK-LABEL: kgen.generator @replace_in_signature_with_shadow
kgen.generator @replace_in_signature_with_shadow<width>() {
  // CHECK: kgen.param.declare width0 = <width>
  // CHECK-NEXT: kgen.param.declare fn: <index>(!kgen.simd<*(0,0), bool>) -> () = <@param_arg>
  // CHECK-NEXT: kgen.param.declare bound: (!kgen.simd<width0, bool>) -> ()
  // CHECK-SAME: = <bind_params(:<index>(!kgen.simd<*(0,0), bool>) -> () fn, width0)>
  kgen.call @callee<width>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @param_arg
kgen.generator @param_arg<width>(%arg0: !kgen.simd<width, bool>) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<width>() always_inline {
  kgen.param.declare fn: <index>(!kgen.simd<*(0,0), bool>) -> () = <@param_arg>
  kgen.param.declare bound: (!kgen.simd<width, bool>) -> () =
    <bind_params(:<index>(!kgen.simd<*(0,0), bool>) -> () fn, width)>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @dependent_types
kgen.generator @dependent_types() {
  // CHECK-NEXT: declare rank = <4>
  kgen.param.declare rank = <4>
  // CHECK: declare rank1 = <1>
  // CHECK-NEXT: declare shape1: array<rank1, index> = <rebind(:array<1, index> [2])>
  // CHECK: declare rank0 = <rank1>
  // CHECK-NEXT: declare shape0: array<rank0, index> = <rebind(:array<rank1, index> shape1)>
  kgen.call @callee<1, :array<1, index> [2]>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @call_me
kgen.generator @call_me<rank, shape: array<rank, index>>() always_inline {
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<rank, shape: array<rank, index>>() always_inline {
  kgen.call @call_me<rank, :array<rank, index> shape>() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @struct_extract
kgen.generator @struct_extract(%arg0: !kgen.struct<(simd<2, f32>)>) {
  kgen.param.declare size = <1>
  kgen.param.declare type: dtype = <si32>
  // CHECK: kgen.struct.extract %0[0] : <(simd<size0, type0>)>
  kgen.call @callee<2, :dtype f32>(%arg0) : (!kgen.struct<(simd<2, f32>)>) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<size, type: dtype>(%arg0: !kgen.struct<(simd<size, type>)>) always_inline {
  kgen.param.declare cond: scalar<bool> = <true>
  kgen.param.if <cond> {
    %0 = kgen.struct.extract %arg0[0] : !kgen.struct<(simd<size, type>)>
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @only_mangle_mangled_captures
kgen.generator @only_mangle_mangled_captures() {
  kgen.param.declare A = <0>
  // CHECK: constant = <A0>
  // CHECK-NEXT: constant = <B>
  kgen.call @callee<1, 1>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A, B>() always_inline {
  kgen.param.declare.region F = () {
    kgen.param.constant = <A>
    kgen.param.constant = <B>
    kgen.return
  }
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<rank, shape: array<rank, index>>() {
  // CHECK: declare another: array<rank0, index> = <shape0>
  kgen.call @mid<rank, :array<rank, index> shape>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @mid
kgen.generator @mid<rank, shape: array<rank, index>>() always_inline {
  kgen.param.declare another: array<rank, index> = <shape>
  kgen.return
}

// -----

// COM: https://github.com/modularml/modular/issues/8586

kgen.generator @unroll<func: <index>() -> ()>() always_inline {
  kgen.param.constant: () -> () = <bind_params(:<index>() -> () func, 1)>
  kgen.return
}

kgen.generator @nested_func_call<func: () -> ()>() always_inline {
  kgen.param.declare.region func_wrapper = () {
    kgen.param.declare.region nested_func = <idx>() {
      kgen.call_param[() -> (): func]()
      kgen.return
    }
    kgen.call @unroll<:<index>() -> () nested_func>() : () -> ()
    kgen.return
  }
  kgen.call_param[() -> (): func_wrapper]()
  kgen.return
}

kgen.generator @pass_it() always_inline {
  kgen.param.declare.region id = () {
    kgen.return
  }
  kgen.call @nested_func_call<:() -> () id>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @main
kgen.generator @main() {
  // CHECK: kgen.param.declare.region id
  // CHECK: kgen.param.declare func1: () -> () = <id>
  // CHECK: kgen.param.declare.region func_wrapper = () {
    // CHECK: kgen.param.declare.region nested_func = <idx>() {
      // CHECK: kgen.call_param[() -> (): func1]
    // CHECK: kgen.param.declare func2: <index>() -> () = <nested_func>
    // CHECK: kgen.param.constant: () -> () = <bind_params(:<index>() -> () func2, 1)>
  // CHECK: kgen.call_param[() -> (): func_wrapper]
  kgen.call @pass_it() : () -> ()

  // CHECK: kgen.param.declare.region id0
  // CHECK: kgen.param.declare func3: () -> () = <id0>
  // CHECK: kgen.param.declare.region func_wrapper0[func_wrapper] = () {
    // CHECK: kgen.param.declare.region nested_func0[nested_func] = <idx0>() {
      // CHECK: kgen.call_param[() -> (): func3]
    // CHECK: kgen.param.declare func4: <index>() -> () = <nested_func0>
    // CHECK: kgen.param.constant: () -> () = <bind_params(:<index>() -> () func4, 1)>
  // CHECK: kgen.call_param[() -> (): func_wrapper0]
  kgen.call @pass_it() : () -> ()
  kgen.return
}

// -----

// COM: Give up on recursive elaboration instead of emitting an error.

kgen.generator @passthrough<cond: scalar<bool>>() always_inline {
  kgen.call @recursive<:scalar<bool> cond>() : () -> ()
  kgen.return
}

kgen.generator @recursive<cond: scalar<bool>>() always_inline {
  kgen.param.if <cond> {
    kgen.call @passthrough<:scalar<bool> false>() : () -> ()
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @root
kgen.generator @root() {
  // CHECK-NEXT: kgen.call @recursive
  kgen.call @recursive<:scalar<bool> true>() : () -> ()
  kgen.return
}

// -----

// COM: This is testing a DenseMap invalidation for nested parameter scopes.
// COM: https://github.com/modularml/modular/issues/10174

kgen.generator @deeply_nested_paramif<value>() always_inline {
  kgen.param.if<eq(value, 0)> {
    kgen.param.yield
  } else {
    kgen.param.if<eq(value, 1)> {
      kgen.param.yield
    } else {
      kgen.param.if<eq(value, 2)> {
        kgen.param.yield
      } else {
        kgen.param.if<eq(value, 3)> {
          kgen.param.yield
        } else {
          kgen.param.if<eq(value, 4)> {
            kgen.param.yield
          } else {
            kgen.param.if<eq(value, 5)> {
              kgen.param.yield
            } else {
              kgen.param.if<eq(value, 6)> {
                kgen.call @deeply_nested_paramif_0<1>() : () -> ()
                kgen.param.yield
              } else {
                kgen.param.if<eq(value, 7)> {
                  kgen.param.yield
                } else {
                  kgen.param.if<eq(value, 8)> {
                    kgen.param.yield
                  } else {
                    kgen.param.if<eq(value, 9)> {
                      kgen.param.yield
                    } else {
                      kgen.param.if<eq(value, 10)> {
                        kgen.param.yield
                      } else {
                        kgen.param.if<eq(value, 11)> {
                          kgen.param.yield
                        } else {
                          kgen.param.yield
                        }
                        kgen.param.yield
                      }
                      kgen.param.yield
                    }
                    kgen.param.yield
                  }
                  kgen.param.yield
                }
                kgen.param.yield
              }
              kgen.param.yield
            }
            kgen.param.yield
          }
          kgen.param.yield
        }
        kgen.param.yield
      }
      kgen.param.yield
    }
    kgen.param.yield
  }
  kgen.return
}

kgen.generator @deeply_nested_paramif_0<value>() always_inline {
  kgen.param.if<eq(value, 0)> {
    kgen.param.yield
  } else {
    kgen.param.if<eq(value, 1)> {
      kgen.param.yield
    } else {
      kgen.param.if<eq(value, 2)> {
        kgen.param.yield
      } else {
        kgen.param.if<eq(value, 3)> {
          kgen.param.yield
        } else {
          kgen.param.if<eq(value, 4)> {
            kgen.param.yield
          } else {
            kgen.param.if<eq(value, 5)> {
              kgen.param.yield
            } else {
              kgen.param.if<eq(value, 6)> {
                kgen.param.yield
              } else {
                kgen.param.if<eq(value, 7)> {
                  kgen.param.yield
                } else {
                  kgen.param.if<eq(value, 8)> {
                    kgen.param.yield
                  } else {
                    kgen.param.if<eq(value, 9)> {
                      kgen.param.yield
                    } else {
                      kgen.param.if<eq(value, 10)> {
                        kgen.param.yield
                      } else {
                        kgen.param.if<eq(value, 11)> {
                          kgen.param.yield
                        } else {
                          kgen.param.yield
                        }
                        kgen.param.yield
                      }
                      kgen.param.yield
                    }
                    kgen.param.yield
                  }
                  kgen.param.yield
                }
                kgen.param.yield
              }
              kgen.param.yield
            }
            kgen.param.yield
          }
          kgen.param.yield
        }
        kgen.param.yield
      }
      kgen.param.yield
    }
    kgen.param.yield
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @call_it
kgen.generator @call_it() {
  // CHECK: kgen.param.if
  kgen.call @deeply_nested_paramif<10>() : () -> ()
  kgen.return
}

// -----

kgen.generator @pass_it() always_inline {
  kgen.param.declare value: i32 = <1>
  kgen.param.declare.region f = () {
    kgen.param.declare.region g = () {
      kgen.param.constant: i32 = <value>
      kgen.return
    }
    kgen.param.declare value0 = <1>
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @main
kgen.generator @main() {
  // CHECK: declare value: f32
  kgen.param.declare value: f32 = <1.0>
  // CHECK: declare value1: i32 = <1>
  // CHECK: kgen.param.constant: i32 = <value1>
  // CHECK: declare value0 = <1>
  kgen.call @pass_it() : () -> ()

  kgen.return
}

// -----

// COM: This test case tricks a simple counter uniquer into mangling two
// COM: parameter decls into the same name.

kgen.generator @inline_me() always_inline {
  kgen.param.declare value = <1>
  kgen.param.declare value0 = <1>
  kgen.param.declare value1 = <1>
  kgen.param.declare value2 = <1>
  kgen.param.declare value3 = <1>
  kgen.param.declare value4 = <1>
  kgen.param.declare value5 = <1>
  kgen.param.declare value6 = <1>
  kgen.param.declare value7 = <1>
  kgen.param.declare value8 = <1>
  kgen.param.declare value9 = <1>
  kgen.return
}

// CHECK-LABEL: kgen.generator @entry
kgen.generator @entry() {
  // CHECK: declare value10 =
  // CHECK: declare value11 =
  kgen.param.declare value1 = <1>
  kgen.param.declare value = <1>
  kgen.call @inline_me() : () -> ()
  kgen.return
}

// -----

kgen.generator @unreachable_and_early_ret() always_inline {
  %true = kgen.param.constant: scalar<bool> = <true>
  hlcf.if %true {
    kgen.return
  } else {
    hlcf.yield
  }
  kgen.unreachable
}

// CHECK-LABEL: kgen.generator @call_it
kgen.generator @call_it() {
  // CHECK-NEXT: hlcf.loop
    // CHECK: hlcf.if
      // CHECK-NEXT: hlcf.break
    // CHECK: kgen.unreachable
  // CHECK-NEXT: }
  // CHECK-NEXT: kgen.return
  kgen.call @unreachable_and_early_ret() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @dontinlineme() -> index
kgen.func @dontinlineme() -> index always_inline {
  // CHECK-NEXT: index.constant
  %0 = index.constant 3
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @caller
kgen.generator @caller() -> index {
  // CHECK-NEXT: kgen.call @dontinlineme
  %0 = kgen.call @dontinlineme() : () -> index
  kgen.return %0 : index
}

// -----

kgen.generator @recursive() always_inline_no_debug {
  kgen.call @recursive() : () -> ()
  kgen.return
}

kgen.generator @trivial() always_inline_no_debug {
  kgen.return
}

// CHECK-LABEL: kgen.generator @top
kgen.generator @top() {
  // CHECK-NEXT: call @recursive
  kgen.call @recursive() : () -> ()
  // CHECK-NOT: call @trivial
  kgen.call @trivial() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @inline_heuristic
kgen.generator @inline_heuristic<A>() {
  // CHECK: %[[V:.*]] = "some.producer"
  // CHECK: %[[R0:.*]] = kgen.rebind %[[V]] : !kgen.param<T> to index
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee<:type index>() : () -> index
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<T: type>() -> !kgen.param<T> always_inline {
  %0 = "some.producer"() : () -> !kgen.param<T>
  kgen.return %0 : !kgen.param<T>
}
