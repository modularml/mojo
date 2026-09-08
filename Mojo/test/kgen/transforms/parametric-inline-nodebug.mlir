// RUN: kgen-opt %s -inline-param="nodebug-only=true optimization-level=2" -mlir-print-debuginfo -split-input-file -allow-unregistered-dialect | FileCheck %s

// SourceLoc immediate
kgen.generator @wrap_source_loc_0() -> !kgen.none always_inline_no_debug {
  %line, %col, %fileName = kgen.source_loc[0]
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator @wrap_source_loc_1() -> !kgen.none always_inline_no_debug {
  %line, %col, %fileName = kgen.source_loc[1]
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator @test_wrap_source_loc_0
kgen.generator @test_wrap_source_loc_0() -> !kgen.none always_inline_no_debug {
  // CHECK: kgen.source_loc[-1]
  // CHECK-NOT: kgen.call
  %0 = kgen.call @wrap_source_loc_0() : () -> !kgen.none loc("some_file.mojo":4:6)
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator @call_wrapped_source_loc_1
kgen.generator @call_wrapped_source_loc_1() -> !kgen.none always_inline_no_debug {
  // CHECK: kgen.source_loc[0]
  // CHECK-NOT: kgen.call
  %0 = kgen.call @wrap_source_loc_1() : () -> !kgen.none
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator @test_wrapped_source_loc_1
kgen.generator @test_wrapped_source_loc_1() -> !kgen.none {
  // CHECK: kgen.source_loc[-1]
  // CHECK-NOT: kgen.call
  %0 = kgen.call @call_wrapped_source_loc_1() : () -> !kgen.none loc("other_file.mojo":10:12)
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator @test_wrapped_source_loc_1_inlined
kgen.generator @test_wrapped_source_loc_1_inlined() -> !kgen.none always_inline_no_debug {
  // CHECK: kgen.source_loc[-1]
  // CHECK-NOT: kgen.call
  %0 = kgen.call @call_wrapped_source_loc_1() : () -> !kgen.none loc("another_file.mojo":42:13)
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// SourceLoc parametric
kgen.generator @wrap_source_loc_param<depth: index>() -> !kgen.none always_inline_no_debug {
  %line, %col, %fileName = kgen.source_loc[depth]
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator @call_wrapped_source_loc_param
kgen.generator @call_wrapped_source_loc_param() -> !kgen.none always_inline_no_debug {
  // CHECK: kgen.param.declare depth = <1>
  // CHECK: kgen.source_loc[to_builtin(:scalar<index> add(from_builtin(depth), -1))]
  // CHECK-NOT: kgen.call
  %0 = kgen.call @wrap_source_loc_param<1>() : () -> !kgen.none
  kgen.return %0 : !kgen.none
}

// CHECK-LABEL: kgen.generator @test_wrap_source_loc_param
kgen.generator @test_wrap_source_loc_param() -> !kgen.none always_inline_no_debug {
  // CHECK: kgen.param.declare [[DEPTH:.+]] = <1>
  // CHECK: kgen.source_loc[to_builtin(:scalar<index> add(from_builtin(depth), -2))]
  // CHECK-NOT: kgen.call
  %0 = kgen.call @call_wrapped_source_loc_param() : () -> !kgen.none loc("some_file.mojo":4:6)
  // CHECK: kgen.param.declare [[DEPTH:.+]] = <1>
  // CHECK: kgen.source_loc[to_builtin(:scalar<index> add(from_builtin(depth0), -3))]
  // CHECK-NOT: kgen.call
  %1 = kgen.call @call_wrapped_source_loc_param() : () -> !kgen.none loc(callsite("some_file.mojo":4:6 at "some_other_file.mojo":5:7))
  kgen.return %0 : !kgen.none
}

kgen.generator @nodebug_inline_me() always_inline_no_debug {
  kgen.param.constant = <1>
  kgen.param.declare.region A = () -> () {
    kgen.return loc(#loc1)
  } loc(#loc1)
  kgen.return
}

kgen.generator @always_inline() always_inline {
  // Inflate function size so not impacted by heuristic.
  kgen.param.declare value1 = <1>
  kgen.param.declare value2 = <1>
  kgen.param.declare value3 = <1>
  kgen.param.declare value4 = <1>
  kgen.return
}

#loc1 = loc("foo.mlir":10:5)
#loc2 = loc("bar.mlir":12:7)
#locInlined = loc(callsite(#loc1 at #loc2))

// CHECK-LABEL: kgen.generator @main
kgen.generator @main() {
  // CHECK-NEXT: kgen.param.constant = <1> loc(#[[LOC_INLINED:.*]])
  // CHECK-NEXT: kgen.param.declare.region A = () {
  // CHECK-NEXT:   kgen.return loc(#[[LOC_CALLEE:.*]])
  // CHECK-NEXT: } {isolated} loc(#[[LOC_CALLEE]])
  kgen.call @nodebug_inline_me() : () -> () loc(#locInlined)
  // CHECK-NEXT: call @always_inline
  kgen.call @always_inline() : () -> ()
  kgen.return
}

// CHECK-DAG: #[[LOC_CALLEE]] = loc("foo.mlir":10:5)
// CHECK-DAG: #[[LOC_INLINED]] = loc("{{.*}}":

// -----

#subprogram = #debuginfo.subprogram<sourceName = <"fake_larger_callee">> : !debuginfo.subroutine<(!debuginfo.unresolved<!kgen.param<T>>) -> (!debuginfo.unresolved<!kgen.param<T>>): DW_CC_normal>
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "arg0"> : !debuginfo.unresolved<!kgen.param<T>>

// CHECK-LABEL: kgen.generator @inline_heuristic
kgen.generator @inline_heuristic<A>(%arg: index) {
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee<:type index>(%arg) : (index) -> index
  // CHECK: kgen.call @larger_callee
  %1 = kgen.call @larger_callee<:type index>(%arg) : (index) -> index
  // CHECK: debuginfo.value
  // CHECK-NOT: kgen.call @fake_larger_callee
  %2 = kgen.call @fake_larger_callee<:type index>(%arg) : (index) -> index
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<T: type>(%arg: !kgen.param<T>) -> !kgen.param<T> always_inline {
  kgen.return %arg : !kgen.param<T>
}

// CHECK-LABEL: kgen.generator @larger_callee
kgen.generator @larger_callee<T: type>(%arg: !kgen.param<T>) -> !kgen.param<T> always_inline {
  kgen.param.declare value1 = <1>
  kgen.param.declare value2 = <1>
  kgen.param.declare value3 = <1>
  kgen.param.declare value4 = <1>
  kgen.return %arg : !kgen.param<T>
}

// CHECK-LABEL: kgen.generator @fake_larger_callee
// This function looks large but is just full of debug ops.
kgen.generator @fake_larger_callee<T: type>(%arg: !kgen.param<T>) -> !kgen.param<T> always_inline {
  debuginfo.value #local_variable = %arg : !kgen.param<T>
  debuginfo.value #local_variable = %arg : !kgen.param<T>
  debuginfo.value #local_variable = %arg : !kgen.param<T>
  debuginfo.kill #local_variable
  kgen.return %arg : !kgen.param<T>
}
