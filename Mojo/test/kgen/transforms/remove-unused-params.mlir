
// RUN: kgen-opt --split-input-file --remove-unused-params --eliminate-dead-symbols %s  | FileCheck %s

kgen.generator @basic_arg_remove_1<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>,%arg2: !kgen.scalar<T>) -> index{
  %l = pop.load %arg1 : !kgen.pointer<index>
  kgen.return %l : index
}

kgen.generator @basic_arg_remove_2<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>, %arg2: !kgen.scalar<T>) -> index{
  %0 = kgen.call @basic_arg_remove_1<:dtype T>(%arg0, %arg1,%arg2) : (index, !kgen.pointer<index>, !kgen.scalar<T>) -> (index)
  kgen.return %0 : index
}

kgen.generator export @basic_arg_export<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>, %arg2: !kgen.scalar<T>) {
  %0 = kgen.call @basic_arg_remove_2<:dtype T>(%arg0, %arg1, %arg2) : (index, !kgen.pointer<index>, !kgen.scalar<T>) -> (index)
  kgen.return
}

// CHECK-LABEL: kgen.generator  export @basic_arg_export<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>, %arg2: !kgen.scalar<T>) {
// CHECK-NEXT: kgen.call @basic_arg_remove_2_REMOVED_ARG(%arg1) : (!kgen.pointer<index>) -> index

// CHECK-LABEL:  kgen.generator  @basic_arg_remove_1_REMOVED_ARG(%arg0: !kgen.pointer<index>) -> index
// CHECK-NEXT: %[[OUT:.*]] = pop.load %arg0 : !kgen.pointer<index>
// CHECK-NEXT: kgen.return %[[OUT]]

// CHECK-LABEL:  kgen.generator  @basic_arg_remove_2_REMOVED_ARG(%arg0: !kgen.pointer<index>) -> index
// CHECK-NEXT: %[[OUT:.*]] = kgen.call @basic_arg_remove_1_REMOVED_ARG(%arg0) : (!kgen.pointer<index>) -> index
// CHECK-NEXT: kgen.return %[[OUT]]


// -----

kgen.generator @recursive_test<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>, %arg2: !kgen.scalar<T>) -> index{
  %0 = kgen.call @recursive_test<:dtype T>(%arg0, %arg1, %arg2) : (index, !kgen.pointer<index>, !kgen.scalar<T>) -> (index)
  kgen.return %0 : index
}

kgen.generator export @recursive_test_entry<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>, %arg2: !kgen.scalar<T>) {
  %0 = kgen.call @recursive_test<:dtype T>(%arg0, %arg1, %arg2) : (index, !kgen.pointer<index>, !kgen.scalar<T>) -> (index)
  kgen.return
}

// CHECK-LABEL: kgen.generator  export @recursive_test_entry<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>, %arg2: !kgen.scalar<T>)
// CHECK-NEXT: kgen.call @recursive_test_REMOVED_ARG() : () -> index
// CHECK-NEXT: kgen.return

// CHECK-LABEL:  kgen.generator  @recursive_test_REMOVED_ARG() -> index
// CHECK-NEXT: %[[OUT:.*]] = kgen.call @recursive_test_REMOVED_ARG() : () -> index
// CHECK-NEXT: kgen.return %[[OUT]]

// -----

// In this test X / Y are switched so only "Z" is unused.
// Technically we could add more logic to remove X / Y too but this is a tradeoff
// of code complexity as we need to guard against lots of cases, e.g <X, X + Y>

kgen.generator @recursive_test_2<X: index, Y: index, Z: index>() -> index {
  %0 = kgen.call @recursive_test_2<:index Y, :index X, :index Z>() : () -> (index)
  kgen.return %0 : index
}

kgen.generator export @recursive_test_2_entry<X: index, Y: index, Z: index>() {
  %0 = kgen.call @recursive_test_2<:index X, :index Y, :index Z>() : () -> (index)
  kgen.return
}

// CHECK-LABEL: kgen.generator  export @recursive_test_2_entry<X, Y, Z>()
// CHECK-NEXT: kgen.call @recursive_test_2_REMOVED_ARG<X, Y>() : () -> index
// CHECK-NEXT: kgen.return

// CHECK-LABEL:  kgen.generator  @recursive_test_2_REMOVED_ARG<X, Y>() -> index
// CHECK-NEXT: %[[OUT:.*]] = kgen.call @recursive_test_2_REMOVED_ARG<Y, X>() : () -> index
// CHECK-NEXT: kgen.return %[[OUT]]


// -----


kgen.generator @test_argument_captured_in_attr<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>, %arg2: !kgen.scalar<T>) attributes {some_metadata = [#kgen.param.decl.ref<"T"> : !kgen.dtype]} {
  kgen.return
}

kgen.generator export @test_argument_captured_in_attr_entry<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>, %arg2: !kgen.scalar<T>) {
  kgen.call @test_argument_captured_in_attr<:dtype T>(%arg0, %arg1, %arg2) : (index, !kgen.pointer<index>, !kgen.scalar<T>) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator  export @test_argument_captured_in_attr_entry<T: dtype>
// CHECK-NEXT: kgen.call @test_argument_captured_in_attr_REMOVED_ARG<:dtype T>() : () -> (

// CHECK-LABEL: kgen.generator  @test_argument_captured_in_attr_REMOVED_ARG<T: dtype>() attributes {some_metadata = [#kgen.param.decl.ref<"T"> : !kgen.dtype]}

// -----


kgen.generator @used_in_param_expr_test<T: dtype>(%arg0: index, %arg1: index) -> index{
  kgen.return %arg0 : index
}

kgen.generator export @used_in_param_expr_test_entry<type: dtype>(%arg0: index, %arg1: index) -> index {
  kgen.param.declare *"OUT`" = <apply(:(index, index) -> index @used_in_param_expr_test<:dtype f32>, 5, 10)>
  %0 = kgen.call @used_in_param_expr_test<:dtype f32>(%arg0, %arg1) : (index, index) -> index
  kgen.return %0 : index
}

// The old one has to exist for the parameter.
// CHECK-LABEL:kgen.generator  @used_in_param_expr_test<T: dtype>(%arg0: index, %arg1: index) -> index {
// CHECK-NEXT: kgen.return

// Param should use old one, call uses the one with the parameters removed.
// CHECK-LABEL:kgen.generator  export @used_in_param_expr_test_entry<type: dtype>(%arg0: index, %arg1: index) -> index {
// CHECK-NEXT: kgen.param.declare *"OUT`" = <apply(:(index, index) -> index @used_in_param_expr_test<:dtype f32>, 5, 10)>
// CHECK-NEXT: kgen.call @used_in_param_expr_test_REMOVED_ARG(%arg0) : (index) -> index

// CHECK-LABEL: kgen.generator  @used_in_param_expr_test_REMOVED_ARG(%arg0: index) -> index


// -----

kgen.generator @with_cycle_3(%arg0: index) -> index{
  %0 = kgen.call @with_cycle_2(%arg0) : (index) -> (index)
  kgen.return %0 : index
}

kgen.generator @with_cycle_2(%arg0: index) -> index{
  %0 = kgen.call @with_cycle_3(%arg0) : (index) -> (index)
  kgen.return %0 : index
}

kgen.generator @with_cycle_1<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>, %arg2: !kgen.scalar<T>) -> index{
  %0 = kgen.call @with_cycle_2(%arg0) : (index) -> (index)
  kgen.return %0 : index
}

kgen.generator export @with_cycle<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>, %arg2: !kgen.scalar<T>) {
  %0 = kgen.call @with_cycle_1<:dtype T>(%arg0, %arg1, %arg2) : (index, !kgen.pointer<index>, !kgen.scalar<T>) -> (index)
  kgen.return
}

// CHECK-LABEL: kgen.generator  export @with_cycle<
// CHECK-NEXT:  kgen.call @with_cycle_1_REMOVED_ARG

// CHECK-LABEL: kgen.generator  @with_cycle_1_REMOVED_ARG(%arg0: index) -> index {
// CHECK-NEXT: kgen.call @with_cycle_2(%arg0) : (index) -> index

// -----

kgen.generator @hanoi<T: dtype>(%arg0: !kgen.scalar<si8>, %arg1: !kgen.scalar<si16>, %arg2: !kgen.scalar<si32>) -> (!kgen.scalar<si8>, !kgen.scalar<si32>) {
  %si8 = pop.cast %arg2 : !kgen.scalar<si32> to !kgen.scalar<si8>
  %si32 = pop.cast %arg0 : !kgen.scalar<si8> to !kgen.scalar<si32>
  %0:2 = kgen.call @hanoi<:dtype T>(%si8, %arg1, %si32) : (!kgen.scalar<si8>, !kgen.scalar<si16>, !kgen.scalar<si32>) -> (!kgen.scalar<si8>, !kgen.scalar<si32>)
  kgen.return %0, %si32 : !kgen.scalar<si8>, !kgen.scalar<si32>
}

kgen.generator export @hanoi_entry<T: dtype>(%arg0: !kgen.scalar<si8>, %arg1: !kgen.scalar<si16>, %arg2: !kgen.scalar<si32>){
  %0:2 = kgen.call @hanoi<:dtype T>(%arg0, %arg1, %arg2) : (!kgen.scalar<si8>, !kgen.scalar<si16>, !kgen.scalar<si32>) -> (!kgen.scalar<si8>, !kgen.scalar<si32>)
  kgen.return
}

// CHECK-LABEL: kgen.generator export @hanoi_entry<T: dtype>(%arg0: !kgen.scalar<si8>, %arg1: !kgen.scalar<si16>, %arg2: !kgen.scalar<si32>)
// CHECK-NEXT: kgen.call @hanoi_REMOVED_ARG(%arg0, %arg2)

// CHECK-LABEL: kgen.generator @hanoi_REMOVED_ARG(%arg0: !kgen.scalar<si8>, %arg1: !kgen.scalar<si32>)
// CHECK-NEXT:  %[[NEW_OP0:.*]] = pop.cast %arg1
// CHECK-NEXT:  %[[NEW_OP1:.*]] = pop.cast %arg0
// CHECK-NEXT:  %2:2 = kgen.call @hanoi_REMOVED_ARG(%[[NEW_OP0]], %[[NEW_OP1]])

// -----

// Test that a parameter referenced only in linkageName is not removed.

kgen.generator @linkage_name_keeps_param<Name: string>() attributes {
  linkageName = #kgen.linkage_name<#kgen.param.decl.ref<"Name"> : !kgen.string, false>
} {
  kgen.return
}

kgen.generator export @linkage_name_keeps_param_entry<Name: string>() {
  kgen.call @linkage_name_keeps_param<:string Name>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @linkage_name_keeps_param<Name: string>() attributes {linkageName = #kgen.linkage_name<#kgen.param.decl.ref<"Name"> : !kgen.string, false> : !kgen.string}

// CHECK-LABEL: kgen.generator export @linkage_name_keeps_param_entry<Name: string>()
// CHECK-NEXT: kgen.call @linkage_name_keeps_param<:string Name>() : () -> ()
