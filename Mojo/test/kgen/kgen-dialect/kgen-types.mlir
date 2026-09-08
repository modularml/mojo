// RUN: kgen-opt %s | kgen-opt --kgen-print-inline-type-values | FileCheck %s
// RUN: kgen-opt -emit-bytecode %s | kgen-opt --kgen-print-inline-type-values | FileCheck %s

// CHECK-LABEL: @genericSugar<scalar: type, T: type>
// CHECK-SAME: %arg0: !kgen.pointer<*"scalar">, %arg1: !kgen.pointer<T>
kgen.generator @genericSugar<scalar: type, T: type>(
  %arg0: !kgen.pointer<*"scalar">, %arg1: !kgen.pointer<T>
) {
  kgen.return
}

// CHECK-LABEL: @int_literal
// CHECK-SAME: %arg0: !pop.int_literal
kgen.func @int_literal(%arg0: !pop.int_literal) {
  kgen.return
}

// CHECK-LABEL: @float_literal
// CHECK-SAME: %arg0: !pop.float_literal
kgen.func @float_literal(%arg0: !pop.float_literal) {
  kgen.return
}

// CHECK-LABEL: @memory_only_struct
// CHECK-SAME: %arg0: !kgen.struct<()>,
// CHECK-SAME: %arg1: !kgen.struct<() memoryOnly>,
// CHECK-SAME: %arg2: !kgen.struct<(index, index)>,
// CHECK-SAME: %arg3: !kgen.struct<(index, index) memoryOnly>,
// CHECK-SAME: %arg4: !kgen.struct<((index, index) -> index)>,
// CHECK-SAME: %arg5: !kgen.struct<((index, index) -> index) memoryOnly>,
// CHECK-SAME: %arg6: !kgen.struct<(index, index) isParamPack>
kgen.func @memory_only_struct(
  %arg0: !kgen.struct<()>,
  %arg1: !kgen.struct<() memoryOnly>,
  %arg2: !kgen.struct<(index, index)>,
  %arg3: !kgen.struct<(index, index) memoryOnly>,
  %arg4: !kgen.struct<((index, index) -> index)>,
  %arg5: !kgen.struct<((index, index) -> index) memoryOnly>,
  %arg6: !kgen.struct<(index, index) isParamPack>
) {
  kgen.return
}

// CHECK-LABEL: @aligned_struct
// CHECK-SAME: %arg0: !kgen.struct<(index) align(64)>,
// CHECK-SAME: %arg1: !kgen.struct<(index) memoryOnly align(128)>
kgen.generator @aligned_struct<param_align>(
  %arg0: !kgen.struct<(index) align(64)>,
  %arg1: !kgen.struct<(index) memoryOnly align(128)>,
  %arg2: !kgen.struct<(index) align(mul(param_align, 2))>
) {
  kgen.return
}

// CHECK-LABEL: @type_printing
kgen.generator @type_printing() {
  // CHECK: type = <struct<()>>
  kgen.param.declare atype: type = <struct<()>>
  // CHECK: type = <struct<()>>
  kgen.param.declare btype: type = <struct<()>>
  // CHECK: type = <struct<()>>
  kgen.param.declare btype: type = <struct<()>>
  // CHECK: type = <struct_inst<"Foo"(data: struct<()>)>>
  kgen.param.declare btype: type = <struct_inst<"Foo"(data: struct<()>)>>
  // CHECK: type = <struct_inst<"Bar"[elemT, size]<:dtype f32, 16>(data: struct<()>) memoryOnly>>
  kgen.param.declare btype: type = <struct_inst<"Bar"[elemT, size]<:dtype f32, 16>(data: struct<()>) memoryOnly>>
  kgen.return
}

// CHECK-LABEL: kgen.generator @variadic_variant
// CHECK-SAME: !kgen.variant<[values]>
// CHECK-SAME-LITERAL: !kgen.variant<[[]]>
kgen.generator @variadic_variant<values: param_list<type>>(%arg0: !kgen.variant<[values]>, %arg1: !kgen.variant<[[]]>) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @parametric_struct
// CHECK-SAME: !kgen.struct<Ts>
// CHECK-SAME: !kgen.struct<Ts memoryOnly>
// CHECK-SAME-LITERAL: !kgen.struct<[index, f32]>
kgen.generator @parametric_struct<Ts: param_list<type>>(
  %arg0: !kgen.struct<Ts>,
  %arg1: !kgen.struct<Ts memoryOnly>,
  %arg2: !kgen.struct<[index, f32]>
) {
  kgen.return
}

// CHECK-LABEL: kgen.func @type_value
kgen.func @type_value() {
  // CHECK: type = <struct<()>>
  kgen.param.declare atype: type = <typevalue<struct<()>>>
  // CHECK: type = <struct<(struct<()>)>>
  kgen.param.declare atype: type = <struct<(typevalue<struct<()>>)>>
  kgen.return
}

// CHECK-LABEL: kgen.func @generator_types
kgen.func @generator_types() {
  // CHECK-NEXT: type = <<>None>
  kgen.param.declare atype: type = <!kgen.generator<<>None>>
  // CHECK-NEXT: type = <<AnyType>i1>
  kgen.param.declare atype: type = <!kgen.generator<<AnyType>i1>>
  // CHECK-NEXT: type = <<AnyType, index, index>i1>
  kgen.param.declare atype: type = <!kgen.generator<<AnyType, index, index>i1>>
  // CHECK-NEXT: type = <<AnyType, index, index>i1>
  kgen.param.declare atype: type = <<AnyType, index, index>i1>
  // CHECK-NEXT: type = <<AnyType, index>(!kgen.param<*(0,0)>, i8) -> !kgen.param<*(0,1)>>
  kgen.param.declare atype: type = <<AnyType, index>(!kgen.param<*(0,0)>, i8) -> !kgen.param<*(0,1)>>
  kgen.return
}

// CHECK-LABEL: kgen.func @func_types
kgen.func @func_types() {
  // CHECK-NEXT: atype: type = <() -> ()>
  kgen.param.declare atype: type = <!kgen.func<() -> ()>>
  // CHECK-NEXT: btype: type = <(index, i8) -> none>
  kgen.param.declare btype: type = <!kgen.func<(index, i8) -> none>>
  kgen.return
}

// COM: Non-null pointer types

// CHECK-LABEL: @nonnull_pointer
// CHECK-SAME: !kgen.pointer<scalar<f32>, nonnull>
kgen.func @nonnull_pointer(%arg0: !kgen.pointer<scalar<f32>, 0, nonnull>) {
  kgen.return
}

// CHECK-LABEL: @nonnull_pointer_addrspace
// CHECK-SAME: !kgen.pointer<scalar<f32>, 1, nonnull>
kgen.func @nonnull_pointer_addrspace(%arg0: !kgen.pointer<scalar<f32>, 1, nonnull>) {
  kgen.return
}
