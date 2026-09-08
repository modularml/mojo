// RUN: kgen-opt %s | kgen-opt | FileCheck %s
// RUN: kgen-opt -emit-bytecode %s | kgen-opt | FileCheck %s

// CHECK-LABEL: @pointer
kgen.generator @pointer<ty: type, address_space>(
  // CHECK-SAME: !kgen.pointer<scalar<f32>>
  %arg0: !kgen.pointer<scalar<f32>>,
  // CHECK-SAME: !kgen.pointer<scalar<f32>, 5>
  %arg1: !kgen.pointer<scalar<f32>, 5>,
  // CHECK-SAME: !kgen.pointer<ty>
  %arg2: !kgen.pointer<ty>,
  // CHECK-SAME: !kgen.pointer<ty, 7>
  %arg3: !kgen.pointer<ty, 7>,
  // CHECK-SAME: !kgen.pointer<scalar<f32>, address_space>
  %arg4: !kgen.pointer<scalar<f32>, address_space>,
  // CHECK-SAME: !kgen.pointer<ty, address_space>
  %arg5: !kgen.pointer<ty, address_space>
) {
  kgen.return
}

// CHECK-LABEL: @array
kgen.generator @array<size, ty: type>(
  // CHECK-SAME: !pop.array<4, scalar<f32>>
  %arg0: !pop.array<4, scalar<f32>>,
  // CHECK-SAME: !pop.array<size, ty>
  %arg1: !pop.array<size, ty>
) {
  kgen.return
}

// CHECK-LABEL: @struct
kgen.generator @struct<size, dtype: dtype, ty: type>(
  // CHECK-SAME: !kgen.struct<(scalar<f32>, simd<4, ui64>)>
  %arg0: !kgen.struct<(scalar<f32>, simd<4, ui64>)>,
  // CHECK-SAME: !kgen.struct<(pointer<simd<4, si8>>, array<24, scalar<si64>>, struct<(scalar<f32>, scalar<f64>)>)>
  %arg1: !kgen.struct<(
    !kgen.pointer<simd<4, si8>>,
    !pop.array<24, scalar<si64>>,
    !kgen.struct<(
      !kgen.scalar<f32>,
      !kgen.scalar<f64>
    )>
  )>,
  // CHECK: !kgen.struct<(ty, array<size, scalar<dtype>>)>
  %arg2: !kgen.struct<(ty, array<size, scalar<dtype>>)>
) {
  kgen.return
}

// CHECK-LABEL: @variadic
kgen.generator @variadic<ty: type>(
  // CHECK-SAME: !kgen.param_list<scalar<f32>>
  %arg0: !kgen.param_list<!kgen.scalar<f32>>,
  // CHECK-SAME: !kgen.param_list<ty>
  %arg1: !kgen.param_list<ty>
) {
  kgen.return
}

// CHECK-LABEL: @union
// CHECK-SAME: !pop.union<>
// CHECK-SAME: !pop.union<i32>
// CHECK-SAME: !pop.union<i32, i64>
kgen.func @union(%arg0: !pop.union<>, %arg1: !pop.union<i32>, %arg2: !pop.union<i32, i64>) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @variadic_union
// CHECK-SAME: !pop.union<[Ts]>
// CHECK-SAME: !pop.union<T0, T1>
kgen.generator @variadic_union<Ts: param_list<type>, T0: type, T1: type>(
  %arg0: !pop.union<[Ts]>,
  %arg1: !pop.union<[[T0, T1]]>
) {
  kgen.return
}
