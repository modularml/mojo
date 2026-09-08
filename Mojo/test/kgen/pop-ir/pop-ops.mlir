// RUN: kgen-opt -verify-parameters %s -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: kgen.generator @pointer_type<dt: dtype>(
kgen.generator @pointer_type<dt: dtype>
  // CHECK-SAME: %{{.*}}: !kgen.pointer<scalar<dt>>,
  (%arg0 : !kgen.pointer<scalar<dt>>,
  // CHECK-SAME: %{{.*}}: !kgen.pointer<scalar<ui8>>,
  %arg1: !kgen.pointer<scalar<ui8>>,
  // CHECK-SAME: %{{.*}}: !kgen.pointer<scalar<invalid>>) {
  %arg2: !kgen.pointer<scalar<invalid>>) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @simd_type<dt: dtype, size>(
kgen.generator @simd_type<dt: dtype, size>
  // CHECK-SAME: %{{.*}}: !kgen.simd<4, dt>,
  (%arg0 : !kgen.simd<4,dt>,
  // CHECK-SAME: %{{.*}}: !kgen.simd<to_builtin(:scalar<index> mul(from_builtin(size), from_builtin(size))), ui8>) {
   %arg1: !kgen.simd<mul(size,size), ui8>) {

  kgen.return
}

// CHECK-LABEL: kgen.func @pop_neg
// CHECK-SAME: %[[ARG0:.*]]: !kgen.scalar<f32>
// CHECK-SAME: %[[ARG1:.*]]: !kgen.scalar<index>
kgen.func @pop_neg(%arg0: !kgen.scalar<f32>,
                   %arg1: !kgen.scalar<index>) -> (!kgen.scalar<f32>,
                                                  !kgen.scalar<index>) {
  // CHECK: pop.neg %[[ARG0]] : !kgen.scalar<f32>
  %0 = pop.neg %arg0 : !kgen.scalar<f32>
  // CHECK: pop.neg %[[ARG1]] : !kgen.scalar<index>
  %1 = pop.neg %arg1 : !kgen.scalar<index>
  kgen.return %0, %1 : !kgen.scalar<f32>, !kgen.scalar<index>
}

// CHECK-LABEL: kgen.func @pop_add
kgen.func @pop_add(%arg0: !kgen.scalar<f32>, %arg1: !kgen.scalar<f32>) -> !kgen.scalar<f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.add %arg0, %arg1 : !kgen.scalar<f32>
  %c = pop.add %arg0, %arg1 : !kgen.scalar<f32>
  kgen.return %c : !kgen.scalar<f32>
}

// CHECK-LABEL: kgen.generator @pop_add2<DT: dtype>
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.scalar<DT>, %[[ARG1:.*]]: !kgen.scalar<DT>) -> !kgen.scalar<DT> {
kgen.generator @pop_add2<DT: dtype>(%a: !kgen.scalar<DT>, %b: !kgen.scalar<DT>) -> !kgen.scalar<DT> {
  // CHECK-NEXT: %[[V0:.*]] = pop.add %[[ARG0]], %[[ARG1]] : !kgen.scalar<DT>
  %c = pop.add %a, %b : !kgen.scalar<DT>
  kgen.return %c : !kgen.scalar<DT>
}

// CHECK-LABEL: kgen.func @pop_add_simd
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.simd<4, f32>, %[[ARG1:.*]]: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
kgen.func @pop_add_simd(%arg0 : !kgen.simd<4, f32>, %arg1 : !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.add %[[ARG0]], %[[ARG1]] : !kgen.simd<4, f32>
  %0 = pop.add %arg0, %arg1 : !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: kgen.func @pop_add_simd_index
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.simd<4, index>, %[[ARG1:.*]]: !kgen.simd<4, index>) -> !kgen.simd<4, index> {
kgen.func @pop_add_simd_index(%arg0 : !kgen.simd<4, index>, %arg1 : !kgen.simd<4, index>) -> !kgen.simd<4, index> {
  // CHECK-NEXT: %[[V0:.*]] = pop.add %[[ARG0]], %[[ARG1]] : !kgen.simd<4, index>
  %0 = pop.add %arg0, %arg1 : !kgen.simd<4, index>
  kgen.return %0 : !kgen.simd<4, index>
}

// CHECK-LABEL: kgen.func @pop_sub
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.scalar<f32>, %[[ARG1:.*]]: !kgen.scalar<f32>) -> !kgen.scalar<f32> {
kgen.func @pop_sub(%arg0 : !kgen.scalar<f32>, %arg1 : !kgen.scalar<f32>) -> !kgen.scalar<f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.sub %[[ARG0]], %[[ARG1]] : !kgen.scalar<f32>
  %0 = pop.sub %arg0, %arg1 : !kgen.scalar<f32>
  kgen.return %0 : !kgen.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_sub_simd
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.simd<4, f32>, %[[ARG1:.*]]: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
kgen.func @pop_sub_simd(%arg0 : !kgen.simd<4, f32>, %arg1 : !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.sub %[[ARG0]], %[[ARG1]] : !kgen.simd<4, f32>
  %0 = pop.sub %arg0, %arg1 : !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: kgen.func @pop_sub_simd_index
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.simd<4, index>, %[[ARG1:.*]]: !kgen.simd<4, index>) -> !kgen.simd<4, index> {
kgen.func @pop_sub_simd_index(%arg0 : !kgen.simd<4, index>, %arg1 : !kgen.simd<4, index>) -> !kgen.simd<4, index> {
  // CHECK-NEXT: %[[V0:.*]] = pop.sub %[[ARG0]], %[[ARG1]] : !kgen.simd<4, index>
  %0 = pop.sub %arg0, %arg1 : !kgen.simd<4, index>
  kgen.return %0 : !kgen.simd<4, index>
}

// CHECK-LABEL: kgen.func @pop_max
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.scalar<f32>, %[[ARG1:.*]]: !kgen.scalar<f32>) -> !kgen.scalar<f32> {
kgen.func @pop_max(%arg0 : !kgen.scalar<f32>, %arg1 : !kgen.scalar<f32>) -> !kgen.scalar<f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.max %[[ARG0]], %[[ARG1]] : !kgen.scalar<f32>
  %0 = pop.max %arg0, %arg1 : !kgen.scalar<f32>
  kgen.return %0 : !kgen.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_max_simd
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.simd<4, f32>, %[[ARG1:.*]]: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
kgen.func @pop_max_simd(%arg0 : !kgen.simd<4, f32>, %arg1 : !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.max %[[ARG0]], %[[ARG1]] : !kgen.simd<4, f32>
  %0 = pop.max %arg0, %arg1 : !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: kgen.func @pop_min
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.scalar<f32>, %[[ARG1:.*]]: !kgen.scalar<f32>) -> !kgen.scalar<f32> {
kgen.func @pop_min(%arg0 : !kgen.scalar<f32>, %arg1 : !kgen.scalar<f32>) -> !kgen.scalar<f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.min %[[ARG0]], %[[ARG1]] : !kgen.scalar<f32>
  %0 = pop.min %arg0, %arg1 : !kgen.scalar<f32>
  kgen.return %0 : !kgen.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_min_simd
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.simd<4, f32>, %[[ARG1:.*]]: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
kgen.func @pop_min_simd(%arg0 : !kgen.simd<4, f32>, %arg1 : !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.min %[[ARG0]], %[[ARG1]] : !kgen.simd<4, f32>
  %0 = pop.min %arg0, %arg1 : !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: kgen.func @pop_mul
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.scalar<f32>, %[[ARG1:.*]]: !kgen.scalar<f32>) -> !kgen.scalar<f32> {
kgen.func @pop_mul(%arg0 : !kgen.scalar<f32>, %arg1 : !kgen.scalar<f32>) -> !kgen.scalar<f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.mul %[[ARG0]], %[[ARG1]] : !kgen.scalar<f32>
  %0 = pop.mul %arg0, %arg1 : !kgen.scalar<f32>
  kgen.return %0 : !kgen.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_mul_simd
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.simd<4, f32>, %[[ARG1:.*]]: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
kgen.func @pop_mul_simd(%arg0 : !kgen.simd<4, f32>, %arg1 : !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.mul %[[ARG0]], %[[ARG1]] : !kgen.simd<4, f32>
  %0 = pop.mul %arg0, %arg1 : !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: kgen.func @pop_div
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.scalar<f32>, %[[ARG1:.*]]: !kgen.scalar<f32>) -> !kgen.scalar<f32> {
kgen.func @pop_div(%arg0 : !kgen.scalar<f32>, %arg1 : !kgen.scalar<f32>) -> !kgen.scalar<f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.div %[[ARG0]], %[[ARG1]] : !kgen.scalar<f32>
  %0 = pop.div %arg0, %arg1 : !kgen.scalar<f32>
  kgen.return %0 : !kgen.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_div_simd
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.simd<4, f32>, %[[ARG1:.*]]: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
kgen.func @pop_div_simd(%arg0 : !kgen.simd<4, f32>, %arg1 : !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.div %[[ARG0]], %[[ARG1]] : !kgen.simd<4, f32>
  %0 = pop.div %arg0, %arg1 : !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: kgen.func @pop_rem
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.scalar<f32>, %[[ARG1:.*]]: !kgen.scalar<f32>) -> !kgen.scalar<f32> {
kgen.func @pop_rem(%arg0 : !kgen.scalar<f32>, %arg1 : !kgen.scalar<f32>) -> !kgen.scalar<f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.rem %[[ARG0]], %[[ARG1]] : !kgen.scalar<f32>
  %0 = pop.rem %arg0, %arg1 : !kgen.scalar<f32>
  kgen.return %0 : !kgen.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_rem_simd
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.simd<4, f32>, %[[ARG1:.*]]: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
kgen.func @pop_rem_simd(%arg0 : !kgen.simd<4, f32>, %arg1 : !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.rem %[[ARG0]], %[[ARG1]] : !kgen.simd<4, f32>
  %0 = pop.rem %arg0, %arg1 : !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: @pop_shifts
kgen.func @pop_shifts(%arg0: !kgen.scalar<si32>, %arg1: !kgen.scalar<si32>,
                      %arg2: !kgen.scalar<index>, %arg3: !kgen.scalar<index>) {
  // CHECK: = pop.shl %{{.*}}, %{{.*}} : !kgen.scalar<si32>
  %0 = pop.shl %arg0, %arg1 : !kgen.scalar<si32>
  // CHECK: = pop.shr %{{.*}}, %{{.*}} : !kgen.scalar<si32>
  %1 = pop.shr %arg0, %arg1 : !kgen.scalar<si32>
  // CHECK: = pop.shr %{{.*}}, %{{.*}} : !kgen.scalar<index>
  %2 = pop.shr %arg2, %arg3 : !kgen.scalar<index>
  // CHECK: = pop.shl %{{.*}}, %{{.*}} : !kgen.scalar<index>
  %3 = pop.shl %arg2, %arg3 : !kgen.scalar<index>
  kgen.return
}

// CHECK-LABEL: @pop_shifts_simd
kgen.func @pop_shifts_simd(%arg0: !kgen.simd<4, si32>, %arg1: !kgen.simd<4, si32>,
                      %arg2: !kgen.simd<4, index>, %arg3: !kgen.simd<4, index>) {
  // CHECK: pop.shl %{{.*}}, %{{.*}} : !kgen.simd<4, si32>
  %0 = pop.shl %arg0, %arg1 : !kgen.simd<4, si32>
  // CHECK: pop.shr %{{.*}}, %{{.*}} : !kgen.simd<4, si32>
  %1 = pop.shr %arg0, %arg1 : !kgen.simd<4, si32>
  // CHECK: = pop.shr %{{.*}}, %{{.*}} : !kgen.simd<4, index>
  %2 = pop.shr %arg2, %arg3 : !kgen.simd<4, index>
  // CHECK: = pop.shl %{{.*}}, %{{.*}} : !kgen.simd<4, index>
  %3 = pop.shl %arg2, %arg3 : !kgen.simd<4, index>
  kgen.return
}

// CHECK-LABEL: kgen.func @pop_fma
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.scalar<f32>, %[[ARG1:.*]]: !kgen.scalar<f32>, %[[ARG2:.*]]: !kgen.scalar<f32>) -> !kgen.scalar<f32> {
kgen.func @pop_fma(%arg0: !kgen.scalar<f32>, %arg1: !kgen.scalar<f32>, %arg2: !kgen.scalar<f32>) -> !kgen.scalar<f32> {
  // CHECK: %[[V0:.*]] = pop.fma %[[ARG0]], %[[ARG1]], %[[ARG2]] : !kgen.scalar<f32>
  %0 = pop.fma %arg0, %arg1, %arg2: !kgen.scalar<f32>
  kgen.return %0 : !kgen.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_fma_simd
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.simd<4, f32>, %[[ARG1:.*]]: !kgen.simd<4, f32>, %[[ARG2:.*]]: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
kgen.func @pop_fma_simd(%arg0 : !kgen.simd<4, f32>, %arg1 : !kgen.simd<4, f32>, %arg2 : !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.fma %[[ARG0]], %[[ARG1]], %[[ARG2]] : !kgen.simd<4, f32>
  %0 = pop.fma %arg0, %arg1, %arg2 : !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: @pop_cmp
kgen.func @pop_cmp(%arg0: !kgen.scalar<f32>, %arg1: !kgen.scalar<f32>) -> !kgen.scalar<bool> {
  // CHECK: pop.cmp ge(%{{.*}}, %{{.*}}) :
  %0 = pop.cmp ge(%arg0, %arg1) : !kgen.scalar<f32>
  kgen.return %0 : !kgen.scalar<bool>
}

kgen.func @pop_cmp_simd(
    %arg0: !kgen.simd<4, si32>, %arg1: !kgen.simd<4, si32>,
    %arg2: !kgen.simd<2, f64>, %arg3: !kgen.simd<2, f64>
  ) -> (!kgen.simd<4, bool>, !kgen.simd<2, bool>) {
  // CHECK: pop.cmp ne(%{{.*}}, %{{.*}}) :
  %0 = pop.cmp ne(%arg0, %arg1) : !kgen.simd<4, si32>
  // CHECK: pop.cmp lt(%{{.*}}, %{{.*}}) :
  %1 = pop.cmp lt(%arg2, %arg3) : !kgen.simd<2, f64>
  kgen.return %0, %1 : !kgen.simd<4, bool>, !kgen.simd<2, bool>
}

// CHECK-LABEL: @pop_and_bool
kgen.func @pop_and_bool(%arg0: !kgen.scalar<bool>, %arg1: !kgen.scalar<bool>,
                        %arg2: !kgen.simd<4, bool>, %arg3: !kgen.simd<4, bool>) {
  // CHECK: pop.simd.and %{{.*}}, %{{.*}} :
  %0 = pop.simd.and %arg0, %arg1 : !kgen.scalar<bool>
  // CHECK: pop.simd.and %{{.*}}, %{{.*}} :
  %1 = pop.simd.and %arg2, %arg3 : !kgen.simd<4, bool>
  kgen.return
}

// CHECK-LABEL: @pop_and
kgen.func @pop_and(%arg0: !kgen.scalar<si32>, %arg1: !kgen.scalar<si32>,
                   %arg2: !kgen.simd<4, si32>, %arg3: !kgen.simd<4, si32>) {
  // CHECK: pop.simd.and %{{.*}}, %{{.*}} :
  %0 = pop.simd.and %arg0, %arg1 : !kgen.scalar<si32>
  // CHECK: pop.simd.and %{{.*}}, %{{.*}} :
  %1 = pop.simd.and %arg2, %arg3 : !kgen.simd<4, si32>
  kgen.return
}

// CHECK-LABEL: @pop_and_index
kgen.func @pop_and_index(%arg0: !kgen.scalar<index>, %arg1: !kgen.scalar<index>,
                       %arg2: !kgen.simd<4, index>, %arg3: !kgen.simd<4, index>) {
  // CHECK: pop.simd.and %{{.*}}, %{{.*}} :
  %0 = pop.simd.and %arg0, %arg1 : !kgen.scalar<index>
  // CHECK: pop.simd.and %{{.*}}, %{{.*}} :
  %1 = pop.simd.and %arg2, %arg3 : !kgen.simd<4, index>
  kgen.return
}

kgen.generator @pop_and_parametric<size, DT: dtype>(
                   %arg0: !kgen.scalar<DT>, %arg1: !kgen.scalar<DT>,
                   %arg2: !kgen.simd<size, DT>, %arg3: !kgen.simd<size, DT>) {
  // CHECK: pop.simd.and
  %0 = pop.simd.and %arg0, %arg1 : !kgen.scalar<DT>
  // CHECK: pop.simd.and
  %1 = pop.simd.and %arg2, %arg3 : !kgen.simd<size, DT>
  kgen.return
}

// CHECK-LABEL: @pop_or_bool
kgen.func @pop_or_bool(%arg0: !kgen.scalar<bool>, %arg1: !kgen.scalar<bool>,
                       %arg2: !kgen.simd<4, bool>, %arg3: !kgen.simd<4, bool>) {
  // CHECK: pop.simd.or %{{.*}}, %{{.*}} :
  %0 = pop.simd.or %arg0, %arg1 : !kgen.scalar<bool>
  // CHECK: pop.simd.or %{{.*}}, %{{.*}} :
  %1 = pop.simd.or %arg2, %arg3 : !kgen.simd<4, bool>
  kgen.return
}

// CHECK-LABEL: @pop_or
kgen.func @pop_or(%arg0: !kgen.scalar<si32>, %arg1: !kgen.scalar<si32>,
                   %arg2: !kgen.simd<4, si32>, %arg3: !kgen.simd<4, si32>) {
  // CHECK: pop.simd.or %{{.*}}, %{{.*}} :
  %0 = pop.simd.or %arg0, %arg1 : !kgen.scalar<si32>
  // CHECK: pop.simd.or %{{.*}}, %{{.*}} :
  %1 = pop.simd.or %arg2, %arg3 : !kgen.simd<4, si32>
  kgen.return
}

// CHECK-LABEL: @pop_or_index
kgen.func @pop_or_index(%arg0: !kgen.scalar<index>, %arg1: !kgen.scalar<index>,
                       %arg2: !kgen.simd<4, index>, %arg3: !kgen.simd<4, index>) {
  // CHECK: pop.simd.or %{{.*}}, %{{.*}} :
  %0 = pop.simd.or %arg0, %arg1 : !kgen.scalar<index>
  // CHECK: pop.simd.or %{{.*}}, %{{.*}} :
  %1 = pop.simd.or %arg2, %arg3 : !kgen.simd<4, index>
  kgen.return
}

kgen.generator @pop_or_parametric<size, DT: dtype>(
                   %arg0: !kgen.scalar<DT>, %arg1: !kgen.scalar<DT>,
                   %arg2: !kgen.simd<size, DT>, %arg3: !kgen.simd<size, DT>) {
  // CHECK: pop.simd.or
  %0 = pop.simd.or %arg0, %arg1 : !kgen.scalar<DT>
  // CHECK: pop.simd.or
  %1 = pop.simd.or %arg2, %arg3 : !kgen.simd<size, DT>
  kgen.return
}

// CHECK-LABEL: @pop_xor_bool
kgen.func @pop_xor_bool(%arg0: !kgen.scalar<bool>, %arg1: !kgen.scalar<bool>,
                   %arg2: !kgen.simd<4, bool>, %arg3: !kgen.simd<4, bool>) {
  // CHECK: pop.simd.xor %{{.*}}, %{{.*}} :
  %0 = pop.simd.xor %arg0, %arg1 : !kgen.scalar<bool>
  // CHECK: pop.simd.xor %{{.*}}, %{{.*}} :
  %1 = pop.simd.xor %arg2, %arg3 : !kgen.simd<4, bool>
  kgen.return
}

// CHECK-LABEL: @pop_xor
kgen.func @pop_xor(%arg0: !kgen.scalar<si32>, %arg1: !kgen.scalar<si32>,
                   %arg2: !kgen.simd<4, si32>, %arg3: !kgen.simd<4, si32>) {
  // CHECK: pop.simd.xor %{{.*}}, %{{.*}} :
  %0 = pop.simd.xor %arg0, %arg1 : !kgen.scalar<si32>
  // CHECK: pop.simd.xor %{{.*}}, %{{.*}} :
  %1 = pop.simd.xor %arg2, %arg3 : !kgen.simd<4, si32>
  kgen.return
}

// CHECK-LABEL: @pop_xor_index
kgen.func @pop_xor_index(%arg0: !kgen.scalar<index>, %arg1: !kgen.scalar<index>,
                       %arg2: !kgen.simd<4, index>, %arg3: !kgen.simd<4, index>) {
  // CHECK: pop.simd.xor %{{.*}}, %{{.*}} :
  %0 = pop.simd.xor %arg0, %arg1 : !kgen.scalar<index>
  // CHECK: pop.simd.xor %{{.*}}, %{{.*}} :
  %1 = pop.simd.xor %arg2, %arg3 : !kgen.simd<4, index>
  kgen.return
}

kgen.generator @pop_xor_parametric<size, DT: dtype>(
                   %arg0: !kgen.scalar<DT>, %arg1: !kgen.scalar<DT>,
                   %arg2: !kgen.simd<size, DT>, %arg3: !kgen.simd<size, DT>) {
  // CHECK: pop.simd.xor
  %0 = pop.simd.xor %arg0, %arg1 : !kgen.scalar<DT>
  // CHECK: pop.simd.xor
  %1 = pop.simd.xor %arg2, %arg3 : !kgen.simd<size, DT>
  kgen.return
}

// CHECK-LABEL: @pop_select
kgen.func @pop_select(%arg0: !kgen.scalar<bool>, %arg1: !kgen.struct<(f32)>, %arg2: !kgen.struct<(f32)>) -> !kgen.struct<(f32)> {
  // CHECK: pop.select %arg0, %arg1, %arg2 : !kgen.struct<(f32)>
  %0 = pop.select %arg0, %arg1, %arg2 : !kgen.struct<(f32)>
  kgen.return %0 : !kgen.struct<(f32)>
}

// CHECK-LABEL: @pop_select_simd
kgen.func @pop_select_simd(
    %arg0: !kgen.simd<4, bool>,
    %arg1: !kgen.simd<4, si32>,
    %arg2: !kgen.simd<4, si32>
  ) -> !kgen.simd<4, si32> {
  // CHECK: pop.simd.select %{{.*}}, %{{.*}}, %{{.*}} :
  %0 = pop.simd.select %arg0, %arg1, %arg2 : !kgen.simd<4, si32>
  kgen.return %0 : !kgen.simd<4, si32>
}

// CHECK-LABEL: @scalar_bitcast
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.generator @scalar_bitcast(%arg0: !kgen.scalar<f32>, %arg1: !kgen.scalar<f64>) -> !kgen.scalar<f32> {
  // CHECK: %[[V0:.*]] = pop.bitcast %[[ARG0]] : !kgen.scalar<f32> to !kgen.scalar<si32>
  %0 = pop.bitcast %arg0 : !kgen.scalar<f32> to !kgen.scalar<si32>
  // CHECK: %[[V1:.*]] = pop.bitcast %[[ARG1]] : !kgen.scalar<f64> to !kgen.scalar<f64>
  %1 = pop.bitcast %arg1 : !kgen.scalar<f64> to !kgen.scalar<f64>
  // CHECK: %[[V2:.*]] = pop.bitcast %[[V0]] : !kgen.scalar<si32> to !kgen.scalar<ui32>
  %2 = pop.bitcast %0 : !kgen.scalar<si32> to !kgen.scalar<ui32>
  // CHECK: %[[V3:.*]] = pop.bitcast %[[V2]] : !kgen.scalar<ui32> to !kgen.scalar<f32>
  %3 = pop.bitcast %2 : !kgen.scalar<ui32> to !kgen.scalar<f32>
  // CHECK: return %[[V3]]
  kgen.return %3 : !kgen.scalar<f32>
}

// CHECK-LABEL: @simd_bitcast
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.generator @simd_bitcast(%arg0: !kgen.simd<4, f32>, %arg1: !kgen.simd<4, f64>) -> !kgen.simd<4, f32> {
  // CHECK: %[[V0:.*]] = pop.bitcast %[[ARG0]] : !kgen.simd<4, f32> to !kgen.simd<4, si32>
  %0 = pop.bitcast %arg0 : !kgen.simd<4, f32> to !kgen.simd<4, si32>
  // CHECK: %[[V1:.*]] = pop.bitcast %[[ARG1]] : !kgen.simd<4, f64> to !kgen.simd<4, f64>
  %1 = pop.bitcast %arg1 : !kgen.simd<4, f64> to !kgen.simd<4, f64>
  // CHECK: %[[V2:.*]] = pop.bitcast %[[V0]] : !kgen.simd<4, si32> to !kgen.simd<4, ui32>
  %2 = pop.bitcast %0 : !kgen.simd<4, si32> to !kgen.simd<4, ui32>
  // CHECK: %[[V3:.*]] = pop.bitcast %[[V2]] : !kgen.simd<4, ui32> to !kgen.simd<4, f32>
  %3 = pop.bitcast %2 : !kgen.simd<4, ui32> to !kgen.simd<4, f32>
  // CHECK: %[[V4:.*]] = pop.bitcast %[[V2]] : !kgen.simd<4, ui32> to !kgen.simd<2, f64>
  %4 = pop.bitcast %2 : !kgen.simd<4, ui32> to !kgen.simd<2, f64>
  // CHECK: %[[V5:.*]] = pop.bitcast %[[V0]] : !kgen.simd<4, si32> to !kgen.simd<128, bool>
  %5 = pop.bitcast %0 : !kgen.simd<4, si32> to !kgen.simd<128, bool>
  // CHECK: return %[[V3]]
  kgen.return %3 : !kgen.simd<4, f32>
}

// CHECK-LABEL: @bitcast_simd_from_bool
kgen.generator @bitcast_simd_from_bool(%a: !kgen.simd<32, bool>) {
  // CHECK: pop.bitcast %arg0 : !kgen.simd<32, bool> to !kgen.scalar<f32>
  %0 = pop.bitcast %a : !kgen.simd<32, bool> to !kgen.simd<1, f32>
  kgen.return
}

// CHECK-LABEL: @bitcast_simd_to_bool
kgen.generator @bitcast_simd_to_bool(%a: !kgen.simd<1, f32>) {
  // CHECK: pop.bitcast %arg0 : !kgen.scalar<f32> to !kgen.simd<32, bool>
  %0 = pop.bitcast %a : !kgen.simd<1, f32> to !kgen.simd<32, bool>
  kgen.return
}

// CHECK-LABEL: @bitcast_parametric
kgen.generator @bitcast_parametric<size1, size2, type1: dtype, type2: dtype>(
  %arg0: !kgen.simd<size1, type1>, %arg1: !kgen.simd<size2, f32>,
  %arg2: !kgen.scalar<type2>
) {
  // CHECK: pop.bitcast %{{.*}} : !kgen.simd<size1, type1> to !kgen.simd<size2, f32>
  %0 = pop.bitcast %arg0 : !kgen.simd<size1, type1> to !kgen.simd<size2, f32>
  // CHECK: pop.bitcast %{{.*}} : !kgen.simd<size2, f32> to !kgen.simd<4, f64>
  %1 = pop.bitcast %arg1 : !kgen.simd<size2, f32> to !kgen.simd<4, f64>
  // CHECK: pop.bitcast %{{.*}} : !kgen.scalar<type2> to !kgen.scalar<f32>
  %2 = pop.bitcast %arg2 : !kgen.scalar<type2> to !kgen.scalar<f32>
  kgen.return
}

// CHECK-LABEL: @pointer_bitcast
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.generator @pointer_bitcast(%arg0: !kgen.pointer<scalar<f32>>, %arg1: !kgen.pointer<simd<4, f64>>) ->
   (!kgen.pointer<simd<4, si32>>, !kgen.pointer<scalar<f64>>) {
  // CHECK: %[[V0:.*]] = pop.pointer.bitcast %[[ARG0]] : !kgen.pointer<scalar<f32>> to !kgen.pointer<simd<4, si32>>
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<scalar<f32>> to !kgen.pointer<simd<4, si32>>
  // CHECK: %[[V1:.*]] = pop.pointer.bitcast %[[ARG1]] : !kgen.pointer<simd<4, f64>> to !kgen.pointer<scalar<f64>>
  %1 = pop.pointer.bitcast %arg1 : !kgen.pointer<simd<4, f64>> to !kgen.pointer<scalar<f64>>
  // CHECK: %{{.*}} = pop.pointer.bitcast %[[ARG0]] : !kgen.pointer<scalar<f32>> to !kgen.pointer<scalar<invalid>>
  %2 = pop.pointer.bitcast %arg0 : !kgen.pointer<scalar<f32>> to !kgen.pointer<scalar<invalid>>
  // CHECK: return %[[V0]], %[[V1]]
  kgen.return %0, %1 : !kgen.pointer<simd<4, si32>>, !kgen.pointer<scalar<f64>>
}

// CHECK-LABEL: @pointer_bitcast_funcptr
kgen.generator @pointer_bitcast_funcptr<T:type>(%arg0: !kgen.generator<() -> ()>) {
  // CHECK: pop.pointer.bitcast %arg0 : !kgen.generator<() -> ()> to !kgen.generator<(i32) -> i32>
  %0 = pop.pointer.bitcast %arg0 : !kgen.generator<() -> ()> to !kgen.generator<(i32) -> i32>
  // CHECK: pop.pointer.bitcast %arg0 : !kgen.generator<() -> ()> to !kgen.param<T>
  %1 = pop.pointer.bitcast %arg0 : !kgen.generator<() -> ()> to !kgen.param<T>
  kgen.return
}

// CHECK-LABEL: @pop_bitcast_paramref
// CHECK-SAME: %[[ARG:[a-z0-9]*]]:
kgen.generator @pop_bitcast_paramref<size1, dt1: dtype, size2, dt2: dtype>(%arg: !kgen.simd<size1,dt1>) {
  // CHECK: pop.bitcast %[[ARG]] : !kgen.simd<size1, dt1> to !kgen.simd<size2, dt2>
  %0 = pop.bitcast %arg : !kgen.simd<size1,dt1> to !kgen.simd<size2,dt2>
  kgen.return
}

// CHECK-LABEL: @scalar_cast
// CHECK-SAME: %[[A:.*]]:
kgen.generator @scalar_cast<DT: dtype>(%a: !kgen.scalar<f32>) -> !kgen.scalar<si32> {
  // CHECK: %[[V0:.*]] = pop.cast %[[A]] : !kgen.scalar<f32> to !kgen.scalar<DT>
  %0 = pop.cast %a : !kgen.scalar<f32> to !kgen.scalar<DT>
  // CHECK: %[[V1:.*]] = pop.cast %[[V0]] : !kgen.scalar<DT> to !kgen.scalar<f64>
  %1 = pop.cast %0 : !kgen.scalar<DT> to !kgen.scalar<f64>
  // CHECK: %[[V2:.*]] = pop.cast %[[V1]] : !kgen.scalar<f64> to !kgen.scalar<si32>
  %2 = pop.cast %1 : !kgen.scalar<f64> to !kgen.scalar<si32>
  // CHECK: return %[[V2]]
  kgen.return %2 : !kgen.scalar<si32>
}

// CHECK-LABEL: @simd_cast
// CHECK-SAME: %[[A:.*]]:
kgen.generator @simd_cast<size, type: dtype>(%a: !kgen.simd<size, f32>) -> !kgen.simd<size, si32> {
  // CHECK: %[[V0:.*]] = pop.cast %[[A]] : !kgen.simd<size, f32> to !kgen.simd<size, type>
  %0 = pop.cast %a : !kgen.simd<size, f32> to !kgen.simd<size, type>
  // CHECK: %[[V1:.*]] = pop.cast %[[V0]] : !kgen.simd<size, type> to !kgen.simd<size, si32>
  %1 = pop.cast %0 : !kgen.simd<size, type> to !kgen.simd<size, si32>
  // CHECK: %[[V2:.*]] = pop.cast %[[V1]] : !kgen.simd<size, si32> to !kgen.simd<size, f64>
  %2 = pop.cast %1 : !kgen.simd<size, si32> to !kgen.simd<size, f64>
  // CHECK: %[[V3:.*]] = pop.cast %[[V2]] : !kgen.simd<size, f64> to !kgen.simd<size, si32>
  %3 = pop.cast %2 : !kgen.simd<size, f64> to !kgen.simd<size, si32>
  // CHECK: return %[[V3]]
  kgen.return %3 : !kgen.simd<size, si32>
}

// CHECK-LABEL: @pop_simd_extractelement
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
// CHECK-SAME: %[[C:[a-z0-9]+]]:
kgen.generator @pop_simd_extractelement<size, type: dtype>(
    %a: !kgen.simd<size, type>,
    %b: !kgen.simd<size, f32>,
    %c: !kgen.simd<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  index.constant
  %idx = index.constant 2
  // CHECK: %[[U:.*]] = pop.simd.extractelement %[[A]][%[[IDX]]] : !kgen.simd<size, type>
  %u = pop.simd.extractelement %a[%idx] : !kgen.simd<size, type>
  // CHECK: %[[V:.*]] = pop.simd.extractelement %[[B]][%[[IDX]]] : !kgen.simd<size, f32>
  %v = pop.simd.extractelement %b[%idx] : !kgen.simd<size, f32>
  // CHECK: %[[w:.*]] = pop.simd.extractelement %[[C]][%[[IDX]]] : !kgen.simd<4, si32>
  %w = pop.simd.extractelement %c[%idx] : !kgen.simd<4, si32>
  kgen.return
}

// CHECK-LABEL: @pop_simd_insertelement
// CHECK-SAME: %[[V0:[a-z0-9]+]]:
// CHECK-SAME: %[[V1:[a-z0-9]+]]:
// CHECK-SAME: %[[V2:[a-z0-9]+]]:
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
// CHECK-SAME: %[[C:[a-z0-9]+]]:
kgen.generator @pop_simd_insertelement<size, DT: dtype>(
    %v0: !kgen.scalar<DT>,
    %v1: !kgen.scalar<f32>,
    %v2: !kgen.scalar<si32>,
    %a: !kgen.simd<size, DT>,
    %b: !kgen.simd<size, f32>,
    %c: !kgen.simd<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  index.constant
  %idx = index.constant 2
  // CHECK: %[[U:.*]] = pop.simd.insertelement %[[V0]], %[[A]][%[[IDX]]] : !kgen.simd<size, DT>
  %u = pop.simd.insertelement %v0, %a[%idx] : !kgen.simd<size, DT>
  // CHECK: %[[V:.*]] = pop.simd.insertelement %[[V1]], %[[B]][%[[IDX]]] : !kgen.simd<size, f32>
  %v = pop.simd.insertelement %v1, %b[%idx] : !kgen.simd<size, f32>
  // CHECK: %[[w:.*]] = pop.simd.insertelement %[[V2]], %[[C]][%[[IDX]]] : !kgen.simd<4, si32>
  %w = pop.simd.insertelement %v2, %c[%idx] : !kgen.simd<4, si32>
  kgen.return
}

// CHECK-LABEL: @pop_simd_shuffle
kgen.generator @pop_simd_shuffle<size, mask: !pop.array<2,index>>(%a: !kgen.simd<size, f32>, %b: !kgen.simd<size, f32>) {
  // CHECK: pop.simd.shuffle <size, f32> %{{.*}}, %{{.*}} -> <2, f32> :array<2, index> [1, 2]
  %0 = pop.simd.shuffle <size, f32> %a, %b -> <2, f32> :array<2, index> [1, 2]
  // CHECK: pop.simd.shuffle <size, f32> %{{.*}}, %{{.*}} -> <4, f32> :array<4, index> [1, 2, 3, 4]
  %1 = pop.simd.shuffle <size, f32> %a, %b -> <4, f32> :array<4, index> [1, 2, 3, 4]
  kgen.return
}

// CHECK-LABEL: @pop_simd_splat
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
kgen.generator @pop_simd_splat<size, DT: dtype>(%a: !kgen.scalar<f32>, %b: !kgen.scalar<DT>) -> (!kgen.simd<4, f32>, !kgen.simd<size, DT>) {
  // CHECK: %[[U:.*]] = pop.simd.splat %[[A]] : !kgen.simd<4, f32>
  %u = pop.simd.splat %a : !kgen.simd<4, f32>
  // CHECK: %[[V:.*]] = pop.simd.splat %[[B]] : !kgen.simd<size, DT>
  %v = pop.simd.splat %b : !kgen.simd<size, DT>
  // CHECK: return %[[U]], %[[V]]
  kgen.return %u, %v : !kgen.simd<4, f32>, !kgen.simd<size, DT>
}

// CHECK-LABEL: @pop_load_store
kgen.generator @pop_load_store<DT: dtype>(%p0: !kgen.pointer<scalar<f32>>, %p1: !kgen.pointer<scalar<DT>>) {
  // CHECK: %[[V0:.*]] = pop.load %{{.*}} : !kgen.pointer<scalar<f32>>
  %0 = pop.load %p0 : !kgen.pointer<scalar<f32>>
  // CHECK: %[[V1:.*]] = pop.load %{{.*}} : !kgen.pointer<scalar<DT>>
  %1 = pop.load %p1 : !kgen.pointer<scalar<DT>>
  // CHECK: %[[V2:.*]] = pop.load volatile %{{.*}} : !kgen.pointer<scalar<f32>>
  %2 = pop.load volatile<1> %p0 : !kgen.pointer<scalar<f32>>
  // CHECK: %[[V3:.*]] = pop.load volatile %{{.*}} : !kgen.pointer<scalar<DT>>
  %3 = pop.load volatile<1> %p1 : !kgen.pointer<scalar<DT>>
  // CHECK: %[[V4:.*]] = pop.load invariant %{{.*}} : !kgen.pointer<scalar<DT>>
  %4 = pop.load invariant<1> %p1 : !kgen.pointer<scalar<DT>>
  // CHECK: [[V5:%.*]] = pop.load atomic acquire %{{.*}} : !kgen.pointer<scalar<DT>>
  %5 = pop.load atomic acquire %p1 : !kgen.pointer<scalar<DT>>
  // CHECK: [[V6:%.*]] = pop.load atomic syncscope("singlethread") acquire %{{.*}} : !kgen.pointer<scalar<DT>>
  %6 = pop.load atomic syncscope("singlethread") acquire %p1 : !kgen.pointer<scalar<DT>>
  // CHECK: pop.store %[[V0]], %{{.*}} : !kgen.pointer<scalar<f32>>
  pop.store %0, %p0 : !kgen.pointer<scalar<f32>>
  // CHECK: pop.store %[[V1]], %{{.*}} : !kgen.pointer<scalar<DT>>
  pop.store %1, %p1 : !kgen.pointer<scalar<DT>>
  // CHECK: pop.store volatile %[[V0]], %{{.*}} : !kgen.pointer<scalar<f32>>
  pop.store volatile<1> %0, %p0 : !kgen.pointer<scalar<f32>>
  // CHECK: pop.store volatile %[[V1]], %{{.*}} : !kgen.pointer<scalar<DT>>
  pop.store volatile<1> %1, %p1 : !kgen.pointer<scalar<DT>>
  // CHECK: pop.store atomic release [[V5]], %{{.*}} : !kgen.pointer<scalar<DT>>
  pop.store atomic release %5, %p1 : !kgen.pointer<scalar<DT>>
  // CHECK: pop.store atomic seq_cst [[V5]], %{{.*}} : !kgen.pointer<scalar<DT>>
  pop.store atomic seq_cst %5, %p1 : !kgen.pointer<scalar<DT>>
  // CHECK: pop.store atomic syncscope("singlethread") release [[V5]], %{{.*}} : !kgen.pointer<scalar<DT>>
  pop.store atomic syncscope("singlethread") release %5, %p1 : !kgen.pointer<scalar<DT>>
  kgen.return
}

// CHECK-LABEL: @pop_load_store_alignment
kgen.generator @pop_load_store_alignment<DT: dtype>(%p0: !kgen.pointer<scalar<f32>>, %p1: !kgen.pointer<scalar<DT>>) {
  // CHECK: pop.load %{{.*}} align<42> : !kgen.pointer<scalar<f32>>
  %0 = pop.load %p0 align<42> : !kgen.pointer<scalar<f32>>
  // CHECK: pop.load %{{.*}} align<8> : !kgen.pointer<scalar<DT>>
  %1 = pop.load %p1 align<8> : !kgen.pointer<scalar<DT>>
  // CHECK: pop.store %{{.*}}, %{{.*}} align<4> : !kgen.pointer<scalar<f32>>
  pop.store %0, %p0 align<4> : !kgen.pointer<scalar<f32>>
  // CHECK: pop.store %{{.*}}, %{{.*}} align<89> : !kgen.pointer<scalar<DT>>
  pop.store %1, %p1 align<89> : !kgen.pointer<scalar<DT>>
  kgen.return
}

// CHECK-LABEL: @alignment_syntax
kgen.generator @alignment_syntax(%arg0: !kgen.pointer<index>) {
  // CHECK: align<#some.int>
  pop.load %arg0 align<#some.int> : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: @pop_load_alignment_generator
kgen.generator @pop_load_alignment_generator<alignment>(%ptr: !kgen.pointer<scalar<f32>>) -> !kgen.scalar<f32> {
  // CHECK: pop.load %{{.*}} align<alignment> : !kgen.pointer<scalar<f32>>
  %0 = pop.load %ptr align<alignment> : !kgen.pointer<scalar<f32>>
  kgen.return %0 : !kgen.scalar<f32>
}

// CHECK-LABEL: @pop_offset
kgen.generator @pop_offset<type: dtype>(%p: !kgen.pointer<scalar<f32>>, %idx: index) {
  // pop.offset %{{.*}}[{{.*}}] : !kgen.pointer<scalar<f32>>
  %0 = pop.offset %p[%idx] : !kgen.pointer<scalar<f32>>
  kgen.return
}

// CHECK-LABEL: @pop_generic_load_store
kgen.generator @pop_generic_load_store<ty: type, dt: dtype, size>(
    %p0: !kgen.pointer<ty>,
    %p1: !kgen.pointer<scalar<dt>>,
    %p2: !kgen.pointer<simd<size, dt>>)
  -> (
    !kgen.param<ty>,
    !kgen.scalar<dt>,
    !kgen.simd<size, dt>
  ) {
  // CHECK: pop.load %{{.*}} : !kgen.pointer<ty>
  // CHECK: pop.store %{{.*}} : !kgen.pointer<ty>
  %0 = pop.load %p0 : !kgen.pointer<ty>
  pop.store %0, %p0 : !kgen.pointer<ty>

  // CHECK: pop.load %{{.*}} : !kgen.pointer<scalar<dt>>
  // CHECK: pop.store %{{.*}} : !kgen.pointer<scalar<dt>>
  %1 = pop.load %p1 : !kgen.pointer<scalar<dt>>
  pop.store %1, %p1 : !kgen.pointer<scalar<dt>>

  // CHECK: pop.load %{{.*}} : !kgen.pointer<simd<size, dt>>
  // CHECK: pop.store %{{.*}} : !kgen.pointer<simd<size, dt>>
  %2 = pop.load %p2 : !kgen.pointer<simd<size, dt>>
  pop.store %2, %p2 : !kgen.pointer<simd<size, dt>>

  kgen.return %0, %1, %2 : !kgen.param<ty>, !kgen.scalar<dt>, !kgen.simd<size, dt>
}

// CHECK-LABEL: @pop_generic_offset
kgen.generator @pop_generic_offset<ty: type>(
    %p0: !kgen.pointer<ty>,
    %p1: !kgen.pointer<simd<4, f32>>,
    %i: index) {
  // CHECK: pop.offset %{{.*}} : !kgen.pointer<ty>
  %0 = pop.offset %p0[%i] : !kgen.pointer<ty>
  // CHECK: pop.offset %{{.*}} : !kgen.pointer<simd<4, f32>>
  %1 = pop.offset %p1[%i] : !kgen.pointer<simd<4, f32>>
  kgen.return
}

// CHECK-LABEL: kgen.generator @parametricAdd
// CHECK-SAME: %[[ARG0:.*]]: !kgen.simd<size, dt>, %[[ARG1:.*]]: !kgen.simd<size, dt>
kgen.generator @parametricAdd<size, dt: dtype>
  (%arg0: !kgen.simd<size, dt>, %arg1: !kgen.simd<size, dt>) -> !kgen.simd<size, dt> {

  // Fully parametric operations are ok!
  // CHECK: %{{.*}} = pop.add %[[ARG0]], %[[ARG1]] : !kgen.simd<size, dt>
  %0 = pop.add %arg0, %arg1 : !kgen.simd<size,dt>
  kgen.return %0 : !kgen.simd<size,dt>
}

// CHECK-LABEL: @stack_allocation
kgen.generator @stack_allocation<size, ty: type, address_space_val>() {
  // CHECK: pop.stack_allocation size x ty
  %0 = pop.stack_allocation size x ty
  // CHECK: pop.stack_allocation 16 x simd<4, f32>
  %1 = pop.stack_allocation 16 x !kgen.simd<4, f32>
  // CHECK: pop.stack_allocation 16 x simd<4, f32> align 8
  %2 = pop.stack_allocation 16 x !kgen.simd<4, f32> align 8
  // CHECK: pop.stack_allocation 16 x simd<4, f32> align size
  %3 = pop.stack_allocation 16 x !kgen.simd<4, f32> align size
  // CHECK: pop.stack_allocation 16 x simd<4, f32> address_space 5
  %4 = pop.stack_allocation 16 x !kgen.simd<4, f32> address_space 5
  // CHECK: pop.stack_allocation 16 x simd<4, f32> address_space 5 align 8
  %5 = pop.stack_allocation 16 x !kgen.simd<4, f32> address_space 5 align 8
  // CHECK: pop.stack_allocation 16 x simd<4, f32> address_space address_space_val
  %6 = pop.stack_allocation 16 x !kgen.simd<4, f32> address_space address_space_val
  kgen.return
}

// CHECK-LABEL: @external_call
kgen.generator @external_call<ty: type, dt: dtype>(%a: !kgen.param<ty>, %b: !kgen.scalar<dt>) {
  // CHECK: pop.external_call @foo(%{{.*}}, %{{.*}})
  %0 = pop.external_call @foo(%a, %b) : (!kgen.param<ty>, !kgen.scalar<dt>) -> !kgen.simd<4, f32>
  // CHECK: pop.external_call @bar(%arg0, %arg1)
  // CHECK-SAME: attributes {funcAttrs = ["noinline", ["alignstack", "16"]],
  // CHECK-SAME: numFixedArgs = 1 : index}
  pop.external_call @bar(%a, %b)
    attributes {funcAttrs = ["noinline", ["alignstack", "16"]],
                numFixedArgs = 1 : index}
    : (!kgen.param<ty>, !kgen.scalar<dt>) -> ()
  kgen.return
}

// CHECK-LABEL: @global_constant
kgen.generator @global_constant() {
  // CHECK: pop.global_constant: i32 = <5>
  pop.global_constant: i32 = <5>
  // CHECK: pop.global_constant: !M.array<4xui32> = <#M.dense_array<0, 1, 2, 3>>
  pop.global_constant: !M.array<4xui32> = <#M.dense_array<0, 1, 2, 3>>
  kgen.return
}

// CHECK-LABEL: @global_alloc
kgen.generator @global_alloc() {
  // CHECK-NEXT: pop.global_alloc "hello" 2 x scalar<si32> address_space 3 align 32
  %0 = pop.global_alloc "hello" 2 x !kgen.scalar<si32> address_space 3 align 32
  kgen.return
}

// CHECK-LABEL: @global_alloc_initialized
kgen.generator @global_alloc_initialized() {
  // CHECK-NEXT: pop.global_alloc "init_test" 1 x scalar<si32> = <42>
  %0 = pop.global_alloc "init_test" 1 x !kgen.scalar<si32> = <42>
  // CHECK-NEXT: pop.global_alloc "init_aligned" 1 x scalar<si32> align 16 = <7>
  %1 = pop.global_alloc "init_aligned" 1 x !kgen.scalar<si32> align 16 = <7>
  kgen.return
}

// CHECK-LABEL: @global_constant_aligned
kgen.generator @global_constant_aligned() {
  // CHECK: pop.global_constant: i32 = <5> align 16
  pop.global_constant: i32 = <5> align 16
  // CHECK: pop.global_constant: !M.array<4xui32> = <#M.dense_array<0, 1, 2, 3>>  align 64
  pop.global_constant: !M.array<4xui32> = <#M.dense_array<0, 1, 2, 3>> align 64
  kgen.return
}

// CHECK-LABEL: @pointer_to_index
kgen.generator @pointer_to_index<ty: type>(%a: !kgen.pointer<ty>, %b: !kgen.pointer<scalar<f32>>) {
  // CHECK: pop.pointer_to_index %{{.*}} : <ty>
  %0 = pop.pointer_to_index %a : !kgen.pointer<ty>
  // CHECK: pop.pointer_to_index %{{.*}} : <scalar<f32>>
  %1 = pop.pointer_to_index %b : !kgen.pointer<scalar<f32>>
  kgen.return
}

// CHECK-LABEL: @struct
kgen.generator @struct<ty: type, dt: dtype>(
  // CHECK-SAME: %[[A:.*]]: !kgen.param
  %a: !kgen.param<ty>,
  // CHECK-SAME: %[[B:.*]]: !kgen.scalar<
  %b: !kgen.scalar<dt>
) -> (!kgen.param<ty>, !kgen.scalar<dt>, !kgen.pointer<ty>) {
  // CHECK: %[[S0:.*]] = kgen.struct.create(%[[A]], %[[B]]) : !kgen.struct<(ty, scalar<dt>)>
  %0 = kgen.struct.create(%a, %b) : !kgen.struct<(ty, scalar<dt>)>
  // CHECK: %[[V0:.*]] = kgen.struct.extract %[[S0]][0] : <(ty, scalar<dt>)>
  %1 = kgen.struct.extract %0[0] : !kgen.struct<(ty, scalar<dt>)>
  // CHECK: %[[V1:.*]] = kgen.struct.extract %[[S0]][1] : <(ty, scalar<dt>)>
  %2 = kgen.struct.extract %0[1] : !kgen.struct<(ty, scalar<dt>)>
  // CHECK: kgen.struct.replace %{{.*}}, %[[S0]][0] : !kgen.struct<(ty, scalar<dt>)>
  %3 = kgen.struct.replace %1, %0[0] : !kgen.struct<(ty, scalar<dt>)>
  // CHECK: kgen.struct.replace %{{.*}}, %{{.*}}[1] : !kgen.struct<(ty, scalar<dt>)>
  %4 = kgen.struct.replace %2, %3[1] : !kgen.struct<(ty, scalar<dt>)>

  // CHECK: %[[STRUCT_PTR:.*]] = pop.stack_allocation
  %struct = pop.stack_allocation 1 x !kgen.struct<(i32, ty)>
  // CHECK: %[[EL_PTR:.*]] = kgen.struct.gep %[[STRUCT_PTR]][1] : <struct<(i32, ty)>>
  %el = kgen.struct.gep %struct[1] : <struct<(i32, ty)>>

  // CHECK: return %[[V0]], %[[V1]], %[[EL_PTR]] : !kgen.param<ty>, !kgen.scalar<dt>, !kgen.pointer<ty>
  kgen.return %1, %2, %el : !kgen.param<ty>, !kgen.scalar<dt>, !kgen.pointer<ty>
}

// CHECK-LABEL: @empty_struct_syntax
kgen.generator @empty_struct_syntax() -> !kgen.struct<()> {
  // CHECK-NEXT: kgen.struct.create() : !kgen.struct<()>
  %0 = kgen.struct.create() : !kgen.struct<()>
  kgen.return %0 : !kgen.struct<()>
}

// CHECK-LABEL: @struct_extract_parametric_index
kgen.generator @struct_extract_parametric_index<I: index>(
  // CHECK-SAME: %[[S:.*]]: !kgen.struct<(i32, f32)>
  %s: !kgen.struct<(i32, f32)>
) {
  // CHECK: kgen.struct.extract %[[S]][0] : <(i32, f32)>
  %0 = kgen.struct.extract %s[0] : !kgen.struct<(i32, f32)>
  // CHECK: kgen.struct.extract %[[S]][1] : <(i32, f32)>
  %1 = kgen.struct.extract %s[1] : !kgen.struct<(i32, f32)>
  // CHECK: kgen.struct.extract %[[S]][I] : <(i32, f32)>
  %2 = kgen.struct.extract %s[I] : !kgen.struct<(i32, f32)>
  // CHECK: kgen.struct.extract %[[S]][to_builtin(:scalar<index> add(from_builtin(I), 1))] : <(i32, f32)>
  %3 = kgen.struct.extract %s[add(I, 1)] : !kgen.struct<(i32, f32)>
  kgen.return
}

// CHECK-LABEL: @pointer_types
kgen.generator @pointer_types<dt: dtype>(
  // CHECK-SAME: %{{.*}}: !kgen.pointer<scalar<dt>>, %{{.*}}: !kgen.pointer<scalar<f32>>, %{{.*}}: !kgen.pointer<scalar<invalid>>
  %arg0: !kgen.pointer<scalar<dt>>, %arg1: !kgen.pointer<scalar<f32>>, %arg2: !kgen.pointer<scalar<invalid>>) {
  kgen.return
}

// CHECK-LABEL: @cast_to_builtin
// CHECK-SAME: %[[ARG0:.*]]: !kgen.scalar<f32>, %[[ARG1:.*]]: !kgen.scalar<si32>
kgen.func @cast_to_builtin(%arg0: !kgen.scalar<f32>, %arg1: !kgen.scalar<si32>) {
  // CHECK: pop.cast_to_builtin %[[ARG0]] : !kgen.scalar<f32> to f32
  %0 = pop.cast_to_builtin %arg0: !kgen.scalar<f32> to f32
  // CHECK: pop.cast_to_builtin %[[ARG1]] : !kgen.scalar<si32> to i32
  %1 = pop.cast_to_builtin %arg1: !kgen.scalar<si32> to i32
  kgen.return
}

// CHECK-LABEL: @cast_from_builtin
// CHECK-SAME: %[[ARG0:.*]]: f32, %[[ARG1:.*]]: ui32
kgen.func @cast_from_builtin(%arg0: f32, %arg1: ui32) {
  // CHECK: pop.cast_from_builtin %[[ARG0]] : f32 to !kgen.scalar<f32>
  %0 = pop.cast_from_builtin %arg0: f32 to !kgen.scalar<f32>
  // CHECK: pop.cast_from_builtin %[[ARG1]] : ui32 to !kgen.scalar<ui32>
  %1 = pop.cast_from_builtin %arg1: ui32 to !kgen.scalar<ui32>
  kgen.return
}

// CHECK-LABEL: @cast_from_builtin_vector
// CHECK-SAME: %[[ARG:.*]]:
kgen.func @cast_from_builtin_vector(%arg0: vector<2xf32>) -> !kgen.simd<2, f32> {
  // CHECK: %[[V0:.*]] = pop.cast_from_builtin %[[ARG]] : vector<2xf32> to !kgen.simd<2, f32>
  %0 = pop.cast_from_builtin %arg0 : vector<2xf32> to !kgen.simd<2, f32>
  // CHECK: kgen.return %[[V0:.*]] : !kgen.simd<2, f32>
  kgen.return %0 : !kgen.simd<2, f32>
}

// CHECK-LABEL: @array_ops
kgen.generator @array_ops<idx, N, T: type, dtype: dtype>(%arg0: !kgen.param<T>)
    -> (!pop.array<2, T>, !kgen.pointer<T>) {
  // CHECK: pop.array.create [%arg0, %arg0] : !pop.array<2, T>
  %0 = pop.array.create [%arg0, %arg0] : !pop.array<2, T>
  // CHECK: pop.array.get %0[1] : !pop.array<2, T>
  %1 = pop.array.get %0[1] : !pop.array<2, T>
  // CHECK: pop.array.replace %{{.*}}, %0[0] : !pop.array<2, T>
  %2 = pop.array.replace %1, %0[0] : !pop.array<2, T>

  // CHECK: %[[ARR_PTR:.*]] = pop.stack_allocation 1 x array<4, T>
  %5 = pop.stack_allocation 1 x !pop.array<4, T>
  // CHECK: %[[IDX:.*]] = index.constant
  %6 = index.constant 2
  // CHECK: pop.array.gep %[[ARR_PTR]][%[[IDX]]] : <array<4, T>>
  %7 = pop.array.gep %5[%6] : <array<4, T>>

  // CHECK: pop.array.get %{{.*}}[idx]
  %8 = pop.array.get %0[idx] : !pop.array<2, T>
  // CHECK: pop.array.replace %arg0, %{{.*}}[idx]
  %9 = pop.array.replace %arg0, %0[idx] : !pop.array<2, T>

  kgen.return %2, %7 : !pop.array<2, T>, !kgen.pointer<T>
}

// CHECK-LABEL: @call_intrinsic
kgen.generator @call_intrinsic<intrin: string>(%arg0: !kgen.scalar<f32>) {
  // CHECK-NEXT: %{{.*}} = pop.call_llvm_intrinsic "llvm.round", (%arg0) : (!kgen.scalar<f32>) -> !kgen.scalar<f32>
  %0 = pop.call_llvm_intrinsic "llvm.round", (%arg0) : (!kgen.scalar<f32>) -> !kgen.scalar<f32>
  // CHECK-NEXT: pop.call_llvm_intrinsic intrin, ()
  pop.call_llvm_intrinsic intrin, () : () -> ()
  kgen.return
}

// CHECK-LABEL: @inline_asm
kgen.generator @inline_asm<ty: type, dt: dtype>(
    %arg0: !kgen.scalar<si32>,
    %arg1: !kgen.scalar<index>,
    %arg2: !kgen.param<ty>,
    %arg3: !kgen.scalar<dt>) {
  // CHECK: pop.inline_asm "bswap $0", "=r,r", (%arg0) : (!kgen.scalar<si32>) -> i8
  %0 = pop.inline_asm "bswap $0", "=r,r", (%arg0) : (!kgen.scalar<si32>) -> i8
  // CHECK: pop.inline_asm "something", "anotherthing", (%arg0, %arg1) :
  // CHECK: (!kgen.scalar<si32>, !kgen.scalar<index>) -> i8
  %1 = pop.inline_asm "something", "anotherthing", (%arg0, %arg1) :
    (!kgen.scalar<si32>, !kgen.scalar<index>) -> i8
  // CHECK: pop.inline_asm side_effecting "something", "anotherthing", (%arg0, %arg1) :
  // CHECK: (!kgen.scalar<si32>, !kgen.scalar<index>) -> i8
  %2 = pop.inline_asm side_effecting<1> "something", "anotherthing", (%arg0, %arg1) :
    (!kgen.scalar<si32>, !kgen.scalar<index>) -> i8
  // CHECK: pop.inline_asm stack_aligned "something", "anotherthing", (%arg0, %arg1) :
  // CHECK: (!kgen.scalar<si32>, !kgen.scalar<index>) -> i8
  %3 = pop.inline_asm stack_aligned<1> "something", "anotherthing", (%arg0, %arg1) :
    (!kgen.scalar<si32>, !kgen.scalar<index>) -> i8
  // CHECK: pop.inline_asm "foo", "=r,=r,r", (%arg0) : (!kgen.scalar<si32>) ->
  // CHECK: !kgen.struct<(ty, scalar<dt>)>
  %4 = pop.inline_asm "foo", "=r,=r,r", (%arg0) : (!kgen.scalar<si32>) ->
    !kgen.struct<(ty, scalar<dt>)>
  // CHECK: pop.inline_asm "bar $0", "=r,r", (%arg2) : (!kgen.param<ty>) -> i8
  %5 = pop.inline_asm "bar $0", "=r,r", (%arg2) : (!kgen.param<ty>) -> i8
  // CHECK: pop.inline_asm "bar $0", "=r,r", (%arg3) : (!kgen.scalar<dt>) -> i8
  %6 = pop.inline_asm "bar $0", "=r,r", (%arg3) : (!kgen.scalar<dt>) -> i8
  // CHECK: [[RET:%.*]]:2 = pop.inline_asm "bar $0", "=r,=r,r", (%arg0) : (!kgen.scalar<si32>) -> (i8, i8)
  %7:2 = pop.inline_asm "bar $0", "=r,=r,r", (%arg0) : (!kgen.scalar<si32>) -> (i8, i8)
  kgen.return
}

// CHECK-LABEL: @usesAGlobal
kgen.func @usesAGlobal() {
  %zero = index.constant 0
  // CHECK: pop.compiler.global_load "aGlobal" : index
  %0 = pop.compiler.global_load "aGlobal" : index
  // CHECK: pop.compiler.global_store "aGlobal", %idx0 : index
  pop.compiler.global_store "aGlobal", %zero : index
  kgen.return
}

// CHECK-LABEL: kgen.generator @atomic_cmpxchg
// CHECK-SAME: %[[PTR:.*]]: !kgen.pointer<scalar<index>>,
// CHECK-SAME: %[[CMP:.*]]: !kgen.scalar<index>,
// CHECK-SAME: %[[NEW:.*]]: !kgen.scalar<index>
kgen.generator @atomic_cmpxchg<scope: string>(%ptr: !kgen.pointer<scalar<index>>,
                                              %cmp: !kgen.scalar<index>,
                                              %new: !kgen.scalar<index>) {
  // CHECK: pop.atomic.cmpxchg %[[PTR]], %[[CMP]], %[[NEW]] monotonic monotonic
  %0 = pop.atomic.cmpxchg %ptr, %cmp, %new monotonic monotonic :
                    !kgen.pointer<scalar<index>>
  // CHECK: pop.atomic.cmpxchg weak %[[PTR]], %[[CMP]], %[[NEW]] monotonic monotonic
  %1 = pop.atomic.cmpxchg weak %ptr, %cmp, %new monotonic monotonic :
                    !kgen.pointer<scalar<index>>
  // CHECK: pop.atomic.cmpxchg %[[PTR]], %[[CMP]], %[[NEW]] seq_cst acq_rel
  %2 = pop.atomic.cmpxchg %ptr, %cmp, %new seq_cst acq_rel :
                    !kgen.pointer<scalar<index>>
  // CHECK: pop.atomic.cmpxchg %[[PTR]], %[[CMP]], %[[NEW]] syncscope(scope) seq_cst acq_rel
  %3 = pop.atomic.cmpxchg %ptr, %cmp, %new syncscope(scope) seq_cst acq_rel :
                    !kgen.pointer<scalar<index>>
  // CHECK: pop.atomic.cmpxchg %[[PTR]], %[[CMP]], %[[NEW]] syncscope("agent") seq_cst acq_rel
  %4 = pop.atomic.cmpxchg %ptr, %cmp, %new syncscope("agent") seq_cst acq_rel :
                    !kgen.pointer<scalar<index>>
  // CHECK: pop.atomic.cmpxchg weak %[[PTR]], %[[CMP]], %[[NEW]] syncscope(scope) seq_cst acq_rel
  %5 = pop.atomic.cmpxchg weak %ptr, %cmp, %new syncscope(scope) seq_cst acq_rel :
                    !kgen.pointer<scalar<index>>
  kgen.return
}

// CHECK-LABEL: kgen.generator @atomic_rmw
// CHECK-SAME: %[[PTR:.*]]: !kgen.pointer<scalar<index>>,
// CHECK-SAME: %[[VAL:.*]]: !kgen.scalar<index>
kgen.generator @atomic_rmw<scope: string>(%ptr: !kgen.pointer<scalar<index>>,
                                          %val: !kgen.scalar<index>) {
  // CHECK: pop.atomic.rmw add(%[[PTR]], %[[VAL]]) monotonic
  %0 = pop.atomic.rmw add(%ptr, %val) monotonic : !kgen.pointer<scalar<index>>
  // CHECK: pop.atomic.rmw sub(%[[PTR]], %[[VAL]]) monotonic
  %1 = pop.atomic.rmw sub(%ptr, %val) monotonic : !kgen.pointer<scalar<index>>
  // CHECK: pop.atomic.rmw xor(%[[PTR]], %[[VAL]]) monotonic
  %2 = pop.atomic.rmw xor(%ptr, %val) monotonic : !kgen.pointer<scalar<index>>
  // CHECK: pop.atomic.rmw min(%[[PTR]], %[[VAL]]) monotonic
  %3 = pop.atomic.rmw min(%ptr, %val) monotonic : !kgen.pointer<scalar<index>>
  // CHECK: pop.atomic.rmw max(%[[PTR]], %[[VAL]]) monotonic
  %4 = pop.atomic.rmw max(%ptr, %val) monotonic : !kgen.pointer<scalar<index>>
  // CHECK: pop.atomic.rmw max(%[[PTR]], %[[VAL]]) syncscope(scope) monotonic
  %5 = pop.atomic.rmw max(%ptr, %val) syncscope(scope)
                                      monotonic : !kgen.pointer<scalar<index>>
  // CHECK: pop.atomic.rmw max(%[[PTR]], %[[VAL]]) syncscope("agent") monotonic
  %6 = pop.atomic.rmw max(%ptr, %val) syncscope("agent")
                                      monotonic : !kgen.pointer<scalar<index>>
  kgen.return
}

// CHECK-LABEL: kgen.func @string_ops(%arg0: !kgen.string, %arg1: !kgen.string, %arg2: !kgen.string) -> index
kgen.func @string_ops(%a: !kgen.string,
                      %src: !kgen.string,
                      %target: !kgen.string) ->  index {
  // CHECK: pop.string.address %arg0
  %0 = pop.string.address %a
  // CHECK: pop.string.size %arg0
  %1 = pop.string.size %a
  kgen.return %1: index
}

// CHECK-LABEL: kgen.generator @dtype_utils
kgen.generator @dtype_utils<DT: dtype>(%arg0: !kgen.dtype) {
  // CHECK: %[[V0:.*]] = pop.dtype.to_ui8 %arg0
  %v0 = pop.dtype.to_ui8 %arg0
  // CHECK: pop.dtype.from_ui8 %[[V0]]
  %x0 = pop.dtype.from_ui8 %v0

  // CHECK: %[[PARAM:.*]] = kgen.param.constant
  %t0 = kgen.param.constant : dtype = <DT>

  // CHECK: %[[V1:.*]] = pop.dtype.to_ui8 %[[PARAM]]
  %v1 = pop.dtype.to_ui8 %t0
  // CHECK: pop.dtype.from_ui8 %[[V1]]
  %x1 = pop.dtype.from_ui8 %v1

  kgen.return
}


// CHECK-LABEL: @aligned_alloc
kgen.func @aligned_alloc(%arg0: index, %arg1: index) {
  // CHECK-NEXT: %0 = pop.aligned_alloc %arg0, %arg1 : <index>
  %0 = pop.aligned_alloc %arg0, %arg1 : <index>
  // CHECK-NEXT: pop.aligned_free %0 : <index>
  pop.aligned_free %0 : <index>
  kgen.return
}

// CHECK-LABEL: @fence
kgen.func @fence() {
  // CHECK: pop.fence acquire
  pop.fence acquire
  // CHECK: pop.fence syncscope("agent") seq_cst
  pop.fence syncscope("agent") seq_cst
  // CHECK: pop.fence syncscope("singlethread") acq_rel
  pop.fence syncscope("singlethread") acq_rel
  kgen.return
}

// CHECK-LABEL: @stack_lifetime
kgen.func @stack_lifetime() {
  %0 = pop.stack_allocation 1 x index marked
  %1 = pop.stack_allocation 1 x index marked
  // CHECK: pop.stack_alloc.lifetime.start(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>
  pop.stack_alloc.lifetime.start(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>
  // CHECK: pop.stack_alloc.lifetime.end(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>
  pop.stack_alloc.lifetime.end(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: @variant_bitcast
kgen.generator @variant_bitcast<idx, Ts: param_list<type>>(%arg0: !kgen.pointer<variant<i32, i64>>, %arg1: !kgen.pointer<variant<[Ts]>>) -> (!kgen.pointer<i64>, !kgen.pointer<i32>) {
  // CHECK-NEXT: pop.variant.bitcast %arg0, <1> : <variant<i32, i64>> as <i64>
  %0 = pop.variant.bitcast %arg0, <1> : <variant<i32, i64>> as <i64>
  // CHECK-NEXT: pop.variant.bitcast %arg1, <idx> : <variant<[Ts]>> as <i32>
  %1 = pop.variant.bitcast %arg1, <idx> : <variant<[Ts]>> as <i32>
  kgen.return %0, %1 : !kgen.pointer<i64>, !kgen.pointer<i32>
}

// CHECK-LABEL: @variant_discr_gep
kgen.generator @variant_discr_gep<Ts: param_list<type>, dt: dtype>(%arg0: !kgen.pointer<variant<i8, i16, i32>>, %arg1: !kgen.pointer<variant<[Ts]>>) {
  // CHECK-NEXT: pop.variant.discr_gep %arg0 : <variant<i8, i16, i32>> as <scalar<ui8>>
  %0 = pop.variant.discr_gep %arg0 : <variant<i8, i16, i32>> as <scalar<ui8>>
  // CHECK-NEXT: pop.variant.discr_gep %arg1 : <variant<[Ts]>> as <scalar<dt>>
  %1 = pop.variant.discr_gep %arg1 : <variant<[Ts]>> as <scalar<dt>>
  kgen.return
}

// CHECK-LABEL: @union_bitcast
kgen.func @union_bitcast(%arg0: !kgen.pointer<union<i32, i64>>) -> !kgen.pointer<i32> {
  // CHECK-NEXT: pop.union.bitcast %arg0 : <union<i32, i64>> as <i32>
  %0 = pop.union.bitcast %arg0 : <union<i32, i64>> as <i32>
  kgen.return %0 : !kgen.pointer<i32>
}

// CHECK-LABEL: @union_wrap_unwrap
kgen.func @union_wrap_unwrap(%arg0: i32) -> i64 {
  // CHECK-NEXT: %0 = pop.union.wrap %arg0 : i32 as <i64, i32>
  %0 = pop.union.wrap %arg0 : i32 as <i64, i32>
  // CHECK-NEXT: %1 = pop.union.unwrap %0 : <i64, i32> as i64
  %1 = pop.union.unwrap %0 : <i64, i32> as i64
  kgen.return %1 : i64
}

// CHECK-LABEL: kgen.func @noalias_cast
kgen.func @noalias_cast(%arg0: !kgen.pointer<index>) {
  // CHECK-NEXT: pop.noalias_pointer_cast %arg0 : <index>
  %0 = pop.noalias_pointer_cast %arg0 : <index>
  kgen.return
}

// CHECK-LABEL: kgen.func @bitcast_scalar_index
kgen.func @bitcast_scalar_index(%a: !kgen.scalar<index>) {
  // CHECK-NEXT: %[[R0:.+]] = pop.bitcast %arg0 : !kgen.scalar<index> to !kgen.scalar<ui64>
  // CHECK-NEXT: %[[R1:.+]] = pop.bitcast %[[R0]] : !kgen.scalar<ui64> to !kgen.scalar<index>
  %0 = pop.bitcast %a : !kgen.scalar<index> to !kgen.scalar<ui64>
  %1 = pop.bitcast %0 : !kgen.scalar<ui64> to !kgen.scalar<index>
  kgen.return
}
