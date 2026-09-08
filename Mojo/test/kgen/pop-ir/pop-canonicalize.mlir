// RUN: kgen-opt -split-input-file %s -canonicalize | FileCheck %s

// -----

// CHECK-LABEL: @neg
kgen.func @neg() -> (!kgen.simd<2, si8>, !kgen.simd<2, f32>) {
  // CHECK-DAG: <1, -1>
  // CHECK-DAG: <"1.25", "-1.25">
  %0 = kgen.param.constant: simd<2, si8> = <<-1, 1>>
  %1 = kgen.param.constant: simd<2, f32> = <<"-1.25", "1.25">>
  %2 = pop.neg %0 : !kgen.simd<2, si8>
  %3 = pop.neg %1 : !kgen.simd<2, f32>
  kgen.return %2, %3 : !kgen.simd<2, si8>, !kgen.simd<2, f32>
}

// CHECK-LABEL: @floor
kgen.func @floor() -> (!kgen.simd<2, f32>, !kgen.simd<2, si8>,
                       !kgen.scalar<index>, !kgen.scalar<uindex>) {
  // CHECK-DAG: <"1", "-2">
  // CHECK-DAG: <3, -4>
  // CHECK-DAG: <7>
  // CHECK-DAG: <8>
  %0 = kgen.param.constant: simd<2, f32> = <<"1.9", "-1.2">>
  %1 = kgen.param.constant: simd<2, si8> = <<3, -4>>
  %2 = kgen.param.constant: scalar<index> = <7>
  %3 = kgen.param.constant: scalar<uindex> = <8>
  %4 = pop.floor %0 : !kgen.simd<2, f32>
  %5 = pop.floor %1 : !kgen.simd<2, si8>
  %6 = pop.floor %2 : !kgen.scalar<index>
  %7 = pop.floor %3 : !kgen.scalar<uindex>
  kgen.return %4, %5, %6, %7 : !kgen.simd<2, f32>, !kgen.simd<2, si8>,
                               !kgen.scalar<index>, !kgen.scalar<uindex>
}

// CHECK-LABEL: @ceil
kgen.func @ceil() -> (!kgen.simd<2, f32>, !kgen.simd<2, si8>,
                       !kgen.scalar<index>, !kgen.scalar<uindex>) {
  // CHECK-DAG: <"2", "-1">
  // CHECK-DAG: <3, -4>
  // CHECK-DAG: <7>
  // CHECK-DAG: <8>
  %0 = kgen.param.constant: simd<2, f32> = <<"1.9", "-1.2">>
  %1 = kgen.param.constant: simd<2, si8> = <<3, -4>>
  %2 = kgen.param.constant: scalar<index> = <7>
  %3 = kgen.param.constant: scalar<uindex> = <8>
  %4 = pop.ceil %0 : !kgen.simd<2, f32>
  %5 = pop.ceil %1 : !kgen.simd<2, si8>
  %6 = pop.ceil %2 : !kgen.scalar<index>
  %7 = pop.ceil %3 : !kgen.scalar<uindex>
  kgen.return %4, %5, %6, %7 : !kgen.simd<2, f32>, !kgen.simd<2, si8>,
                               !kgen.scalar<index>, !kgen.scalar<uindex>
}

// CHECK-LABEL: @trunc
kgen.func @trunc() -> (!kgen.simd<2, f32>, !kgen.simd<2, si8>,
                       !kgen.scalar<index>, !kgen.scalar<uindex>) {
  // CHECK-DAG: <"1", "-1">
  // CHECK-DAG: <3, -4>
  // CHECK-DAG: <7>
  // CHECK-DAG: <8>
  %0 = kgen.param.constant: simd<2, f32> = <<"1.9", "-1.2">>
  %1 = kgen.param.constant: simd<2, si8> = <<3, -4>>
  %2 = kgen.param.constant: scalar<index> = <7>
  %3 = kgen.param.constant: scalar<uindex> = <8>
  %4 = pop.trunc %0 : !kgen.simd<2, f32>
  %5 = pop.trunc %1 : !kgen.simd<2, si8>
  %6 = pop.trunc %2 : !kgen.scalar<index>
  %7 = pop.trunc %3 : !kgen.scalar<uindex>
  kgen.return %4, %5, %6, %7 : !kgen.simd<2, f32>, !kgen.simd<2, si8>,
                               !kgen.scalar<index>, !kgen.scalar<uindex>
}

// CHECK-LABEL: @add
kgen.func @add() -> (!kgen.scalar<si8>, !kgen.scalar<f32>) {
  // CHECK-DAG: <4>
  // CHECK-DAG: <"-2.5">
  %0 = kgen.param.constant: scalar<si8> = <<2>>
  %1 = kgen.param.constant: scalar<f32> = <<"-1.25">>
  %2 = pop.add %0, %0 : !kgen.scalar<si8>
  %3 = pop.add %1, %1 : !kgen.scalar<f32>
  kgen.return %2, %3 : !kgen.scalar<si8>, !kgen.scalar<f32>
}

// CHECK-LABEL: @add_zero
kgen.func @add_zero(%arg0: !kgen.simd<2, si8>,
                 %arg1: !kgen.scalar<f32>) -> (!kgen.simd<2, si8>,
                                              !kgen.simd<2, si8>,
                                              !kgen.simd<2, si8>,
                                              !kgen.scalar<f32>) {
  %0 = kgen.param.constant: simd<2, si8> = <<0, 0>>
  // CHECK: %[[ZERO_ONE:.*]] = kgen.param.constant: simd<2, si8> = <<0, 1>>
  %1 = kgen.param.constant: simd<2, si8> = <<0, 1>>
  // CHECK: %[[FP_ZERO:.*]] = kgen.param.constant: scalar<f32> = <"0">
  %2 = kgen.param.constant: scalar<f32> = <<"0.0">>

  // x+0 -> x
  %3 = pop.add %arg0, %0 : !kgen.simd<2, si8>
  // 0+x -> x
  %4 = pop.add %0, %arg0 : !kgen.simd<2, si8>

  // negative case: %1 is not all zeros
  // CHECK: %[[R1:.*]] = pop.add %arg0, %[[ZERO_ONE]] : !kgen.simd<2, si8>
  %5 = pop.add %arg0, %1 : !kgen.simd<2, si8>

  // negative case: non-integer type
  // CHECK: %[[R2:.*]] = pop.add %arg1, %[[FP_ZERO]] : !kgen.scalar<f32>
  %6 = pop.add %arg1, %2 : !kgen.scalar<f32>

  // First two outputs are reduced to %arg0, remaining two are intact
  // CHECK: return %arg0, %arg0, %[[R1]], %[[R2]]
  kgen.return %3, %4, %5, %6 : !kgen.simd<2, si8>, !kgen.simd<2, si8>, !kgen.simd<2, si8>, !kgen.scalar<f32>
}

// CHECK-LABEL: @sub
kgen.func @sub() -> (!kgen.scalar<si8>, !kgen.scalar<f32>) {
  // CHECK-DAG: <-2>
  // CHECK-DAG: <"-1.25">
  %0 = kgen.param.constant: scalar<si8> = <<2>>
  %1 = kgen.param.constant: scalar<si8> = <<4>>
  %2 = kgen.param.constant: scalar<f32> = <<"1.25">>
  %3 = kgen.param.constant: scalar<f32> = <<"2.5">>
  %4 = pop.sub %0, %1 : !kgen.scalar<si8>
  %5 = pop.sub %2, %3 : !kgen.scalar<f32>
  kgen.return %4, %5 : !kgen.scalar<si8>, !kgen.scalar<f32>
}

// CHECK-LABEL: @sub_zero
kgen.func @sub_zero(%arg0: !kgen.simd<2, si8>,
                 %arg1: !kgen.scalar<f32>) -> (!kgen.simd<2, si8>,
                                              !kgen.simd<2, si8>,
                                              !kgen.simd<2, si8>,
                                              !kgen.scalar<f32>) {
  // CHECK-DAG: %[[ZERO_ZERO:.*]] = kgen.param.constant: simd<2, si8> = <0>
  %0 = kgen.param.constant: simd<2, si8> = <<0, 0>>
  // CHECK-DAG: %[[ZERO_ONE:.*]] = kgen.param.constant: simd<2, si8> = <<0, 1>>
  %1 = kgen.param.constant: simd<2, si8> = <<0, 1>>
  // CHECK-DAG: %[[FP_ZERO:.*]] = kgen.param.constant: scalar<f32> = <"0">
  %2 = kgen.param.constant: scalar<f32> = <<"0.0">>

  // x-0 -> x
  %3 = pop.sub %arg0, %0 : !kgen.simd<2, si8>

  // negative case: 0-x
  // CHECK: %[[R0:.*]] = pop.sub %[[ZERO_ZERO]], %arg0 : !kgen.simd<2, si8>
  %4 = pop.sub %0, %arg0 : !kgen.simd<2, si8>

  // negative case: vector value %1 is not all zeros
  // CHECK: %[[R1:.*]] = pop.sub %arg0, %[[ZERO_ONE]] : !kgen.simd<2, si8>
  %5 = pop.sub %arg0, %1 : !kgen.simd<2, si8>

  // negative case: non-integer type
  // CHECK: %[[R2:.*]] = pop.sub %arg1, %[[FP_ZERO]] : !kgen.scalar<f32>
  %6 = pop.sub %arg1, %2 : !kgen.scalar<f32>

  // First output is reduced to %arg0, remaining ones are intact
  // CHECK: return %arg0, %[[R0]], %[[R1]], %[[R2]]
  kgen.return %3, %4, %5, %6 : !kgen.simd<2, si8>, !kgen.simd<2, si8>, !kgen.simd<2, si8>, !kgen.scalar<f32>
}


// CHECK-LABEL: @mul
kgen.func @mul() -> (!kgen.scalar<si8>, !kgen.scalar<f32>) {
  // CHECK-DAG: <4>
  // CHECK-DAG: <"6.25">
  %0 = kgen.param.constant: scalar<si8> = <<2>>
  %1 = kgen.param.constant: scalar<f32> = <<"2.5">>
  %2 = pop.mul %0, %0 : !kgen.scalar<si8>
  %3 = pop.mul %1, %1 : !kgen.scalar<f32>
  kgen.return %2, %3 : !kgen.scalar<si8>, !kgen.scalar<f32>
}

// CHECK-LABEL: @mul_zero_one
kgen.func @mul_zero_one(
    %si32: !kgen.scalar<si32>,
    %vsi32 : !kgen.simd<2, si32>) -> (!kgen.scalar<si32>, !kgen.scalar<si32>,
                                     !kgen.scalar<si32>, !kgen.scalar<si32>,
                                     !kgen.simd<2, si32>, !kgen.simd<2, si32>,
                                     !kgen.simd<2, si32>, !kgen.simd<2, si32>) {
  %zero = kgen.param.constant: scalar<si32> = <<0>>
  %one = kgen.param.constant: scalar<si32> = <<1>>
  %0 = pop.mul %si32, %zero : !kgen.scalar<si32>
  %1 = pop.mul %zero, %si32 : !kgen.scalar<si32>

  %2 = pop.mul %si32, %one : !kgen.scalar<si32>
  %3 = pop.mul %one, %si32 : !kgen.scalar<si32>

  %vzero = kgen.param.constant: simd<2, si32> = <<0, 0>>
  %vone = kgen.param.constant: simd<2, si32> = <<1, 1>>

  %4 = pop.mul %vsi32, %vzero : !kgen.simd<2, si32>
  %5 = pop.mul %vzero, %vsi32 : !kgen.simd<2, si32>

  %6 = pop.mul %vsi32, %vone : !kgen.simd<2, si32>
  %7 = pop.mul %vone, %vsi32 : !kgen.simd<2, si32>

  // CHECK-DAG: %[[SIMD_ZERO:.*]] = kgen.param.constant: simd<2, si32> = <0>
  // CHECK-DAG: %[[ZERO:.*]] = kgen.param.constant: scalar<si32> = <0>
  // CHECK: kgen.return %[[ZERO]], %[[ZERO]], %arg0, %arg0, %[[SIMD_ZERO]], %[[SIMD_ZERO]], %arg1, %arg1
  kgen.return %0, %1, %2, %3,
              %4, %5, %6, %7 : !kgen.scalar<si32>, !kgen.scalar<si32>,
                               !kgen.scalar<si32>, !kgen.scalar<si32>,
                               !kgen.simd<2, si32>, !kgen.simd<2, si32>,
                               !kgen.simd<2, si32>, !kgen.simd<2, si32>
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="e-p:64:64", simd_bit_width = 128, index_bit_width = 64>} {
// CHECK-LABEL: @div
kgen.func @div(%arg0: !kgen.scalar<si64>, %arg1: !kgen.simd<2, si32>, %arg2: !kgen.simd<4, index>, %arg3: !kgen.simd<4, uindex>) -> (
    !kgen.scalar<si4>, !kgen.scalar<ui4>, !kgen.scalar<f32>,
    !kgen.scalar<si64>, !kgen.simd<2, si32>, !kgen.scalar<ui32>,
    !kgen.simd<2, ui32>, !kgen.scalar<index>, !kgen.simd<4, index>,
    !kgen.simd<4, index>, !kgen.simd<4, uindex>
  ) {
  // CHECK-DAG: %[[TWO_INDEX:.*]] = kgen.param.constant: scalar<index> = <2>
  // CHECK-DAG: %[[ONE_U32:.*]] = kgen.param.constant: scalar<ui32> = <1>
  // CHECK-DAG: %[[ONE_TWO:.*]] = kgen.param.constant: simd<2, ui32> = <<1, 2>>
  // CHECK-DAG: %[[TWO:.*]] = kgen.param.constant: scalar<si64> = <2>
  // CHECK-DAG: %[[SIMD_FOUR:.*]] = kgen.param.constant: simd<2, si32> = <4>
  // CHECK-DAG: %[[SIMD_INDEX:.*]] = kgen.param.constant: simd<4, index> = <<2, 4, 8, 16>>
  // CHECK-DAG: %[[SIMD_UINDEX:.*]] = kgen.param.constant: simd<4, uindex> = <<1, 2, 3, 4>>

  %0 = kgen.param.constant: scalar<si4> = <7>
  %1 = kgen.param.constant: scalar<si4> = <-2>
  %2 = kgen.param.constant: scalar<ui4> = <7>
  %3 = kgen.param.constant: scalar<ui4> = <-2>
  %4 = kgen.param.constant: scalar<f32> = <"2.5">
  %5 = kgen.param.constant: scalar<f32> = <"2">

  // CHECK-DAG: <si4> = <-3
  %6 = pop.div %0, %1 : !kgen.scalar<si4>

  // CHECK-DAG: <ui4> = <0>
  %7 = pop.div %2, %3 : !kgen.scalar<ui4>

  // CHECK-DAG: <"1.25">
  %8 = pop.div %4, %5 : !kgen.scalar<f32>

  // Check the canonicalization of x / 2^n into x >> n

  // CHECK-DAG: pop.div %arg0, %[[TWO]] : !kgen.scalar<si64>
  %two_int = kgen.param.constant: scalar<si64> = <2>
  %9 = pop.div %arg0, %two_int : !kgen.scalar<si64>

  // CHECK-DAG: pop.div %arg1, %[[SIMD_FOUR]]
  %four_simd = kgen.param.constant: simd<2, si32> = <<4, 4>>
  %10 = pop.div %arg1, %four_simd : !kgen.simd<2, si32>

  // CHECK-DAG: %[[U32:.*]] = pop.cast %{{.*}} : !kgen.scalar<si64> to !kgen.scalar<ui32>
  // CHECK-DAG: pop.shr %[[U32]], %[[ONE_U32]] : !kgen.scalar<ui32>
  %two_uint = kgen.param.constant: scalar<ui32> = <2>
  %uarg0 = pop.cast %arg0 : !kgen.scalar<si64> to !kgen.scalar<ui32>
  %11 = pop.div %uarg0, %two_uint : !kgen.scalar<ui32>

  // CHECK-DAG: %[[SIMD_U32:.*]] = pop.cast %{{.*}} : !kgen.simd<2, si32> to !kgen.simd<2, ui32>
  // CHECK-DAG: pop.shr %[[SIMD_U32]], %[[ONE_TWO]] : !kgen.simd<2, ui32>
  %uarg1 = pop.cast %arg1 : !kgen.simd<2, si32> to !kgen.simd<2, ui32>
  %two_four_simd = kgen.param.constant: simd<2, ui32> = <<2, 4>>
  %12 = pop.div %uarg1, %two_four_simd : !kgen.simd<2, ui32>

  // CHECK-DAG: %[[INDEX:.*]] = pop.cast %{{.*}} : !kgen.scalar<si64> to !kgen.scalar<index>
  // CHECK-DAG: pop.div %[[INDEX]], %[[TWO_INDEX]]
  %two_index = kgen.param.constant: scalar<index> = <2>
  %iarg0 = pop.cast %arg0 : !kgen.scalar<si64> to !kgen.scalar<index>
  %13 = pop.div %iarg0, %two_index : !kgen.scalar<index>

  // CHECK-DAG: pop.div %arg2, %[[SIMD_INDEX]] : !kgen.simd<4, index>
  %14 = kgen.param.constant: simd<4, index> = <<2, 4, 8, 16>>
  %15 = pop.div %arg2, %14 : !kgen.simd<4, index>

  // CHECK-DAG: simd<4, index> = <<4, 2, 1, 2>>
  %16 = kgen.param.constant: simd<4, index> = <<9, 9, 8, 32>>
  %17 = kgen.param.constant: simd<4, index> = <<2, 4, 8, 16>>
  %18 = pop.div %16, %17 : !kgen.simd<4, index>

  // CHECK-DAG: pop.shr %arg3, %[[SIMD_UINDEX]]
  %19 = kgen.param.constant: simd<4, uindex> = <<2, 4, 8, 16>>
  %20 = pop.div %arg3, %19 : !kgen.simd<4, uindex>

  kgen.return %6, %7, %8, %9, %10, %11, %12, %13, %15, %18, %20 : !kgen.scalar<si4>, !kgen.scalar<ui4>,
                                                   !kgen.scalar<f32>, !kgen.scalar<si64>,
                                                   !kgen.simd<2, si32>, !kgen.scalar<ui32>,
                                                   !kgen.simd<2, ui32>, !kgen.scalar<index>,
                                                   !kgen.simd<4, index>, !kgen.simd<4, index>,
                                                   !kgen.simd<4, uindex>
}

// CHECK-LABEL: @div_zero
kgen.func @div_zero() -> (!kgen.scalar<si4>, !kgen.scalar<f32>) {
  %0 = kgen.param.constant: scalar<si4> = <0>
  %1 = kgen.param.constant: scalar<f32> = <"0">
  // Integer division by zero is UB - don't fold.
  // CHECK-DAG: %[[ZERO:.*]] = kgen.param.constant: scalar<si4> = <0>
  %2 = pop.div %0, %0 : !kgen.scalar<si4>
  // Float division by zero folds per IEEE 754 (0.0/0.0 = NaN).
  // CHECK-DAG: %[[NAN:.*]] = kgen.param.constant: scalar<f32> = <"NaN">
  %3 = pop.div %1, %1 : !kgen.scalar<f32>
  // CHECK: %[[DIV:.*]] = pop.div %[[ZERO]], %[[ZERO]] : !kgen.scalar<si4>
  // CHECK: kgen.return %[[DIV]], %[[NAN]] : !kgen.scalar<si4>, !kgen.scalar<f32>
  kgen.return %2, %3 : !kgen.scalar<si4>, !kgen.scalar<f32>
}
}

// -----

// CHECK-LABEL: @rem
kgen.func @rem() -> (!kgen.scalar<si4>, !kgen.scalar<ui4>, !kgen.scalar<f32>, !kgen.scalar<f64>) {
  // CHECK-DAG: <si4> = <1
  // CHECK-DAG: <ui4> = <7>
  // CHECK-DAG: <"0.5">
  // CHECK-DAG: <"1.140{{.*}}">
  %0 = kgen.param.constant: scalar<si4> = <7>
  %1 = kgen.param.constant: scalar<si4> = <-2>
  %2 = kgen.param.constant: scalar<ui4> = <7>
  %3 = kgen.param.constant: scalar<ui4> = <-2>
  %4 = kgen.param.constant: scalar<f32> = <"2.5">
  %5 = kgen.param.constant: scalar<f32> = <"2">
  %6 = pop.rem %0, %1 : !kgen.scalar<si4>
  %7 = pop.rem %2, %3 : !kgen.scalar<ui4>
  %8 = pop.rem %4, %5 : !kgen.scalar<f32>
  %9 = kgen.param.constant: scalar<f64> = <"3.14">
  %10 = kgen.param.constant: scalar<f64> = <"2.0">
  %11 = pop.rem %9, %10 : !kgen.scalar<f64>
  kgen.return %6, %7, %8, %11 : !kgen.scalar<si4>, !kgen.scalar<ui4>, !kgen.scalar<f32>, !kgen.scalar<f64>
}

// CHECK-LABEL: @max
kgen.func @max() -> (!kgen.scalar<si4>, !kgen.scalar<f32>, !kgen.scalar<f32>) {
  // CHECK-DAG: <-1>
  // CHECK-DAG: <"2">
  // CHECK-DAG: <"1.25">
  %0 = kgen.param.constant: scalar<si4> = <-2>
  %1 = kgen.param.constant: scalar<si4> = <-1>
  %2 = kgen.param.constant: scalar<f32> = <"1.25">
  %3 = kgen.param.constant: scalar<f32> = <"2">
  %4 = kgen.param.constant: scalar<f32> = <"NaN">
  %5 = pop.max %0, %1 : !kgen.scalar<si4>
  %6 = pop.max %2, %3 : !kgen.scalar<f32>
  %7 = pop.max %2, %4 : !kgen.scalar<f32>
  kgen.return %5, %6, %7 : !kgen.scalar<si4>, !kgen.scalar<f32>, !kgen.scalar<f32>
}

// CHECK-LABEL: @min
kgen.func @min() -> (!kgen.scalar<ui4>, !kgen.scalar<f32>, !kgen.scalar<f32>) {
  // CHECK-DAG: <0>
  // CHECK-DAG: <"-2">
  // CHECK-DAG: <"1.25">
  %0 = kgen.param.constant: scalar<ui4> = <0>
  %1 = kgen.param.constant: scalar<ui4> = <-1>
  %2 = kgen.param.constant: scalar<f32> = <"1.25">
  %3 = kgen.param.constant: scalar<f32> = <"-2">
  %4 = kgen.param.constant: scalar<f32> = <"NaN">
  %5 = pop.min %0, %1 : !kgen.scalar<ui4>
  %6 = pop.min %2, %3 : !kgen.scalar<f32>
  %7 = pop.min %2, %4 : !kgen.scalar<f32>
  kgen.return %5, %6, %7 : !kgen.scalar<ui4>, !kgen.scalar<f32>, !kgen.scalar<f32>
}

// CHECK-LABEL: @min_max_identity
kgen.func @min_max_identity(%arg0 : !kgen.scalar<ui4>, %arg1 : !kgen.scalar<ui4>) ->
 (!kgen.scalar<ui4>, !kgen.scalar<ui4>, !kgen.scalar<ui4>, !kgen.scalar<ui4>) {
  // Confirm that folding does not erroneously act on min(x,y) and max(x,y)
  %0 = pop.min %arg0, %arg1 : !kgen.scalar<ui4>
  %1 = pop.max %arg0, %arg1 : !kgen.scalar<ui4>
  // min(x,x) -> x
  %2 = pop.min %arg0, %arg0 : !kgen.scalar<ui4>
  // max(x,x) -> x
  %3 = pop.max %arg0, %arg0 : !kgen.scalar<ui4>
  // CHECK: return %0, %1, %arg0, %arg0
  kgen.return %0, %1, %2, %3 : !kgen.scalar<ui4>, !kgen.scalar<ui4>, !kgen.scalar<ui4>, !kgen.scalar<ui4>
}

// CHECK-LABEL: @shl
kgen.func @shl() -> !kgen.scalar<ui4> {
  // CHECK-NEXT: <12>
  %0 = kgen.param.constant: scalar<ui4> = <6>
  %1 = kgen.param.constant: scalar<ui4> = <1>
  %2 = pop.shl %0, %1 : !kgen.scalar<ui4>

  kgen.return %2 : !kgen.scalar<ui4>
}

// CHECK-LABEL: @shr
kgen.func @shr() -> (!kgen.scalar<ui4>, !kgen.scalar<si4>) {
  // CHECK-DAG: <3>
  // CHECK-DAG: <-4>
  %0 = kgen.param.constant: scalar<ui4> = <7>
  %1 = kgen.param.constant: scalar<ui4> = <1>
  %2 = kgen.param.constant: scalar<si4> = <-7>
  %3 = kgen.param.constant: scalar<si4> = <1>
  %4 = pop.shr %0, %1 : !kgen.scalar<ui4>
  %5 = pop.shr %2, %3 : !kgen.scalar<si4>
  kgen.return %4, %5 : !kgen.scalar<ui4>, !kgen.scalar<si4>
}

// CHECK-LABEL: @fma
kgen.func @fma() -> (!kgen.scalar<si8>, !kgen.scalar<f32>) {
  // CHECK-DAG: <6>
  // CHECK-DAG: <"8.75">
  %0 = kgen.param.constant: scalar<si8> = <2>
  %1 = kgen.param.constant: scalar<f32> = <"2.5">
  %2 = pop.fma %0, %0, %0 : !kgen.scalar<si8>
  %3 = pop.fma %1, %1, %1 : !kgen.scalar<f32>
  kgen.return %2, %3 : !kgen.scalar<si8>, !kgen.scalar<f32>
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="e-p:64:64", simd_bit_width = 128, index_bit_width = 64>} {
// CHECK-LABEL: @cmp_eq
kgen.func @cmp_eq() -> (!kgen.simd<2, bool>, !kgen.scalar<bool>) {
  // CHECK-DAG: <false, true>
  // CHECK-DAG: <false>
  %0 = kgen.param.constant: simd<2, si8> = <<1, 2>>
  %1 = kgen.param.constant: simd<2, si8> = <<-2, 2>>
  %2 = pop.cmp eq(%0, %1) : !kgen.simd<2, si8>
  %3 = kgen.param.constant: scalar<f32> = <"1">
  %4 = kgen.param.constant: scalar<f32> = <"2">
  %5 = pop.cmp eq(%3, %4) : !kgen.scalar<f32>
  kgen.return %2, %5 : !kgen.simd<2, bool>, !kgen.scalar<bool>
}

// CHECK-LABEL: @cmp_ne
kgen.func @cmp_ne() -> (!kgen.simd<2, bool>, !kgen.scalar<bool>) {
  // CHECK-DAG: <true, false>
  // CHECK-DAG: <true>
  %0 = kgen.param.constant: simd<2, si8> = <<1, 2>>
  %1 = kgen.param.constant: simd<2, si8> = <<-2, 2>>
  %2 = pop.cmp ne(%0, %1) : !kgen.simd<2, si8>
  %3 = kgen.param.constant: scalar<f32> = <"1">
  %4 = kgen.param.constant: scalar<f32> = <"2">
  %5 = pop.cmp ne(%3, %4) : !kgen.scalar<f32>
  kgen.return %2, %5 : !kgen.simd<2, bool>, !kgen.scalar<bool>
}

// CHECK-LABEL: @cmp_lt
kgen.func @cmp_lt() -> (!kgen.simd<2, bool>, !kgen.scalar<bool>) {
  // CHECK-DAG: <false>
  // CHECK-DAG: <true>
  %0 = kgen.param.constant: simd<2, si8> = <<1, 2>>
  %1 = kgen.param.constant: simd<2, si8> = <<-2, 2>>
  %2 = pop.cmp lt(%0, %1) : !kgen.simd<2, si8>
  %3 = kgen.param.constant: scalar<f32> = <"1">
  %4 = kgen.param.constant: scalar<f32> = <"2">
  %5 = pop.cmp lt(%3, %4) : !kgen.scalar<f32>
  kgen.return %2, %5 : !kgen.simd<2, bool>, !kgen.scalar<bool>
}

// CHECK-LABEL: @cmp_gt
kgen.func @cmp_gt() -> (!kgen.simd<2, bool>, !kgen.scalar<bool>) {
  // CHECK-DAG: <true, false>
  // CHECK-DAG: <false>
  %0 = kgen.param.constant: simd<2, si8> = <<1, 2>>
  %1 = kgen.param.constant: simd<2, si8> = <<-2, 2>>
  %2 = pop.cmp gt(%0, %1) : !kgen.simd<2, si8>
  %3 = kgen.param.constant: scalar<f32> = <"1">
  %4 = kgen.param.constant: scalar<f32> = <"2">
  %5 = pop.cmp gt(%3, %4) : !kgen.scalar<f32>
  kgen.return %2, %5 : !kgen.simd<2, bool>, !kgen.scalar<bool>
}

// CHECK-LABEL: @cmp_le
kgen.func @cmp_le() -> (!kgen.simd<2, bool>, !kgen.scalar<bool>) {
  // CHECK-DAG: <false, true>
  // CHECK-DAG: <true>
  %0 = kgen.param.constant: simd<2, si8> = <<1, 2>>
  %1 = kgen.param.constant: simd<2, si8> = <<-2, 2>>
  %2 = pop.cmp le(%0, %1) : !kgen.simd<2, si8>
  %3 = kgen.param.constant: scalar<f32> = <"1">
  %4 = kgen.param.constant: scalar<f32> = <"2">
  %5 = pop.cmp le(%3, %4) : !kgen.scalar<f32>
  kgen.return %2, %5 : !kgen.simd<2, bool>, !kgen.scalar<bool>
}

// CHECK-LABEL: @cmp_ge
kgen.func @cmp_ge() -> (!kgen.simd<2, bool>, !kgen.scalar<bool>) {
  // CHECK-DAG: <true>
  // CHECK-DAG: <false>
  %0 = kgen.param.constant: simd<2, si8> = <<1, 2>>
  %1 = kgen.param.constant: simd<2, si8> = <<-2, 2>>
  %2 = pop.cmp ge(%0, %1) : !kgen.simd<2, si8>
  %3 = kgen.param.constant: scalar<f32> = <"1">
  %4 = kgen.param.constant: scalar<f32> = <"2">
  %5 = pop.cmp ge(%3, %4) : !kgen.scalar<f32>
  kgen.return %2, %5 : !kgen.simd<2, bool>, !kgen.scalar<bool>
}

// CHECK-LABEL: @cmp_index
kgen.func @cmp_index() -> !kgen.scalar<bool> {
  // CHECK: <false>
  %0 = kgen.param.constant: scalar<index> = <4294967296>
  %1 = kgen.param.constant: scalar<index> = <8589934592>
  %2 = pop.cmp eq(%0, %1) : !kgen.scalar<index>
  kgen.return %2 : !kgen.scalar<bool>
}

// CHECK-LABEL: @cmp_eq_self
kgen.func @cmp_eq_self(%simd: !kgen.simd<2, si8>) -> !kgen.simd<2, bool> {
  // CHECK-DAG: <true>
  %0 = pop.cmp eq(%simd, %simd) : !kgen.simd<2, si8>
  kgen.return %0 : !kgen.simd<2, bool>
}

// Floats cannot be removed symbolically as they could be NAN.
// CHECK-LABEL: @cmp_eq_self_float
kgen.func @cmp_eq_self_float(%simd: !kgen.simd<2, f32>) -> !kgen.simd<2, bool> {
  // CHECK-NEXT: %[[RES:.*]] = pop.cmp eq
  // CHECK-NEXT: kgen.return %[[RES]]
  %0 = pop.cmp eq(%simd, %simd) : !kgen.simd<2, f32>
  kgen.return %0 : !kgen.simd<2, bool>
}

// CHECK-LABEL: @cmp_ne_self
kgen.func @cmp_ne_self(%simd: !kgen.simd<2, si8>) -> !kgen.simd<2, bool> {
  // CHECK-DAG: <false>
  %0 = pop.cmp ne(%simd, %simd) : !kgen.simd<2, si8>
  kgen.return %0 : !kgen.simd<2, bool>
}

// CHECK-LABEL: @cmp_lt_self
kgen.func @cmp_lt_self(%simd: !kgen.simd<2, si8>) -> !kgen.simd<2, bool> {
  // CHECK-DAG: <false>
  %0 = pop.cmp lt(%simd, %simd) : !kgen.simd<2, si8>
  kgen.return %0 : !kgen.simd<2, bool>
}

// CHECK-LABEL: @cmp_gt_self
kgen.func @cmp_gt_self(%simd: !kgen.simd<2, si8>) -> !kgen.simd<2, bool> {
  // CHECK-DAG: <false>
  %0 = pop.cmp gt(%simd, %simd) : !kgen.simd<2, si8>
  kgen.return %0 : !kgen.simd<2, bool>
}

// CHECK-LABEL: @cmp_ge_self
kgen.func @cmp_ge_self(%simd: !kgen.simd<2, si8>) -> !kgen.simd<2, bool> {
  // CHECK-DAG: <true>
  %0 = pop.cmp ge(%simd, %simd) : !kgen.simd<2, si8>
  kgen.return %0 : !kgen.simd<2, bool>
}

// CHECK-LABEL: @cmp_le_self
kgen.func @cmp_le_self(%simd: !kgen.simd<2, si8>) -> !kgen.simd<2, bool> {
  // CHECK-DAG: <true>
  %0 = pop.cmp le(%simd, %simd) : !kgen.simd<2, si8>
  kgen.return %0 : !kgen.simd<2, bool>
}

// CHECK-LABEL: @cmp_true_false
kgen.func @cmp_true_false(%simd: !kgen.simd<2, bool>) -> (!kgen.simd<2, bool>, !kgen.simd<2, bool>, !kgen.simd<2, bool>, !kgen.simd<2, bool>) {
  %true = kgen.param.constant: simd<2, bool> = <<true, true>>
  %false = kgen.param.constant: simd<2, bool> = <<false, false>>
  %0 = pop.cmp eq(%true, %simd) : <2, bool>
  %1 = pop.cmp ne(%simd, %false) : <2, bool>
  %2 = pop.cmp eq(%simd, %true) : <2, bool>
  %3 = pop.cmp ne(%false, %simd) : <2, bool>
  // CHECK-NEXT: return %arg0, %arg0, %arg0, %arg0
  kgen.return %0, %1, %2, %3 : !kgen.simd<2, bool>, !kgen.simd<2, bool>, !kgen.simd<2, bool>, !kgen.simd<2, bool>
}

// CHECK-LABEL: @cmp_unsigned
kgen.func @cmp_unsigned(%simd: !kgen.simd<2, ui8>) -> (
    !kgen.simd<2, bool>, !kgen.simd<2, bool>,
    !kgen.simd<2, bool>, !kgen.simd<2, bool>,
    !kgen.simd<2, bool>, !kgen.simd<2, bool>,
    !kgen.simd<2, bool>, !kgen.simd<2, bool>
) {
  %zero = kgen.param.constant: simd<2, ui8> = <<0, 0>>

  // CHECK: %[[ZERO:.*]] = kgen.param.constant: simd<2, ui8> = <0>
  // CHECK: %[[TRUE:.*]] = kgen.param.constant: simd<2, bool> = <true>
  // CHECK: %[[FALSE:.*]] = kgen.param.constant: simd<2, bool> = <false>
  %0 = pop.cmp ge(%simd, %zero) : <2, ui8>
  %2 = pop.cmp gt(%zero, %simd) : <2, ui8>
  %5 = pop.cmp le(%zero, %simd) : <2, ui8>
  %7 = pop.cmp lt(%simd, %zero) : <2, ui8>

  // CHECK: %[[UNOPTIMIZED_GE:.*]] = pop.cmp ge(%[[ZERO]], %arg0)
  %1 = pop.cmp ge(%zero, %simd) : <2, ui8>
  // CHECK: %[[UNOPTIMIZED_GT:.*]] = pop.cmp gt(%arg0, %[[ZERO]])
  %3 = pop.cmp gt(%simd, %zero) : <2, ui8>
  // CHECK: %[[UNOPTIMIZED_LE:.*]] = pop.cmp le(%arg0, %[[ZERO]])
  %4 = pop.cmp le(%simd, %zero) : <2, ui8>
  // CHECK: %[[UNOPTIMIZED_LT:.*]] = pop.cmp lt(%[[ZERO]], %arg0)
  %6 = pop.cmp lt(%zero, %simd) : <2, ui8>

  // CHECK-NEXT: return %[[TRUE]], %[[UNOPTIMIZED_GE]], %[[FALSE]], %[[UNOPTIMIZED_GT]], %[[UNOPTIMIZED_LE]], %[[TRUE]], %[[UNOPTIMIZED_LT]], %[[FALSE]]
  kgen.return %0, %1, %2, %3,
              %4, %5, %6, %7 : !kgen.simd<2, bool>, !kgen.simd<2, bool>,
                               !kgen.simd<2, bool>, !kgen.simd<2, bool>,
                               !kgen.simd<2, bool>, !kgen.simd<2, bool>,
                               !kgen.simd<2, bool>, !kgen.simd<2, bool>
}

// CHECK-LABEL: @and
kgen.func @and() -> !kgen.scalar<ui4> {
  // CHECK-NEXT <1>
  %0 = kgen.param.constant: scalar<ui4> = <7>
  %1 = kgen.param.constant: scalar<ui4> = <9>
  %2 = pop.simd.and %0, %1 : !kgen.scalar<ui4>
  kgen.return %2 : !kgen.scalar<ui4>
}

// CHECK-LABEL: @or
kgen.func @or() -> !kgen.scalar<ui4> {
  // CHECK-NEXT <15>
  %0 = kgen.param.constant: scalar<ui4> = <6>
  %1 = kgen.param.constant: scalar<ui4> = <9>
  %2 = pop.simd.or %0, %1 : !kgen.scalar<ui4>
  kgen.return %2 : !kgen.scalar<ui4>
}

// CHECK-LABEL: @xor
kgen.func @xor() -> !kgen.scalar<ui4> {
  // CHECK-NEXT <2>
  %0 = kgen.param.constant: scalar<ui4> = <5>
  %1 = kgen.param.constant: scalar<ui4> = <7>
  %2 = pop.simd.xor %0, %1 : !kgen.scalar<ui4>
  kgen.return %2 : !kgen.scalar<ui4>
}

// CHECK-LABEL: @xor_zero
kgen.func @xor_zero(%arg0: !kgen.simd<2, ui16>) -> !kgen.simd<2, ui16> {
  %0 = kgen.param.constant: simd<2, ui16> = <0>
  %1 = pop.simd.xor %0, %arg0 : !kgen.simd<2, ui16>
  // CHECK-NEXT: return %arg0
  kgen.return %1 : !kgen.simd<2, ui16>
}

// CHECK-LABEL: @not_not
kgen.func @not_not(%arg0: !kgen.scalar<bool>) ->!kgen.scalar<bool>{
  %0 = kgen.param.constant: scalar<bool> = <true>
  %1 = pop.simd.xor %arg0, %0 : !kgen.scalar<bool>
  %2 = pop.simd.xor %1, %0 : !kgen.scalar<bool>
  // CHECK-NEXT: return %arg0
  kgen.return %2 : !kgen.scalar<bool>
}

// CHECK-LABEL: @not_not_const
kgen.func @not_not_const() ->!kgen.scalar<bool>{
  %arg0 = kgen.param.constant: scalar<bool> = <false>
  %0 = kgen.param.constant: scalar<bool> = <true>
  %1 = pop.simd.xor %arg0, %0 : !kgen.scalar<bool>
  %2 = pop.simd.xor %1, %0 : !kgen.scalar<bool>
  // CHECK-NEXT: %simd = kgen.param.constant: scalar<bool> = <false>
  // CHECK-NEXT: return %simd
  kgen.return %2 : !kgen.scalar<bool>
}

// CHECK-LABEL: @bool_and
kgen.func @bool_and() -> i1 {
  // CHECK-NEXT: <1>
  %0 = kgen.param.constant: i1 = <1>
  %1 = kgen.param.constant: i1 = <1>
  // CHECK-NOT: pop.add
  %2 = pop.and %0, %1
  kgen.return %2 : i1
}

// CHECK-LABEL: @bool_and_true
kgen.func @bool_and_true(%arg0: i1) -> i1 {
  %0 = kgen.param.constant: i1 = <1>
  %1 = pop.and %arg0, %0
  // CHECK-NEXT: kgen.return %arg0
  kgen.return %1 : i1
}

// CHECK-LABEL: @bool_and_false
kgen.func @bool_and_false(%arg0: i1) -> i1 {
  // CHECK-NEXT: <0>
  %0 = kgen.param.constant: i1 = <0>
  // CHECK-NOT: pop.add
  %1 = pop.and %arg0, %0
  kgen.return %1 : i1
}

// CHECK-LABEL: @bool_or
kgen.func @bool_or() -> i1 {
  // CHECK-NEXT: <0>
  %0 = kgen.param.constant: i1 = <0>
  %1 = kgen.param.constant: i1 = <0>
  // CHECK-NOT: pop.or
  %2 = pop.or %0, %1
  kgen.return %2 : i1
}

// CHECK-LABEL: @bool_or_true
kgen.func @bool_or_true(%arg0: i1) -> i1 {
  // CHECK-NEXT: <1>
  %0 = kgen.param.constant: i1 = <1>
  // CHECK-NOT: pop.or
  %1 = pop.or %arg0, %0
  kgen.return %1 : i1
}

// CHECK-LABEL: @bool_or_false
kgen.func @bool_or_false(%arg0: i1) -> i1 {
  %0 = kgen.param.constant: i1 = <0>
  %1 = pop.or %arg0, %0
  // CHECK-NEXT: kgen.return %arg0
  kgen.return %1 : i1
}

// CHECK-LABEL: @bool_xor
kgen.func @bool_xor() -> i1 {
  // CHECK-NEXT: <1>
  %0 = kgen.param.constant: i1 = <1>
  %1 = kgen.param.constant: i1 = <0>
  // CHECK-NOT: pop.xor
  %2 = pop.xor %0, %1
  kgen.return %2 : i1
}

// CHECK-LABEL: @bool_xor_false
kgen.func @bool_xor_false(%arg0: i1) -> i1 {
  %0 = kgen.param.constant: i1 = <0>
  %1 = pop.xor %0, %arg0
  // CHECK-NEXT: return %arg0
  kgen.return %1 : i1
}

// CHECK-LABEL: @bool_not_not
kgen.func @bool_not_not(%arg0: i1) ->i1{
  %0 = kgen.param.constant: i1 = <1>
  %1 = pop.xor %arg0, %0
  %2 = pop.xor %1, %0
  // CHECK-NEXT: return %arg0
  kgen.return %2 : i1
}

// CHECK-LABEL: @mask_ones
kgen.func @mask_ones(%arg0: !kgen.simd<2, ui4>) -> !kgen.simd<2, ui4> {
  %0 = kgen.param.constant: simd<2, ui4> = <15>
  %1 = kgen.param.constant: simd<2, ui4> = <15>
  %2 = pop.simd.xor %arg0, %0 : !kgen.simd<2, ui4>
  %3 = pop.simd.xor %2, %1 : !kgen.simd<2, ui4>
  // CHECK-NEXT: return %arg0
  kgen.return %3 : !kgen.simd<2, ui4>
}

// CHECK-LABEL: @simd_select
kgen.func @simd_select() -> !kgen.simd<2, si4> {
  // CHECK-NEXT: <1, 4>
  %0 = kgen.param.constant: simd<2, si4> = <<1, 3>>
  %1 = kgen.param.constant: simd<2, si4> = <<2, 4>>
  %2 = kgen.param.constant: simd<2, bool> = <<true, false>>
  %3 = pop.simd.select %2, %0, %1 : !kgen.simd<2, si4>
  kgen.return %3 : !kgen.simd<2, si4>
}

// CHECK-LABEL: @simd_select_true_false
kgen.func @simd_select_true_false(%arg0: !kgen.simd<2, bool>) -> !kgen.simd<2, bool> {
  // CHECK-NEXT: return %arg0
  %true = kgen.param.constant: simd<2, bool> = <true>
  %false = kgen.param.constant: simd<2, bool> = <false>
  %0 = pop.simd.select %arg0, %true, %false : <2, bool>
  kgen.return %0 : !kgen.simd<2, bool>
}

// CHECK-LABEL: @simd_select_false_true
kgen.func @simd_select_false_true(%arg0: !kgen.simd<2, bool>) -> !kgen.simd<2, bool> {
  // CHECK-NEXT: %[[TRUE:.*]] = kgen.param.constant: simd<2, bool> = <true>
  // CHECK-NEXT: %0 = pop.simd.xor %arg0, %[[TRUE]]
  // CHECK-NEXT: return %0
  %true = kgen.param.constant: simd<2, bool> = <<true, true>>
  %false = kgen.param.constant: simd<2, bool> = <<false, false>>
  %0 = pop.simd.select %arg0, %false, %true : <2, bool>
  kgen.return %0 : !kgen.simd<2, bool>
}

// CHECK-LABEL: @simd_select_equal
kgen.func @simd_select_equal(%arg0: !kgen.simd<2, bool>, %arg1: !kgen.simd<2, bool>) -> !kgen.simd<2, bool> {
  %0 = pop.simd.select %arg0, %arg1, %arg1 : <2, bool>
  // CHECK-NEXT: return %arg1
  kgen.return %0 : !kgen.simd<2, bool>
}

// CHECK-LABEL: @simd_select_all_true
kgen.func @simd_select_all_true(%arg0: !kgen.simd<2, f32>, %arg1: !kgen.simd<2, f32>) -> !kgen.simd<2, f32> {
  // CHECK: (%[[ARG0:.*]]: !kgen.simd<2, f32>, %[[ARG1:.*]]: !kgen.simd<2, f32>)
  // CHECK-NEXT: kgen.return %[[ARG0]]

  %true = kgen.param.constant: simd<2, bool> = <<true, true>>
  %0 = pop.simd.select %true, %arg0, %arg1 : <2, f32>
  kgen.return %0 : !kgen.simd<2, f32>
}

// CHECK-LABEL: @simd_select_all_false
kgen.func @simd_select_all_false(%arg0: !kgen.simd<2, f32>, %arg1: !kgen.simd<2, f32>) -> !kgen.simd<2, f32> {
  // CHECK: (%[[ARG0:.*]]: !kgen.simd<2, f32>, %[[ARG1:.*]]: !kgen.simd<2, f32>)
  // CHECK-NEXT: kgen.return %[[ARG1]]

  %true = kgen.param.constant: simd<2, bool> = <<false, false>>
  %0 = pop.simd.select %true, %arg0, %arg1 : <2, f32>
  kgen.return %0 : !kgen.simd<2, f32>
}

// CHECK-LABEL: @bitcast
kgen.func @bitcast() -> (!kgen.simd<2, bf16>, !kgen.simd<2, f16>) {
  // CHECK-DAG: <"0.125", "8">
  // CHECK-DAG: <"5.9605E-8", "1.1921E-7">
  %0 = kgen.param.constant: simd<2, si16> = <<1, 2>>
  %1 = kgen.param.constant: simd<2, f16> = <<"1.5", "2.5">>
  %2 = pop.bitcast %0 : !kgen.simd<2, si16> to !kgen.simd<2, f16>
  %3 = pop.bitcast %1 : !kgen.simd<2, f16> to !kgen.simd<2, bf16>
  kgen.return %3, %2 : !kgen.simd<2, bf16>, !kgen.simd<2, f16>
}

// CHECK-LABEL: @bitcast_size_change
kgen.func @bitcast_size_change() -> (!kgen.simd<4, si16>) {
  // CHECK: <<1, 0, 2, 0>>
  %0 = kgen.param.constant: simd<2, si32> = <<1, 2>>
  %1 = pop.bitcast %0 : !kgen.simd<2, si32> to !kgen.simd<4, si16>
  kgen.return %1 : !kgen.simd<4, si16>
}

// CHECK-LABEL: @bitcast_index
kgen.func @bitcast_index() -> (!kgen.simd<2, ui64>, !kgen.simd<2, index>,
                               !kgen.simd<2, index>, !kgen.simd<2, f64>) {
  %ui64_const = kgen.param.constant: simd<2, ui64> = <<1, 2>>
  %f64_const = kgen.param.constant: simd<2, f64> = <<"1.5", "2.5">>
  %index_const = kgen.param.constant: simd<2, index> = <<3, 4>>

  %2 = pop.bitcast %index_const : !kgen.simd<2, index> to !kgen.simd<2, ui64>
  %3 = pop.bitcast %ui64_const : !kgen.simd<2, ui64> to !kgen.simd<2, index>
  %4 = pop.bitcast %f64_const : !kgen.simd<2, f64> to !kgen.simd<2, index>
  %5 = pop.bitcast %index_const : !kgen.simd<2, index> to !kgen.simd<2, f64>
  // CHECK: [[SIMD_UI64:%.*]] = kgen.param.constant: simd<2, ui64> = <<3, 4>>
  // CHECK-NEXT: [[SIMD_INDEX:%.*]] = kgen.param.constant: simd<2, index> = <<1, 2>>
  // CHECK-NEXT: [[SIMD_INDEX_F64:%.*]] = kgen.param.constant: simd<2, index> = <<4609434218613702656, 4612811918334230528>>
  // CHECK-NEXT: [[SIMD_F64:%.*]] = kgen.param.constant: simd<2, f64> = <<"1.4821969375237396E-323", "1.9762625833649862E-323">>
  // CHECK: kgen.return [[SIMD_UI64]], [[SIMD_INDEX]], [[SIMD_INDEX_F64]], [[SIMD_F64]]
  kgen.return %2, %3, %4, %5 : !kgen.simd<2, ui64>, !kgen.simd<2, index>,
                               !kgen.simd<2, index>, !kgen.simd<2, f64>
}

// CHECK-LABEL: @bitcast_bool_nonfoldable
kgen.func @bitcast_bool_nonfoldable(%arg0: !kgen.scalar<bool>) -> (!kgen.scalar<ui1>) {
  // Non-constant bool operand: bitcast should not be folded.
  // CHECK-NEXT: pop.bitcast %arg0 : !kgen.scalar<bool> to !kgen.scalar<ui1>
  %0 = pop.bitcast %arg0 : !kgen.scalar<bool> to !kgen.scalar<ui1>
  kgen.return %0 : !kgen.scalar<ui1>
}

// CHECK-LABEL: @bitcast_from_bool
kgen.func @bitcast_from_bool() -> (!kgen.scalar<ui1>) {
  %true = kgen.param.constant: scalar<bool> = <true>
  // CHECK: kgen.param.constant: scalar<ui1> = <1>
  %0 = pop.bitcast %true : !kgen.scalar<bool> to !kgen.scalar<ui1>
  kgen.return %0 : !kgen.scalar<ui1>
}

// CHECK-LABEL: @bitcast_bool_to_ui1_false
kgen.func @bitcast_bool_to_ui1_false() -> (!kgen.scalar<ui1>) {
  %false = kgen.param.constant: scalar<bool> = <false>
  // CHECK: kgen.param.constant: scalar<ui1> = <0>
  %0 = pop.bitcast %false : !kgen.scalar<bool> to !kgen.scalar<ui1>
  kgen.return %0 : !kgen.scalar<ui1>
}

// CHECK-LABEL: @bitcast_ui1_to_bool
kgen.func @bitcast_ui1_to_bool() -> (!kgen.scalar<bool>) {
  %one = kgen.param.constant: scalar<ui1> = <1>
  // CHECK: kgen.param.constant: scalar<bool> = <true>
  %0 = pop.bitcast %one : !kgen.scalar<ui1> to !kgen.scalar<bool>
  kgen.return %0 : !kgen.scalar<bool>
}

// CHECK-LABEL: @bitcast_pack_bools_to_ui8
kgen.func @bitcast_pack_bools_to_ui8() -> (!kgen.scalar<ui8>) {
  // 0b01010111 = 0x57 = 87
  %bools = kgen.param.constant: simd<8, bool> = <<true, true, true, false, true, false, true, false>>
  // CHECK: kgen.param.constant: scalar<ui8> = <87>
  %0 = pop.bitcast %bools : !kgen.simd<8, bool> to !kgen.scalar<ui8>
  kgen.return %0 : !kgen.scalar<ui8>
}

// CHECK-LABEL: @bitcast_unpack_ui8_to_bools
kgen.func @bitcast_unpack_ui8_to_bools() -> (!kgen.simd<8, bool>) {
  // 87 = 0x57 = 0b01010111
  %byte = kgen.param.constant: scalar<ui8> = <87>
  // CHECK: kgen.param.constant: simd<8, bool> = <<true, true, true, false, true, false, true, false>>
  %0 = pop.bitcast %byte : !kgen.scalar<ui8> to !kgen.simd<8, bool>
  kgen.return %0 : !kgen.simd<8, bool>
}

// CHECK-LABEL: @bitcast_all_false_bools
kgen.func @bitcast_all_false_bools() -> (!kgen.scalar<ui8>) {
  %bools = kgen.param.constant: simd<8, bool> = <<false, false, false, false, false, false, false, false>>
  // CHECK: kgen.param.constant: scalar<ui8> = <0>
  %0 = pop.bitcast %bools : !kgen.simd<8, bool> to !kgen.scalar<ui8>
  kgen.return %0 : !kgen.scalar<ui8>
}

// CHECK-LABEL: @bitcast_all_true_bools
kgen.func @bitcast_all_true_bools() -> (!kgen.scalar<ui8>) {
  %bools = kgen.param.constant: simd<8, bool> = <<true, true, true, true, true, true, true, true>>
  // CHECK: kgen.param.constant: scalar<ui8> = <255>
  %0 = pop.bitcast %bools : !kgen.simd<8, bool> to !kgen.scalar<ui8>
  kgen.return %0 : !kgen.scalar<ui8>
}

// CHECK-LABEL: @bitcast_pack_bools_to_simd
kgen.func @bitcast_pack_bools_to_simd() -> (!kgen.simd<2, ui8>) {
  // First 8 bools → 87 (0x57), next 8 bools → 170 (0xAA)
  %bools = kgen.param.constant: simd<16, bool> = <<true, true, true, false, true, false, true, false, false, true, false, true, false, true, false, true>>
  // CHECK: kgen.param.constant: simd<2, ui8> = <<87, 170>>
  %0 = pop.bitcast %bools : !kgen.simd<16, bool> to !kgen.simd<2, ui8>
  kgen.return %0 : !kgen.simd<2, ui8>
}

// CHECK-LABEL: @bitcast_unpack_simd_to_bools
kgen.func @bitcast_unpack_simd_to_bools() -> (!kgen.simd<16, bool>) {
  %bytes = kgen.param.constant: simd<2, ui8> = <<87, 170>>
  // CHECK: kgen.param.constant: simd<16, bool> = <<true, true, true, false, true, false, true, false, false, true, false, true, false, true, false, true>>
  %0 = pop.bitcast %bytes : !kgen.simd<2, ui8> to !kgen.simd<16, bool>
  kgen.return %0 : !kgen.simd<16, bool>
}

// CHECK-LABEL: @pointer_bitcast
kgen.func @pointer_bitcast() -> !kgen.pointer<si32> {
  // CHECK-NEXT: pointer<si32> = <0>
  %0 = kgen.param.constant: pointer<si64> = <0>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<si64> to !kgen.pointer<si32>
  kgen.return %1 : !kgen.pointer<si32>
}

// CHECK-LABEL: @pointer_bitcast_of_bitcast
kgen.func @pointer_bitcast_of_bitcast(%arg0: !kgen.pointer<si32>) -> !kgen.pointer<f32> {
  // CHECK-NEXT: %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<si32> to !kgen.pointer<f32>
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<si32> to !kgen.pointer<f64>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<f64> to !kgen.pointer<f32>
  // CHECK-NEXT: return %0
  kgen.return %1 : !kgen.pointer<f32>
}

// A bitcast from a pointer-to-array to a pointer-to-element is the address of
// element 0, so it is rewritten to an explicit `pop.array.gep %p[0]` (lets
// SROA see element-wise access instead of an opaque cast).
// CHECK-LABEL: @pointer_bitcast_array_to_elem0
kgen.func @pointer_bitcast_array_to_elem0(%arg0: !kgen.pointer<array<4, si32>>) -> !kgen.pointer<si32> {
  // CHECK: %[[Z:.+]] = kgen.param.constant = <0>
  // CHECK: %[[G:.+]] = pop.array.gep %arg0[%[[Z]]] : <array<4, si32>>
  // CHECK: return %[[G]]
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<array<4, si32>> to !kgen.pointer<si32>
  kgen.return %0 : !kgen.pointer<si32>
}

// Element type differs from the array element type: this is a genuine
// reinterpret, not element-0 access, so it must stay a bitcast.
// CHECK-LABEL: @pointer_bitcast_array_reinterpret
kgen.func @pointer_bitcast_array_reinterpret(%arg0: !kgen.pointer<array<4, si32>>) -> !kgen.pointer<f32> {
  // CHECK: pop.pointer.bitcast %arg0
  // CHECK-NOT: pop.array.gep
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<array<4, si32>> to !kgen.pointer<f32>
  kgen.return %0 : !kgen.pointer<f32>
}

// Bitcast that also changes address space is a genuine reinterpret (the
// resulting element pointer could not be expressed by `pop.array.gep`, whose
// result is always the default address space), so it must stay a bitcast.
// CHECK-LABEL: @pointer_bitcast_array_addrspace
kgen.func @pointer_bitcast_array_addrspace(%arg0: !kgen.pointer<array<4, si32>>) -> !kgen.pointer<si32, 5> {
  // CHECK: pop.pointer.bitcast %arg0
  // CHECK-NOT: pop.array.gep
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<array<4, si32>> to !kgen.pointer<si32, 5>
  kgen.return %0 : !kgen.pointer<si32, 5>
}

// A non-default address space *on the source array* still canonicalizes:
// `pop.array.gep` always yields a default-address-space element pointer, which
// is exactly the element-0 pointer type here, so the rewrite is valid.
// CHECK-LABEL: @pointer_bitcast_array_src_addrspace
kgen.func @pointer_bitcast_array_src_addrspace(%arg0: !kgen.pointer<array<4, si32>, 5>) -> !kgen.pointer<si32> {
  // CHECK: %[[Z:.+]] = kgen.param.constant = <0>
  // CHECK: %[[G:.+]] = pop.array.gep %arg0[%[[Z]]] : <array<4, si32>, 5>
  // CHECK: return %[[G]]
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<array<4, si32>, 5> to !kgen.pointer<si32>
  kgen.return %0 : !kgen.pointer<si32>
}

// Nested array: a bitcast to the (array-typed) element pointer is element 0 of
// the outer array, so it is rewritten to `pop.array.gep %p[0]` as well.
// CHECK-LABEL: @pointer_bitcast_nested_array_to_elem0
kgen.func @pointer_bitcast_nested_array_to_elem0(%arg0: !kgen.pointer<array<4, array<2, si32>>>) -> !kgen.pointer<array<2, si32>> {
  // CHECK: %[[Z:.+]] = kgen.param.constant = <0>
  // CHECK: %[[G:.+]] = pop.array.gep %arg0[%[[Z]]] : <array<4, array<2, si32>>>
  // CHECK: return %[[G]]
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<array<4, array<2, si32>>> to !kgen.pointer<array<2, si32>>
  kgen.return %0 : !kgen.pointer<array<2, si32>>
}

// CHECK-LABEL: @cast_bool
kgen.func @cast_bool() -> (!kgen.scalar<f16>, !kgen.scalar<ui8>, !kgen.scalar<index>) {
  %c = kgen.param.constant: scalar<bool> = <true>
  // CHECK-DAG: %[[C0:.*]] = kgen{{.*}}<"1">
  %0 = pop.cast %c : !kgen.scalar<bool> to !kgen.scalar<f16>
  // CHECK-DAG: %[[C1:.*]] = kgen{{.*}}ui8{{.*}}<1>
  %1 = pop.cast %c : !kgen.scalar<bool> to !kgen.scalar<ui8>
  // CHECK-DAG: %[[C2:.*]] = kgen{{.*}}index{{.*}}<1>
  %2 = pop.cast %c : !kgen.scalar<bool> to !kgen.scalar<index>
  // CHECK-NEXT: return %[[C0]], %[[C1]], %[[C2]]
  kgen.return %0, %1, %2 : !kgen.scalar<f16>, !kgen.scalar<ui8>, !kgen.scalar<index>
}

// CHECK-LABEL: @cast_bf16
kgen.func @cast_bf16() -> (!kgen.scalar<f16>, !kgen.scalar<ui8>, !kgen.scalar<bool>, !kgen.scalar<index>) {
  %c = kgen.param.constant: scalar<bf16> = <"10.125">
  // CHECK-DAG: %[[C0:.*]] = kgen{{.*}}<"10.125">
  %0 = pop.cast %c : !kgen.scalar<bf16> to !kgen.scalar<f16>
  // CHECK-DAG: %[[C1:.*]] = kgen{{.*}}ui8{{.*}}<10>
  %1 = pop.cast %c : !kgen.scalar<bf16> to !kgen.scalar<ui8>
  // CHECK-DAG: %[[C2:.*]] = kgen{{.*}}<true>
  %2 = pop.cast %c : !kgen.scalar<bf16> to !kgen.scalar<bool>
  // CHECK-DAG: %[[C3:.*]] = kgen{{.*}}index{{.*}}<10>
  %3 = pop.cast %c : !kgen.scalar<bf16> to !kgen.scalar<index>
  // CHECK-NEXT: return %[[C0]], %[[C1]], %[[C2]], %[[C3]]
  kgen.return %0, %1, %2, %3 : !kgen.scalar<f16>, !kgen.scalar<ui8>, !kgen.scalar<bool>, !kgen.scalar<index>
}

// CHECK-LABEL: @cast_si128
kgen.func @cast_si128() -> (!kgen.scalar<f16>, !kgen.scalar<ui8>, !kgen.scalar<bool>, !kgen.scalar<index>) {
  %c = kgen.param.constant: scalar<si128> = <18446744073709551615>
  // CHECK-DAG: %[[C0:.*]] = kgen{{.*}}<"+Inf">
  %0 = pop.cast %c : !kgen.scalar<si128> to !kgen.scalar<f16>
  // CHECK-DAG: %[[C1:.*]] = kgen{{.*}}ui8{{.*}}<255>
  %1 = pop.cast %c : !kgen.scalar<si128> to !kgen.scalar<ui8>
  // CHECK-DAG: %[[C2:.*]] = kgen{{.*}}<true>
  %2 = pop.cast %c : !kgen.scalar<si128> to !kgen.scalar<bool>
  // CHECK-DAG: %[[C3:.*]] = kgen{{.*}}index{{.*}}<-1>
  %3 = pop.cast %c : !kgen.scalar<si128> to !kgen.scalar<index>
  // CHECK-NEXT: return %[[C0]], %[[C1]], %[[C2]], %[[C3]]
  kgen.return %0, %1, %2, %3 : !kgen.scalar<f16>, !kgen.scalar<ui8>, !kgen.scalar<bool>, !kgen.scalar<index>
}

// CHECK-LABEL: @cast_uindex_index
kgen.func @cast_uindex_index() -> (!kgen.scalar<index>, !kgen.scalar<uindex>) {
  // CHECK-DAG: %[[C0:.*]] = kgen{{.*}}index{{.*}}<4294967296>
  // CHECK-DAG: %[[C1:.*]] = kgen{{.*}}uindex{{.*}}<4294967297>
  %cu = kgen.param.constant: scalar<uindex> = <4294967296>
  %ci = kgen.param.constant: scalar<index> = <4294967297>
  %0 = pop.cast %cu : !kgen.scalar<uindex> to !kgen.scalar<index>
  %1 = pop.cast %ci : !kgen.scalar<index> to !kgen.scalar<uindex>
  // CHECK-NEXT: return %[[C0]], %[[C1]]
  kgen.return %0, %1 : !kgen.scalar<index>, !kgen.scalar<uindex>
}
}

// -----

// Big-endian (BE) bool bitcast tests.
module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="E-p:64:64", simd_bit_width = 128, index_bit_width = 64>} {

// CHECK-LABEL: @bitcast_pack_bools_to_ui8_be
kgen.func @bitcast_pack_bools_to_ui8_be() -> (!kgen.scalar<ui8>) {
  // Same input as LE test, but big-endian packs elem0 into MSB.
  // BE: 0b11101010 = 234
  %bools = kgen.param.constant: simd<8, bool> = <<true, true, true, false, true, false, true, false>>
  // CHECK: kgen.param.constant: scalar<ui8> = <234>
  %0 = pop.bitcast %bools : !kgen.simd<8, bool> to !kgen.scalar<ui8>
  kgen.return %0 : !kgen.scalar<ui8>
}

// CHECK-LABEL: @bitcast_unpack_ui8_to_bools_be
kgen.func @bitcast_unpack_ui8_to_bools_be() -> (!kgen.simd<8, bool>) {
  // 87 = 0b01010111, BE: bit7→elem0, bit0→elem7
  %byte = kgen.param.constant: scalar<ui8> = <87>
  // CHECK: kgen.param.constant: simd<8, bool> = <<false, true, false, true, false, true, true, true>>
  %0 = pop.bitcast %byte : !kgen.scalar<ui8> to !kgen.simd<8, bool>
  kgen.return %0 : !kgen.simd<8, bool>
}

// CHECK-LABEL: @bitcast_pack_bools_to_simd_be
kgen.func @bitcast_pack_bools_to_simd_be() -> (!kgen.simd<2, ui8>) {
  // First 8 bools BE → 234 (0xEA), next 8 bools BE → 85 (0x55)
  %bools = kgen.param.constant: simd<16, bool> = <<true, true, true, false, true, false, true, false, false, true, false, true, false, true, false, true>>
  // CHECK: kgen.param.constant: simd<2, ui8> = <<234, 85>>
  %0 = pop.bitcast %bools : !kgen.simd<16, bool> to !kgen.simd<2, ui8>
  kgen.return %0 : !kgen.simd<2, ui8>
}

// CHECK-LABEL: @bitcast_unpack_simd_to_bools_be
kgen.func @bitcast_unpack_simd_to_bools_be() -> (!kgen.simd<16, bool>) {
  // 87 BE → F,T,F,T,F,T,T,T; 170 = 0xAA BE → T,F,T,F,T,F,T,F
  %bytes = kgen.param.constant: simd<2, ui8> = <<87, 170>>
  // CHECK: kgen.param.constant: simd<16, bool> = <<false, true, false, true, false, true, true, true, true, false, true, false, true, false, true, false>>
  %0 = pop.bitcast %bytes : !kgen.simd<2, ui8> to !kgen.simd<16, bool>
  kgen.return %0 : !kgen.simd<16, bool>
}

}

// -----

// CHECK-LABEL: @cast_si64_ui64_index
kgen.func @cast_si64_ui64_index(%in : !kgen.scalar<si64>) -> !kgen.scalar<index> {
  // CHECK-NEXT: pop.cast {{.*}} !kgen.scalar<si64> to !kgen.scalar<ui64>
  %1 = pop.cast %in : !kgen.scalar<si64> to !kgen.scalar<ui64>
  // CHECK-NEXT: pop.cast {{.*}} !kgen.scalar<ui64> to !kgen.scalar<index>
  %2 = pop.cast %1 : !kgen.scalar<ui64> to !kgen.scalar<index>

  kgen.return %2 : !kgen.scalar<index>
}

// CHECK-LABEL: @cast_fp_too_big
kgen.func @cast_fp_too_big() -> (!kgen.scalar<si2>, !kgen.scalar<si2>) {
  // CHECK-DAG: <1>
  // CHECK-DAG: <"7">
  %0 = kgen.param.constant: scalar<f16> = <"1.5">
  %1 = kgen.param.constant: scalar<f16> = <"7">
  %2 = pop.cast %0 : !kgen.scalar<f16> to !kgen.scalar<si2>
  // CHECK: %[[TOO_BIG:.*]] = pop.cast
  %3 = pop.cast %1 : !kgen.scalar<f16> to !kgen.scalar<si2>
  // CHECK-NEXT: return %{{.*}}, %[[TOO_BIG]]
  kgen.return %2, %3 : !kgen.scalar<si2>, !kgen.scalar<si2>
}

// CHECK-LABEL: @cast_index_to_int
kgen.func @cast_index_to_int() -> !kgen.scalar<si32> {
  // CHECK-NEXT: scalar<si32> = <-2>
  %0 = kgen.param.constant: scalar<index> = <-2>
  %1 = pop.cast %0 : !kgen.scalar<index> to !kgen.scalar<si32>
  kgen.return %1 : !kgen.scalar<si32>
}

// CHECK-LABEL: @cast_index_to_index
kgen.func @cast_index_to_index() -> !kgen.scalar<index> {
  // CHECK-NEXT: scalar<index> = <99999999999>
  // CHECK-NOT: pop.cast
  %0 = kgen.param.constant: scalar<index> = <99999999999>
  %1 = pop.cast %0 : !kgen.scalar<index> to !kgen.scalar<index>
  kgen.return %1 : !kgen.scalar<index>
}

// CHECK-LABEL: @cast_index_to_si64_large
kgen.func @cast_index_to_si64_large() -> !kgen.scalar<si64> {
  // CHECK-NEXT: kgen.param.constant: scalar<index> = <-8664705627211539068>
  %0 = kgen.param.constant: scalar<index> = <-8664705627211539068>
  %1 = pop.cast %0 : !kgen.scalar<index> to !kgen.scalar<si64>
  kgen.return %1 : !kgen.scalar<si64>
}

// CHECK-LABEL: @cast_index_to_si32_large
kgen.func @cast_index_to_si32_large() -> !kgen.scalar<si32> {
  // CHECK-NEXT: kgen.param.constant: scalar<si32> = <-1095082620>
  %0 = kgen.param.constant: scalar<index> = <-8664705627211539068>
  %1 = pop.cast %0 : !kgen.scalar<index> to !kgen.scalar<si32>
  kgen.return %1 : !kgen.scalar<si32>
}

// CHECK-LABEL: @cast_and_truncate
kgen.func @cast_and_truncate(%v0 : !kgen.simd<2, si64>) -> !kgen.simd<2, si32> {
  // CHECK-NEXT: pop.cast %arg0 : !kgen.simd<2, si64> to !kgen.simd<2, si32>
  // CHECK-NOT: pop.cast
  %v1 = pop.cast %v0 : !kgen.simd<2, si64> to !kgen.simd<2, si32>
  %v2 = pop.cast %v1 : !kgen.simd<2, si32> to !kgen.simd<2, si64>
  %v3 = pop.cast %v2 : !kgen.simd<2, si64> to !kgen.simd<2, si32>
  kgen.return %v3 : !kgen.simd<2, si32>
}

// CHECK-LABEL: @sext_and_sext
kgen.func @sext_and_sext(%v0 : !kgen.simd<2, si16>) -> !kgen.simd<2, si64> {
  // CHECK-NEXT: pop.cast %arg0 : !kgen.simd<2, si16> to !kgen.simd<2, si64>
  %v1 = pop.cast %v0 : !kgen.simd<2, si16> to !kgen.simd<2, si32>
  %v2 = pop.cast %v1 : !kgen.simd<2, si32> to !kgen.simd<2, si64>
  kgen.return %v2 : !kgen.simd<2, si64>
}

// CHECK-LABEL: @zext_and_zext
kgen.func @zext_and_zext(%v0 : !kgen.simd<2, ui16>) -> !kgen.simd<2, ui64> {
  // CHECK-NEXT: pop.cast %arg0 : !kgen.simd<2, ui16> to !kgen.simd<2, ui64>
  %v1 = pop.cast %v0 : !kgen.simd<2, ui16> to !kgen.simd<2, ui32>
  %v2 = pop.cast %v1 : !kgen.simd<2, ui32> to !kgen.simd<2, ui64>
  kgen.return %v2 : !kgen.simd<2, ui64>
}

// The chain zero-extends; a direct si32 -> ui64 cast would sign-extend.
// CHECK-LABEL: @sign_flip_at_same_width_then_extend
kgen.func @sign_flip_at_same_width_then_extend(
    %v0 : !kgen.scalar<si32>) -> !kgen.scalar<ui64> {
  // CHECK-NEXT: pop.cast %arg0 : !kgen.scalar<si32> to !kgen.scalar<ui32>
  %v1 = pop.cast %v0 : !kgen.scalar<si32> to !kgen.scalar<ui32>
  // CHECK-NEXT: pop.cast {{.*}} : !kgen.scalar<ui32> to !kgen.scalar<ui64>
  %v2 = pop.cast %v1 : !kgen.scalar<ui32> to !kgen.scalar<ui64>
  kgen.return %v2 : !kgen.scalar<ui64>
}

// Same flip while widening: still zero-extends, so it cannot collapse.
// CHECK-LABEL: @sign_flip_while_widening_then_extend
kgen.func @sign_flip_while_widening_then_extend(
    %v0 : !kgen.scalar<si16>) -> !kgen.scalar<ui64> {
  // CHECK-NEXT: pop.cast %arg0 : !kgen.scalar<si16> to !kgen.scalar<ui32>
  %v1 = pop.cast %v0 : !kgen.scalar<si16> to !kgen.scalar<ui32>
  // CHECK-NEXT: pop.cast {{.*}} : !kgen.scalar<ui32> to !kgen.scalar<ui64>
  %v2 = pop.cast %v1 : !kgen.scalar<ui32> to !kgen.scalar<ui64>
  kgen.return %v2 : !kgen.scalar<ui64>
}

// The mirror flip: the chain sign-extends, a direct ui32 -> si64 zero-extends.
// CHECK-LABEL: @unsigned_to_signed_flip_at_same_width
kgen.func @unsigned_to_signed_flip_at_same_width(
    %v0 : !kgen.scalar<ui32>) -> !kgen.scalar<si64> {
  // CHECK-NEXT: pop.cast %arg0 : !kgen.scalar<ui32> to !kgen.scalar<si32>
  // CHECK-NEXT: pop.cast {{.*}} : !kgen.scalar<si32> to !kgen.scalar<si64>
  %v1 = pop.cast %v0 : !kgen.scalar<ui32> to !kgen.scalar<si32>
  %v2 = pop.cast %v1 : !kgen.scalar<si32> to !kgen.scalar<si64>
  kgen.return %v2 : !kgen.scalar<si64>
}

// Foldable: zero-extending to si32 clears the sign bit, so the sext matches.
// CHECK-LABEL: @unsigned_to_signed_flip_still_folds
kgen.func @unsigned_to_signed_flip_still_folds(
    %v0 : !kgen.scalar<ui16>) -> !kgen.scalar<si64> {
  // CHECK-NEXT: pop.cast %arg0 : !kgen.scalar<ui16> to !kgen.scalar<si64>
  %v1 = pop.cast %v0 : !kgen.scalar<ui16> to !kgen.scalar<si32>
  %v2 = pop.cast %v1 : !kgen.scalar<si32> to !kgen.scalar<si64>
  kgen.return %v2 : !kgen.scalar<si64>
}

// CHECK-LABEL: @fpext_and_fptoint
kgen.func @fpext_and_fptoint(%arg0 : !kgen.simd<2, f16>) -> (!kgen.simd<2, si32>, !kgen.simd<2, ui32>) {
  // COM: fptosi(fpext)
  %v12 = pop.cast %arg0 : !kgen.simd<2, f16> to !kgen.simd<2, f32>
  %v13 = pop.cast %v12 : !kgen.simd<2, f32> to !kgen.simd<2, si32>

  // COM: fptoui(fpext)
  %v14 = pop.cast %arg0 : !kgen.simd<2, f16> to !kgen.simd<2, f32>
  %v15 = pop.cast %v14 : !kgen.simd<2, f32> to !kgen.simd<2, ui32>

  kgen.return %v13, %v15 : !kgen.simd<2, si32>, !kgen.simd<2, ui32>
}

// CHECK-LABEL: @intext_and_fptoint
kgen.func @intext_and_fptoint(%arg0 : !kgen.simd<2, ui16>, %arg1 : !kgen.simd<2, si16>) -> (!kgen.simd<2, f32>, !kgen.simd<2, f32>) {
  // COM: uitofp(zext)
  %v12 = pop.cast %arg0 : !kgen.simd<2, ui16> to !kgen.simd<2, ui32>
  %v13 = pop.cast %v12 : !kgen.simd<2, ui32> to !kgen.simd<2, f32>

  // COM: sitofp(sext)
  %v14 = pop.cast %arg1 : !kgen.simd<2, si16> to !kgen.simd<2, si32>
  %v15 = pop.cast %v14 : !kgen.simd<2, si32> to !kgen.simd<2, f32>

  kgen.return %v13, %v15 : !kgen.simd<2, f32>, !kgen.simd<2, f32>
}

// CHECK-LABEL: @unsupported_intcast_and_intcast
kgen.func @unsupported_intcast_and_intcast(%v0 : !kgen.simd<2, ui8>) -> !kgen.simd<2, si64> {
  // CHECK-NEXT: pop.cast %arg0 : !kgen.simd<2, ui8> to !kgen.simd<2, ui16>
  %v1 = pop.cast %v0 : !kgen.simd<2, ui8> to !kgen.simd<2, ui16>
  // CHECK-NEXT: pop.cast %0 : !kgen.simd<2, ui16> to !kgen.simd<2, si32>
  %v2 = pop.cast %v1 : !kgen.simd<2, ui16> to !kgen.simd<2, si32>
  // CHECK-NEXT: pop.cast %1 : !kgen.simd<2, si32> to !kgen.simd<2, ui64>
  %v3 = pop.cast %v2 : !kgen.simd<2, si32> to !kgen.simd<2, ui64>
  // CHECK-NEXT: pop.cast %2 : !kgen.simd<2, ui64> to !kgen.simd<2, si64>
  %v4 = pop.cast %v3 : !kgen.simd<2, ui64> to !kgen.simd<2, si64>
  kgen.return %v4 : !kgen.simd<2, si64>
}

// CHECK-LABEL: @fpext_and_fpext
kgen.func @fpext_and_fpext(%arg0 : !kgen.simd<2, f16>) -> !kgen.simd<2, f64> {
  // COM: fpext(fpext)
  %v8 = pop.cast %arg0 : !kgen.simd<2, f16> to !kgen.simd<2, f32>
  // CHECK-NEXT: pop.cast %arg0 : !kgen.simd<2, f16> to !kgen.simd<2, f64>
  %v9 = pop.cast %v8 : !kgen.simd<2, f32> to !kgen.simd<2, f64>
  kgen.return %v9 : !kgen.simd<2, f64>
}

// CHECK-LABEL: @fpext_and_fptrunc
kgen.func @fpext_and_fptrunc(%arg0 : !kgen.simd<2, f16>) -> !kgen.simd<2, f32> {
  // COM: fptrunc(fpext)
  %v10 = pop.cast %arg0 : !kgen.simd<2, f16> to !kgen.simd<2, f64>
  // CHECK-NEXT: pop.cast %arg0 : !kgen.simd<2, f16> to !kgen.simd<2, f32>
  %v11 = pop.cast %v10 : !kgen.simd<2, f64> to !kgen.simd<2, f32>
  kgen.return %v11 : !kgen.simd<2, f32>
}

// Rounding twice can differ from rounding once, so T2 has to stay.
// CHECK-LABEL: @fptrunc_and_fptrunc
kgen.func @fptrunc_and_fptrunc(%v0 : !kgen.simd<2, f64>) -> !kgen.simd<2, f16> {
  // CHECK-NEXT: pop.cast %arg0 : !kgen.simd<2, f64> to !kgen.simd<2, f32>
  // CHECK-NEXT: pop.cast {{.*}} : !kgen.simd<2, f32> to !kgen.simd<2, f16>
  %v1 = pop.cast %v0 : !kgen.simd<2, f64> to !kgen.simd<2, f32>
  %v2 = pop.cast %v1 : !kgen.simd<2, f32> to !kgen.simd<2, f16>
  kgen.return %v2 : !kgen.simd<2, f16>
}

// A `fast` cast accepts a result that differs from the double rounding.
// CHECK-LABEL: @fptrunc_and_fptrunc_fast
kgen.func @fptrunc_and_fptrunc_fast(
    %v0 : !kgen.simd<2, f64>) -> !kgen.simd<2, f16> {
  // CHECK-NEXT: pop.cast %arg0 : !kgen.simd<2, f64> to !kgen.simd<2, f16>
  %v1 = pop.cast %v0 {fastmathFlags = #pop.fmf<fast>} : !kgen.simd<2, f64> to !kgen.simd<2, f32>
  %v2 = pop.cast %v1 {fastmathFlags = #pop.fmf<fast>} : !kgen.simd<2, f32> to !kgen.simd<2, f16>
  kgen.return %v2 : !kgen.simd<2, f16>
}

// Only the rewritten cast's own `fast` licenses dropping T2's rounding.
// CHECK-LABEL: @fptrunc_and_fptrunc_inner_fast_only
kgen.func @fptrunc_and_fptrunc_inner_fast_only(
    %v0 : !kgen.simd<2, f64>) -> !kgen.simd<2, f16> {
  // CHECK-NEXT: pop.cast %arg0 {fastmathFlags = #pop.fmf<fast>} : !kgen.simd<2, f64> to !kgen.simd<2, f32>
  // CHECK-NEXT: pop.cast {{.*}} : !kgen.simd<2, f32> to !kgen.simd<2, f16>
  %v1 = pop.cast %v0 {fastmathFlags = #pop.fmf<fast>} : !kgen.simd<2, f64> to !kgen.simd<2, f32>
  %v2 = pop.cast %v1 : !kgen.simd<2, f32> to !kgen.simd<2, f16>
  kgen.return %v2 : !kgen.simd<2, f16>
}

// T3 is still wider than T1, so T2 keeps bits the folded cast would drop.
// CHECK-LABEL: @widen_then_truncate_below_t2
kgen.func @widen_then_truncate_below_t2(
    %v0 : !kgen.simd<2, si16>) -> !kgen.simd<2, si32> {
  // CHECK-NEXT: pop.cast %arg0 : !kgen.simd<2, si16> to !kgen.simd<2, si64>
  // CHECK-NEXT: pop.cast {{.*}} : !kgen.simd<2, si64> to !kgen.simd<2, si32>
  %v1 = pop.cast %v0 : !kgen.simd<2, si16> to !kgen.simd<2, si64>
  %v2 = pop.cast %v1 : !kgen.simd<2, si64> to !kgen.simd<2, si32>
  kgen.return %v2 : !kgen.simd<2, si32>
}

// T3 reinterprets T2 at its width, but T2 truncated T1 first.
// CHECK-LABEL: @truncate_then_reinterpret
kgen.func @truncate_then_reinterpret(
    %v0 : !kgen.simd<2, si64>) -> !kgen.simd<2, ui32> {
  // CHECK-NEXT: pop.cast %arg0 : !kgen.simd<2, si64> to !kgen.simd<2, si32>
  // CHECK-NEXT: pop.cast {{.*}} : !kgen.simd<2, si32> to !kgen.simd<2, ui32>
  %v1 = pop.cast %v0 : !kgen.simd<2, si64> to !kgen.simd<2, si32>
  %v2 = pop.cast %v1 : !kgen.simd<2, si32> to !kgen.simd<2, ui32>
  kgen.return %v2 : !kgen.simd<2, ui32>
}

// T3 reinterprets T2 at its width with the opposite sign.
// CHECK-LABEL: @widen_then_reinterpret_other_sign
kgen.func @widen_then_reinterpret_other_sign(
    %v0 : !kgen.simd<2, si16>) -> !kgen.simd<2, ui32> {
  // CHECK-NEXT: pop.cast %arg0 : !kgen.simd<2, si16> to !kgen.simd<2, si32>
  // CHECK-NEXT: pop.cast {{.*}} : !kgen.simd<2, si32> to !kgen.simd<2, ui32>
  %v1 = pop.cast %v0 : !kgen.simd<2, si16> to !kgen.simd<2, si32>
  %v2 = pop.cast %v1 : !kgen.simd<2, si32> to !kgen.simd<2, ui32>
  kgen.return %v2 : !kgen.simd<2, ui32>
}

// COM: test chain of int2fp/fp2int conversions
// CHECK-LABEL: @unsupported_conv_conv
kgen.func @unsupported_conv_conv(%arg0 : !kgen.simd<2, ui16>, %arg1 : !kgen.simd<2, si16>) -> (!kgen.simd<2, ui32>, !kgen.simd<2, f64>, !kgen.simd<2, si32>, !kgen.simd<2, f64>, !kgen.simd<2, f16>) {
  // COM: fptoui(uitofp)
  // CHECK-NEXT: pop.cast %arg0 : !kgen.simd<2, ui16> to !kgen.simd<2, f32>
  %0 = pop.cast %arg0 : !kgen.simd<2, ui16> to !kgen.simd<2, f32>
  // CHECK: pop.cast %0 : !kgen.simd<2, f32> to !kgen.simd<2, ui32>
  %v2 = pop.cast %0 : !kgen.simd<2, f32> to !kgen.simd<2, ui32>

  // COM: fpext(uitofp)
  // CHECK-NEXT: pop.cast %arg0 : !kgen.simd<2, ui16> to !kgen.simd<2, f32>
  %1 = pop.cast %arg0 : !kgen.simd<2, ui16> to !kgen.simd<2, f32>
  // CHECK-NEXT: pop.cast %2 : !kgen.simd<2, f32> to !kgen.simd<2, f64>
  %v3 = pop.cast %1 : !kgen.simd<2, f32> to !kgen.simd<2, f64>

  // COM: fptrunc(sitofp)
  // CHECK-NEXT: pop.cast %arg1 : !kgen.simd<2, si16> to !kgen.simd<2, f32>
  %2 = pop.cast %arg1 : !kgen.simd<2, si16> to !kgen.simd<2, f32>
  // CHECK-NEXT: pop.cast %4 : !kgen.simd<2, f32> to !kgen.simd<2, si32>
  %v5 = pop.cast %2 : !kgen.simd<2, f32> to !kgen.simd<2, si32>

  // COM: fpext(sitofp)
  // CHECK-NEXT: pop.cast %arg1 : !kgen.simd<2, si16> to !kgen.simd<2, f32>
  %3 = pop.cast %arg1 : !kgen.simd<2, si16> to !kgen.simd<2, f32>
  // CHECK-NEXT: pop.cast %6 : !kgen.simd<2, f32> to !kgen.simd<2, f64>
  %v6 = pop.cast %3 : !kgen.simd<2, f32> to !kgen.simd<2, f64>

  // COM: fptrunc(sitofp)
  // CHECK-NEXT: pop.cast %arg1 : !kgen.simd<2, si16> to !kgen.simd<2, f32>
  %4 = pop.cast %arg1 : !kgen.simd<2, si16> to !kgen.simd<2, f32>
  // CHECK: pop.cast %8 : !kgen.simd<2, f32> to !kgen.simd<2, f16>
  %v7 = pop.cast %4 : !kgen.simd<2, f32> to !kgen.simd<2, f16>

  kgen.return %v2, %v3, %v5, %v6, %v7 : !kgen.simd<2, ui32>, !kgen.simd<2, f64>, !kgen.simd<2, si32>, !kgen.simd<2, f64>, !kgen.simd<2, f16>
}

// CHECK-LABEL: @simd_extractelement
kgen.func @simd_extractelement() -> (!kgen.scalar<si8>) {
  // CHECK-NEXT: <20>
  %idx1 = index.constant 1
  %0 = kgen.param.constant: simd<2, si8> = <<10, 20>>
  %1 = pop.simd.extractelement %0[%idx1] : !kgen.simd<2, si8>
  kgen.return %1 : !kgen.scalar<si8>
}

// CHECK-LABEL: @simd_extractelement_scalar
kgen.func @simd_extractelement_scalar(%scalar : !kgen.scalar<f32>, %arg : index) -> (!kgen.scalar<f32>) {
  // CHECK: (%[[SCALAR:.*]]: !kgen.scalar<f32>, %[[ARG:.*]]: index)
  // CHECK-NEXT: kgen.return %[[SCALAR]]
  %1 = pop.simd.extractelement %scalar[%arg] : !kgen.scalar<f32>
  kgen.return %1 : !kgen.scalar<f32>
}

// CHECK-LABEL: @simd_insertelement
kgen.func @simd_insertelement() -> (!kgen.simd<2, si8>) {
  // CHECK-NEXT: <30, 20>
  %idx0 = index.constant 0
  %0 = kgen.param.constant: simd<2, si8> = <<10, 20>>
  %1 = kgen.param.constant: scalar<si8> = <30>
  %2 = pop.simd.insertelement %1, %0[%idx0] : !kgen.simd<2, si8>
  kgen.return %2 : !kgen.simd<2, si8>
}

// CHECK-LABEL: @simd_insertelement2
kgen.func @simd_insertelement2() -> (!kgen.simd<2, si8>) {
  // CHECK-NEXT: <30, 0>
  %idx0 = index.constant 0

  // Start from uninitialized memory instead of a constant.
  %0 = kgen.param.constant: simd<2, si8> = <#interp.uninitmem : !kgen.simd<2, si8>>
  %1 = kgen.param.constant: scalar<si8> = <30>
  %2 = pop.simd.insertelement %1, %0[%idx0] : !kgen.simd<2, si8>
  kgen.return %2 : !kgen.simd<2, si8>
}

// CHECK-LABEL: @simd_shuffle
kgen.func @simd_shuffle() -> !kgen.simd<2, si8> {
  // CHECK-NEXT: <3, 2>
  %0 = kgen.param.constant: scalar<si8> = <<2>>
  %1 = kgen.param.constant: scalar<si8> = <<3>>
  %2 = pop.simd.shuffle <1, si8> %0, %1 -> <2, si8> :array<2, index> [1, 0]
  kgen.return %2 : !kgen.simd<2, si8>
}

// CHECK-LABEL: @simd_splat_scalar
kgen.func @simd_splat_scalar(%arg0: !kgen.scalar<si8>) -> !kgen.scalar<si8> {
  // CHECK: (%[[ARG0:.*]]: !kgen.scalar<si8>)
  // CHECK-NEXT: kgen.return %[[ARG0]]
  %1 = pop.simd.splat %arg0 : !kgen.scalar<si8>
  kgen.return %1 : !kgen.scalar<si8>
}

// CHECK-LABEL: @simd_splat
kgen.func @simd_splat() -> !kgen.simd<2, si8> {
  // CHECK-NEXT: <2>
  %0 = kgen.param.constant: scalar<si8> = <<2>>
  %1 = pop.simd.splat %0 : !kgen.simd<2, si8>
  kgen.return %1 : !kgen.simd<2, si8>
}

// CHECK-LABEL: @array_create
kgen.func @array_create() -> !pop.array<2, index> {
  // CHECK-NEXT: constant: array<2, index> = <[0, 0]>
  %idx0 = index.constant 0
  %0 = pop.array.create [%idx0, %idx0] : !pop.array<2, index>
  kgen.return %0 : !pop.array<2, index>
}

// CHECK-LABEL: @array_repeat
kgen.func @array_repeat() -> !pop.array<3, index> {
  // CHECK-NEXT: constant: array<3, index> = <[0, 1, 0]>
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %0 = pop.array.repeat [%idx0, %idx1] : !pop.array<3, index>
  kgen.return %0 : !pop.array<3, index>
}

// CHECK-LABEL: @array_repeat_get
kgen.func @array_repeat_get(%arg0: index, %arg1: index) -> (index, index, index) {
  // CHECK: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
  // CHECK-NEXT: kgen.return %[[ARG0]], %[[ARG1]], %[[ARG0]] : index, index, index

  %0 = pop.array.repeat [%arg0, %arg1] : !pop.array<3, index>
  %1 = pop.array.get %0[0] : !pop.array<3, index>
  %2 = pop.array.get %0[1] : !pop.array<3, index>
  %3 = pop.array.get %0[2] : !pop.array<3, index>
  kgen.return %1, %2, %3 : index, index, index
}

// CHECK-LABEL: @array_get
kgen.func @array_get() -> index {
  // CHECK-NEXT: constant = <1>
  %0 = kgen.param.constant: array<2, index> = <[0, 1]>
  %1 = pop.array.get %0[1] : !pop.array<2, index>
  kgen.return %1 : index
}

// CHECK-LABEL: @array_get_unknown
kgen.func @array_get_unknown() -> index {
  %array = kgen.param.constant: array<4, index> = <#interp.uninitmem>
  %0 = pop.array.get %array[0] : !pop.array<4, index>

  // CHECK: %[[OUT:.*]] = kgen.param.constant = <#interp.uninitmem>
  // CHECK-NEXT: kgen.return %[[OUT]]

  kgen.return %0 : index
}

// CHECK-LABEL: @array_get_non_const_0
kgen.func @array_get_non_const_0(%arg0: index, %arg1: index) -> index {
  // CHECK: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
  // CHECK: kgen.return %[[ARG0]]
  %0 = pop.array.create [%arg0, %arg1] : !pop.array<2, index>
  %1 = pop.array.get %0[0] : !pop.array<2, index>
  kgen.return %1 : index
}

// CHECK-LABEL: @array_get_non_const_1
kgen.func @array_get_non_const_1(%arg0: index, %arg1: index) -> index {
  // CHECK: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
  // CHECK: kgen.return %[[ARG1]]
  %0 = pop.array.create [%arg0, %arg1] : !pop.array<2, index>
  %1 = pop.array.get %0[1] : !pop.array<2, index>
  kgen.return %1 : index
}

// CHECK-LABEL: @array_gep
kgen.func @array_gep(%array: !kgen.pointer<array<1, index>>, %idx: index) -> !kgen.pointer<index> {
  // CHECK: (%[[ARRAY:.*]]: !kgen.pointer<array<1, index>>, %[[IDX:.*]]: index)
  // CHECK-NEXT: %[[ZERO:.*]] = kgen.param.constant = <0>
  // CHECK-NEXT: %[[GEP:.*]] = pop.array.gep %[[ARRAY]][%[[ZERO]]]
  // CHECK-NEXT: kgen.return %[[GEP]]
  %1 = pop.array.gep %array[%idx] : !kgen.pointer<array<1, index>>
  kgen.return %1 : !kgen.pointer<index>
}

// CHECK-LABEL: @array_gep_unchanged
kgen.func @array_gep_unchanged(%array: !kgen.pointer<array<2, index>>, %idx: index) -> !kgen.pointer<index> {
  // CHECK: (%[[ARRAY:.*]]: !kgen.pointer<array<2, index>>, %[[IDX:.*]]: index)
  // CHECK-NEXT: %[[GEP:.*]] = pop.array.gep %[[ARRAY]][%[[IDX]]]
  // CHECK-NEXT: kgen.return %[[GEP]]
  %1 = pop.array.gep %array[%idx] : !kgen.pointer<array<2, index>>
  kgen.return %1 : !kgen.pointer<index>
}

// CHECK-LABEL: @array_replace
kgen.func @array_replace() -> !pop.array<2, index> {
  // CHECK-NEXT: constant: array<2, index> = <[0, 1]>
  %0 = kgen.param.constant: array<2, index> = <[0, 0]>
  %1 = index.constant 1
  %2 = pop.array.replace %1, %0[1] : !pop.array<2, index>
  kgen.return %2 : !pop.array<2, index>
}

kgen.func @array_replace_with_create(%idx0: index, %idx1: index, %idx2: index) -> !pop.array<3, index> {
  // CHECK: (%[[IDX0:.*]]: index, %[[IDX1:.*]]: index, %[[IDX2:.*]]: index)
  // CHECK-NEXT: %[[OUT:.*]] = pop.array.create [%[[IDX0]], %[[IDX1]], %[[IDX2]]]
  // CHECK-NEXT: kgen.return %[[OUT]]
  %0 = pop.array.create [%idx0, %idx0, %idx0] : !pop.array<3, index>
  %1 = pop.array.replace %idx1, %0[1] : !pop.array<3, index>
  %2 = pop.array.replace %idx2, %1[2] : !pop.array<3, index>
  kgen.return %2 : !pop.array<3, index>
}

kgen.func @array_replace_with_const(%idx0: index, %idx1: index) -> !pop.array<3, index> {
  // CHECK: (%[[IDX0:.*]]: index, %[[IDX1:.*]]: index)
  // CHECK-NEXT: %[[C0:.*]] = kgen.param.constant = <0>
  // CHECK-NEXT: %[[OUT:.*]] = pop.array.create [%[[C0]], %[[IDX0]], %[[IDX1]]]
  // CHECK-NEXT: kgen.return %[[OUT]]
  %0 = kgen.param.constant: array<3, index> = <[0, 0, 0]>
  %1 = pop.array.replace %idx0, %0[1] : !pop.array<3, index>
  %2 = pop.array.replace %idx1, %1[2] : !pop.array<3, index>
  kgen.return %2 : !pop.array<3, index>
}

// CHECK-LABEL: @pointer_to_index
kgen.func @pointer_to_index() -> index {
  // CHECK-DAG: <1>
  %0 = kgen.param.constant: pointer<i8> = <#interp.pointer<1>>
  %1 = pop.pointer_to_index %0 : !kgen.pointer<i8>
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.func @bitcast_ptr_to_index
kgen.func @bitcast_ptr_to_index(%arg0: !kgen.pointer<none>) -> index {
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<index>
  // CHECK-NEXT: %0 = pop.pointer_to_index %arg0
  %1 = pop.pointer_to_index %0 : !kgen.pointer<index>
  // CHECK-NEXT: return %0
  kgen.return %1 : index
}

// CHECK-LABEL: @cast_to_builtin
kgen.func @cast_to_builtin() -> (
    vector<2xi1>, vector<2xindex>, vector<2xi4>, vector<2xbf16>,
    i1, index, ui8, f16) {
  // CHECK-DAG: %[[C0:.*]] = kgen{{.*}}vector<2xi1> = <#M.dense_array<true, false>>
  // CHECK-DAG: %[[C1:.*]] = kgen{{.*}}vector<2xindex> = <#M.dense_array<1, 2>>
  // CHECK-DAG: %[[C2:.*]] = kgen{{.*}}vector<2xi4> = <#M.dense_array<3, 4>>
  // CHECK-DAG: %[[C3:.*]] = kgen{{.*}}vector<2xbf16> = <#M.dense_array<1.5{{0+}}e+00, 2.5{{0+}}e+00>>
  // CHECK-DAG: %[[C4:.*]] = kgen{{.*}}i1 = <1>
  // CHECK-DAG: %[[C5:.*]] = kgen{{.*}}constant = <10>
  // CHECK-DAG: %[[C6:.*]] = kgen{{.*}}ui8 = <66>
  // CHECK-DAG: %[[C7:.*]] = kgen{{.*}}f16 = <5.5{{0+}}e+00>
  %0 = kgen.param.constant: simd<2, bool> = <<true, false>>
  %1 = kgen.param.constant: simd<2, index> = <<1, 2>>
  %2 = kgen.param.constant: simd<2, si4> = <<3, 4>>
  %3 = kgen.param.constant: simd<2, bf16> = <<"1.5", "2.5">>
  %4 = kgen.param.constant: scalar<bool> = <<true>>
  %5 = kgen.param.constant: scalar<index> = <<10>>
  %6 = kgen.param.constant: scalar<ui8> = <<66>>
  %7 = kgen.param.constant: scalar<f16> = <<"5.5">>

  %a0 = pop.cast_to_builtin %0 : !kgen.simd<2, bool> to vector<2xi1>
  %a1 = pop.cast_to_builtin %1 : !kgen.simd<2, index> to vector<2xindex>
  %a2 = pop.cast_to_builtin %2 : !kgen.simd<2, si4> to vector<2xi4>
  %a3 = pop.cast_to_builtin %3 : !kgen.simd<2, bf16> to vector<2xbf16>
  %a4 = pop.cast_to_builtin %4 : !kgen.scalar<bool> to i1
  %a5 = pop.cast_to_builtin %5 : !kgen.scalar<index> to index
  %a6 = pop.cast_to_builtin %6 : !kgen.scalar<ui8> to ui8
  %a7 = pop.cast_to_builtin %7 : !kgen.scalar<f16> to f16

  // CHECK: return %[[C0]], %[[C1]], %[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[C6]], %[[C7]]
  kgen.return %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7 :
    vector<2xi1>, vector<2xindex>, vector<2xi4>, vector<2xbf16>,
    i1, index, ui8, f16
}

// CHECK-LABEL: @cast_from_builtin
kgen.func @cast_from_builtin() -> (
    !kgen.simd<2, bool>, !kgen.simd<2, index>, !kgen.simd<2, si4>, !kgen.simd<2, bf16>,
    !kgen.scalar<bool>, !kgen.scalar<index>, !kgen.scalar<ui8>, !kgen.scalar<f16>,
    !kgen.scalar<index>) {
  // CHECK-DAG: %[[C0:.*]] = kgen{{.*}}simd<2, bool> = <<true, false>>
  // CHECK-DAG: %[[C1:.*]] = kgen{{.*}}simd<2, index> = <<1, 2>>
  // CHECK-DAG: %[[C2:.*]] = kgen{{.*}}simd<2, si4> = <<3, 4>>
  // CHECK-DAG: %[[C3:.*]] = kgen{{.*}}simd<2, bf16> = <<"1.5", "2.5">>
  // CHECK-DAG: %[[C4:.*]] = kgen{{.*}}scalar<bool> = <true>
  // CHECK-DAG: %[[C5:.*]] = kgen{{.*}}scalar<index> = <10>
  // CHECK-DAG: %[[C6:.*]] = kgen{{.*}}scalar<ui8> = <66>
  // CHECK-DAG: %[[C7:.*]] = kgen{{.*}}scalar<f16> = <"5.5">
  // CHECK-DAG: %[[C8:.*]] = kgen{{.*}}scalar<index> = <8>
  %0 = kgen.param.constant: vector<2xi1> = <#M.dense_array<true, false>>
  %1 = kgen.param.constant: vector<2xindex> = <#M.dense_array<1, 2>>
  %2 = kgen.param.constant: vector<2xi4> = <#M.dense_array<3, 4>>
  %3 = kgen.param.constant: vector<2xbf16> = <#M.dense_array<1.5, 2.5>>
  %4 = kgen.param.constant: i1 = <1>
  %5 = kgen.param.constant = <10>
  %6 = kgen.param.constant: ui8 = <66>
  %7 = kgen.param.constant: f16 = <5.5>
  %8 = index.constant 8

  %a0 = pop.cast_from_builtin %0 : vector<2xi1> to !kgen.simd<2, bool>
  %a1 = pop.cast_from_builtin %1 : vector<2xindex> to !kgen.simd<2, index>
  %a2 = pop.cast_from_builtin %2 : vector<2xi4> to !kgen.simd<2, si4>
  %a3 = pop.cast_from_builtin %3 : vector<2xbf16> to !kgen.simd<2, bf16>
  %a4 = pop.cast_from_builtin %4 : i1 to !kgen.scalar<bool>
  %a5 = pop.cast_from_builtin %5 : index to !kgen.scalar<index>
  %a6 = pop.cast_from_builtin %6 : ui8 to !kgen.scalar<ui8>
  %a7 = pop.cast_from_builtin %7 : f16 to !kgen.scalar<f16>
  %a8 = pop.cast_from_builtin %8 : index to !kgen.scalar<index>

  // CHECK: return %[[C0]], %[[C1]], %[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[C6]], %[[C7]], %[[C8]]
  kgen.return %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7, %a8 :
    !kgen.simd<2, bool>, !kgen.simd<2, index>, !kgen.simd<2, si4>, !kgen.simd<2, bf16>,
    !kgen.scalar<bool>, !kgen.scalar<index>, !kgen.scalar<ui8>, !kgen.scalar<f16>, !kgen.scalar<index>
}

// CHECK-LABEL: @cast_from_parameter
kgen.generator @cast_from_parameter<N>() -> !kgen.scalar<index> {
  %0 = kgen.param.constant = <N>
  // CHECK: pop.cast_from_builtin
  %1 = pop.cast_from_builtin %0 : index to !kgen.scalar<index>
  kgen.return %1 : !kgen.scalar<index>
}

// CHECK-LABEL: @dtype_to_ui8(
kgen.func @dtype_to_ui8() -> ui8 {
  // CHECK-NEXT: kgen.param.constant: ui8 = <1>
  %0 = kgen.param.constant: dtype = <bool>
  %1 = pop.dtype.to_ui8 %0
  kgen.return %1 : ui8
}

// CHECK-LABEL: @dtype_from_ui8(
kgen.func @dtype_from_ui8() -> !kgen.dtype {
  // CHECK-NEXT: kgen.param.constant: dtype = <bool>
  %0 = kgen.param.constant: ui8 = <1>
  %1 = pop.dtype.from_ui8 %0
  kgen.return %1 : !kgen.dtype
}

// CHECK-LABEL: @fold_offset
// CHECK: ([[ARG0:%.*]]:
kgen.func @fold_offset(%arg0: !kgen.pointer<index>) -> (!kgen.pointer<index>) {
  // CHECK-NEXT: kgen.return [[ARG0]]
  %0 = kgen.param.constant = <0>
  %1 = pop.offset %arg0[%0] : !kgen.pointer<index>
  kgen.return %1 : !kgen.pointer<index>
}

// CHECK-LABEL: @select
kgen.func @select(%arg0: !kgen.scalar<bool>, %arg1: i32, %arg2: i32) -> (i32, i32) {
  // CHECK-NEXT: kgen.return %arg1, %arg2
  %true = kgen.param.constant: scalar<bool> = <true>
  %0 = pop.select %arg0, %arg1, %arg1 : i32
  %1 = pop.select %true, %arg2, %arg1 : i32
  kgen.return %0, %1 : i32, i32
}

// CHECK-LABEL: @select_to_cond
kgen.func @select_to_cond(%cond: !kgen.scalar<bool>) -> !kgen.scalar<bool> {
  // CHECK-NEXT: kgen.return %arg0
  %true = kgen.param.constant: scalar<bool> = <true>
  %false = kgen.param.constant: scalar<bool> = <false>
  %0 = pop.select %cond, %true, %false : !kgen.scalar<bool>
  kgen.return %0: !kgen.scalar<bool>
}


// CHECK-LABEL: @string_ops
kgen.func @string_ops() -> index {
  %str = kgen.param.constant: string = <"four">
  // CHECK-DAG: kgen.param.constant = <4>
  %0 = pop.string.size %str
  %hello_world = kgen.param.constant: string = <"hello world">
  %world = kgen.param.constant: string = <" world">
  %empty_str = kgen.param.constant: string = <"">
  kgen.return %0 : index
}


// CHECK-LABEL: @offset_of_offset
kgen.func @offset_of_offset(%arg0: !kgen.pointer<index>) -> !kgen.pointer<index> {
  %idx3 = index.constant 3
  %0 = pop.offset %arg0[%idx3] : !kgen.pointer<index>
  %idx200 = index.constant 200
  // CHECK: %0 = pop.offset %arg0[%index203]
  %1 = pop.offset %0[%idx200] : !kgen.pointer<index>
  // CHECK: return %0
  kgen.return %1 : !kgen.pointer<index>
}

// CHECK-LABEL: @array_gep_offset
kgen.func @array_gep_offset(%arg1: !pop.array<4, index>, %arg2: index) -> index {
  // CHECK: %0 = pop.stack_allocation 1 x array<4, index>
  %0 = kgen.param.constant = <0>
  %array = pop.stack_allocation 1 x array<4, index>
  // CHECK-NEXT: pop.store %arg0, %0
  pop.store %arg1, %array : !kgen.pointer<array<4, index>>
  %gep = pop.array.gep %array[%0] : <array<4, index>>
  // Fold gep+offset into just gep.
  // CHECK: %1 = pop.array.gep %0[%arg1]
  // CHECK: %2 = pop.load %1
  %offset = pop.offset %gep[%arg2] : !kgen.pointer<index>
  %load = pop.load %offset : !kgen.pointer<index>
  kgen.return %load : index
}

// CHECK-LABEL: @large_int_memory_leak
// COM: Ensure that memory is correctly freed from a SIMDAttr.
kgen.func @large_int_memory_leak() -> !kgen.scalar<si128> {
  // CHECK: constant: scalar<si128> = <1234>
  %0 = kgen.param.constant: si128 = <1234>
  %1 = pop.cast_from_builtin %0 : si128 to !kgen.scalar<si128>
  kgen.return %1 : !kgen.scalar<si128>
}

// CHECK-LABEL: kgen.func @select_true_false
kgen.func @select_true_false(%arg0: !kgen.scalar<bool>) -> !kgen.scalar<bool> {
  // CHECK-NEXT: return %arg0 : !kgen.scalar<bool>
  %0 = kgen.param.constant: scalar<bool> = <true>
  %1 = kgen.param.constant: scalar<bool> = <false>
  %2 = pop.select %arg0, %0, %1 : !kgen.scalar<bool>
  kgen.return %2 : !kgen.scalar<bool>
}

// CHECK-LABEL: kgen.func @select_of_select
kgen.func @select_of_select(%arg0: !kgen.scalar<bool>, %arg1: index, %arg2: index, %arg3: index) -> (index, index) {
  // CHECK-NEXT: %0 = pop.select %arg0, %arg1, %arg3
  %0 = pop.select %arg0, %arg1, %arg2 : index
  %1 = pop.select %arg0, %0, %arg3 : index
  // CHECK-NEXT: %1 = pop.select %arg0, %arg1, %arg3
  %2 = pop.select %arg0, %arg2, %arg3 : index
  %3 = pop.select %arg0, %arg1, %2 : index
  // CHECK-NEXT: return %0, %1
  kgen.return %1, %3 : index, index
}

// CHECK-LABEL: kgen.func @lifetime_markers
kgen.func @lifetime_markers() {
  pop.stack_alloc.lifetime.start()
  pop.stack_alloc.lifetime.end()
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @load_bitcast
kgen.func @load_bitcast(%arg0: !kgen.pointer<pointer<none>>) -> !kgen.pointer<index> {
  // CHECK-NEXT: %0 = pop.load %arg0 : !kgen.pointer<pointer<none>>
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<pointer<none>> to !kgen.pointer<pointer<index>>
  // CHECK-NEXT: %1 = pop.pointer.bitcast %0 : !kgen.pointer<none> to !kgen.pointer<index>
  %1 = pop.load %0 : !kgen.pointer<pointer<index>>
  // CHECK-NEXT: return %1
  kgen.return %1 : !kgen.pointer<index>
}

// CHECK-LABEL: @load_bitcast_ptr_ptr
kgen.func @load_bitcast_ptr_ptr(%arg0: !kgen.pointer<none>) -> !kgen.pointer<none> {
  // CHECK-NEXT: %0 = pop.pointer.bitcast %arg0
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<pointer<none>>
  // CHECK-NEXT: pop.load %0
  %1 = pop.load %0 : !kgen.pointer<pointer<none>>
  kgen.return %1 : !kgen.pointer<none>
}

// CHECK-LABEL: @load_bitcast_func_ptr
kgen.func @load_bitcast_func_ptr(%arg0: !kgen.generator<() -> ()>) ->
                                  (!kgen.pointer<index>, !kgen.pointer<index>, !kgen.pointer<index>, !kgen.pointer<index>) {
  // CHECK-NEXT: %0 = pop.pointer.bitcast %arg0
  %0 = pop.pointer.bitcast %arg0 : !kgen.generator<() -> ()> to !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.load %0
  %1 = pop.load %0 : !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.load volatile %0
  %2 = pop.load volatile<1> %0 : !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.load volatile invariant %0
  %3 = pop.load volatile<1> invariant<1> %0 : !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.load atomic acquire %0
  %4 = pop.load atomic acquire %0 : !kgen.pointer<pointer<index>>
  kgen.return %1, %2, %3, %4 : !kgen.pointer<index>, !kgen.pointer<index>, !kgen.pointer<index>, !kgen.pointer<index>
}

// CHECK-LABEL: @load_of_store
kgen.func @load_of_store(%arg0: !kgen.pointer<index>, %arg1: index) -> index {
  // CHECK-NEXT: pop.store %arg1, %arg0
  pop.store %arg1, %arg0 : !kgen.pointer<index>
  %1 = pop.load %arg0 : !kgen.pointer<index>
  // CHECK-NEXT: return %arg1
  kgen.return %1 : index
}

// CHECK-LABEL: @load_of_store_atomic
// CHECK-SAME: [[PTR:%[a-z0-9]*]]:
// CHECK-SAME: [[VAL:%[a-z0-9]*]]:
kgen.func @load_of_store_atomic(%arg0: !kgen.pointer<index>, %arg1: index) -> index {
  // CHECK-NEXT: pop.store atomic release [[VAL]], [[PTR]]
  pop.store atomic release %arg1, %arg0 : !kgen.pointer<index>
  // CHECK-NEXT: pop.load atomic acquire [[PTR]]
  %1 = pop.load atomic acquire %arg0 : !kgen.pointer<index>
  kgen.return %1 : index
}

// CHECK-LABEL: @atomic_store_not_canonicalized
// CHECK-SAME: [[PTR:%[a-z0-9]*]]:
// CHECK-SAME: [[VAL:%[a-z0-9]*]]:
kgen.func @atomic_store_not_canonicalized(%p: !kgen.pointer<scalar<f32>>, %v: !kgen.scalar<f32>) {
  // CHECK: pop.store atomic release [[VAL]], [[PTR]]
  pop.store atomic release %v, %p : !kgen.pointer<scalar<f32>>
  // CHECK: pop.store atomic release [[VAL]], [[PTR]]
  pop.store atomic release %v, %p : !kgen.pointer<scalar<f32>>
  kgen.return
}

// CHECK-LABEL: @store_unknown
kgen.func @store_unknown(%ptr : !kgen.pointer<array<4, index>>) {
  %array = kgen.param.constant: array<4, index> = <#interp.uninitmem>
  pop.store %array, %ptr align<1> : !kgen.pointer<array<4, index>>

  // CHECK-NEXT: kgen.return
  kgen.return
}

// CHECK-LABEL: @store_bitcast
kgen.func @store_bitcast(%arg0: !kgen.pointer<index>, %arg1: !kgen.pointer<pointer<none>>) {
  // CHECK-NEXT: %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<index> to !kgen.pointer<none>
  %0 = pop.pointer.bitcast %arg1 : !kgen.pointer<pointer<none>> to !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.store %0, %arg1 : !kgen.pointer<pointer<none>>
  pop.store %arg0, %0 : !kgen.pointer<pointer<index>>
  // CHECK-NEXT: %1 = pop.pointer.bitcast %arg0 : !kgen.pointer<index> to !kgen.pointer<none>
  // CHECK-NEXT: pop.store volatile %1, %arg1 : !kgen.pointer<pointer<none>>
  %1 = pop.pointer.bitcast %arg0 : !kgen.pointer<index> to !kgen.pointer<none>
  pop.store volatile<1> %1, %arg1 : !kgen.pointer<pointer<none>>
  kgen.return
}

// CHECK-LABEL: @bitcast_free
kgen.func @bitcast_free(%arg0: !kgen.pointer<none>) {
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<index>
  // CHECK-NEXT: pop.aligned_free %arg0
  pop.aligned_free %0 : <index>
  kgen.return
}

// CHECK-LABEL: @variant_bitcast
kgen.func @variant_bitcast() -> !kgen.pointer<i32> {
  // CHECK: constant: pointer<i32> = <0>
  %0 = kgen.param.constant: pointer<variant<i32, i64>> = <0>
  %1 = pop.variant.bitcast %0, <0> : <variant<i32, i64>> as <i32>
  kgen.return %1 : !kgen.pointer<i32>
}

// CHECK-LABEL: @union_bitcast
kgen.func @union_bitcast() -> !kgen.pointer<i32> {
  %0 = kgen.param.constant: pointer<union<i32>> = <0>
  // CHECK-NEXT: constant: pointer<i32> = <0>
  %1 = pop.union.bitcast %0 : <union<i32>> as <i32>
  kgen.return %1 : !kgen.pointer<i32>
}

// CHECK-LABEL: @union_wrap
kgen.func @union_wrap() -> !pop.union<i32> {
  %0 = kgen.param.constant: i32 = <42>
  // CHECK-NEXT: constant: union<i32> = <{:i32 42}>
  %1 = pop.union.wrap %0 : i32 as <i32>
  kgen.return %1 : !pop.union<i32>
}

// CHECK-LABEL: @union_unwrap
kgen.func @union_unwrap() -> i32 {
  %0 = kgen.param.constant: union<i32> = <{:i32 42}>
  // CHECK-NEXT: constant: i32 = <42>
  %1 = pop.union.unwrap %0 : <i32> as i32
  kgen.return %1 : i32
}

kgen.func @union_unwrap_type() -> i64 {
  %0 = kgen.param.constant: union<i32, i64> = <{:i32 42}>
  // CHECK: pop.union.unwrap
  %1 = pop.union.unwrap %0 : <i32, i64> as i64
  kgen.return %1 : i64
}

// CHECK-LABEL: @wrap_unwrap
kgen.func @wrap_unwrap(%arg0: !pop.union<i32, i64>) -> !pop.union<i32, i64> {
  %0 = pop.union.unwrap %arg0 : <i32, i64> as i64
  %1 = pop.union.wrap %0 : i64 as <i32, i64>
  // CHECK-NEXT: return %arg0
  kgen.return %1 : !pop.union<i32, i64>
}

// CHECK-LABEL: @wrap_unwrap_type
kgen.func @wrap_unwrap_type(%arg0: !pop.union<i32>) -> !pop.union<i32, i64> {
  // CHECK-NEXT: pop.union.unwrap
  %0 = pop.union.unwrap %arg0 : <i32> as i32
  %1 = pop.union.wrap %0 : i32 as <i32, i64>
  kgen.return %1 : !pop.union<i32, i64>
}

// CHECK-LABEL: @unwrap_wrap
kgen.func @unwrap_wrap(%arg0: i32) -> i32 {
  %0 = pop.union.wrap %arg0 : i32 as <i32, i64>
  %1 = pop.union.unwrap %0 : <i32, i64> as i32
  // CHECK-NEXT: return %arg0
  kgen.return %1 : i32
}

// CHECK-LABEL: @unwrap_wrap_type
kgen.func @unwrap_wrap_type(%arg0: i32) -> i64 {
  // CHECK-NEXT: pop.union.wrap
  %0 = pop.union.wrap %arg0 : i32 as <i32, i64>
  %1 = pop.union.unwrap %0 : <i32, i64> as  i64
  kgen.return %1 : i64
}


// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="e-p:32:32">} {
  // CHECK-LABEL: @cast_const_ui8_to_index_folding
  kgen.func @cast_const_ui8_to_index_folding() -> !kgen.scalar<index> {
    // CHECK-NEXT: kgen.param.constant: scalar<index> = <128>
    %in = kgen.param.constant: scalar<ui8> = <128>
    %out = pop.cast %in : !kgen.scalar<ui8> to !kgen.scalar<index>
    kgen.return %out : !kgen.scalar<index>
  }

  // CHECK-LABEL: @cast_si64_ui64_index
  kgen.func @cast_si64_ui64_index(%in : !kgen.scalar<si64>) -> !kgen.scalar<index> {
    // CHECK-NEXT: pop.cast {{.*}} : !kgen.scalar<si64> to !kgen.scalar<index>
    %1 = pop.cast %in : !kgen.scalar<si64> to !kgen.scalar<ui64>
    %2 = pop.cast %1 : !kgen.scalar<ui64> to !kgen.scalar<index>

    kgen.return %2 : !kgen.scalar<index>
  }

  // CHECK-LABEL: @cast_si64_index_f64
  kgen.func @cast_si64_index_f64(%in : !kgen.scalar<si64>) -> !kgen.scalar<f64> {
    // CHECK-NEXT:pop.cast {{.*}} : !kgen.scalar<si64> to !kgen.scalar<index>
    %1 = pop.cast %in : !kgen.scalar<si64> to !kgen.scalar<index>
    // CHECK-NEXT:pop.cast {{.*}} : !kgen.scalar<index> to !kgen.scalar<f64>
    %2 = pop.cast %1 : !kgen.scalar<index> to !kgen.scalar<f64>

    kgen.return %2 : !kgen.scalar<f64>
  }

  // CHECK-LABEL: @cast_si32_index_f64
  kgen.func @cast_si32_index_f64(%in : !kgen.scalar<si32>) -> !kgen.scalar<f64> {
    // CHECK-NEXT:pop.cast {{.*}} : !kgen.scalar<si32> to !kgen.scalar<index>
    %1 = pop.cast %in : !kgen.scalar<si32> to !kgen.scalar<index>
    // CHECK-NEXT:pop.cast {{.*}} : !kgen.scalar<index> to !kgen.scalar<f64>
    %2 = pop.cast %1 : !kgen.scalar<index> to !kgen.scalar<f64>

    kgen.return %2 : !kgen.scalar<f64>
  }

  // CHECK-LABEL: @cast_index_ui32_f64
  kgen.func @cast_index_ui32_f64(%in : !kgen.scalar<index>) -> !kgen.scalar<f64> {
    // CHECK-NEXT:pop.cast {{.*}} : !kgen.scalar<index> to !kgen.scalar<ui32>
    %1 = pop.cast %in : !kgen.scalar<index> to !kgen.scalar<ui32>
    // CHECK-NEXT:pop.cast {{.*}} : !kgen.scalar<ui32> to !kgen.scalar<f64>
    %2 = pop.cast %1 : !kgen.scalar<ui32> to !kgen.scalar<f64>

    kgen.return %2 : !kgen.scalar<f64>
  }

  // CHECK-LABEL: @bitcast_index
  kgen.func @bitcast_index() -> (!kgen.simd<2, ui32>, !kgen.simd<2, index>,
                                 !kgen.simd<2, index>, !kgen.simd<2, f32>) {
    %ui32_const = kgen.param.constant: simd<2, ui32> = <<1, 2>>
    %f32_const = kgen.param.constant: simd<2, f32> = <<"1.5", "2.5">>
    %index_const = kgen.param.constant: simd<2, index> = <<3, 4>>

    %2 = pop.bitcast %index_const : !kgen.simd<2, index> to !kgen.simd<2, ui32>
    %3 = pop.bitcast %ui32_const : !kgen.simd<2, ui32> to !kgen.simd<2, index>
    %4 = pop.bitcast %f32_const : !kgen.simd<2, f32> to !kgen.simd<2, index>
    %5 = pop.bitcast %index_const : !kgen.simd<2, index> to !kgen.simd<2, f32>
    // CHECK: [[SIMD_UI32:%.*]] = kgen.param.constant: simd<2, ui32> = <<3, 4>>
    // CHECK-NEXT: [[SIMD_INDEX:%.*]] = kgen.param.constant: simd<2, index> = <<1, 2>>
    // CHECK-NEXT: [[SIMD_INDEX_F32:%.*]] = kgen.param.constant: simd<2, index> = <<1069547520, 1075838976>>
    // CHECK-NEXT: [[SIMD_F32:%.*]] = kgen.param.constant: simd<2, f32> = <<"4.20389539E-45", "5.60519386E-45">>
    // CHECK: kgen.return [[SIMD_UI32]], [[SIMD_INDEX]], [[SIMD_INDEX_F32]], [[SIMD_F32]]
    kgen.return %2, %3, %4, %5 : !kgen.simd<2, ui32>, !kgen.simd<2, index>,
                                 !kgen.simd<2, index>, !kgen.simd<2, f32>
  }

  // CHECK-LABEL: @shl_index
  kgen.func @shl_index() -> (!kgen.scalar<index>, !kgen.scalar<index>) {
    %neg_one = kgen.param.constant: scalar<index> = <-1>
    %one = kgen.param.constant: scalar<index> = <1>
    %_33 = kgen.param.constant: scalar<index> = <33>
    // CHECK-DAG: [[N1:%.*]] = kgen.param.constant: scalar<index> = <-1>
    // CHECK-DAG: [[S1:%.*]] = kgen.param.constant: scalar<index> = <1>
    // CHECK-DAG: [[S33:%.*]] = kgen.param.constant: scalar<index> = <33>
    // Cannot fold these as shifting more than LHS' bit width is UB
    // CHECK: [[T0:%.*]] = pop.shl [[N1]], [[S33]] : !kgen.scalar<index>
    %res0 = pop.shl %neg_one, %_33 : !kgen.scalar<index>
    // CHECK: [[T1:%.*]] = pop.shl [[S1]], [[S33]] : !kgen.scalar<index>
    %res1 = pop.shl %one, %_33 : !kgen.scalar<index>

    // CHECK: kgen.return [[T0]], [[T1]]
    kgen.return %res0, %res1 : !kgen.scalar<index>, !kgen.scalar<index>
  }

  // CHECK-LABEL: @shr_index
  kgen.func @shr_index() -> (!kgen.scalar<index>, !kgen.scalar<index>) {
    // CHECK-DAG: [[N1:%.*]] = kgen.param.constant: scalar<index> = <-1>
    // CHECK-DAG: [[S63:%.*]] = kgen.param.constant: scalar<index> = <63>
    %neg1 = kgen.param.constant: scalar<index> = <-1>
    %63 = kgen.param.constant: scalar<index> = <63>
    %2 = kgen.param.constant: scalar<index> = <2>
    %neg16 = kgen.param.constant: scalar<index> = <-16>
    // Cannot fold this as shifting more than LHS' bit width is UB
    // CHECK-DAG: %[[RES:.*]] = pop.shr [[N1]], [[S63]] : !kgen.scalar<index>
    %res0 = pop.shr %neg1, %63 : !kgen.scalar<index>
    // CHECK-DAG: %[[RES1:.*]] = kgen.param.constant: scalar<index> = <-4>
    %res1 = pop.shr %neg16, %2 : !kgen.scalar<index>

    // CHECK: kgen.return %[[RES]], %[[RES1]]
    kgen.return %res0, %res1 : !kgen.scalar<index>, !kgen.scalar<index>
  }

  // CHECK-LABEL: @index_div
  kgen.func @index_div() -> (!kgen.scalar<index>, !kgen.scalar<index>) {
    %0 = kgen.param.constant: scalar<index> = <1073741826>
    %1 = kgen.param.constant: scalar<index> = <268435457>
    %2 = kgen.param.constant: scalar<index> = <2>

    // CHECK: %[[RES:.*]] = kgen.param.constant: scalar<index> = <536870913>
    %r0 = pop.div %0, %2 : !kgen.scalar<index>
    // CHECK: %[[RES1:.*]] = kgen.param.constant: scalar<index> = <134217728>
    %r1 = pop.div %1, %2 : !kgen.scalar<index>

    // CHECK: kgen.return %[[RES]], %[[RES1]] : !kgen.scalar<index>, !kgen.scalar<index>
    kgen.return %r0, %r1 : !kgen.scalar<index>, !kgen.scalar<index>
  }

  // CHECK-LABEL: @uindex_shr
  kgen.func @uindex_shr() -> (!kgen.scalar<uindex>) {
    %0 = kgen.param.constant: scalar<uindex> = <2147483648>
    %1 = kgen.param.constant: scalar<uindex> = <2>

    // CHECK: %[[RES:.*]] = kgen.param.constant: scalar<uindex> = <536870912>
    %r0 = pop.shr %0, %1 : !kgen.scalar<uindex>

    // CHECK: kgen.return %[[RES]] : !kgen.scalar<uindex>
    kgen.return %r0 : !kgen.scalar<uindex>
  }

  // CHECK-LABEL: @uindex_div
  kgen.func @uindex_div() -> !kgen.scalar<uindex> {
    %0 = kgen.param.constant: scalar<uindex> = <4294967295>
    %1 = kgen.param.constant: scalar<uindex> = <10>

    // CHECK: %[[RES:.*]] = kgen.param.constant: scalar<uindex> = <429496729>
    %6 = pop.div %0, %1 : !kgen.scalar<uindex>
    // CHECK: kgen.return %[[RES]] : !kgen.scalar<uindex>
    kgen.return %6 : !kgen.scalar<uindex>
  }

  // CHECK-LABEL: @uindex_rem
  kgen.func @uindex_rem() -> !kgen.scalar<uindex> {
    %0 = kgen.param.constant: scalar<uindex> = <4294967295>
    %1 = kgen.param.constant: scalar<uindex> = <10>

    // CHECK: %[[RES:.*]] = kgen.param.constant: scalar<uindex> = <5>
    %6 = pop.rem %0, %1 : !kgen.scalar<uindex>
    // CHECK: kgen.return %[[RES]] : !kgen.scalar<uindex>
    kgen.return %6 : !kgen.scalar<uindex>
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="e-p:64:64">} {
  // CHECK-LABEL: @cast_const_ui8_to_index_folding
  kgen.func @cast_const_ui8_to_index_folding() -> !kgen.scalar<index> {
    // CHECK-NEXT: kgen.param.constant: scalar<index> = <128>
    %in = kgen.param.constant: scalar<ui8> = <128>
    %out = pop.cast %in : !kgen.scalar<ui8> to !kgen.scalar<index>
    kgen.return %out : !kgen.scalar<index>
  }

  // CHECK-LABEL: @cast_si64_ui64_index
  kgen.func @cast_si64_ui64_index(%in : !kgen.scalar<si64>) -> !kgen.scalar<index> {
    // CHECK-NEXT: pop.cast {{.*}} : !kgen.scalar<si64> to !kgen.scalar<index>
    %1 = pop.cast %in : !kgen.scalar<si64> to !kgen.scalar<ui64>
    %2 = pop.cast %1 : !kgen.scalar<ui64> to !kgen.scalar<index>

    kgen.return %2 : !kgen.scalar<index>
  }

  // CHECK-LABEL: @cast_si64_index_f64
  kgen.func @cast_si64_index_f64(%in : !kgen.scalar<si64>) -> !kgen.scalar<f64> {
    // CHECK-NEXT:pop.cast {{.*}} : !kgen.scalar<si64> to !kgen.scalar<index>
    %1 = pop.cast %in : !kgen.scalar<si64> to !kgen.scalar<index>
    // CHECK-NEXT:pop.cast {{.*}} : !kgen.scalar<index> to !kgen.scalar<f64>
    %2 = pop.cast %1 : !kgen.scalar<index> to !kgen.scalar<f64>

    kgen.return %2 : !kgen.scalar<f64>
  }

  // CHECK-LABEL: @cast_si32_index_f64
  kgen.func @cast_si32_index_f64(%in : !kgen.scalar<si32>) -> !kgen.scalar<f64> {
    // CHECK-NEXT:pop.cast {{.*}} : !kgen.scalar<si32> to !kgen.scalar<index>
    %1 = pop.cast %in : !kgen.scalar<si32> to !kgen.scalar<index>
    // CHECK-NEXT:pop.cast {{.*}} : !kgen.scalar<index> to !kgen.scalar<f64>
    %2 = pop.cast %1 : !kgen.scalar<index> to !kgen.scalar<f64>

    kgen.return %2 : !kgen.scalar<f64>
  }

  // CHECK-LABEL: @cast_index_ui32_f64
  kgen.func @cast_index_ui32_f64(%in : !kgen.scalar<index>) -> !kgen.scalar<f64> {
    // CHECK-NEXT:pop.cast {{.*}} : !kgen.scalar<index> to !kgen.scalar<ui32>
    %1 = pop.cast %in : !kgen.scalar<index> to !kgen.scalar<ui32>
    // CHECK-NEXT:pop.cast {{.*}} : !kgen.scalar<ui32> to !kgen.scalar<f64>
    %2 = pop.cast %1 : !kgen.scalar<ui32> to !kgen.scalar<f64>

    kgen.return %2 : !kgen.scalar<f64>
  }

  // CHECK-LABEL: @shl_index
  kgen.func @shl_index() -> (!kgen.scalar<index>, !kgen.scalar<index>) {
    %neg_one = kgen.param.constant: scalar<index> = <-1>
    %one = kgen.param.constant: scalar<index> = <1>
    %_33 = kgen.param.constant: scalar<index> = <33>
    // CHECK: %[[RES0:.*]] = kgen.param.constant: scalar<index> = <-8589934592>
    %res0 = pop.shl %neg_one, %_33 : !kgen.scalar<index>
    // CHECK: %[[RES1:.*]] = kgen.param.constant: scalar<index> = <8589934592>
    %res1 = pop.shl %one, %_33 : !kgen.scalar<index>

    // CHECK: kgen.return %[[RES0]], %[[RES1]]
    kgen.return %res0, %res1 : !kgen.scalar<index>, !kgen.scalar<index>
  }

  // CHECK-LABEL: @shr_index
  kgen.func @shr_index() -> (!kgen.scalar<index>, !kgen.scalar<index>) {
    %neg1 = kgen.param.constant: scalar<index> = <-1>
    %63 = kgen.param.constant: scalar<index> = <63>
    %2 = kgen.param.constant: scalar<index> = <2>
    %neg16 = kgen.param.constant: scalar<index> = <-16>
    // CHECK: %[[RES:.*]] = kgen.param.constant: scalar<index> = <-1>
    %res0 = pop.shr %neg1, %63 : !kgen.scalar<index>
    // CHECK: %[[RES1:.*]] = kgen.param.constant: scalar<index> = <-4>
    %res1 = pop.shr %neg16, %2 : !kgen.scalar<index>

    // CHECK: kgen.return %[[RES]], %[[RES1]]
    kgen.return %res0, %res1 : !kgen.scalar<index>, !kgen.scalar<index>
  }

  // CHECK-LABEL: @index_div
  kgen.func @index_div() -> (!kgen.scalar<index>, !kgen.scalar<index>) {
    %0 = kgen.param.constant: scalar<index> = <8589934594>
    %1 = kgen.param.constant: scalar<index> = <4294967298>
    %2 = kgen.param.constant: scalar<index> = <2>

    // CHECK: %[[RES:.*]] = kgen.param.constant: scalar<index> = <4294967297>
    %r0 = pop.div %0, %2 : !kgen.scalar<index>
    // CHECK: %[[RES1:.*]] = kgen.param.constant: scalar<index> = <2147483649>
    %r1 = pop.div %1, %2 : !kgen.scalar<index>

    // CHECK: kgen.return %[[RES]], %[[RES1]] : !kgen.scalar<index>, !kgen.scalar<index>
    kgen.return %r0, %r1 : !kgen.scalar<index>, !kgen.scalar<index>
  }

  // CHECK-LABEL: @uindex_shr
  kgen.func @uindex_shr() -> (!kgen.scalar<uindex>) {
    %0 = kgen.param.constant: scalar<uindex> = <9223372036854775808>
    %1 = kgen.param.constant: scalar<uindex> = <2>

    // CHECK: %[[RES:.*]] = kgen.param.constant: scalar<uindex> = <2305843009213693952>
    %r0 = pop.shr %0, %1 : !kgen.scalar<uindex>

    // CHECK: kgen.return %[[RES]] : !kgen.scalar<uindex>
    kgen.return %r0 : !kgen.scalar<uindex>
  }

  // CHECK-LABEL: @uindex_div
  kgen.func @uindex_div() -> !kgen.scalar<uindex> {
    %0 = kgen.param.constant: scalar<uindex> = <18446744073709551615>
    %1 = kgen.param.constant: scalar<uindex> = <10>

    // CHECK: %[[RES:.*]] = kgen.param.constant: scalar<uindex> = <1844674407370955161>
    %6 = pop.div %0, %1 : !kgen.scalar<uindex>
    // CHECK: kgen.return %[[RES]] : !kgen.scalar<uindex>
    kgen.return %6 : !kgen.scalar<uindex>
  }

  // CHECK-LABEL: @uindex_rem
  kgen.func @uindex_rem() -> !kgen.scalar<uindex> {
    %0 = kgen.param.constant: scalar<uindex> = <18446744073709551615>
    %1 = kgen.param.constant: scalar<uindex> = <10>

    // CHECK: %[[RES:.*]] = kgen.param.constant: scalar<uindex> = <5>
    %6 = pop.rem %0, %1 : !kgen.scalar<uindex>
    // CHECK: kgen.return %[[RES]] : !kgen.scalar<uindex>
    kgen.return %6 : !kgen.scalar<uindex>
  }
}

// -----

// COM: Index folds go through the same path as integer folds. We just need to
// check that ops can fold for index dtypes and do not fold when the results
// don't match.
kgen.func @index_folds() -> (!kgen.scalar<index>, !kgen.scalar<index>) {
  // differ between 64-bit and 32-bit arithmetic.

  %0 = kgen.param.constant: scalar<index> = <8589934594>
  // CHECK-DAG: %[[C1:.*]] = kgen.param.constant: scalar<index> = <4294967298>
  %1 = kgen.param.constant: scalar<index> = <4294967298>
  // CHECK-DAG: %[[C2:.*]] = kgen.param.constant: scalar<index> = <1>
  %2 = kgen.param.constant: scalar<index> = <1>

  // CHECK-DAG: %[[RES:.*]] = kgen.param.constant: scalar<index> = <4294967297>
  %3 = pop.shr %0, %2 : !kgen.scalar<index>
  // CHECK: %[[RES1:.*]] = pop.shr %[[C1]], %[[C2]] : !kgen.scalar<index>
  %4 = pop.shr %1, %2 : !kgen.scalar<index>
  // CHECK: kgen.return %[[RES]], %[[RES1]] : !kgen.scalar<index>, !kgen.scalar<index>
  kgen.return %3, %4 : !kgen.scalar<index>, !kgen.scalar<index>
}

// -----

// Test that comparing two constant uindex values doesn't crash when the index
// bit width is unknown. Without target_info, foldSIMDOpResult can't fold when
// 32-bit and 64-bit results might differ (values >= 2^32), so the code falls
// through to the unsigned comparison optimization. We verify it doesn't assert
// and correctly returns without folding.
// CHECK-LABEL: @cmp_uindex_both_const_no_target_info
kgen.func @cmp_uindex_both_const_no_target_info() -> (!kgen.scalar<bool>, !kgen.scalar<bool>) {
  // Use a value that truncates to 0 in 32-bit (e.g. 2^32), then compare it to 1,
  // so that comparisons differ between 32-bit (0 < 1 = true) and 64-bit (0x100000000 < 1 = false)
  %large = kgen.param.constant: scalar<uindex> = <4294967296>
  %one = kgen.param.constant: scalar<uindex> = <1>
  // CHECK-DAG: %[[RES0:.*]] = pop.cmp ge
  %0 = pop.cmp ge(%large, %one) : !kgen.scalar<uindex>
  // CHECK-DAG: %[[RES1:.*]] = pop.cmp lt
  %1 = pop.cmp lt(%large, %one) : !kgen.scalar<uindex>
  // CHECK-NEXT: kgen.return %[[RES0]], %[[RES1]]
  kgen.return %0, %1 : !kgen.scalar<bool>, !kgen.scalar<bool>
}

// -----

// A `pop.load` of a bitcast from a struct pointer to a pointer to the struct's
// leading (offset-zero, nested) element is rewritten into a `kgen.struct.gep`
// chain feeding the load, dropping the opaque cast so SROA/mem2reg can
// decompose the allocation.
// CHECK-LABEL: @load_bitcast_leading_struct_element
// CHECK-SAME: (%[[P:.*]]: !kgen.pointer<struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>>)
kgen.func @load_bitcast_leading_struct_element(
    %p: !kgen.pointer<struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>>)
    -> !kgen.pointer<none> {
  // CHECK-NOT: pop.pointer.bitcast
  // CHECK: %[[G0:.*]] = kgen.struct.gep %[[P]][0]
  // CHECK: %[[G1:.*]] = kgen.struct.gep %[[G0]][0]
  // CHECK: %[[G2:.*]] = kgen.struct.gep %[[G1]][0]
  // CHECK: %[[L:.*]] = pop.load %[[G2]]
  // CHECK: kgen.return %[[L]]
  %cast = pop.pointer.bitcast %p : !kgen.pointer<struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>> to !kgen.pointer<pointer<none>>
  %load = pop.load %cast : !kgen.pointer<pointer<none>>
  kgen.return %load : !kgen.pointer<none>
}

// -----

// A bitcast to a type that is not the leading element is a real
// reinterpretation and must be left alone.
// CHECK-LABEL: @load_bitcast_not_leading
kgen.func @load_bitcast_not_leading(%p: !kgen.pointer<struct<(index, index)>>)
    -> !kgen.scalar<si64> {
  // CHECK: pop.pointer.bitcast
  %cast = pop.pointer.bitcast %p : !kgen.pointer<struct<(index, index)>> to !kgen.pointer<scalar<si64>>
  %load = pop.load %cast : !kgen.pointer<scalar<si64>>
  kgen.return %load : !kgen.scalar<si64>
}

// -----

// A bitcast that also changes address space is not a pure type pun and must be
// left alone.
// CHECK-LABEL: @load_bitcast_addrspace
kgen.func @load_bitcast_addrspace(%p: !kgen.pointer<struct<(index, index)>>)
    -> index {
  // CHECK: pop.pointer.bitcast
  %cast = pop.pointer.bitcast %p : !kgen.pointer<struct<(index, index)>> to !kgen.pointer<index, 3>
  %load = pop.load %cast : !kgen.pointer<index, 3>
  kgen.return %load : index
}

// -----

// Volatile (and atomic) loads are left untouched: the leading-element rewrite
// bails on them, so the bitcast and the volatile load both survive.
// CHECK-LABEL: @load_bitcast_leading_volatile
kgen.func @load_bitcast_leading_volatile(
    %p: !kgen.pointer<struct<(pointer<none>)>>) -> !kgen.pointer<none> {
  // CHECK-NOT: kgen.struct.gep
  // CHECK: pop.pointer.bitcast
  // CHECK: pop.load volatile
  %cast = pop.pointer.bitcast %p : !kgen.pointer<struct<(pointer<none>)>> to !kgen.pointer<pointer<none>>
  %load = pop.load volatile<1> %cast : !kgen.pointer<pointer<none>>
  kgen.return %load : !kgen.pointer<none>
}

// -----

// Single-level leading struct element: load(bitcast) -> load(struct.gep[0]).
// CHECK-LABEL: @load_bitcast_leading_single
kgen.func @load_bitcast_leading_single(
    %p: !kgen.pointer<struct<(pointer<none>)>>) -> !kgen.pointer<none> {
  // CHECK-NOT: pop.pointer.bitcast
  // CHECK: %[[G:.*]] = kgen.struct.gep %{{.*}}[0]
  // CHECK: pop.load %[[G]]
  %c = pop.pointer.bitcast %p : !kgen.pointer<struct<(pointer<none>)>> to !kgen.pointer<pointer<none>>
  %l = pop.load %c : !kgen.pointer<pointer<none>>
  kgen.return %l : !kgen.pointer<none>
}

// -----

// Leading element two struct levels down (field 0 at each level).
// CHECK-LABEL: @load_bitcast_leading_two_level
kgen.func @load_bitcast_leading_two_level(
    %p: !kgen.pointer<struct<(struct<(scalar<f32>, scalar<f64>)>)>>)
    -> !kgen.scalar<f32> {
  // CHECK-NOT: pop.pointer.bitcast
  // CHECK: %[[G0:.*]] = kgen.struct.gep %{{.*}}[0]
  // CHECK: %[[G1:.*]] = kgen.struct.gep %[[G0]][0]
  // CHECK: pop.load %[[G1]]
  %c = pop.pointer.bitcast %p : !kgen.pointer<struct<(struct<(scalar<f32>, scalar<f64>)>)>> to !kgen.pointer<scalar<f32>>
  %l = pop.load %c : !kgen.pointer<scalar<f32>>
  kgen.return %l : !kgen.scalar<f32>
}

// -----

// A non-leading nested field (index 1) is a real reinterpretation: keep the
// cast.
// CHECK-LABEL: @load_bitcast_deeper_non_leading
kgen.func @load_bitcast_deeper_non_leading(
    %p: !kgen.pointer<struct<(struct<(scalar<f32>, scalar<f64>)>)>>)
    -> !kgen.scalar<f64> {
  // CHECK: pop.pointer.bitcast
  %c = pop.pointer.bitcast %p : !kgen.pointer<struct<(struct<(scalar<f32>, scalar<f64>)>)>> to !kgen.pointer<scalar<f64>>
  %l = pop.load %c : !kgen.pointer<scalar<f64>>
  kgen.return %l : !kgen.scalar<f64>
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="e-p:64:64", simd_bit_width = 128, index_bit_width = 32>} {

// A same-width T2 is never folded away, even when its sign matches T1's.
// CHECK-LABEL: @same_width_same_sign_does_not_fold
kgen.func @same_width_same_sign_does_not_fold(%v0 : !kgen.scalar<si32>) -> !kgen.scalar<si64> {
  // CHECK-NEXT: pop.cast %arg0 : !kgen.scalar<si32> to !kgen.scalar<index>
  // CHECK-NEXT: pop.cast {{.*}} : !kgen.scalar<index> to !kgen.scalar<si64>
  %v1 = pop.cast %v0 : !kgen.scalar<si32> to !kgen.scalar<index>
  %v2 = pop.cast %v1 : !kgen.scalar<index> to !kgen.scalar<si64>
  kgen.return %v2 : !kgen.scalar<si64>
}

// Same width but T2 is signed where T1 is not, so the sext must stay.
// CHECK-LABEL: @same_width_sign_differs_does_not_fold
kgen.func @same_width_sign_differs_does_not_fold(%v0 : !kgen.scalar<ui32>) -> !kgen.scalar<si64> {
  // CHECK-NEXT: pop.cast %arg0 : !kgen.scalar<ui32> to !kgen.scalar<index>
  // CHECK-NEXT: pop.cast {{.*}} : !kgen.scalar<index> to !kgen.scalar<si64>
  %v1 = pop.cast %v0 : !kgen.scalar<ui32> to !kgen.scalar<index>
  %v2 = pop.cast %v1 : !kgen.scalar<index> to !kgen.scalar<si64>
  kgen.return %v2 : !kgen.scalar<si64>
}

}
