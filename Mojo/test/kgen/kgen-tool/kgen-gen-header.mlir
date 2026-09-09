// RUN: kgen %s -emit=header -func="someKernel:f32(f32,index)" | FileCheck %s --check-prefixes=SCALAR
// RUN: kgen %s -emit=header -func="someMetaScalarKernel" | FileCheck %s --check-prefixes=SCALARMETA
// RUN: kgen %s -emit=header -func="litNoneKernel" | FileCheck %s --check-prefixes=VOID
// RUN: kgen %s -emit=header -func="oneElemStruct" | FileCheck %s --check-prefixes=ONESTRUCT
// RUN: kgen %s -emit=header -func="oneVariadic" | FileCheck %s --check-prefixes=ONEVARIADIC
// RUN: kgen %s -emit=header -func="twoVariadic" | FileCheck %s --check-prefixes=TWOVARIADIC

kgen.func export @someKernel(%arg1: f32, %arg2: index) cabi -> f32 {
  kgen.return %arg1 : f32
}
// SCALAR: extern float someKernel(float, ssize_t);

kgen.func export @someMetaScalarKernel(%arg0: !kgen.simd<1, f32>) cabi -> !kgen.simd<1, f32> {
  kgen.return %arg0 : !kgen.simd<1, f32>
}
// SCALARMETA: extern float someMetaScalarKernel(float);

kgen.func export @litNoneKernel() cabi -> !pop.array<0, i1> {
  %0 = kgen.param.constant: array<0, i1> = <[]>
  kgen.return %0 : !pop.array<0, i1>
}

// VOID: extern void litNoneKernel();

kgen.func export @oneElemStruct(%arg0: i32) cabi -> !kgen.struct<(i32)> {
  %0 = kgen.param.constant: struct<(i32)> = <{ 0 }>
  kgen.return %0 : !kgen.struct<(i32)>
}

// ONESTRUCT: extern int32_t oneElemStruct(int32_t);

kgen.func export @oneVariadic(%arg0: !kgen.param_list<f32>) cabi -> !kgen.struct<(i32)> {
  %0 = kgen.param.constant: struct<(i32)> = <{ 0 }>
  kgen.return %0 : !kgen.struct<(i32)>
}

// ONEVARIADIC: extern int32_t oneVariadic(void *, ssize_t);

kgen.func export @twoVariadic(%arg0: !kgen.param_list<!kgen.struct<(i32, i32)>>,
                              %arg1: !kgen.param_list<i32>) cabi -> !kgen.struct<(i32)> {
  %0 = kgen.param.constant: struct<(i32)> = <{ 0 }>
  kgen.return %0 : !kgen.struct<(i32)>
}

// TWOVARIADIC: extern int32_t twoVariadic(void *, ssize_t, void *, ssize_t);
