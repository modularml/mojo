// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt -emit-bytecode -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @simd_constants
kgen.generator @simd_constants<N, value: !kgen.simd<N, si32>>() {
  // CHECK: simd<2, f32> = <<"12.375", "77">>
  %0 = kgen.param.constant: simd<2, f32> = <<"12.375", "77">>
  // CHECK: scalar<si64> = <1234>
  %1 = kgen.param.constant: scalar<si64> = <1234>
  // CHECK: simd<2, bool> = <<true, false>>
  %2 = kgen.param.constant: simd<2, bool> = <<true, false>>
  // CHECK: scalar<f64> = <"0.01171875">
  %3 = kgen.param.constant: scalar<f64> = <"0.01171875">
  // CHECK: simd<6, ui2> = <<0, 1, 2, 3, 3, 2>>
  %4 = kgen.param.constant: simd<6, ui2> = <<0, 1, 2, 3, 3, 2>>
  // CHECK: simd<2, index> = <<-54321, 12345>>
  %5 = kgen.param.constant: simd<2, index> = <<-54321, 12345>>
  // CHECK: scalar<f32> = <"0.100000001">
  %6 = kgen.param.constant: scalar<f32> = <"0.1">


  // CHECK: scalar<f16> = <"1.7285E-5">
  kgen.param.constant: scalar<f16> = <"1.7285E-5">

  // CHECK: #kgen.simd<1, 2>
  "simd.const"() {a = #kgen.simd<1, 2> : !kgen.simd<2, si32>} : () -> ()
  // CHECK: #kgen<simd 1>
  "simd.const"() {a = #kgen<simd 1> : !kgen.simd<2, si32>} : () -> ()
  // CHECK: simd<N, si32> = <value>
  kgen.param.constant: simd<N, si32> = <value>
  kgen.return
}

// CHECK-LABEL: @array_constants
kgen.generator @array_constants<T: type, A: !kgen.param<T>>() {
  // CHECK: array<2, index> = <[1, 2]>
  kgen.param.constant: array<2, index> = <[1, 2]>
  // CHECK: array<2, dtype> = <[ui4, si4]>
  kgen.param.constant: array<2, dtype> = <[ui4, si4]>
  // CHECK: array<2, T> = <[A, A]>
  kgen.param.constant: array<2, T> = <[A, A]>
  kgen.return
}

// CHECK-LABEL: @variadic_constants
kgen.generator @variadic_constants<T: type, value: si32>() {
  // CHECK: param_list<si32> = <[1, value]>
  kgen.param.constant: param_list<si32> = <[1, value]>
  // CHECK: param_list<T> = <[]>
  kgen.param.constant: param_list<T> = <[]>
  kgen.return
}

// CHECK-LABEL: @union_constants
kgen.func @union_constants() {
  // CHECK: constant: union<i32, i64> = <{:i32 42}>
  kgen.param.constant: union<i32, i64> = <{:i32 42}>
  kgen.return
}

// CHECK: f0 = #pop.union<:i32 42> : !pop.union<i32, i64>
"union.attr"() { f0 = #pop.union<:i32 42> : !pop.union<i32, i64> } : () -> ()

// CHECK: f0 = #pop.fmf<none>
// CHECK: f1 = #pop.fmf<reassoc>
// CHECK: f2 = #pop.fmf<nnan|ninf|reassoc>
// CHECK: f3 = #pop.fmf<fast>
"enums.op"() {
  f0 = #pop.fmf<none>,
  f1 = #pop.fmf<reassoc>,
  f2 = #pop.fmf<reassoc|ninf|nnan>,
  f3 = #pop.fmf<fast>
} : () -> ()

// CHECK: f0 = #kgen.param.expr<and, #kgen<simd -1> : !kgen.simd<4, si32>, #kgen.unknown : !kgen.simd<4, si32>> : !kgen.simd<4, si32>
"simd_and.attr"() { f0 = #kgen.param.expr<and,#kgen.unknown : !kgen.simd<4, si32>, #kgen<simd -1> : !kgen.simd<4, si32>> : !kgen.simd<4, si32> } : () -> ()

// CHECK: f0 = #pop.cast<#kgen.unknown : !kgen.scalar<si32>> : !kgen.scalar<ui32>
"pop_cast.op"() { f0 = #pop.cast< #kgen.unknown : !kgen.scalar<si32>> : !kgen.scalar<ui32> } : () -> ()

// CHECK: f0 = #kgen.cast_from_builtin<#kgen.unknown : si64> : !kgen.scalar<si64>
"pop_cast_from_builtin.op"() { f0 = #kgen.cast_from_builtin<#kgen.unknown : si64> : !kgen.scalar<si64> } : () -> ()

// CHECK: f0 = #kgen.cast_to_builtin<#kgen.unknown : !kgen.scalar<f16>> : f16
"pop_cast_to_builtin.op"() { f0 = #kgen.cast_to_builtin<#kgen.unknown: !kgen.scalar<f16>> : f16 } : () -> ()

// CHECK: f0 = #kgen.simd_splat<#kgen.unknown : !kgen.scalar<f16>> : !kgen.simd<4, f16>
"pop_simd_splat.op"() { f0 = #kgen.simd_splat<#kgen.unknown : !kgen.scalar<f16>> : !kgen.simd<4, f16> } : () -> ()

// CHECK: f0 = #pop.dtype_to_ui8<*?> : ui8
"pop_dtype_to_ui8.op"() { f0 = #pop.dtype_to_ui8<*?> : ui8 } : () -> ()

// CHECK: f0 = #pop.dtype_from_ui8<#kgen.unknown : ui8> : !kgen.dtype
"pop_dtype_from_ui8.op"() { f0 = #pop.dtype_from_ui8<#kgen.unknown : ui8> : !kgen.dtype } : () -> ()

// CHECK: f0 = #kgen.param.expr<xor, #kgen<simd -1> : !kgen.simd<4, si32>, #kgen.unknown : !kgen.simd<4, si32>> : !kgen.simd<4, si32>
"simd_xor.attr"() { f0 = #kgen.param.expr<xor,#kgen.unknown : !kgen.simd<4, si32>, #kgen<simd -1> : !kgen.simd<4, si32>> : !kgen.simd<4, si32> } : () -> ()

// Equality against an unknown value stays symbolic.
// CHECK: f0 = #kgen.param.expr<eq, #kgen<simd -1> : !kgen.simd<4, si32>, #kgen.unknown : !kgen.simd<4, si32>> : !kgen.simd<4, bool>
"simd_cmp.attr"() { f0 = #kgen.param.expr<eq, #kgen.unknown : !kgen.simd<4, si32>, #kgen<simd -1> : !kgen.simd<4, si32>> : !kgen.simd<4, bool> } : () -> ()

// CHECK: f0 = #pop.simd_reduce_or<#kgen.unknown : !kgen.simd<4, si32>> : !kgen.scalar<si32>
"simd_reduce_or.attr"() { f0 = #pop.simd_reduce_or<#kgen.unknown : !kgen.simd<4, si32>> : !kgen.scalar<si32> } : () -> ()

// `shl(x, c)` canonicalizes to `mul(x, 1<<c)`.
// CHECK: f0 = #kgen.param.expr<mul, #kgen<simd 2> : !kgen.simd<4, si32>, #kgen.unknown : !kgen.simd<4, si32>> : !kgen.simd<4, si32>
"simd_shl.attr"() { f0 = #pop.simd_shl<#kgen.unknown : !kgen.simd<4, si32>, #kgen<simd 1> : !kgen.simd<4, si32>> } : () -> ()

// CHECK: f0 = #kgen.param.expr<shr, #kgen.unknown : !kgen.simd<4, si32>, #kgen<simd 1> : !kgen.simd<4, si32>> : !kgen.simd<4, si32>
"simd_shr.attr"() { f0 = #pop.simd_shr<#kgen.unknown : !kgen.simd<4, si32>, #kgen<simd 1> : !kgen.simd<4, si32>> } : () -> ()

// CHECK: f0 = #pop.variadic_to_array<:param_list<index> v> : !pop.array<#kgen.param_list.size<:param_list<index> v>, index>
"variadic_to_array.attr"() { f0 = #pop.variadic_to_array<:!kgen.param_list<index> #kgen.param.decl.ref<"v">>
      : !pop.array<#kgen.param_list.size<:!kgen.param_list<index> #kgen.param.decl.ref<"v">>, index> } : () -> ()

// CHECK: f0 = #pop.array<1, 2, 3> : !pop.array<3, index>
"variadic_to_array_fold.attr"() { f0 = #pop.variadic_to_array<:!kgen.param_list<index> #kgen.param_list<1, 2, 3> > : !pop.array<3, index> } : () -> ()

// CHECK:      a0 = #kgen<simd 0>
// CHECK-SAME: a1 = #kgen<simd 1>
// CHECK-SAME: a2 = #kgen<simd 0>
// CHECK-SAME: a3 = #kgen<simd 1>
// CHECK-SAME: a4 = #kgen<simd -1>
// CHECK-SAME: a5 = #kgen<simd 4294967295>
// CHECK-SAME: a6 = #kgen<simd -1>
// CHECK-SAME: a7 = #kgen<simd 255>
// CHECK-SAME: a8 = #kgen<simd -1>
// CHECK-SAME: a9 = #kgen<simd 65535>

// CHECK-SAME: b0 = #kgen<simd -1>
// CHECK-SAME: b1 = #kgen<simd 18446744073709551615>

// CHECK-SAME: c0 = #kgen<simd -1>
// CHECK-SAME: c1 = #kgen<simd 255>
// CHECK-SAME: c2 = #kgen<simd true>
// CHECK-SAME: c3 = #kgen<simd true>
// CHECK-SAME: c4 = #kgen<simd true>
// CHECK-SAME: c5 = #kgen<simd false>

// CHECK-SAME: d0 = #kgen<simd "1">
// CHECK-SAME: d1 = #kgen<simd "1">

// CHECK-SAME: e0 = #kgen<simd 255>

// CHECK-SAME: z6 = #kgen<simd -1>
// CHECK-SAME: z7 = #kgen<simd 18446744073709551615>
// CHECK-SAME: z8 = #kgen<simd -1>
// CHECK-SAME: z9 = #kgen<simd 340282366920938463463374607431768211455>
"literal_converts"() {
    a0 = #pop.int_literal_convert<0> : !kgen.scalar<si32>,
    a1 = #pop.int_literal_convert<1> : !kgen.scalar<si32>,
    a2 = #pop.int_literal_convert<0> : !kgen.scalar<ui32>,
    a3 = #pop.int_literal_convert<1> : !kgen.scalar<ui32>,
    a4 = #pop.int_literal_convert<-1> : !kgen.scalar<si32>,
    a5 = #pop.int_literal_convert<-1> : !kgen.scalar<ui32>,
    a6 = #pop.int_literal_convert<-1> : !kgen.scalar<si8>,
    a7 = #pop.int_literal_convert<-1> : !kgen.scalar<ui8>,
    a8 = #pop.int_literal_convert<-1> : !kgen.scalar<si16>,
    a9 = #pop.int_literal_convert<-1> : !kgen.scalar<ui16>,

    b0 = #pop.int_literal_convert<-1> : !kgen.scalar<si64>,
    b1 = #pop.int_literal_convert<-1> : !kgen.scalar<ui64>,

    c0 = #pop.int_literal_convert<65535> : !kgen.scalar<si8>,
    c1 = #pop.int_literal_convert<65535> : !kgen.scalar<ui8>,
    c2 = #pop.int_literal_convert<65535> : !kgen.scalar<bool>,
    c3 = #pop.int_literal_convert<65534> : !kgen.scalar<bool>,
    c4 = #pop.int_literal_convert<1> : !kgen.scalar<bool>,
    c5 = #pop.int_literal_convert<0> : !kgen.scalar<bool>,

    d0 = #pop.int_literal_convert<1> : !kgen.scalar<f32>,
    d1 = #pop.int_literal_convert<1> : !kgen.scalar<f64>,

    e0 = #pop.int_literal_convert<-1> : !kgen.simd<4, ui8>,

    z6 = #pop.int_literal_convert<-1> : !kgen.scalar<index>,
    z7 = #pop.int_literal_convert<-1> : !kgen.scalar<uindex>,
    z8 = #pop.int_literal_convert<-1> : !kgen.scalar<si128>,
    z9 = #pop.int_literal_convert<-1> : !kgen.scalar<ui128>
} : () -> ()
