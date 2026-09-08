// RUN: kgen-opt -lower-kgen-to-llvm -split-input-file %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="skylake-avx512", features="+fma", data_layout="", simd_bit_width=128, tune_cpu="skylake-avx512">} {

// CHECK-LABEL: llvm.func internal @trivial
// CHECK-SAME: (%[[ARG0:.*]]: i32
// CHECK-SAME: ["target-cpu", "skylake-avx512"]
// CHECK-SAME: ["target-features", "+fma"]
// CHECK-SAME: ["tune-cpu", "skylake-avx512"]
// CHECK-NEXT: llvm.return %[[ARG0]] : i32
kgen.func @trivial(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

// CHECK: llvm.func internal @none_type() -> !llvm.struct<()>
kgen.func @none_type() -> !kgen.none {
  // CHECK: [[NONE:%.*]] = llvm.mlir.undef : !llvm.struct<()>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: llvm.func internal @convert_pop_types
// CHECK-SAME: %{{.*}}: f32
// CHECK-SAME: %{{.*}}: !llvm.ptr
// CHECK-SAME: %{{.*}}: vector<4xf32>

kgen.func @convert_pop_types(
    %arg0: !kgen.simd<1, f32>,
    %arg1: !kgen.pointer<simd<1, f32>>,
    %arg2: !kgen.simd<4, f32>) {
  kgen.return
}

kgen.func @trivial_simd(%arg0: !kgen.simd<1, f32>) -> !kgen.simd<1, f32> {
  kgen.return %arg0 : !kgen.simd<1, f32>
}

kgen.func @no_result(%arg0: !kgen.simd<1, f32>) {
  kgen.return
}

kgen.func @two_results(%arg0: !kgen.simd<1, f32>) -> (!kgen.simd<1, f32>, !kgen.simd<1, f32>) {
  kgen.return %arg0, %arg0 : !kgen.simd<1, f32>, !kgen.simd<1, f32>
}

// CHECK-LABEL: llvm.func internal @convert_call
// CHECK-SAME: %[[ARG0:.*]]: f32
kgen.func @convert_call(%arg0: !kgen.simd<1, f32>) {
  // CHECK: llvm.call @trivial_simd(%[[ARG0]]) : (f32) -> f32
  %0 = kgen.call @trivial_simd(%arg0) : (!kgen.simd<1, f32>) -> !kgen.simd<1, f32>
  // CHECK: llvm.call @no_result(%[[ARG0]]) : (f32) -> ()
  kgen.call @no_result(%arg0) : (!kgen.simd<1, f32>) -> ()
  // CHECK: %[[PACK:.*]] = llvm.call @two_results(%[[ARG0]]) : (f32) -> !llvm.struct<(f32, f32)>
  %1:2 = kgen.call @two_results(%arg0) : (!kgen.simd<1, f32>) -> (!kgen.simd<1, f32>, !kgen.simd<1, f32>)
  // CHECK: llvm.extractvalue %[[PACK]][0]
  // CHECK: llvm.extractvalue %[[PACK]][1]

  // CHECK: llvm.call tail @trivial_simd(%[[ARG0]]) : (f32) -> f32
  kgen.call tail @trivial_simd(%arg0) : (!kgen.simd<1, f32>) -> !kgen.simd<1, f32>

  // CHECK: llvm.call musttail @trivial_simd(%[[ARG0]]) : (f32) -> f32
  kgen.call musttail @trivial_simd(%arg0) : (!kgen.simd<1, f32>) -> !kgen.simd<1, f32>

  kgen.return
}

//CHECK: "noinline"
kgen.func @test_no_inline(%a: i64) no_inline {
  kgen.return
}

//CHECK: "alwaysinline"
kgen.func @test_always_inline(%a: i64) always_inline {
  kgen.return
}

kgen.func @reference_me(%a: i64) -> i64 {
  kgen.return %a : i64
}

// CHECK-LABEL: @address_dtype
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr
// CHECK-SAME: %[[ARG1:.*]]: vector<4x!llvm.ptr>
kgen.func @address_dtype(%arg0 : !kgen.simd<1, address>, %arg1 : !kgen.simd<4, address>) {
  kgen.return
}

// CHECK-LABEL: @uninitmem
kgen.func @uninitmem() -> index {
  // CHECK-NEXT: llvm.mlir.undef : i64
  %0 = kgen.param.constant = <#interp.uninitmem : index>
  kgen.return %0 : index
}

kgen.func @constant_str() -> !kgen.string {
  // CHECK: %[[LENGTH:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[STRUCT:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  // CHECK: %[[GLOBAL_STR:.*]] = llvm.mlir.addressof @[[STATIC_STRING:.*]] : !llvm.ptr
  // CHECK: %[[GEP:.*]] = llvm.bitcast %[[GLOBAL_STR]] : !llvm.ptr to !llvm.ptr
  // CHECK: %[[VAL0:.*]] = llvm.insertvalue %[[GEP]], %[[STRUCT]][0] : !llvm.struct<(ptr, i64)>
  // CHECK: %[[VAL1:.*]] = llvm.insertvalue %[[LENGTH]], %[[VAL0]][1] : !llvm.struct<(ptr, i64)>
  %0 = kgen.param.constant: string = <"AB">
  // CHECK: llvm.return %[[VAL1]] : !llvm.struct<(ptr, i64)>
  kgen.return %0 : !kgen.string
}

kgen.func @constant_str_2() -> !kgen.string {
  // CHECK: llvm.mlir.addressof @[[STATIC_STRING]] : !llvm.ptr
  %0 = kgen.param.constant: string = <"AB">
  kgen.return %0 : !kgen.string
}

// CHECK-LABEL: @empty_str
kgen.func @empty_str() -> !kgen.string {
  // CHECK: %[[LENGTH:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %[[STRUCT:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  // CHECK: %[[GLOBAL_STR:.*]] = llvm.mlir.addressof @[[STATIC_EMPTY_STRING:.*]] : !llvm.ptr
  // CHECK: %[[GEP:.*]] = llvm.bitcast %[[GLOBAL_STR]] : !llvm.ptr to !llvm.ptr
  // CHECK: %[[VAL0:.*]] = llvm.insertvalue %[[GEP]], %[[STRUCT]][0] : !llvm.struct<(ptr, i64)>
  // CHECK: %[[VAL1:.*]] = llvm.insertvalue %[[LENGTH]], %[[VAL0]][1] : !llvm.struct<(ptr, i64)>
  %0 = kgen.param.constant: string = <"">
  // CHECK: llvm.return %[[VAL1]] : !llvm.struct<(ptr, i64)>
  kgen.return %0 : !kgen.string
}

// CHECK-LABEL: @test_unreachable
kgen.func @test_unreachable() -> !kgen.simd<1, f32> {
  // CHECK-NEXT: llvm.trap
  // CHECK-NEXT: llvm.unreachable
  kgen.unreachable
}

// CHECK-LABEL: @address_of
kgen.func @address_of() -> !kgen.generator<() -> !kgen.scalar<f32>> {
  // CHECK: llvm.mlir.addressof @test_unreachable : !llvm.ptr
  %0 = kgen.param.constant: () -> !kgen.scalar<f32> = <@test_unreachable>
  kgen.return %0 : !kgen.generator<() -> !kgen.scalar<f32>>
}

// CHECK: llvm.func @used_internally
kgen.func export @used_internally() cabi -> !kgen.struct<(i32, i32)>{
  kgen.unreachable
}

// CHECK: llvm.func internal @used_func
kgen.func @used_func() {
  // CHECK-NEXT: call @used_internally
  kgen.call @used_internally() : () -> !kgen.struct<(i32, i32)>
  kgen.return
}

// CHECK: llvm.mlir.global internal constant @[[STATIC_STRING]]("AB\00") {addr_space = 0 : i32, alignment = 16 : i64}
// CHECK: llvm.mlir.global internal constant @[[STATIC_EMPTY_STRING]]("\00") {addr_space = 0 : i32, alignment = 16 : i64}

}

// -----

// `abi`, when set, is recorded as a `target-abi` module flag (mirroring
// clang), not a function attribute like `target-cpu`/`target-features`.
module attributes {M.target_info = #M.target<triple="", arch="skylake-avx512", features="+fma", data_layout="", simd_bit_width=128, abi="lp64d">} {

// CHECK: llvm.module_flags [#llvm.mlir.module_flag<error, "target-abi", "lp64d">]
// CHECK-LABEL: llvm.func internal @with_abi
// CHECK-SAME: ["target-cpu", "skylake-avx512"]
// CHECK-SAME: ["target-features", "+fma"]
// CHECK-NOT: "target-abi"
kgen.func @with_abi(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @struct_constant
kgen.func @struct_constant() -> !kgen.struct<(array<1, i32>, struct<(i32, i32)>)> {
  // CHECK: %0 = llvm.mlir.undef : !llvm.struct<(array<1 x i32>, struct<(i32, i32)>)>
  // CHECK: %1 = llvm.mlir.undef : !llvm.array<1 x i32>
  // CHECK: %2 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %3 = llvm.insertvalue %2, %1[0] : !llvm.array<1 x i32>
  // CHECK: %4 = llvm.insertvalue %3, %0[0] : !llvm.struct<(array<1 x i32>, struct<(i32, i32)>)>
  // CHECK: %5 = llvm.mlir.undef : !llvm.struct<(i32, i32)>
  // CHECK: %6 = llvm.mlir.constant(2 : i32) : i32
  // CHECK: %7 = llvm.insertvalue %6, %5[0] : !llvm.struct<(i32, i32)>
  // CHECK: %8 = llvm.mlir.constant(3 : i32) : i32
  // CHECK: %9 = llvm.insertvalue %8, %7[1] : !llvm.struct<(i32, i32)>
  // CHECK: %10 = llvm.insertvalue %9, %4[1] : !llvm.struct<(array<1 x i32>, struct<(i32, i32)>)>
  // CHECK: llvm.return %10 : !llvm.struct<(array<1 x i32>, struct<(i32, i32)>)>
  %0 = kgen.param.constant: struct<(array<1, i32>, struct<(i32, i32)>)> =
    <{ [1], { 2, 3 } }>
  kgen.return %0 : !kgen.struct<(array<1, i32>, struct<(i32, i32)>)>
}

// CHECK-LABEL: @pointer_constant
kgen.func @pointer_constant() -> !kgen.pointer<*?> {
  // CHECK: %0 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
  // CHECK: llvm.return %1 : !llvm.ptr
  %null = kgen.param.constant: pointer<*?> = <#interp.pointer<0>>
  kgen.return %null : !kgen.pointer<*?>
}

// CHECK-LABEL: @empty_struct_with_never
kgen.func @empty_struct_with_never() throws -> !kgen.struct<(union<struct<()>, !kgen.never>, scalar<ui8>)> {
  %struct = kgen.param.constant: struct<(union<struct<()>, !kgen.never>, scalar<ui8>)> = <{ {:struct<()> #interp.uninitmem}, 0 }>
  // CHECK: llvm.return %{{.*}} : !llvm.struct<(struct<()>, i8)>
  kgen.return %struct : !kgen.struct<(union<struct<()>, !kgen.never>, scalar<ui8>)>
}

}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=64>} {

// CHECK-LABEL: llvm.func internal @coro
// CHECK-SAME: coroutineType = !llvm.struct<(i64, ptr, ptr, ptr, ptr, ptr, ptr)>
kgen.func @coro() attributes {coroutineType = !kgen.struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>)>} {
  kgen.return
}

}
