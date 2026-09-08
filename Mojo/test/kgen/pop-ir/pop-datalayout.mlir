// RUN: kgen-opt %s | FileCheck %s

#target = #kgen.target<triple="", arch="", features="", data_layout="i64:64:64", simd_bit_width=128> : !kgen.target
#i32_align8 = #kgen.target<triple="", arch="", features="", data_layout="i32:64:64", simd_bit_width=128> : !kgen.target

// CHECK-LABEL: @pop_sizeof_alignof
kgen.generator @pop_sizeof_alignof<N, T:type, DT:dtype>() {
  // CHECK-NEXT: <1>
  kgen.param.constant: index = <get_sizeof(array<1, i8>, #target)>
  // CHECK-NEXT: <4>
  kgen.param.constant: index = <get_sizeof(array<4, i6>, #target)>
  // CHECK-NEXT: <get_sizeof(array<N, i8>, #kgen.target<{{.*}}>)>
  kgen.param.constant: index = <get_sizeof(array<N, i8>, #target)>
  // CHECK-NEXT: <1>
  kgen.param.constant: index = <get_alignof(array<1, i8>, #target)>
  // CHECK-NEXT: <4>
  kgen.param.constant: index = <get_alignof(array<4, i30>, #target)>
  // CHECK-NEXT: <1>
  kgen.param.constant: index = <get_alignof(array<N, i8>, #target)>

  // CHECK-NEXT: <8>
  kgen.param.constant: index = <get_sizeof(pointer<scalar<invalid>>, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: index = <get_alignof(pointer<array<4, i32>>, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: index = <get_sizeof(pointer<T>, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: index = <get_alignof(pointer<T>, #target)>

  // CHECK-NEXT: <4>
  kgen.param.constant: index = <get_sizeof(scalar<si32>, #target)>
  // CHECK-NEXT: <1>
  kgen.param.constant: index = <get_alignof(scalar<si4>, #target)>
  // CHECK-NEXT: <get_sizeof(scalar<DT>, #kgen.target<{{.*}}>)>
  kgen.param.constant: index = <get_sizeof(scalar<DT>, #target)>
  // CHECK-NEXT: <get_alignof(scalar<DT>, #kgen.target<{{.*}}>)>
  kgen.param.constant: index = <get_alignof(scalar<DT>, #target)>

  // CHECK-NEXT: <16>
  kgen.param.constant: index = <get_sizeof(simd<4, f32>, #target)>
  // CHECK-NEXT: <16>
  kgen.param.constant: index = <get_alignof(simd<4, f32>, #target)>
  // CHECK-NEXT: <get_sizeof(simd<N, f32>, #kgen.target<{{.*}}>)>
  kgen.param.constant: index = <get_sizeof(simd<N, f32>, #target)>
  // CHECK-NEXT: <get_alignof(simd<4, DT>, #kgen.target<{{.*}}>)>
  kgen.param.constant: index = <get_alignof(simd<4, DT>, #target)>

  // CHECK-NEXT: <0>
  kgen.param.constant: index = <get_sizeof(struct<()>, #target)>
  // CHECK-NEXT: <24>
  kgen.param.constant: index = <get_sizeof(struct<(i8, i32, i64, i32)>, #target)>
  // CHECK-NEXT: <1>
  kgen.param.constant: index = <get_alignof(struct<()>, #target)>
  // CHECK-NEXT: <4>
  kgen.param.constant: index = <get_alignof(struct<(i8, i32, i16)>, #target)>
  // CHECK-NEXT: <16>
  kgen.param.constant: index = <get_sizeof(struct<(i32, i8)>, #i32_align8)>

  // CHECK-NEXT: <8>
  kgen.param.constant: index = <get_sizeof(variant<i32, i16>, #target)>
  // CHECK-NEXT: <1>
  kgen.param.constant: index = <get_alignof(variant<i1, i2, i3, i4>, #target)>

  // CHECK-NEXT: <16>
  kgen.param.constant: index = <get_sizeof(param_list<i32>, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: index = <get_alignof(param_list<i32>, #target)>

  // CHECK-NEXT: <16>
  kgen.param.constant: index = <get_sizeof(union<simd<4, f32>, i64>, #target)>
  // Union alignment is now the max alignment of variant types (simd<4, f32> = 16).
  // CHECK-NEXT: <16>
  kgen.param.constant: index = <get_alignof(union<simd<4, f32>, i64>, #target)>

  kgen.return
}

// CHECK-LABEL: @simd_normal()
kgen.generator @simd_normal() {
  // FIXME: get_alignof isn't implemented in terms of DataLayout::getFloatABIAlign.
  // https://github.com/modularml/modular/issues/28137

  // CHECK-NEXT: <16>
  kgen.param.constant: index = <get_alignof(simd<4, si32>, #kgen.target<triple="", arch="", features="", data_layout="p:32:32-v128:256", simd_bit_width=128>)>
  // CHECK-NEXT: <8>
  kgen.param.constant: index = <get_alignof(simd<2, si32>, #kgen.target<triple="", arch="", features="", data_layout="p:32:32-v64:64-v128:256", simd_bit_width=128>)>
  // CHECK-NEXT: <4>
  kgen.param.constant: index = <get_alignof(f32, #kgen.target<triple="", arch="", features="", data_layout="p:32:32", simd_bit_width=128>)>
  // CHECK-NEXT: <8>
  kgen.param.constant: index = <get_alignof(f32, #kgen.target<triple="", arch="", features="", data_layout="p:32:32-f32:64:64", simd_bit_width=128>)>
  // CHECK-NEXT: <4>
  kgen.param.constant: index = <get_alignof(f32, #kgen.target<triple="", arch="", features="", data_layout="p:32:32-f32:32:32", simd_bit_width=128>)>

  // CHECK-NEXT: <0>
  kgen.param.constant: index = <get_sizeof(scalar<invalid>, #target)>
  kgen.return
}

// `get_sizeof` returns the alloc size: the store size rounded up to the
// alignment, i.e. the stride between adjacent array elements. The two differ
// for non-power-of-two SIMD vectors and for structs whose `align(N)` exceeds
// their natural alignment.
// CHECK-LABEL: @sizeof_rounds_up_to_alignment()
kgen.generator @sizeof_rounds_up_to_alignment() {
  // CHECK-NEXT: <16>
  kgen.param.constant: index = <get_sizeof(simd<3, f32>, #target)>
  // CHECK-NEXT: <16>
  kgen.param.constant: index = <get_alignof(simd<3, f32>, #target)>
  // CHECK-NEXT: <32>
  kgen.param.constant: index = <get_sizeof(simd<5, f32>, #target)>
  // CHECK-NEXT: <32>
  kgen.param.constant: index = <get_sizeof(simd<6, f32>, #target)>
  // CHECK-NEXT: <32>
  kgen.param.constant: index = <get_alignof(simd<6, f32>, #target)>
  // Sub-byte element vectors stay bit-packed within their padded size.
  // CHECK-NEXT: <1>
  kgen.param.constant: index = <get_sizeof(simd<3, si1>, #target)>
  // CHECK-NEXT: <2>
  kgen.param.constant: index = <get_sizeof(simd<3, si4>, #target)>

  // A struct's explicit alignment pads its size up to a full stride.
  // CHECK-NEXT: <32>
  kgen.param.constant: index = <get_sizeof(struct<(f32) align(32)>, #target)>
  // CHECK-NEXT: <32>
  kgen.param.constant: index = <get_alignof(struct<(f32) align(32)>, #target)>
  // A non-power-of-two SIMD field occupies its full padded stride.
  // CHECK-NEXT: <32>
  kgen.param.constant: index = <get_sizeof(struct<(simd<3, f32>, f32)>, #target)>
  kgen.return
}

// CHECK-LABEL: @simd_bitpacked()
kgen.generator @simd_bitpacked() {
  // CHECK-NEXT: <1>
  kgen.param.constant: index = <get_sizeof(scalar<si4>, #target)>
  // CHECK-NEXT: <2>
  kgen.param.constant: index = <get_sizeof(simd<4, si4>, #target)>
  // CHECK-NEXT: <4>
  kgen.param.constant: index = <get_sizeof(scalar<index>, #kgen.target<triple="", arch="", features="", data_layout="p:32:32", simd_bit_width=128>)>
  // CHECK-NEXT: <8>
  kgen.param.constant: index = <get_sizeof(simd<2, address>, #kgen.target<triple="", arch="", features="", data_layout="p:32:32", simd_bit_width=128>)>
  kgen.return
}
