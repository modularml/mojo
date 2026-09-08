// RUN: kgen-opt -lower-kgen-to-llvm --split-input-file %s | FileCheck %s

!struct1 = !kgen.struct<(struct<(f32)>, array<4, f32>)>

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @struct_construct
kgen.func @struct_construct(%a: !kgen.struct<(f32)>, %b: !pop.array<4, f32>) -> !struct1 {
  // CHECK: %[[S0:.*]] = llvm.mlir.undef : !llvm.struct<(struct<(f32)>, array<4 x f32>)>
  // CHECK: %[[S1:.*]] = llvm.insertvalue %{{.*}}, %[[S0]][0]
  // CHECK: %[[S2:.*]] = llvm.insertvalue %{{.*}}, %[[S1]][1]
  %0 = kgen.struct.create(%a, %b) : !struct1
  kgen.return %0 : !struct1
}

// CHECK-LABEL: @struct_construct_one
kgen.func @struct_construct_one(%a: f32) -> !kgen.struct<(f32)> {
  // CHECK: %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(f32)>
  %0 = kgen.struct.create(%a) : !kgen.struct<(f32)>
  kgen.return %0 : !kgen.struct<(f32)>
}

// CHECK-LABEL: @struct_insert
kgen.func @struct_insert(%a: !kgen.struct<(f32, f32)>, %b: f32) -> !kgen.struct<(f32, f32)> {
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(f32, f32)>
  %0 = kgen.struct.replace %b, %a[0] : !kgen.struct<(f32, f32)>
  kgen.return %0 : !kgen.struct<(f32, f32)>
}

// CHECK-LABEL: @struct_insert_one
kgen.func @struct_insert_one(%a: !kgen.struct<(f32)>, %b: f32) -> !kgen.struct<(f32)> {
  // CHECK: llvm.insertvalue %arg1, %arg0[0] : !llvm.struct<(f32)>
  %0 = kgen.struct.replace %b, %a[0] : !kgen.struct<(f32)>
  kgen.return %0 : !kgen.struct<(f32)>
}

// CHECK-LABEL: @struct_extract
kgen.func @struct_extract(%a: !kgen.struct<(f32, f32)>) -> f32 {
  // CHECK: llvm.extractvalue %{{.*}}[0]
  %0 = kgen.struct.extract %a[0] : !kgen.struct<(f32, f32)>
  kgen.return %0 : f32
}

// CHECK-LABEL: @struct_extract_one
kgen.func @struct_extract_one(%a: !kgen.struct<(f32)>) -> f32 {
  // CHECK: llvm.extractvalue %arg0[0] : !llvm.struct<(f32)>
  %0 = kgen.struct.extract %a[0] : !kgen.struct<(f32)>
  kgen.return %0 : f32
}

// CHECK-LABEL: @struct_gep
kgen.func @struct_gep(%a: !kgen.pointer<struct<(i32, i64)>>) -> !kgen.pointer<i64> {
  // CHECK: llvm.getelementptr %{{.*}}[0, 1] : (!llvm.ptr) -> !llvm.ptr
  %0 = kgen.struct.gep %a[1] : <struct<(i32, i64)>>
  kgen.return %0 : !kgen.pointer<i64>
}

// CHECK-LABEL: @struct_gep_one
kgen.func @struct_gep_one(%a: !kgen.pointer<struct<(i32)>>) -> !kgen.pointer<i32> {
  // CHECK: llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr
  %0 = kgen.struct.gep %a[0] : <struct<(i32)>>
  kgen.return %0 : !kgen.pointer<i32>
}

// CHECK-LABEL: @struct_gep_as
kgen.func @struct_gep_as(%a: !kgen.pointer<struct<(i32, i64)>, 4>) -> !kgen.pointer<i64, 4> {
  // CHECK: llvm.getelementptr %{{.*}}[0, 1] : (!llvm.ptr<4>) -> !llvm.ptr<4>
  %0 = kgen.struct.gep %a[1] : <struct<(i32, i64)>, 4>
  kgen.return %0 : !kgen.pointer<i64, 4>
}

// CHECK-LABEL: @struct_aligned_64_1
kgen.func @struct_aligned_64_1(%a: i32) -> !kgen.struct<(i32) align(64)> {
  // CHECK: %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i32, array<60 x i8>)>
  %0 = kgen.struct.create(%a) : !kgen.struct<(i32) align(64)>
  kgen.return %0 : !kgen.struct<(i32) align(64)>
}

// CHECK-LABEL: @struct_aligned_64_2
kgen.func @struct_aligned_64_2(%a: i32) -> !kgen.struct<(i32, i32) align(64)> {
  // CHECK: %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i32, i32, array<56 x i8>)>
  %0 = kgen.struct.create(%a, %a) : !kgen.struct<(i32, i32) align(64)>
  kgen.return %0 : !kgen.struct<(i32, i32) align(64)>
}

// CHECK-LABEL: @struct_aligned_64_nested
// A nested struct with its own alignment
kgen.func @struct_aligned_64_nested(%a: i32) -> !kgen.struct<(i32, !kgen.struct<(i32) align(8)>) align(8)> {
  // CHECK: %3 = llvm.insertvalue %arg0, %2[0] : !llvm.struct<(i32, array<4 x i8>, struct<(i32, array<4 x i8>)>)>
  %0 = kgen.struct.create(%a) : !kgen.struct<(i32) align(8)>
  %1 = kgen.struct.create(%a, %0) : !kgen.struct<(i32, !kgen.struct<(i32) align(8)>) align(8)>
  kgen.return %1 : !kgen.struct<(i32, !kgen.struct<(i32) align(8)>) align(8)>
}

}

// -----

// COM: A data layout where aggregates are aligned to 8 bytes
module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="a:64:64", simd_bit_width=128>} {

// CHECK-LABEL: @struct_aligned_64_1
// COM: Check we don't insert unnecessary padding
kgen.func @struct_aligned_64_1(%a: i32) -> !kgen.struct<(i32) align(8)> {
  // CHECK: %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i32)>
  %0 = kgen.struct.create(%a) : !kgen.struct<(i32) align(8)>
  kgen.return %0 : !kgen.struct<(i32) align(8)>
}

// CHECK-LABEL: @struct_aligned_64_2
kgen.func @struct_aligned_64_2(%a: i32, %b: i16) -> !kgen.struct<(i32, i16) align(8)> {
  // CHECK: %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i32, i16)>
  %0 = kgen.struct.create(%a, %b) : !kgen.struct<(i32, i16) align(8)>
  kgen.return %0 : !kgen.struct<(i32, i16) align(8)>
}

// CHECK-LABEL: @struct_aligned_64_3
// COM: We still have to pad if we're asked to overly align
kgen.func @struct_aligned_64_3(%a: i32, %b: i16) -> !kgen.struct<(i32, i32) align(16)> {
  // CHECK: %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i32, i32, array<8 x i8>)>
  %0 = kgen.struct.create(%a, %a) : !kgen.struct<(i32, i32) align(16)>
  kgen.return %0 : !kgen.struct<(i32, i32) align(16)>
}

// CHECK-LABEL: @struct_aligned_64_nested_1
// A nested struct with its own alignment - both are handled by the DataLayout automatically
kgen.func @struct_aligned_64_nested_1(%a: i32) -> !kgen.struct<(i32, !kgen.struct<(i32) align(8)>) align(8)> {
  // CHECK: %3 = llvm.insertvalue %arg0, %2[0] : !llvm.struct<(i32, struct<(i32)>)>
  %0 = kgen.struct.create(%a) : !kgen.struct<(i32) align(8)>
  %1 = kgen.struct.create(%a, %0) : !kgen.struct<(i32, !kgen.struct<(i32) align(8)>) align(8)>
  kgen.return %1 : !kgen.struct<(i32, !kgen.struct<(i32) align(8)>) align(8)>
}

// CHECK-LABEL: @struct_aligned_64_nested_2
// The inner struct is 16-byte aligned so needs padding before it to keep it
// aligned, and inside it to keep itself the right size.
kgen.func @struct_aligned_64_nested_2(%a: i32) -> !kgen.struct<(i32, !kgen.struct<(i32) align(16)>) align(8)> {
  // CHECK: %3 = llvm.insertvalue %arg0, %2[0] : !llvm.struct<(i32, array<12 x i8>, struct<(i32, array<12 x i8>)>)>
  %0 = kgen.struct.create(%a) : !kgen.struct<(i32) align(16)>
  %1 = kgen.struct.create(%a, %0) : !kgen.struct<(i32, !kgen.struct<(i32) align(16)>) align(8)>
  kgen.return %1 : !kgen.struct<(i32, !kgen.struct<(i32) align(16)>) align(8)>
}

// CHECK-LABEL: @struct_aligned_64_nested_3
// The inner struct is 16-byte aligned so needs padding before it to keep it
// aligned, and inside it to keep itself the right size. It also needs padding
// after it to bump it back up to the overly-aligned 16 byte requirement.
kgen.func @struct_aligned_64_nested_3(%a: i32, %b: i8) -> !kgen.struct<(i32, !kgen.struct<(i32) align(16)>, i8) align(8)> {
  // CHECK: %3 = llvm.insertvalue %arg0, %2[0] : !llvm.struct<(i32, array<12 x i8>, struct<(i32, array<12 x i8>)>, i8, array<15 x i8>)>
  %0 = kgen.struct.create(%a) : !kgen.struct<(i32) align(16)>
  %1 = kgen.struct.create(%a, %0, %b) : !kgen.struct<(i32, !kgen.struct<(i32) align(16)>, i8) align(8)>
  kgen.return %1 : !kgen.struct<(i32, !kgen.struct<(i32) align(16)>, i8) align(8)>
}

}
