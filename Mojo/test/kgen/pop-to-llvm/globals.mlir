// RUN: kgen-opt -split-input-file -allow-unregistered-dialect -lower-global-pop-to-llvm %s | FileCheck %s --check-prefix=CHECK

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @external_call
  kgen.func @external_call(%a: !kgen.simd<1, ui32>, %b: !kgen.pointer<i32>) -> !kgen.simd<4, f64> {
    // CHECK: llvm.call @foo
    %0 = pop.external_call @foo(%a) attributes {
      funcAttrs = ["noinline", "noreturn"],
      memory = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read, errnoMem = none, targetMem0 = none, targetMem1 = none>
    } : (!kgen.simd<1, ui32>) -> !kgen.simd<4, f64>
    // CHECK: llvm.call @bar
    %1 = pop.external_call @bar(%b) attributes {argAttrs = [{llvm.noalias}], resAttrs = {llvm.signext}} : (!kgen.pointer<i32>) -> i32

    // CHECK: llvm.call @bar
    %2 = pop.external_call @bar(%b) attributes {argAttrs = [{llvm.noalias}], resAttrs = {llvm.signext}} : (!kgen.pointer<i32>) -> i32
    kgen.return %0 : !kgen.simd<4, f64>
  }
  // CHECK: llvm.func @foo(i32) -> vector<4xf64>
  // CHECK-SAME: memory_effects = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read, errnoMem = none, targetMem0 = none, targetMem1 = none>
  // CHECK-SAME: passthrough = ["noinline", "noreturn"
  // CHECK: llvm.func @bar(!llvm.ptr {llvm.noalias}) -> (i32 {llvm.signext})
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @external_call_variadic
  kgen.func @external_call_variadic(%a: !kgen.simd<1, ui32>) {
    // CHECK: llvm.call @foo
    pop.external_call @foo (%a) attributes {numFixedArgs = 1 : index} : (!kgen.simd<1, ui32>) -> ()
    // CHECK: llvm.call @foo
    pop.external_call @foo (%a, %a) attributes {numFixedArgs = 1 : index} : (!kgen.simd<1, ui32>, !kgen.simd<1, ui32>) -> ()
    kgen.return
  }
  // CHECK: llvm.func @foo(i32, ...)
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @global_constant
  kgen.func @global_constant() {
    // CHECK: llvm.mlir.addressof @global_constant_0 : !llvm.ptr
    %0 = pop.global_constant: ui32 = <5>
    // CHECK: llvm.mlir.addressof @global_constant_0 : !llvm.ptr
    %1 = pop.global_constant: ui32 = <5>
    // CHECK: llvm.mlir.addressof @global_constant_1 : !llvm.ptr
    %2 = pop.global_constant: simd<2, si32> = <<2, 5>>
    kgen.return
  }

  // CHECK-LABEL: kgen.func @global_alloc
  kgen.func @global_alloc() -> !kgen.pointer<scalar<f32>, 3> {
    // CHECK-NEXT: %0 = llvm.mlir.addressof @my_alloc : !llvm.ptr<3>
    // CHECK-NEXT: %1 = llvm.bitcast %0 : !llvm.ptr<3> to !llvm.ptr<3>
    %0 = pop.global_alloc "my_alloc" 2 x !kgen.scalar<f32> address_space 3 align 4
    kgen.return %0 : !kgen.pointer<scalar<f32>, 3>
  }

  // CHECK-LABEL: llvm.mlir.global internal @my_alloc() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<2 x f32>
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @global_alloc_initialized
  kgen.func @global_alloc_initialized() -> !kgen.pointer<scalar<si32>> {
    // CHECK: llvm.mlir.addressof @my_init : !llvm.ptr
    // CHECK: llvm.bitcast
    %0 = pop.global_alloc "my_init" 1 x !kgen.scalar<si32> align 4 = <31>
    kgen.return %0 : !kgen.pointer<scalar<si32>>
  }

  // CHECK: llvm.mlir.global internal @my_init() {addr_space = 0 : i32, alignment = 4 : i64} : !llvm.array<1 x i32> {
  // CHECK-NEXT: %0 = llvm.mlir.constant(31 : i32) : i32
  // CHECK-NEXT: %1 = llvm.mlir.undef : !llvm.array<1 x i32>
  // CHECK-NEXT: %2 = llvm.insertvalue %0, %1[0] : !llvm.array<1 x i32>
  // CHECK-NEXT: llvm.return %2 : !llvm.array<1 x i32>
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @aligned_globals
  kgen.func @aligned_globals() {

    // Same value + same alignment = same constant.
    // CHECK: llvm.mlir.addressof @global_constant
    %0 = pop.global_constant: ui32 = <5> align 4
    // CHECK: llvm.mlir.addressof @global_constant
    %1 = pop.global_constant: ui32 = <5> align 4

    // Same value + different alignment = different constant.
    // CHECK: llvm.mlir.addressof @global_constant_0
    %2 = pop.global_constant: ui32 = <5> align 16

    // CHECK: llvm.mlir.addressof @global_constant_1
    %3 = pop.global_constant: simd<2, si32> = <<2, 5>> align 64
    kgen.return
  }

  // CHECK: llvm.mlir.global internal constant @global_constant() {addr_space = 0 : i32, alignment = 4 : i64} : i32 {
  // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(5 : i32) : i32

  // CHECK: llvm.mlir.global internal constant @global_constant_0() {addr_space = 0 : i32, alignment = 16 : i64} : i32 {
  // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(5 : i32) : i32

  // CHECK: llvm.mlir.global internal constant @global_constant_1() {addr_space = 0 : i32, alignment = 64 : i64} : vector<2xi32>
  // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(#M.dense_array<2, 5> : vector<2xi32>) : vector<2xi32>
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @global_array_constant
  kgen.func @global_array_constant() {
    // CHECK: llvm.mlir.addressof @global_constant
    %0 = pop.global_constant: array<4, ui32> = <[1, 2, 3, 4]>
    kgen.return
  }
  // CHECK: llvm.mlir.global internal constant @global_constant() {
  // CHECK: %0 = llvm.mlir.undef : !llvm.array<4 x i32>
  // CHECK: llvm.return %{{.*}} : !llvm.array<4 x i32>
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: llvm.func @alloc_free
  llvm.func @alloc_free() {
    %size = index.constant 1
    %align = index.constant 8
    // CHECK: [[RAW_PTR:%.*]] = llvm.call @KGEN_CompilerRT_AlignedAlloc
    // CHECK-NEXT: [[PTR:%.*]] = llvm.bitcast [[RAW_PTR]] : !llvm.ptr to !llvm.ptr
    %0 = pop.aligned_alloc %align, %size : <index>
    // CHECK: [[RAW_PTR:%.*]] = llvm.bitcast [[PTR]] : !llvm.ptr to !llvm.ptr
    // CHECK-NEXT: llvm.call @KGEN_CompilerRT_AlignedFree([[RAW_PTR]])
    pop.aligned_free %0 : <index>
    llvm.return
  }

  // CHECK: llvm.func @KGEN_CompilerRT_AlignedAlloc(i64 {llvm.allocalign}, i64) -> (!llvm.ptr {llvm.noalias})
  // CHECK-DAG: ["allockind", "41"]
  // CHECK-DAG: ["allocsize", "8589934591"]
  // CHECK-DAG: ["alloc-family", "kgen_aligned_allocator"]

  // CHECK: llvm.func @KGEN_CompilerRT_AlignedFree(!llvm.ptr {llvm.allocptr})
  // CHECK-DAG: ["allockind", "4"]
  // CHECK-DAG: ["alloc-family", "kgen_aligned_allocator"]
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @external_call
kgen.func @external_call(%a : !kgen.struct<(scalar<si32>)>,
                         %b : !kgen.struct<(scalar<si32>, scalar<f32>, scalar<f32>)>) {
   // Struct operands are passed directly — LLVM's backend handles C ABI.
   // CHECK: llvm.call @call1(%{{.*}}) : (!llvm.struct<(i32)>) -> i32
   %0 = pop.external_call @call1(%a) : (!kgen.struct<(scalar<si32>)>) -> !kgen.scalar<si32>

   // CHECK: llvm.call @call3(%{{.*}}) : (!llvm.struct<(i32, f32, f32)>) -> i32
   %1 = pop.external_call @call3(%b) : (!kgen.struct<(scalar<si32>, scalar<f32>, scalar<f32>)>) -> !kgen.scalar<si32>
   kgen.return
}
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @noalias_cast
kgen.func @noalias_cast(%arg0: !kgen.pointer<index>) -> index {
  // CHECK: llvm.call @__kgen_noalias_cast(%0)
  %0 = pop.noalias_pointer_cast %arg0 : !kgen.pointer<index>
  %1 = pop.load %0 : !kgen.pointer<index>
  kgen.return %1 : index
}

// CHECK: llvm.func internal @__kgen_noalias_cast
// CHECK-SAME: (%arg0: !llvm.ptr {llvm.noalias}) -> (!llvm.ptr {llvm.noalias})
// CHECK-SAME: ["alwaysinline", "mustprogress", "nofree", "norecurse", "nosync",
// CHECK-SAME:  "nounwind", "willreturn", ["memory", "0"]
// CHECK-NEXT: return %arg0

}
