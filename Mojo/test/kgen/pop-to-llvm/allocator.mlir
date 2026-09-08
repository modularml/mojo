// RUN: kgen-opt %s -split-input-file -lower-to-llvm | kgen-translate -split-input-file -mlir-to-llvmir | FileCheck %s

// CHECK: declare noalias ptr @KGEN_CompilerRT_AlignedAlloc(i64 allocalign, i64) [[ALLOC_ATTRS:#[0-9]+]]
// CHECK: declare void @KGEN_CompilerRT_AlignedFree(ptr allocptr) [[FREE_ATTRS:#[0-9]+]]

// CHECK: attributes [[ALLOC_ATTRS]] =
// CHECK-SAME: allockind("alloc,uninitialized,aligned")
// CHECK-SAME: allocsize(1)
// CHECK-SAME: "alloc-family"="kgen_aligned_allocator"

// CHECK: attributes [[FREE_ATTRS]] =
// CHECK-SAME: allockind("free")
// CHECK-SAME: "alloc-family"="kgen_aligned_allocator"

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  kgen.func @alloc_free() {
    %size = index.constant 1
    %align = index.constant 8
    %0 = pop.aligned_alloc %align, %size : <index>
    pop.aligned_free %0 : <index>
    kgen.return
  }
}
