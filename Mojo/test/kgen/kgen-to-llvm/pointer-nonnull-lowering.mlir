// RUN: kgen-opt %s --lower-kgen-to-llvm | FileCheck %s

// Test LLVM lowering of non-null pointer types.
// This verifies that `nonnull` and `noundef` attributes are correctly applied
// during conversion to LLVM IR.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=64>} {

// Non-null pointer argument gets nonnull attribute.
// CHECK-LABEL: @nonnull_arg
// CHECK-SAME: !llvm.ptr {llvm.nonnull, llvm.noundef}
kgen.func @nonnull_arg(%arg0: !kgen.pointer<scalar<f32>, 0, nonnull>) {
  kgen.return
}

// Nullable pointer does NOT get nonnull attribute (but gets noundef from
// calling convention).
// CHECK-LABEL: @nullable_arg
// CHECK-SAME: !llvm.ptr {llvm.noundef}
// CHECK-NOT: llvm.nonnull
kgen.func @nullable_arg(%arg0: !kgen.pointer<scalar<f32>>) {
  kgen.return
}

// Non-null pointer with non-default address space.
// CHECK-LABEL: @nonnull_addrspace
// CHECK-SAME: !llvm.ptr<1> {llvm.nonnull, llvm.noundef}
kgen.func @nonnull_addrspace(%arg0: !kgen.pointer<scalar<f32>, 1, nonnull>) {
  kgen.return
}

// Non-null pointer return type gets nonnull attribute.
// CHECK-LABEL: @return_nonnull
// CHECK-SAME: -> (!llvm.ptr {llvm.nonnull, llvm.noundef})
kgen.func @return_nonnull() -> !kgen.pointer<scalar<f32>, 0, nonnull> {
  %ptr = builtin.unrealized_conversion_cast to !kgen.pointer<scalar<f32>, 0, nonnull>
  kgen.return %ptr : !kgen.pointer<scalar<f32>, 0, nonnull>
}

// Nullable pointer return does NOT get nonnull attribute.
// CHECK-LABEL: @return_nullable
// CHECK-SAME: -> !llvm.ptr
// CHECK-NOT: llvm.nonnull
kgen.func @return_nullable() -> !kgen.pointer<scalar<f32>> {
  %ptr = builtin.unrealized_conversion_cast to !kgen.pointer<scalar<f32>>
  kgen.return %ptr : !kgen.pointer<scalar<f32>>
}

// Non-null arg and return attributes coexist correctly.
// CHECK-LABEL: @nonnull_arg_and_return
// CHECK-SAME: (%arg0: !llvm.ptr {llvm.nonnull, llvm.noundef}) -> (!llvm.ptr {llvm.nonnull, llvm.noundef})
kgen.func @nonnull_arg_and_return(%arg0: !kgen.pointer<scalar<f32>, 0, nonnull>)
    -> !kgen.pointer<scalar<f32>, 0, nonnull> {
  kgen.return %arg0 : !kgen.pointer<scalar<f32>, 0, nonnull>
}

}
