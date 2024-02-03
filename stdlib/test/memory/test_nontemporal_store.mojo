# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from sys.info import simdwidthof

from memory.buffer import Buffer, NDBuffer

from utils.index import Index
from utils.list import DimList

# To generate assembly, decorate the following function with `@export`
# and run `kgen -emit -S %s` on intel machines with non-temporal store
# support. There should be NT intructions in the assembly.
# E.x. on Skylake
# 	vmovaps	.LCPI1_1(%rip), %zmm0
# 	vmovntps	%zmm0, 128(%rsp)


# CHECK-LABEL: test_non_temporal_store
fn test_non_temporal_store(m: Int):
    print("== test_non_temporal_store")

    alias buffer_size = 128
    alias alignment = 64  # Bytes
    alias simd_width = simdwidthof[DType.float32]()

    let b1 = Buffer[DType.float32, buffer_size].stack_allocation()
    for i in range(buffer_size):
        b1[i] = Float32(i)

    let b2 = Buffer[DType.float32, buffer_size].aligned_stack_allocation[
        alignment
    ]()

    for i in range(0, buffer_size, simd_width):
        b2.simd_nt_store[simd_width](i, b1.simd_load[simd_width](i))

    let b3 = NDBuffer[
        DType.float32,
        2,
        DimList(buffer_size // simd_width, simd_width),
    ](b1.data)
    let b4 = NDBuffer[
        DType.float32,
        2,
        DimList(buffer_size // simd_width, simd_width),
    ](b2.data)

    for j in range(buffer_size // simd_width):
        b4.simd_nt_store[simd_width](
            Index(j, 0), b3.simd_load[simd_width](Index(j, 0))
        )

    # The following introduces dependency on the input value so that
    # the compiler won't optimize away the buffer copy. If we replace
    # `m` with the constant value, the simd load/store is optimized away.

    # CHECK: 0.0
    print(b2[m])

    # CHECK: 0.0
    print(b4[Index(0, m)])


fn main():
    test_non_temporal_store(0)
