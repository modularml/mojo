# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Roundtrip tests for the `TileCopier` structs in tile_io.mojo.

Each kernel moves a source tile through one or more intermediate address
spaces (SHARED, LOCAL) and back out to a destination DRAM tile, using
only `TileTensor` and the concrete `TileCopier` structs (no
`LayoutTensor` / `ManagedLayoutTensor`). The host verifies that the
destination tile matches the source.

Coverage:

- GENERIC -> SHARED -> GENERIC (unswizzled)
- GENERIC -> LOCAL -> GENERIC (unswizzled)
- GENERIC -> SHARED -> LOCAL -> SHARED -> GENERIC (exercises
  SharedToLocal and LocalToShared unswizzled paths)
- GENERIC -> LOCAL -> SHARED -> GENERIC with swizzle (exercises the
  correctness-critical swizzled paths of LocalToShared and
  SharedToGeneric, which must use identical swizzled addresses for
  writes and reads to round-trip).
- GENERIC -> SHARED (async cp.async) -> GENERIC (exercises
  GenericToSharedAsyncTileCopier on NVIDIA, falling back to a
  synchronous load/store on AMD/Apple). Covered for all three legal
  vector widths: 4 bytes (`element_size=1`, float32), 8 bytes
  (`vectorize[1, 2]`), and 16 bytes (`vectorize[1, 4]`); a 16-byte
  bf16 case (`vectorize[1, 8]`) covers the realistic
  half-precision matmul prologue shape.
- GENERIC -> SHARED (async cp.async, swizzled) -> GENERIC (swizzled)
  (exercises the swizzled write path of GenericToSharedAsyncTileCopier
  against the existing swizzled SharedToGeneric reader).
- GENERIC -> SHARED (async cp.async, masked=True with full src) ->
  GENERIC (smoke-tests the masked write path of
  GenericToSharedAsyncTileCopier; runtime-shape zero-fill verification
  is a follow-up — see the kernel's docstring).
"""

from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_all,
)

from layout import Idx, TileTensor, row_major
from layout.swizzle import Swizzle, make_swizzle
from layout.tile_io import (
    GenericToLocalTileCopier,
    GenericToSharedAsyncTileCopier,
    GenericToSharedTileCopier,
    LocalToGenericTileCopier,
    LocalToSharedTileCopier,
    SharedToGenericTileCopier,
    SharedToLocalTileCopier,
)
from layout.tile_tensor import stack_allocation

from std.testing import assert_equal


# 4x4 tile, distributed over 4 threads (2x2 thread layout -> each thread
# owns a 2x2 fragment).
comptime _N = 4
comptime _NUM_ELEMENTS = _N * _N
comptime _BLOCK_DIM = 4


def generic_to_shared_to_generic_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Roundtrip a tile through shared memory: GENERIC -> SHARED -> GENERIC."""
    comptime thread_layout = row_major(Idx[2], Idx[2])

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())
    var smem = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_N, _N]()
    )

    GenericToSharedTileCopier[thread_layout]().copy(smem, src)
    barrier()
    SharedToGenericTileCopier[thread_layout]().copy(dst, smem)


def generic_to_local_to_generic_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Roundtrip a tile through registers: GENERIC -> LOCAL -> GENERIC.

    Each thread holds its own 2x2 fragment in local memory.
    """
    comptime thread_layout = row_major(Idx[2], Idx[2])

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())
    # Per-thread fragment of a 4x4 tile under a 2x2 thread layout.
    var local = stack_allocation[dtype=DType.float32, address_space=.LOCAL](
        row_major[2, 2]()
    )

    GenericToLocalTileCopier[thread_layout]().copy(local, src)
    LocalToGenericTileCopier[thread_layout]().copy(dst, local)


def shared_local_shared_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Roundtrip a tile through shared -> local -> shared -> generic.

    GENERIC -> SHARED loads src into `smem_in`.
    SHARED -> LOCAL distributes `smem_in` into per-thread fragments.
    LOCAL -> SHARED writes those fragments into a second `smem_out`.
    SHARED -> GENERIC reads `smem_out` back to `dst`.
    """
    comptime thread_layout = row_major(Idx[2], Idx[2])

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())

    var smem_in = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_N, _N]()
    )
    var smem_out = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_N, _N]()
    )
    var local = stack_allocation[dtype=DType.float32, address_space=.LOCAL](
        row_major[2, 2]()
    )

    GenericToSharedTileCopier[thread_layout]().copy(smem_in, src)
    barrier()
    SharedToLocalTileCopier[thread_layout]().copy(local, smem_in)
    LocalToSharedTileCopier[thread_layout]().copy(smem_out, local)
    barrier()
    SharedToGenericTileCopier[thread_layout]().copy(dst, smem_out)


def swizzled_local_to_shared_to_generic_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Swizzled roundtrip: GENERIC -> LOCAL -> SHARED (swizzled) ->
    GENERIC (swizzled).

    Exercises the correctness-critical swizzled paths in LocalToShared
    and SharedToGeneric. A `None` swizzle on the SHARED -> GENERIC read
    would produce garbage (that copier's docstring flags this), so this
    test is the guard that the two swizzled branches compute identical
    swizzled addresses per thread / per index.
    """
    comptime thread_layout = row_major(Idx[2], Idx[2])
    comptime swizzle = Swizzle(1, 0, 2)

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())

    var smem = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_N, _N]()
    )
    var local = stack_allocation[dtype=DType.float32, address_space=.LOCAL](
        row_major[2, 2]()
    )

    GenericToLocalTileCopier[thread_layout]().copy(local, src)
    LocalToSharedTileCopier[thread_layout, swizzle=swizzle]().copy(smem, local)
    barrier()
    SharedToGenericTileCopier[thread_layout, swizzle=swizzle]().copy(dst, smem)


def async_generic_to_shared_to_generic_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Roundtrip a tile through shared memory using `cp.async` for the
    DRAM->SMEM leg: GENERIC -> SHARED (async) -> GENERIC.

    The async copy must be committed and waited on before the destination
    tile can be read back; on AMD / Apple the underlying intrinsic falls
    back to synchronous loads, but the commit/wait calls remain valid
    no-ops.
    """
    comptime thread_layout = row_major(Idx[2], Idx[2])

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())
    var smem = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_N, _N]()
    )

    GenericToSharedAsyncTileCopier[thread_layout]().copy(smem, src)
    async_copy_commit_group()
    async_copy_wait_all()
    barrier()
    SharedToGenericTileCopier[thread_layout]().copy(dst, smem)


def async_generic_to_shared_to_generic_8b_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Roundtrip with 8-byte (2x f32) async copies.

    Vectorizing the 4x4 tile by [1, 2] gives a 4x2 logical layout with
    `element_size=2`; under a 2x2 thread layout each thread issues two
    8-byte `cp.async` operations.
    """
    comptime thread_layout = row_major(Idx[2], Idx[2])

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())
    var smem = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_N, _N]()
    )

    GenericToSharedAsyncTileCopier[thread_layout]().copy(
        smem.vectorize[1, 2](), src.vectorize[1, 2]()
    )
    async_copy_commit_group()
    async_copy_wait_all()
    barrier()
    SharedToGenericTileCopier[thread_layout]().copy(dst, smem)


def async_generic_to_shared_to_generic_16b_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Roundtrip with 16-byte (4x f32) async copies.

    Vectorizing the 4x4 tile by [1, 4] gives a 4x1 logical layout with
    `element_size=4`; under a 4x1 thread layout each thread issues a
    single 16-byte `cp.async` covering one row.
    """
    comptime thread_layout = row_major(Idx[4], Idx[1])

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())
    var smem = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_N, _N]()
    )

    GenericToSharedAsyncTileCopier[thread_layout]().copy(
        smem.vectorize[1, 4](), src.vectorize[1, 4]()
    )
    async_copy_commit_group()
    async_copy_wait_all()
    barrier()
    SharedToGenericTileCopier[thread_layout]().copy(dst, smem)


# 4x8 bf16 tile: vectorize[1, 8] yields 16-byte (8x bf16) cp.async issues, the
# realistic prologue shape for half-precision matmul kernels.
comptime _BF16_ROWS = 4
comptime _BF16_COLS = 8
comptime _BF16_NUM_ELEMENTS = _BF16_ROWS * _BF16_COLS


def async_generic_to_shared_to_generic_16b_bf16_kernel(
    src_ptr: MutPointer[BFloat16, MutAnyOrigin],
    dst_ptr: MutPointer[BFloat16, MutAnyOrigin],
):
    """Roundtrip with 16-byte (8x bf16) async copies.

    Vectorizing the 4x8 bf16 tile by [1, 8] gives a 4x1 logical layout
    with `element_size=8`; under a 4x1 thread layout each thread issues
    a single 16-byte `cp.async` covering one row of bf16 values. This is
    the byte width and dtype the copier is most likely to see in real
    matmul prologues.
    """
    comptime thread_layout = row_major(Idx[4], Idx[1])

    var src = TileTensor(src_ptr, row_major[_BF16_ROWS, _BF16_COLS]())
    var dst = TileTensor(dst_ptr, row_major[_BF16_ROWS, _BF16_COLS]())
    var smem = stack_allocation[dtype=DType.bfloat16, address_space=.SHARED](
        row_major[_BF16_ROWS, _BF16_COLS]()
    )

    GenericToSharedAsyncTileCopier[thread_layout]().copy(
        smem.vectorize[1, 8](), src.vectorize[1, 8]()
    )
    async_copy_commit_group()
    async_copy_wait_all()
    barrier()
    SharedToGenericTileCopier[thread_layout]().copy(dst, smem)


def masked_async_generic_to_shared_to_generic_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Smoke-test the masked async write path with a fully in-bounds src.

    `masked=True` makes the copier compute a per-vector bound using
    `src.dim[0]() * row_stride - src_frag_offset` and skip vectors that
    fall past it. With a 4x4 static src that matches the smem layout,
    the bound covers every vector, so this test proves the masked code
    path compiles end-to-end and does not false-trigger zero-fills when
    the source is fully in bounds.

    Verifying the actual zero-fill behavior requires a runtime-shaped
    src (e.g. `Coord(Int(num_rows), Idx[_N])`) so `src.dim[0]()`
    can be smaller than the distribute layout's row count. That follow-up
    test is tracked alongside the production attention adopters that will
    exercise this path.
    """
    comptime thread_layout = row_major(Idx[2], Idx[2])

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())
    var smem = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_N, _N]()
    )

    GenericToSharedAsyncTileCopier[thread_layout, masked=True]().copy(smem, src)
    async_copy_commit_group()
    async_copy_wait_all()
    barrier()
    SharedToGenericTileCopier[thread_layout]().copy(dst, smem)


def access_size_swizzled_vectorized_async_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Production-shaped swizzle + vectorize on the async path.

    Uses `make_swizzle[..., access_size=simd_size]` so the swizzle
    `base = log2(simd_size)`, guaranteeing bits below `simd_size`
    are never permuted (and therefore cp.async destination offsets
    stay naturally aligned to `simd_size * size_of[dtype]()`).
    This is the shape mha/mla/qmatmul use in production.
    """
    comptime thread_layout = row_major(Idx[2], Idx[2])
    comptime simd_size = 2
    comptime swizzle = make_swizzle[
        num_rows=2, row_size=_N, access_size=simd_size
    ]()

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())
    var smem = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_N, _N]()
    )

    GenericToSharedAsyncTileCopier[thread_layout, swizzle=swizzle]().copy(
        smem.vectorize[1, simd_size](),
        src.vectorize[1, simd_size](),
    )
    async_copy_commit_group()
    async_copy_wait_all()
    barrier()
    SharedToGenericTileCopier[thread_layout, swizzle=swizzle]().copy(
        dst.vectorize[1, simd_size](),
        smem.vectorize[1, simd_size](),
    )


def swizzled_async_generic_to_shared_to_generic_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Swizzled async roundtrip: GENERIC -> SHARED (cp.async, swizzled) ->
    GENERIC (swizzled).

    Exercises the swizzled write path of `GenericToSharedAsyncTileCopier`.
    The destination tile is read back with `SharedToGenericTileCopier`
    using the same swizzle, so any disagreement between the two
    swizzled-address calculations would surface as a mismatched roundtrip.
    """
    comptime thread_layout = row_major(Idx[2], Idx[2])
    comptime swizzle = Swizzle(1, 0, 2)

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())
    var smem = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_N, _N]()
    )

    GenericToSharedAsyncTileCopier[thread_layout, swizzle=swizzle]().copy(
        smem, src
    )
    async_copy_commit_group()
    async_copy_wait_all()
    barrier()
    SharedToGenericTileCopier[thread_layout, swizzle=swizzle]().copy(dst, smem)


def _run_roundtrip[
    kernel_fn: def(
        MutPointer[Float32, MutAnyOrigin],
        MutPointer[Float32, MutAnyOrigin],
    ) thin -> None,
](name: String, ctx: DeviceContext) raises:
    print("==", name)

    var src_host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    for i in range(_NUM_ELEMENTS):
        src_host[i] = Float32(i + 1)

    var src_dev = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    var dst_dev = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(src_dev, src_host)

    ctx.enqueue_function[kernel_fn](
        src_dev, dst_dev, grid_dim=(1), block_dim=(_BLOCK_DIM)
    )

    var dst_host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(dst_host, dst_dev)
    ctx.synchronize()

    for i in range(_NUM_ELEMENTS):
        assert_equal(dst_host[i], src_host[i])


def test_generic_to_shared_to_generic(ctx: DeviceContext) raises:
    _run_roundtrip[generic_to_shared_to_generic_kernel](
        "test_generic_to_shared_to_generic", ctx
    )


def test_generic_to_local_to_generic(ctx: DeviceContext) raises:
    _run_roundtrip[generic_to_local_to_generic_kernel](
        "test_generic_to_local_to_generic", ctx
    )


def test_shared_local_shared_roundtrip(ctx: DeviceContext) raises:
    _run_roundtrip[shared_local_shared_kernel](
        "test_shared_local_shared_roundtrip", ctx
    )


def test_swizzled_local_to_shared_to_generic(ctx: DeviceContext) raises:
    _run_roundtrip[swizzled_local_to_shared_to_generic_kernel](
        "test_swizzled_local_to_shared_to_generic", ctx
    )


def test_async_generic_to_shared_to_generic(ctx: DeviceContext) raises:
    _run_roundtrip[async_generic_to_shared_to_generic_kernel](
        "test_async_generic_to_shared_to_generic", ctx
    )


def test_async_generic_to_shared_to_generic_8b(ctx: DeviceContext) raises:
    _run_roundtrip[async_generic_to_shared_to_generic_8b_kernel](
        "test_async_generic_to_shared_to_generic_8b", ctx
    )


def test_async_generic_to_shared_to_generic_16b(ctx: DeviceContext) raises:
    _run_roundtrip[async_generic_to_shared_to_generic_16b_kernel](
        "test_async_generic_to_shared_to_generic_16b", ctx
    )


def test_async_generic_to_shared_to_generic_16b_bf16(
    ctx: DeviceContext,
) raises:
    var name = "test_async_generic_to_shared_to_generic_16b_bf16"
    print("==", name)

    var src_host = ctx.enqueue_create_host_buffer[.bfloat16](_BF16_NUM_ELEMENTS)
    for i in range(_BF16_NUM_ELEMENTS):
        src_host[i] = BFloat16(i + 1)

    var src_dev = ctx.enqueue_create_buffer[.bfloat16](_BF16_NUM_ELEMENTS)
    var dst_dev = ctx.enqueue_create_buffer[.bfloat16](_BF16_NUM_ELEMENTS)
    ctx.enqueue_copy(src_dev, src_host)

    ctx.enqueue_function[async_generic_to_shared_to_generic_16b_bf16_kernel](
        src_dev, dst_dev, grid_dim=(1), block_dim=(_BF16_ROWS)
    )

    var dst_host = ctx.enqueue_create_host_buffer[.bfloat16](_BF16_NUM_ELEMENTS)
    ctx.enqueue_copy(dst_host, dst_dev)
    ctx.synchronize()

    for i in range(_BF16_NUM_ELEMENTS):
        assert_equal(dst_host[i], src_host[i])


def test_swizzled_async_generic_to_shared_to_generic(
    ctx: DeviceContext,
) raises:
    _run_roundtrip[swizzled_async_generic_to_shared_to_generic_kernel](
        "test_swizzled_async_generic_to_shared_to_generic", ctx
    )


def test_masked_async_generic_to_shared_to_generic(
    ctx: DeviceContext,
) raises:
    _run_roundtrip[masked_async_generic_to_shared_to_generic_kernel](
        "test_masked_async_generic_to_shared_to_generic", ctx
    )


def test_access_size_swizzled_vectorized_async(ctx: DeviceContext) raises:
    _run_roundtrip[access_size_swizzled_vectorized_async_kernel](
        "test_access_size_swizzled_vectorized_async", ctx
    )


def main() raises:
    with DeviceContext() as ctx:
        test_generic_to_shared_to_generic(ctx)
        test_generic_to_local_to_generic(ctx)
        test_shared_local_shared_roundtrip(ctx)
        test_swizzled_local_to_shared_to_generic(ctx)
        test_async_generic_to_shared_to_generic(ctx)
        test_async_generic_to_shared_to_generic_8b(ctx)
        test_async_generic_to_shared_to_generic_16b(ctx)
        test_async_generic_to_shared_to_generic_16b_bf16(ctx)
        test_swizzled_async_generic_to_shared_to_generic(ctx)
        test_masked_async_generic_to_shared_to_generic(ctx)
        test_access_size_swizzled_vectorized_async(ctx)
