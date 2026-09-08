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
"""Benchmarks tile_io copy helpers against LayoutTensor copy helpers."""

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu import thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.host.compile import get_gpu_target
from std.memory import bitcast
from std.os import abort
from std.sys import (
    align_of,
    get_defined_dtype,
    get_defined_int,
    simd_width_of,
    size_of,
)
from std.testing import assert_equal

from layout import Idx, Layout, LayoutTensor, TileTensor, row_major
from layout.layout_tensor import copy_dram_to_sram, copy_sram_to_dram
from layout.tile_io import GenericToSharedTileCopier, SharedToGenericTileCopier
from layout.tile_tensor import stack_allocation


@fieldwise_init
struct _CopyDirection(Equatable, TrivialRegisterPassable):
    var _value: Int32

    comptime DRAMToSRAM = Self(0)
    comptime SRAMToDRAM = Self(1)

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return self._value != other._value


@always_inline
def _thread_offset[
    N: Int, thread_cols: Int, simd_size: Int
](worker_idx: Int) -> Int:
    return (worker_idx // thread_cols) * N + (
        worker_idx % thread_cols
    ) * simd_size


@always_inline
def _manual_copy[
    dtype: DType,
    M: Int,
    N: Int,
    thread_rows: Int,
    thread_cols: Int,
    simd_size: Int,
    direction: _CopyDirection,
](
    dram_src_ptr: ImmPointer[Scalar[dtype], ...],
    dram_dst_ptr: MutPointer[Scalar[dtype], ...],
    smem_ptr: MutPointer[Scalar[dtype], address_space=.SHARED, ...],
):
    """Copies one leg manually using the benchmark's per-thread layout."""
    comptime rows_per_thread = M // thread_rows
    comptime row_stride = thread_rows * N
    comptime alignment = align_of[SIMD[dtype, simd_size]]()
    var base = _thread_offset[N, thread_cols, simd_size](Int(thread_idx.x))

    comptime for i in range(rows_per_thread):
        var offset = base + i * row_stride
        comptime if direction == _CopyDirection.DRAMToSRAM:
            smem_ptr.store[alignment=alignment](
                offset,
                dram_src_ptr.load[width=simd_size, alignment=alignment](offset),
            )
        else:
            dram_dst_ptr.store[alignment=alignment](
                offset,
                smem_ptr.load[width=simd_size, alignment=alignment](offset),
            )


@always_inline
def _tile_io_copy[
    dtype: DType,
    M: Int,
    N: Int,
    thread_rows: Int,
    thread_cols: Int,
    simd_size: Int,
    tile_io_dram_to_sram: Bool,
    tile_io_sram_to_dram: Bool,
](
    src_ptr: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    dst_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
):
    comptime thread_layout = row_major(Idx[thread_rows], Idx[thread_cols])

    var smem = stack_allocation[dtype=dtype, address_space=.SHARED](
        row_major[M, N]()
    ).vectorize[1, simd_size]()

    comptime if tile_io_dram_to_sram:
        var src = TileTensor(src_ptr, row_major[M, N]()).vectorize[
            1, simd_size
        ]()
        GenericToSharedTileCopier[thread_layout]().copy(smem, src)
    else:
        _manual_copy[
            dtype,
            M,
            N,
            thread_rows,
            thread_cols,
            simd_size,
            _CopyDirection.DRAMToSRAM,
        ](src_ptr, dst_ptr, smem.ptr)

    barrier()

    comptime if tile_io_sram_to_dram:
        var dst = TileTensor(dst_ptr, row_major[M, N]()).vectorize[
            1, simd_size
        ]()
        SharedToGenericTileCopier[thread_layout]().copy(dst, smem)
    else:
        _manual_copy[
            dtype,
            M,
            N,
            thread_rows,
            thread_cols,
            simd_size,
            _CopyDirection.SRAMToDRAM,
        ](src_ptr, dst_ptr, smem.ptr)


@always_inline
def _layout_tensor_copy[
    dtype: DType,
    tensor_layout: Layout,
    M: Int,
    N: Int,
    thread_rows: Int,
    thread_cols: Int,
    simd_size: Int,
    layout_tensor_dram_to_sram: Bool,
    layout_tensor_sram_to_dram: Bool,
](
    src_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    dst_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
):
    comptime thread_layout = Layout.row_major(thread_rows, thread_cols)

    var smem = LayoutTensor[
        dtype,
        tensor_layout,
        MutAnyOrigin,
        address_space=.SHARED,
    ].stack_allocation()

    comptime if layout_tensor_dram_to_sram:
        var src = LayoutTensor[dtype, tensor_layout, MutAnyOrigin](src_ptr)
        copy_dram_to_sram[thread_layout=thread_layout](
            smem.vectorize[1, simd_size](), src.vectorize[1, simd_size]()
        )
    else:
        _manual_copy[
            dtype,
            M,
            N,
            thread_rows,
            thread_cols,
            simd_size,
            _CopyDirection.DRAMToSRAM,
        ](src_ptr, dst_ptr, smem.ptr)

    barrier()

    comptime if layout_tensor_sram_to_dram:
        var dst = LayoutTensor[dtype, tensor_layout, MutAnyOrigin](dst_ptr)
        copy_sram_to_dram[thread_layout=thread_layout](
            dst.vectorize[1, simd_size](), smem.vectorize[1, simd_size]()
        )
    else:
        _manual_copy[
            dtype,
            M,
            N,
            thread_rows,
            thread_cols,
            simd_size,
            _CopyDirection.SRAMToDRAM,
        ](src_ptr, dst_ptr, smem.ptr)


def tile_io_copy_roundtrip_kernel[
    dtype: DType,
    M: Int,
    N: Int,
    thread_rows: Int,
    thread_cols: Int,
    simd_size: Int,
](
    src_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    dst_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
):
    """Copies a TileTensor from global to shared memory and back."""
    _tile_io_copy[dtype, M, N, thread_rows, thread_cols, simd_size, True, True](
        src_ptr, dst_ptr
    )


def tile_io_dram_to_sram_kernel[
    dtype: DType,
    M: Int,
    N: Int,
    thread_rows: Int,
    thread_cols: Int,
    simd_size: Int,
](
    src_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    dst_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
):
    """Benchmarks the tile_io global-to-shared copy with a common drain."""
    _tile_io_copy[
        dtype, M, N, thread_rows, thread_cols, simd_size, True, False
    ](src_ptr, dst_ptr)


def tile_io_sram_to_dram_kernel[
    dtype: DType,
    M: Int,
    N: Int,
    thread_rows: Int,
    thread_cols: Int,
    simd_size: Int,
](
    src_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    dst_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
):
    """Benchmarks the tile_io shared-to-global copy with a common fill."""
    _tile_io_copy[
        dtype, M, N, thread_rows, thread_cols, simd_size, False, True
    ](src_ptr, dst_ptr)


def layout_tensor_copy_roundtrip_kernel[
    dtype: DType,
    tensor_layout: Layout,
    M: Int,
    N: Int,
    thread_rows: Int,
    thread_cols: Int,
    simd_size: Int,
](
    src: LayoutTensor[dtype, tensor_layout, MutAnyOrigin],
    dst: LayoutTensor[dtype, tensor_layout, MutAnyOrigin],
):
    """Copies a LayoutTensor from global to shared memory and back."""
    _layout_tensor_copy[
        dtype,
        tensor_layout,
        M,
        N,
        thread_rows,
        thread_cols,
        simd_size,
        True,
        True,
    ](src.ptr, dst.ptr)


def layout_tensor_dram_to_sram_kernel[
    dtype: DType,
    tensor_layout: Layout,
    M: Int,
    N: Int,
    thread_rows: Int,
    thread_cols: Int,
    simd_size: Int,
](
    src: LayoutTensor[dtype, tensor_layout, MutAnyOrigin],
    dst_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
):
    """Benchmarks LayoutTensor DRAM-to-SRAM copy with a common drain."""
    _layout_tensor_copy[
        dtype,
        tensor_layout,
        M,
        N,
        thread_rows,
        thread_cols,
        simd_size,
        True,
        False,
    ](src.ptr, dst_ptr)


def layout_tensor_sram_to_dram_kernel[
    dtype: DType,
    tensor_layout: Layout,
    M: Int,
    N: Int,
    thread_rows: Int,
    thread_cols: Int,
    simd_size: Int,
](
    src_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    dst: LayoutTensor[dtype, tensor_layout, MutAnyOrigin],
):
    """Benchmarks LayoutTensor SRAM-to-DRAM copy with a common fill."""
    _layout_tensor_copy[
        dtype,
        tensor_layout,
        M,
        N,
        thread_rows,
        thread_cols,
        simd_size,
        False,
        True,
    ](src_ptr, dst.ptr)


@always_inline
def _assert_buffers_equal[
    dtype: DType, simd_size: Int
](
    actual_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    expected_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    num_elements: Int,
    label: String,
) raises:
    """Checks host buffers a SIMD vector at a time."""
    var i = 0
    while i + simd_size <= num_elements:
        var actual = actual_ptr.load[width=simd_size, alignment=1](i)
        var expected = expected_ptr.load[width=simd_size, alignment=1](i)
        comptime if dtype.is_float8():
            var actual_bits = bitcast[.uint8, simd_size](actual)
            var expected_bits = bitcast[.uint8, simd_size](expected)
            if actual_bits != expected_bits:
                for lane in range(simd_size):
                    assert_equal(
                        actual_bits[lane],
                        expected_bits[lane],
                        String(label, " mismatch at element ", i + lane),
                    )
        else:
            if actual != expected:
                for lane in range(simd_size):
                    assert_equal(
                        actual[lane],
                        expected[lane],
                        String(label, " mismatch at element ", i + lane),
                    )
        i += simd_size

    while i < num_elements:
        comptime if dtype.is_float8():
            assert_equal(
                bitcast[.uint8](actual_ptr[i]),
                bitcast[.uint8](expected_ptr[i]),
                String(label, " mismatch at element ", i),
            )
        else:
            assert_equal(
                actual_ptr[i],
                expected_ptr[i],
                String(label, " mismatch at element ", i),
            )
        i += 1


def bench_copy_roundtrip[
    dtype: DType,
    M: Int,
    N: Int,
    num_threads: Int,
](ctx: DeviceContext, mut bench: Bench) raises:
    comptime simd_size = simd_width_of[dtype, target=get_gpu_target()]()
    comptime tensor_layout = Layout.row_major(M, N)

    comptime assert (
        N % simd_size == 0
    ), "N must be divisible by the dtype SIMD width."
    comptime assert (
        num_threads * simd_size
    ) % N == 0, "num_threads * simd_width must be divisible by N."

    comptime thread_rows = num_threads * simd_size // N
    comptime thread_cols = N // simd_size
    comptime num_elements = M * N

    var src_host = ctx.enqueue_create_host_buffer[dtype](num_elements)
    for i in range(num_elements):
        # Uses a small prime modulus to create a deterministic, non-power-of-two
        # input pattern while keeping values compact across dtypes.
        src_host[i] = Float32(i % 251).cast[dtype]()

    var src_dev = ctx.enqueue_create_buffer[dtype](num_elements)
    var tile_io_dst_dev = ctx.enqueue_create_buffer[dtype](num_elements)
    var layout_tensor_dst_dev = ctx.enqueue_create_buffer[dtype](num_elements)
    ctx.enqueue_copy(src_dev, src_host)

    var src_tensor = LayoutTensor[dtype, tensor_layout](src_dev)
    var layout_tensor_dst = LayoutTensor[dtype, tensor_layout](
        layout_tensor_dst_dev
    )

    comptime tile_io_kernel = tile_io_copy_roundtrip_kernel[
        dtype, M, N, thread_rows, thread_cols, simd_size
    ]
    comptime layout_tensor_kernel = layout_tensor_copy_roundtrip_kernel[
        dtype, tensor_layout, M, N, thread_rows, thread_cols, simd_size
    ]
    comptime tile_io_dram_to_sram_kernel_type = tile_io_dram_to_sram_kernel[
        dtype, M, N, thread_rows, thread_cols, simd_size
    ]
    comptime layout_tensor_dram_to_sram_kernel_type = (
        layout_tensor_dram_to_sram_kernel[
            dtype, tensor_layout, M, N, thread_rows, thread_cols, simd_size
        ]
    )
    comptime tile_io_sram_to_dram_kernel_type = tile_io_sram_to_dram_kernel[
        dtype, M, N, thread_rows, thread_cols, simd_size
    ]
    comptime layout_tensor_sram_to_dram_kernel_type = (
        layout_tensor_sram_to_dram_kernel[
            dtype, tensor_layout, M, N, thread_rows, thread_cols, simd_size
        ]
    )

    ctx.enqueue_function[tile_io_kernel](
        src_dev,
        tile_io_dst_dev,
        grid_dim=(1,),
        block_dim=(num_threads,),
    )
    ctx.enqueue_function[layout_tensor_kernel](
        src_tensor,
        layout_tensor_dst,
        grid_dim=(1,),
        block_dim=(num_threads,),
    )
    var tile_io_dst_host = ctx.enqueue_create_host_buffer[dtype](num_elements)
    var layout_tensor_dst_host = ctx.enqueue_create_host_buffer[dtype](
        num_elements
    )
    ctx.enqueue_copy(tile_io_dst_host, tile_io_dst_dev)
    ctx.enqueue_copy(layout_tensor_dst_host, layout_tensor_dst_dev)
    ctx.synchronize()
    _assert_buffers_equal[dtype, simd_size](
        tile_io_dst_host.unsafe_ptr(),
        src_host.unsafe_ptr(),
        num_elements,
        "tile_io_copy_roundtrip",
    )
    _assert_buffers_equal[dtype, simd_size](
        layout_tensor_dst_host.unsafe_ptr(),
        src_host.unsafe_ptr(),
        num_elements,
        "layout_tensor_copy_roundtrip",
    )

    ctx.enqueue_function[tile_io_dram_to_sram_kernel_type](
        src_dev,
        tile_io_dst_dev,
        grid_dim=(1,),
        block_dim=(num_threads,),
    )
    ctx.enqueue_function[layout_tensor_dram_to_sram_kernel_type](
        src_tensor,
        layout_tensor_dst_dev,
        grid_dim=(1,),
        block_dim=(num_threads,),
    )
    ctx.enqueue_copy(tile_io_dst_host, tile_io_dst_dev)
    ctx.enqueue_copy(layout_tensor_dst_host, layout_tensor_dst_dev)
    ctx.synchronize()
    _assert_buffers_equal[dtype, simd_size](
        tile_io_dst_host.unsafe_ptr(),
        src_host.unsafe_ptr(),
        num_elements,
        "tile_io_dram_to_sram",
    )
    _assert_buffers_equal[dtype, simd_size](
        layout_tensor_dst_host.unsafe_ptr(),
        src_host.unsafe_ptr(),
        num_elements,
        "layout_tensor_dram_to_sram",
    )

    ctx.enqueue_function[tile_io_sram_to_dram_kernel_type](
        src_dev,
        tile_io_dst_dev,
        grid_dim=(1,),
        block_dim=(num_threads,),
    )
    ctx.enqueue_function[layout_tensor_sram_to_dram_kernel_type](
        src_dev,
        layout_tensor_dst,
        grid_dim=(1,),
        block_dim=(num_threads,),
    )
    ctx.enqueue_copy(tile_io_dst_host, tile_io_dst_dev)
    ctx.enqueue_copy(layout_tensor_dst_host, layout_tensor_dst_dev)
    ctx.synchronize()
    _assert_buffers_equal[dtype, simd_size](
        tile_io_dst_host.unsafe_ptr(),
        src_host.unsafe_ptr(),
        num_elements,
        "tile_io_sram_to_dram",
    )
    _assert_buffers_equal[dtype, simd_size](
        layout_tensor_dst_host.unsafe_ptr(),
        src_host.unsafe_ptr(),
        num_elements,
        "layout_tensor_sram_to_dram",
    )

    @always_inline
    def bench_tile_io_roundtrip(mut b: Bencher) {var}:
        @always_inline
        def tile_io_roundtrip_launch(ctx: DeviceContext) raises {var}:
            ctx.enqueue_function[tile_io_kernel](
                src_dev,
                tile_io_dst_dev,
                grid_dim=(1,),
                block_dim=(num_threads,),
            )

        bencher_iter_custom(b, tile_io_roundtrip_launch, ctx)

    @always_inline
    def bench_layout_tensor_roundtrip(mut b: Bencher) {var}:
        @always_inline
        def layout_tensor_roundtrip_launch(ctx: DeviceContext) raises {var}:
            var src = LayoutTensor[dtype, tensor_layout, MutAnyOrigin](
                src_dev.unsafe_ptr()
            )
            var dst = LayoutTensor[dtype, tensor_layout, MutAnyOrigin](
                layout_tensor_dst_dev.unsafe_ptr()
            )
            ctx.enqueue_function[layout_tensor_kernel](
                src,
                dst,
                grid_dim=(1,),
                block_dim=(num_threads,),
            )

        bencher_iter_custom(b, layout_tensor_roundtrip_launch, ctx)

    @always_inline
    def bench_tile_io_dram_to_sram(mut b: Bencher) {var}:
        @always_inline
        def tile_io_dram_to_sram_launch(ctx: DeviceContext) raises {var}:
            ctx.enqueue_function[tile_io_dram_to_sram_kernel_type](
                src_dev,
                tile_io_dst_dev,
                grid_dim=(1,),
                block_dim=(num_threads,),
            )

        bencher_iter_custom(b, tile_io_dram_to_sram_launch, ctx)

    @always_inline
    def bench_layout_tensor_dram_to_sram(mut b: Bencher) {var}:
        @always_inline
        def layout_tensor_dram_to_sram_launch(ctx: DeviceContext) raises {var}:
            var src = LayoutTensor[dtype, tensor_layout, MutAnyOrigin](
                src_dev.unsafe_ptr()
            )
            ctx.enqueue_function[layout_tensor_dram_to_sram_kernel_type](
                src,
                layout_tensor_dst_dev,
                grid_dim=(1,),
                block_dim=(num_threads,),
            )

        bencher_iter_custom(b, layout_tensor_dram_to_sram_launch, ctx)

    @always_inline
    def bench_tile_io_sram_to_dram(mut b: Bencher) {var}:
        @always_inline
        def tile_io_sram_to_dram_launch(ctx: DeviceContext) raises {var}:
            ctx.enqueue_function[tile_io_sram_to_dram_kernel_type](
                src_dev,
                tile_io_dst_dev,
                grid_dim=(1,),
                block_dim=(num_threads,),
            )

        bencher_iter_custom(b, tile_io_sram_to_dram_launch, ctx)

    @always_inline
    def bench_layout_tensor_sram_to_dram(mut b: Bencher) {var}:
        @always_inline
        def layout_tensor_sram_to_dram_launch(ctx: DeviceContext) raises {var}:
            var dst = LayoutTensor[dtype, tensor_layout, MutAnyOrigin](
                layout_tensor_dst_dev.unsafe_ptr()
            )
            ctx.enqueue_function[layout_tensor_sram_to_dram_kernel_type](
                src_dev,
                dst,
                grid_dim=(1,),
                block_dim=(num_threads,),
            )

        bencher_iter_custom(b, layout_tensor_sram_to_dram_launch, ctx)

    comptime roundtrip_bytes = 2 * num_elements * size_of[dtype]()
    comptime one_way_bytes = num_elements * size_of[dtype]()
    var roundtrip_bandwidth = ThroughputMeasure(
        BenchMetric.bytes, roundtrip_bytes
    )
    var one_way_bandwidth = ThroughputMeasure(BenchMetric.bytes, one_way_bytes)
    var input_id = String(M, "x", N, "_threads", num_threads, "_", dtype)

    bench.bench_function(
        bench_tile_io_roundtrip,
        BenchId("tile_io_copy_roundtrip", input_id=input_id),
        [roundtrip_bandwidth],
    )
    bench.bench_function(
        bench_layout_tensor_roundtrip,
        BenchId("layout_tensor_copy_roundtrip", input_id=input_id),
        [roundtrip_bandwidth],
    )
    bench.bench_function(
        bench_tile_io_dram_to_sram,
        BenchId("tile_io_dram_to_sram", input_id=input_id),
        [one_way_bandwidth],
    )
    bench.bench_function(
        bench_layout_tensor_dram_to_sram,
        BenchId("layout_tensor_dram_to_sram", input_id=input_id),
        [one_way_bandwidth],
    )
    bench.bench_function(
        bench_tile_io_sram_to_dram,
        BenchId("tile_io_sram_to_dram", input_id=input_id),
        [one_way_bandwidth],
    )
    bench.bench_function(
        bench_layout_tensor_sram_to_dram,
        BenchId("layout_tensor_sram_to_dram", input_id=input_id),
        [one_way_bandwidth],
    )


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .float32]()
    comptime M = get_defined_int["M", 64]()
    comptime N = get_defined_int["N", 64]()
    comptime num_threads = get_defined_int["num_threads", 128]()

    var bench = Bench()
    with DeviceContext() as ctx:
        bench_copy_roundtrip[dtype, M, N, num_threads](ctx, bench)

    bench.dump_report()
