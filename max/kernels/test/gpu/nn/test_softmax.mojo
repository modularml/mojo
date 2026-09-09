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

from std.math import isclose, isnan
from std.utils.numerics import min_or_neg_inf
from std.random import rand, random_float64, seed
from std.sys import has_amd_gpu_accelerator, simd_width_of

from max.gpu import WARP_SIZE
from max.gpu.host import DeviceContext, get_gpu_target
from layout import (
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    coord_to_index_list,
    row_major,
)
from layout._utils import ManagedLayoutTensor
from nn.softmax import (
    _online_softmax_kernel,
    _softmax_cpu,
    _softmax_gpu,
    softmax_with_temperature,
)
from std.testing import assert_almost_equal, assert_true

from std.utils import IndexList


def test_gpu_softmax(ctx: DeviceContext) raises:
    print("== test_gpu_softmax")

    comptime type = DType.float32
    comptime rank = 3
    var shape = IndexList[rank](3, 5, 515)
    var in_host_ptr = ctx.enqueue_create_host_buffer[type](
        shape.flattened_length()
    )
    var in_device_ptr = ctx.enqueue_create_buffer[type](
        shape.flattened_length()
    )
    comptime layout_dyn = Layout.row_major[rank]()
    var in_host = LayoutTensor[type, layout_dyn](
        in_host_ptr, RuntimeLayout[layout_dyn].row_major(shape)
    )
    var in_device = LayoutTensor[type, layout_dyn](
        in_device_ptr.unsafe_ptr(), RuntimeLayout[layout_dyn].row_major(shape)
    )
    var out_host_ptr = ctx.enqueue_create_host_buffer[type](
        shape.flattened_length()
    )
    var out_ref_ptr = ctx.enqueue_create_host_buffer[type](
        shape.flattened_length()
    )
    var out_device_ptr = ctx.enqueue_create_buffer[type](
        shape.flattened_length()
    )
    var out_host = LayoutTensor[type, layout_dyn](
        out_host_ptr, RuntimeLayout[layout_dyn].row_major(shape)
    )
    var out_ref = LayoutTensor[type, layout_dyn](
        out_ref_ptr, RuntimeLayout[layout_dyn].row_major(shape)
    )
    rand[type](in_host_ptr.as_span())
    ctx.enqueue_copy(in_device_ptr, in_host_ptr)

    @__parameter
    @__copy_capture(in_device)
    def input_fn_device[
        _simd_width: Int
    ](coords: Coord) -> SIMD[type, _simd_width]:
        return in_device.load[width=_simd_width](coord_to_index_list(coords))

    @__parameter
    @__copy_capture(in_host)
    def input_fn_host[
        _simd_width: Int
    ](coords: Coord) -> SIMD[type, _simd_width]:
        return in_host.load[width=_simd_width](coord_to_index_list(coords))

    _softmax_gpu[type, 1, rank, input_fn_device](
        Coord(shape),
        TileTensor(out_device_ptr, row_major(Coord(shape))),
        rank - 1,
        ctx,
    )

    _softmax_cpu[type, 1, rank, origin_of()._mlir_origin, input_fn_host](
        Coord(shape),
        TileTensor(out_ref.ptr, row_major(Coord(shape))),
        rank - 1,
    )

    ctx.synchronize()
    ctx.enqueue_copy(out_host_ptr, out_device_ptr)
    ctx.synchronize()

    for i in range(shape.flattened_length()):
        if not isclose(
            LayoutTensor[out_ref.dtype, Layout.row_major(UNKNOWN_VALUE)](
                out_ref.ptr,
                RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                    IndexList[1](out_ref.size())
                ),
            )[i],
            LayoutTensor[out_host.dtype, Layout.row_major(UNKNOWN_VALUE)](
                out_host.ptr,
                RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                    IndexList[1](out_host.size())
                ),
            )[i],
            atol=1e-4,
            rtol=1e-5,
        ):
            print("ERROR. Mismatch at flattened idx:", i)
            assert_true(False)

    _ = in_device
    _ = in_host
    _ = in_device_ptr
    _ = out_device_ptr


def test_gpu_softmax_half[test_type: DType](ctx: DeviceContext) raises:
    print("== test_gpu_softmax_half")
    comptime seed_val = 42
    seed(seed_val)

    comptime ref_type = DType.float32
    comptime rank = 3

    var shape = IndexList[rank](3, 5, 515)
    var length = shape.flattened_length()

    comptime layout_dyn = Layout.row_major[rank]()

    var in_host_ref_ptr = alloc[Scalar[ref_type]](length)
    var in_device_ref_ptr = ctx.enqueue_create_buffer[ref_type](length)
    var in_host_test_ptr = alloc[Scalar[test_type]](length)
    var in_device_test_ptr = ctx.enqueue_create_buffer[test_type](length)
    var in_device_ref = LayoutTensor[ref_type, layout_dyn](
        in_device_ref_ptr.unsafe_ptr(),
        RuntimeLayout[layout_dyn].row_major(shape),
    )
    var in_device_test = LayoutTensor[test_type, layout_dyn](
        in_device_test_ptr.unsafe_ptr(),
        RuntimeLayout[layout_dyn].row_major(shape),
    )

    var out_host_ref_ptr = alloc[Scalar[ref_type]](length)
    var out_device_ref_ptr = ctx.enqueue_create_buffer[ref_type](length)
    var out_host_test_ptr = alloc[Scalar[test_type]](length)
    var out_device_test_ptr = ctx.enqueue_create_buffer[test_type](length)

    # first fill BF16 pointer with random values, then cast to FP32 to
    # circumvent precision loss on casting of input. Skew the values to simulate
    # precision loss
    for i in range(length):
        # TODO use randn when GCC Float64 -> Float16 truncation is fixed #33932
        in_host_test_ptr[i] = (
            random_float64(1, 10).cast[.float32]().cast[test_type]()
        )
        in_host_ref_ptr[i] = in_host_test_ptr[i].cast[ref_type]()

    ctx.enqueue_copy(in_device_test_ptr, in_host_test_ptr)
    ctx.enqueue_copy(in_device_ref_ptr, in_host_ref_ptr)

    @__parameter
    @__copy_capture(in_device_ref)
    def input_fn_ref[
        _simd_width: Int
    ](coords: Coord) -> SIMD[ref_type, _simd_width]:
        return in_device_ref.load[width=_simd_width](
            coord_to_index_list(coords)
        )

    @__parameter
    @__copy_capture(in_device_test)
    def input_fn_test[
        _simd_width: Int
    ](coords: Coord) -> SIMD[test_type, _simd_width]:
        return in_device_test.load[width=_simd_width](
            coord_to_index_list(coords)
        )

    _softmax_gpu[ref_type, 1, rank, input_fn_ref](
        Coord(shape),
        TileTensor(out_device_ref_ptr, row_major(Coord(shape))),
        rank - 1,
        ctx,
    )

    _softmax_gpu[test_type, 1, rank, input_fn_test](
        Coord(shape),
        TileTensor(out_device_test_ptr, row_major(Coord(shape))),
        rank - 1,
        ctx,
    )

    ctx.synchronize()
    ctx.enqueue_copy(out_host_ref_ptr, out_device_ref_ptr)
    ctx.enqueue_copy(out_host_test_ptr, out_device_test_ptr)
    ctx.synchronize()

    for i in range(length):
        var ref_val = out_host_ref_ptr[i]
        var test_val = out_host_test_ptr[i].cast[ref_type]()
        assert_almost_equal(ref_val, test_val, atol=1e-2)

    _ = in_device_ref_ptr
    _ = in_device_test_ptr
    _ = in_device_test
    _ = in_device_ref


def test_gpu_softmax_warp_short_axis[
    rank: Int,
    //,
    test_type: DType,
    shape: IndexList[rank],
    has_prologue_fusion: Bool = False,
](ctx: DeviceContext) raises:
    """Regression for `_softmax_gpu` over the inner axis of a rank-`rank`
    tensor of static `shape`.

    Inner axis `shape[rank-1] <= WARP_SIZE` takes the warp-per-row fast path;
    """
    print(
        "== test_gpu_softmax_warp_short_axis",
        test_type,
        shape,
        "has_prologue_fusion=",
        has_prologue_fusion,
    )
    seed(42)

    comptime ref_type = DType.float32
    comptime row_size = shape[rank - 1]
    var length = shape.flattened_length()
    var num_rows = length // row_size

    comptime layout_dyn = Layout.row_major[rank]()

    var in_host_test_ptr = alloc[Scalar[test_type]](length)
    var in_device_test_ptr = ctx.enqueue_create_buffer[test_type](length)
    var in_host_test = LayoutTensor[test_type, layout_dyn](
        in_host_test_ptr,
        RuntimeLayout[layout_dyn].row_major(shape),
    )
    var in_device_test = LayoutTensor[test_type, layout_dyn](
        in_device_test_ptr.unsafe_ptr(),
        RuntimeLayout[layout_dyn].row_major(shape),
    )

    var out_host_test_ptr = alloc[Scalar[test_type]](length)
    var out_device_test_ptr = ctx.enqueue_create_buffer[test_type](length)
    var out_ref_ptr = alloc[Scalar[ref_type]](length)

    for i in range(length):
        in_host_test_ptr[i] = (
            random_float64(-3, 3).cast[ref_type]().cast[test_type]()
        )

    ctx.enqueue_copy(in_device_test_ptr, in_host_test_ptr)

    @__parameter
    @__copy_capture(in_device_test)
    def input_fn_device[
        _simd_width: Int
    ](coords: Coord) -> SIMD[test_type, _simd_width]:
        return in_device_test.load[width=_simd_width](
            coord_to_index_list(coords)
        )

    @__parameter
    @__copy_capture(in_host_test)
    def input_fn_host[
        _simd_width: Int
    ](coords: Coord) -> SIMD[ref_type, _simd_width]:
        return in_host_test.load[width=_simd_width](
            coord_to_index_list(coords)
        ).cast[ref_type]()

    _softmax_gpu[
        test_type,
        1,
        rank,
        input_fn_device,
        has_prologue_fusion=has_prologue_fusion,
    ](
        Coord(shape),
        TileTensor(out_device_test_ptr, row_major(Coord(shape))),
        rank - 1,
        ctx,
    )

    _softmax_cpu[ref_type, 1, rank, origin_of()._mlir_origin, input_fn_host](
        Coord(shape),
        TileTensor(out_ref_ptr, row_major(Coord(shape))),
        rank - 1,
    )

    ctx.synchronize()
    ctx.enqueue_copy(out_host_test_ptr, out_device_test_ptr)
    ctx.synchronize()

    # GPU output matches the fp32 CPU reference elementwise.
    for i in range(length):
        assert_almost_equal(
            out_host_test_ptr[i].cast[ref_type](),
            out_ref_ptr[i],
            atol=1e-2,
        )

    # Every softmax row sums to ~1.
    for r in range(num_rows):
        var s = Scalar[ref_type](0)
        for j in range(row_size):
            s += out_host_test_ptr[r * row_size + j].cast[ref_type]()
        assert_almost_equal(s, Scalar[ref_type](1), atol=1e-2)

    in_host_test_ptr.free()
    out_host_test_ptr.free()
    out_ref_ptr.free()
    _ = in_device_test_ptr
    _ = out_device_test_ptr


def test_gpu_softmax_verify_shapes[test_type: DType](ctx: DeviceContext) raises:
    """Ported from `verify_softmax.py`: drives `test_gpu_softmax_warp_short_axis`
    over shapes that cross the `WARP_SIZE` boundary -- the warp-per-row fast
    path (inner axis <= WARP_SIZE) and the block-per-row path (> WARP_SIZE) --
    including the customer inner dim 24 and the boundary at WARP_SIZE.
    """
    print("== test_gpu_softmax_verify_shapes", test_type)
    # Warp-per-row path (inner axis <= WARP_SIZE), exercising both the flat fast
    # path (has_prologue_fusion=False) and the true-coordinate fusion-safe path
    # (has_prologue_fusion=True). Both must match the reference. Includes the
    # customer inner dim 24 and the boundary at WARP_SIZE.
    test_gpu_softmax_warp_short_axis[test_type, IndexList[4](1, 1, 1, 1)](ctx)
    test_gpu_softmax_warp_short_axis[test_type, IndexList[4](2, 3, 5, 7)](ctx)
    test_gpu_softmax_warp_short_axis[test_type, IndexList[4](16, 16, 16, 16)](
        ctx
    )
    # Customer inner dim.
    test_gpu_softmax_warp_short_axis[test_type, IndexList[4](8, 16, 8, 24)](ctx)
    test_gpu_softmax_warp_short_axis[
        test_type, IndexList[4](8, 16, 8, 24), has_prologue_fusion=True
    ](ctx)
    # == WARP_SIZE boundary.
    test_gpu_softmax_warp_short_axis[test_type, IndexList[4](4, 8, 16, 32)](ctx)
    test_gpu_softmax_warp_short_axis[
        test_type, IndexList[4](4, 8, 16, 32), has_prologue_fusion=True
    ](ctx)
    # Block-per-row path (inner axis > WARP_SIZE; has_prologue_fusion unused there).
    test_gpu_softmax_warp_short_axis[test_type, IndexList[4](4, 8, 16, 33)](ctx)
    test_gpu_softmax_warp_short_axis[test_type, IndexList[4](4, 8, 16, 64)](ctx)
    test_gpu_softmax_warp_short_axis[test_type, IndexList[4](2, 4, 8, 128)](ctx)


def test_gpu_softmax_large_vocab[test_type: DType](ctx: DeviceContext) raises:
    """Exercises the vectorized path of the online op-level softmax, using a
    vocab-sized inner dimension (the case the Eagle/sampler softmax hits).
    """
    print("== test_gpu_softmax_large_vocab")
    comptime seed_val = 42
    seed(seed_val)

    comptime ref_type = DType.float32
    comptime rank = 2
    var shape = IndexList[rank](32, 128256)
    var length = shape.flattened_length()

    comptime layout_dyn = Layout.row_major[rank]()
    var runtime_layout = RuntimeLayout[layout_dyn].row_major(shape)

    var in_ref = ManagedLayoutTensor[ref_type, layout_dyn](runtime_layout, ctx)
    var in_test = ManagedLayoutTensor[test_type, layout_dyn](
        runtime_layout, ctx
    )
    var out_ref = ManagedLayoutTensor[ref_type, layout_dyn](runtime_layout, ctx)
    var out_test = ManagedLayoutTensor[test_type, layout_dyn](
        runtime_layout, ctx
    )

    var in_host_test = in_test.tensor[update=False]()
    var in_host_ref = in_ref.tensor[update=False]()
    for i in range(length):
        var v = random_float64(-3, 3).cast[.float32]().cast[test_type]()
        in_host_test.ptr[i] = v
        in_host_ref.ptr[i] = v.cast[ref_type]()

    var in_device_ref = in_ref.device_tensor()
    var in_device_test = in_test.device_tensor()

    @__parameter
    @__copy_capture(in_device_ref)
    def input_fn_ref[
        _simd_width: Int
    ](coords: Coord) -> SIMD[ref_type, _simd_width]:
        return in_device_ref.load[width=_simd_width](
            coord_to_index_list(coords)
        )

    @__parameter
    @__copy_capture(in_device_test)
    def input_fn_test[
        _simd_width: Int
    ](coords: Coord) -> SIMD[test_type, _simd_width]:
        return in_device_test.load[width=_simd_width](
            coord_to_index_list(coords)
        )

    _softmax_gpu[
        ref_type,
        simd_width_of[ref_type, target=get_gpu_target()](),
        rank,
        input_fn_ref,
    ](
        Coord(shape),
        TileTensor(out_ref.device_data.value(), row_major(Coord(shape))),
        rank - 1,
        ctx,
    )

    _softmax_gpu[
        test_type,
        simd_width_of[test_type, target=get_gpu_target()](),
        rank,
        input_fn_test,
    ](
        Coord(shape),
        TileTensor(out_test.device_data.value(), row_major(Coord(shape))),
        rank - 1,
        ctx,
    )

    var out_host_ref = out_ref.tensor()
    var out_host_test = out_test.tensor()
    for i in range(length):
        var ref_val = out_host_ref.ptr[i]
        var test_val = out_host_test.ptr[i].cast[ref_type]()
        assert_almost_equal(ref_val, test_val, atol=1e-2)


def test_gpu_softmax_masked_split[test_type: DType](ctx: DeviceContext) raises:
    """Drives a `-inf`-masked large-vocab input through the split-K path of
    `_softmax_gpu` and checks it against `_softmax_cpu`.

    Few rows over a long inner axis force `num_splits > 1`, so this covers the
    split-K combine's masked-chunk handling: a split whose whole chunk is
    `-inf` must contribute `exp_sum = 0` (not NaN), and a fully-masked row must
    reproduce the single-block kernel's NaN behavior. There is no other test
    that pushes `-inf` through this kernel.
    """
    print("== test_gpu_softmax_masked_split", test_type)
    seed(42)

    comptime ref_type = DType.float32
    comptime rank = 2
    # Few rows over a long axis so `num_splits > 1` (fp32 and bf16) and the
    # split path is exercised.
    var shape = IndexList[rank](4, 32768)
    comptime row_size = 32768
    var length = shape.flattened_length()
    var num_rows = length // row_size

    comptime layout_dyn = Layout.row_major[rank]()

    var neg_inf = min_or_neg_inf[test_type]()
    var in_host_ptr = alloc[Scalar[test_type]](length)
    var in_device_ptr = ctx.enqueue_create_buffer[test_type](length)
    var in_host = LayoutTensor[test_type, layout_dyn](
        in_host_ptr, RuntimeLayout[layout_dyn].row_major(shape)
    )
    var in_device = LayoutTensor[test_type, layout_dyn](
        in_device_ptr.unsafe_ptr(), RuntimeLayout[layout_dyn].row_major(shape)
    )

    for i in range(length):
        in_host_ptr[i] = (
            random_float64(-3, 3).cast[ref_type]().cast[test_type]()
        )

    # Row 0 fully masked (expects NaN); row 1's `-inf` prefix fully masks a
    # leading split chunk (the "contributes 0, not NaN" landmine); row 2 masked
    # tail; row 3 unmasked.
    for j in range(row_size):
        in_host_ptr[0 * row_size + j] = neg_inf
    for j in range(8192):
        in_host_ptr[1 * row_size + j] = neg_inf
    for j in range(24576, row_size):
        in_host_ptr[2 * row_size + j] = neg_inf

    ctx.enqueue_copy(in_device_ptr, in_host_ptr)

    var out_host_ptr = alloc[Scalar[test_type]](length)
    var out_device_ptr = ctx.enqueue_create_buffer[test_type](length)
    var out_ref_ptr = alloc[Scalar[ref_type]](length)

    @__parameter
    @__copy_capture(in_device)
    def input_fn_device[
        _simd_width: Int
    ](coords: Coord) -> SIMD[test_type, _simd_width]:
        return in_device.load[width=_simd_width](coord_to_index_list(coords))

    @__parameter
    @__copy_capture(in_host)
    def input_fn_host[
        _simd_width: Int
    ](coords: Coord) -> SIMD[ref_type, _simd_width]:
        return in_host.load[width=_simd_width](
            coord_to_index_list(coords)
        ).cast[ref_type]()

    _softmax_gpu[
        test_type,
        simd_width_of[test_type, target=get_gpu_target()](),
        rank,
        input_fn_device,
    ](
        Coord(shape),
        TileTensor(out_device_ptr, row_major(Coord(shape))),
        rank - 1,
        ctx,
    )

    _softmax_cpu[ref_type, 1, rank, origin_of()._mlir_origin, input_fn_host](
        Coord(shape),
        TileTensor(out_ref_ptr, row_major(Coord(shape))),
        rank - 1,
    )

    ctx.synchronize()
    ctx.enqueue_copy(out_host_ptr, out_device_ptr)
    ctx.synchronize()

    # Match the reference where finite; NaN where the reference is NaN.
    for i in range(length):
        var ref_val = out_ref_ptr[i]
        var got = out_host_ptr[i].cast[ref_type]()
        if isnan(ref_val):
            assert_true(isnan(got))
        else:
            assert_almost_equal(got, ref_val, atol=1e-4, rtol=1e-5)

    # Finite (partially/un-masked) rows still sum to ~1.
    for r in range(num_rows):
        if isnan(out_ref_ptr[r * row_size]):
            continue
        var s = Scalar[ref_type](0)
        for j in range(row_size):
            s += out_host_ptr[r * row_size + j].cast[ref_type]()
        assert_almost_equal(s, Scalar[ref_type](1), atol=1e-2)

    in_host_ptr.free()
    out_host_ptr.free()
    out_ref_ptr.free()
    _ = in_device_ptr
    _ = out_device_ptr
    _ = in_device


def test_gpu_online_softmax[
    WM: Int, WN: Int, transpose_fragments: Bool
](ctx: DeviceContext) raises:
    print("== test_online_softmax")

    comptime type = DType.float32
    comptime rank = 3
    comptime seqlen = 256

    # For testing purpose, call online softmax twice and each time updates half
    # seq_len. Limit to WM rows and arrange warps in N dim.
    comptime shape = IndexList[rank](1, WM, seqlen)
    comptime num_warps = seqlen // (2 * WN)
    comptime num_threads = num_warps * WARP_SIZE

    var in_host_ptr = ctx.enqueue_create_host_buffer[type](
        shape.flattened_length()
    )
    var out_host_ptr = ctx.enqueue_create_host_buffer[type](
        shape.flattened_length()
    )
    var out_ref_ptr = ctx.enqueue_create_host_buffer[type](
        shape.flattened_length()
    )

    comptime layout_dyn = Layout.row_major[rank]()
    var in_host = LayoutTensor[type, layout_dyn](
        in_host_ptr, RuntimeLayout[layout_dyn].row_major(shape)
    )
    var out_ref = LayoutTensor[type, layout_dyn](
        out_ref_ptr, RuntimeLayout[layout_dyn].row_major(shape)
    )

    var in_device_ptr = ctx.enqueue_create_buffer[type](
        shape.flattened_length()
    )
    var out_device_ptr = ctx.enqueue_create_buffer[type](
        shape.flattened_length()
    )

    var in_device = LayoutTensor[type, Layout.row_major(shape[1], shape[2])](
        in_device_ptr
    )
    var out_device = LayoutTensor[type, Layout.row_major(shape[1], shape[2])](
        out_device_ptr
    )

    rand[type](in_host_ptr.as_span())

    ctx.enqueue_copy(in_device_ptr, in_host_ptr)
    comptime kernel = _online_softmax_kernel[
        WM,
        WN,
        DType.float32,
        Layout.row_major(shape[1], shape[2]),
        transpose_fragments,
    ]

    ctx.enqueue_function[kernel](
        in_device,
        out_device,
        grid_dim=1,
        block_dim=num_threads,
    )

    @__parameter
    @__copy_capture(in_host)
    def input_fn_host[
        _simd_width: Int
    ](coords: Coord) -> SIMD[type, _simd_width]:
        return in_host.load[width=_simd_width](coord_to_index_list(coords))

    _softmax_cpu[type, 1, rank, origin_of()._mlir_origin, input_fn_host](
        Coord(shape),
        TileTensor(out_ref.ptr, row_major(Coord(shape))),
        rank - 1,
    )

    ctx.synchronize()
    ctx.enqueue_copy(out_host_ptr, out_device_ptr)
    ctx.synchronize()

    for i in range(shape.flattened_length()):
        assert_almost_equal(
            out_host_ptr[i], out_ref_ptr[i], atol=1e-4, rtol=1e-5
        )

    _ = in_device_ptr
    _ = out_device_ptr


def test_gpu_logsoftmax(ctx: DeviceContext) raises:
    print("== test_gpu_logsoftmax")

    comptime type = DType.float32
    comptime rank = 3

    @__parameter
    def _test_shape(shape: IndexList[rank]) raises:
        var in_host_ptr = ctx.enqueue_create_host_buffer[type](
            shape.flattened_length()
        )
        var in_device_ptr = ctx.enqueue_create_buffer[type](
            shape.flattened_length()
        )
        comptime layout_dyn = Layout.row_major[rank]()
        var in_host = LayoutTensor[type, layout_dyn](
            in_host_ptr, RuntimeLayout[layout_dyn].row_major(shape)
        )
        var in_device = LayoutTensor[type, layout_dyn](
            in_device_ptr.unsafe_ptr(),
            RuntimeLayout[layout_dyn].row_major(shape),
        )
        var out_host_ptr = ctx.enqueue_create_host_buffer[type](
            shape.flattened_length()
        )
        var out_ref_ptr = ctx.enqueue_create_host_buffer[type](
            shape.flattened_length()
        )
        var out_device_ptr = ctx.enqueue_create_buffer[type](
            shape.flattened_length()
        )
        var out_ref = LayoutTensor[type, layout_dyn](
            out_ref_ptr, RuntimeLayout[layout_dyn].row_major(shape)
        )
        rand[type](in_host_ptr.as_span())
        ctx.enqueue_copy(in_device_ptr, in_host_ptr)

        @__parameter
        @__copy_capture(in_device)
        def input_fn_device[
            _simd_width: Int
        ](coords: Coord) -> SIMD[type, _simd_width]:
            return in_device.load[width=_simd_width](
                coord_to_index_list(coords)
            )

        @__parameter
        @__copy_capture(in_host)
        def input_fn_host[
            _simd_width: Int
        ](coords: Coord) -> SIMD[type, _simd_width]:
            return in_host.load[width=_simd_width](coord_to_index_list(coords))

        _softmax_gpu[type, 1, rank, input_fn_device, logsoftmax=True](
            Coord(shape),
            TileTensor(out_device_ptr, row_major(Coord(shape))),
            rank - 1,
            ctx,
        )

        _softmax_cpu[
            type,
            1,
            rank,
            origin_of()._mlir_origin,
            input_fn_host,
            logsoftmax=True,
        ](
            Coord(shape),
            TileTensor(out_ref.ptr, row_major(Coord(shape))),
            rank - 1,
        )

        ctx.synchronize()
        ctx.enqueue_copy(out_host_ptr, out_device_ptr)
        ctx.synchronize()

        for i in range(shape.flattened_length()):
            var expected = out_ref_ptr[i]
            var got = out_host_ptr[i]
            if not isclose(expected, got, atol=1e-4, rtol=1e-5):
                print(
                    "ERROR. Mismatch at flattened idx:",
                    i,
                    "expected:",
                    expected,
                    "got:",
                    got,
                )
                assert_true(False)

        _ = in_device
        _ = in_host
        _ = in_device_ptr
        _ = out_device_ptr

    # Test multi-thread row processing (row_size=515 > BLOCK_SIZE=128)
    _test_shape(IndexList[rank](3, 5, 515))
    # Test single-thread row processing (row_size=4 < BLOCK_SIZE=128)
    _test_shape(IndexList[rank](1, 1, 4))


def test_gpu_softmax_temperature[per_row: Bool](ctx: DeviceContext) raises:
    """Test GPU softmax_with_temperature against CPU reference.

    Parameters:
        per_row: If True, use per-row temperature array; otherwise scalar.
    """

    comptime if per_row:
        print("== test_gpu_softmax_temperature (per_row)")
    else:
        print("== test_gpu_softmax_temperature (scalar)")

    comptime type = DType.float32
    comptime rank = 2
    comptime batch_size = 4
    comptime vocab_size = 512
    var shape = IndexList[rank](batch_size, vocab_size)
    var length = shape.flattened_length()

    # Input logits.
    var in_host_ptr = ctx.enqueue_create_host_buffer[type](length)
    var in_device = ctx.enqueue_create_buffer[type](length)
    rand[type](in_host_ptr.as_span())
    for i in range(length):
        in_host_ptr[i] *= 10.0
    ctx.enqueue_copy(in_device, in_host_ptr)

    # GPU output.
    var out_device = ctx.enqueue_create_buffer[type](length)

    var rt_layout = row_major(Coord(batch_size, vocab_size))
    var in_tt = TileTensor(in_device, rt_layout)
    var out_tt = TileTensor(out_device, rt_layout)

    # Temperature: scalar or per-row array.
    var temp_host_ptr = ctx.enqueue_create_host_buffer[type](batch_size)
    var temp_device = ctx.enqueue_create_buffer[type](batch_size)

    comptime if per_row:
        rand[type](temp_host_ptr.as_span())
        for i in range(batch_size):
            temp_host_ptr[i] = temp_host_ptr[i] * 1.5 + 0.5
        ctx.enqueue_copy(temp_device, temp_host_ptr)
        var temp_tt = TileTensor(temp_device, row_major(batch_size))
        softmax_with_temperature(
            ctx,
            in_tt,
            out_tt,
            temperature_arr=temp_tt.as_unsafe_any_origin().as_immut(),
        )
    else:
        var temperature = Scalar[type](0.7)
        # Fill uniform so CPU reference loop works the same way.
        for i in range(batch_size):
            temp_host_ptr[i] = temperature
        softmax_with_temperature(ctx, in_tt, out_tt, temperature=temperature)

    # CPU reference: standard softmax on logits / T per row.
    comptime layout_dyn = Layout.row_major[rank]()
    var scaled_host_ptr = ctx.enqueue_create_host_buffer[type](length)
    var scaled_host = LayoutTensor[type, layout_dyn](
        scaled_host_ptr, RuntimeLayout[layout_dyn].row_major(shape)
    )
    var in_host = LayoutTensor[type, layout_dyn](
        in_host_ptr, RuntimeLayout[layout_dyn].row_major(shape)
    )
    for row in range(batch_size):
        for col in range(vocab_size):
            scaled_host[row, col] = in_host[row, col] / temp_host_ptr[row]
    var ref_host_ptr = ctx.enqueue_create_host_buffer[type](length)
    var out_ref = LayoutTensor[type, layout_dyn](
        ref_host_ptr, RuntimeLayout[layout_dyn].row_major(shape)
    )

    @__parameter
    @__copy_capture(scaled_host)
    def input_fn_cpu[
        _simd_width: Int
    ](coords: Coord) -> SIMD[type, _simd_width]:
        return scaled_host.load[width=_simd_width](coord_to_index_list(coords))

    _softmax_cpu[type, 1, rank, origin_of()._mlir_origin, input_fn_cpu](
        Coord(shape),
        TileTensor(out_ref.ptr, row_major(Coord(shape))),
        rank - 1,
    )

    ctx.synchronize()
    var out_host_ptr = ctx.enqueue_create_host_buffer[type](length)
    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.synchronize()

    for i in range(length):
        if not isclose(out_host_ptr[i], ref_host_ptr[i], atol=1e-4, rtol=1e-5):
            print(
                "ERROR. Mismatch at idx:",
                i,
                "expected:",
                ref_host_ptr[i],
                "got:",
                out_host_ptr[i],
            )
            assert_true(False)

    _ = in_device
    _ = out_device
    _ = temp_device
    _ = scaled_host


def main() raises:
    with DeviceContext() as ctx:
        test_gpu_softmax(ctx)
        test_gpu_softmax_half[.bfloat16](ctx)
        test_gpu_softmax_half[.float16](ctx)
        test_gpu_softmax_warp_short_axis[.bfloat16, IndexList[2](12, 5)](ctx)
        test_gpu_softmax_warp_short_axis[.float16, IndexList[2](12, 5)](ctx)
        test_gpu_softmax_verify_shapes[.bfloat16](ctx)
        test_gpu_softmax_verify_shapes[.float32](ctx)
        test_gpu_softmax_large_vocab[.bfloat16](ctx)
        test_gpu_softmax_large_vocab[.float32](ctx)
        test_gpu_softmax_masked_split[.float32](ctx)
        test_gpu_softmax_masked_split[.bfloat16](ctx)
        test_gpu_logsoftmax(ctx)
        test_gpu_softmax_temperature[per_row=False](ctx)
        test_gpu_softmax_temperature[per_row=True](ctx)
        # Test general online-softmax, communicating data via shared memory.

        test_gpu_online_softmax[32, 32, False](ctx)
        # Test covering entire row within one warp
        test_gpu_online_softmax[16, 128, False](ctx)

        comptime if has_amd_gpu_accelerator():
            test_gpu_online_softmax[32, 32, True](ctx)
            # Test covering entire row within one warp
            test_gpu_online_softmax[16, 128, True](ctx)
