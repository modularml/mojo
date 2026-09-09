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
"""Tests for dual-concat kernels (two independent concats in one launch)."""

from layout import Coord, Idx, TensorLayout, TileTensor, row_major
from nn.concat import (
    _fused_dual_concat_gpu,
    elementwise_epilogue_type,
)
from std.sys.info import is_gpu

from max.gpu.host import DeviceContext
from std.testing import assert_equal
from std.utils import IndexList, StaticTuple


# ===-----------------------------------------------------------------------===#
# Inner-most single-dim fast path (axis == rank-1, each input has dim 1)
# ===-----------------------------------------------------------------------===#


def test_dual_concat_inner_most_single_dim(ctx: DeviceContext) raises:
    """Two independent concats along the last axis, each input has size 1 in
    that dim. This exercises _fused_dual_concat_inner_most_single_dim."""
    print("== test_dual_concat_inner_most_single_dim")

    comptime dtype = DType.float32
    comptime rank = 3

    # --- Concat 0: 3 inputs of shape [2, 4, 1] → output [2, 4, 3] ---
    comptime in_layout_0 = row_major[2, 4, 1]()
    comptime out_layout_0 = row_major[2, 4, 3]()

    var a0_buf = ctx.enqueue_create_buffer[dtype](in_layout_0.product())
    var a1_buf = ctx.enqueue_create_buffer[dtype](in_layout_0.product())
    var a2_buf = ctx.enqueue_create_buffer[dtype](in_layout_0.product())
    var out0_buf = ctx.enqueue_create_buffer[dtype](out_layout_0.product())

    var a0_host = ctx.enqueue_create_host_buffer[dtype](in_layout_0.product())
    var a1_host = ctx.enqueue_create_host_buffer[dtype](in_layout_0.product())
    var a2_host = ctx.enqueue_create_host_buffer[dtype](in_layout_0.product())
    ctx.synchronize()

    for i in range(in_layout_0.product()):
        a0_host[i] = Float32(10 + i)
        a1_host[i] = Float32(20 + i)
        a2_host[i] = Float32(30 + i)

    ctx.enqueue_copy(a0_buf, a0_host)
    ctx.enqueue_copy(a1_buf, a1_host)
    ctx.enqueue_copy(a2_buf, a2_host)

    # --- Concat 1: 2 inputs of shape [2, 4, 1] → output [2, 4, 2] ---
    comptime out_layout_1 = row_major[2, 4, 2]()

    var b0_buf = ctx.enqueue_create_buffer[dtype](in_layout_0.product())
    var b1_buf = ctx.enqueue_create_buffer[dtype](in_layout_0.product())
    var out1_buf = ctx.enqueue_create_buffer[dtype](out_layout_1.product())

    var b0_host = ctx.enqueue_create_host_buffer[dtype](in_layout_0.product())
    var b1_host = ctx.enqueue_create_host_buffer[dtype](in_layout_0.product())
    ctx.synchronize()

    for i in range(in_layout_0.product()):
        b0_host[i] = Float32(100 + i)
        b1_host[i] = Float32(200 + i)

    ctx.enqueue_copy(b0_buf, b0_host)
    ctx.enqueue_copy(b1_buf, b1_host)
    ctx.synchronize()

    var in_shape_0 = IndexList[rank](2, 4, 1)
    var a0 = TileTensor(a0_buf, row_major(Coord(in_shape_0)))
    var a1 = TileTensor(a1_buf, row_major(Coord(in_shape_0)))
    var a2 = TileTensor(a2_buf, row_major(Coord(in_shape_0)))
    var out0 = TileTensor(out0_buf, row_major(Coord(IndexList[rank](2, 4, 3))))

    var b0 = TileTensor(b0_buf, row_major(Coord(in_shape_0)))
    var b1 = TileTensor(b1_buf, row_major(Coord(in_shape_0)))
    var out1 = TileTensor(out1_buf, row_major(Coord(IndexList[rank](2, 4, 2))))

    var input_shapes_0 = StaticTuple[IndexList[rank], 3](
        in_shape_0, in_shape_0, in_shape_0
    )
    var input_shapes_1 = StaticTuple[IndexList[rank], 2](in_shape_0, in_shape_0)

    @__parameter
    @always_inline
    @__copy_capture(a0, a1, a2)
    def input_fn_0[
        input_index: Int, width: Int, _rank: Int, alignment: Int = 1
    ](indices: IndexList[_rank]) -> SIMD[dtype, width]:
        var coord = Coord(indices)
        comptime if input_index == 0:
            return rebind[SIMD[dtype, width]](a0.load[width=width](coord))
        elif input_index == 1:
            return rebind[SIMD[dtype, width]](a1.load[width=width](coord))
        else:
            return rebind[SIMD[dtype, width]](a2.load[width=width](coord))

    @__parameter
    @always_inline
    @__copy_capture(out0)
    def output_0_fn[
        c_type: DType, _rank: Int, width: SIMDLength = 1, *, alignment: Int = 1
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        out0.store[width=width](Coord(indices), rebind[SIMD[dtype, width]](val))

    @__parameter
    @always_inline
    @__copy_capture(b0, b1)
    def input_fn_1[
        input_index: Int, width: Int, _rank: Int, alignment: Int = 1
    ](indices: IndexList[_rank]) -> SIMD[dtype, width]:
        var coord = Coord(indices)
        comptime if input_index == 0:
            return rebind[SIMD[dtype, width]](b0.load[width=width](coord))
        else:
            return rebind[SIMD[dtype, width]](b1.load[width=width](coord))

    @__parameter
    @always_inline
    @__copy_capture(out1)
    def output_1_fn[
        c_type: DType, _rank: Int, width: SIMDLength = 1, *, alignment: Int = 1
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        out1.store[width=width](Coord(indices), rebind[SIMD[dtype, width]](val))

    _fused_dual_concat_gpu[
        rank,
        dtype,
        input_fn_0,
        output_0_fn,
        3,
        input_fn_1,
        output_1_fn,
        2,
        out0.LayoutType,
        out1.LayoutType,
    ](
        rank - 1,
        input_shapes_0,
        out0.as_unsafe_any_origin(),
        input_shapes_1,
        out1.as_unsafe_any_origin(),
        ctx,
    )

    # Read back and verify
    var out0_host = ctx.enqueue_create_host_buffer[dtype](
        out_layout_0.product()
    )
    var out1_host = ctx.enqueue_create_host_buffer[dtype](
        out_layout_1.product()
    )
    ctx.enqueue_copy(out0_host, out0_buf)
    ctx.enqueue_copy(out1_host, out1_buf)
    ctx.synchronize()

    var out0_tile = TileTensor(out0_host, out_layout_0)
    var out1_tile = TileTensor(out1_host, out_layout_1)

    # Verify concat 0: output[i,j,k] == input_k[i,j,0]
    for i in range(2):
        for j in range(4):
            var flat = i * 4 + j
            assert_equal(out0_tile[i, j, 0], Float32(10 + flat))
            assert_equal(out0_tile[i, j, 1], Float32(20 + flat))
            assert_equal(out0_tile[i, j, 2], Float32(30 + flat))

    # Verify concat 1: output[i,j,k] == input_k[i,j,0]
    for i in range(2):
        for j in range(4):
            var flat = i * 4 + j
            assert_equal(out1_tile[i, j, 0], Float32(100 + flat))
            assert_equal(out1_tile[i, j, 1], Float32(200 + flat))

    _ = a0_buf
    _ = a1_buf
    _ = a2_buf
    _ = b0_buf
    _ = b1_buf
    _ = out0_buf
    _ = out1_buf


# ===-----------------------------------------------------------------------===#
# General elementwise path (axis != rank-1)
# ===-----------------------------------------------------------------------===#


def test_dual_concat_general_axis(ctx: DeviceContext) raises:
    """Two independent concats along axis=1 (not the last axis).
    This exercises _fused_dual_concat_gpu_elementwise → dual_elementwise."""
    print("== test_dual_concat_general_axis")

    comptime dtype = DType.float32
    comptime rank = 3

    # --- Concat 0: 2 inputs [2, 3, 4] and [2, 5, 4] → output [2, 8, 4] ---
    comptime in0_l0 = row_major[2, 3, 4]()
    comptime in0_l1 = row_major[2, 5, 4]()
    comptime out0_l = row_major[2, 8, 4]()

    var c0_buf = ctx.enqueue_create_buffer[dtype](in0_l0.product())
    var c1_buf = ctx.enqueue_create_buffer[dtype](in0_l1.product())
    var out0_buf = ctx.enqueue_create_buffer[dtype](out0_l.product())

    var c0_host = ctx.enqueue_create_host_buffer[dtype](in0_l0.product())
    var c1_host = ctx.enqueue_create_host_buffer[dtype](in0_l1.product())
    ctx.synchronize()

    for i in range(in0_l0.product()):
        c0_host[i] = Float32(i)
    for i in range(in0_l1.product()):
        c1_host[i] = Float32(100 + i)

    ctx.enqueue_copy(c0_buf, c0_host)
    ctx.enqueue_copy(c1_buf, c1_host)

    # --- Concat 1: 2 inputs [2, 2, 4] and [2, 6, 4] → output [2, 8, 4] ---
    comptime in1_l0 = row_major[2, 2, 4]()
    comptime in1_l1 = row_major[2, 6, 4]()
    comptime out1_l = row_major[2, 8, 4]()

    var d0_buf = ctx.enqueue_create_buffer[dtype](in1_l0.product())
    var d1_buf = ctx.enqueue_create_buffer[dtype](in1_l1.product())
    var out1_buf = ctx.enqueue_create_buffer[dtype](out1_l.product())

    var d0_host = ctx.enqueue_create_host_buffer[dtype](in1_l0.product())
    var d1_host = ctx.enqueue_create_host_buffer[dtype](in1_l1.product())
    ctx.synchronize()

    for i in range(in1_l0.product()):
        d0_host[i] = Float32(1000 + i)
    for i in range(in1_l1.product()):
        d1_host[i] = Float32(2000 + i)

    ctx.enqueue_copy(d0_buf, d0_host)
    ctx.enqueue_copy(d1_buf, d1_host)
    ctx.synchronize()

    var c0 = TileTensor(c0_buf, row_major(Coord(IndexList[rank](2, 3, 4))))
    var c1 = TileTensor(c1_buf, row_major(Coord(IndexList[rank](2, 5, 4))))
    var out0 = TileTensor(out0_buf, row_major(Coord(IndexList[rank](2, 8, 4))))

    var d0 = TileTensor(d0_buf, row_major(Coord(IndexList[rank](2, 2, 4))))
    var d1 = TileTensor(d1_buf, row_major(Coord(IndexList[rank](2, 6, 4))))
    var out1 = TileTensor(out1_buf, row_major(Coord(IndexList[rank](2, 8, 4))))

    var input_shapes_0 = StaticTuple[IndexList[rank], 2](
        IndexList[rank](2, 3, 4),
        IndexList[rank](2, 5, 4),
    )
    var input_shapes_1 = StaticTuple[IndexList[rank], 2](
        IndexList[rank](2, 2, 4),
        IndexList[rank](2, 6, 4),
    )

    @__parameter
    @always_inline
    @__copy_capture(c0, c1)
    def input_fn_0[
        input_index: Int, width: Int, _rank: Int, alignment: Int = 1
    ](indices: IndexList[_rank]) -> SIMD[dtype, width]:
        var coord = Coord(indices)
        comptime if input_index == 0:
            return rebind[SIMD[dtype, width]](c0.load[width=width](coord))
        else:
            return rebind[SIMD[dtype, width]](c1.load[width=width](coord))

    @__parameter
    @always_inline
    @__copy_capture(out0)
    def output_0_fn[
        c_type: DType, _rank: Int, width: SIMDLength = 1, *, alignment: Int = 1
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        out0.store[width=width](Coord(indices), rebind[SIMD[dtype, width]](val))

    @__parameter
    @always_inline
    @__copy_capture(d0, d1)
    def input_fn_1[
        input_index: Int, width: Int, _rank: Int, alignment: Int = 1
    ](indices: IndexList[_rank]) -> SIMD[dtype, width]:
        var coord = Coord(indices)
        comptime if input_index == 0:
            return rebind[SIMD[dtype, width]](d0.load[width=width](coord))
        else:
            return rebind[SIMD[dtype, width]](d1.load[width=width](coord))

    @__parameter
    @always_inline
    @__copy_capture(out1)
    def output_1_fn[
        c_type: DType, _rank: Int, width: SIMDLength = 1, *, alignment: Int = 1
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        out1.store[width=width](Coord(indices), rebind[SIMD[dtype, width]](val))

    _fused_dual_concat_gpu[
        rank,
        dtype,
        input_fn_0,
        output_0_fn,
        2,
        input_fn_1,
        output_1_fn,
        2,
        out0.LayoutType,
        out1.LayoutType,
    ](
        1,
        input_shapes_0,
        out0.as_unsafe_any_origin(),
        input_shapes_1,
        out1.as_unsafe_any_origin(),
        ctx,
    )

    # Read back and verify
    var out0_host = ctx.enqueue_create_host_buffer[dtype](out0_l.product())
    var out1_host = ctx.enqueue_create_host_buffer[dtype](out1_l.product())
    ctx.enqueue_copy(out0_host, out0_buf)
    ctx.enqueue_copy(out1_host, out1_buf)
    ctx.synchronize()

    var out0_tile = TileTensor(out0_host, out0_l)
    var out1_tile = TileTensor(out1_host, out1_l)

    # Verify concat 0: first 3 rows from c0 (arange), next 5 rows from c1
    var c0_tile = TileTensor(c0_host, in0_l0)
    var c1_tile = TileTensor(c1_host, in0_l1)
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert_equal(
                    out0_tile[i, j, k],
                    c0_tile[i, j, k],
                    msg="concat0 c0 region",
                )
        for j in range(5):
            for k in range(4):
                assert_equal(
                    out0_tile[i, 3 + j, k],
                    c1_tile[i, j, k],
                    msg="concat0 c1 region",
                )

    # Verify concat 1: first 2 rows from d0, next 6 rows from d1
    var d0_tile = TileTensor(d0_host, in1_l0)
    var d1_tile = TileTensor(d1_host, in1_l1)
    for i in range(2):
        for j in range(2):
            for k in range(4):
                assert_equal(
                    out1_tile[i, j, k],
                    d0_tile[i, j, k],
                    msg="concat1 d0 region",
                )
        for j in range(6):
            for k in range(4):
                assert_equal(
                    out1_tile[i, 2 + j, k],
                    d1_tile[i, j, k],
                    msg="concat1 d1 region",
                )

    _ = c0_buf
    _ = c1_buf
    _ = d0_buf
    _ = d1_buf
    _ = out0_buf
    _ = out1_buf


def test_dual_concat_inner_most_static_vs_dynamic(ctx: DeviceContext) raises:
    """Static-divisor fold (KERN-2872) equivalence check for the dual kernel.

    `_fused_dual_concat_inner_most_single_dim` decomposes the flattened row
    index over BOTH `output_0` and `output_1` independently. This runs the dual
    launch twice on identical inputs -- once with fully static output layouts
    (`row_major[...]()`, so each decomposition folds to magic-multiply + shift)
    and once with dynamic `Coord(IndexList)` output layouts (runtime `IDIV`) --
    and asserts both outputs are bit-identical across the two paths. rank-4 with
    non-trivial outer dims so dims 1..rank-2 actually divide.
    """
    comptime dtype = DType.float32
    comptime rank = 4
    comptime d0 = 3
    comptime d1 = 5
    comptime d2 = 7
    comptime size_0 = 3
    comptime size_1 = 2

    comptime in_layout = row_major[d0, d1, d2, 1]()
    comptime static_out_layout_0 = row_major[d0, d1, d2, size_0]()
    comptime static_out_layout_1 = row_major[d0, d1, d2, size_1]()
    comptime n_rows = d0 * d1 * d2

    var in_shape = IndexList[rank](d0, d1, d2, 1)
    var input_shapes_0 = StaticTuple[IndexList[rank], size_0](
        in_shape, in_shape, in_shape
    )
    var input_shapes_1 = StaticTuple[IndexList[rank], size_1](
        in_shape, in_shape
    )

    # Shared inputs: each gets a disjoint value range so the concat is verifiable
    # and the two output-layout variants compare meaningfully.
    var a0_buf = ctx.enqueue_create_buffer[dtype](n_rows)
    var a1_buf = ctx.enqueue_create_buffer[dtype](n_rows)
    var a2_buf = ctx.enqueue_create_buffer[dtype](n_rows)
    var b0_buf = ctx.enqueue_create_buffer[dtype](n_rows)
    var b1_buf = ctx.enqueue_create_buffer[dtype](n_rows)
    var stage = ctx.enqueue_create_host_buffer[dtype](n_rows)
    ctx.synchronize()

    for buf_idx in range(5):
        for i in range(n_rows):
            stage[i] = Float32(buf_idx * n_rows + i)
        if buf_idx == 0:
            ctx.enqueue_copy(a0_buf, stage)
        elif buf_idx == 1:
            ctx.enqueue_copy(a1_buf, stage)
        elif buf_idx == 2:
            ctx.enqueue_copy(a2_buf, stage)
        elif buf_idx == 3:
            ctx.enqueue_copy(b0_buf, stage)
        else:
            ctx.enqueue_copy(b1_buf, stage)
    ctx.synchronize()

    # Four output buffers: static/dynamic for each of the two concat groups.
    var out0_static_buf = ctx.enqueue_create_buffer[dtype](n_rows * size_0)
    var out1_static_buf = ctx.enqueue_create_buffer[dtype](n_rows * size_1)
    var out0_dyn_buf = ctx.enqueue_create_buffer[dtype](n_rows * size_0)
    var out1_dyn_buf = ctx.enqueue_create_buffer[dtype](n_rows * size_1)

    # Inputs use a static layout in both runs (the fold keys off the OUTPUT
    # layout); only the output layout type differs between the two launches.
    var a0 = TileTensor(a0_buf, in_layout)
    var a1 = TileTensor(a1_buf, in_layout)
    var a2 = TileTensor(a2_buf, in_layout)
    var b0 = TileTensor(b0_buf, in_layout)
    var b1 = TileTensor(b1_buf, in_layout)

    @__parameter
    @always_inline
    @__copy_capture(a0, a1, a2)
    def input_fn_0[
        input_index: Int, width: Int, _rank: Int, alignment: Int = 1
    ](indices: IndexList[_rank]) -> SIMD[dtype, width]:
        var coord = Coord(indices)
        comptime if input_index == 0:
            return rebind[SIMD[dtype, width]](a0.load[width=width](coord))
        elif input_index == 1:
            return rebind[SIMD[dtype, width]](a1.load[width=width](coord))
        else:
            return rebind[SIMD[dtype, width]](a2.load[width=width](coord))

    @__parameter
    @always_inline
    @__copy_capture(b0, b1)
    def input_fn_1[
        input_index: Int, width: Int, _rank: Int, alignment: Int = 1
    ](indices: IndexList[_rank]) -> SIMD[dtype, width]:
        var coord = Coord(indices)
        comptime if input_index == 0:
            return rebind[SIMD[dtype, width]](b0.load[width=width](coord))
        else:
            return rebind[SIMD[dtype, width]](b1.load[width=width](coord))

    # --- Static-output run (fold fires on both decompositions). ---
    var out0_static = TileTensor(out0_static_buf, static_out_layout_0)
    var out1_static = TileTensor(out1_static_buf, static_out_layout_1)

    @__parameter
    @always_inline
    @__copy_capture(out0_static)
    def output_0_static_fn[
        c_type: DType, _rank: Int, width: SIMDLength = 1, *, alignment: Int = 1
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        out0_static.store[width=width](
            Coord(indices), rebind[SIMD[dtype, width]](val)
        )

    @__parameter
    @always_inline
    @__copy_capture(out1_static)
    def output_1_static_fn[
        c_type: DType, _rank: Int, width: SIMDLength = 1, *, alignment: Int = 1
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        out1_static.store[width=width](
            Coord(indices), rebind[SIMD[dtype, width]](val)
        )

    _fused_dual_concat_gpu[
        rank,
        dtype,
        input_fn_0,
        output_0_static_fn,
        size_0,
        input_fn_1,
        output_1_static_fn,
        size_1,
        out0_static.LayoutType,
        out1_static.LayoutType,
    ](
        input_shapes_0,
        out0_static.as_unsafe_any_origin(),
        input_shapes_1,
        out1_static.as_unsafe_any_origin(),
        ctx,
    )

    # --- Dynamic-output run (runtime divide on both decompositions). ---
    var out0_dyn = TileTensor(
        out0_dyn_buf, row_major(Coord(IndexList[rank](d0, d1, d2, size_0)))
    )
    var out1_dyn = TileTensor(
        out1_dyn_buf, row_major(Coord(IndexList[rank](d0, d1, d2, size_1)))
    )

    @__parameter
    @always_inline
    @__copy_capture(out0_dyn)
    def output_0_dyn_fn[
        c_type: DType, _rank: Int, width: SIMDLength = 1, *, alignment: Int = 1
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        out0_dyn.store[width=width](
            Coord(indices), rebind[SIMD[dtype, width]](val)
        )

    @__parameter
    @always_inline
    @__copy_capture(out1_dyn)
    def output_1_dyn_fn[
        c_type: DType, _rank: Int, width: SIMDLength = 1, *, alignment: Int = 1
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        out1_dyn.store[width=width](
            Coord(indices), rebind[SIMD[dtype, width]](val)
        )

    _fused_dual_concat_gpu[
        rank,
        dtype,
        input_fn_0,
        output_0_dyn_fn,
        size_0,
        input_fn_1,
        output_1_dyn_fn,
        size_1,
        out0_dyn.LayoutType,
        out1_dyn.LayoutType,
    ](
        input_shapes_0,
        out0_dyn.as_unsafe_any_origin(),
        input_shapes_1,
        out1_dyn.as_unsafe_any_origin(),
        ctx,
    )

    # Both output groups must be bit-identical across static-fold and dynamic.
    var out0_static_host = ctx.enqueue_create_host_buffer[dtype](
        n_rows * size_0
    )
    var out1_static_host = ctx.enqueue_create_host_buffer[dtype](
        n_rows * size_1
    )
    var out0_dyn_host = ctx.enqueue_create_host_buffer[dtype](n_rows * size_0)
    var out1_dyn_host = ctx.enqueue_create_host_buffer[dtype](n_rows * size_1)
    ctx.enqueue_copy(out0_static_host, out0_static_buf)
    ctx.enqueue_copy(out1_static_host, out1_static_buf)
    ctx.enqueue_copy(out0_dyn_host, out0_dyn_buf)
    ctx.enqueue_copy(out1_dyn_host, out1_dyn_buf)
    ctx.synchronize()

    for i in range(n_rows * size_0):
        assert_equal(
            out0_static_host[i],
            out0_dyn_host[i],
            msg="output_0 static-fold diverged from dynamic-divide",
        )
    for i in range(n_rows * size_1):
        assert_equal(
            out1_static_host[i],
            out1_dyn_host[i],
            msg="output_1 static-fold diverged from dynamic-divide",
        )

    _ = a0_buf
    _ = a1_buf
    _ = a2_buf
    _ = b0_buf
    _ = b1_buf
    _ = out0_static_buf
    _ = out1_static_buf
    _ = out0_dyn_buf
    _ = out1_dyn_buf


def main() raises:
    comptime if is_gpu():
        with DeviceContext() as ctx:
            test_dual_concat_inner_most_single_dim(ctx)
            test_dual_concat_general_axis(ctx)
            test_dual_concat_inner_most_static_vs_dynamic(ctx)
