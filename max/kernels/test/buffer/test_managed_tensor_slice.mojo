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
"""Tests for ManagedTensorSlice - a tensor view type for custom graph operations.
"""

from extensibility import get_row_major_tensor_spec_static
from extensibility import ManagedTensorSlice, IOSpec
from extensibility.managed_tensor_slice import (
    OutputFusion,
    StaticTensorSpec,
    _IndexListToTileLayout,
)
from layout import coord_to_index_list
from std.memory import AddressSpace
from std.sys import align_of
from max.gpu.host import DeviceContext
from extensibility import foreach
from layout import Coord
from std.testing import assert_equal, TestSuite

from std.utils import IndexList


def test_basic_construction() raises:
    """Test basic ManagedTensorSlice construction from pointer and shape."""
    var storage = Array[Float32, 3 * 4](fill={})
    # Shape-only constructor computes row-major strides automatically
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 3, 4]()
    var tensor = ManagedTensorSlice[io_spec=IOSpec.Unknown, static_spec=spec](
        storage.unsafe_ptr(), IndexList[2](3, 4)
    )

    assert_equal(tensor.rank, 2)
    assert_equal(tensor.size(), 12)


def test_shape_and_strides() raises:
    """Test shape() and strides() methods."""
    var storage = Array[Float32, 2 * 3 * 4](fill={})
    comptime spec = get_row_major_tensor_spec_static[
        DType.float32, 3, 2, 3, 4
    ]()
    var tensor = ManagedTensorSlice[io_spec=IOSpec.Unknown, static_spec=spec](
        storage.unsafe_ptr(), IndexList[3](2, 3, 4)
    )

    var shape = tensor.shape()
    assert_equal(shape[0], 2)
    assert_equal(shape[1], 3)
    assert_equal(shape[2], 4)

    var strides = tensor.strides()
    assert_equal(strides[0], 12)
    assert_equal(strides[1], 4)
    assert_equal(strides[2], 1)


def test_dim_size() raises:
    """Test dim_size methods (compile-time and runtime)."""
    var storage = Array[Float32, 5 * 7](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 5, 7]()
    var tensor = ManagedTensorSlice[io_spec=IOSpec.Unknown, static_spec=spec](
        storage.unsafe_ptr(), IndexList[2](5, 7)
    )

    # Test compile-time dim_size
    assert_equal(tensor.dim_size[0](), 5)
    assert_equal(tensor.dim_size[1](), 7)

    # Test runtime dim_size
    assert_equal(tensor.dim_size(0), 5)
    assert_equal(tensor.dim_size(1), 7)


def test_getitem_setitem() raises:
    """Test __getitem__ and __setitem__ operations."""
    var storage = Array[Float32, 2 * 3](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 2, 3]()
    var tensor = ManagedTensorSlice[
        mut=True, io_spec=IOSpec.Unknown, static_spec=spec
    ](storage.unsafe_ptr(), IndexList[2](2, 3))

    # Set values
    tensor[0, 0] = 1.0
    tensor[0, 1] = 2.0
    tensor[0, 2] = 3.0
    tensor[1, 0] = 4.0
    tensor[1, 1] = 5.0
    tensor[1, 2] = 6.0

    # Get values using variadic indices
    assert_equal(tensor[0, 0], 1.0)
    assert_equal(tensor[0, 1], 2.0)
    assert_equal(tensor[0, 2], 3.0)
    assert_equal(tensor[1, 0], 4.0)
    assert_equal(tensor[1, 1], 5.0)
    assert_equal(tensor[1, 2], 6.0)

    # Get values using IndexList
    assert_equal(tensor[IndexList[2](0, 2)], 3.0)
    assert_equal(tensor[IndexList[2](1, 0)], 4.0)


def test_simd_load_store() raises:
    """Test SIMD load and store operations."""
    var storage = Array[Float32, 8](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 1, 8]()
    var tensor = ManagedTensorSlice[
        mut=True, io_spec=IOSpec.Unknown, static_spec=spec
    ](storage.unsafe_ptr(), IndexList[1](8))

    # Store a SIMD vector
    var vec = SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0)
    tensor.store(IndexList[1](0), vec)

    var vec2 = SIMD[.float32, 4](5.0, 6.0, 7.0, 8.0)
    tensor.store(IndexList[1](4), vec2)

    # Load and verify
    var loaded = tensor.load[4](IndexList[1](0))
    assert_equal(loaded, SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0))

    var loaded2 = tensor.load[4](IndexList[1](4))
    assert_equal(loaded2, SIMD[.float32, 4](5.0, 6.0, 7.0, 8.0))


def test_to_layout_tensor() raises:
    """Test to_layout_tensor() conversion."""
    var storage = Array[Float32, 3 * 4](
        fill_with=lambda (i: Int) -> Float32: Float32(i)
    )
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 3, 4]()
    var tensor = ManagedTensorSlice[
        mut=True, io_spec=IOSpec.Unknown, static_spec=spec
    ](storage.unsafe_ptr(), IndexList[2](3, 4))

    # Convert to LayoutTensor
    var layout_tensor = tensor.to_layout_tensor()

    # Verify the layout tensor has the same data
    assert_equal(layout_tensor[0, 0], 0.0)
    assert_equal(layout_tensor[1, 1], 5.0)
    assert_equal(layout_tensor[2, 3], 11.0)

    # Verify dimensions
    assert_equal(Int(layout_tensor.runtime_layout.shape[0]), 3)
    assert_equal(Int(layout_tensor.runtime_layout.shape[1]), 4)

    # TODO(GEX-4147): ManagedTensorSlice needs to carry the Array's origin
    # `tensor` holds an untracked pointer into `storage`; keep `storage` alive
    # until the last read through it.
    _ = storage^


def test_stride_length() raises:
    """Test stride_length methods."""
    var storage = Array[Float32, 3 * 5](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 3, 5]()
    var tensor = ManagedTensorSlice[io_spec=IOSpec.Unknown, static_spec=spec](
        storage.unsafe_ptr(), IndexList[2](3, 5)
    )

    # Test compile-time stride_length
    assert_equal(tensor.stride_length[0](), 5)
    assert_equal(tensor.stride_length[1](), 1)

    # Test runtime stride_length
    assert_equal(tensor.stride_length(0), 5)
    assert_equal(tensor.stride_length(1), 1)


def test_simd_load_store_2d() raises:
    """Test SIMD load and store operations on 2D tensor."""
    var storage = Array[Float32, 4 * 8](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 4, 8]()
    var tensor = ManagedTensorSlice[
        mut=True, io_spec=IOSpec.Unknown, static_spec=spec
    ](storage.unsafe_ptr(), IndexList[2](4, 8))

    # Store vectors in each row
    for i in range(4):
        var vec = SIMD[.float32, 4](
            Float32(i * 10),
            Float32(i * 10 + 1),
            Float32(i * 10 + 2),
            Float32(i * 10 + 3),
        )
        tensor.store(IndexList[2](i, 0), vec)

        var vec2 = SIMD[.float32, 4](
            Float32(i * 10 + 4),
            Float32(i * 10 + 5),
            Float32(i * 10 + 6),
            Float32(i * 10 + 7),
        )
        tensor.store(IndexList[2](i, 4), vec2)

    # Load and verify
    var loaded_row0 = tensor.load[4](IndexList[2](0, 0))
    assert_equal(loaded_row0, SIMD[.float32, 4](0.0, 1.0, 2.0, 3.0))

    var loaded_row2 = tensor.load[4](IndexList[2](2, 4))
    assert_equal(loaded_row2, SIMD[.float32, 4](24.0, 25.0, 26.0, 27.0))

    var loaded_row3 = tensor.load[4](IndexList[2](3, 0))
    assert_equal(loaded_row3, SIMD[.float32, 4](30.0, 31.0, 32.0, 33.0))


def test_to_tile_tensor() raises:
    """Test to_tile_tensor() conversion."""
    var storage = Array[Float32, 3 * 4](
        fill_with=lambda (i: Int) -> Float32: Float32(i)
    )
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 3, 4]()
    var tensor = ManagedTensorSlice[
        mut=True, io_spec=IOSpec.Unknown, static_spec=spec
    ](storage.unsafe_ptr(), IndexList[2](3, 4))

    # Convert to TileTensor
    var tile_tensor = tensor.to_tile_tensor[.int64]()

    # Verify the layout tensor has the same data
    comptime assert tile_tensor.flat_rank == 2
    assert_equal(tile_tensor[0, 0], 0.0)
    assert_equal(tile_tensor[1, 1], 5.0)
    assert_equal(tile_tensor[2, 3], 11.0)

    # Verify dimensions
    assert_equal(tile_tensor.layout.shape[0]().value(), 3)
    assert_equal(tile_tensor.layout.shape[1]().value(), 4)

    # TODO(GEX-4147): ManagedTensorSlice needs to carry the Array's origin
    # `tensor` holds an untracked pointer into `storage`; keep `storage` alive
    # until the last read through it.
    _ = storage^


def test_shape_coord_static() raises:
    """Test shape_coord() preserves fully-static shape information."""
    var storage = Array[Float32, 3 * 4](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 3, 4]()
    var tensor = ManagedTensorSlice[io_spec=IOSpec.Unknown, static_spec=spec](
        storage.unsafe_ptr(), IndexList[2](3, 4)
    )

    var shape = tensor.shape_coord()

    # Every dimension is statically known, so this is a compile-time fact.
    comptime assert shape.all_dims_known
    comptime assert shape.element_types[0].static_value == 3
    comptime assert shape.element_types[1].static_value == 4

    # The runtime values still round-trip correctly.
    var index_list = coord_to_index_list(shape)
    assert_equal(index_list[0], 3)
    assert_equal(index_list[1], 4)


def test_shape_coord_mixed() raises:
    """Test shape_coord() encodes static dims while filling dynamic ones."""
    var storage = Array[Float32, 2 * 4](fill={})
    # dim 0 is dynamic (-1), dim 1 is static (4); strides are row-major.
    comptime mixed_layout = _IndexListToTileLayout[
        IndexList[2](-1, 4), IndexList[2](4, 1)
    ]
    comptime mixed_spec = StaticTensorSpec[
        DType.float32, 2, static_layout=mixed_layout
    ](align_of[DType.float32](), AddressSpace.GENERIC)
    var tensor = ManagedTensorSlice[
        io_spec=IOSpec.Unknown, static_spec=mixed_spec
    ](storage.unsafe_ptr(), IndexList[2](2, 4))

    var shape = tensor.shape_coord()

    # The static/dynamic structure is preserved in the Coord's type.
    comptime assert not shape.all_dims_known
    comptime assert not shape.element_types[0].is_static_value
    comptime assert shape.element_types[1].is_static_value
    comptime assert shape.element_types[1].static_value == 4

    # The dynamic dimension is filled from the runtime shape.
    var index_list = coord_to_index_list(shape)
    assert_equal(index_list[0], 2)
    assert_equal(index_list[1], 4)


def test_strides_coord_static() raises:
    """Test strides_coord() preserves fully-static stride information."""
    var storage = Array[Float32, 3 * 4](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 3, 4]()
    var tensor = ManagedTensorSlice[io_spec=IOSpec.Unknown, static_spec=spec](
        storage.unsafe_ptr(), IndexList[2](3, 4)
    )

    var strides = tensor.strides_coord()

    # Row-major strides are statically known, so this is a compile-time fact.
    comptime assert strides.all_dims_known
    comptime assert strides.element_types[0].static_value == 4
    comptime assert strides.element_types[1].static_value == 1

    # The runtime values still round-trip correctly.
    var index_list = coord_to_index_list(strides)
    assert_equal(index_list[0], 4)
    assert_equal(index_list[1], 1)


def test_strides_coord_mixed() raises:
    """Test strides_coord() encodes static strides while filling dynamic ones.
    """
    var storage = Array[Float32, 2 * 4](fill={})
    # Shape is static (2, 4); stride 0 is dynamic (-1) and stride 1 is static.
    comptime mixed_layout = _IndexListToTileLayout[
        IndexList[2](2, 4), IndexList[2](-1, 1)
    ]
    comptime mixed_spec = StaticTensorSpec[
        DType.float32, 2, static_layout=mixed_layout
    ](align_of[DType.float32](), AddressSpace.GENERIC)
    var tensor = ManagedTensorSlice[
        io_spec=IOSpec.Unknown, static_spec=mixed_spec
    ](storage.unsafe_ptr(), IndexList[2](2, 4), IndexList[2](4, 1))

    var strides = tensor.strides_coord()

    # The static/dynamic structure is preserved in the Coord's type.
    comptime assert not strides.all_dims_known
    comptime assert not strides.element_types[0].is_static_value
    comptime assert strides.element_types[1].is_static_value
    comptime assert strides.element_types[1].static_value == 1

    # The dynamic stride is filled from the runtime strides.
    var index_list = coord_to_index_list(strides)
    assert_equal(index_list[0], 4)
    assert_equal(index_list[1], 1)


# ===----------------------------------------------------------------------=== #
# `foreach` — the value (unified-closure) overload and its parametric twin.
#
# Every case runs from a parametric helper: the callback's return type has to
# name `dtype` as a parameter reference for it to unify with the overload's
# inferred `dtype`. A literal (`SIMD[DType.float32, width]`) does not.
#
# A test that needs two buffers carves both out of one `Array`. Two separate
# `Array`s can end up sharing a stack slot once `unsafe_ptr()` is taken, which
# silently aliases them.
# ===----------------------------------------------------------------------=== #


def _lane_ramp[dtype: DType, width: Int](start: Int) -> SIMD[dtype, width]:
    """`[start, start + 1, ...]` — a body that varies per lane, so a wrong
    index or a collapsed vector shows up as a wrong element."""
    var v = SIMD[dtype, width](0)
    for lane in range(width):
        v[lane] = Scalar[dtype](start + lane)
    return v


def _flat_view[
    dtype: DType, n: Int
](
    ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    out result: ManagedTensorSlice[
        mut=True,
        io_spec=IOSpec.Unknown,
        static_spec=get_row_major_tensor_spec_static[dtype, 1, n](),
    ],
):
    """A readable/writable rank-1 view, for setting up inputs and checking
    results without going back through the raw pointer."""
    return {ptr, IndexList[1](n)}


def _flat_output[
    dtype: DType, n: Int
](
    ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    out result: ManagedTensorSlice[
        io_spec=IOSpec.Output,
        static_spec=get_row_major_tensor_spec_static[dtype, 1, n](),
    ],
):
    return {ptr, IndexList[1](n)}


def _flat_input[
    dtype: DType, n: Int
](
    ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    out result: ManagedTensorSlice[
        io_spec=IOSpec.Input,
        static_spec=get_row_major_tensor_spec_static[dtype, 1, n](),
    ],
):
    return {ptr, IndexList[1](n)}


def _fill_ramp[
    dtype: DType, n: Int
](tensor: ManagedTensorSlice[mut=True, dtype=dtype, rank=1, ...]):
    for i in range(n):
        tensor.store(IndexList[1](i), SIMD[dtype, 1](Scalar[dtype](i)))


def _check_value_form_rank1[dtype: DType]() raises:
    comptime N = 32
    var storage = Array[Scalar[dtype], N](fill={})
    var base = storage.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var out = _flat_output[dtype, N](base)
    var view = _flat_view[dtype, N](base)

    var bias = Scalar[dtype](3)

    @inline(.always)
    def body[width: Int](idx: Coord) {var bias} -> SIMD[dtype, width]:
        return SIMD[dtype, width](bias)

    var ctx = DeviceContext(api="cpu")
    foreach(body, out, ctx)

    for i in range(N):
        assert_equal(view.load[1](IndexList[1](i)), Scalar[dtype](3))
    _ = storage^


def test_foreach_value_form_rank1() raises:
    """The value overload resolves and runs over a rank-1 tensor."""
    _check_value_form_rank1[DType.float32]()


def _check_value_form_rank2[dtype: DType]() raises:
    comptime ROWS = 3
    comptime COLS = 4
    var storage = Array[Scalar[dtype], ROWS * COLS](fill={})
    var base = storage.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    comptime spec = get_row_major_tensor_spec_static[dtype, 2, ROWS, COLS]()
    var out = ManagedTensorSlice[io_spec=IOSpec.Output, static_spec=spec](
        base, IndexList[2](ROWS, COLS)
    )
    var view = ManagedTensorSlice[
        mut=True, io_spec=IOSpec.Unknown, static_spec=spec
    ](base, IndexList[2](ROWS, COLS))

    var scale = Scalar[dtype](10)

    # `Coord` carries the index of the vector's first element, so the body
    # fills the remaining lanes along the last axis itself. A wrong index or a
    # collapsed vector shows up as a wrong element.
    @inline(.always)
    def body[width: Int](idx: Coord) {var scale} -> SIMD[dtype, width]:
        var il = coord_to_index_list(idx)
        var base_val = Scalar[dtype](il[0]) * scale + Scalar[dtype](il[1])
        var v = SIMD[dtype, width](0)
        for lane in range(width):
            v[lane] = base_val + Scalar[dtype](lane)
        return v

    var ctx = DeviceContext(api="cpu")
    foreach(body, out, ctx)

    for r in range(ROWS):
        for c in range(COLS):
            assert_equal(
                view.load[1](IndexList[2](r, c)), Scalar[dtype](r * 10 + c)
            )
    _ = storage^


def test_foreach_value_form_rank2() raises:
    """The value overload handles rank 2, including `Coord` conversion."""
    _check_value_form_rank2[DType.float32]()


def _check_value_matches_parametric[dtype: DType]() raises:
    comptime N = 32
    var storage = Array[Scalar[dtype], 2 * N](fill={})
    var base = storage.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var value_ptr = base
    var param_ptr = base.unsafe_offset(N)

    var value_out = _flat_output[dtype, N](value_ptr)
    var param_out = _flat_output[dtype, N](param_ptr)
    var value_view = _flat_view[dtype, N](value_ptr)
    var param_view = _flat_view[dtype, N](param_ptr)

    var ctx = DeviceContext(api="cpu")
    var two = Scalar[dtype](2)

    # Both bodies compute `2 * i + 1`. The value form carries its multiplier in
    # a capture list; the comptime form cannot capture at all without
    # `@__parameter`, so it spells the multiplier inline. Same arithmetic, two
    # dispatch paths — any divergence is the new overload's fault.
    @inline(.always)
    def value_body[width: Int](idx: Coord) {var two} -> SIMD[dtype, width]:
        return _lane_ramp[dtype, width](coord_to_index_list(idx)[0]) * two + 1

    foreach(value_body, value_out, ctx)

    @inline(.always)
    def param_body[width: Int](idx: Coord) capturing -> SIMD[dtype, width]:
        return _lane_ramp[dtype, width](coord_to_index_list(idx)[0]) * 2 + 1

    foreach[param_body](param_out, ctx)

    for i in range(N):
        assert_equal(
            value_view.load[1](IndexList[1](i)),
            param_view.load[1](IndexList[1](i)),
        )
        assert_equal(
            value_view.load[1](IndexList[1](i)), Scalar[dtype](i * 2 + 1)
        )
    _ = storage^


def test_foreach_value_matches_parametric() raises:
    """Differential: both overloads produce identical results for one body."""
    _check_value_matches_parametric[DType.float32]()


def _check_capture_outer_tensor[dtype: DType]() raises:
    comptime N = 32
    var storage = Array[Scalar[dtype], 2 * N](fill={})
    var base = storage.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var in_ptr = base
    var out_ptr = base.unsafe_offset(N)

    _fill_ramp[dtype, N](_flat_view[dtype, N](in_ptr))
    # A mut view, not an `IOSpec.Input` one, even though the migrated examples
    # capture inputs: `IOSpec.Input` means `mut=False`, which
    # `simd_load_from_managed_tensor_slice` turns into an LLVM
    # `!invariant.load`. Filling a buffer and reading it back in one function
    # breaks that promise — at N=32 the optimizer forwarded the `Array`
    # zero-init past `_fill_ramp` for the first vector. Real ops never write
    # their own input, and the overload does not depend on the `IOSpec`.
    var x = _flat_view[dtype, N](in_ptr)
    var out = _flat_output[dtype, N](out_ptr)
    var view = _flat_view[dtype, N](out_ptr)

    # The shape every migrated `max/examples/` caller uses: capture the input
    # tensor and read it through `load`.
    @inline(.always)
    def body[width: Int](idx: Coord) {var x} -> SIMD[dtype, width]:
        return x.load[width](idx) + 1

    var ctx = DeviceContext(api="cpu")
    foreach(body, out, ctx)

    for i in range(N):
        assert_equal(view.load[1](IndexList[1](i)), Scalar[dtype](i + 1))
    _ = storage^


def test_foreach_captures_outer_tensor() raises:
    """A captured input tensor, the shape the migrated examples use."""
    _check_capture_outer_tensor[DType.float32]()


def _check_capture_tensor_and_scalar[dtype: DType]() raises:
    comptime N = 32
    var storage = Array[Scalar[dtype], 2 * N](fill={})
    var base = storage.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var in_ptr = base
    var out_ptr = base.unsafe_offset(N)

    _fill_ramp[dtype, N](_flat_view[dtype, N](in_ptr))
    # A mut view, for the invariant-load reason given above.
    var x = _flat_view[dtype, N](in_ptr)
    var out = _flat_output[dtype, N](out_ptr)
    var view = _flat_view[dtype, N](out_ptr)
    var addend = Scalar[dtype](7)

    # The `add_constant` shape: a tensor and a loose scalar in one capture list.
    @inline(.always)
    def body[width: Int](idx: Coord) {var x, var addend} -> SIMD[dtype, width]:
        return x.load[width](idx) + addend

    var ctx = DeviceContext(api="cpu")
    foreach(body, out, ctx)

    for i in range(N):
        assert_equal(view.load[1](IndexList[1](i)), Scalar[dtype](i + 7))
    _ = storage^


def test_foreach_captures_tensor_and_scalar() raises:
    """Mixed capture list: an outer tensor alongside an outer scalar."""
    _check_capture_tensor_and_scalar[DType.float32]()


def _check_capture_input_tensor[dtype: DType]() raises:
    comptime N = 32
    comptime FILL = 7
    # The one test that captures a real `IOSpec.Input` tensor, so the shape the
    # migrated examples use is covered with its invariant loads intact. It can
    # do that only because the input is filled once, at construction, and never
    # written again: with nothing stale in the buffer there is nothing for the
    # optimizer to forward, which is the hazard `_check_capture_outer_tensor`
    # documents. The ramp lives in the body instead of the buffer, so the
    # expected value still varies per index.
    var in_storage = Array[Scalar[dtype], N](fill=Scalar[dtype](FILL))
    var out_storage = Array[Scalar[dtype], N](fill={})
    var in_ptr = in_storage.unsafe_ptr().unsafe_origin_cast[
        MutUntrackedOrigin
    ]()
    var out_ptr = out_storage.unsafe_ptr().unsafe_origin_cast[
        MutUntrackedOrigin
    ]()

    var x = _flat_input[dtype, N](in_ptr)
    var out = _flat_output[dtype, N](out_ptr)
    var view = _flat_view[dtype, N](out_ptr)

    @inline(.always)
    def body[width: Int](idx: Coord) {var x} -> SIMD[dtype, width]:
        return x.load[width](idx) + _lane_ramp[dtype, width](
            coord_to_index_list(idx)[0]
        )

    var ctx = DeviceContext(api="cpu")
    foreach(body, out, ctx)

    for i in range(N):
        assert_equal(view.load[1](IndexList[1](i)), Scalar[dtype](FILL + i))
    _ = in_storage^
    _ = out_storage^


def test_foreach_captures_input_tensor() raises:
    """A captured `IOSpec.Input` tensor, invariant loads and all."""
    _check_capture_input_tensor[DType.float32]()


def _check_simd_width_one[dtype: DType]() raises:
    comptime N = 32
    var storage = Array[Scalar[dtype], N](fill={})
    var base = storage.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var out = _flat_output[dtype, N](base)
    var view = _flat_view[dtype, N](base)
    var one = Scalar[dtype](1)

    # `simd_width=1` is what `image_pipeline.mojo` and `grayscale.mojo` pass;
    # the body asserts it actually arrives.
    @inline(.always)
    def body[width: Int](idx: Coord) {var one} -> SIMD[dtype, width]:
        comptime assert width == 1, "simd_width=1 was not honored"
        return SIMD[dtype, width](
            Scalar[dtype](coord_to_index_list(idx)[0]) * one
        )

    var ctx = DeviceContext(api="cpu")
    foreach[simd_width=1](body, out, ctx)

    for i in range(N):
        assert_equal(view.load[1](IndexList[1](i)), Scalar[dtype](i))
    _ = storage^


def test_foreach_honors_simd_width() raises:
    """A non-default `simd_width` reaches the body."""
    _check_simd_width_one[DType.float32]()


@fieldwise_init
struct _MarkingOutFusion[fusion_dtype: DType](OutputFusion):
    """Records stores through the fusion path, offset by 100 so a store that
    bypassed the fusion is distinguishable from one that used it."""

    var dst: Pointer[Scalar[Self.fusion_dtype], MutUntrackedOrigin]

    def store[
        dtype: DType,
        rank: Int,
        simd_width: SIMDLength,
        element_alignment: Int = 1,
    ](self, idx: IndexList[rank], val: SIMD[dtype, simd_width]):
        var start = idx[rank - 1]
        for lane in range(Int(simd_width)):
            self.dst.unsafe_offset(start + lane).unsafe_store(
                rebind[Scalar[Self.fusion_dtype]](val[lane]) + 100
            )


def _check_fused_store[dtype: DType]() raises:
    comptime N = 32
    # `direct` is what the tensor points at; `fused` is where the fusion
    # writes. A `foreach` that stored straight to the data pointer would fill
    # `direct` and leave `fused` untouched — the regression this guards.
    var storage = Array[Scalar[dtype], 2 * N](fill={})
    var base = storage.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var direct_ptr = base
    var fused_ptr = base.unsafe_offset(N)

    var direct_view = _flat_view[dtype, N](direct_ptr)
    var fused_view = _flat_view[dtype, N](fused_ptr)
    for i in range(N):
        direct_view.store(IndexList[1](i), SIMD[dtype, 1](Scalar[dtype](-1)))
        fused_view.store(IndexList[1](i), SIMD[dtype, 1](Scalar[dtype](-1)))

    var out = _flat_output[dtype, N](direct_ptr)
    var fused_out = out._bind_to_fused_output(_MarkingOutFusion(fused_ptr))
    var one = Scalar[dtype](1)

    @inline(.always)
    def body[width: Int](idx: Coord) {var one} -> SIMD[dtype, width]:
        return SIMD[dtype, width](
            Scalar[dtype](coord_to_index_list(idx)[0]) * one
        )

    var ctx = DeviceContext(api="cpu")
    foreach[simd_width=1](body, fused_out, ctx)

    for i in range(N):
        assert_equal(
            fused_view.load[1](IndexList[1](i)), Scalar[dtype](i + 100)
        )
        assert_equal(direct_view.load[1](IndexList[1](i)), Scalar[dtype](-1))
    _ = storage^


def test_foreach_routes_through_fused_store() raises:
    """The value overload stores through `_fused_store`, not the data pointer.

    Without this the wrapper could write straight to the tensor and silently
    break output fusion while every other test still passed.
    """
    _check_fused_store[DType.float32]()


def _check_parametric_form_still_resolves[dtype: DType]() raises:
    comptime N = 32
    var storage = Array[Scalar[dtype], N](fill={})
    var base = storage.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var out = _flat_output[dtype, N](base)
    var view = _flat_view[dtype, N](base)

    # No capture list: a `capturing` body written without `@__parameter` does
    # not actually capture, so anything it needs has to come from `idx`.
    @inline(.always)
    def body[width: Int](idx: Coord) capturing -> SIMD[dtype, width]:
        return _lane_ramp[dtype, width](coord_to_index_list(idx)[0]) + 5

    var ctx = DeviceContext(api="cpu")
    foreach[body](out, ctx)

    for i in range(N):
        assert_equal(view.load[1](IndexList[1](i)), Scalar[dtype](i + 5))
    _ = storage^


def test_foreach_parametric_form_still_resolves() raises:
    """Regression guard: adding the value overload left the comptime one
    callable and unambiguous."""
    _check_parametric_form_still_resolves[DType.float32]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
