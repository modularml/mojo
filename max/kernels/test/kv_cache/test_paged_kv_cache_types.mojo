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

from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCache,
    PagedKVCacheCollection,
)
from layout import (
    DefaultEngine,
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    UNKNOWN_VALUE,
    Coord,
)
from std.memory import alloc
from std.sys import size_of
from std.testing import assert_true

from std.utils.coord import dyn_coord
from std.utils.index import IndexList
from std.collections import OptionalReg

comptime kv_params = KVCacheStaticParams(num_heads=16, head_size=16)


def do_test[
    page_size: Int,
    layout_block_size: Int,
    scale_dtype: Optional[DType] = None,
]() raises:
    comptime batch_size = 16
    comptime max_num_blocks = 100
    comptime shape = IndexList[6](
        100,
        2,
        1,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )

    var blocks_ptr = List(length=shape.flattened_length(), fill=Float32(0))
    var blocks = LayoutTensor[.float32, Layout.row_major[6]()](
        blocks_ptr, RuntimeLayout[Layout.row_major[6]()].row_major(shape)
    )
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    var cache_lengths_ptr = List(length=batch_size, fill=UInt32(0))
    var cache_lengths = LayoutTensor[.uint32, layout_1d](
        cache_lengths_ptr,
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size)),
    )
    comptime layout_2d = Layout.row_major[2]()
    var lookup_table_ptr = List(
        length=batch_size * max_num_blocks, fill=UInt32(0)
    )
    var lookup_table = LayoutTensor[.uint32, layout_2d](
        lookup_table_ptr,
        RuntimeLayout[layout_2d].row_major(
            IndexList[2](batch_size, max_num_blocks)
        ),
    )
    for i in range(batch_size):
        cache_lengths[i] = UInt32(i)
        for j in range(max_num_blocks):
            lookup_table[i, j] = UInt32(j)

    var max_seq_length = UInt32(2048)
    var max_cache_length = UInt32(2048)

    # Concrete scales element type: the real scale dtype when set, else the
    # block dtype (`float32`). Used for both the `scales` declaration and its
    # assignment so the compiler folds the types without a rebind.
    comptime scales_dtype = scale_dtype.or_else(DType.float32)
    var scales: OptionalReg[
        LayoutTensor[scales_dtype, Layout.row_major[6](), MutUntrackedOrigin]
    ] = None

    comptime if scale_dtype == DType.float8_e4m3fn:
        # Use the same shape as the blocks.
        var scales_ptr = alloc[Scalar[scales_dtype]](shape.flattened_length())
        scales = LayoutTensor[scales_dtype, Layout.row_major[6]()](
            scales_ptr,
            RuntimeLayout[Layout.row_major[6]()].row_major(shape),
        ).fill(0)

    var collection = PagedKVCacheCollection[
        DType.float32,
        kv_params,
        page_size,
        scale_dtype_=scale_dtype,
    ](
        LayoutTensor[blocks.dtype, Layout.row_major[6]()](
            blocks.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                blocks.runtime_layout.shape.value,
                blocks.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, cache_lengths.dtype, Layout(UNKNOWN_VALUE)](
            cache_lengths.ptr.as_imm().as_unsafe_any_origin(),
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                cache_lengths.runtime_layout.shape.value,
                cache_lengths.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, lookup_table.dtype, Layout.row_major[2]()](
            lookup_table.ptr,
            RuntimeLayout[Layout.row_major[2]()](
                lookup_table.runtime_layout.shape.value,
                lookup_table.runtime_layout.stride.value,
            ),
        ),
        max_seq_length,
        max_cache_length,
        scales,
    )

    comptime layout = Layout(
        IntTuple(layout_block_size, kv_params.head_size),
        IntTuple(kv_params.num_heads * kv_params.head_size, 1),
    )

    var cache = collection.get_key_cache(1)
    _ = cache.block_paged_ptr[layout_block_size](1, layout_block_size, 0)


def test_paged_kv_cache_stride_is_unknown() raises:
    """Test that PagedKVCache has UNKNOWN stride[0] for view tensor correctness.

    PagedKVCache is a 4D view of a 6D parent tensor. The outer stride depends
    on num_layers from the parent tensor, which is unknown at compile time.
    Setting stride[0] = UNKNOWN_VALUE ensures offset calculations use the
    runtime stride rather than an incorrect compile-time value.

    This is a regression test - previously Layout.row_major() computed stride[0]
    from just the 4D shape, which was incorrect for view tensors.
    """
    comptime CacheType = PagedKVCache[
        DType.float32,
        kv_params,
        16,
        MutUntrackedOrigin,
        DefaultEngine[element_width=1],
        ImmUntrackedOrigin,
        ImmUntrackedOrigin,
        MutUntrackedOrigin,
    ]

    # Verify stride[0] is UNKNOWN_VALUE
    comptime stride_0 = CacheType.blocks_layout.stride[0].value()
    assert_true(
        stride_0 == UNKNOWN_VALUE,
        String("PagedKVCache stride[0] should be UNKNOWN_VALUE (-1), got ")
        + String(stride_0),
    )

    # Verify inner strides are still known (enables partial constant folding)
    comptime stride_1 = CacheType.blocks_layout.stride[1].value()
    comptime stride_2 = CacheType.blocks_layout.stride[2].value()
    comptime stride_3 = CacheType.blocks_layout.stride[3].value()

    comptime expected_stride_1 = kv_params.num_heads * kv_params.head_size
    comptime expected_stride_2 = kv_params.head_size
    comptime expected_stride_3 = 1

    assert_true(
        stride_1 == expected_stride_1,
        String("PagedKVCache stride[1] should be ")
        + String(expected_stride_1)
        + ", got "
        + String(stride_1),
    )
    assert_true(
        stride_2 == expected_stride_2,
        String("PagedKVCache stride[2] should be ")
        + String(expected_stride_2)
        + ", got "
        + String(stride_2),
    )
    assert_true(
        stride_3 == expected_stride_3,
        String("PagedKVCache stride[3] should be ")
        + String(expected_stride_3)
        + ", got "
        + String(stride_3),
    )


def test_paged_kv_cache_offset_correctness() raises:
    """Test that PagedKVCache offset calculations use correct runtime strides.

    This test verifies that when accessing elements through a PagedKVCache view,
    the correct values from the underlying 6D tensor are returned. This catches
    bugs where compile-time strides don't match the actual memory layout.

    The 6D parent tensor has shape:
        [num_blocks, 2, num_layers, page_size, num_heads, head_size]

    The 4D PagedKVCache view has shape:
        [num_blocks, page_size, num_heads, head_size]

    The view's stride[0] must account for the skipped dimensions (2, num_layers).
    """
    comptime num_blocks = 4
    comptime num_layers = 3
    comptime page_size = 2
    comptime num_heads = 2
    comptime head_size = 4

    comptime kv_params_small = KVCacheStaticParams(
        num_heads=num_heads, head_size=head_size
    )

    # 6D shape: [num_blocks, 2, num_layers, page_size, num_heads, head_size]
    comptime shape_6d = IndexList[6](
        num_blocks, 2, num_layers, page_size, num_heads, head_size
    )
    comptime total_elems = shape_6d.flattened_length()

    # Allocate and fill with unique values (value = flattened index)
    var blocks_ptr = List(length=total_elems, fill=Float32(0))
    for i in range(total_elems):
        blocks_ptr[i] = Float32(i)

    var blocks = LayoutTensor[.float32, Layout.row_major[6]()](
        blocks_ptr, RuntimeLayout[Layout.row_major[6]()].row_major(shape_6d)
    )

    # Create minimal supporting tensors
    comptime batch_size = 1
    var cache_lengths_ptr = List(length=batch_size, fill=UInt32(0))
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    var cache_lengths = LayoutTensor[.uint32, layout_1d](
        cache_lengths_ptr,
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size)),
    )

    comptime layout_2d = Layout.row_major[2]()
    var lookup_table_ptr = List(length=batch_size * num_blocks, fill=UInt32(0))
    for i in range(num_blocks):
        lookup_table_ptr[i] = UInt32(i)  # Identity mapping
    var lookup_table = LayoutTensor[.uint32, layout_2d](
        lookup_table_ptr,
        RuntimeLayout[layout_2d].row_major(
            IndexList[2](batch_size, num_blocks)
        ),
    )

    # Create collection
    var collection = PagedKVCacheCollection[
        DType.float32, kv_params_small, page_size
    ](
        LayoutTensor[.float32, Layout.row_major[6]()](
            blocks.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                blocks.runtime_layout.shape.value,
                blocks.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, Layout(UNKNOWN_VALUE)](
            cache_lengths.ptr,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                cache_lengths.runtime_layout.shape.value,
                cache_lengths.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, Layout.row_major[2]()](
            lookup_table.ptr,
            RuntimeLayout[Layout.row_major[2]()](
                lookup_table.runtime_layout.shape.value,
                lookup_table.runtime_layout.stride.value,
            ),
        ),
        UInt32(page_size),
        UInt32(page_size),
    )

    # Test: Access element through key cache view's blocks tensor and verify correct value
    # Get key cache for layer 1 (kv_idx=0, layer_idx=1)
    var key_cache = collection.get_key_cache(1)

    # Directly access the blocks tensor using IndexList
    # 4D coords [block=1, page=0, head=1, dim=2]
    # In 6D, this corresponds to [block=1, kv=0, layer=1, page=0, head=1, dim=2]
    #
    # 6D strides (row-major):
    #   stride[0] = 2 * 3 * 2 * 2 * 4 = 96
    #   stride[1] = 3 * 2 * 2 * 4 = 48
    #   stride[2] = 2 * 2 * 4 = 16
    #   stride[3] = 2 * 4 = 8
    #   stride[4] = 4
    #   stride[5] = 1
    #
    # Expected 6D offset = 1*96 + 0*48 + 1*16 + 0*8 + 1*4 + 2*1 = 96 + 16 + 4 + 2 = 118
    comptime expected_6d_offset = (
        1 * (2 * num_layers * page_size * num_heads * head_size)
        + 0 * (num_layers * page_size * num_heads * head_size)
        + 1 * (page_size * num_heads * head_size)
        + 0 * (num_heads * head_size)
        + 1 * head_size
        + 2
    )

    # The 4D view's runtime stride[0] should be 2 * num_layers * page_size * num_heads * head_size
    # If the bug existed (using compile-time stride[0] = page_size * num_heads * head_size),
    # we'd compute wrong offset: 1*16 + 0*8 + 1*4 + 2 = 22 (wrong!)

    # Access via the blocks TileTensor - this tests the layout offset computation
    var idx = dyn_coord[.int64]((1, 0, 1, 2))
    var value = key_cache.blocks.raw_load[width=1](key_cache.blocks.layout(idx))
    var expected_value = Float32(expected_6d_offset)

    assert_true(
        value == expected_value,
        String("PagedKVCache returned wrong value! Got ")
        + String(value)
        + " but expected "
        + String(expected_value)
        + " (at 6D offset "
        + String(expected_6d_offset)
        + "). This indicates stride[0] is using incorrect compile-time value.",
    )


def test_paged_kv_cache_quantization() raises:
    comptime CacheType = PagedKVCache[
        DType.float32,
        kv_params,
        16,
        MutUntrackedOrigin,
        DefaultEngine[element_width=1],
        ImmUntrackedOrigin,
        ImmUntrackedOrigin,
        MutUntrackedOrigin,
        scale_dtype_=DType.float8_e4m3fn,
        quantization_granularity_=256,
    ]
    assert_true(CacheType.quantization_enabled, "Quantization not enabled")
    assert_true(
        CacheType.quantization_granularity == 256,
        "Incorrect quantization granularity",
    )


def test_scales_resolve_through_their_own_lookup_table() raises:
    """A distinct scales_lookup_table must resolve scale pages independently
    of lookup_table, not silently fall back to it.
    """
    comptime kv_params_small = KVCacheStaticParams(num_heads=1, head_size=4)
    comptime page_size = 1
    comptime granularity = 4
    comptime batch_size = 1
    comptime num_value_blocks = 1
    comptime num_scale_blocks = 3

    comptime blocks_shape = IndexList[6](
        num_value_blocks,
        2,
        1,
        page_size,
        kv_params_small.num_heads,
        kv_params_small.head_size,
    )
    var blocks_ptr = List(
        length=blocks_shape.flattened_length(), fill=Float32(0)
    )
    var blocks = LayoutTensor[.float32, Layout.row_major[6]()](
        blocks_ptr,
        RuntimeLayout[Layout.row_major[6]()].row_major(blocks_shape),
    )

    comptime layout_1d = Layout(UNKNOWN_VALUE)
    var cache_lengths_ptr = List(length=batch_size, fill=UInt32(1))
    var cache_lengths = LayoutTensor[.uint32, layout_1d](
        cache_lengths_ptr,
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size)),
    )

    comptime layout_2d = Layout.row_major[2]()

    # Both LUTs are `alloc`'d rather than backed by a `List`: the collection's
    # `scales_lookup_table` parameter shares one origin type with
    # `lookup_table` (both resolve through the same `lookup_table_origin`
    # struct param), and `alloc[T](n)` returns a fixed `MutUntrackedOrigin`
    # regardless of call site, so both tensors unify to the same type even
    # though they're separately allocated.

    # Values resolve through block 0.
    var lookup_table_ptr = alloc[UInt32](batch_size)
    lookup_table_ptr[0] = UInt32(0)
    var lookup_table = LayoutTensor[mut=False, .uint32, layout_2d](
        lookup_table_ptr,
        RuntimeLayout[layout_2d].row_major(IndexList[2](batch_size, 1)),
    )

    # Scales resolve through block 2 -- distinct from what lookup_table
    # points at, and only reachable through scales_lookup_table.
    var scales_lookup_table_ptr = alloc[UInt32](batch_size)
    scales_lookup_table_ptr[0] = UInt32(2)
    var scales_lookup_table_opt: OptionalReg[
        LayoutTensor[mut=False, .uint32, layout_2d, MutUntrackedOrigin]
    ] = LayoutTensor[mut=False, .uint32, layout_2d](
        scales_lookup_table_ptr,
        RuntimeLayout[layout_2d].row_major(IndexList[2](batch_size, 1)),
    )

    comptime scales_shape = IndexList[6](
        num_scale_blocks, 2, 1, page_size, kv_params_small.num_heads, 1
    )
    var scales_ptr = alloc[Float32](scales_shape.flattened_length())
    var scales = LayoutTensor[.float32, Layout.row_major[6]()](
        scales_ptr,
        RuntimeLayout[Layout.row_major[6]()].row_major(scales_shape),
    ).fill(Float32(-1.0))
    # Block 0 -- where lookup_table points -- holds a decoy value that a
    # buggy fallback to lookup_table would read instead.
    scales[0, 0, 0, 0, 0, 0] = Float32(111.0)
    # Block 2 -- where scales_lookup_table points -- holds the real value.
    scales[2, 0, 0, 0, 0, 0] = Float32(222.0)

    var scales_opt: OptionalReg[
        LayoutTensor[.float32, Layout.row_major[6](), MutUntrackedOrigin]
    ] = scales

    var collection = PagedKVCacheCollection[
        DType.float32,
        kv_params_small,
        page_size,
        scale_dtype_=DType.float32,
        quantization_granularity_=granularity,
    ](
        LayoutTensor[blocks.dtype, Layout.row_major[6]()](
            blocks.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                blocks.runtime_layout.shape.value,
                blocks.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, cache_lengths.dtype, Layout(UNKNOWN_VALUE)](
            cache_lengths.ptr.as_imm().as_unsafe_any_origin(),
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                cache_lengths.runtime_layout.shape.value,
                cache_lengths.runtime_layout.stride.value,
            ),
        ),
        lookup_table,
        UInt32(page_size),
        UInt32(page_size),
        scales_opt,
        scales_lookup_table_opt,
    )

    var key_cache = collection.get_key_cache(0)
    var scale = key_cache.load_scale[width=1](
        bs=0, head_idx=0, tok_idx=0, head_dim_idx=0
    )

    assert_true(
        Float32(scale) == Float32(222.0),
        String("scale resolved through the wrong LUT: got ")
        + String(scale)
        + ", expected 222.0 (via scales_lookup_table, not lookup_table)",
    )


def _build_collection_and_page_stride[
    page_size: Int, num_layers: Int
](page_stride: Int) raises -> Int:
    """Returns the byte distance between consecutive pages of a collection.

    Builds a paged collection over `num_blocks` pages, then measures the page
    stride the way a kernel experiences it: the distance between the pointers
    two adjacent physical pages resolve to. `page_stride` is forwarded
    verbatim, so -1 exercises the packed path.
    """
    comptime num_blocks = 8
    comptime batch_size = 1
    comptime shape = IndexList[6](
        num_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )

    # The pool owes `num_blocks * page_stride` elements when a padded stride is
    # supplied, since the tensor below still describes packed pages.
    var packed_page = (
        2 * num_layers * page_size * kv_params.num_heads * kv_params.head_size
    )
    var page = packed_page if page_stride < 0 else page_stride
    var blocks_ptr = List(length=num_blocks * page, fill=Float32(0))
    var blocks = LayoutTensor[.float32, Layout.row_major[6]()](
        blocks_ptr, RuntimeLayout[Layout.row_major[6]()].row_major(shape)
    )

    comptime layout_1d = Layout(UNKNOWN_VALUE)
    var cache_lengths_ptr = List(length=batch_size, fill=UInt32(0))
    var cache_lengths = LayoutTensor[.uint32, layout_1d](
        cache_lengths_ptr,
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size)),
    )
    comptime layout_2d = Layout.row_major[2]()
    var lookup_table_ptr = List(length=batch_size * num_blocks, fill=UInt32(0))
    var lookup_table = LayoutTensor[.uint32, layout_2d](
        lookup_table_ptr,
        RuntimeLayout[layout_2d].row_major(
            IndexList[2](batch_size, num_blocks)
        ),
    )
    for j in range(num_blocks):
        lookup_table[0, j] = UInt32(j)

    var collection = PagedKVCacheCollection[
        DType.float32, kv_params, page_size
    ](
        LayoutTensor[blocks.dtype, Layout.row_major[6]()](
            blocks.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                blocks.runtime_layout.shape.value,
                blocks.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, cache_lengths.dtype, Layout(UNKNOWN_VALUE)](
            cache_lengths.ptr.as_imm().as_unsafe_any_origin(),
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                cache_lengths.runtime_layout.shape.value,
                cache_lengths.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, lookup_table.dtype, Layout.row_major[2]()](
            lookup_table.ptr,
            RuntimeLayout[Layout.row_major[2]()](
                lookup_table.runtime_layout.shape.value,
                lookup_table.runtime_layout.stride.value,
            ),
        ),
        UInt32(page_size * num_blocks),
        UInt32(page_size * num_blocks),
        page_stride=page_stride,
    )

    var cache = collection.get_key_cache(0)
    var page0 = cache.block_paged_ptr[page_size](0, 0, 0)
    var page1 = cache.block_paged_ptr[page_size](0, page_size, 0)
    return Int(page1) - Int(page0)


def test_page_stride_of_minus_one_keeps_pages_packed() raises:
    """A -1 stride must reproduce the geometry callers already depend on."""
    comptime page_size = 16
    comptime num_layers = 3
    var measured = _build_collection_and_page_stride[page_size, num_layers](-1)
    comptime packed = (
        2 * num_layers * page_size * kv_params.num_heads * kv_params.head_size
    )
    comptime element_size = size_of[Float32]()
    assert_true(
        measured == packed * element_size,
        String("packed page stride should be ")
        + String(packed * element_size)
        + " bytes, got "
        + String(measured),
    )


def test_page_stride_overrides_the_packed_distance() raises:
    """A padded page is reached at the supplied stride, not the packed one."""
    comptime page_size = 16
    comptime num_layers = 3
    comptime packed = (
        2 * num_layers * page_size * kv_params.num_heads * kv_params.head_size
    )
    # A padded page: the pad is a whole number of rows, which the allocator
    # so that the row division in `_stride()` stays exact.
    comptime row = kv_params.num_heads * kv_params.head_size
    comptime padded = packed + 16 * row
    comptime assert padded % row == 0, "a padded page must be whole rows"

    var measured = _build_collection_and_page_stride[page_size, num_layers](
        padded
    )
    comptime element_size = size_of[Float32]()
    assert_true(
        measured == padded * element_size,
        String("padded page stride should be ")
        + String(padded * element_size)
        + " bytes, got "
        + String(measured),
    )


def main() raises:
    test_page_stride_of_minus_one_keeps_pages_packed()
    test_page_stride_overrides_the_packed_distance()
    test_paged_kv_cache_stride_is_unknown()
    test_paged_kv_cache_offset_correctness()
    test_paged_kv_cache_quantization()
    test_scales_resolve_through_their_own_lookup_table()
    do_test[16, 16]()
    do_test[64, 16]()
    do_test[128, 64]()
    do_test[128, 64, DType.float8_e4m3fn]()
