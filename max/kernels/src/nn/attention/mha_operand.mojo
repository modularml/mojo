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
"""KV-cache operand descriptors and TMA tile builders for MHA GPU kernels.

Provides the types and helper functions that describe how Q, K, and V
tensors (including paged KV caches) are laid out in GPU memory and how
TMA (tensor memory accelerator) descriptors are constructed for them.
"""

from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapL2Promotion, TensorMapSwizzle
from kv_cache.types import (
    KVCacheT,
    PagedRowIndices,
    _populate_via_row_idx,
    create_flat_scale_tma_tile,
    flat_scale_window,
    kv_num_sub_tiles,
    kv_sub_tile_rows,
    kv_tma_fold_chunks,
    padded_depth,
    swizzle_granularity,
)
from layout import (
    Layout,
    LayoutTensor,
    DefaultEngine,
    TensorEngine,
    UNKNOWN_VALUE,
)
from layout.tile_layout import (
    Layout as MixedLayout,
    RowMajorLayout,
    TensorLayout,
)
from layout.tma_async import (
    SplitLastDimTMATensorTile,
    SharedMemBarrier,
    _gather4_box_width,
    create_split_tma,
    TMATensorTile,
    create_tensor_tile,
    create_tma_tile_gather4,
)
from layout.tile_tensor import TileTensor
from layout.tile_layout import row_major
from layout.coord import Idx, Coord
from std.math import ceildiv

from std.utils import Index, IndexList

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder


trait MHAOperand(DevicePassable, TrivialRegisterPassable):
    """This serves as the trait to support arguments to our MHA kernel."""

    comptime dtype: DType
    comptime scale_dtype: DType
    comptime page_size: Int
    comptime quantization_enabled: Bool = False
    comptime quantization_granularity: Int

    # TODO: change this to return a LayoutTensor once MOCO-1471 is fixed
    @always_inline
    def block_paged_ptr[
        tile_size: Int,
    ](
        self,
        batch_idx: UInt32,
        start_tok_idx: UInt32,
        head_idx: UInt32,
        head_dim_idx: UInt32 = 0,
    ) -> UnsafePointer[Scalar[Self.dtype], ImmutAnyOrigin]:
        ...

    @always_inline
    def block_paged_tile[
        layout_t: TensorLayout,
        //,
        tile_size: Int,
    ](
        self,
        batch_idx: UInt32,
        start_tok_idx: UInt32,
        head_idx: UInt32,
        layout_val: layout_t,
        head_dim_idx: UInt32 = 0,
    ) -> TileTensor[Self.dtype, layout_t, ImmutAnyOrigin]:
        """Wraps block_paged_ptr in a TileTensor with the caller's layout.

        Parameters:
            layout_t: The `TensorLayout` of the returned `TileTensor`
                (inferred).
            tile_size: Tile size in rows used to compute the paged block
                pointer.

        Args:
            batch_idx: Batch index of the request.
            start_tok_idx: Starting token index within the batch.
            head_idx: KV head index.
            layout_val: Concrete `layout_t` instance describing the tile
                layout.
            head_dim_idx: Index along the head dimension (defaults to 0).
        """
        return TileTensor[Self.dtype, layout_t, ImmutAnyOrigin](
            ptr=self.block_paged_ptr[tile_size](
                batch_idx, start_tok_idx, head_idx, head_dim_idx
            ),
            layout=layout_val,
        )

    @always_inline
    def scales_block_paged_ptr(
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], ImmutAnyOrigin]:
        ...

    @always_inline
    def load_scale[
        width: Int
    ](
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[Self.scale_dtype, width]:
        ...

    @always_inline
    def cache_length(self, batch_idx: Int) -> Int:
        """Returns the length of the cache for a given batch index."""
        ...

    @always_inline
    def max_context_length(self) -> UInt32:
        """Returns the maximum cache length in a given batch index."""
        ...

    @always_inline
    def num_kv_rows(self) -> Int:
        """Returns the total number of virtual rows in the KV memory view.

        For paged caches this accounts for the paging stride so that TMA
        descriptors can be sized to cover the entire address space.
        """
        ...

    @always_inline
    def row_idx(self, batch_idx: UInt32, start_tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        ...

    @always_inline
    def populate[
        BN: Int,
        base_alignment: Int,
        pair_cta: Bool = False,
        is_leader: Bool = True,
    ](self, batch_idx: UInt32, base_kv_row: UInt32) -> PagedRowIndices[
        BN, Self.page_size, pair_cta, is_leader
    ]:
        """Populate a full `PagedRowIndices[BN, ...]` for a BN-row tile.

        Returns the precomputed physical row indices for the `num_pages`
        sub-tile pages covering the BN-row range starting at
        `base_kv_row` for `batch_idx`. Both K's TMA (which may cover only
        a subset in `pair_cta` mode) and V's TMA (which covers the full
        range) can then consume the result without any lazy LUT lookup.

        `base_alignment` is a comptime promise that
        `base_kv_row % base_alignment == 0` at runtime: typically
        `mask.start_column_alignment[...]()`. The `PagedKVCache`
        override uses it to pick the largest legal SIMD chunk for its
        LUT vector load and to skip the intra-page divmod when
        `base_alignment % page_size == 0`.

        Default implementation: scalar loop over `num_pages` calls to
        `row_idx`. Overrides (e.g. `PagedKVCache`) replace this with a
        single SIMD load from the underlying lookup table.
        """

        def _row(batch_idx: UInt32, start_tok_idx: UInt32) {imm} -> UInt32:
            return self.row_idx(batch_idx, start_tok_idx)

        return _populate_via_row_idx[BN, Self.page_size, pair_cta, is_leader](
            batch_idx, base_kv_row, _row
        )

    @always_inline
    def get_tma_row(self, encoded_index: Int32) -> Int32:
        """Convert an encoded sparse index to a physical TMA row.

        For paged caches the encoded index is
        ``physical_block * page_size + offset`` and this method returns
        ``physical_block * stride + offset``. Non-paged operands return
        the encoded index unchanged.
        """
        ...

    @always_inline
    def create_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        depth: Int,
        BK: Int = padded_depth[Self.dtype, swizzle_mode, depth](),
        fold_chunks: Int = 1,
        row_major: Bool = False,
    ](self, ctx: DeviceContext) raises -> SplitLastDimTMATensorTile[
        Self.dtype,
        IndexList[3](BN, 1, BK),
        swizzle_mode,
    ]:
        """Creates a TMA tile for efficient GPU memory transfers.
        This is useful for `k-major` MMA operations where we don't
        need to mask any extra rows.

        When `fold_chunks >= 2` the contiguous depth chunks are folded into one
        rank-4 TMA descriptor (SM100 K-only optimization). The caller must pass the
        value from `kv_tma_fold_chunks` and use the same value at the `tma_copy_k`
        issue site. `1` (default) is the original per-chunk behavior.

        When `row_major` is `True` (and `fold_chunks >= 2`) the fold uses the
        rank-5 chunk-inner (page-dense) box instead of the rank-4 chunk-outer
        box, so a tile can span multiple pages with one TMA per page. The same
        value must be used at the `tma_copy_k`/`tma_copy_v` issue site and the
        P@V MMA consumer descriptor."""
        ...

    @always_inline
    def create_scale_tma_tile[
        BMN: Int
    ](self, ctx: DeviceContext) raises -> TMATensorTile[
        Self.scale_dtype,
        2,
        Index(1, BMN),
        Index(1, BMN),
    ]:
        """Creates a TMA tile for efficient GPU memory transfers.
        This is useful for `m-major` MMA operations where we don't
        need to mask any extra rows."""
        ...

    @always_inline
    def create_index_scale_tma_tile[
        TILE: Int
    ](self, ctx: DeviceContext) raises -> TMATensorTile[
        Self.dtype,
        2,
        Index(1, flat_scale_window[Self.dtype, TILE]()),
        Index(1, flat_scale_window[Self.dtype, TILE]()),
    ]:
        """Creates a flat TMA tile over the scale pool this operand addresses.

        Only meaningful for an operand that IS a scale pool, so the default
        rejects; `block_paged_ptr` must already address scales. The box is
        `TILE` scales plus one alignment unit of slack -- a TMA's global start
        must be 16-byte aligned and a key index is the innermost coordinate
        here, so an unaligned caller rounds its base down and skips the
        residual.
        """
        comptime assert False, (
            "create_index_scale_tma_tile is only meaningful for an operand"
            " that ADDRESSES scales"
        )

    @always_inline
    def create_rope_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int,
        padded_depth: Int,
    ](self, ctx: DeviceContext) raises -> SplitLastDimTMATensorTile[
        DType.bfloat16,
        IndexList[3](BN, 1, BK),
        swizzle_mode,
    ]:
        """Creates a BF16 TMA tile for the rope portion of the per-tensor rope-aware KV cache.
        """
        ...

    @always_inline
    def create_gather4_tma_tile[
        tile_width: Int,
        tile_stride: Int = tile_width,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tile_height: Int = 4,
        tma_dtype: DType = Self.dtype,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](self, ctx: DeviceContext) raises -> TMATensorTile[
        tma_dtype,
        2,
        tile_shape=IndexList[2](
            tile_height,
            _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
        ),
    ]:
        """Creates a 2D TMA gather4 descriptor for this operand.

        The descriptor views the data as a flat 2D matrix of
        ``[num_kv_rows, tile_width]`` and is configured for gather4 operations
        that load 4 non-contiguous rows per TMA instruction. The box width
        is derived from the swizzle mode; for SWIZZLE_NONE it equals
        ``tile_width``.

        When ``tma_dtype`` differs from ``Self.dtype``, the underlying data
        pointer is bitcast to ``tma_dtype`` at descriptor creation time.

        Parameters:
            tile_width: Number of elements per row to load (box width) in
                ``tma_dtype`` elements.
            tile_stride: Row stride in elements in global memory. Defaults to
                ``tile_width``. Use a larger value when the global row is wider
                than the portion to load.
            swizzle_mode: TMA swizzle mode for shared memory access pattern.
                Defaults to SWIZZLE_NONE.
            tile_height: Number of rows in the tile. Must be a multiple of 4.
                Defaults to 4 for backward compatibility.
            tma_dtype: The data type used for the TMA descriptor. Defaults to
                ``Self.dtype``. When different, the pointer is bitcast.
            l2_promotion: L2 cache promotion hint for TMA loads. Defaults to
                NONE.

        Args:
            ctx: The CUDA device context used to create the TMA descriptor.

        Returns:
            A TMATensorTile with box width derived from the swizzle mode.
        """
        ...

    @always_inline
    def create_rope_gather4_tma_tile[
        tile_width: Int,
        padded_depth: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tile_height: Int = 4,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](self, ctx: DeviceContext) raises -> TMATensorTile[
        DType.bfloat16,
        2,
        tile_shape=IndexList[2](
            tile_height,
            _gather4_box_width[.bfloat16, tile_width, swizzle_mode](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[.bfloat16, tile_width, swizzle_mode](),
        ),
    ]:
        """Creates a BF16 gather4 TMA descriptor for the rope portion of the
        KV cache.

        For the per-tensor rope-aware layout each token row is
        ``padded_depth`` FP8 bytes (content) followed by BF16 rope elements.
        This method offsets the base pointer by ``padded_depth`` bytes,
        reinterprets as BF16, and creates a gather4 TMA descriptor.

        Parameters:
            tile_width: Number of BF16 elements per row in global memory.
            padded_depth: Byte offset from row start to the rope data.
            swizzle_mode: TMA swizzle mode for shared memory access pattern.
            tile_height: Number of rows in the tile. Must be a multiple of 4.
            l2_promotion: L2 cache promotion hint for TMA loads. Defaults to
                NONE.

        Args:
            ctx: The CUDA device context used to create the TMA descriptor.

        Returns:
            A BF16 TMATensorTile configured for gather4.
        """
        ...

    @always_inline
    def scales_raw_ptr(
        self,
    ) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Returns the base pointer to the quantization scales tensor.

        Returns a null pointer for operands without quantization support.
        """
        ...


struct KVCacheMHAOperand[
    cache_t: KVCacheT,
](MHAOperand, TrivialRegisterPassable):
    """An implementation for `mo.opaque` KVCacheT arguments to MHA kernels.

    We can eventually remove this trait and just add it as a sub-trait in the
    KVCacheT type, but we need to solve some cyclic dependencies first.

    Parameters:
        cache_t: The concrete `KVCacheT` type backing this operand and
            providing the paged KV cache storage and lookup tables.
    """

    comptime dtype = Self.cache_t.dtype
    comptime scale_dtype = Self.cache_t.scale_dtype
    comptime page_size = Self.cache_t.page_size_
    comptime quantization_enabled = Self.cache_t.quantization_enabled
    comptime quantization_granularity = Self.cache_t.quantization_granularity
    var cache: Self.cache_t

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "KVCacheMHAOperand"

    def __init__(out self, cache: Self.cache_t):
        self.cache = cache

    @always_inline
    def block_paged_ptr[
        tile_size: Int
    ](
        self,
        batch_idx: UInt32,
        start_tok_idx: UInt32,
        head_idx: UInt32,
        head_dim_idx: UInt32 = 0,
    ) -> UnsafePointer[Scalar[Self.dtype], ImmutAnyOrigin]:
        return self.cache.block_paged_ptr[tile_size](
            Int(batch_idx), Int(start_tok_idx), Int(head_idx), Int(head_dim_idx)
        )

    @always_inline
    def scales_block_paged_ptr(
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], ImmutAnyOrigin]:
        return self.cache.scales_block_paged_ptr(
            batch_idx, start_tok_idx, head_idx, head_dim_idx
        )

    @always_inline
    def load_scale[
        width: Int
    ](
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[Self.scale_dtype, width]:
        return self.cache.load_scale[width=width](
            batch_idx, head_idx, start_tok_idx, head_dim_idx
        )

    @always_inline
    def cache_length(self, batch_idx: Int) -> Int:
        return self.cache.cache_length(batch_idx)

    @always_inline
    def max_context_length(self) -> UInt32:
        return self.cache.max_context_length()

    @always_inline
    def num_kv_rows(self) -> Int:
        """Returns the total number of virtual rows in the KV memory view."""
        return self.cache.num_kv_rows()

    @always_inline
    def row_idx(self, batch_idx: UInt32, start_tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix.

        Args:
            batch_idx: Batch index of the request.
            start_tok_idx: Starting token index within the batch.
        """
        return self.cache.row_idx(batch_idx, start_tok_idx)

    @always_inline
    def populate[
        BN: Int,
        base_alignment: Int,
        pair_cta: Bool = False,
        is_leader: Bool = True,
    ](self, batch_idx: UInt32, base_kv_row: UInt32) -> PagedRowIndices[
        BN, Self.page_size, pair_cta, is_leader
    ]:
        """Delegate to the underlying cache's `populate`.

        `PagedKVCache.populate` overrides with a SIMD lookup-table read;
        other cache types fall through to the scalar default on
        `KVCacheT.populate`. `base_alignment` is the comptime alignment
        of `base_kv_row` (typically `mask.start_column_alignment[...]()`).

        Parameters:
            BN: Number of rows in the tile to populate.
            base_alignment: Comptime promise that `base_kv_row %
                base_alignment == 0` at runtime.
            pair_cta: Whether K and V run as a paired CTA pair where K
                covers only a subset of the range (defaults to `False`).
            is_leader: Whether this CTA is the leader that computes the
                row indices (defaults to `True`).

        Args:
            batch_idx: Batch index of the request.
            base_kv_row: Starting KV row index of the `BN`-row tile.
        """
        return self.cache.populate[BN, base_alignment, pair_cta, is_leader](
            batch_idx, base_kv_row
        )

    @always_inline
    def get_tma_row(self, encoded_index: Int32) -> Int32:
        """Convert an encoded sparse index to a physical TMA row.

        Args:
            encoded_index: Encoded sparse row index. For paged caches this
                is `physical_block * page_size + offset`.
        """
        return self.cache.get_tma_row(encoded_index)

    @always_inline
    def create_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        depth: Int,
        BK: Int = padded_depth[Self.dtype, swizzle_mode, depth](),
        fold_chunks: Int = 1,
        row_major: Bool = False,
    ](
        self,
        ctx: DeviceContext,
        out tma: SplitLastDimTMATensorTile[
            Self.dtype,
            IndexList[3](BN, 1, BK),
            swizzle_mode,
        ],
    ) raises:
        """Creates a TMA tile for efficient GPU memory transfers.

        Parameters:
            swizzle_mode: TMA swizzle mode for shared memory access pattern.
            BN: Number of rows loaded per TMA instruction.
            depth: Full depth of the K/V dimension in global memory, in
                elements.
            BK: Block size along the depth dimension in elements (defaults
                to `padded_depth[Self.dtype, swizzle_mode, depth]()`).
            fold_chunks: Number of contiguous depth chunks folded into one
                TMA descriptor (defaults to 1).
            row_major: When `True` with `fold_chunks >= 2`, use the rank-5
                chunk-inner box so a tile can span multiple pages (defaults
                to `False`).

        Args:
            ctx: The CUDA device context used to create the TMA descriptor.
        """
        # Forward to the underlying cache's implementation
        # TODO: remove `comptime assert` when the `where` clause is enough
        comptime assert (
            BK % swizzle_granularity[Self.dtype, swizzle_mode]()
        ) == 0
        tma = rebind[type_of(tma)](
            self.cache.create_tma_tile[
                swizzle_mode,
                BN=BN,
                BK=BK,
                fold_chunks=fold_chunks,
                row_major=row_major,
            ](ctx)
        )

    @always_inline
    def create_scale_tma_tile[
        BMN: Int
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            Self.scale_dtype,
            2,
            Index(1, BMN),
            Index(1, BMN),
        ],
    ) raises:
        """Creates a TMA tile for efficient GPU memory transfers.
        This is useful for `m-major` MMA operations where we don't
        need to mask any extra rows.

        Parameters:
            BMN: Number of scale elements loaded per TMA tile row.

        Args:
            ctx: The CUDA device context used to create the TMA descriptor.
        """
        comptime assert False, "create_scale_tma_tile is not implemented"

    @always_inline
    def create_rope_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int,
        padded_depth: Int,
    ](
        self,
        ctx: DeviceContext,
        out tma: SplitLastDimTMATensorTile[
            DType.bfloat16,
            IndexList[3](BN, 1, BK),
            swizzle_mode,
        ],
    ) raises:
        """Delegates to the underlying KVCache to create a BF16 rope TMA tile.

        Parameters:
            swizzle_mode: TMA swizzle mode for shared memory access pattern.
            BN: Number of rows loaded per TMA instruction.
            BK: Block size along the depth dimension in BF16 elements.
            padded_depth: Byte offset from row start to the BF16 rope data,
                skipping the FP8 content prefix.

        Args:
            ctx: The CUDA device context used to create the TMA descriptor.
        """
        tma = self.cache.create_rope_tma_tile[
            swizzle_mode, BN=BN, BK=BK, padded_depth=padded_depth
        ](ctx)

    @always_inline
    def create_gather4_tma_tile[
        tile_width: Int,
        tile_stride: Int = tile_width,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tile_height: Int = 4,
        tma_dtype: DType = Self.dtype,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            tma_dtype,
            2,
            tile_shape=IndexList[2](
                tile_height,
                _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
            ),
            desc_shape=IndexList[2](
                1,
                _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
            ),
        ],
    ) raises:
        """Creates a 2D TMA gather4 descriptor for this KV cache operand.

        Parameters:
            tile_width: Number of elements per row to load (box width) in
                `tma_dtype` elements.
            tile_stride: Row stride in elements in global memory. Defaults to
                `tile_width`. Use a larger value when the global row is wider
                than the portion to load.
            swizzle_mode: TMA swizzle mode for shared memory access pattern.
                Defaults to `SWIZZLE_NONE`.
            tile_height: Number of rows in the tile. Must be a multiple of 4.
                Defaults to 4 for backward compatibility.
            tma_dtype: Data type used for the TMA descriptor. Defaults to
                `Self.dtype`. When different, the pointer is bitcast.
            l2_promotion: L2 cache promotion hint for TMA loads. Defaults to
                `NONE`.

        Args:
            ctx: The CUDA device context used to create the TMA descriptor.
        """
        tma = rebind[type_of(tma)](
            self.cache.create_gather4_tma_tile[
                tile_height=tile_height,
                tile_width=tile_width,
                tile_stride=tile_stride,
                swizzle_mode=swizzle_mode,
                tma_dtype=tma_dtype,
                l2_promotion=l2_promotion,
            ](ctx)
        )

    @always_inline
    def create_rope_gather4_tma_tile[
        tile_width: Int,
        padded_depth: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tile_height: Int = 4,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            DType.bfloat16,
            2,
            tile_shape=IndexList[2](
                tile_height,
                _gather4_box_width[.bfloat16, tile_width, swizzle_mode](),
            ),
            desc_shape=IndexList[2](
                1,
                _gather4_box_width[.bfloat16, tile_width, swizzle_mode](),
            ),
        ],
    ) raises:
        """Delegates to the underlying KVCache to create a BF16 rope gather4
        TMA tile.

        For the per-tensor rope-aware layout each token row is `padded_depth`
        FP8 bytes (content) followed by BF16 rope elements.

        Parameters:
            tile_width: Number of BF16 elements per row in global memory.
            padded_depth: Byte offset from row start to the BF16 rope data,
                skipping the FP8 content prefix.
            swizzle_mode: TMA swizzle mode for shared memory access pattern.
                Defaults to `SWIZZLE_NONE`.
            tile_height: Number of rows in the tile. Must be a multiple of 4.
                Defaults to 4.
            l2_promotion: L2 cache promotion hint for TMA loads. Defaults to
                `NONE`.

        Args:
            ctx: The CUDA device context used to create the TMA descriptor.
        """
        tma = rebind[type_of(tma)](
            self.cache.create_rope_gather4_tma_tile[
                tile_height=tile_height,
                tile_width=tile_width,
                padded_depth=padded_depth,
                swizzle_mode=swizzle_mode,
                l2_promotion=l2_promotion,
            ](ctx)
        )

    @always_inline
    def scales_raw_ptr(
        self,
    ) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Returns the base pointer to the quantization scales tensor."""
        return rebind[UnsafePointer[Float32, MutAnyOrigin]](
            self.cache.scales_raw_ptr()
        )


struct KVCacheScalesMHAOperand[
    cache_t: KVCacheT,
](MHAOperand, TrivialRegisterPassable):
    """An MHAOperand that accesses the scales field of a KVCache.

    This is useful for MLA attention where k_s (per-token scales) are stored
    in the scales field of the k cache with quantization_granularity = head_size.
    The scales have shape [num_blocks, page_size, num_heads, head_dim_granularity].

    Parameters:
        cache_t: The concrete `KVCacheT` type whose scales field this operand
            accesses.
    """

    comptime dtype = Self.cache_t.scale_dtype
    comptime scale_dtype = Self.dtype
    comptime page_size = Self.cache_t.page_size_
    comptime quantization_enabled = Self.cache_t.quantization_enabled
    comptime quantization_granularity = Self.cache_t.quantization_granularity
    var cache: Self.cache_t

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "KVCacheScalesMHAOperand"

    def __init__(out self, cache: Self.cache_t):
        self.cache = cache

    @always_inline
    def block_paged_ptr[
        tile_size: Int
    ](
        self,
        batch_idx: UInt32,
        start_tok_idx: UInt32,
        head_idx: UInt32,
        head_dim_idx: UInt32 = 0,
    ) -> UnsafePointer[Scalar[Self.dtype], ImmutAnyOrigin]:
        # Forward to scales_block_paged_ptr instead of block_paged_ptr
        return self.cache.scales_block_paged_ptr(
            Int(batch_idx), Int(start_tok_idx), Int(head_idx), Int(head_dim_idx)
        )

    @always_inline
    def scales_block_paged_ptr(
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], ImmutAnyOrigin]:
        # SAFETY: KVCacheScalesMHAOperand doesn't support block-paged scales;
        # callers only dereference behind comptime quantization guards.
        return UnsafePointer[
            Scalar[Self.scale_dtype], ImmutAnyOrigin
        ].unsafe_dangling()

    @always_inline
    def load_scale[
        width: Int
    ](
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[Self.scale_dtype, width]:
        return SIMD[Self.scale_dtype, width](0)

    @always_inline
    def cache_length(self, batch_idx: Int) -> Int:
        return self.cache.cache_length(batch_idx)

    @always_inline
    def max_context_length(self) -> UInt32:
        return self.cache.max_context_length()

    @always_inline
    def num_kv_rows(self) -> Int:
        """Returns the total number of virtual rows in the KV memory view."""
        return self.cache.num_kv_rows()

    @always_inline
    def row_idx(self, batch_idx: UInt32, start_tok_idx: UInt32) -> UInt32:
        """Returns the row idx in the SCALE pool -- what this operand addresses.

        The scale pool has its own lookup table and its own block stride, which
        coincide with the value pool's only when the caller leaves the scales
        LUT unset. `block_paged_ptr` already forwards to the scales, so this
        must too.
        """
        return self.cache.scale_row_idx(batch_idx, start_tok_idx)

    @always_inline
    def get_tma_row(self, encoded_index: Int32) -> Int32:
        """Convert an encoded sparse index to a physical TMA row."""
        return self.cache.get_tma_row(encoded_index)

    @always_inline
    def create_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        depth: Int,
        BK: Int = padded_depth[Self.dtype, swizzle_mode, depth](),
        fold_chunks: Int = 1,
        row_major: Bool = False,
    ](
        self,
        ctx: DeviceContext,
        out tma: SplitLastDimTMATensorTile[
            Self.dtype,
            IndexList[3](BN, 1, BK),
            swizzle_mode,
        ],
    ) raises:
        """TMA not supported for KVCacheScalesMHAOperand."""
        comptime assert False, "TMA not supported for KVCacheScalesMHAOperand"

    @always_inline
    def create_scale_tma_tile[
        BMN: Int
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            Self.scale_dtype,
            2,
            Index(1, BMN),
            Index(1, BMN),
        ],
    ) raises:
        comptime assert False, "create_scale_tma_tile is not implemented"

    @always_inline
    def create_index_scale_tma_tile[
        TILE: Int
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            Self.dtype,
            2,
            Index(1, flat_scale_window[Self.dtype, TILE]()),
            Index(1, flat_scale_window[Self.dtype, TILE]()),
        ],
    ) raises:
        """A flat window on the cache's scale pool.

        This operand IS the scales, so unlike `create_scale_tma_tile` -- which
        describes a companion buffer and stays unimplemented here -- this one
        is the operand's whole point.
        """
        return self.cache.create_index_scale_tma_tile[TILE](ctx)

    @always_inline
    def create_rope_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int,
        padded_depth: Int,
    ](
        self,
        ctx: DeviceContext,
        out tma: SplitLastDimTMATensorTile[
            DType.bfloat16,
            IndexList[3](BN, 1, BK),
            swizzle_mode,
        ],
    ) raises:
        """Not supported for KVCacheScalesMHAOperand."""
        comptime assert (
            False
        ), "create_rope_tma_tile is not supported for KVCacheScalesMHAOperand"

    @always_inline
    def create_gather4_tma_tile[
        tile_width: Int,
        tile_stride: Int = tile_width,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tile_height: Int = 4,
        tma_dtype: DType = Self.dtype,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            tma_dtype,
            2,
            tile_shape=IndexList[2](
                tile_height,
                _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
            ),
            desc_shape=IndexList[2](
                1,
                _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
            ),
        ],
    ) raises:
        """Not supported for KVCacheScalesMHAOperand.

        Parameters:
            tile_width: Number of elements per row to load (box width) in
                `tma_dtype` elements.
            tile_stride: Row stride in elements in global memory. Defaults
                to `tile_width`.
            swizzle_mode: TMA swizzle mode for shared memory access pattern.
                Defaults to `SWIZZLE_NONE`.
            tile_height: Number of rows in the tile. Must be a multiple of 4.
                Defaults to 4.
            tma_dtype: Data type used for the TMA descriptor. Defaults to
                `Self.dtype`.
            l2_promotion: L2 cache promotion hint for TMA loads. Defaults to
                `NONE`.

        Args:
            ctx: The CUDA device context used to create the TMA descriptor.
        """
        comptime assert False, (
            "create_gather4_tma_tile is not supported for"
            " KVCacheScalesMHAOperand"
        )

    @always_inline
    def create_rope_gather4_tma_tile[
        tile_width: Int,
        padded_depth: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tile_height: Int = 4,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            DType.bfloat16,
            2,
            tile_shape=IndexList[2](
                tile_height,
                _gather4_box_width[.bfloat16, tile_width, swizzle_mode](),
            ),
            desc_shape=IndexList[2](
                1,
                _gather4_box_width[.bfloat16, tile_width, swizzle_mode](),
            ),
        ],
    ) raises:
        """Not supported for KVCacheScalesMHAOperand."""
        comptime assert False, (
            "create_rope_gather4_tma_tile is not supported for"
            " KVCacheScalesMHAOperand"
        )

    @always_inline
    def scales_raw_ptr(
        self,
    ) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Returns a dangling pointer. KVCacheScalesMHAOperand already points to the
        scales pointer."""
        # SAFETY: Callers access scales through the operand's own pointer, not
        # this raw_ptr; only used behind comptime quantization guards.
        return UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()


@always_inline
def _null_scale_tile_tensor[
    scale_dtype: DType,
    scale_layout: TensorLayout,
]() -> TileTensor[scale_dtype, scale_layout, ImmutAnyOrigin]:
    """Builds a null (dangling) scale `TileTensor` for the no-scale path.

    `scale_layout` is trait-typed, so re-materialize the concrete `Layout` to
    construct the value, then implicitly convert to the trait-typed slot. Used
    as the default `scale_buffer` argument so `scale_origin` binds to
    `ImmutAnyOrigin` (matching the legacy default-arg behavior).
    """
    comptime NullScaleLayout = MixedLayout[
        shape_types=scale_layout._shape_types,
        stride_types=scale_layout._stride_types,
    ]
    return rebind[TileTensor[scale_dtype, scale_layout, ImmutAnyOrigin]](
        TileTensor[scale_dtype, NullScaleLayout, ImmutAnyOrigin](
            ptr=UnsafePointer[
                Scalar[scale_dtype], ImmutAnyOrigin
            ].unsafe_dangling(),
            layout=NullScaleLayout(),
        )
    )


struct LayoutTensorMHAOperand[
    origin: ImmOrigin,
    scale_origin: ImmOrigin,
    //,
    dtype_: DType,
    buffer_layout: TensorLayout,
    scale_dtype_: DType = .float32,
    scale_buffer_layout: TensorLayout = MixedLayout[
        shape_types=Coord[].element_types,
        stride_types=Coord[].element_types,
    ],
    buffer_engine: TensorEngine = DefaultEngine[element_width=1],
    scale_buffer_engine: TensorEngine = DefaultEngine[element_width=1],
](MHAOperand, TrivialRegisterPassable):
    """An implementation for contiguous tensor arguments to MHA kernels.

    Parameters:
        origin: Origin of the K/V `buffer` `TileTensor` (inferred).
        scale_origin: Origin of the `scale_buffer` `TileTensor`
            (inferred).
        dtype_: Element type of the K/V `buffer`.
        buffer_layout: `TensorLayout` of the K/V `buffer`.
        scale_dtype_: Element type of the `scale_buffer` (defaults to
            `DType.float32`).
        scale_buffer_layout: `TensorLayout` of the `scale_buffer`. A
            rank-0 layout disables quantization (defaults to a rank-0
            `MixedLayout`).
        buffer_engine: `TensorEngine` of the K/V `buffer` (defaults to
            `DefaultEngine`).
        scale_buffer_engine: `TensorEngine` of the `scale_buffer`
            (defaults to `DefaultEngine`).
    """

    comptime dtype = Self.dtype_
    comptime scale_dtype = Self.scale_dtype_
    comptime page_size = 0
    comptime layout_rank = Self.buffer_layout.rank
    comptime scale_rank = Self.scale_buffer_layout.rank
    comptime layout_dim: Int = Self.buffer_layout._shape_types[
        Self.layout_rank - 1
    ].static_value
    # `scale_dim` is only meaningful (and only read) when quantization is
    # enabled. For the no-scale path (`scale_rank == 0`) fall back to 1 so the
    # `ceildiv` below stays well-formed without indexing an empty shape list.
    comptime scale_dim: Int = Self.scale_buffer_layout._shape_types[
        Self.scale_rank - 1
    ].static_value if Self.scale_rank != 0 else 1
    comptime quantization_granularity: Int = ceildiv(
        Self.layout_dim, Self.scale_dim
    )
    comptime quantization_enabled: Bool = Self.scale_buffer_layout.rank != 0

    var buffer: TileTensor[
        Self.dtype, Self.buffer_layout, Self.origin, Engine=Self.buffer_engine
    ]
    var scale_buffer: TileTensor[
        Self.scale_dtype,
        Self.scale_buffer_layout,
        Self.scale_origin,
        Engine=Self.scale_buffer_engine,
    ]
    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "LayoutTensorMHAOperand"

    def __init__(
        out self,
        buffer: TileTensor[
            Self.dtype,
            Self.buffer_layout,
            Self.origin,
            Engine=Self.buffer_engine,
        ],
        scale_buffer: TileTensor[
            Self.scale_dtype,
            Self.scale_buffer_layout,
            Self.scale_origin,
            Engine=Self.scale_buffer_engine,
        ] = _null_scale_tile_tensor[
            Self.scale_dtype, Self.scale_buffer_layout
        ](),
    ):
        self.buffer = buffer
        self.scale_buffer = scale_buffer

    @always_inline
    def block_paged_ptr[
        tile_size: Int
    ](
        self,
        batch_idx: UInt32,
        start_tok_idx: UInt32,
        head_idx: UInt32,
        head_dim_idx: UInt32 = 0,
    ) -> UnsafePointer[Scalar[Self.dtype], ImmutAnyOrigin]:
        var ret_ptr = self.buffer.ptr + self.buffer.layout[
            linear_idx_type=type_of(self.buffer).linear_idx_type
        ](
            Coord(
                Int(batch_idx),
                Int(start_tok_idx),
                Int(head_idx),
                Int(head_dim_idx),
            )
        )
        return ret_ptr.as_imm().as_unsafe_any_origin()

    @always_inline
    def scales_block_paged_ptr(
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], ImmutAnyOrigin]:
        var ret_ptr = self.scale_buffer.ptr + self.scale_buffer.layout[
            linear_idx_type=type_of(self.scale_buffer).linear_idx_type
        ](
            Coord(
                Int(batch_idx),
                Int(start_tok_idx),
                Int(head_idx),
                Int(head_dim_idx // Self.quantization_granularity),
            )
        )
        return ret_ptr.as_imm().as_unsafe_any_origin()

    @always_inline
    def load_scale[
        width: Int
    ](
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[Self.scale_dtype, width]:
        return self.scale_buffer.load[width=width](
            Coord(
                Int(batch_idx),
                Int(start_tok_idx),
                Int(head_idx),
                Int(head_dim_idx // Self.quantization_granularity),
            )
        )

    @always_inline
    def cache_length(self, batch_idx: Int) -> Int:
        # Contiguous tensor path assumes BSHD layout and all cache entries have
        # the same length.
        return Int(self.buffer.dim[1]())

    @always_inline
    def max_context_length(self) -> UInt32:
        return UInt32(Int(self.buffer.dim[1]()))

    @always_inline
    def num_kv_rows(self) -> Int:
        """Returns the total number of virtual rows (batch * seq_len)."""
        return Int(self.buffer.dim[0]()) * Int(self.buffer.dim[1]())

    @always_inline
    def row_idx(self, batch_idx: UInt32, start_tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        return batch_idx * UInt32(Int(self.buffer.dim[1]())) + start_tok_idx

    @always_inline
    def get_tma_row(self, encoded_index: Int32) -> Int32:
        """Convert an encoded sparse index to a physical TMA row.

        Non-paged operand: identity (no paging translation needed).
        """
        return encoded_index

    @always_inline
    def create_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        depth: Int,
        BK: Int = padded_depth[Self.dtype, swizzle_mode, depth](),
        fold_chunks: Int = 1,
        row_major: Bool = False,
    ](
        self,
        ctx: DeviceContext,
        out tma: SplitLastDimTMATensorTile[
            Self.dtype,
            IndexList[3](BN, 1, BK),
            swizzle_mode,
        ],
    ) raises:
        """Creates a TMA tile for efficient GPU memory transfers."""
        # View the 4D buffer as a 2D matrix [batch*seq, heads*head_dim]
        comptime assert (
            BK % swizzle_granularity[Self.dtype, swizzle_mode]()
        ) == 0, String("BN = ", BN, "\ndepth = ", depth, "\nBK = ", BK)
        var rows = Int(self.buffer.dim[0]()) * Int(self.buffer.dim[1]())
        comptime smem_shape = IndexList[3](BN, 1, BK)
        comptime gmem_shape = IndexList[3](UNKNOWN_VALUE, UNKNOWN_VALUE, depth)

        tma = create_split_tma[
            smem_shape,
            gmem_shape,
            swizzle_mode=swizzle_mode,
            fold_chunks=fold_chunks,
            row_major=row_major,
        ](
            ctx,
            self.buffer.ptr.as_imm().as_unsafe_any_origin(),
            rows,
            Int(self.buffer.dim[2]()),
        )

    @always_inline
    def create_scale_tma_tile[
        BMN: Int
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            Self.scale_dtype,
            2,
            Index(1, BMN),
            Index(1, BMN),
        ],
    ) raises:
        var total_elements = self.scale_buffer.num_elements()
        debug_assert(
            total_elements % 4 == 0,
            (
                "Total_elements must be divisible by 4. Otherwise, the Nvidia"
                " Driver will crash."
            ),
        )
        var scale_tensor = TileTensor(
            self.scale_buffer.ptr,
            row_major(Coord(Idx[1], total_elements)),
        )
        return create_tensor_tile[
            Index(1, BMN),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
            __desc_shape=Index(1, BMN),
        ](ctx, scale_tensor)

    @always_inline
    def create_rope_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int,
        padded_depth: Int,
    ](
        self,
        ctx: DeviceContext,
        out tma: SplitLastDimTMATensorTile[
            DType.bfloat16,
            IndexList[3](BN, 1, BK),
            swizzle_mode,
        ],
    ) raises:
        """Not supported for LayoutTensorMHAOperand.

        Parameters:
            swizzle_mode: TMA swizzle mode for shared memory access
                pattern.
            BN: Number of rows loaded per TMA instruction.
            BK: Block size along the depth dimension in BF16 elements.
            padded_depth: Byte offset from row start to the BF16 rope
                data, skipping the FP8 content prefix.

        Args:
            ctx: The CUDA device context used to create the TMA
                descriptor.
        """
        comptime assert (
            False
        ), "create_rope_tma_tile is not supported for LayoutTensorMHAOperand"

    @always_inline
    def create_gather4_tma_tile[
        tile_width: Int,
        tile_stride: Int = tile_width,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tile_height: Int = 4,
        tma_dtype: DType = Self.dtype,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            tma_dtype,
            2,
            tile_shape=IndexList[2](
                tile_height,
                _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
            ),
            desc_shape=IndexList[2](
                1,
                _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
            ),
        ],
    ) raises:
        """Creates a 2D TMA gather4 descriptor for this contiguous operand.

        Parameters:
            tile_width: Number of elements per row to load (box width)
                in `tma_dtype` elements.
            tile_stride: Row stride in elements in global memory.
                Defaults to `tile_width`. Use a larger value when the
                global row is wider than the portion to load.
            swizzle_mode: TMA swizzle mode for shared memory access
                pattern. Defaults to `SWIZZLE_NONE`.
            tile_height: Number of rows in the tile. Must be a multiple
                of 4. Defaults to 4 for backward compatibility.
            tma_dtype: Data type used for the TMA descriptor. Defaults
                to `Self.dtype`. When different, the pointer is bitcast.
            l2_promotion: L2 cache promotion hint for TMA loads.
                Defaults to `NONE`.

        Args:
            ctx: The CUDA device context used to create the TMA
                descriptor.
        """
        # View the 4D buffer as a 2D matrix [batch*seq, tile_width]
        var rows = Int(self.buffer.dim[0]()) * Int(self.buffer.dim[1]())
        tma = create_tma_tile_gather4[
            tma_dtype,
            tile_height=tile_height,
            tile_width=tile_width,
            tile_stride=tile_stride,
            swizzle_mode=swizzle_mode,
            l2_promotion=l2_promotion,
        ](ctx, self.buffer.ptr.bitcast[Scalar[tma_dtype]](), rows)

    @always_inline
    def create_rope_gather4_tma_tile[
        tile_width: Int,
        padded_depth: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tile_height: Int = 4,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            DType.bfloat16,
            2,
            tile_shape=IndexList[2](
                tile_height,
                _gather4_box_width[.bfloat16, tile_width, swizzle_mode](),
            ),
            desc_shape=IndexList[2](
                1,
                _gather4_box_width[.bfloat16, tile_width, swizzle_mode](),
            ),
        ],
    ) raises:
        """Not supported for LayoutTensorMHAOperand.

        Parameters:
            tile_width: Number of BF16 elements per row in global
                memory.
            padded_depth: Byte offset from row start to the BF16 rope
                data, skipping the FP8 content prefix.
            swizzle_mode: TMA swizzle mode for shared memory access
                pattern. Defaults to `SWIZZLE_NONE`.
            tile_height: Number of rows in the tile. Must be a multiple
                of 4. Defaults to 4.
            l2_promotion: L2 cache promotion hint for TMA loads.
                Defaults to `NONE`.

        Args:
            ctx: The CUDA device context used to create the TMA
                descriptor.
        """
        comptime assert False, (
            "create_rope_gather4_tma_tile is not supported for"
            " LayoutTensorMHAOperand"
        )

    @always_inline
    def scales_raw_ptr(
        self,
    ) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Returns a dangling pointer. Contiguous operands do not support
        quantization."""
        # SAFETY: LayoutTensor operands are never quantized; callers only
        # dereference behind comptime quantization guards.
        return UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()


struct RaggedMHAOperand[
    origin: ImmOrigin,
    cache_origin: ImmOrigin,
    //,
    dtype_: DType,
    layout: TensorLayout,
    cache_layout: TensorLayout,
    scale_dtype_: DType = dtype_,
    scale_layout: TensorLayout = RowMajorLayout[
        *Coord[Int64, Int64].element_types
    ],
](MHAOperand, TrivialRegisterPassable):
    """An implementation for ragged contiguous tensor arguments to MHA kernels.
    """

    comptime dtype = Self.dtype_
    comptime scale_dtype = Self.scale_dtype_
    comptime page_size = 0
    comptime quantization_granularity = 0
    var buffer: TileTensor[Self.dtype, Self.layout, Self.origin]

    @__allow_legacy_any_origin_fields
    var scale_buffer: TileTensor[
        Self.scale_dtype, Self.scale_layout, ImmutAnyOrigin
    ]
    var cache_row_offsets: TileTensor[
        .uint32, Self.cache_layout, Self.cache_origin
    ]

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "RaggedMHAOperand"

    def __init__(
        out self,
        buffer: TileTensor[Self.dtype, Self.layout, Self.origin],
        cache_row_offsets: TileTensor[
            .uint32, Self.cache_layout, Self.cache_origin
        ],
    ):
        comptime assert (
            buffer.rank == 3
        ), "only support rank 3 inputs for ragged inputs."
        comptime assert (
            cache_row_offsets.rank == 1
        ), "only support rank 1 inputs for cache offsets."
        self.buffer = buffer
        self.cache_row_offsets = cache_row_offsets
        # No-scale operand: build a null scale buffer. `scale_layout` is
        # trait-typed, so re-materialize the concrete `Layout` and rebind it
        # to the field's exact type.
        comptime NullScaleLayout = MixedLayout[
            shape_types=Self.scale_layout._shape_types,
            stride_types=Self.scale_layout._stride_types,
        ]
        self.scale_buffer = rebind[
            TileTensor[Self.scale_dtype, Self.scale_layout, ImmutAnyOrigin]
        ](
            TileTensor[Self.scale_dtype, NullScaleLayout, ImmutAnyOrigin](
                ptr=UnsafePointer[
                    Scalar[Self.scale_dtype], ImmutAnyOrigin
                ].unsafe_dangling(),
                layout=NullScaleLayout(),
            )
        )

    def __init__(
        out self,
        buffer: TileTensor[Self.dtype, Self.layout, Self.origin],
        scale_buffer: TileTensor[
            Self.scale_dtype, Self.scale_layout, ImmutAnyOrigin
        ],
        cache_row_offsets: TileTensor[
            .uint32, Self.cache_layout, Self.cache_origin
        ],
    ):
        self.buffer = buffer
        self.cache_row_offsets = cache_row_offsets
        self.scale_buffer = scale_buffer

    @always_inline
    def block_paged_ptr[
        tile_size: Int
    ](
        self,
        batch_idx: UInt32,
        start_tok_idx: UInt32,
        head_idx: UInt32,
        head_dim_idx: UInt32 = 0,
    ) -> UnsafePointer[Scalar[Self.dtype], ImmutAnyOrigin]:
        var global_token_idx = Int(
            self.cache_row_offsets[Int(batch_idx)] + start_tok_idx
        )
        var ret_ptr = self.buffer.ptr + self.buffer.layout[
            linear_idx_type=type_of(self.buffer).linear_idx_type
        ](
            Coord(
                global_token_idx,
                Int(head_idx),
                Int(head_dim_idx),
            )
        )
        return ret_ptr.as_imm().as_unsafe_any_origin()

    @always_inline
    def scales_block_paged_ptr(
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], ImmutAnyOrigin]:
        # SAFETY: Ragged operands don't support quantization; callers only
        # dereference behind comptime quantization guards.
        return UnsafePointer[
            Scalar[Self.scale_dtype], ImmutAnyOrigin
        ].unsafe_dangling()

    @always_inline
    def load_scale[
        width: Int
    ](
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[Self.scale_dtype, width]:
        return SIMD[Self.scale_dtype, width](0)

    @always_inline
    def cache_length(self, batch_idx: Int) -> Int:
        return Int(
            self.cache_row_offsets[batch_idx + 1]
            - self.cache_row_offsets[batch_idx]
        )

    @always_inline
    def max_context_length(self) -> UInt32:
        comptime assert (
            False
        ), "For RaggedMHAOperand, max_context_length is not implemented."

    @always_inline
    def num_kv_rows(self) -> Int:
        """Returns the total number of tokens in the ragged buffer."""
        return Int(self.buffer.dim[0]())

    @always_inline
    def row_idx(self, batch_idx: UInt32, start_tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        return self.cache_row_offsets[Int(batch_idx)][0] + start_tok_idx

    @always_inline
    def get_tma_row(self, encoded_index: Int32) -> Int32:
        """Convert an encoded sparse index to a physical TMA row.

        Non-paged operand: identity (no paging translation needed).
        """
        return encoded_index

    @always_inline
    def create_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        depth: Int,
        BK: Int = padded_depth[Self.dtype, swizzle_mode, depth](),
        fold_chunks: Int = 1,
        row_major: Bool = False,
    ](
        self,
        ctx: DeviceContext,
        out tma: SplitLastDimTMATensorTile[
            Self.dtype,
            IndexList[3](BN, 1, BK),
            swizzle_mode,
        ],
    ) raises:
        """Creates a TMA tile for efficient GPU memory transfers."""
        # View as [total_tokens, heads*head_dim]
        comptime assert (
            BK % swizzle_granularity[Self.dtype, swizzle_mode]()
        ) == 0
        var rows = Int(self.buffer.dim[0]())  # total tokens
        comptime smem_shape = IndexList[3](BN, 1, BK)
        comptime gmem_shape = IndexList[3](UNKNOWN_VALUE, UNKNOWN_VALUE, depth)

        tma = create_split_tma[
            smem_shape,
            gmem_shape,
            swizzle_mode=swizzle_mode,
            fold_chunks=fold_chunks,
            row_major=row_major,
        ](
            ctx,
            self.buffer.ptr.as_imm().as_unsafe_any_origin(),
            rows,
            Int(self.buffer.dim[1]()),
        )

    @always_inline
    def create_scale_tma_tile[
        BMN: Int
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            Self.scale_dtype,
            2,
            Index(1, BMN),
            Index(1, BMN),
        ],
    ) raises:
        # if per token scale, treat as 1D tensor
        comptime if Self.scale_layout.rank == 2:
            var total_elements = self.scale_buffer.num_elements()
            debug_assert(
                total_elements % 4 == 0,
                (
                    "Total_elements must be divisible by 4. Otherwise, the"
                    " Nvidia Driver will crash."
                ),
            )
            var scale_tensor = TileTensor(
                self.scale_buffer.ptr,
                row_major(Coord(Idx[1], total_elements)),
            )
            return create_tensor_tile[
                Index(1, BMN),
                swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
                __desc_shape=Index(1, BMN),
            ](ctx, scale_tensor)

        # if per token per head scale, treat as 2D tensor with shape [num_heads, total_seq_len]
        elif Self.scale_layout.rank == 3:
            comptime num_heads = Self.scale_layout.static_shape[0]
            var total_seq_len = Int(self.buffer.dim[1]())

            var scale_tensor = TileTensor(
                self.scale_buffer.ptr,
                row_major(Coord(Idx[num_heads], total_seq_len)),
            )

            return create_tensor_tile[
                Index(1, BMN),
                swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
                __desc_shape=Index(1, BMN),
            ](ctx, scale_tensor)

        else:
            comptime assert False, (
                "scale_layout must be 2D(per token) or 3D(per token per head)"
                " tensor."
            )

    @always_inline
    def create_index_scale_tma_tile[
        TILE: Int
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            Self.dtype,
            2,
            Index(1, flat_scale_window[Self.dtype, TILE]()),
            Index(1, flat_scale_window[Self.dtype, TILE]()),
        ],
    ) raises:
        """A flat window on the buffer this operand addresses.

        Over `buffer`, not `scale_buffer`: the FP8 indexer constructs this
        operand directly over the k-scale tensor, so the scales ARE the primary
        buffer and `create_scale_tma_tile` above would describe the (unset)
        companion one.
        """
        comptime assert Self.layout.rank <= 3, (
            "the flat scale window needs one scalar per token, so the buffer"
            " may carry only degenerate trailing dims"
        )
        var total_elements = self.buffer.num_elements()
        debug_assert[assert_mode="safe"](
            total_elements == Int(self.buffer.dim[0]()),
            (
                "the flat scale window assumes one element per token; this"
                " buffer has more than one"
            ),
        )
        return create_flat_scale_tma_tile[
            Self.dtype, flat_scale_window[Self.dtype, TILE]()
        ](ctx, self.buffer.ptr, total_elements)

    @always_inline
    def create_rope_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int,
        padded_depth: Int,
    ](
        self,
        ctx: DeviceContext,
        out tma: SplitLastDimTMATensorTile[
            DType.bfloat16,
            IndexList[3](BN, 1, BK),
            swizzle_mode,
        ],
    ) raises:
        """Not supported for RaggedMHAOperand."""
        comptime assert (
            False
        ), "create_rope_tma_tile is not supported for RaggedMHAOperand"

    @always_inline
    def create_gather4_tma_tile[
        tile_width: Int,
        tile_stride: Int = tile_width,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tile_height: Int = 4,
        tma_dtype: DType = Self.dtype,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            tma_dtype,
            2,
            tile_shape=IndexList[2](
                tile_height,
                _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
            ),
            desc_shape=IndexList[2](
                1,
                _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
            ),
        ],
    ) raises:
        """Creates a 2D TMA gather4 descriptor for this ragged operand."""
        # View the ragged buffer as a 2D matrix [total_tokens, tile_width]
        var rows = Int(self.buffer.dim[0]())
        tma = create_tma_tile_gather4[
            tma_dtype,
            tile_height=tile_height,
            tile_width=tile_width,
            tile_stride=tile_stride,
            swizzle_mode=swizzle_mode,
            l2_promotion=l2_promotion,
        ](ctx, self.buffer.ptr.bitcast[Scalar[tma_dtype]](), rows)

    @always_inline
    def create_rope_gather4_tma_tile[
        tile_width: Int,
        padded_depth: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tile_height: Int = 4,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            DType.bfloat16,
            2,
            tile_shape=IndexList[2](
                tile_height,
                _gather4_box_width[.bfloat16, tile_width, swizzle_mode](),
            ),
            desc_shape=IndexList[2](
                1,
                _gather4_box_width[.bfloat16, tile_width, swizzle_mode](),
            ),
        ],
    ) raises:
        """Not supported for RaggedMHAOperand."""
        comptime assert (
            False
        ), "create_rope_gather4_tma_tile is not supported for RaggedMHAOperand"

    @always_inline
    def scales_raw_ptr(
        self,
    ) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Returns a dangling pointer. Ragged operands do not support
        quantization."""
        # SAFETY: Ragged operands don't support quantization; callers only
        # dereference behind comptime quantization guards.
        return UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
