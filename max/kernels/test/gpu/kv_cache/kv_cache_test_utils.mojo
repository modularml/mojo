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
"""Shared utilities for KV cache tests."""

from std.math import ceildiv
from std.random import shuffle
from std.utils.numerics import isinf, isnan

from max.gpu.host import DeviceBuffer, DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor

from std.utils import Index, IndexList


# Mirror of `_LUT_TAIL_PAD` in
# `max/python/max/kv_cache/paged_kv_cache/cache_manager.py`. Bump in
# lockstep if the Python value changes.
comptime _LUT_TAIL_PAD = 16


def padded_lut_cols(cols: Int) -> Int:
    """Mirror of `_padded_lut_cols` in
    `max/python/max/kv_cache/paged_kv_cache/cache_manager.py`.

    `PagedKVCache.populate`'s SIMD path requires the LUT row stride to
    be a multiple of 8 (chunk alignment, chunk capped at 8) and at
    least `cols + 15` (so a 16-wide SIMD load at any valid
    `first_lut_idx` stays in-bounds). Production allocates with this
    padding; tests must do the same.
    """
    return ((cols + 7) // 8) * 8 + _LUT_TAIL_PAD


def randperm(n: Int) -> List[Int]:
    """Return a uniformly random permutation of `[0, n)`.

    Builds the identity permutation `[0, n)` and shuffles it in place with the
    stdlib Fisher-Yates `shuffle`.

    Use this instead of `random_distinct` when the caller also needs the
    *complement* of the sample: `perm[:k]` is a uniform sample of `k` distinct
    values without replacement, and `perm[k:]` is exactly the unsampled
    remainder. That turns a "do something to every value we did NOT pick" loop
    (e.g. poisoning unreferenced KV blocks) into a slice walk, with no
    membership `Set` and no per-element `in` probe.

    Args:
        n: Size of the permutation; elements are `[0, n)`.

    Returns:
        A list holding a random permutation of `[0, n)`.
    """
    var perm: List[Int] = [i for i in range(n)]
    shuffle(perm)
    return perm^


def random_distinct(n: Int, k: Int) -> List[Int]:
    """Sample `k` distinct integers uniformly without replacement from `[0, n)`.

    The first `k` elements of a uniform random permutation are themselves a
    uniform sample of `k` distinct values without replacement, so this takes
    the `k`-element prefix of `randperm(n)`.

    This replaces the rejection-sampling idiom (draw into a `Set[Int]`,
    redraw on collision), whose expected work blows up as `k` approaches `n`:
    the last few distinct values each take on average `n / (n - i)` draws, so
    selecting nearly all of a population is roughly `n * H_n` draws
    (coupon-collector). The paged-KV LUT routinely needs almost every block,
    which is exactly that worst case. A full shuffle is `O(n)` with no redraws.

    Constraints (caller-enforced): `0 <= k <= n`.

    Args:
        n: Size of the population `[0, n)` to sample from.
        k: Number of distinct values to return.

    Returns:
        A list of `k` distinct integers in `[0, n)`, in sampled order.
    """
    var perm = randperm(n)
    perm.shrink(k)
    return perm^


def assert_no_nan_inf[
    dtype: DType, layout: Layout
](
    mut output: ManagedLayoutTensor[dtype, layout],
    name: StaticString = "output",
) raises:
    """Assert no NaN/Inf is present in `output`.

    Copies the managed tensor's device buffer back to host (via
    `tensor[update=True]()`) and linearly scans every element. Raises on the
    first NaN or Inf with the element's flat index, total size, and the
    caller-supplied `name`. Use immediately after a kernel + `synchronize()`
    to give a named, indexed failure rather than relying on tolerance
    comparisons (which mask NaN-vs-NaN matches and produce vague error
    messages).
    """
    var host = output.tensor[update=True]()
    var n = host.runtime_layout.size()
    for i in range(n):
        var v = host.ptr[i].cast[.float32]()
        if isnan(v):
            raise Error(
                String("NaN at element ")
                + String(i)
                + " of "
                + String(n)
                + " in '"
                + String(name)
                + "'"
            )
        if isinf(v):
            raise Error(
                String("Inf at element ")
                + String(i)
                + " of "
                + String(n)
                + " in '"
                + String(name)
                + "'"
            )


struct _KVCacheTestTensor[dtype: DType, layout: Layout, rank: Int](Copyable):
    comptime tensor_type = LayoutTensor[Self.dtype, Self.layout, ImmutAnyOrigin]

    var shape: IndexList[Self.rank]
    var host_ptr: MutPointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var device_buf: Optional[DeviceBuffer[Self.dtype]]

    def __init__(out self, shape: IndexList[Self.rank]):
        self.shape = shape
        self.host_ptr = alloc[Scalar[Self.dtype]](shape.flattened_length())
        self.device_buf = None

    def __deinit__(deinit self):
        self.host_ptr.free()

    def copy_to_device(mut self, ctx: DeviceContext) raises:
        self.device_buf = ctx.enqueue_create_buffer[Self.dtype](
            self.shape.flattened_length()
        )
        ctx.enqueue_copy(self.device_buf.value(), self.host_ptr)

    def host_tensor(self) -> Self.tensor_type:
        return self._tensor(self.host_ptr)

    def device_tensor(self) -> Self.tensor_type:
        return self._tensor(self.device_buf.value().unsafe_ptr())

    def _runtime_layout(self) -> RuntimeLayout[Self.layout]:
        return RuntimeLayout[Self.layout].row_major(self.shape)

    def _tensor(self, ptr: Pointer[Scalar[Self.dtype], _]) -> Self.tensor_type:
        return Self.tensor_type(
            ptr.as_imm().as_unsafe_any_origin(), self._runtime_layout()
        )


struct CacheLengthsTable(Copyable):
    var cache_lengths: _KVCacheTestTensor[
        DType.uint32, Layout(UNKNOWN_VALUE), 1
    ]
    var input_row_offsets: _KVCacheTestTensor[
        DType.uint32, Layout(UNKNOWN_VALUE), 1
    ]

    var batch_size: Int
    var max_full_context_length: Int
    var max_seq_length_batch: Int
    var total_length: Int

    def __init__(out self, batch_size: Int):
        self.batch_size = batch_size
        self.cache_lengths = type_of(self.cache_lengths)(Index(batch_size))
        self.input_row_offsets = type_of(self.input_row_offsets)(
            Index(batch_size + 1)
        )
        self.max_full_context_length = 0
        self.max_seq_length_batch = 0
        self.total_length = 0

    def _build(
        mut self,
        prompt_lens: List[Int],
        cache_lens: List[Int],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        var cache_lengths_ptr = self.cache_lengths.host_ptr
        var input_row_offsets_ptr = self.input_row_offsets.host_ptr

        var max_full_context_length = 0
        var max_seq_length_batch = 0
        var total_length = 0

        for batch, (prompt_len, cache_len) in enumerate(
            zip(prompt_lens, cache_lens)
        ):
            cache_lengths_ptr[batch] = UInt32(cache_len)
            input_row_offsets_ptr[batch] = UInt32(total_length)

            max_full_context_length = max(
                max_full_context_length, cache_len + prompt_len
            )
            max_seq_length_batch = max(max_seq_length_batch, prompt_len)
            total_length += prompt_len

        input_row_offsets_ptr[self.batch_size] = UInt32(total_length)

        self.max_full_context_length = max_full_context_length
        self.max_seq_length_batch = max_seq_length_batch
        self.total_length = total_length

        if ctx:
            self.cache_lengths.copy_to_device(ctx.value())
            self.input_row_offsets.copy_to_device(ctx.value())

    @staticmethod
    def build(
        prompt_lens: List[Int],
        cache_lens: List[Int],
        ctx: Optional[DeviceContext],
    ) raises -> Self:
        var batch_size = len(prompt_lens)
        var cache_lengths_table = Self(batch_size)
        cache_lengths_table._build(prompt_lens, cache_lens, ctx)
        return cache_lengths_table^


struct PagedLookupTable[page_size: Int](Copyable):
    var paged_lut: _KVCacheTestTensor[.uint32, Layout.row_major[2](), 2]

    def __init__(
        out self, batch_size: Int, max_full_context_length: Int
    ) raises:
        # Pad the LUT inner dim to honor `PagedKVCache.populate`'s SIMD
        # padding invariant — see `padded_lut_cols`.
        self.paged_lut = type_of(self.paged_lut)(
            Index(
                batch_size,
                padded_lut_cols(
                    ceildiv(max_full_context_length, Self.page_size)
                ),
            )
        )

    def _build(
        mut self,
        prompt_lens: List[Int],
        cache_lens: List[Int],
        num_paged_blocks: Int,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        var batch_size = len(prompt_lens)

        var host_tensor = LayoutTensor[.uint32, type_of(self.paged_lut).layout](
            self.paged_lut.host_ptr,
            self.paged_lut._runtime_layout(),
        )
        # Sample one distinct paged block per page across the whole batch up
        # front, then hand them out in iteration order. Total pages needed is
        # <= num_paged_blocks by construction.
        var total_pages = 0
        for batch in range(batch_size):
            total_pages += ceildiv(
                prompt_lens[batch] + cache_lens[batch], Self.page_size
            )
        var blocks = random_distinct(num_paged_blocks, total_pages)

        var page_pos = 0
        for batch in range(batch_size):
            var seq_len = prompt_lens[batch] + cache_lens[batch]

            for block_idx in range(0, ceildiv(seq_len, Self.page_size)):
                host_tensor[batch, block_idx] = UInt32(blocks[page_pos])
                page_pos += 1

        if ctx:
            self.paged_lut.copy_to_device(ctx.value())

    @staticmethod
    def build(
        prompt_lens: List[Int],
        cache_lens: List[Int],
        max_full_context_length: Int,
        num_paged_blocks: Int,
        ctx: Optional[DeviceContext],
    ) raises -> Self:
        var batch_size = len(prompt_lens)
        var paged_lut = Self(batch_size, max_full_context_length)
        paged_lut._build(prompt_lens, cache_lens, num_paged_blocks, ctx)
        return paged_lut^

    @staticmethod
    def build[
        batch_size: Int
    ](
        prompt_lens: IndexList[batch_size],
        cache_lens: IndexList[batch_size],
        max_full_context_length: Int,
        num_paged_blocks: Int,
        ctx: DeviceContext,
    ) raises -> Self:
        @__parameter
        def _to_list(idx_list: IndexList) -> List[Int]:
            var list = List[Int](capacity=idx_list.size)
            for i in range(idx_list.size):
                list.append(idx_list[i])
            return list^

        return Self.build(
            _to_list(prompt_lens),
            _to_list(cache_lens),
            max_full_context_length,
            num_paged_blocks,
            ctx,
        )

    def host_tensor(self) -> type_of(self.paged_lut).tensor_type:
        return self.paged_lut.host_tensor()

    def device_tensor(self) -> type_of(self.paged_lut).tensor_type:
        return self.paged_lut.device_tensor()
