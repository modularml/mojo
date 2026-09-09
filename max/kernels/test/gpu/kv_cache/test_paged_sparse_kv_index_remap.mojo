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
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""GPU test for row-offset sparse KV remap (same dispatch as sparse MLA MOGG path)."""

from max.gpu.host import DeviceContext
from std.memory import alloc

from std.testing import assert_equal

from kv_cache.paged_sparse_kv_index_remap import (
    paged_sparse_kv_logical_to_physical_indices_from_row_offsets_dispatch,
)


@always_inline
def _find_batch_for_row_ref(
    r: Int,
    row_offsets: ImmPointer[UInt32, _],
    num_batches: Int,
) -> UInt32:
    """Matches production ``_find_batch_for_row`` (test golden only)."""
    var ru = UInt32(r)
    for b in range(num_batches):
        if ru >= row_offsets[b] and ru < row_offsets[b + 1]:
            return UInt32(b)
    return UInt32(0)


@always_inline
def _remap_one_ref(
    log_t: Int32,
    batch_u32: UInt32,
    lut: ImmPointer[UInt32, _],
    lut_cols: Int,
    lut_rows: Int,
    page_size: Int,
    invalid_block_id: UInt32,
) -> Int32:
    """Matches production ``_remap_one`` (test golden only)."""
    if log_t < 0:
        return log_t
    var t = Int(log_t)
    var bi = Int(batch_u32)
    if bi >= lut_rows:
        return Int32(-1)
    var page_idx = t // page_size
    if page_idx >= lut_cols:
        return Int32(-1)
    var tok_in_page = t % page_size
    var block_id = lut[bi * lut_cols + page_idx]
    if block_id >= invalid_block_id:
        return Int32(-1)
    return Int32(Int(block_id) * page_size + tok_in_page)


def _reference_row_offsets_remap(
    logical: ImmPointer[Int32, _],
    row_offsets: ImmPointer[UInt32, _],
    lut: ImmPointer[UInt32, _],
    physical_out: MutPointer[Int32, _],
    num_indices: Int,
    lut_cols: Int,
    lut_rows: Int,
    indices_stride: Int,
    invalid_block_id: UInt32,
    num_batches: Int,
    logical_stride0: Int,
    logical_stride1: Int,
    page_size: Int,
):
    """CPU golden mirroring ``paged_sparse_kv_logical_to_physical_indices_from_row_offsets_dispatch``.
    """
    for i in range(num_indices):
        var r = i // indices_stride
        var c = i - r * indices_stride
        var loff = r * logical_stride0 + c * logical_stride1
        var batch_u32 = _find_batch_for_row_ref(r, row_offsets, num_batches)
        physical_out[i] = _remap_one_ref(
            logical[loff],
            batch_u32,
            lut,
            lut_cols,
            lut_rows,
            page_size,
            invalid_block_id,
        )


def main() raises:
    comptime PAGE_SIZE = 128
    comptime INVALID_BLOCK = 512
    comptime LUT_ROWS = 2
    comptime LUT_COLS = 4
    comptime NUM_IDX = 5
    comptime INDICES_STRIDE = 1
    comptime NUM_BATCHES = 2
    comptime LOG_S0 = 1
    comptime LOG_S1 = 1

    # Sparse rows 0..3 → batch 0; row 4 → batch 1 (indices_stride=1 ⇒ tid i → row i).
    var h_row_off = alloc[UInt32](NUM_BATCHES + 1)
    h_row_off[0] = 0
    h_row_off[1] = 4
    h_row_off[2] = 5

    var h_lut = alloc[UInt32](LUT_ROWS * LUT_COLS)
    h_lut[0 * LUT_COLS + 0] = 10
    h_lut[0 * LUT_COLS + 1] = 5
    h_lut[0 * LUT_COLS + 2] = INVALID_BLOCK
    h_lut[0 * LUT_COLS + 3] = INVALID_BLOCK
    h_lut[1 * LUT_COLS + 0] = 3
    h_lut[1 * LUT_COLS + 1] = 20
    h_lut[1 * LUT_COLS + 2] = INVALID_BLOCK
    h_lut[1 * LUT_COLS + 3] = INVALID_BLOCK

    var h_log = alloc[Int32](NUM_IDX)
    h_log[0] = 0
    h_log[1] = 129
    h_log[2] = -1
    h_log[3] = 256
    h_log[4] = 200

    var h_expected = alloc[Int32](NUM_IDX)
    _reference_row_offsets_remap(
        h_log,
        h_row_off,
        h_lut,
        h_expected,
        NUM_IDX,
        LUT_COLS,
        LUT_ROWS,
        INDICES_STRIDE,
        UInt32(INVALID_BLOCK),
        NUM_BATCHES,
        LOG_S0,
        LOG_S1,
        PAGE_SIZE,
    )

    assert_equal(Int(h_expected[0]), 10 * PAGE_SIZE + 0)
    assert_equal(Int(h_expected[1]), 5 * PAGE_SIZE + 1)
    assert_equal(Int(h_expected[2]), -1)
    assert_equal(Int(h_expected[3]), -1)
    assert_equal(Int(h_expected[4]), 20 * PAGE_SIZE + 72)

    var h_out = alloc[Int32](NUM_IDX)

    with DeviceContext() as ctx:
        var d_log = ctx.enqueue_create_buffer[.int32](NUM_IDX)
        var d_row_off = ctx.enqueue_create_buffer[.uint32](NUM_BATCHES + 1)
        var d_lut = ctx.enqueue_create_buffer[.uint32](LUT_ROWS * LUT_COLS)
        var d_out = ctx.enqueue_create_buffer[.int32](NUM_IDX)

        ctx.enqueue_copy(d_log, h_log)
        ctx.enqueue_copy(d_row_off, h_row_off)
        ctx.enqueue_copy(d_lut, h_lut)

        paged_sparse_kv_logical_to_physical_indices_from_row_offsets_dispatch[
            "gpu", PAGE_SIZE
        ](
            d_out.unsafe_ptr(),
            d_log.unsafe_ptr(),
            d_row_off.unsafe_ptr(),
            d_lut.unsafe_ptr(),
            NUM_IDX,
            LUT_COLS,
            LUT_ROWS,
            INDICES_STRIDE,
            UInt32(INVALID_BLOCK),
            NUM_BATCHES,
            LOG_S0,
            LOG_S1,
            ctx,
        )

        ctx.enqueue_copy(h_out, d_out)

        # TODO(GPUA-92)
        # `DeviceBuffer.unsafe_ptr()` returns an untracked-origin pointer, so
        # these buffers are otherwise destroyed at the `unsafe_ptr()` calls in
        # the dispatch above — before the async kernel actually runs — freeing
        # their device memory and causing a use-after-free.
        _ = d_log^
        _ = d_row_off^
        _ = d_lut^

        ctx.synchronize()

    for i in range(NUM_IDX):
        assert_equal(Int(h_out[i]), Int(h_expected[i]), "gpu remap mismatch")

    h_row_off.free()
    h_log.free()
    h_lut.free()
    h_expected.free()
    h_out.free()
