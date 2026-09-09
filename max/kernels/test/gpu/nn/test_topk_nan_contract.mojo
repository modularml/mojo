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

"""A NaN in a row must be skipped, not allowed to hide the row's real maximum.

`TopK_2.insert` folds with a strict `>`, so a NaN never wins -- the contract
`test_argmax_streaming_gpu.mojo` documents ("NaN is skipped rather than
propagated"). `TopKHeap`, the register heap the two-stage `topk_gpu` uses for
its per-thread scan, has to agree, because a greedy request (`top_k == 1`)
takes that path rather than the fused `topk_fi` one.

The row here puts a NaN at index 0 and the true maximum at index 16384. With
`block_size = 256` and `num_blocks_per_input = 8` the scan stride is 2048, so
one thread sees both: the NaN first (filling a heap slot, and becoming the
heap's eviction threshold) and the maximum on its ninth insert, once the heap
is full. Any NaN-unaware eviction rule silently drops the maximum there.
"""

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from nn.topk import topk_gpu
from std.testing import assert_equal
from std.utils.numerics import nan

comptime DTYPE = DType.float32
comptime IDX = DType.int64
comptime BLOCK_SIZE = 256
comptime NUM_BLOCKS = 8
comptime N = 32768
comptime STRIDE = BLOCK_SIZE * NUM_BLOCKS
comptime NAN_IDX = 0
comptime MAX_IDX = 8 * STRIDE  # ninth element of the NaN thread's scan


def check_nan_does_not_hide_max(ctx: DeviceContext, batch_size: Int) raises:
    var in_buf = ctx.enqueue_create_buffer[DTYPE](batch_size * N)
    var out_vals = ctx.enqueue_create_buffer[DTYPE](batch_size)
    var out_idxs = ctx.enqueue_create_buffer[IDX](batch_size)

    var in_t = TileTensor(in_buf, row_major(Coord(batch_size, N)))
    with in_buf.map_to_host() as h:
        var t = TileTensor(h, row_major(Coord(batch_size, N)))
        for b in range(batch_size):
            for i in range(N):
                t[b, i] = Scalar[DTYPE](0.0)
            t[b, NAN_IDX] = nan[DTYPE]()
            t[b, MAX_IDX] = Scalar[DTYPE](5.0)
    with out_idxs.map_to_host() as h:
        for b in range(batch_size):
            h[b] = Scalar[IDX](-7)

    topk_gpu[sampling=False, largest=True](
        ctx,
        1,
        in_t.as_unsafe_any_origin().as_immut(),
        TileTensor(out_vals, row_major(Coord(batch_size, 1))),
        TileTensor(out_idxs, row_major(Coord(batch_size, 1))),
        block_size=BLOCK_SIZE,
        num_blocks_per_input=NUM_BLOCKS,
    )
    ctx.synchronize()

    with out_idxs.map_to_host() as h:
        for b in range(batch_size):
            assert_equal(Int(h[b]), MAX_IDX)

    _ = in_buf^
    _ = out_vals^
    _ = out_idxs^


def check_all_nan_row_is_in_range(ctx: DeviceContext) raises:
    """An all-NaN row has no real maximum; the reported index must still be a
    legal offset, since callers use it to index the vocabulary."""
    var in_buf = ctx.enqueue_create_buffer[DTYPE](N)
    var out_vals = ctx.enqueue_create_buffer[DTYPE](1)
    var out_idxs = ctx.enqueue_create_buffer[IDX](1)

    var in_t = TileTensor(in_buf, row_major(Coord(1, N)))
    with in_buf.map_to_host() as h:
        var t = TileTensor(h, row_major(Coord(1, N)))
        for i in range(N):
            t[0, i] = nan[DTYPE]()
    with out_idxs.map_to_host() as h:
        h[0] = Scalar[IDX](-7)

    topk_gpu[sampling=False, largest=True](
        ctx,
        1,
        in_t.as_unsafe_any_origin().as_immut(),
        TileTensor(out_vals, row_major(Coord(1, 1))),
        TileTensor(out_idxs, row_major(Coord(1, 1))),
        block_size=BLOCK_SIZE,
        num_blocks_per_input=NUM_BLOCKS,
    )
    ctx.synchronize()

    with out_idxs.map_to_host() as h:
        var idx = Int(h[0])
        if idx < 0 or idx >= N:
            raise Error(
                "all-NaN row produced out-of-range index " + String(idx)
            )

    _ = in_buf^
    _ = out_vals^
    _ = out_idxs^


def main() raises:
    with DeviceContext() as ctx:
        check_nan_does_not_hide_max(ctx, 1)
        check_nan_does_not_hide_max(ctx, 4)
        check_all_nan_row_is_in_range(ctx)
        print("test_topk_nan_contract: OK")
