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
"""Non-WS FA4 correctness when a trailing KV tile has fewer valid pages than
page slots. Regression test for KERN-3392.

With `0 < page_size < BN`, a `BN`-row V tile spans several pages, and the
last tile of a sequence can be short: only some of its page slots hold
in-bounds KV rows. `O += P @ V` contracts the whole tile and masked keys give
`P == 0`, so reading a skipped page slot gives `0 * non-finite == NaN` unless
the kernel cuts the contraction at the loaded boundary.

`num_keys = 584` at `page_size = 64` dispatches to `BN = 192`, leaving a
trailing tile with one valid page slot out of three. That tile sits in the
second split-K partition, so the contraction cut has to be measured in the
partition's own frame. `valid_length = 128` selects the non-WS BM=128 path,
and the tile count stays under the KV stage count so the short tile lands on
a ring slot no earlier tile of this launch has written.

A skipped slot holds whatever the previous occupant of the SM's shared memory
left there, so a regression only surfaces when those bytes decode as
non-finite bf16. The poison launch fills every SM's shared memory with bf16
NaN first, making the failure deterministic.
"""

from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.memory import external_memory
from max.gpu.sync import barrier
from max.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv, isnan, rsqrt
from std.random import seed
from std.sys import size_of
from std.utils import IndexList
from std.utils.numerics import nan

from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import CausalMask


comptime dtype = DType.bfloat16
comptime head_size = 64
comptime page_size = 64
comptime num_q_heads = 64
comptime kv_params = KVCacheStaticParams(num_heads=8, head_size=head_size)
comptime group = num_q_heads // kv_params.num_heads

comptime valid_length = 128
comptime cache_length = 456
comptime num_keys = cache_length + valid_length

# Nearly all of an SM100 SM's 228 KB, so no second block shares the SM.
comptime poison_bytes = 227 * 1024
comptime poison_elems = poison_bytes // size_of[dtype]()
comptime poison_threads = 1024
# Several times the SM count, so every SM takes at least one block.
comptime poison_blocks = 4 * DeviceContext.default_device_info.sm_count
comptime reps = 10


def poison_smem_kernel(sink: MutPointer[Float32, MutAnyOrigin]):
    """Fills this block's whole dynamic SMEM allocation with bf16 NaN."""
    var smem = external_memory[
        Scalar[dtype],
        address_space=.SHARED,
        alignment=16,
        name="poison_smem",
    ]()
    var poison = nan[dtype]()
    for i in range(Int(thread_idx.x), poison_elems, Int(block_dim.x)):
        smem[i] = poison
    # Read one element back so the stores are not optimized away.
    barrier()
    if thread_idx.x == 0:
        sink[0] = smem[poison_elems - 1].cast[.float32]()


def main() raises:
    comptime row_offsets_layout = Layout(UNKNOWN_VALUE)
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    comptime qo_layout = Layout.row_major(UNKNOWN_VALUE, num_q_heads, head_size)
    comptime paged_lut_layout = Layout.row_major[2]()
    comptime kv_block_layout = Layout.row_major[6]()
    comptime qo_shape = IndexList[3](valid_length, num_q_heads, head_size)
    comptime qo_size = valid_length * num_q_heads * head_size

    var scale = rsqrt(Float32(head_size))
    seed(0x5151)

    with DeviceContext() as ctx:
        var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](2)
        row_offsets_host[0] = 0
        row_offsets_host[1] = UInt32(valid_length)
        var row_offsets_dev = ctx.enqueue_create_buffer[.uint32](2)
        ctx.enqueue_copy(row_offsets_dev, row_offsets_host)
        var row_offsets = LayoutTensor[mut=False, .uint32, row_offsets_layout](
            row_offsets_dev,
            RuntimeLayout[row_offsets_layout].row_major(IndexList[1](2)),
        )

        var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](1)
        cache_lengths_host[0] = UInt32(cache_length)
        var cache_lengths_dev = ctx.enqueue_create_buffer[.uint32](1)
        ctx.enqueue_copy(cache_lengths_dev, cache_lengths_host)
        var cache_lengths = LayoutTensor[
            mut=False, .uint32, cache_lengths_layout
        ](
            cache_lengths_dev,
            RuntimeLayout[cache_lengths_layout].row_major(IndexList[1](1)),
        )

        var q_host = ctx.enqueue_create_host_buffer[dtype](qo_size)
        random(
            LayoutTensor[dtype, qo_layout](
                q_host.unsafe_ptr(),
                RuntimeLayout[qo_layout].row_major(qo_shape),
            )
        )
        var q_dev = ctx.enqueue_create_buffer[dtype](qo_size)
        ctx.enqueue_copy(q_dev, q_host)
        var q = LayoutTensor[mut=False, dtype, qo_layout](
            q_dev, RuntimeLayout[qo_layout].row_major(qo_shape)
        )

        # One spare block backs the LUT tail padding: the kernel reads LUT
        # columns a full tile at a time, so it can read past the last real
        # page.
        var num_pages = ceildiv(num_keys, page_size)
        var num_blocks = num_pages + 1
        var kv_shape = IndexList[6](
            num_blocks, 2, 1, page_size, kv_params.num_heads, head_size
        )
        var kv_size = (
            num_blocks * 2 * page_size * kv_params.num_heads * head_size
        )
        var kv_host = ctx.enqueue_create_host_buffer[dtype](kv_size)
        random(
            LayoutTensor[dtype, kv_block_layout](
                kv_host.unsafe_ptr(),
                RuntimeLayout[kv_block_layout].row_major(kv_shape),
            )
        )
        var kv_dev = ctx.enqueue_create_buffer[dtype](kv_size)
        ctx.enqueue_copy(kv_dev, kv_host)
        var kv_blocks = LayoutTensor[dtype, kv_block_layout](
            kv_dev, RuntimeLayout[kv_block_layout].row_major(kv_shape)
        )

        var lut_cols = ceildiv(num_pages, 8) * 8 + 16
        var lut_host = ctx.enqueue_create_host_buffer[.uint32](lut_cols)
        for i in range(lut_cols):
            lut_host[i] = UInt32(num_pages if i >= num_pages else i)
        var lut_dev = ctx.enqueue_create_buffer[.uint32](lut_cols)
        ctx.enqueue_copy(lut_dev, lut_host)
        var lut = LayoutTensor[mut=False, .uint32, paged_lut_layout](
            lut_dev,
            RuntimeLayout[paged_lut_layout].row_major(
                IndexList[2](1, lut_cols)
            ),
        )

        var kv_collection = PagedKVCacheCollection[dtype, kv_params, page_size](
            kv_blocks.as_unsafe_any_origin(),
            cache_lengths,
            lut,
            UInt32(valid_length),
            UInt32(num_keys),
        )
        var k_cache = kv_collection.get_key_cache(0)
        var v_cache = kv_collection.get_value_cache(0)

        # Another kernel on the device can scrub the poison between the two
        # launches, so run the pair several times and check every output.
        var sink_dev = ctx.enqueue_create_buffer[.float32](1)
        var test_dev = ctx.enqueue_create_buffer[dtype](reps * qo_size)
        for r in range(reps):
            ctx.enqueue_function[poison_smem_kernel](
                sink_dev,
                grid_dim=(poison_blocks, 1),
                block_dim=(poison_threads, 1),
                shared_mem_bytes=poison_bytes,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(poison_bytes)
                ),
            )
            flash_attention[ragged=True](
                LayoutTensor[dtype, qo_layout](
                    test_dev.unsafe_ptr() + r * qo_size,
                    RuntimeLayout[qo_layout].row_major(qo_shape),
                ),
                q,
                k_cache,
                v_cache,
                CausalMask(),
                row_offsets,
                scale,
                ctx,
            )

        var ref_dev = ctx.enqueue_create_buffer[dtype](qo_size)
        mha_gpu_naive[ragged=True](
            q,
            k_cache,
            v_cache,
            CausalMask(),
            LayoutTensor[dtype, qo_layout](
                ref_dev.unsafe_ptr(),
                RuntimeLayout[qo_layout].row_major(qo_shape),
            ),
            row_offsets,
            scale,
            1,
            valid_length,
            num_keys,
            num_q_heads,
            head_size,
            group,
            ctx,
        )

        var test_host = ctx.enqueue_create_host_buffer[dtype](reps * qo_size)
        var ref_host = ctx.enqueue_create_host_buffer[dtype](qo_size)
        ctx.enqueue_copy(test_host, test_dev)
        ctx.enqueue_copy(ref_host, ref_dev)
        ctx.synchronize()

        var nan_count = 0
        var max_abs_diff = Float32(0)
        for i in range(reps * qo_size):
            var got = test_host[i].cast[.float32]()
            if isnan(got):
                nan_count += 1
                continue
            var want = ref_host[i % qo_size].cast[.float32]()
            max_abs_diff = max(max_abs_diff, abs(got - want))

        _ = q_dev^
        _ = kv_dev^
        _ = lut_dev^
        _ = row_offsets_dev^
        _ = cache_lengths_dev^
        _ = sink_dev^
        _ = test_dev^
        _ = ref_dev^

        print("nan_count =", nan_count, " max-abs diff =", max_abs_diff)
        if nan_count > 0:
            raise Error("NaN x" + String(nan_count))
        if max_abs_diff > 0.02:
            raise Error("max-abs diff too large: " + String(max_abs_diff))
        print("PASSED")
