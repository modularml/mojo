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

from max.gpu.host import DeviceContext
from std.sys.info import _has_blackwell_tcgen05
from nn.index_fp8 import fp8_index, fp8_index_naive
from std.random import rand
from layout import Idx, TileTensor, row_major
from std.testing import assert_almost_equal


def test_index_fp8[
    num_heads: Int,
    depth: Int,
](batch_size: Int, seq_len: Int, num_keys: Int, ctx: DeviceContext) raises:
    print(
        "test_index_fp8 with params:",
        "num_heads:",
        num_heads,
        "depth:",
        depth,
        "batch_size:",
        batch_size,
        "seq_len:",
        seq_len,
        "num_keys:",
        num_keys,
    )
    var q_size = batch_size * seq_len * num_heads * depth
    var qs_size = batch_size * seq_len * num_heads
    var k_size = batch_size * num_keys * depth
    var ks_size = batch_size * num_keys
    var o_size = batch_size * seq_len * num_keys

    var q_ptr = ctx.enqueue_create_host_buffer[.float8_e4m3fn](q_size)
    var qs_ptr = ctx.enqueue_create_host_buffer[.float32](qs_size)
    var k_ptr = ctx.enqueue_create_host_buffer[.float8_e4m3fn](k_size)
    var ks_ptr = ctx.enqueue_create_host_buffer[.float32](ks_size)
    var o_ptr = ctx.enqueue_create_host_buffer[.float32](o_size)
    var o_ref_ptr = ctx.enqueue_create_host_buffer[.float32](o_size)
    var input_row_offsets = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    var cache_row_offsets = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )

    var q_device_ptr = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    var qs_device_ptr = ctx.enqueue_create_buffer[.float32](qs_size)
    var k_device_ptr = ctx.enqueue_create_buffer[.float8_e4m3fn](k_size)
    var ks_device_ptr = ctx.enqueue_create_buffer[.float32](ks_size)
    var input_row_offsets_device_ptr = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    var cache_row_offsets_device_ptr = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    var o_device_ptr = ctx.enqueue_create_buffer[.float32](o_size)
    var o_device_ref_ptr = ctx.enqueue_create_buffer[.float32](o_size)

    rand(q_ptr.as_span())
    rand(qs_ptr.as_span())
    rand(k_ptr.as_span())
    rand(ks_ptr.as_span())

    # input row offsets and cache row offsets
    for i in range(batch_size):
        input_row_offsets[i] = UInt32(i * seq_len)
        cache_row_offsets[i] = UInt32(i * num_keys)
    input_row_offsets[batch_size] = UInt32(batch_size * seq_len)
    cache_row_offsets[batch_size] = UInt32(batch_size * num_keys)

    ctx.enqueue_copy(q_device_ptr, q_ptr)
    ctx.enqueue_copy(qs_device_ptr, qs_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)
    ctx.enqueue_copy(ks_device_ptr, ks_ptr)
    ctx.enqueue_copy(input_row_offsets_device_ptr, input_row_offsets)
    ctx.enqueue_copy(cache_row_offsets_device_ptr, cache_row_offsets)

    # Ragged Q tensor: [total_seq_len, num_heads, depth]
    var q_device = TileTensor(
        q_device_ptr,
        row_major((batch_size * seq_len, Idx[num_heads], Idx[depth])),
    )

    var qs_device = TileTensor(
        qs_device_ptr,
        row_major((batch_size * seq_len, Idx[num_heads])),
    )

    var k_device = TileTensor(
        k_device_ptr,
        row_major((batch_size * num_keys, Idx[1], Idx[depth])),
    )

    var ks_device = TileTensor(
        ks_device_ptr,
        row_major(batch_size * num_keys),
    )

    var o_device = TileTensor(
        o_device_ptr,
        row_major((batch_size * seq_len, num_keys)),
    )

    var o_ref_device = TileTensor(
        o_device_ref_ptr,
        row_major((batch_size * seq_len, num_keys)),
    )

    var input_row_offsets_device = TileTensor(
        input_row_offsets_device_ptr,
        row_major(batch_size + 1),
    )

    var cache_row_offsets_device = TileTensor[mut=False, Engine=_](
        cache_row_offsets_device_ptr,
        row_major(batch_size + 1),
    )

    fp8_index[num_heads, depth](
        o_device,
        q_device,
        qs_device,
        k_device,
        ks_device,
        input_row_offsets_device,
        cache_row_offsets_device,
        batch_size,
        seq_len,
        num_keys,
        ctx,
    )
    ctx.synchronize()
    ctx.enqueue_copy(o_ptr, o_device_ptr)

    fp8_index_naive[num_heads, depth](
        o_ref_device,
        q_device,
        qs_device,
        k_device,
        ks_device,
        input_row_offsets_device,
        cache_row_offsets_device,
        batch_size,
        seq_len,
        num_keys,
        ctx,
    )
    ctx.synchronize()
    ctx.enqueue_copy(o_ref_ptr, o_device_ref_ptr)
    ctx.synchronize()

    for b in range(batch_size):
        for s in range(seq_len):
            for k in range(num_keys):
                var expect = o_ref_ptr[
                    b * seq_len * num_keys + s * num_keys + k
                ]
                var actual = o_ptr[b * seq_len * num_keys + s * num_keys + k]

                if abs((actual - expect)) > 1e-2:
                    print(b, s, k, actual, expect)
                assert_almost_equal(actual, expect, atol=1e-2, rtol=1e-3)

    _ = q_device_ptr
    _ = qs_device_ptr
    _ = k_device_ptr
    _ = ks_device_ptr
    _ = input_row_offsets_device_ptr
    _ = cache_row_offsets_device_ptr
    _ = o_device_ptr
    _ = o_device_ref_ptr


def main() raises:
    with DeviceContext() as ctx:
        test_index_fp8[num_heads=128, depth=128](2, 128, 128, ctx)
        test_index_fp8[num_heads=128, depth=128](2, 32, 32, ctx)
        test_index_fp8[num_heads=128, depth=128](4, 200, 200, ctx)
        test_index_fp8[num_heads=128, depth=128](1, 501, 501, ctx)
        test_index_fp8[num_heads=128, depth=128](3, 600, 600, ctx)
        test_index_fp8[num_heads=128, depth=128](4, 722, 722, ctx)
        test_index_fp8[num_heads=128, depth=128](5, 32, 64, ctx)
        test_index_fp8[num_heads=128, depth=128](2, 128, 256, ctx)

        test_index_fp8[num_heads=64, depth=128](2, 128, 128, ctx)
        test_index_fp8[num_heads=64, depth=128](2, 32, 32, ctx)
        test_index_fp8[num_heads=64, depth=128](4, 200, 200, ctx)
        test_index_fp8[num_heads=64, depth=128](1, 501, 501, ctx)
        test_index_fp8[num_heads=64, depth=128](3, 600, 600, ctx)
        test_index_fp8[num_heads=64, depth=128](4, 722, 722, ctx)
        # k-scale TMA alignment. The SM100 scorer stages the per-key scales
        # through a flat `1 x total_keys` TMA view, where a key index IS the
        # innermost coordinate and so must be 16-byte (4-f32) aligned. The
        # per-tile term is always a multiple of `BM_key`, leaving the ragged
        # batch base `b * num_keys` as the only misaligning term -- and 723
        # walks all four residues in one case (0, 3, 2, 1). Before the producer
        # rounded that base down, this faulted `CUDA_ERROR_ILLEGAL_INSTRUCTION`
        # on the odd-residue entries; 722 above covers residue 2 alone.
        test_index_fp8[num_heads=64, depth=128](4, 723, 723, ctx)
        test_index_fp8[num_heads=64, depth=128](5, 32, 64, ctx)
        test_index_fp8[num_heads=64, depth=128](2, 128, 256, ctx)

        # depth=64 stays on the scalar kernel on every target (the SM100 gate
        # requires depth == 128): regression coverage for the Q staging, whose
        # old layout-distributed copy silently staged nothing at this depth.
        test_index_fp8[num_heads=64, depth=64](2, 32, 64, ctx)
        test_index_fp8[num_heads=64, depth=64](3, 5, 33, ctx)
        test_index_fp8[num_heads=128, depth=64](2, 16, 32, ctx)

        # num_heads=32 (GLM 5.x replicated indexer): N_TOKENS=4 boundary. The
        # scalar fallback also supports 32, so these run on every target.
        test_index_fp8[num_heads=32, depth=128](2, 128, 128, ctx)
        test_index_fp8[num_heads=32, depth=128](2, 3, 64, ctx)
        test_index_fp8[num_heads=32, depth=128](1, 4, 64, ctx)
        test_index_fp8[num_heads=32, depth=128](4, 5, 200, ctx)
        test_index_fp8[num_heads=32, depth=128](3, 200, 200, ctx)
        # Alternate-N-tile route (`N_TOKENS_ALT=3` from `nn/index_fp8.mojo`).
        # Reached only when the key range is deep enough to open the key-split arm
        # (>= 64 key tiles = 8192 keys) AND 3 divides seq_len while the default 4
        # does not, with the alt tile's own block count inside
        # `_KEYSPLIT_MAX_TOKEN_TILES`. That last clause confines this to speculative
        # widths: at nh=32 it admits exactly seq_len 3, 6 and 9, so no prefill shape
        # can reach the tile. These two are therefore the ONLY cases covering the
        # MMA_N=96 kernel. 8192 is BM_key-aligned; 8300 leaves a 108-key tail, so a
        # dropped or over-run final tile cannot pass on tolerance alone. 8192 is also
        # exactly `_KEYSPLIT_MIN_KEY_TILES * BM_key`, so raising that threshold would
        # silently demote the seq_len 6 cell to a third negative control -- check the
        # tile is still reached rather than trusting a pass.
        #
        # Both cells are UNIFORM, so every token on the alt tile is live: the
        # epilogue's per-token liveness guard is covered here only on the default
        # tile (by the two negative controls, which do carry partial blocks). The
        # ragged dead-token case on the alt tile lives in `test_mla_index_fp8`,
        # whose `seq_lens=[6, 4]` cells put two dead tokens in the second block.
        # Negative controls first, at the same key depth, so that poisoning the
        # alternate tile (a comptime factor gated on `MMA_N != 128`) reaches every
        # one of them before the first expected failure: 3 divides neither 2 nor
        # 5, so both stay on the default 4-token tile -- which also keeps the
        # 256-thread tile covered on the key-split arm.
        test_index_fp8[num_heads=32, depth=128](3, 2, 8300, ctx)
        test_index_fp8[num_heads=32, depth=128](3, 5, 8300, ctx)
        test_index_fp8[num_heads=32, depth=128](2, 6, 8192, ctx)
        test_index_fp8[num_heads=32, depth=128](3, 9, 8300, ctx)

        # TP-head-sharded indexer counts exist only on the SM100 tensor-core
        # path (the scalar fallback's [16, 8] thread layout copies nothing
        # below 16 heads), so they are compile-gated to Blackwell.
        comptime if _has_blackwell_tcgen05():
            # num_heads=8 (GLM 32 heads over 4 ranks; DSv3.2 64 over 8):
            # N_TOKENS=16, cover partial tiles around the 16-token boundary.
            test_index_fp8[num_heads=8, depth=128](2, 128, 128, ctx)
            test_index_fp8[num_heads=8, depth=128](2, 17, 32, ctx)
            test_index_fp8[num_heads=8, depth=128](4, 200, 200, ctx)
            test_index_fp8[num_heads=8, depth=128](1, 501, 501, ctx)
            test_index_fp8[num_heads=8, depth=128](3, 15, 64, ctx)
            test_index_fp8[num_heads=8, depth=128](5, 16, 64, ctx)
            test_index_fp8[num_heads=8, depth=128](2, 1, 256, ctx)

            # num_heads=4 (GLM 32 heads over 8 ranks): N_TOKENS=32 boundary.
            test_index_fp8[num_heads=4, depth=128](2, 128, 128, ctx)
            test_index_fp8[num_heads=4, depth=128](2, 33, 64, ctx)
            test_index_fp8[num_heads=4, depth=128](1, 32, 64, ctx)
            test_index_fp8[num_heads=4, depth=128](2, 31, 256, ctx)
            test_index_fp8[num_heads=4, depth=128](4, 200, 200, ctx)
            test_index_fp8[num_heads=4, depth=128](1, 1, 64, ctx)
