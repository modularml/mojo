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

from std.collections import OptionalReg
from std.math import rsqrt
from std.memory import unsafe_memcpy
from std.random import random_ui64, seed
from std.sys import get_defined_bool

from max.gpu.host import DeviceContext
from kv_cache_test_utils import random_distinct
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor
from layout._fillers import random
from nn.attention.gpu.amd_structured.config import (
    mha_decode_fold_tile_q_seq_len,
    mha_decode_fold_wide_mma,
)
from nn.attention.gpu.mha import _mha_decode_fold_ok, flash_attention
from nn.attention.mha_mask import CausalMask
from std.testing import assert_almost_equal, assert_true

from std.utils import Index, IndexList

# Selects the AMD-only fp8 arm, which covers the decode fold's padded wide-MFMA
# tile. `main` calls exactly one arm, so neither target elaborates the other's
# kernels.
comptime FOLD_FP8 = get_defined_bool["FOLD_FP8", False]()

comptime kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
comptime llama_num_q_heads = 32

# MiniMax-M3's dense-layer shape at TP4: 16 query heads over one KV head, so
# `group == num_heads` and the AMD decode query-token fold is eligible.
comptime kv_params_single_kv = KVCacheStaticParams(num_heads=1, head_size=128)
comptime fold_num_q_heads = 16

# The EAGLE3 draft shape at TP4: 16 query heads over 16 KV heads (`group == 1`),
# which folds on the token-strided Q/O path.
comptime kv_params_one_q_per_kv = KVCacheStaticParams(
    num_heads=16, head_size=128
)


def execute_ragged_flash_attention[
    num_q_heads: Int,
    dtype: DType,
    kv_params: KVCacheStaticParams,
    output_type: DType = dtype,
    sink: Bool = False,
](
    valid_lengths: List[Int],
    max_seq_len_cache: Int,
    cache_lengths: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
) raises:
    comptime num_blocks = 32

    var batch_size = len(valid_lengths)
    debug_assert(
        batch_size < num_blocks,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured num_blocks (",
        num_blocks,
        ")",
    )
    assert len(valid_lengths) == len(
        cache_lengths
    ), "expected valid_lengths and cache_lengths size to be equal"

    # Define layouts
    comptime input_row_offsets_layout = Layout(UNKNOWN_VALUE)
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    comptime valid_lengths_layout = Layout(UNKNOWN_VALUE)
    comptime lookup_table_layout = Layout(UNKNOWN_VALUE)
    comptime q_ragged_static_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, kv_params.head_size
    )
    comptime q_padded_static_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_q_heads, kv_params.head_size
    )
    comptime kv_block_static_layout = Layout.row_major[6]()

    var total_length = 0
    var max_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        max_context_length = max(
            max_context_length, cache_lengths[i] + valid_lengths[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths[i])
        total_length += valid_lengths[i]

    # Create managed tensors for offset and length metadata.
    var input_row_offsets = ManagedLayoutTensor[
        .uint32, input_row_offsets_layout
    ](
        RuntimeLayout[input_row_offsets_layout].row_major(
            Index(batch_size + 1)
        ),
        ctx,
    )
    var cache_lengths_managed = ManagedLayoutTensor[
        .uint32, cache_lengths_layout
    ](
        RuntimeLayout[cache_lengths_layout].row_major(Index(batch_size)),
        ctx,
    )
    var valid_lengths_managed = ManagedLayoutTensor[
        .uint32, valid_lengths_layout
    ](
        RuntimeLayout[valid_lengths_layout].row_major(Index(batch_size)),
        ctx,
    )

    # Initialize row offsets and lengths
    var input_row_offsets_host = input_row_offsets.tensor[update=False]()
    var running_total = 0
    for i in range(batch_size):
        input_row_offsets_host[i] = UInt32(running_total)
        running_total += valid_lengths[i]
    input_row_offsets_host[batch_size] = UInt32(running_total)

    var cache_lengths_host = cache_lengths_managed.tensor[update=False]()
    var valid_lengths_host = valid_lengths_managed.tensor[update=False]()
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])
        valid_lengths_host[i] = UInt32(valid_lengths[i])

    # Create q tensors
    var q_ragged_shape = IndexList[3](
        total_length, num_q_heads, kv_params.head_size
    )
    var q_ragged_runtime_layout = RuntimeLayout[
        q_ragged_static_layout
    ].row_major(q_ragged_shape)

    var q_padded_shape = IndexList[4](
        batch_size, max_prompt_length, num_q_heads, kv_params.head_size
    )
    var q_padded_runtime_layout = RuntimeLayout[
        q_padded_static_layout
    ].row_major(q_padded_shape)

    var q_ragged = ManagedLayoutTensor[dtype, q_ragged_static_layout](
        q_ragged_runtime_layout, ctx
    )
    var q_padded = ManagedLayoutTensor[dtype, q_padded_static_layout](
        q_padded_runtime_layout, ctx
    )

    # Initialize q_ragged with random data
    var q_ragged_host = q_ragged.tensor()
    random(q_ragged_host)

    # Also initialize q_padded by copying from q_ragged
    var q_padded_host = q_padded.tensor()
    # copy over the ragged values to the padded tensor.
    # Don't worry about padded values, we won't read them.
    for bs in range(batch_size):
        var unpadded_seq_len = valid_lengths[bs]
        var ragged_start_idx = Int(input_row_offsets_host[bs])
        var padded_ptr = q_padded_host.ptr + (
            bs * max_prompt_length * num_q_heads * kv_params.head_size
        )
        var ragged_ptr = q_ragged_host.ptr + (
            ragged_start_idx * num_q_heads * kv_params.head_size
        )
        unsafe_memcpy(
            dest=padded_ptr,
            src=ragged_ptr,
            count=unpadded_seq_len * num_q_heads * kv_params.head_size,
        )

    # Create output tensors
    var ref_output_shape = IndexList[4](
        batch_size, max_prompt_length, num_q_heads, kv_params.head_size
    )
    var ref_output_runtime_layout = RuntimeLayout[
        q_padded_static_layout
    ].row_major(ref_output_shape)
    var ref_output = ManagedLayoutTensor[output_type, q_padded_static_layout](
        ref_output_runtime_layout, ctx
    )

    var test_output_shape = IndexList[3](
        total_length, num_q_heads, kv_params.head_size
    )
    var test_output_runtime_layout = RuntimeLayout[
        q_ragged_static_layout
    ].row_major(test_output_shape)
    var test_output = ManagedLayoutTensor[output_type, q_ragged_static_layout](
        test_output_runtime_layout, ctx
    )

    # Initialize kv_block with random data using regular host memory
    # (not host-pinned memory via map_to_host) to avoid exhausting
    # the limited host-pinned memory buffer cache
    var kv_block_shape = IndexList[6](
        num_blocks,
        2,
        num_layers,
        max_seq_len_cache,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var kv_block_runtime_layout = RuntimeLayout[
        kv_block_static_layout
    ].row_major(kv_block_shape)

    var kv_block = ManagedLayoutTensor[dtype, kv_block_static_layout](
        kv_block_runtime_layout, ctx
    )
    var kv_block_host_tensor = kv_block.tensor()
    random(kv_block_host_tensor)

    # Create lookup table
    var lookup_table_managed = ManagedLayoutTensor[
        .uint32, lookup_table_layout
    ](
        RuntimeLayout[lookup_table_layout].row_major(Index(batch_size)),
        ctx,
    )

    # Initialize lookup table with random block indices
    var lookup_table_host = lookup_table_managed.tensor[update=False]()
    # Assign each batch entry a distinct block. `random_ui64` is inclusive, so
    # the original draw range `[0, num_blocks - 1]` is a population of
    # `num_blocks` blocks.
    var lut_blocks = random_distinct(num_blocks, batch_size)
    for idx in range(batch_size):
        lookup_table_host[idx] = UInt32(lut_blocks[idx])

    # Create layout tensors for GPU operations
    var input_row_offsets_tensor = input_row_offsets.device_tensor()
    var valid_lengths_tensor = valid_lengths_managed.device_tensor()

    var cache_lengths_tensor = cache_lengths_managed.device_tensor()

    var lookup_table_tensor = lookup_table_managed.device_tensor()

    var kv_block_tensor = kv_block.device_tensor()

    var kv_collection_device = ContinuousBatchingKVCacheCollection[
        dtype, kv_params
    ](
        kv_block_tensor,
        cache_lengths_tensor,
        lookup_table_tensor,
        UInt32(max_prompt_length),
        UInt32(max_context_length),
    )
    var k_cache_device = kv_collection_device.get_key_cache(layer_idx)
    var v_cache_device = kv_collection_device.get_value_cache(layer_idx)

    # Create sink weights
    var sink_weights_shape = IndexList[1](num_q_heads)
    var sink_weights = ManagedLayoutTensor[
        dtype, Layout.row_major(UNKNOWN_VALUE)
    ](
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            sink_weights_shape
        ),
        ctx,
    )

    # Initialize sink weights with varying negative values
    var sink_weights_host = sink_weights.tensor[update=False]()
    for h in range(num_q_heads):
        sink_weights_host[h] = Scalar[dtype](-2.0 - 0.5 * Float64(h))

    var sink_weights_device_tensor: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None

    comptime if sink:
        sink_weights_device_tensor = LayoutTensor[
            dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ](
            sink_weights.device_tensor().ptr,
            sink_weights.device_tensor().runtime_layout,
        )

    var q_ragged_tensor = q_ragged.device_tensor()
    var q_padded_tensor = q_padded.device_tensor()
    var test_output_tensor = test_output.device_tensor()
    var ref_output_tensor = ref_output.device_tensor()

    # ragged execution with sink weights
    flash_attention[ragged=True, sink=sink](
        test_output_tensor,
        q_ragged_tensor,
        k_cache_device,
        v_cache_device,
        CausalMask(),
        input_row_offsets_tensor,
        rsqrt(Float32(kv_params.head_size)),
        ctx,
        sink_weights=sink_weights_device_tensor,
    )
    # padded execution
    flash_attention[sink=sink, naive_kernel=True](
        ref_output_tensor,
        q_padded_tensor,
        k_cache_device,
        v_cache_device,
        CausalMask(),
        valid_lengths_tensor,
        rsqrt(Float32(kv_params.head_size)),
        ctx,
        sink_weights=sink_weights_device_tensor,
    )
    # Verify results
    var row_offsets_tensor = input_row_offsets.tensor()
    var test_out_tensor = test_output.tensor()
    var ref_out_tensor = ref_output.tensor()

    comptime rtol = 6e-2 if dtype.is_float8() else (
        1e-2 if dtype == .bfloat16 else 1e-4
    )

    for bs in range(batch_size):
        var prompt_len = valid_lengths[bs]
        var ragged_offset = Int(row_offsets_tensor[bs])
        for s in range(prompt_len):
            for h in range(num_q_heads):
                for hd in range(kv_params.head_size):
                    var ref_val = ref_out_tensor[bs, s, h, hd]
                    var test_val = test_out_tensor[ragged_offset + s, h, hd]
                    try:
                        assert_almost_equal(
                            ref_val,
                            test_val,
                            rtol=rtol,
                            atol=5e-3,  # numerical instability between naive and optimized kernels
                        )
                    except e:
                        print(
                            "MISMATCH:",
                            bs,
                            s,
                            h,
                            hd,
                            ref_val,
                            test_val,
                        )
                        raise e^


def execute_flash_attention_suite(ctx: DeviceContext) raises:
    comptime dtypes = (DType.float32, DType.bfloat16)

    for bs in [1, 16]:
        comptime for dtype_idx in range(len(dtypes)):
            comptime dtype = rebind[DType](dtypes[dtype_idx])

            var ce_cache_sizes = List[Int]()
            var ce_seq_lens = List[Int]()
            var tg_cache_sizes = List[Int]()
            var tg_seq_lens = List[Int]()
            for _ in range(bs):
                tg_seq_lens.append(1)
                tg_cache_sizes.append(Int(random_ui64(512, 1024)))
                ce_seq_lens.append(Int(random_ui64(512, 1024)))
                ce_cache_sizes.append(0)
            print("CE", bs, dtype)
            execute_ragged_flash_attention[
                llama_num_q_heads, dtype, kv_params_llama3
            ](ce_seq_lens, 1024, ce_cache_sizes, 2, 1, ctx)

            print("TG", bs, dtype)
            execute_ragged_flash_attention[
                llama_num_q_heads, dtype, kv_params_llama3
            ](tg_seq_lens, 1024, tg_cache_sizes, 2, 0, ctx)

    # edge cases
    var short_ce_seq_len: List[Int] = [2]
    var short_ce_cache_size: List[Int] = [0]
    execute_ragged_flash_attention[
        llama_num_q_heads, DType.bfloat16, kv_params_llama3
    ](short_ce_seq_len, 1024, short_ce_cache_size, 2, 1, ctx)


def test_flash_attention_with_sink_weights(ctx: DeviceContext) raises:
    var valid_lengths: List[Int] = [100, 200, 300]
    var max_seq_len_cache = 1024
    var cache_lengths: List[Int] = [100, 200, 300]
    var num_layers = 1
    var layer_idx = 0

    execute_ragged_flash_attention[
        llama_num_q_heads, DType.float32, kv_params_llama3, sink=True
    ](
        valid_lengths,
        max_seq_len_cache,
        cache_lengths,
        num_layers,
        layer_idx,
        ctx,
    )

    execute_ragged_flash_attention[
        llama_num_q_heads, DType.bfloat16, kv_params_llama3, sink=True
    ](
        valid_lengths,
        max_seq_len_cache,
        cache_lengths,
        num_layers,
        layer_idx,
        ctx,
    )

    valid_lengths: List[Int] = [1, 1, 1]
    print("Testing TG")
    execute_ragged_flash_attention[
        llama_num_q_heads, DType.float32, kv_params_llama3, sink=True
    ](
        valid_lengths,
        max_seq_len_cache,
        cache_lengths,
        num_layers,
        layer_idx,
        ctx,
    )
    print("Testing TG BF16")
    execute_ragged_flash_attention[
        llama_num_q_heads, DType.bfloat16, kv_params_llama3, sink=True
    ](
        valid_lengths,
        max_seq_len_cache,
        cache_lengths,
        num_layers,
        layer_idx,
        ctx,
    )


def test_speculative_decode_query_lengths(ctx: DeviceContext) raises:
    """Verify-step query lengths (S > 1) against the padded naive reference.

    On AMD these route to the decode kernel's query-token fold, which writes rows
    `input_row_offsets[b] .. +S`. A NON-UNIFORM batch is what catches a wrong row
    base — a short sequence would overwrite its neighbour's tokens — since the
    comparison reads each output row at its ragged offset. Elsewhere these lengths
    take prefill and are still valid coverage.
    """
    var max_seq_len_cache = 1024

    # Uniform S, one kernel instantiation per S on the fold path. S=8 is the
    # 1+7 verify step: 16 query heads over one KV head is BM = 128.
    for s in [2, 3, 4, 5, 6, 7, 8]:
        var seq_lens = List[Int]()
        var cache_lens = List[Int]()
        for _ in range(4):
            seq_lens.append(s)
            cache_lens.append(Int(random_ui64(512, 1000)))
        execute_ragged_flash_attention[
            fold_num_q_heads, DType.bfloat16, kv_params_single_kv
        ](seq_lens, max_seq_len_cache, cache_lens, 2, 0, ctx)

    # Non-uniform S in one launch (drafts accepted to different depths, including
    # a plain 1-token decode): the fold dispatches at S = max(seq_lens) and each
    # CTA clamps to its own runtime length.
    var mixed_seq_lens: List[Int] = [4, 1, 2, 4, 3]
    var mixed_cache_lens: List[Int] = [900, 512, 700, 1000, 613]
    execute_ragged_flash_attention[
        fold_num_q_heads, DType.bfloat16, kv_params_single_kv
    ](mixed_seq_lens, max_seq_len_cache, mixed_cache_lens, 2, 0, ctx)

    # Same at the widest arm (S=8), where a short sequence's pad rows span
    # warps rather than lanes.
    var wide_mixed_seq_lens: List[Int] = [8, 1, 6, 3, 7]
    var wide_mixed_cache_lens: List[Int] = [900, 512, 700, 1000, 613]
    execute_ragged_flash_attention[
        fold_num_q_heads, DType.bfloat16, kv_params_single_kv
    ](wide_mixed_seq_lens, max_seq_len_cache, wide_mixed_cache_lens, 2, 0, ctx)

    # Single sequence, and a cache length far short of the BN=128 key tile.
    var one_seq: List[Int] = [4]
    var one_cache: List[Int] = [29]
    execute_ragged_flash_attention[
        fold_num_q_heads, DType.bfloat16, kv_params_single_kv
    ](one_seq, max_seq_len_cache, one_cache, 2, 0, ctx)

    # `group == 1`: a folded CTA's rows are the S tokens, stepping by the BSHD
    # token stride rather than by depth. S=5 and S=9 are EAGLE3 step-0 widths.
    for s in [2, 3, 4, 5, 6, 7, 8, 9]:
        var one_q_seq_lens = List[Int]()
        var one_q_cache_lens = List[Int]()
        for _ in range(4):
            one_q_seq_lens.append(s)
            one_q_cache_lens.append(Int(random_ui64(512, 1000)))
        execute_ragged_flash_attention[
            fold_num_q_heads, DType.bfloat16, kv_params_one_q_per_kv
        ](one_q_seq_lens, max_seq_len_cache, one_q_cache_lens, 2, 0, ctx)

    # Non-uniform S at `group == 1` — the production case, since a step-0
    # declares K+2 but feeds K+1 rows. Exercises pad-row clamping where dead
    # rows sit `num_heads*depth` apart instead of adjacent.
    var one_q_mixed_cache_lens: List[Int] = [900, 512, 700, 1000, 613]
    var one_q_mixed_seq_lens: List[Int] = [5, 1, 4, 2, 3]
    execute_ragged_flash_attention[
        fold_num_q_heads, DType.bfloat16, kv_params_one_q_per_kv
    ](
        one_q_mixed_seq_lens,
        max_seq_len_cache,
        one_q_mixed_cache_lens,
        2,
        0,
        ctx,
    )
    var one_q_wide_mixed_seq_lens: List[Int] = [9, 1, 8, 2, 6]
    execute_ragged_flash_attention[
        fold_num_q_heads, DType.bfloat16, kv_params_one_q_per_kv
    ](
        one_q_wide_mixed_seq_lens,
        max_seq_len_cache,
        one_q_mixed_cache_lens,
        2,
        0,
        ctx,
    )


def test_decode_fold_padded_tile_fp8(ctx: DeviceContext) raises:
    """The AMD decode fold's padded wide-MFMA tile, which is fp8-only.

    Every fold case above is bf16 and the wide arm is gated on
    `dtype.is_float8()`, so those exercise only the narrow 16-row geometry.
    Padding is what earns this its own arm: the tile's height stops equalling
    the tokens a sequence carries, so every Q/output/split-K stride has to come
    from the real width instead. On the ragged path those offsets are genuinely
    runtime, which is why the fold admits ragged where it excludes padded
    batches — but "should be inert" is not evidence.

    Both arms read the identical fp8 query and cache, so the naive reference
    needs no quantisation round-trip and any mismatch is the fold's.
    """
    comptime fp8 = DType.float8_e4m3fn
    comptime depth = kv_params_single_kv.head_size

    # The padded band's ends: 3 is the pad floor (48 source rows) and 8 is the
    # tile itself, so iterating between them spans every width that pads.
    comptime first_padded_width = 3
    comptime last_padded_width = 8

    # A green comparison proves nothing about the padded wide arm unless these
    # shapes take it — the fold could fall back to prefill and still be right.
    comptime for S in range(first_padded_width, last_padded_width + 1):
        assert_true(
            _mha_decode_fold_ok[
                fp8, depth, fold_num_q_heads, fold_num_q_heads, S, ragged=True
            ](),
            "ragged fold must admit this width",
        )
        assert_true(
            mha_decode_fold_tile_q_seq_len[
                fp8, fold_num_q_heads, fold_num_q_heads, S
            ]()
            == 8,
            "width must pad onto the wide tile",
        )

    # The predicate reads the PADDED slot count, not the source width:
    # `[fp8, 16, 16, 4]` is False, so checking that alone would read as the arm
    # being unreachable.
    assert_true(
        mha_decode_fold_wide_mma[fp8, fold_num_q_heads, fold_num_q_heads, 8]()
    )

    # Each padded width, one kernel instantiation apiece.
    var cache_lens: List[Int] = [900, 512, 700, 1000]
    for s in range(first_padded_width, last_padded_width + 1):
        var seq_lens = List[Int]()
        for _ in range(len(cache_lens)):
            seq_lens.append(s)
        execute_ragged_flash_attention[
            fold_num_q_heads, fp8, kv_params_single_kv, DType.bfloat16
        ](seq_lens, 1024, cache_lens, 2, 0, ctx)

    # Sequences shorter than the dispatched width, the real risk: the fold
    # dispatches at S = max(seq_lens) and pads THAT, so a short sequence's rows
    # are dead twice over and a tile-derived stride lands them on the next
    # sequence's rows. Only a non-uniform batch sees it.
    var mixed_cache_lens: List[Int] = [900, 512, 700, 1000, 613]

    var mixed_seq_lens: List[Int] = [4, 1, 2, 4, 3]
    execute_ragged_flash_attention[
        fold_num_q_heads, fp8, kv_params_single_kv, DType.bfloat16
    ](mixed_seq_lens, 1024, mixed_cache_lens, 2, 0, ctx)

    # At the widest arm a short sequence's dead rows span whole warps.
    var wide_mixed_seq_lens: List[Int] = [8, 1, 6, 3, 7]
    execute_ragged_flash_attention[
        fold_num_q_heads, fp8, kv_params_single_kv, DType.bfloat16
    ](wide_mixed_seq_lens, 1024, mixed_cache_lens, 2, 0, ctx)

    # Every sequence below the pad floor, so the tile is mostly dead rows.
    var narrow_mixed_seq_lens: List[Int] = [3, 1, 3, 2, 1]
    execute_ragged_flash_attention[
        fold_num_q_heads, fp8, kv_params_single_kv, DType.bfloat16
    ](narrow_mixed_seq_lens, 1024, mixed_cache_lens, 2, 0, ctx)

    # One sequence whose cache falls far short of the BN=128 key tile.
    var one_seq: List[Int] = [4]
    var one_cache: List[Int] = [29]
    execute_ragged_flash_attention[
        fold_num_q_heads, fp8, kv_params_single_kv, DType.bfloat16
    ](one_seq, 1024, one_cache, 2, 0, ctx)

    var wide_seq: List[Int] = [8]
    execute_ragged_flash_attention[
        fold_num_q_heads, fp8, kv_params_single_kv, DType.bfloat16
    ](wide_seq, 1024, one_cache, 2, 0, ctx)


def main() raises:
    seed(42)
    with DeviceContext() as ctx:
        comptime if FOLD_FP8:
            test_decode_fold_padded_tile_fp8(ctx)
        else:
            execute_flash_attention_suite(ctx)

            # Test sink weights functionality
            test_flash_attention_with_sink_weights(ctx)

            test_speculative_decode_query_lengths(ctx)
