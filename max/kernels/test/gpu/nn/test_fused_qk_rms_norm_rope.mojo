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
"""Correctness test for the fused QK RMSNorm + RoPE kernel.

Verifies `fused_qk_rms_norm_rope_ragged_paged` against the two-step reference
(`fused_qk_rms_norm_ragged_paged` followed by `fused_qk_rope_ragged`) on
identical inputs. Covers the two MiniMax-M3 RoPE geometries:

  * head_dim=128, rope_dim=128 (full rope, main attention): 64 Q-heads, 4 KV-heads.
  * head_dim=128, rope_dim=64  (partial rope, sparse indexer): 4 Q-heads, 1 KV-head.

Both BF16 and non-interleaved (safetensors) RoPE, matching M3.

Also verifies `fused_dual_qk_rms_norm_rope_ragged_paged` (the M3 main+indexer
fusion) is bit-exact (`assert_equal`) to two separate single-kernel launches.
"""

from std.collections import Set
from std.math import ceildiv
from std.random import random_ui64

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout._fillers import random
from std.memory import unsafe_memcpy

from nn.fused_qk_rope import fused_qk_rope_ragged
from nn.kv_cache import (
    fused_dual_qk_rms_norm_rope_ragged_paged,
    fused_qk_rms_norm_ragged_paged,
    fused_qk_rms_norm_rope_ragged_paged,
)
from std.testing import assert_almost_equal, assert_equal

from std.utils import Index, IndexList


def run_fused_qk_rms_norm_rope[
    dtype: DType,
    head_size: Int,
    rope_dim: Int,
    num_q_heads: Int,
    num_k_heads: Int,
](
    ctx: DeviceContext,
    # The two-step reference rounds the normed Q/K to BF16 *before* RoPE (the
    # rms_norm kernel writes a BF16 intermediate that the rope kernel reads
    # back). The fused kernel keeps the normed value in FP32 through the
    # rotation and rounds once at the end, so it is strictly more accurate but
    # differs from the reference by ~1 BF16 ULP of the normed value, amplified
    # through the complex multiply. Tolerate that intermediate-rounding gap.
    rtol: Float64 = 6e-2,
    atol: Float64 = 1e-2,
) raises:
    print(
        "== run_fused_qk_rms_norm_rope dtype=",
        dtype,
        " head_size=",
        head_size,
        " rope_dim=",
        rope_dim,
        " num_q_heads=",
        num_q_heads,
        " num_k_heads=",
        num_k_heads,
    )

    comptime kv_params = KVCacheStaticParams(
        num_heads=num_k_heads, head_size=head_size
    )
    comptime num_paged_blocks = 32
    comptime page_size = 128
    comptime num_layers = 1
    comptime layer_idx = 0
    comptime max_seq_len = 1024
    comptime weight_offset = 1.0
    var epsilon = Float32(1e-6)

    var prompt_lens = [16, 24, 8, 32]
    var cache_lens = [0, 7, 13, 5]
    var batch_size = len(prompt_lens)

    var total_length = 0
    var max_cache_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        total_length += prompt_lens[i]
        max_cache_length = max(max_cache_length, cache_lens[i])
        max_full_context_length = max(
            max_full_context_length, cache_lens[i] + prompt_lens[i]
        )
        max_prompt_length = max(max_prompt_length, prompt_lens[i])

    comptime cache_lengths_layout = Layout.row_major(UNKNOWN_VALUE)
    comptime kv_block_layout = Layout.row_major(
        UNKNOWN_VALUE,
        2,
        UNKNOWN_VALUE,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    comptime paged_lut_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var row_offsets_tile_layout = row_major(batch_size + 1)
    comptime freqs_tile_layout = row_major[max_seq_len, rope_dim]()

    var row_offsets_shape = Index(batch_size + 1)
    var cache_lengths_shape = Index(batch_size)
    var q_ragged_shape = IndexList[3](total_length, num_q_heads, head_size)
    var kv_block_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var paged_lut_shape = IndexList[2](
        batch_size, ceildiv(max_full_context_length, page_size)
    )
    var freqs_shape = IndexList[2](max_seq_len, rope_dim)

    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    var q_in_device = ctx.enqueue_create_buffer[dtype](
        q_ragged_shape.flattened_length()
    )
    var q_norm_ref_device = ctx.enqueue_create_buffer[dtype](
        q_ragged_shape.flattened_length()
    )
    var q_rope_ref_device = ctx.enqueue_create_buffer[dtype](
        q_ragged_shape.flattened_length()
    )
    var q_fused_device = ctx.enqueue_create_buffer[dtype](
        q_ragged_shape.flattened_length()
    )
    var gamma_q_device = ctx.enqueue_create_buffer[dtype](head_size)
    var gamma_k_device = ctx.enqueue_create_buffer[dtype](head_size)
    var kv_block_ref_device = ctx.enqueue_create_buffer[dtype](
        kv_block_shape.flattened_length()
    )
    var kv_block_fused_device = ctx.enqueue_create_buffer[dtype](
        kv_block_shape.flattened_length()
    )
    var paged_lut_device = ctx.enqueue_create_buffer[.uint32](
        paged_lut_shape.flattened_length()
    )
    var freqs_device = ctx.enqueue_create_buffer[dtype](
        freqs_shape.flattened_length()
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    var offset = 0
    for i in range(batch_size):
        row_offsets_host[i] = UInt32(offset)
        cache_lengths_host[i] = UInt32(cache_lens[i])
        offset += prompt_lens[i]
    row_offsets_host[batch_size] = UInt32(offset)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    comptime q_ragged_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, head_size
    )
    var q_ragged_runtime_layout = RuntimeLayout[q_ragged_layout].row_major(
        q_ragged_shape
    )
    with q_in_device.map_to_host() as q_in_host:
        var q_in_tensor = LayoutTensor[dtype, q_ragged_layout](
            q_in_host, q_ragged_runtime_layout
        )
        random(q_in_tensor)

    comptime gamma_layout = Layout.row_major(head_size)
    var gamma_runtime_layout = RuntimeLayout[gamma_layout].row_major(
        Index(head_size)
    )
    with gamma_q_device.map_to_host() as gamma_q_host:
        var gamma_q_tensor = LayoutTensor[dtype, gamma_layout](
            gamma_q_host, gamma_runtime_layout
        )
        random(gamma_q_tensor)
    with gamma_k_device.map_to_host() as gamma_k_host:
        var gamma_k_tensor = LayoutTensor[dtype, gamma_layout](
            gamma_k_host, gamma_runtime_layout
        )
        random(gamma_k_tensor)

    comptime freqs_layout = Layout.row_major(max_seq_len, rope_dim)
    var freqs_runtime_layout = RuntimeLayout[freqs_layout].row_major(
        freqs_shape
    )
    with freqs_device.map_to_host() as freqs_host:
        var freqs_init = LayoutTensor[dtype, freqs_layout](
            freqs_host, freqs_runtime_layout
        )
        random(freqs_init)

    var kv_block_runtime_layout = RuntimeLayout[kv_block_layout].row_major(
        kv_block_shape
    )
    var kv_block_host = ctx.enqueue_create_host_buffer[dtype](
        kv_block_shape.flattened_length()
    )
    var kv_block_host_tensor = LayoutTensor[dtype, kv_block_layout](
        kv_block_host.unsafe_ptr(), kv_block_runtime_layout
    )
    random(kv_block_host_tensor)
    ctx.enqueue_copy(kv_block_ref_device, kv_block_host)
    ctx.enqueue_copy(kv_block_fused_device, kv_block_host)
    ctx.synchronize()

    var paged_lut_runtime_layout = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )
    with paged_lut_device.map_to_host() as paged_lut_host:
        var paged_lut_tensor = LayoutTensor[.uint32, paged_lut_layout](
            paged_lut_host, paged_lut_runtime_layout
        )
        var paged_lut_set = Set[Int]()
        for bs in range(batch_size):
            var seq_len = cache_lens[bs] + prompt_lens[bs]
            for block_idx in range(0, ceildiv(seq_len, page_size)):
                var randval = Int(random_ui64(0, num_paged_blocks - 1))
                while randval in paged_lut_set:
                    randval = Int(random_ui64(0, num_paged_blocks - 1))
                paged_lut_set.add(randval)
                paged_lut_tensor[bs, block_idx] = UInt32(randval)

    var row_offsets_tensor = TileTensor(
        row_offsets_device, row_offsets_tile_layout
    )
    var q_in_tt = TileTensor(
        q_in_device,
        row_major((total_length, Idx[num_q_heads], Idx[head_size])),
    )
    var q_norm_ref_tt = TileTensor(
        q_norm_ref_device,
        row_major((total_length, Idx[num_q_heads], Idx[head_size])),
    )
    var q_rope_ref_tt = TileTensor(
        q_rope_ref_device,
        row_major((total_length, Idx[num_q_heads], Idx[head_size])),
    )
    var q_fused_tt = TileTensor(
        q_fused_device,
        row_major((total_length, Idx[num_q_heads], Idx[head_size])),
    )
    var gamma_q_tt = TileTensor(gamma_q_device, row_major[head_size]())
    var gamma_k_tt = TileTensor(gamma_k_device, row_major[head_size]())
    var freqs_tt = TileTensor(freqs_device, freqs_tile_layout)

    var cache_lengths_tensor = LayoutTensor[
        mut=False, .uint32, Layout(UNKNOWN_VALUE)
    ](
        cache_lengths_device,
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(cache_lengths_shape),
    )
    var paged_lut_tensor = LayoutTensor[
        mut=False, .uint32, Layout.row_major[2]()
    ](
        paged_lut_device,
        RuntimeLayout[Layout.row_major[2]()].row_major(paged_lut_shape),
    )

    var ref_kv_block_tensor = LayoutTensor[dtype, kv_block_layout](
        kv_block_ref_device, kv_block_runtime_layout
    )
    var fused_kv_block_tensor = LayoutTensor[dtype, kv_block_layout](
        kv_block_fused_device, kv_block_runtime_layout
    )
    var ref_collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        LayoutTensor[dtype, Layout.row_major[6]()](
            ref_kv_block_tensor.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                ref_kv_block_tensor.runtime_layout.shape.value.canonicalize(),
                ref_kv_block_tensor.runtime_layout.stride.value.canonicalize(),
            ),
        ),
        cache_lengths_tensor,
        paged_lut_tensor,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
    )
    var fused_collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        LayoutTensor[dtype, Layout.row_major[6]()](
            fused_kv_block_tensor.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                fused_kv_block_tensor.runtime_layout.shape.value.canonicalize(),
                fused_kv_block_tensor.runtime_layout.stride.value.canonicalize(),
            ),
        ),
        cache_lengths_tensor,
        paged_lut_tensor,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
    )

    fused_qk_rms_norm_ragged_paged[target="gpu", multiply_before_cast=True](
        q_in_tt,
        ref_collection,
        gamma_q_tt,
        gamma_k_tt,
        epsilon,
        Scalar[dtype](weight_offset),
        UInt32(layer_idx),
        row_offsets_tensor,
        q_norm_ref_tt,
        ctx,
    )
    fused_qk_rope_ragged[
        ref_collection.CacheType, interleaved=False, target="gpu"
    ](
        q_proj=q_norm_ref_tt,
        input_row_offsets=row_offsets_tensor,
        kv_collection=ref_collection,
        freqs_cis=freqs_tt,
        position_ids=None,
        layer_idx=UInt32(layer_idx),
        output=q_rope_ref_tt,
        context=ctx,
    )

    @always_inline
    @__parameter
    @__copy_capture(q_in_tt)
    def q_input_fn[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) -> SIMD[dtype, width]:
        return q_in_tt.load[width=width](Coord(Index(token, head, col)))

    fused_qk_rms_norm_rope_ragged_paged[
        target="gpu",
        multiply_before_cast=True,
        interleaved=False,
        q_input_fn=q_input_fn,
    ](
        fused_collection,
        gamma_q_tt,
        gamma_k_tt,
        freqs_tt,
        epsilon,
        Scalar[dtype](weight_offset),
        UInt32(layer_idx),
        row_offsets_tensor,
        q_fused_tt,
        ctx,
    )
    ctx.synchronize()

    print("comparing Q")
    with q_rope_ref_device.map_to_host() as q_rope_ref_host:
        with q_fused_device.map_to_host() as q_fused_host:
            var ref_t = LayoutTensor[dtype, q_ragged_layout](
                q_rope_ref_host, q_ragged_runtime_layout
            )
            var fused_t = LayoutTensor[dtype, q_ragged_layout](
                q_fused_host, q_ragged_runtime_layout
            )
            for tok in range(total_length):
                for h in range(num_q_heads):
                    for d in range(head_size):
                        assert_almost_equal(
                            fused_t[tok, h, d],
                            ref_t[tok, h, d],
                            rtol=rtol,
                            atol=atol,
                        )

    ctx.enqueue_copy(kv_block_host, kv_block_ref_device)
    var kv_block_fused_host = ctx.enqueue_create_host_buffer[dtype](
        kv_block_shape.flattened_length()
    )
    ctx.enqueue_copy(kv_block_fused_host, kv_block_fused_device)
    var paged_lut_host_ptr = ctx.enqueue_create_host_buffer[.uint32](
        paged_lut_shape.flattened_length()
    )
    ctx.enqueue_copy(paged_lut_host_ptr, paged_lut_device)
    ctx.synchronize()

    var ref_kv_host_tensor = LayoutTensor[dtype, kv_block_layout](
        kv_block_host.unsafe_ptr(), kv_block_runtime_layout
    )
    var fused_kv_host_tensor = LayoutTensor[dtype, kv_block_layout](
        kv_block_fused_host.unsafe_ptr(), kv_block_runtime_layout
    )
    var paged_lut_host_tensor = LayoutTensor[.uint32, paged_lut_layout](
        paged_lut_host_ptr.unsafe_ptr(), paged_lut_runtime_layout
    )

    print("comparing K")
    for bs in range(batch_size):
        var cache_len = cache_lens[bs]
        for tok in range(prompt_lens[bs]):
            var ctx_pos = cache_len + tok
            var block = paged_lut_host_tensor[bs, ctx_pos // page_size]
            var in_page = ctx_pos % page_size
            for h in range(kv_params.num_heads):
                for d in range(head_size):
                    # kv_idx 0 == key cache.
                    assert_almost_equal(
                        fused_kv_host_tensor[
                            Int(block), 0, layer_idx, in_page, h, d
                        ],
                        ref_kv_host_tensor[
                            Int(block), 0, layer_idx, in_page, h, d
                        ],
                        rtol=rtol,
                        atol=atol,
                    )

    # FIXME(MSTDL-2742): HostBuffer is origin incorrect.
    _ = Pointer(to=kv_block_host).as_unsafe_any_origin()[]
    _ = Pointer(to=kv_block_fused_host).as_unsafe_any_origin()[]

    _ = row_offsets_device^
    _ = cache_lengths_device^
    _ = q_in_device^
    _ = q_norm_ref_device^
    _ = q_rope_ref_device^
    _ = q_fused_device^
    _ = gamma_q_device^
    _ = gamma_k_device^
    _ = kv_block_ref_device^
    _ = kv_block_fused_device^
    _ = paged_lut_device^
    _ = freqs_device^


def run_fused_dual_qk_rms_norm_rope[
    dtype: DType,
    head_size: Int,
    rope_dim: Int,
    main_q_heads: Int,
    main_k_heads: Int,
    index_q_heads: Int,
    index_k_heads: Int,
](ctx: DeviceContext) raises:
    """Bit-exact check of the dual (main + indexer) fused RMSNorm+RoPE launch.

    Runs the single-QK kernel twice (main band, then indexer band) to produce
    the reference Q outputs and K caches, then runs the dual kernel once. The
    dual kernel shares the single kernel's per-row helper, so the two paths must
    be byte-identical; this verifies the four-band grid dispatch routes every
    (token, head) to the correct band cache / gamma / output / epsilon. The two
    caches use different KV-head counts, so `main_cache_t != index_cache_t`,
    exercising the two-`cache_t` instantiation the M3 sparse path needs.
    """
    print(
        "== run_fused_dual_qk_rms_norm_rope dtype=",
        dtype,
        " head_size=",
        head_size,
        " rope_dim=",
        rope_dim,
        " main=",
        main_q_heads,
        "/",
        main_k_heads,
        " index=",
        index_q_heads,
        "/",
        index_k_heads,
    )

    comptime main_kv_params = KVCacheStaticParams(
        num_heads=main_k_heads, head_size=head_size
    )
    comptime index_kv_params = KVCacheStaticParams(
        num_heads=index_k_heads, head_size=head_size
    )
    comptime num_paged_blocks = 32
    comptime page_size = 128
    comptime num_layers = 1
    comptime layer_idx = 0
    comptime max_seq_len = 1024
    comptime weight_offset = 1.0
    # Distinct per-band epsilons to verify the dual kernel routes each band's
    # epsilon rather than sharing one.
    var main_epsilon = Float32(1e-6)
    var index_epsilon = Float32(1e-5)
    comptime interleaved = False

    var prompt_lens = [16, 24, 8, 32]
    var cache_lens = [0, 7, 13, 5]
    var batch_size = len(prompt_lens)

    var total_length = 0
    var max_cache_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        total_length += prompt_lens[i]
        max_cache_length = max(max_cache_length, cache_lens[i])
        max_full_context_length = max(
            max_full_context_length, cache_lens[i] + prompt_lens[i]
        )
        max_prompt_length = max(max_prompt_length, prompt_lens[i])

    comptime main_kv_block_layout = Layout.row_major(
        UNKNOWN_VALUE,
        2,
        UNKNOWN_VALUE,
        page_size,
        main_kv_params.num_heads,
        main_kv_params.head_size,
    )
    comptime index_kv_block_layout = Layout.row_major(
        UNKNOWN_VALUE,
        2,
        UNKNOWN_VALUE,
        page_size,
        index_kv_params.num_heads,
        index_kv_params.head_size,
    )
    comptime paged_lut_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var main_kv_block_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        main_kv_params.num_heads,
        main_kv_params.head_size,
    )
    var index_kv_block_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        index_kv_params.num_heads,
        index_kv_params.head_size,
    )
    var paged_lut_shape = IndexList[2](
        batch_size, ceildiv(max_full_context_length, page_size)
    )
    var freqs_shape = IndexList[2](max_seq_len, rope_dim)

    # Shared metadata.
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    var paged_lut_device = ctx.enqueue_create_buffer[.uint32](
        paged_lut_shape.flattened_length()
    )
    var freqs_device = ctx.enqueue_create_buffer[dtype](
        freqs_shape.flattened_length()
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    var offset = 0
    for i in range(batch_size):
        row_offsets_host[i] = UInt32(offset)
        cache_lengths_host[i] = UInt32(cache_lens[i])
        offset += prompt_lens[i]
    row_offsets_host[batch_size] = UInt32(offset)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    comptime freqs_layout = Layout.row_major(max_seq_len, rope_dim)
    var freqs_runtime_layout = RuntimeLayout[freqs_layout].row_major(
        freqs_shape
    )
    with freqs_device.map_to_host() as freqs_host:
        var freqs_init = LayoutTensor[dtype, freqs_layout](
            freqs_host, freqs_runtime_layout
        )
        random(freqs_init)

    var paged_lut_runtime_layout = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )
    with paged_lut_device.map_to_host() as paged_lut_host:
        var paged_lut_tensor_h = LayoutTensor[.uint32, paged_lut_layout](
            paged_lut_host, paged_lut_runtime_layout
        )
        var paged_lut_set = Set[Int]()
        for bs in range(batch_size):
            var seq_len = cache_lens[bs] + prompt_lens[bs]
            for block_idx in range(0, ceildiv(seq_len, page_size)):
                var randval = Int(random_ui64(0, num_paged_blocks - 1))
                while randval in paged_lut_set:
                    randval = Int(random_ui64(0, num_paged_blocks - 1))
                paged_lut_set.add(randval)
                paged_lut_tensor_h[bs, block_idx] = UInt32(randval)

    # Per-band gammas.
    var gamma_main_q_device = ctx.enqueue_create_buffer[dtype](head_size)
    var gamma_main_k_device = ctx.enqueue_create_buffer[dtype](head_size)
    var gamma_index_q_device = ctx.enqueue_create_buffer[dtype](head_size)
    var gamma_index_k_device = ctx.enqueue_create_buffer[dtype](head_size)
    comptime gamma_layout = Layout.row_major(head_size)
    var gamma_runtime_layout = RuntimeLayout[gamma_layout].row_major(
        Index(head_size)
    )
    for gamma_dev in [
        gamma_main_q_device,
        gamma_main_k_device,
        gamma_index_q_device,
        gamma_index_k_device,
    ]:
        with gamma_dev.map_to_host() as gamma_host:
            var gamma_tensor = LayoutTensor[dtype, gamma_layout](
                gamma_host, gamma_runtime_layout
            )
            random(gamma_tensor)

    # Per-band Q inputs.
    var q_main_shape = IndexList[3](total_length, main_q_heads, head_size)
    var q_index_shape = IndexList[3](total_length, index_q_heads, head_size)
    comptime q_main_layout = Layout.row_major(
        UNKNOWN_VALUE, main_q_heads, head_size
    )
    comptime q_index_layout = Layout.row_major(
        UNKNOWN_VALUE, index_q_heads, head_size
    )
    var q_main_runtime_layout = RuntimeLayout[q_main_layout].row_major(
        q_main_shape
    )
    var q_index_runtime_layout = RuntimeLayout[q_index_layout].row_major(
        q_index_shape
    )

    var q_main_in_device = ctx.enqueue_create_buffer[dtype](
        q_main_shape.flattened_length()
    )
    var q_index_in_device = ctx.enqueue_create_buffer[dtype](
        q_index_shape.flattened_length()
    )
    with q_main_in_device.map_to_host() as q_main_in_host:
        random(
            LayoutTensor[dtype, q_main_layout](
                q_main_in_host, q_main_runtime_layout
            )
        )
    with q_index_in_device.map_to_host() as q_index_in_host:
        random(
            LayoutTensor[dtype, q_index_layout](
                q_index_in_host, q_index_runtime_layout
            )
        )

    var q_main_ref_device = ctx.enqueue_create_buffer[dtype](
        q_main_shape.flattened_length()
    )
    var q_main_fused_device = ctx.enqueue_create_buffer[dtype](
        q_main_shape.flattened_length()
    )
    var q_index_ref_device = ctx.enqueue_create_buffer[dtype](
        q_index_shape.flattened_length()
    )
    var q_index_fused_device = ctx.enqueue_create_buffer[dtype](
        q_index_shape.flattened_length()
    )

    # Per-band K caches: identical random init copied into a ref and a fused
    # buffer so the read-modify-write is comparable bit for bit.
    var main_kv_runtime_layout = RuntimeLayout[main_kv_block_layout].row_major(
        main_kv_block_shape
    )
    var index_kv_runtime_layout = RuntimeLayout[
        index_kv_block_layout
    ].row_major(index_kv_block_shape)
    var main_kv_host = ctx.enqueue_create_host_buffer[dtype](
        main_kv_block_shape.flattened_length()
    )
    var index_kv_host = ctx.enqueue_create_host_buffer[dtype](
        index_kv_block_shape.flattened_length()
    )
    random(
        LayoutTensor[dtype, main_kv_block_layout](
            main_kv_host.unsafe_ptr(), main_kv_runtime_layout
        )
    )
    random(
        LayoutTensor[dtype, index_kv_block_layout](
            index_kv_host.unsafe_ptr(), index_kv_runtime_layout
        )
    )
    var main_kv_ref_device = ctx.enqueue_create_buffer[dtype](
        main_kv_block_shape.flattened_length()
    )
    var main_kv_fused_device = ctx.enqueue_create_buffer[dtype](
        main_kv_block_shape.flattened_length()
    )
    var index_kv_ref_device = ctx.enqueue_create_buffer[dtype](
        index_kv_block_shape.flattened_length()
    )
    var index_kv_fused_device = ctx.enqueue_create_buffer[dtype](
        index_kv_block_shape.flattened_length()
    )
    ctx.enqueue_copy(main_kv_ref_device, main_kv_host)
    ctx.enqueue_copy(main_kv_fused_device, main_kv_host)
    ctx.enqueue_copy(index_kv_ref_device, index_kv_host)
    ctx.enqueue_copy(index_kv_fused_device, index_kv_host)
    ctx.synchronize()

    # TileTensors / LayoutTensors over the shared + per-band buffers.
    var row_offsets_tt = TileTensor(
        row_offsets_device, row_major(batch_size + 1)
    )
    var freqs_tt = TileTensor(freqs_device, row_major[max_seq_len, rope_dim]())
    var gamma_main_q_tt = TileTensor(
        gamma_main_q_device, row_major[head_size]()
    )
    var gamma_main_k_tt = TileTensor(
        gamma_main_k_device, row_major[head_size]()
    )
    var gamma_index_q_tt = TileTensor(
        gamma_index_q_device, row_major[head_size]()
    )
    var gamma_index_k_tt = TileTensor(
        gamma_index_k_device, row_major[head_size]()
    )
    var q_main_in_tt = TileTensor(
        q_main_in_device,
        row_major((total_length, Idx[main_q_heads], Idx[head_size])),
    )
    var q_index_in_tt = TileTensor(
        q_index_in_device,
        row_major((total_length, Idx[index_q_heads], Idx[head_size])),
    )
    var q_main_ref_tt = TileTensor(
        q_main_ref_device,
        row_major((total_length, Idx[main_q_heads], Idx[head_size])),
    )
    var q_main_fused_tt = TileTensor(
        q_main_fused_device,
        row_major((total_length, Idx[main_q_heads], Idx[head_size])),
    )
    var q_index_ref_tt = TileTensor(
        q_index_ref_device,
        row_major((total_length, Idx[index_q_heads], Idx[head_size])),
    )
    var q_index_fused_tt = TileTensor(
        q_index_fused_device,
        row_major((total_length, Idx[index_q_heads], Idx[head_size])),
    )

    var cache_lengths_tt = LayoutTensor[
        mut=False, .uint32, Layout(UNKNOWN_VALUE)
    ](
        cache_lengths_device,
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(Index(batch_size)),
    )
    var paged_lut_tt = LayoutTensor[mut=False, .uint32, Layout.row_major[2]()](
        paged_lut_device,
        RuntimeLayout[Layout.row_major[2]()].row_major(paged_lut_shape),
    )

    var main_kv_ref_t = LayoutTensor[dtype, main_kv_block_layout](
        main_kv_ref_device, main_kv_runtime_layout
    )
    var main_kv_fused_t = LayoutTensor[dtype, main_kv_block_layout](
        main_kv_fused_device, main_kv_runtime_layout
    )
    var index_kv_ref_t = LayoutTensor[dtype, index_kv_block_layout](
        index_kv_ref_device, index_kv_runtime_layout
    )
    var index_kv_fused_t = LayoutTensor[dtype, index_kv_block_layout](
        index_kv_fused_device, index_kv_runtime_layout
    )

    var main_ref_collection = PagedKVCacheCollection[
        dtype, main_kv_params, page_size
    ](
        LayoutTensor[dtype, Layout.row_major[6]()](
            main_kv_ref_t.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                main_kv_ref_t.runtime_layout.shape.value.canonicalize(),
                main_kv_ref_t.runtime_layout.stride.value.canonicalize(),
            ),
        ),
        cache_lengths_tt,
        paged_lut_tt,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
    )
    var main_fused_collection = PagedKVCacheCollection[
        dtype, main_kv_params, page_size
    ](
        LayoutTensor[dtype, Layout.row_major[6]()](
            main_kv_fused_t.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                main_kv_fused_t.runtime_layout.shape.value.canonicalize(),
                main_kv_fused_t.runtime_layout.stride.value.canonicalize(),
            ),
        ),
        cache_lengths_tt,
        paged_lut_tt,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
    )
    var index_ref_collection = PagedKVCacheCollection[
        dtype, index_kv_params, page_size
    ](
        LayoutTensor[dtype, Layout.row_major[6]()](
            index_kv_ref_t.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                index_kv_ref_t.runtime_layout.shape.value.canonicalize(),
                index_kv_ref_t.runtime_layout.stride.value.canonicalize(),
            ),
        ),
        cache_lengths_tt,
        paged_lut_tt,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
    )
    var index_fused_collection = PagedKVCacheCollection[
        dtype, index_kv_params, page_size
    ](
        LayoutTensor[dtype, Layout.row_major[6]()](
            index_kv_fused_t.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                index_kv_fused_t.runtime_layout.shape.value.canonicalize(),
                index_kv_fused_t.runtime_layout.stride.value.canonicalize(),
            ),
        ),
        cache_lengths_tt,
        paged_lut_tt,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
    )

    @always_inline
    @__parameter
    @__copy_capture(q_main_in_tt)
    def main_q_input_fn[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) -> SIMD[dtype, width]:
        return q_main_in_tt.load[width=width](Coord(Index(token, head, col)))

    @always_inline
    @__parameter
    @__copy_capture(q_index_in_tt)
    def index_q_input_fn[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) -> SIMD[dtype, width]:
        return q_index_in_tt.load[width=width](Coord(Index(token, head, col)))

    # Reference: two separate single-kernel launches.
    fused_qk_rms_norm_rope_ragged_paged[
        target="gpu",
        multiply_before_cast=True,
        interleaved=interleaved,
        q_input_fn=main_q_input_fn,
    ](
        main_ref_collection,
        gamma_main_q_tt,
        gamma_main_k_tt,
        freqs_tt,
        main_epsilon,
        Scalar[dtype](weight_offset),
        UInt32(layer_idx),
        row_offsets_tt,
        q_main_ref_tt,
        ctx,
    )
    fused_qk_rms_norm_rope_ragged_paged[
        target="gpu",
        multiply_before_cast=True,
        interleaved=interleaved,
        q_input_fn=index_q_input_fn,
    ](
        index_ref_collection,
        gamma_index_q_tt,
        gamma_index_k_tt,
        freqs_tt,
        index_epsilon,
        Scalar[dtype](weight_offset),
        UInt32(layer_idx),
        row_offsets_tt,
        q_index_ref_tt,
        ctx,
    )

    # Fused: one dual launch.
    fused_dual_qk_rms_norm_rope_ragged_paged[
        target="gpu",
        multiply_before_cast=True,
        interleaved=interleaved,
        main_q_input_fn=main_q_input_fn,
        index_q_input_fn=index_q_input_fn,
    ](
        main_fused_collection,
        index_fused_collection,
        gamma_main_q_tt,
        gamma_main_k_tt,
        gamma_index_q_tt,
        gamma_index_k_tt,
        freqs_tt,
        main_epsilon,
        index_epsilon,
        Scalar[dtype](weight_offset),
        UInt32(layer_idx),
        row_offsets_tt,
        q_main_fused_tt,
        q_index_fused_tt,
        ctx,
    )
    ctx.synchronize()

    print("comparing Q (main)")
    with q_main_ref_device.map_to_host() as ref_host:
        with q_main_fused_device.map_to_host() as fused_host:
            var ref_t = LayoutTensor[dtype, q_main_layout](
                ref_host, q_main_runtime_layout
            )
            var fused_t = LayoutTensor[dtype, q_main_layout](
                fused_host, q_main_runtime_layout
            )
            for tok in range(total_length):
                for h in range(main_q_heads):
                    for d in range(head_size):
                        assert_equal(fused_t[tok, h, d], ref_t[tok, h, d])

    print("comparing Q (index)")
    with q_index_ref_device.map_to_host() as ref_host:
        with q_index_fused_device.map_to_host() as fused_host:
            var ref_t = LayoutTensor[dtype, q_index_layout](
                ref_host, q_index_runtime_layout
            )
            var fused_t = LayoutTensor[dtype, q_index_layout](
                fused_host, q_index_runtime_layout
            )
            for tok in range(total_length):
                for h in range(index_q_heads):
                    for d in range(head_size):
                        assert_equal(fused_t[tok, h, d], ref_t[tok, h, d])

    var paged_lut_check = ctx.enqueue_create_host_buffer[.uint32](
        paged_lut_shape.flattened_length()
    )
    ctx.enqueue_copy(paged_lut_check, paged_lut_device)
    ctx.synchronize()
    var paged_lut_check_t = LayoutTensor[.uint32, paged_lut_layout](
        paged_lut_check.unsafe_ptr(), paged_lut_runtime_layout
    )

    print("comparing K (main)")
    var main_kv_ref_out = ctx.enqueue_create_host_buffer[dtype](
        main_kv_block_shape.flattened_length()
    )
    var main_kv_fused_out = ctx.enqueue_create_host_buffer[dtype](
        main_kv_block_shape.flattened_length()
    )
    ctx.enqueue_copy(main_kv_ref_out, main_kv_ref_device)
    ctx.enqueue_copy(main_kv_fused_out, main_kv_fused_device)
    ctx.synchronize()
    var main_kv_ref_out_t = LayoutTensor[dtype, main_kv_block_layout](
        main_kv_ref_out.unsafe_ptr(), main_kv_runtime_layout
    )
    var main_kv_fused_out_t = LayoutTensor[dtype, main_kv_block_layout](
        main_kv_fused_out.unsafe_ptr(), main_kv_runtime_layout
    )
    for bs in range(batch_size):
        var cache_len = cache_lens[bs]
        for tok in range(prompt_lens[bs]):
            var ctx_pos = cache_len + tok
            var block = paged_lut_check_t[bs, ctx_pos // page_size]
            var in_page = ctx_pos % page_size
            for h in range(main_kv_params.num_heads):
                for d in range(head_size):
                    assert_equal(
                        main_kv_fused_out_t[
                            Int(block), 0, layer_idx, in_page, h, d
                        ],
                        main_kv_ref_out_t[
                            Int(block), 0, layer_idx, in_page, h, d
                        ],
                    )

    print("comparing K (index)")
    var index_kv_ref_out = ctx.enqueue_create_host_buffer[dtype](
        index_kv_block_shape.flattened_length()
    )
    var index_kv_fused_out = ctx.enqueue_create_host_buffer[dtype](
        index_kv_block_shape.flattened_length()
    )
    ctx.enqueue_copy(index_kv_ref_out, index_kv_ref_device)
    ctx.enqueue_copy(index_kv_fused_out, index_kv_fused_device)
    ctx.synchronize()
    var index_kv_ref_out_t = LayoutTensor[dtype, index_kv_block_layout](
        index_kv_ref_out.unsafe_ptr(), index_kv_runtime_layout
    )
    var index_kv_fused_out_t = LayoutTensor[dtype, index_kv_block_layout](
        index_kv_fused_out.unsafe_ptr(), index_kv_runtime_layout
    )
    for bs in range(batch_size):
        var cache_len = cache_lens[bs]
        for tok in range(prompt_lens[bs]):
            var ctx_pos = cache_len + tok
            var block = paged_lut_check_t[bs, ctx_pos // page_size]
            var in_page = ctx_pos % page_size
            for h in range(index_kv_params.num_heads):
                for d in range(head_size):
                    assert_equal(
                        index_kv_fused_out_t[
                            Int(block), 0, layer_idx, in_page, h, d
                        ],
                        index_kv_ref_out_t[
                            Int(block), 0, layer_idx, in_page, h, d
                        ],
                    )

    # FIXME(MSTDL-2742): HostBuffer is origin incorrect.
    _ = Pointer(to=main_kv_host).as_unsafe_any_origin()[]
    _ = Pointer(to=index_kv_host).as_unsafe_any_origin()[]
    _ = Pointer(to=main_kv_ref_out).as_unsafe_any_origin()[]
    _ = Pointer(to=main_kv_fused_out).as_unsafe_any_origin()[]
    _ = Pointer(to=index_kv_ref_out).as_unsafe_any_origin()[]
    _ = Pointer(to=index_kv_fused_out).as_unsafe_any_origin()[]

    # Re-run both ops with an FP8 main-Q output and compare byte for byte; the
    # bf16 equality above misses that threading. Must stay last: these launches
    # re-rope the shared K caches in place -- Q reproduces exactly, K does not.
    print("comparing Q (main) at an FP8 output dtype: dual vs single")
    comptime q_out_dtype = DType.float8_e4m3fn
    var q_main_single_fp8_device = ctx.enqueue_create_buffer[q_out_dtype](
        q_main_shape.flattened_length()
    )
    var q_main_dual_fp8_device = ctx.enqueue_create_buffer[q_out_dtype](
        q_main_shape.flattened_length()
    )
    var q_main_single_fp8_tt = TileTensor(
        q_main_single_fp8_device,
        row_major((total_length, Idx[main_q_heads], Idx[head_size])),
    )
    var q_main_dual_fp8_tt = TileTensor(
        q_main_dual_fp8_device,
        row_major((total_length, Idx[main_q_heads], Idx[head_size])),
    )

    fused_qk_rms_norm_rope_ragged_paged[
        target="gpu",
        multiply_before_cast=True,
        interleaved=interleaved,
        q_input_fn=main_q_input_fn,
    ](
        main_ref_collection,
        gamma_main_q_tt,
        gamma_main_k_tt,
        freqs_tt,
        main_epsilon,
        Scalar[dtype](weight_offset),
        UInt32(layer_idx),
        row_offsets_tt,
        q_main_single_fp8_tt,
        ctx,
    )
    fused_dual_qk_rms_norm_rope_ragged_paged[
        target="gpu",
        multiply_before_cast=True,
        interleaved=interleaved,
        main_q_input_fn=main_q_input_fn,
        index_q_input_fn=index_q_input_fn,
    ](
        main_fused_collection,
        index_fused_collection,
        gamma_main_q_tt,
        gamma_main_k_tt,
        gamma_index_q_tt,
        gamma_index_k_tt,
        freqs_tt,
        main_epsilon,
        index_epsilon,
        Scalar[dtype](weight_offset),
        UInt32(layer_idx),
        row_offsets_tt,
        q_main_dual_fp8_tt,
        q_index_fused_tt,
        ctx,
    )
    ctx.synchronize()

    var q_main_numel = q_main_shape.flattened_length()
    with q_main_single_fp8_device.map_to_host() as single_host:
        with q_main_dual_fp8_device.map_to_host() as dual_host:
            # Raw bytes: an FP8 NaN never compares equal to itself.
            var single_u8 = single_host.unsafe_ptr().bitcast[UInt8]()
            var dual_u8 = dual_host.unsafe_ptr().bitcast[UInt8]()
            var live = 0
            for i in range(q_main_numel):
                assert_equal(dual_u8[i], single_u8[i])
                # Both arms share the cast, so an out-of-range Q gives the same
                # byte on both sides -- the compare above is blind to it. A
                # finite out-of-range value saturates on both vendors (NVPTX
                # lowers to `cvt.rn.satfinite`, AMDGPU clamps before
                # `cvt.pk.fp8.f32`), so the tell is a value pinned at the
                # limit. Testing `>= 0x7E` catches NaN (0x7F) too.
                if (single_u8[i] & 0x7F) >= 0x7E:
                    raise Error(
                        String(
                            "FP8 Q sits at the e4m3 limit at element ",
                            i,
                            (
                                ": a value left the representable range and"
                                " both arms round it the same way"
                            ),
                        )
                    )
                # 0x00/0x80 are +/-0; two all-zero buffers would compare equal
                # while proving nothing.
                if (single_u8[i] & 0x7F) != 0:
                    live += 1
            if live * 2 < q_main_numel:
                raise Error(
                    String(
                        "FP8 Q comparison is near-vacuous: only ",
                        live,
                        " of ",
                        q_main_numel,
                        " bytes are nonzero",
                    )
                )

    _ = q_main_single_fp8_device^
    _ = q_main_dual_fp8_device^
    _ = row_offsets_device^
    _ = cache_lengths_device^
    _ = paged_lut_device^
    _ = freqs_device^
    _ = gamma_main_q_device^
    _ = gamma_main_k_device^
    _ = gamma_index_q_device^
    _ = gamma_index_k_device^
    _ = q_main_in_device^
    _ = q_index_in_device^
    _ = q_main_ref_device^
    _ = q_main_fused_device^
    _ = q_index_ref_device^
    _ = q_index_fused_device^
    _ = main_kv_ref_device^
    _ = main_kv_fused_device^
    _ = index_kv_ref_device^
    _ = index_kv_fused_device^


def main() raises:
    with DeviceContext() as ctx:
        run_fused_qk_rms_norm_rope[
            DType.bfloat16,
            head_size=128,
            rope_dim=128,
            num_q_heads=64,
            num_k_heads=4,
        ](ctx)
        run_fused_qk_rms_norm_rope[
            DType.bfloat16,
            head_size=128,
            rope_dim=64,
            num_q_heads=4,
            num_k_heads=1,
        ](ctx)
        # Dual (main + indexer) fusion, bit-exact vs two single launches.
        # Main and indexer use different KV-head counts (distinct cache_t) and
        # the M3 sparse geometry: head_dim=128, rope_dim=64, non-interleaved.
        run_fused_dual_qk_rms_norm_rope[
            DType.bfloat16,
            head_size=128,
            rope_dim=64,
            main_q_heads=16,
            main_k_heads=4,
            index_q_heads=4,
            index_k_heads=1,
        ](ctx)
