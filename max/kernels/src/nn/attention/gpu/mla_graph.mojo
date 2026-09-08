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

"""Provides manually fused GPU graph kernels for Multi-head Latent Attention (MLA).

Defines the fused RoPE and RMSNorm kernel, the FP8 prefill and decode branches,
and the combined prefill-decode graph that up-projects the latent KV cache and
performs MLA attention with dynamic FP8 scaling.
"""


from std.collections import Optional, OptionalReg
from std.math import align_up, ceildiv

from std.sys import simd_width_of, size_of
from std.sys.info import align_of
from std.utils.index import Index, IndexList

from max.algorithm.functional import _elementwise_impl_gpu
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    global_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.info import _is_sm10x_gpu
from std.utils.coord import Coord, Idx, coord_to_index_list
from layout import (
    Coord,
    Idx,
    TensorLayout,
    TileTensor,
    row_major,
)
from layout.layout import *
from layout.tile_layout import Layout as TileLayout
from linalg.bmm import _batched_matmul_gpu, batched_matmul_dynamic_scaled_fp8
from linalg.matmul import matmul
from std.utils.index import StaticTuple
from std.utils.numerics import get_accum_type
from linalg.fp8_quantization import (
    matmul_dynamic_scaled_fp8,
    quantize_dynamic_scaled_fp8,
    batched_quantize_dynamic_scaled_fp8,
)
from internal_utils.fp8_utils import cast_saturating
from nn._ragged_utils import get_batch_and_token_idx_from_row_offsets
from nn.fused_qk_rope import rope_k_cache, rope_q_proj, rope_value
from nn.kv_cache import KVCollectionT, KVCacheT
from nn.kv_cache_ragged import (
    generic_flare_mla_decode_kv_cache_ragged,
    generic_flare_mla_prefill_kv_cache_ragged,
)
from nn.attention.gpu.mla import _k_cache_to_buffer, mla_decode_max_seq_len
from nn.attention.gpu.nvidia.sm100.mla_prefill import (
    mla_sm100_prefill_sparse,
    mla_sm100_prefill_sparse_fp8,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_qkv_fp8 import (
    mla_prefill_sparse_qkv_fp8,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_utils import (
    MLASparseConfig,
)
from nn.normalization import _rms_norm_warp_tiling_subkernel


# Manually fused MLA RoPE and RMSNorm kernel
# ===-----------------------------------------------------------------------===#


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
@__name(t"fused_rope_rmsnorm_{dtype}")
def fused_rope_rmsnorm_kernel[
    dtype: DType,
    freq_dtype: DType,
    gamma_dtype: DType,
    QRopeOutputLayoutType: TensorLayout,
    QRopeLayoutType: TensorLayout,
    InputRowOffsetsLayoutType: TensorLayout,
    FreqsCisLayoutType: TensorLayout,
    GammaLayoutType: TensorLayout,
    cache_t: KVCacheT,
    block_size: Int,
    n_rope_blocks: Int,
    n_rms_blocks: Int,
](
    q_rope_output: TileTensor[
        mut=True, dtype, QRopeOutputLayoutType, MutUntrackedOrigin
    ],
    q_rope: TileTensor[dtype, QRopeLayoutType, ImmUntrackedOrigin],
    input_row_offsets: TileTensor[
        .uint32, InputRowOffsetsLayoutType, ImmUntrackedOrigin
    ],
    freqs_cis: TileTensor[freq_dtype, FreqsCisLayoutType, ImmUntrackedOrigin],
    gamma: TileTensor[gamma_dtype, GammaLayoutType, ImmUntrackedOrigin],
    k_cache: cache_t,
    epsilon: Float32,
) -> None:
    """Fused GPU kernel that applies RoPE to query projections and RMSNorm to KV
    cache.

    This kernel processes tokens in parallel across GPU blocks, with separate
    block groups handling RoPE and RMSNorm operations. The RoPE blocks apply
    rotary position embeddings to both the query rope part (in-place) and the
    key cache rope part (in-place). The RMSNorm blocks normalize the first
    `kv_norm_dim` elements of the key cache entries.

    Parameters:
        dtype: Data type of query tensors.
        freq_dtype: Data type of frequency cosine/sine values.
        gamma_dtype: Data type of RMSNorm gamma weights.
        QRopeOutputLayoutType: Layout types of the output query rope tensor.
        QRopeLayoutType: Layout types of the input query rope tensor.
        InputRowOffsetsLayoutType: Layout types of the row offset indices tensor.
        FreqsCisLayoutType: Layout types of the frequency tensor.
        GammaLayoutType: Layout types of the gamma weights tensor.
        cache_t: Type of the KV cache.
        block_size: Number of threads per block.
        n_rope_blocks: Number of blocks allocated for RoPE computation.
        n_rms_blocks: Number of blocks allocated for RMSNorm computation.

    Args:
        q_rope_output: Output tensor for RoPE-applied query projections.
            Shape: [tot_seq_len, num_heads, rope_dim].
        q_rope: Input query rope projections. Shape: [tot_seq_len, num_heads, rope_dim].
        input_row_offsets: Row offsets indicating request boundaries.
            Shape: [num_batches + 1].
        freqs_cis: Precomputed RoPE frequency values. Shape: [max_seq_len, rope_dim].
        gamma: RMSNorm gamma weights. Shape: [kv_norm_dim].
        k_cache: Key cache to apply RoPE and RMSNorm to.
        epsilon: Small constant for numerical stability in RMSNorm.
    """
    comptime assert (
        cache_t.kv_params.num_heads == 1
    ), "num_heads should be 1 for MLA"
    comptime assert q_rope_output.flat_rank == 3
    comptime assert q_rope.flat_rank == 3
    comptime assert input_row_offsets.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2
    comptime assert gamma.flat_rank == 1

    # Evidence asserts for TileTensor load/store Coord constraints.
    comptime assert (
        TileTensor[freq_dtype, FreqsCisLayoutType, ImmUntrackedOrigin].flat_rank
        >= 2
    )
    comptime assert (
        TileTensor[gamma_dtype, GammaLayoutType, ImmUntrackedOrigin].flat_rank
        >= 1
    )

    comptime num_q_heads = q_rope.static_shape[1]
    comptime rope_dim = q_rope.static_shape[2]
    comptime kv_norm_dim = gamma.static_shape[0]

    var worker_idx = block_idx.y
    var num_workers = grid_dim.y
    var num_tokens = q_rope.dim(0)

    with PDL():
        for global_token_idx in range(worker_idx, Int(num_tokens), num_workers):
            var batch_idx, token_idx = get_batch_and_token_idx_from_row_offsets(
                input_row_offsets, global_token_idx
            )
            var post_seq_idx = k_cache.cache_length(batch_idx) + token_idx

            # First n_rope_blocks blocks of this worker process RoPE.
            if block_idx.x < n_rope_blocks:
                comptime q_width = simd_width_of[dtype]()
                comptime assert (
                    rope_dim % q_width == 0
                ), "rope_dim should be divisible by q_width"

                var head_idx, head_dim_idx = divmod(
                    global_idx.x * q_width, rope_dim
                )
                var f_c = freqs_cis.load[width=q_width](
                    (post_seq_idx, head_dim_idx)
                )

                if head_idx < num_q_heads:
                    rope_q_proj[interleaved=True](
                        q_rope,
                        q_rope_output,
                        Index(global_token_idx, head_idx, head_dim_idx),
                        f_c,
                        rope_dim,
                    )
                elif head_idx == num_q_heads:
                    rope_k_cache[interleaved=True](
                        k_cache,
                        batch_idx,
                        0,  # num_k_heads is 1 for MLA
                        post_seq_idx,
                        head_dim_idx + kv_norm_dim,
                        f_c,
                        rope_dim,
                    )

            # The last block of this worker processes RMSNorm.
            else:
                comptime k_dtype = cache_t.dtype
                comptime k_width = simd_width_of[k_dtype]()
                comptime accum_type = get_accum_type[k_dtype]()
                comptime warps_per_block = block_size // WARP_SIZE

                comptime assert (
                    kv_norm_dim % k_width == 0
                ), "kv_norm_dim should be divisible by k_width"

                var vec_data = SIMD[accum_type, k_width](0)
                var gamma_val = SIMD[gamma_dtype, k_width](0)

                var idx = thread_idx.x * k_width
                if idx < kv_norm_dim:
                    vec_data = k_cache.load[width=k_width](
                        batch_idx,
                        0,  # num_k_heads is 1 for MLA
                        post_seq_idx,
                        idx,
                    ).cast[accum_type]()
                    # Prefetch gamma before reduction.
                    gamma_val = gamma.load[
                        width=k_width,
                        alignment=align_of[SIMD[gamma_dtype, k_width]](),
                    ](Coord(idx))

                var norm_val = _rms_norm_warp_tiling_subkernel[
                    warps_per_block,
                    False,  # Do not multiply the gamma before casting.
                ](
                    global_token_idx,
                    idx,
                    vec_data,
                    gamma_val,
                    epsilon,
                    0.0,
                    kv_norm_dim,
                )

                if idx < kv_norm_dim:
                    k_cache.store(
                        batch_idx,
                        0,  # num_k_heads is 1 for MLA
                        post_seq_idx,
                        idx,
                        cast_saturating[k_dtype](norm_val),
                    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
@__name(t"fused_rope_rmsnorm_quantization_{dtype}_{out_rope_dtype}")
def fused_rope_rmsnorm_quantization_kernel[
    dtype: DType,
    freq_dtype: DType,
    gamma_dtype: DType,
    QRopeOutputLayoutType: TensorLayout,
    QRopeLayoutType: TensorLayout,
    InputRowOffsetsLayoutType: TensorLayout,
    FreqsCisLayoutType: TensorLayout,
    GammaLayoutType: TensorLayout,
    cache_t: KVCacheT,
    block_size: Int,
    n_rope_blocks: Int,
    n_rms_blocks: Int,
    out_rope_dtype: DType,
    kv_input_fn: def[width: Int](IndexList[2]) capturing -> SIMD[
        DType.bfloat16, width
    ],
](
    q_rope_output: TileTensor[
        mut=True, out_rope_dtype, QRopeOutputLayoutType, MutUntrackedOrigin
    ],
    q_rope: TileTensor[dtype, QRopeLayoutType, ImmUntrackedOrigin],
    input_row_offsets: TileTensor[
        .uint32, InputRowOffsetsLayoutType, ImmUntrackedOrigin
    ],
    freqs_cis: TileTensor[freq_dtype, FreqsCisLayoutType, ImmUntrackedOrigin],
    gamma: TileTensor[gamma_dtype, GammaLayoutType, ImmUntrackedOrigin],
    k_cache: cache_t,
    epsilon: Float32,
) -> None:
    """Fused GPU kernel that applies RoPE to query projections and RMSNorm to KV
    cache, reading the inputs from a KV buffer and quantizing the final results
    before writing to the KVCache object.

    This kernel processes tokens in parallel across GPU blocks, with separate
    block groups handling RoPE and RMSNorm operations. The RoPE blocks apply
    rotary position embeddings to both the query rope part (in-place) and the
    key cache rope part (in-place). The RMSNorm blocks normalize the first
    `kv_norm_dim` elements of the key cache entries.

    Parameters:
        dtype: Data type of query tensors.
        freq_dtype: Data type of frequency cosine/sine values.
        gamma_dtype: Data type of RMSNorm gamma weights.
        QRopeOutputLayoutType: Layout types of the output query rope tensor.
        QRopeLayoutType: Layout types of the input query rope tensor.
        InputRowOffsetsLayoutType: Layout types of the row offset indices tensor.
        FreqsCisLayoutType: Layout types of the frequency tensor.
        GammaLayoutType: Layout types of the gamma weights tensor.
        cache_t: Type of the KV cache.
        block_size: Number of threads per block.
        n_rope_blocks: Number of blocks allocated for RoPE computation.
        n_rms_blocks: Number of blocks allocated for RMSNorm computation.
        out_rope_dtype: Data type of the RoPE output.
        kv_input_fn: Input lambda function to load the KV latent values. Shape:
            [tot_seq_len, cache_head_dim]. Where cache_head_dim = kv_lora_rank
            + qk_rope_head_dim.

    Args:
        q_rope_output: Output tensor for RoPE-applied query projections.
            Shape: [tot_seq_len, num_heads, rope_dim].
        q_rope: Input query rope projections. Shape: [tot_seq_len, num_heads, rope_dim].
        input_row_offsets: Row offsets indicating request boundaries.
            Shape: [num_batches + 1].
        freqs_cis: Precomputed RoPE frequency values. Shape: [max_seq_len, rope_dim].
        gamma: RMSNorm gamma weights. Shape: [kv_norm_dim].
        k_cache: Key cache to apply RoPE and RMSNorm to.
        epsilon: Small constant for numerical stability in RMSNorm.
    """
    comptime assert (
        cache_t.kv_params.num_heads == 1
    ), "num_heads should be 1 for MLA"
    comptime assert q_rope_output.flat_rank == 3
    comptime assert q_rope.flat_rank == 3
    comptime assert input_row_offsets.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2
    comptime assert gamma.flat_rank == 1

    # Evidence asserts for TileTensor load/store Coord constraints.
    comptime assert (
        TileTensor[freq_dtype, FreqsCisLayoutType, ImmUntrackedOrigin].flat_rank
        >= 2
    )
    comptime assert (
        TileTensor[gamma_dtype, GammaLayoutType, ImmUntrackedOrigin].flat_rank
        >= 1
    )

    comptime num_q_heads = q_rope.static_shape[1]
    comptime rope_dim = q_rope.static_shape[2]
    comptime kv_norm_dim = gamma.static_shape[0]
    comptime cache_dtype = cache_t.dtype
    comptime k_width = simd_width_of[dtype]()

    var worker_idx = block_idx.y
    var num_workers = grid_dim.y
    var num_tokens = q_rope.dim(0)

    with PDL():
        for global_token_idx in range(worker_idx, Int(num_tokens), num_workers):
            var batch_idx, token_idx = get_batch_and_token_idx_from_row_offsets(
                input_row_offsets, global_token_idx
            )
            var post_seq_idx = k_cache.cache_length(batch_idx) + token_idx

            # First n_rope_blocks blocks of this worker process RoPE.
            if block_idx.x < n_rope_blocks:
                comptime q_width = simd_width_of[dtype]()
                comptime assert (
                    rope_dim % q_width == 0
                ), "rope_dim should be divisible by q_width"

                var head_idx, head_dim_idx = divmod(
                    global_idx.x * q_width, rope_dim
                )
                var f_c = freqs_cis.load[width=q_width](
                    (post_seq_idx, head_dim_idx)
                )

                if head_idx < num_q_heads:
                    # One `alignment` covers both the load and the store, so it
                    # must hold for the narrower of the two element types.
                    rope_q_proj[
                        interleaved=True,
                        alignment=align_of[SIMD[out_rope_dtype, q_width]](),
                    ](
                        q_rope,
                        q_rope_output,
                        Index(global_token_idx, head_idx, head_dim_idx),
                        f_c,
                        rope_dim,
                    )
                elif head_idx == num_q_heads:
                    var val = kv_input_fn[width=k_width](
                        {global_token_idx, kv_norm_dim + head_dim_idx}
                    )
                    var roped_val = rope_value(val, f_c).cast[dtype]()
                    k_cache.store(
                        batch_idx,
                        0,  # num_k_heads is 1 for MLA
                        post_seq_idx,
                        head_dim_idx + kv_norm_dim,
                        cast_saturating[cache_dtype](roped_val),
                    )

            # The last block of this worker processes RMSNorm.
            else:
                comptime accum_type = get_accum_type[dtype]()
                comptime warps_per_block = block_size // WARP_SIZE

                comptime assert (
                    kv_norm_dim % k_width == 0
                ), "kv_norm_dim should be divisible by k_width"

                var vec_data = SIMD[accum_type, k_width](0)
                var gamma_val = SIMD[gamma_dtype, k_width](0)

                var idx = thread_idx.x * k_width
                if idx < kv_norm_dim:
                    vec_data = kv_input_fn[width=k_width](
                        {global_token_idx, idx}
                    ).cast[accum_type]()
                    # Prefetch gamma before reduction.
                    gamma_val = gamma.load[
                        width=k_width,
                        alignment=align_of[SIMD[gamma_dtype, k_width]](),
                    ](Coord(idx))

                var norm_val = _rms_norm_warp_tiling_subkernel[
                    warps_per_block,
                    False,  # Do not multiply the gamma before casting.
                ](
                    global_token_idx,
                    idx,
                    vec_data,
                    gamma_val,
                    epsilon,
                    0.0,
                    kv_norm_dim,
                )

                if idx < kv_norm_dim:
                    k_cache.store(
                        batch_idx,
                        0,  # num_k_heads is 1 for MLA
                        post_seq_idx,
                        idx,
                        cast_saturating[cache_dtype](norm_val),
                    )


@always_inline
def mla_fused_rope_rmsnorm_quantization[
    dtype: DType,
    freq_dtype: DType,
    gamma_dtype: DType,
    collection_t: KVCollectionT,
    out_rope_dtype: DType,
    //,
    kv_input_fn: def[width: Int](IndexList[2]) capturing -> SIMD[
        DType.bfloat16, width
    ],
](
    q_rope_output: TileTensor[mut=True, out_rope_dtype, ...],
    q_rope: TileTensor[mut=False, dtype, ...],
    input_row_offsets: TileTensor[mut=False, .uint32, ...],
    freqs_cis: TileTensor[mut=False, freq_dtype, ...],
    gamma: TileTensor[mut=False, gamma_dtype, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    epsilon: Float32,
    ctx: DeviceContext,
) raises:
    """Launches the fused RoPE and RMSNorm kernel for MLA attention.

    This function fuses three operations:
    1. RoPE applied to query and key cache rope parts.
    2. RMSNorm applied to the non-rope portion of the key cache.
    3. Quantization of the final results before writing to the KVCache object.

    Parameters:
        dtype: Data type of query tensors.
        freq_dtype: Data type of frequency cosine/sine values.
        gamma_dtype: Data type of RMSNorm gamma weights.
        collection_t: Type of the KV cache collection.
        out_rope_dtype: Data type of the RoPE output values.
        kv_input_fn: Input lambda function to load the KV latent values. Shape:
            [tot_seq_len, cache_head_dim]. Where cache_head_dim = kv_lora_rank
            + qk_rope_head_dim.

    Args:
        q_rope_output: Output tensor for RoPE-applied query projections.
            Shape: [tot_seq_len, num_heads, rope_dim].
        q_rope: Input query rope projections. Shape: [tot_seq_len, num_heads, rope_dim].
        input_row_offsets: Row offsets indicating request boundaries.
            Shape: [num_batches + 1].
        freqs_cis: Precomputed RoPE frequency values. Shape: [max_seq_len, rope_dim].
        gamma: RMSNorm gamma weights. Shape: [kv_norm_dim].
        kv_collection: Paged KV cache collection.
        layer_idx: Index of the current transformer layer.
        epsilon: Small constant for numerical stability in RMSNorm.
        ctx: Device context for kernel execution.
    """
    comptime hw_info = ctx.default_device_info
    comptime sm_count = hw_info.sm_count
    comptime num_q_heads = q_rope.static_shape[1]
    comptime num_k_heads = 1  # Fixed to 1 for MLA.
    comptime rope_dim = q_rope.static_shape[2]
    comptime kv_norm_dim = gamma.static_shape[0]

    comptime assert (
        q_rope_output.static_shape[2] == rope_dim
    ), "q_rope_output and q_rope must have the same head_size"
    comptime assert (
        q_rope_output.rank == 3 and q_rope.rank == 3
    ), "q_rope_output and q_rope must be rank 3"
    comptime assert (
        rope_dim + kv_norm_dim == collection_t.kv_params.head_size
    ), "rope_dim + kv_norm_dim must be equal to kvcache head_size"

    # Default block size used by the `elementwise` function on Blackwell.
    comptime block_size = 128
    comptime kernel_simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime n_rope_elems = (num_q_heads + num_k_heads) * rope_dim

    # Make sure that we can use one block to process the rmsnorm.
    comptime assert kv_norm_dim <= block_size * kernel_simd_width, (
        "kv_norm_dim must be less than or equal to block_size *"
        " kernel_simd_width"
    )
    comptime n_rope_blocks = ceildiv(
        ceildiv(n_rope_elems, kernel_simd_width), block_size
    )
    comptime n_rms_blocks = 1  # Fixed to 1 for MLA.

    var max_workers = (
        sm_count
        * (hw_info.max_thread_block_size // block_size)
        // (n_rope_blocks + n_rms_blocks)
    )
    var num_workers = min(max_workers, Int(q_rope.dim(0)))

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))

    comptime kernel = fused_rope_rmsnorm_quantization_kernel[
        dtype,
        freq_dtype,
        gamma_dtype,
        q_rope_output.LayoutType,
        q_rope.LayoutType,
        input_row_offsets.LayoutType,
        freqs_cis.LayoutType,
        gamma.LayoutType,
        type_of(k_cache),
        block_size,
        n_rope_blocks,
        n_rms_blocks,
        out_rope_dtype,
        kv_input_fn,
    ]

    ctx.enqueue_function[kernel](
        q_rope_output,
        q_rope.as_immut(),
        input_row_offsets.as_immut(),
        freqs_cis.as_immut(),
        gamma.as_immut(),
        k_cache,
        epsilon,
        grid_dim=(n_rope_blocks + n_rms_blocks, num_workers, 1),
        block_dim=block_size,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


# ===-----------------------------------------------------------------------===#
# Manually fused MLA prefill branch (FP8)
# ===-----------------------------------------------------------------------===#


def mla_prefill_branch_fp8[
    dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    collection_t: KVCollectionT,
    //,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    mask_str: StaticString,
    kv_input_fn: def[width: Int](IndexList[2]) capturing -> SIMD[
        DType.bfloat16, width
    ],
    target: StaticString = "cpu",
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    q: TileTensor[dtype, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    freqs_cis: TileTensor[_, address_space=.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space=.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    buffer_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    cache_offsets: TileTensor[mut=True, .uint32, address_space=.GENERIC, ...],
    buffer_length: Int,
    w_k: TileTensor[fp8_dtype, address_space=.GENERIC, ...],
    w_k_scale: TileTensor[fp8_scale_dtype, address_space=.GENERIC, ...],
    w_uv: TileTensor[fp8_dtype, address_space=.GENERIC, ...],
    w_uv_scale: TileTensor[fp8_scale_dtype, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """
    This is a manually fused kernel that performs the following operations:
    - Apply RoPE to the query and the key cache (in-place).
    - Apply RMSNorm to the non-rope portion of the key cache (in-place).
    - Copy the KV latent values from PagedKVCache to a contiguous buffer.
    - Quantize the KV latent values to fp8.
    - Up-project the latent KV values to full K and V through two matmuls.
    - Perform MLA prefill.

    Parameters:
        dtype: Data type of the input and output tensors.
        fp8_dtype: Data type of the fp8 input and output tensors.
        fp8_scale_dtype: Data type of the fp8 scale input and output tensors.
        collection_t: Type of the KV collection.
        m_scale_granularity: Granularity of the scale for M dimension of the
            matrix multiplication.
        n_scale_granularity: Granularity of the scale for N dimension of the
            matrix multiplication.
        k_scale_granularity: Granularity of the scale for K dimension of the
            matrix multiplication.
        mask_str: Mask variant.
        kv_input_fn: Input lambda function to load the KV latent values. Shape:
            [tot_seq_len, cache_head_dim]. Where cache_head_dim = kv_lora_rank
            + qk_rope_head_dim.
        target: Target device.

    Args:
        output: Output tensor of shape [tot_seq_len, num_heads, v_head_dim].
        q: Combined query tensor containing both nope and rope parts. Shape:
            [tot_seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim].
        input_row_offsets: Indicates where each request starts and ends in
            `q`. Shape: [num_batches + 1].
        freqs_cis: Precomputed RoPE frequency values for rotary position
            embeddings. Shape: [max_seq_len, qk_rope_head_dim].
        kv_norm_gamma: RMSNorm gamma weights for normalizing the KV cache.
            Shape: [kv_lora_rank].
        kv_collection: Paged KV Cache object.
        layer_idx: Layer index.
        scale: Scale for the attention calculation.
        epsilon: Small constant for numerical stability in RMSNorm.
        buffer_row_offsets: Indicates where each request's KV latent values
            should be stored in the contiguous K buffer. This is a 1D tensor
            of shape [num_batches + 1].
        cache_offsets: Indicates the starting token position in the KV cache
            from which to copy KV latent values for each request. This is a 1D
            tensor of shape [num_batches + 1].
        buffer_length: The total number of tokens in the KV cache. Scalar.
        w_k: Weight matrix for up-projecting the latent cache to full K. Shape:
            [num_heads * qk_nope_head_dim, kv_latent_dim].
        w_k_scale: Scale tensor for `w_k`.
        w_uv: Weight tensor for projecting latent values to V. Shape:
            [num_heads, v_head_dim, kv_latent_dim].
        w_uv_scale: Scale tensor for `w_uv`.
        ctx: Device context.
    """
    comptime kv_params = collection_t.kv_params
    comptime assert kv_params.is_mla, "kv_params.is_mla should be true"
    comptime assert kv_params.num_heads == 1, "kv_params.num_heads should be 1"

    comptime num_heads = q.static_shape[1]
    comptime q_head_dim = q.static_shape[2]
    comptime v_head_dim = output.static_shape[2]
    comptime k_cache_dim = kv_params.head_size

    comptime assert w_k.shape_known, "w_k's shape should be static"
    comptime kv_latent_dim = w_k.static_shape[1]
    # Derive the rotary width from cache+latent, not freqs_cis, so a NoPE model
    # with no rotary table can still reach this op.
    comptime qk_rope_head_dim = k_cache_dim - kv_latent_dim
    comptime qk_nope_head_dim = q_head_dim - qk_rope_head_dim
    comptime assert (
        qk_rope_head_dim == 0 or freqs_cis.static_shape[1] == qk_rope_head_dim
    ), "freqs_cis's width should be k_cache_dim - kv_latent_dim."
    comptime assert (
        w_k.static_shape[0] == num_heads * qk_nope_head_dim
    ), "w_k.shape[0] should be equal to num_heads * qk_nope_head_dim"
    comptime assert w_uv.shape_known, "w_uv's shape should be static"
    comptime assert (
        w_uv.static_shape[0] == num_heads
    ), "w_uv.shape[0] should be equal to num_heads"
    comptime assert (
        w_uv.static_shape[1] == v_head_dim
    ), "w_uv.shape[1] should be equal to v_head_dim"
    comptime assert (
        w_uv.static_shape[2] == kv_latent_dim
    ), "w_uv.shape[2] should be equal to kv_latent_dim"

    comptime assert m_scale_granularity == 1, "m_scale_granularity should be 1"
    comptime assert (
        n_scale_granularity == k_scale_granularity
        and n_scale_granularity in (64, 128)
    ), (
        "n_scale_granularity and k_scale_granularity must be equal and in"
        " (64, 128)"
    )

    # Return early if we have no tokens to process.
    if buffer_length == 0:
        return

    var seq_len = Int(q.dim(0))

    # =========================================================================#
    # QK RoPE and K cache RMSNorm                                              #
    # =========================================================================#

    # Create a view of the `q` tensor that only contains the last
    # qk_rope_head_dim columns of each Q head.
    var q_rope = TileTensor(
        q.ptr + qk_nope_head_dim,
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * q_head_dim], Idx[q_head_dim], Idx[1]),
        ),
    )

    # In-place update of the rope part of the `q` tensor
    var q_rope_mut = TileTensor(
        q_rope.ptr.unsafe_mut_cast[True](),
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * q_head_dim], Idx[q_head_dim], Idx[1]),
        ),
    )

    mla_fused_rope_rmsnorm_quantization[kv_input_fn=kv_input_fn](
        q_rope_mut,
        q_rope.as_unsafe_any_origin(),  # hack aliasing.
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        kv_collection,
        layer_idx,
        epsilon,
        ctx,
    )

    # =========================================================================#
    # Up-project the latent KV cache to full K and V                           #
    # =========================================================================#

    # First, dump the k cache to a contiguous buffer
    # allocate a buffer for raw latent KV values
    var k_latent_buf = ctx.enqueue_create_buffer[dtype](
        buffer_length * kv_latent_dim
    )
    var k_latent = TileTensor(
        k_latent_buf,
        row_major(buffer_length, Idx[kv_latent_dim]),
    )

    # copy the k cache to the latent buffer
    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    _k_cache_to_buffer(
        buffer_row_offsets,
        cache_offsets,
        k_cache,
        Int32(buffer_length),
        k_latent,
        ctx,
    )

    # quantize the latent KV values to fp8
    # allocate buffers for fp8 latent KV values and scales
    # TODO: Fused the _k_cache_to_buffer with the quantize_dynamic_scaled_fp8
    var fp8_k_latent_buf = ctx.enqueue_create_buffer[fp8_dtype](
        buffer_length * kv_latent_dim
    )
    var fp8_k_latent = TileTensor(
        fp8_k_latent_buf,
        row_major(buffer_length, Idx[kv_latent_dim]),
    )

    # the scales are stored in a transposed, padded format
    comptime scales_m_padding = 16 // size_of[fp8_scale_dtype]()
    var scales_padded_m = align_up(buffer_length, scales_m_padding)
    var fp8_k_latent_scale_buf = ctx.enqueue_create_buffer[fp8_scale_dtype](
        scales_padded_m * ceildiv(kv_latent_dim, k_scale_granularity)
    )
    var fp8_k_latent_scale = TileTensor(
        fp8_k_latent_scale_buf,
        row_major(
            (Idx[ceildiv(kv_latent_dim, k_scale_granularity)], scales_padded_m)
        ),
    )

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var k_latent} -> SIMD[k_latent.dtype, width]:
        return k_latent.load[width=width]((row, col))

    quantize_dynamic_scaled_fp8[
        in_dtype=k_latent.dtype,
        group_size_or_per_token=k_scale_granularity,
        num_cols=k_latent.static_shape[1],
    ](
        input_fn,
        fp8_k_latent,
        fp8_k_latent_scale,
        1200.0,
        ctx,
        Int(k_latent.dim[0]()),
    )

    # Up-project latent KV values to K.
    var k_buf = ctx.enqueue_create_buffer[dtype](
        buffer_length * num_heads * qk_nope_head_dim
    )
    var k_flat = TileTensor(
        k_buf,
        row_major(buffer_length, Idx[num_heads * qk_nope_head_dim]),
    )
    matmul_dynamic_scaled_fp8[
        input_scale_granularity="block",
        weight_scale_granularity="block",
        m_scale_granularity=m_scale_granularity,
        n_scale_granularity=n_scale_granularity,
        k_scale_granularity=k_scale_granularity,
        transpose_b=True,
        target=target,
    ](
        k_flat,
        fp8_k_latent,
        w_k,
        fp8_k_latent_scale,
        w_k_scale,
        ctx,
    )

    # Reuse decode's rank-3 w_uv by flattening [H, Dv, K] -> [H*Dv, K].
    var w_v = TileTensor(
        w_uv.ptr,
        row_major(Idx[num_heads * v_head_dim], Idx[kv_latent_dim]),
    )
    var w_v_scale = TileTensor(
        w_uv_scale.ptr,
        row_major(
            (
                Idx[num_heads * (v_head_dim // n_scale_granularity)],
                Idx[kv_latent_dim // k_scale_granularity],
            )
        ),
    )

    # Up-project latent KV values to V.
    var v_buf = ctx.enqueue_create_buffer[dtype](
        buffer_length * num_heads * v_head_dim
    )
    var v_flat = TileTensor(
        v_buf,
        row_major(buffer_length, Idx[num_heads * v_head_dim]),
    )
    matmul_dynamic_scaled_fp8[
        input_scale_granularity="block",
        weight_scale_granularity="block",
        m_scale_granularity=m_scale_granularity,
        n_scale_granularity=n_scale_granularity,
        k_scale_granularity=k_scale_granularity,
        transpose_b=True,
        target=target,
    ](
        v_flat,
        fp8_k_latent,
        w_v,
        fp8_k_latent_scale,
        w_v_scale,
        ctx,
    )

    var k = TileTensor(
        k_buf,
        row_major((buffer_length, Idx[num_heads], Idx[qk_nope_head_dim])),
    )
    var v = TileTensor(
        v_buf,
        row_major(buffer_length, Idx[num_heads], Idx[v_head_dim]),
    )

    generic_flare_mla_prefill_kv_cache_ragged[
        target=target,
        mask_str=mask_str,
    ](
        q,
        k,
        v,
        buffer_row_offsets,
        cache_offsets,
        input_row_offsets,
        kv_collection,
        layer_idx,
        scale,
        output,
        ctx,
    )


# ===-----------------------------------------------------------------------===#
# Manually fused MLA decode branch (FP8)
# ===-----------------------------------------------------------------------===#


@always_inline
def quantize_and_bmm_fp8_helper[
    dtype: DType,
    c_dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    target: StaticString = "cpu",
](
    c: TileTensor[mut=True, c_dtype, address_space=.GENERIC, ...],
    a: TileTensor[dtype, address_space=.GENERIC, ...],
    b: TileTensor[fp8_dtype, address_space=.GENERIC, ...],
    b_scales: TileTensor[fp8_scale_dtype, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """
    Helper function to quantize and perform a batched matrix multiplication.
    This function uses the transposed view of the input tensor `a`.

    Parameters:
        dtype: Data type of the input tensor `a`.
        c_dtype: Data type of the output tensor `c`. When this is an FP8 type
            the matmul epilogue quantizes as it stores, so no separate
            conversion pass is needed downstream.
        fp8_dtype: Data type of the FP8 quantized tensors.
        fp8_scale_dtype: Data type of the FP8 scale tensors.
        m_scale_granularity: Granularity of the scale for the M dimension of
            the batched matrix multiplication.
        n_scale_granularity: Granularity of the scale for the N dimension of
            the batched matrix multiplication.
        k_scale_granularity: Granularity of the scale for the K dimension of
            the batched matrix multiplication, also the quantization group
            size for `a`.
        target: Target device for the batched matrix multiplication (defaults
            to "cpu").

    Args:
        c: Output tensor for the batched matrix multiplication result. Shape:
            [batch, m, n].
        a: Input tensor to quantize to FP8 and use as the left operand, loaded
            via a transposed view. Shape: [m, batch, k].
        b: FP8 weight tensor used as the right operand. Shape: [batch, n, k].
        b_scales: Scale tensor for `b`.
        ctx: Device context for buffer allocation and kernel execution.
    """

    # Evidence assert for TileTensor load Coord constraint.
    comptime assert type_of(a).flat_rank >= 3

    comptime B = a.static_shape[1]
    comptime K = a.static_shape[2]
    comptime N = b.static_shape[1]

    var m = Int(a.dim(0))

    # allocate buffers for quantized a and its scales
    var fp8_a_buf = ctx.enqueue_create_buffer[fp8_dtype](B * m * K)
    var fp8_a = TileTensor(fp8_a_buf, row_major(Idx[B], m, Idx[K]))

    # the scales are stored in a transposed, padded format
    comptime scales_m_padding = 16 // size_of[fp8_scale_dtype]()
    var scales_padded_m = align_up(m, scales_m_padding)
    var fp8_a_scale_buf = ctx.enqueue_create_buffer[fp8_scale_dtype](
        B * ceildiv(K, k_scale_granularity) * scales_padded_m
    )
    var fp8_a_scale = TileTensor(
        fp8_a_scale_buf,
        row_major(
            (Idx[B], Idx[ceildiv(K, k_scale_granularity)], scales_padded_m)
        ),
    )

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](batch: Int, row: Int, col: Int) {var a} -> SIMD[dtype, width]:
        # First transpose the q_nope tensor from [row, batch, col] to [batch, row, col].
        comptime assert a.flat_rank == 3
        return a.load[width=width]((row, batch, col))

    batched_quantize_dynamic_scaled_fp8[
        in_dtype=dtype,
        group_size_or_per_token=k_scale_granularity,
        num_cols=K,
    ](
        input_fn,
        fp8_a,
        fp8_a_scale,
        1200.0,
        ctx,
        num_rows=m,
        batch_size=B,
    )

    batched_matmul_dynamic_scaled_fp8[
        input_scale_granularity="block",
        weight_scale_granularity="block",
        m_scale_granularity=m_scale_granularity,
        n_scale_granularity=n_scale_granularity,
        k_scale_granularity=k_scale_granularity,
        transpose_b=True,
        target=target,
    ](
        c,
        fp8_a,
        b,
        fp8_a_scale,
        b_scales,
        ctx,
    )


def mla_decode_branch_fp8[
    dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    collection_t: KVCollectionT,
    //,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    mask_str: StaticString,
    kv_input_fn: def[width: Int](IndexList[2]) capturing -> SIMD[
        DType.bfloat16, width
    ],
    target: StaticString = "cpu",
    sparse_mla: Bool = False,
    # Read-once shared-index MTP fold (KERN-3141); threaded to flare_mla_decoding
    # (fp8-KV only; on the bf16 branch it is parity plumbing, always False).
    fold_shared_index: Bool = False,
    # Off keeps Q in `dtype`, so the two stagings can be compared.
    fp8_q: Bool = True,
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    q: TileTensor[dtype, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    freqs_cis: TileTensor[_, address_space=.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space=.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    w_uk: TileTensor[fp8_dtype, address_space=.GENERIC, ...],
    w_uk_scale: TileTensor[fp8_scale_dtype, address_space=.GENERIC, ...],
    w_uv: TileTensor[fp8_dtype, address_space=.GENERIC, ...],
    w_uv_scale: TileTensor[fp8_scale_dtype, address_space=.GENERIC, ...],
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    ctx: DeviceContext,
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    attn_sink_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    extra_k: OptionalReg[collection_t.CacheType] = None,
    extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_indices_stride: Int = 0,
    extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_scales_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    # Capturable-graph scalar forwarded from the MoGG op input list.
    num_partitions_in: Optional[Int] = None,
    # Logical sparse indices for position-based causal masking; `None` keeps
    # the prior slot-count behavior. See mla_decode_utils.mojo.
    logical_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
) raises:
    """
    This is a manually fused kernel that performs the following operations:
    - Apply RoPE to the query and the key cache (in-place).
    - Apply RMSNorm to the non-rope portion of the key cache (in-place).
    - Project q_nope to kv_latent_dim through a fp8 batched matmul:
        q_nope_proj = q_nope_t @ w_uk.
    - Concatenate q_nope_proj and q_rope:
        q_full = concat(q_nope_proj, q_rope, axis=2).
    - Perform MLA decode.
    - Project raw_output to v_head_dim through another fp8 batched matmul:
        output = raw_output_t @ w_uv.

    Parameters:
        dtype: Data type of the input and output tensors.
        fp8_dtype: Data type of the fp8 input and output tensors.
        fp8_scale_dtype: Data type of the fp8 scale input and output tensors.
        collection_t: Type of the KV collection.
        m_scale_granularity: Granularity of the scale for M dimension of the
            matrix multiplication.
        n_scale_granularity: Granularity of the scale for N dimension of the
            matrix multiplication.
        k_scale_granularity: Granularity of the scale for K dimension of the
            matrix multiplication.
        mask_str: Mask variant.
        kv_input_fn: Input lambda function to load the KV latent values. Shape:
            [tot_seq_len, cache_head_dim]. Where cache_head_dim = kv_lora_rank
            + qk_rope_head_dim.
        target: Target device.
        sparse_mla: Whether to use sparse MLA.
        fold_shared_index: Whether to enable the default-off read-once
            shared-index fold on the sparse FP8 decode path, which packs the
            folded MTP query positions into one CTA and gathers their single
            shared top-k list once.
        fp8_q: Whether to stage Q in the FP8 cache dtype where the sparse
            decode supports it, so the producers below quantize as they store.
            Off stages Q in `dtype` and routes to the sparse kernel that reads
            a wider Q.

    Args:
        output: Output tensor of shape [tot_seq_len, num_heads, v_head_dim].
        q: Combined query tensor containing both nope and rope parts. Shape:
            [tot_seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim].
        input_row_offsets: Indicates where each request starts and ends in
            `q`. Shape: [num_batches + 1].
        freqs_cis: Precomputed RoPE frequency values for rotary position
            embeddings. Shape: [max_seq_len, qk_rope_head_dim].
        kv_norm_gamma: RMSNorm gamma weights for normalizing the KV cache.
            Shape: [kv_lora_rank].
        kv_collection: Paged KV Cache object.
        layer_idx: Layer index.
        scale: Scale for the attention calculation.
        epsilon: Small constant for numerical stability in RMSNorm.
        w_uk: Weight matrix for projecting the non-rope part of each query head to
            KV latent space. Shape: [num_heads, kv_latent_dim, qk_nope_head_dim].
        w_uk_scale: The scale for the w_uk weight matrix. Shape varies
            depending on the FP8 block-scaling granularity used for weight
            quantization.
        w_uv: Weight matrix for projecting the output of the attention back to
            each head's original space. Shape: [num_heads, v_head_dim, kv_latent_dim].
        w_uv_scale: The scale for the w_uv weight matrix. Shape varies
            depending on the FP8 block-scaling granularity used for weight
            quantization.
        scalar_args_buf: Packed MLA dispatch metadata buffer.
        ctx: Device context.
        d_indices: Sparse decode packed indices (null when dense).
        indices_stride: Row stride in ``d_indices``.
        topk_lengths: Per-batch valid top-k counts.
        attn_sink_ptr: Optional per-batch attention sink weights.
        extra_k: Optional second key cache operand (see ``flare_mla_decoding``).
        extra_d_indices: Extra-stream sparse indices.
        extra_indices_stride: Stride for ``extra_d_indices``.
        extra_topk_lengths: Extra-stream per-batch lengths.
        extra_scales_ptr: Extra-stream scales.
        num_partitions_in: Capturable-graph num_partitions override.
        logical_indices: Logical sparse indices for position-based causal
            masking; `None` keeps the prior slot-count behavior.
    """

    comptime kv_params = collection_t.kv_params
    comptime assert kv_params.is_mla, "kv_params.is_mla should be true"
    comptime assert kv_params.num_heads == 1, "kv_params.num_heads should be 1"

    comptime num_heads = q.static_shape[1]
    comptime q_head_dim = q.static_shape[2]
    comptime v_head_dim = output.static_shape[2]
    comptime k_cache_dim = kv_params.head_size

    comptime assert (
        w_uk.shape_known and w_uv.shape_known
    ), "w_uk and w_uv's shapes should be static"
    comptime kv_latent_dim = w_uk.static_shape[1]
    comptime qk_rope_head_dim = k_cache_dim - kv_latent_dim
    comptime qk_nope_head_dim = q_head_dim - qk_rope_head_dim
    comptime assert (
        qk_rope_head_dim == 0 or freqs_cis.static_shape[1] == qk_rope_head_dim
    ), "freqs_cis's width should be k_cache_dim - kv_latent_dim."
    comptime assert (
        w_uk.static_shape[2] == qk_nope_head_dim
    ), "w_uk.static_shape[2] should be equal to qk_nope_head_dim"
    comptime assert (
        w_uv.static_shape[1] == v_head_dim
    ), "w_uv.static_shape[1] should be equal to v_head_dim"

    var seq_len = Int(q.dim(0))

    if seq_len == 0:
        return

    # Sparse decode over a unit-scale FP8 latent cache on SM100 hands the
    # kernel an FP8 Q so the dispatch routes to the native-FP8 sparse kernel
    # (MLA_SM100_Decode_Sparse_QKV_FP8). Both producers below quantize as they
    # store, so Q is staged in FP8 directly, matching the bf16-weights branch.
    comptime native_fp8_sparse = (
        fp8_q
        and sparse_mla
        and dtype == .bfloat16
        and collection_t.CacheType.dtype == .float8_e4m3fn
        and not collection_t.CacheType.quantization_enabled
        and _is_sm10x_gpu(ctx.default_device_info)
    )
    comptime mla_decode_input_dtype = (
        collection_t.CacheType.dtype if native_fp8_sparse else dtype
    )

    # First, create a input buffer for the mla decode kernel
    var mla_decode_input_buf = ctx.enqueue_create_buffer[
        mla_decode_input_dtype
    ](seq_len * num_heads * k_cache_dim)
    var mla_decode_input = TileTensor(
        mla_decode_input_buf,
        row_major(seq_len, Idx[num_heads], Idx[k_cache_dim]),
    )

    # =========================================================================#
    # Project the non-rope part of each query head to kv_latent_dim            #
    # =========================================================================#

    # The first qk_nope_head_dim columns of each Q head will be up-projected to
    # kv_latent_dim through a fp8 batched matmul.

    # We start by create a view of the input tensor `q` that only contains the
    # first qk_nope_head_dim columns of each Q head.
    var q_nope = TileTensor(
        q.ptr,
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_nope_head_dim]),
            (Idx[num_heads * q_head_dim], Idx[q_head_dim], Idx[1]),
        ),
    )

    # Then create a view of the mla_decode_input tensor that only contains the
    # first kv_latent_dim columns of each Q head.
    var mla_decode_input_nope = TileTensor(
        mla_decode_input.ptr,
        TileLayout(
            (Idx[num_heads], seq_len, Idx[kv_latent_dim]),
            (Idx[k_cache_dim], Idx[num_heads * k_cache_dim], Idx[1]),
        ),
    )

    # Proceed with the fp8 batched matmul
    # This helper function uses the transposed view of the input tensor `q_nope`.
    quantize_and_bmm_fp8_helper[
        m_scale_granularity=m_scale_granularity,
        n_scale_granularity=n_scale_granularity,
        k_scale_granularity=k_scale_granularity,
        target=target,
    ](mla_decode_input_nope, q_nope, w_uk, w_uk_scale, ctx)

    # =========================================================================#
    # QK RoPE and K cache RMSNorm                                              #
    # =========================================================================#

    # Create a view of the `q` tensor that only contains the last
    # qk_rope_head_dim columns of each Q head.
    var q_rope = TileTensor(
        q.ptr + qk_nope_head_dim,
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * q_head_dim], Idx[q_head_dim], Idx[1]),
        ),
    )

    # Create a view of the `mla_decode_input` tensor that only contains the last
    # qk_rope_head_dim columns of each Q head.
    var mla_decode_input_rope = TileTensor(
        mla_decode_input.ptr + kv_latent_dim,
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * k_cache_dim], Idx[k_cache_dim], Idx[1]),
        ),
    )

    mla_fused_rope_rmsnorm_quantization[kv_input_fn=kv_input_fn](
        mla_decode_input_rope,
        q_rope,
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        kv_collection,
        layer_idx,
        epsilon,
        ctx,
    )

    # =========================================================================#
    # MLA decode                                                               #
    # =========================================================================#

    var raw_output_buf = ctx.enqueue_create_buffer[dtype](
        seq_len * num_heads * kv_latent_dim
    )
    var raw_output = TileTensor(
        raw_output_buf,
        row_major(seq_len, Idx[num_heads], Idx[kv_latent_dim]),
    )

    generic_flare_mla_decode_kv_cache_ragged[
        target=target,
        mask_str=mask_str,
        sparse_mla=sparse_mla,
        fold_shared_index=fold_shared_index,
    ](
        mla_decode_input,
        input_row_offsets,
        kv_collection,
        layer_idx,
        scale,
        raw_output,
        scalar_args_buf,
        ctx,
        d_indices=d_indices,
        indices_stride=indices_stride,
        topk_lengths=topk_lengths,
        attn_sink_ptr=attn_sink_ptr,
        extra_k=extra_k,
        extra_d_indices=extra_d_indices,
        extra_indices_stride=extra_indices_stride,
        extra_topk_lengths=extra_topk_lengths,
        extra_scales_ptr=extra_scales_ptr,
        num_partitions_in=num_partitions_in,
        logical_indices=logical_indices,
    )

    # Create a view of the output tensor with logical shape
    # [num_heads, seq_len, v_head_dim], and map directly to
    # [seq_len, num_heads, v_head_dim] physical memory.
    var output_t = TileTensor(
        output.ptr,
        TileLayout(
            (Idx[num_heads], seq_len, Idx[v_head_dim]),
            (Idx[v_head_dim], Idx[num_heads * v_head_dim], Idx[1]),
        ),
    )

    # Another batched matmul to project the raw output to the original space
    # This helper function uses the transposed view of the input tensor `raw_output`.
    quantize_and_bmm_fp8_helper[
        dtype=dtype,
        fp8_dtype=fp8_dtype,
        fp8_scale_dtype=fp8_scale_dtype,
        m_scale_granularity=m_scale_granularity,
        n_scale_granularity=n_scale_granularity,
        k_scale_granularity=k_scale_granularity,
        target=target,
    ](output_t, raw_output, w_uv, w_uv_scale, ctx)


@always_inline
def mla_prefill_branch_sparse_fp8[
    dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    collection_t: KVCollectionT,
    //,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    kv_input_fn: def[width: Int](IndexList[2]) capturing -> SIMD[
        DType.bfloat16, width
    ],
    indices_stride: Int,
    target: StaticString = "cpu",
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    q: TileTensor[dtype, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    freqs_cis: TileTensor[_, address_space=.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space=.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    w_uk: TileTensor[fp8_dtype, address_space=.GENERIC, ...],
    w_uk_scale: TileTensor[fp8_scale_dtype, address_space=.GENERIC, ...],
    w_uv: TileTensor[fp8_dtype, address_space=.GENERIC, ...],
    w_uv_scale: TileTensor[fp8_scale_dtype, address_space=.GENERIC, ...],
    ctx: DeviceContext,
    d_indices: UnsafePointer[Int32, MutAnyOrigin],
    topk_lengths: UnsafePointer[Int32, MutAnyOrigin],
    attn_sink_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]],
) raises:
    """Sparse MLA prefill branch (DSv3.2/GLM absorbed shape, FP8 weights).

    Reuses `mla_decode_branch_fp8`'s absorbed-Q construction (q_nope up-proj via
    `w_uk` + RoPE/RMSNorm) and the identical `w_uv` output up-projection, and
    swaps the attention call to the existing `mla_sm100_prefill_sparse` kernel
    over the paged BF16 latent cache. The caller (`.fp8.sparse` op) has already
    remapped `d_indices` from logical to physical rows, so they are passed
    straight through. Only supported for a BF16 KV cache; the FP8-cache case is
    handled by the caller (dense fallback).
    """
    comptime kv_params = collection_t.kv_params
    comptime assert kv_params.is_mla, "kv_params.is_mla should be true"
    comptime assert kv_params.num_heads == 1, "kv_params.num_heads should be 1"

    comptime num_heads = q.static_shape[1]
    comptime q_head_dim = q.static_shape[2]
    comptime v_head_dim = output.static_shape[2]
    comptime k_cache_dim = kv_params.head_size

    comptime assert (
        w_uk.shape_known and w_uv.shape_known
    ), "w_uk and w_uv's shapes should be static"
    comptime kv_latent_dim = w_uk.static_shape[1]
    comptime qk_rope_head_dim = k_cache_dim - kv_latent_dim
    comptime qk_nope_head_dim = q_head_dim - qk_rope_head_dim
    comptime assert (
        qk_rope_head_dim == 0 or freqs_cis.static_shape[1] == qk_rope_head_dim
    ), "freqs_cis's width should be k_cache_dim - kv_latent_dim."
    comptime assert (
        w_uk.static_shape[2] == qk_nope_head_dim
    ), "w_uk.static_shape[2] should be equal to qk_nope_head_dim"
    comptime assert (
        w_uv.static_shape[1] == v_head_dim
    ), "w_uv.static_shape[1] should be equal to v_head_dim"

    var seq_len = Int(q.dim(0))
    if seq_len == 0:
        return

    var mla_decode_input_buf = ctx.enqueue_create_buffer[dtype](
        seq_len * num_heads * k_cache_dim
    )
    var mla_decode_input = TileTensor(
        mla_decode_input_buf,
        row_major(seq_len, Idx[num_heads], Idx[k_cache_dim]),
    )

    var q_nope = TileTensor(
        q.ptr,
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_nope_head_dim]),
            (Idx[num_heads * q_head_dim], Idx[q_head_dim], Idx[1]),
        ),
    )
    var mla_decode_input_nope = TileTensor(
        mla_decode_input.ptr,
        TileLayout(
            (Idx[num_heads], seq_len, Idx[kv_latent_dim]),
            (Idx[k_cache_dim], Idx[num_heads * k_cache_dim], Idx[1]),
        ),
    )
    quantize_and_bmm_fp8_helper[
        m_scale_granularity=m_scale_granularity,
        n_scale_granularity=n_scale_granularity,
        k_scale_granularity=k_scale_granularity,
        target=target,
    ](mla_decode_input_nope, q_nope, w_uk, w_uk_scale, ctx)

    var q_rope = TileTensor(
        q.ptr + qk_nope_head_dim,
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * q_head_dim], Idx[q_head_dim], Idx[1]),
        ),
    )
    var mla_decode_input_rope = TileTensor(
        mla_decode_input.ptr + kv_latent_dim,
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * k_cache_dim], Idx[k_cache_dim], Idx[1]),
        ),
    )
    mla_fused_rope_rmsnorm_quantization[kv_input_fn=kv_input_fn](
        mla_decode_input_rope,
        q_rope,
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        kv_collection,
        layer_idx,
        epsilon,
        ctx,
    )

    var raw_output_buf = ctx.enqueue_create_buffer[dtype](
        seq_len * num_heads * kv_latent_dim
    )
    var raw_output = TileTensor(
        raw_output_buf,
        row_major(seq_len, Idx[num_heads], Idx[kv_latent_dim]),
    )

    # `d_indices` / `topk_lengths` are int32 buffers reinterpreted as uint32:
    # invalid `-1` slots become 0xFFFFFFFF and are rejected by the kernel's
    # `idx >= 0` gather producer.
    var indices_tt = TileTensor(
        d_indices.bitcast[UInt32](),
        row_major(seq_len * indices_stride),
    )
    var topk_lengths_tt = TileTensor(
        topk_lengths.bitcast[UInt32](),
        row_major(seq_len),
    )
    var attn_sink_opt = Optional[UnsafePointer[Float32, ImmutAnyOrigin]](None)
    if attn_sink_ptr:
        attn_sink_opt = UnsafePointer[Float32, ImmutAnyOrigin](
            attn_sink_ptr.value()
        )

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    # Unit-scale FP8 latent cache on SM100 runs the native-FP8 sparse prefill
    # kernel (QK^T and P*V in FP8, no dequant) when the head count fits its
    # tiles: num_q_heads == 128 (cta_group=2) or a multiple of 8 in (0, 64]
    # (cta_group=1). `quantize_and_bmm_fp8_helper` ties its output dtype to
    # the bf16 input, so Q is converted with a saturating-cast pass (unlike
    # the bf16-weights branch, which stages FP8 directly).
    comptime native_fp8_sparse = (
        dtype == .bfloat16
        and collection_t.CacheType.dtype == .float8_e4m3fn
        and not collection_t.CacheType.quantization_enabled
        and _is_sm10x_gpu(ctx.default_device_info)
        and (num_heads == 128 or (num_heads <= 64 and num_heads % 8 == 0))
    )
    comptime if collection_t.CacheType.dtype.is_float8():
        comptime if native_fp8_sparse:
            var q_fp8_buf = ctx.enqueue_create_buffer[.float8_e4m3fn](
                seq_len * num_heads * k_cache_dim
            )
            var q_fp8 = TileTensor(
                q_fp8_buf,
                row_major(seq_len, Idx[num_heads], Idx[k_cache_dim]),
            )
            var mla_decode_input_bf16 = TileTensor(
                mla_decode_input.ptr.bitcast[BFloat16](),
                row_major(seq_len, Idx[num_heads], Idx[k_cache_dim]),
            )
            convert_bf16_to_fp8_e4m3fn(mla_decode_input_bf16, q_fp8, ctx)

            comptime cta_group = 2 if num_heads == 128 else 1
            comptime b_topk = 128 if cta_group == 2 else 64
            comptime num_mbars = 2 if cta_group == 2 else 4
            comptime config = MLASparseConfig[
                DType.bfloat16,
                b_topk_=b_topk,
                num_mbars_=num_mbars,
                cta_group_=cta_group,
            ](
                num_q_heads=num_heads,
                num_kv_heads=1,
                # Tile stays laid out for 576. The kernel zero-fills the NoPE tail.
                qk_depth=kv_latent_dim + 64,
                input_qk_depth=k_cache_dim,
                v_depth=kv_latent_dim,
                indices_stride=indices_stride,
                group=num_heads,
            )
            mla_prefill_sparse_qkv_fp8[
                config=config,
                group=num_heads,
                q_depth=k_cache_dim,
                scale_block_size=0,
            ](
                raw_output,
                q_fp8,
                k_cache,
                indices_tt,
                topk_lengths_tt,
                attn_sink_opt,
                scale,
                Int32(indices_stride),
                ctx,
            )
        else:
            # FP8 latent cache with bf16 Q: run the FP8 sparse-prefill kernel
            # directly over the quantized cache (no BF16 staging). Today the
            # cache carries no dequant scales (scale_dtype=int8 => quantization
            # disabled), so read at unit scale (scale_block_size=0), mirroring
            # the sparse-DECODE kernel's read. scales_ptr is unused at
            # scale_block_size=0; pass a non-null dummy (SnapMLA/SERVOPT-1094
            # will supply real scales + a positive scale_block_size here once
            # the cache carries them).
            var dummy_scales = (
                raw_output_buf.unsafe_ptr()
                .bitcast[Float32]()
                .as_unsafe_any_origin()
            )
            mla_sm100_prefill_sparse_fp8[
                num_q_heads=num_heads,
                qk_depth=k_cache_dim,
                v_depth=kv_latent_dim,
                indices_stride=indices_stride,
                scale_block_size=0,
            ](
                raw_output,
                mla_decode_input,
                k_cache,
                indices_tt,
                topk_lengths_tt,
                attn_sink_opt,
                dummy_scales,
                scale,
                ctx,
            )
    else:
        mla_sm100_prefill_sparse[
            num_q_heads=num_heads,
            qk_depth=k_cache_dim,
            v_depth=kv_latent_dim,
            indices_stride=indices_stride,
        ](
            raw_output,
            mla_decode_input,
            k_cache,
            indices_tt,
            topk_lengths_tt,
            attn_sink_opt,
            scale,
            ctx,
        )

    var output_t = TileTensor(
        output.ptr,
        TileLayout(
            (Idx[num_heads], seq_len, Idx[v_head_dim]),
            (Idx[v_head_dim], Idx[num_heads * v_head_dim], Idx[1]),
        ),
    )
    quantize_and_bmm_fp8_helper[
        dtype=dtype,
        fp8_dtype=fp8_dtype,
        fp8_scale_dtype=fp8_scale_dtype,
        m_scale_granularity=m_scale_granularity,
        n_scale_granularity=n_scale_granularity,
        k_scale_granularity=k_scale_granularity,
        target=target,
    ](output_t, raw_output, w_uv, w_uv_scale, ctx)


# ===-----------------------------------------------------------------------===#
# MLA prefill-decode graph (FP8)
# ===-----------------------------------------------------------------------===#


@always_inline
def mla_prefill_decode_graph_fp8[
    dtype: DType,
    fp8_dtype: DType,
    fp8_scale_dtype: DType,
    collection_t: KVCollectionT,
    //,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    mask_str: StaticString,
    kv_input_fn: def[width: Int](IndexList[2]) capturing -> SIMD[
        DType.bfloat16, width
    ],
    target: StaticString = "cpu",
    sparse_mla: Bool = False,
    sparse_indices_stride: Int = 0,
    # Read-once shared-index MTP fold (KERN-3141); threaded to flare_mla_decoding
    # (fp8-KV only; on the bf16 branch it is parity plumbing, always False).
    fold_shared_index: Bool = False,
    # Stage Q in the FP8 cache dtype where the sparse decode supports it; see
    # `mla_decode_branch_fp8`.
    fp8_q: Bool = True,
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    q: TileTensor[dtype, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    freqs_cis: TileTensor[_, address_space=.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space=.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    buffer_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    cache_offsets: TileTensor[mut=True, .uint32, address_space=.GENERIC, ...],
    buffer_length: Int,
    max_seq_len: Int,
    w_k: TileTensor[fp8_dtype, address_space=.GENERIC, ...],
    w_k_scale: TileTensor[fp8_scale_dtype, address_space=.GENERIC, ...],
    w_uk: TileTensor[fp8_dtype, address_space=.GENERIC, ...],
    w_uk_scale: TileTensor[fp8_scale_dtype, address_space=.GENERIC, ...],
    w_uv: TileTensor[fp8_dtype, address_space=.GENERIC, ...],
    w_uv_scale: TileTensor[fp8_scale_dtype, address_space=.GENERIC, ...],
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    ctx: DeviceContext,
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    attn_sink_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    extra_k: OptionalReg[collection_t.CacheType] = None,
    extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_indices_stride: Int = 0,
    extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_scales_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    # Capturable-graph scalar forwarded from the MoGG op input list.
    num_partitions_in: Optional[Int] = None,
) raises:
    """
    This is a manually fused kernel that performs the following operations:
    - Perform MLA prefill or decode based on the maximum sequence length.

    Parameters:
        dtype: Data type of the input and output tensors (inferred).
        fp8_dtype: Data type of the fp8 input and output tensors (inferred).
        fp8_scale_dtype: Data type of the fp8 scale input and output tensors
            (inferred).
        collection_t: Type of the KV collection (inferred).
        m_scale_granularity: Granularity of the scale for M dimension of the
            matrix multiplication.
        n_scale_granularity: Granularity of the scale for N dimension of the
            matrix multiplication.
        k_scale_granularity: Granularity of the scale for K dimension of the
            matrix multiplication.
        mask_str: Mask variant.
        kv_input_fn: Input lambda function to load the KV latent values. Shape:
            [tot_seq_len, cache_head_dim]. Where cache_head_dim = kv_lora_rank
            + qk_rope_head_dim.
        target: Target device (defaults to "cpu").
        sparse_mla: Whether to use sparse MLA (defaults to False).
        sparse_indices_stride: Row stride of the sparse decode index buffer
            (defaults to 0).
        fold_shared_index: Whether to use the read-once shared-index MTP
            fold threaded to `flare_mla_decoding` (defaults to False).
        fp8_q: Whether to stage Q in the FP8 cache dtype on the sparse decode
            path (defaults to True); see `mla_decode_branch_fp8`.

    Args:
        output: Output tensor of shape [tot_seq_len, num_heads, v_head_dim].
        q: Combined query tensor containing both nope and rope parts. Shape:
            [tot_seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim].
        input_row_offsets: Indicates where each request starts and ends in
            `q`. Shape: [num_batches + 1].
        freqs_cis: Precomputed RoPE frequency values for rotary position
            embeddings. Shape: [max_seq_len, qk_rope_head_dim].
        kv_norm_gamma: RMSNorm gamma weights for normalizing the KV cache.
            Shape: [kv_lora_rank].
        kv_collection: Paged KV Cache object.
        layer_idx: Layer index.
        scale: Scale for the attention calculation.
        epsilon: Small constant for numerical stability in RMSNorm.
        buffer_row_offsets: Indicates where each request's KV latent values
            should be stored in the contiguous K buffer. This is a 1D tensor
            of shape [num_batches + 1].
        cache_offsets: Indicates the starting token position in the KV cache
            from which to copy KV latent values for each request. This is a 1D
            tensor of shape [num_batches + 1].
        buffer_length: The total number of tokens in the KV cache. Scalar.
        max_seq_len: Maximum sequence length in the batch, used to select
            prefill versus decode.
        w_k: Weight matrix for up-projecting the latent cache to full K. Shape:
            [num_heads * qk_nope_head_dim, kv_latent_dim].
        w_k_scale: Scale tensor for `w_k`.
        w_uk: Weight matrix for projecting the non-rope part of each query head
            to KV latent space. Shape: [num_heads, kv_latent_dim,
            qk_nope_head_dim].
        w_uk_scale: The scale for the `w_uk` weight matrix. Shape varies
            depending on the FP8 block-scaling granularity used for weight
            quantization.
        w_uv: Weight tensor for projecting latent values to V. Shape:
            [num_heads, v_head_dim, kv_latent_dim].
        w_uv_scale: Scale tensor for `w_uv`.
        scalar_args_buf: Packed MLA dispatch metadata buffer.
        ctx: Device context.
        d_indices: Sparse decode packed indices (null when dense, defaults to
            None).
        indices_stride: Row stride in `d_indices` (defaults to 0).
        topk_lengths: Per-batch valid top-k counts (defaults to None).
        attn_sink_ptr: Optional per-batch attention sink weights (defaults to
            None).
        extra_k: Optional second key cache operand (see
            `flare_mla_decoding`, defaults to None).
        extra_d_indices: Extra-stream sparse indices (defaults to None).
        extra_indices_stride: Stride for `extra_d_indices` (defaults to 0).
        extra_topk_lengths: Extra-stream per-batch lengths (defaults to None).
        extra_scales_ptr: Extra-stream scales (defaults to None).
        num_partitions_in: Capturable-graph num_partitions override (defaults
            to None).
    """

    var seq_len = q.dim(0)

    if seq_len == 0:
        return

    # When running verification with MTP we want to use the decode branch.
    if (
        max_seq_len
        <= mla_decode_max_seq_len[
            collection_t.CacheType.dtype, q.static_shape[1]
        ]()
    ):
        mla_decode_branch_fp8[
            m_scale_granularity=m_scale_granularity,
            n_scale_granularity=n_scale_granularity,
            k_scale_granularity=k_scale_granularity,
            mask_str=mask_str,
            kv_input_fn=kv_input_fn,
            target=target,
            sparse_mla=sparse_mla,
            fold_shared_index=fold_shared_index,
            fp8_q=fp8_q,
        ](
            output,
            q,
            input_row_offsets,
            freqs_cis,
            kv_norm_gamma,
            kv_collection,
            layer_idx,
            scale,
            epsilon,
            w_uk,
            w_uk_scale,
            w_uv,
            w_uv_scale,
            scalar_args_buf,
            ctx,
            d_indices,
            indices_stride,
            topk_lengths,
            attn_sink_ptr,
            extra_k,
            extra_d_indices,
            extra_indices_stride,
            extra_topk_lengths,
            extra_scales_ptr,
            num_partitions_in,
        )

    else:
        comptime if sparse_mla:
            # Sparse MLA prefill for BOTH bf16 and fp8 latent caches: the
            # branch comptime-dispatches the attention kernel on the cache
            # dtype (fp8 cache => mla_sm100_prefill_sparse_fp8 read at unit
            # scale, mirroring sparse decode; bf16 cache =>
            # mla_sm100_prefill_sparse). Replaces the dense FP8-cache fallback
            # that previously ran here.
            mla_prefill_branch_sparse_fp8[
                m_scale_granularity=m_scale_granularity,
                n_scale_granularity=n_scale_granularity,
                k_scale_granularity=k_scale_granularity,
                kv_input_fn=kv_input_fn,
                indices_stride=sparse_indices_stride,
                target=target,
            ](
                output,
                q,
                input_row_offsets,
                freqs_cis,
                kv_norm_gamma,
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                w_uk,
                w_uk_scale,
                w_uv,
                w_uv_scale,
                ctx,
                d_indices.value(),
                topk_lengths.value(),
                attn_sink_ptr,
            )
        else:
            # Dense prefill for NON-sparse MLA only. Sparse MLA (both bf16 and
            # fp8 caches) is handled by the sparse branch above.
            mla_prefill_branch_fp8[
                m_scale_granularity=m_scale_granularity,
                n_scale_granularity=n_scale_granularity,
                k_scale_granularity=k_scale_granularity,
                mask_str=mask_str,
                kv_input_fn=kv_input_fn,
                target=target,
            ](
                output,
                q,
                input_row_offsets,
                freqs_cis,
                kv_norm_gamma,
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                buffer_row_offsets,
                cache_offsets,
                buffer_length,
                w_k,
                w_k_scale,
                w_uv,
                w_uv_scale,
                ctx,
            )


@always_inline
def convert_bf16_to_fp8_e4m3fn(
    input_buffer: TileTensor[mut=False, .bfloat16, ...],
    output_buffer: TileTensor[mut=True, .float8_e4m3fn, ...],
    context: DeviceContext,
) raises:
    """Convert bfloat16 weights to E4M3FN format.

    Args:
        input_buffer: Input tensor in bfloat16 format.
        output_buffer: Output tensor to store E4M3FN format.
        context: Device context for kernel execution.
    """
    # Runtime assertions for dynamic dimensions
    comptime assert (
        input_buffer.rank == output_buffer.rank
    ), "Input and output must have the same rank"

    @always_inline
    @__parameter
    @__copy_capture(input_buffer, output_buffer)
    def convert_kernel[
        width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]):
        comptime assert rank == 2 or rank == 3, "rank should be 2 or 3"

        output_buffer.store_linear(
            idx,
            cast_saturating[.float8_e4m3fn](
                input_buffer.load_linear[width](idx)
            ),
        )

    comptime target_simd_width = simd_width_of[
        DType.bfloat16, target=get_gpu_target()
    ]()

    def convert_kernel_unified[width: Int, alignment: Int = 1](idx: Coord):
        convert_kernel[width, idx.rank, alignment](coord_to_index_list(idx))

    comptime if input_buffer.rank == 2:
        _elementwise_impl_gpu[
            simd_width=target_simd_width,
            trace_description="mla_bf16_to_fp8_convert",
        ](
            convert_kernel_unified,
            shape=(
                Int(input_buffer.dim[0]()),
                Int(input_buffer.dim[1]()),
            ),
            ctx=context,
        )
    else:
        _elementwise_impl_gpu[
            simd_width=target_simd_width,
            trace_description="mla_bf16_to_fp8_convert",
        ](
            convert_kernel_unified,
            shape=(
                Int(input_buffer.dim[0]()),
                Int(input_buffer.dim[1]()),
                Int(input_buffer.dim[2]()),
            ),
            ctx=context,
        )


# ===-----------------------------------------------------------------------===#
# Manually fused MLA prefill branch (BF16)
# ===-----------------------------------------------------------------------===#


def mla_prefill_branch_bf16[
    collection_t: KVCollectionT,
    //,
    mask_str: StaticString,
    kv_input_fn: def[width: Int](IndexList[2]) capturing -> SIMD[
        DType.bfloat16, width
    ],
    target: StaticString = "cpu",
](
    output: TileTensor[mut=True, .bfloat16, address_space=.GENERIC, ...],
    q: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    freqs_cis: TileTensor[_, address_space=.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space=.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    buffer_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    cache_offsets: TileTensor[mut=True, .uint32, address_space=.GENERIC, ...],
    buffer_length: Int,
    w_k: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    w_uv: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """BF16 MLA prefill path.

    Applies RoPE and RMSNorm, up-projects latent KV to full K and V, then runs
    prefill attention.

    Parameters:
        collection_t: Type of the KV collection (inferred).
        mask_str: Mask variant.
        kv_input_fn: Input lambda function to load the KV latent values. Shape:
            [tot_seq_len, cache_head_dim]. Where cache_head_dim = kv_lora_rank
            + qk_rope_head_dim.
        target: Target device (defaults to "cpu").

    Args:
        output: Output tensor of shape [tot_seq_len, num_heads, v_head_dim].
        q: Combined query tensor containing both nope and rope parts. Shape:
            [tot_seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim].
        input_row_offsets: Indicates where each request starts and ends in
            `q`. Shape: [num_batches + 1].
        freqs_cis: Precomputed RoPE frequency values for rotary position
            embeddings. Shape: [max_seq_len, qk_rope_head_dim].
        kv_norm_gamma: RMSNorm gamma weights for normalizing the KV cache.
            Shape: [kv_lora_rank].
        kv_collection: Paged KV Cache object.
        layer_idx: Layer index.
        scale: Scale for the attention calculation.
        epsilon: Small constant for numerical stability in RMSNorm.
        buffer_row_offsets: Indicates where each request's KV latent values
            should be stored in the contiguous K buffer. This is a 1D tensor
            of shape [num_batches + 1].
        cache_offsets: Indicates the starting token position in the KV cache
            from which to copy KV latent values for each request. This is a 1D
            tensor of shape [num_batches + 1].
        buffer_length: The total number of tokens in the KV cache. Scalar.
        w_k: Weight matrix for up-projecting the latent cache to full K. Shape:
            [num_heads * qk_nope_head_dim, kv_latent_dim].
        w_uv: Weight tensor for projecting latent values to V. Shape:
            [num_heads, v_head_dim, kv_latent_dim].
        ctx: Device context.
    """
    comptime kv_params = collection_t.kv_params
    comptime assert kv_params.is_mla, "kv_params.is_mla should be true"
    comptime assert kv_params.num_heads == 1, "kv_params.num_heads should be 1"

    comptime num_heads = q.static_shape[1]
    comptime q_head_dim = q.static_shape[2]
    comptime v_head_dim = output.static_shape[2]
    comptime k_cache_dim = kv_params.head_size

    comptime assert w_k.shape_known, "w_k's shape should be static"
    comptime kv_latent_dim = w_k.static_shape[1]
    comptime qk_rope_head_dim = k_cache_dim - kv_latent_dim
    comptime qk_nope_head_dim = q_head_dim - qk_rope_head_dim
    comptime assert (
        qk_rope_head_dim == 0 or freqs_cis.static_shape[1] == qk_rope_head_dim
    ), "freqs_cis's width should be k_cache_dim - kv_latent_dim."
    comptime assert (
        w_k.static_shape[0] == num_heads * qk_nope_head_dim
    ), "w_k.shape[0] should be equal to num_heads * qk_nope_head_dim"
    comptime assert w_uv.shape_known, "w_uv's shape should be static"
    comptime assert (
        w_uv.static_shape[0] == num_heads
    ), "w_uv.shape[0] should be equal to num_heads"
    comptime assert (
        w_uv.static_shape[1] == v_head_dim
    ), "w_uv.shape[1] should be equal to v_head_dim"
    comptime assert (
        w_uv.static_shape[2] == kv_latent_dim
    ), "w_uv.shape[2] should be equal to kv_latent_dim"

    if buffer_length == 0:
        return

    var seq_len = Int(q.dim(0))
    if seq_len == 0:
        return

    # =========================================================================#
    # QK RoPE and K cache RMSNorm                                              #
    # =========================================================================#

    # Create a view of the `q` tensor that only contains the last
    # qk_rope_head_dim columns of each Q head.
    var q_rope = TileTensor(
        q.ptr + qk_nope_head_dim,
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * q_head_dim], Idx[q_head_dim], Idx[1]),
        ),
    )

    # In-place update of the rope part of the `q` tensor
    var q_rope_mut = TileTensor(
        q_rope.ptr.unsafe_mut_cast[True](),
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * q_head_dim], Idx[q_head_dim], Idx[1]),
        ),
    )

    mla_fused_rope_rmsnorm_quantization[kv_input_fn=kv_input_fn](
        q_rope_mut,
        q_rope.as_unsafe_any_origin(),  # hack aliasing.
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        kv_collection,
        layer_idx,
        epsilon,
        ctx,
    )

    # allocate buffers for latent KV
    var k_latent_buf = ctx.enqueue_create_buffer[.bfloat16](
        buffer_length * kv_latent_dim
    )
    var k_latent = TileTensor(
        k_latent_buf,
        row_major(buffer_length, Idx[kv_latent_dim]),
    )

    var buffer_length_int = Int(buffer_length)
    var k_cache = kv_collection.get_key_cache(Int(layer_idx))

    _k_cache_to_buffer(
        buffer_row_offsets,
        cache_offsets,
        k_cache,
        Int32(buffer_length_int),
        k_latent,
        ctx,
    )

    comptime if collection_t.CacheType.dtype.is_float8():
        # Allocate FP8 buffers for K and V
        var k_fp8_buf = ctx.enqueue_create_buffer[.float8_e4m3fn](
            buffer_length * num_heads * qk_nope_head_dim
        )
        var k_fp8_flat = TileTensor(
            k_fp8_buf,
            row_major((buffer_length, Idx[num_heads * qk_nope_head_dim])),
        )

        var v_fp8_buf = ctx.enqueue_create_buffer[.float8_e4m3fn](
            buffer_length * num_heads * v_head_dim
        )
        var v_fp8_flat = TileTensor(
            v_fp8_buf,
            row_major(buffer_length, Idx[num_heads * v_head_dim]),
        )

        # K matmul with internal FP8 conversion
        matmul[
            target=target,
            transpose_b=True,
        ](
            k_fp8_flat,
            k_latent,
            w_k,
            Optional(ctx),
        )

        var w_v = TileTensor(
            w_uv.ptr,
            row_major(Idx[num_heads * v_head_dim], Idx[kv_latent_dim]),
        )

        # V matmul with internal FP8 conversion
        matmul[
            target=target,
            transpose_b=True,
        ](
            v_fp8_flat,
            k_latent,
            w_v,
            Optional(ctx),
        )

        # Create 3D views for attention kernel
        var k_fp8 = TileTensor(
            k_fp8_buf,
            row_major((buffer_length, Idx[num_heads], Idx[qk_nope_head_dim])),
        )
        var v_fp8 = TileTensor(
            v_fp8_buf,
            row_major((buffer_length, Idx[num_heads], Idx[v_head_dim])),
        )

        # Allocate FP8 buffer for Q and convert
        var q_fp8_buf = ctx.enqueue_create_buffer[.float8_e4m3fn](
            seq_len * num_heads * q_head_dim
        )
        var q_fp8 = TileTensor(
            q_fp8_buf,
            row_major(seq_len, Idx[num_heads], Idx[q_head_dim]),
        )
        convert_bf16_to_fp8_e4m3fn(q, q_fp8, ctx)

        # Pass FP8 tensors to attention kernel
        generic_flare_mla_prefill_kv_cache_ragged[
            target=target,
            mask_str=mask_str,
        ](
            q_fp8,
            k_fp8,
            v_fp8,
            buffer_row_offsets,
            cache_offsets,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            ctx,
        )
    else:
        # Standard BF16 path
        var k_buf = ctx.enqueue_create_buffer[.bfloat16](
            buffer_length * num_heads * qk_nope_head_dim
        )
        var k_flat = TileTensor(
            k_buf,
            row_major((buffer_length, Idx[num_heads * qk_nope_head_dim])),
        )
        matmul[target=target, transpose_b=True](
            k_flat,
            k_latent,
            w_k,
            Optional(ctx),
        )

        var w_v = TileTensor(
            w_uv.ptr,
            row_major(Idx[num_heads * v_head_dim], Idx[kv_latent_dim]),
        )
        var v_buf = ctx.enqueue_create_buffer[.bfloat16](
            buffer_length * num_heads * v_head_dim
        )
        var v_flat = TileTensor(
            v_buf,
            row_major(buffer_length, Idx[num_heads * v_head_dim]),
        )
        matmul[target=target, transpose_b=True](
            v_flat,
            k_latent,
            w_v,
            Optional(ctx),
        )

        var k = TileTensor(
            k_buf,
            row_major((buffer_length, Idx[num_heads], Idx[qk_nope_head_dim])),
        )
        var v = TileTensor(
            v_buf,
            row_major((buffer_length, Idx[num_heads], Idx[v_head_dim])),
        )

        generic_flare_mla_prefill_kv_cache_ragged[
            target=target,
            mask_str=mask_str,
        ](
            q,
            k,
            v,
            buffer_row_offsets,
            cache_offsets,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# Manually fused MLA decode branch (BF16)
# ===-----------------------------------------------------------------------===#


def mla_decode_branch_bf16[
    collection_t: KVCollectionT,
    //,
    mask_str: StaticString,
    kv_input_fn: def[width: Int](IndexList[2]) capturing -> SIMD[
        DType.bfloat16, width
    ],
    target: StaticString = "cpu",
    sparse_mla: Bool = False,
](
    output: TileTensor[mut=True, .bfloat16, address_space=.GENERIC, ...],
    q: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    freqs_cis: TileTensor[_, address_space=.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space=.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    w_uk: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    w_uv: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    ctx: DeviceContext,
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    attn_sink_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    # Capturable-graph scalar forwarded from the MoGG op input list.
    num_partitions_in: Optional[Int] = None,
) raises:
    """BF16 MLA decode path.

    Applies RoPE and RMSNorm, projects q_nope to latent space, concatenates with
    q_rope, and runs decode.

    Parameters:
        collection_t: Type of the KV collection (inferred).
        mask_str: Mask variant.
        kv_input_fn: Input lambda function to load the KV latent values. Shape:
            [tot_seq_len, cache_head_dim]. Where cache_head_dim = kv_lora_rank
            + qk_rope_head_dim.
        target: Target device (defaults to "cpu").
        sparse_mla: Whether to use sparse MLA (defaults to False).

    Args:
        output: Output tensor of shape [tot_seq_len, num_heads, v_head_dim].
        q: Combined query tensor containing both nope and rope parts. Shape:
            [tot_seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim].
        input_row_offsets: Indicates where each request starts and ends in
            `q`. Shape: [num_batches + 1].
        freqs_cis: Precomputed RoPE frequency values for rotary position
            embeddings. Shape: [max_seq_len, qk_rope_head_dim].
        kv_norm_gamma: RMSNorm gamma weights for normalizing the KV cache.
            Shape: [kv_lora_rank].
        kv_collection: Paged KV Cache object.
        layer_idx: Layer index.
        scale: Scale for the attention calculation.
        epsilon: Small constant for numerical stability in RMSNorm.
        w_uk: Weight matrix for projecting the non-rope part of each query head
            to KV latent space. Shape: [num_heads, kv_latent_dim,
            qk_nope_head_dim].
        w_uv: Weight matrix for projecting the output of the attention back to
            each head's original space. Shape: [num_heads, v_head_dim,
            kv_latent_dim].
        scalar_args_buf: Packed MLA dispatch metadata buffer.
        ctx: Device context.
        d_indices: Sparse decode packed indices (null when dense).
        indices_stride: Row stride in `d_indices` (defaults to 0).
        topk_lengths: Per-batch valid top-k counts.
        attn_sink_ptr: Optional per-batch attention sink weights.
        num_partitions_in: Capturable-graph num_partitions override.
    """
    comptime kv_params = collection_t.kv_params
    comptime assert kv_params.is_mla, "kv_params.is_mla should be true"
    comptime assert kv_params.num_heads == 1, "kv_params.num_heads should be 1"

    comptime num_heads = q.static_shape[1]
    comptime q_head_dim = q.static_shape[2]
    comptime v_head_dim = output.static_shape[2]
    comptime k_cache_dim = kv_params.head_size

    comptime assert (
        w_uk.shape_known and w_uv.shape_known
    ), "w_uk and w_uv's shapes should be static"
    comptime kv_latent_dim = w_uk.static_shape[1]
    comptime qk_rope_head_dim = k_cache_dim - kv_latent_dim
    comptime qk_nope_head_dim = q_head_dim - qk_rope_head_dim
    comptime assert (
        qk_rope_head_dim == 0 or freqs_cis.static_shape[1] == qk_rope_head_dim
    ), "freqs_cis's width should be k_cache_dim - kv_latent_dim."
    comptime assert (
        w_uk.static_shape[2] == qk_nope_head_dim
    ), "w_uk.static_shape[2] should be equal to qk_nope_head_dim"
    comptime assert (
        w_uv.static_shape[2] == kv_latent_dim
    ), "w_uv.static_shape[2] should be equal to kv_latent_dim"
    comptime assert (
        w_uv.static_shape[1] == v_head_dim
    ), "w_uv.static_shape[1] should be equal to v_head_dim"

    var seq_len = Int(q.dim(0))
    if seq_len == 0:
        return

    # Sparse decode over a unit-scale FP8 latent cache on SM100 stages Q in
    # FP8 so the dispatch routes to the native-FP8 sparse kernel
    # (MLA_SM100_Decode_Sparse_QKV_FP8, SWIZZLE_64B FP8 Q TMA). Other sparse
    # configs keep bf16 Q: the BF16/converter sparse kernels' Q TMA uses
    # SWIZZLE_128B over BK=576 elements, which requires bf16 (1152 B).
    # Non-sparse bf16 decode may still stage Q in the KV cache dtype when the
    # cache is FP8.
    comptime native_fp8_sparse = (
        sparse_mla
        and collection_t.CacheType.dtype == .float8_e4m3fn
        and not collection_t.CacheType.quantization_enabled
        and _is_sm10x_gpu(ctx.default_device_info)
    )
    comptime mla_decode_input_dtype = (
        DType.bfloat16 if (
            sparse_mla and not native_fp8_sparse
        ) else collection_t.CacheType.dtype
    )
    var mla_decode_input_buf = ctx.enqueue_create_buffer[
        mla_decode_input_dtype
    ](seq_len * num_heads * k_cache_dim)
    var mla_decode_input = TileTensor(
        mla_decode_input_buf,
        row_major(seq_len, Idx[num_heads], Idx[k_cache_dim]),
    )

    # =========================================================================#
    # QK RoPE and K cache RMSNorm                                              #
    # =========================================================================#

    # Create a view of the `q` tensor that only contains the last
    # qk_rope_head_dim columns of each Q head.
    var q_rope = TileTensor(
        q.ptr + qk_nope_head_dim,
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * q_head_dim], Idx[q_head_dim], Idx[1]),
        ),
    )

    # Create a view of the `mla_decode_input` tensor that only contains the last
    # qk_rope_head_dim columns of each Q head.
    var mla_decode_input_rope = TileTensor(
        mla_decode_input.ptr + kv_latent_dim,
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * k_cache_dim], Idx[k_cache_dim], Idx[1]),
        ),
    )

    mla_fused_rope_rmsnorm_quantization[kv_input_fn=kv_input_fn](
        mla_decode_input_rope,
        q_rope,
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        kv_collection,
        layer_idx,
        epsilon,
        ctx,
    )

    # =========================================================================#
    # Project the non-rope part of each query head to kv_latent_dim            #
    # =========================================================================#

    # Create a view of the `q` tensor that only contains the first
    # qk_nope_head_dim columns of each Q head. Also transposed to
    # [num_heads, seq_len, qk_nope_head_dim].
    var q_nope_t = TileTensor(
        q.ptr,
        TileLayout(
            (Idx[num_heads], seq_len, Idx[qk_nope_head_dim]),
            (Idx[q_head_dim], Idx[num_heads * q_head_dim], Idx[1]),
        ),
    )

    # Then create a view of the mla_decode_input tensor that only contains the
    # first kv_latent_dim columns of each Q head.
    var mla_decode_input_nope = TileTensor(
        mla_decode_input.ptr,
        TileLayout(
            (Idx[num_heads], seq_len, Idx[kv_latent_dim]),
            (Idx[k_cache_dim], Idx[num_heads * k_cache_dim], Idx[1]),
        ),
    )

    _batched_matmul_gpu[transpose_b=True](
        mla_decode_input_nope, q_nope_t, w_uk, ctx
    )

    # Perform MLA decode
    var raw_output_buf = ctx.enqueue_create_buffer[.bfloat16](
        seq_len * num_heads * kv_latent_dim
    )
    var raw_output = TileTensor(
        raw_output_buf,
        row_major(seq_len, Idx[num_heads], Idx[kv_latent_dim]),
    )

    generic_flare_mla_decode_kv_cache_ragged[
        target=target,
        mask_str=mask_str,
        sparse_mla=sparse_mla,
    ](
        mla_decode_input,
        input_row_offsets,
        kv_collection,
        layer_idx,
        scale,
        raw_output,
        scalar_args_buf,
        ctx,
        d_indices=d_indices,
        indices_stride=indices_stride,
        topk_lengths=topk_lengths,
        attn_sink_ptr=attn_sink_ptr,
        num_partitions_in=num_partitions_in,
    )

    # Create a view of the raw output tensor with logical shape
    # [num_heads, seq_len, kv_latent_dim], and map directly to
    # [seq_len, num_heads, kv_latent_dim] physical memory.
    var raw_output_t = TileTensor(
        raw_output_buf,
        TileLayout(
            (Idx[num_heads], seq_len, Idx[kv_latent_dim]),
            (Idx[kv_latent_dim], Idx[num_heads * kv_latent_dim], Idx[1]),
        ),
    )

    # Create a view of the output tensor with logical shape
    # [num_heads, seq_len, v_head_dim], and map directly to
    # [seq_len, num_heads, v_head_dim] physical memory.
    var output_t = TileTensor(
        output.ptr,
        TileLayout(
            (Idx[num_heads], seq_len, Idx[v_head_dim]),
            (Idx[v_head_dim], Idx[num_heads * v_head_dim], Idx[1]),
        ),
    )

    _batched_matmul_gpu[transpose_b=True](output_t, raw_output_t, w_uv, ctx)


# ===-----------------------------------------------------------------------===#
# Manually fused MLA sparse prefill branch (BF16)
# ===-----------------------------------------------------------------------===#


@always_inline
def mla_prefill_branch_sparse_bf16[
    collection_t: KVCollectionT,
    //,
    kv_input_fn: def[width: Int](IndexList[2]) capturing -> SIMD[
        DType.bfloat16, width
    ],
    indices_stride: Int,
    target: StaticString = "cpu",
](
    output: TileTensor[mut=True, .bfloat16, address_space=.GENERIC, ...],
    q: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    freqs_cis: TileTensor[_, address_space=.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space=.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    w_uk: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    w_uv: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    ctx: DeviceContext,
    d_indices: UnsafePointer[Int32, MutAnyOrigin],
    topk_lengths: UnsafePointer[Int32, MutAnyOrigin],
    attn_sink_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]],
) raises:
    """Sparse MLA prefill branch (DSv3.2/GLM absorbed shape, BF16 weights).

    BF16 analogue of `mla_prefill_branch_sparse_fp8`: reuses
    `mla_decode_branch_bf16`'s absorbed-Q construction (q_nope up-proj via `w_uk`
    + RoPE/RMSNorm) and the identical `w_uv` output up-projection, and swaps the
    attention call to the existing `mla_sm100_prefill_sparse` kernel over the
    paged BF16 latent cache. The caller (`.sparse` op) has already remapped
    `d_indices` from logical to physical rows, so they are passed straight
    through. Only supported for a BF16 KV cache.
    """
    comptime kv_params = collection_t.kv_params
    comptime assert kv_params.is_mla, "kv_params.is_mla should be true"
    comptime assert kv_params.num_heads == 1, "kv_params.num_heads should be 1"

    comptime num_heads = q.static_shape[1]
    comptime q_head_dim = q.static_shape[2]
    comptime v_head_dim = output.static_shape[2]
    comptime k_cache_dim = kv_params.head_size

    comptime assert (
        w_uk.shape_known and w_uv.shape_known
    ), "w_uk and w_uv's shapes should be static"
    comptime kv_latent_dim = w_uk.static_shape[1]
    comptime qk_rope_head_dim = k_cache_dim - kv_latent_dim
    comptime qk_nope_head_dim = q_head_dim - qk_rope_head_dim
    comptime assert (
        qk_rope_head_dim == 0 or freqs_cis.static_shape[1] == qk_rope_head_dim
    ), "freqs_cis's width should be k_cache_dim - kv_latent_dim."
    comptime assert (
        w_uk.static_shape[2] == qk_nope_head_dim
    ), "w_uk.static_shape[2] should be equal to qk_nope_head_dim"
    comptime assert (
        w_uv.static_shape[1] == v_head_dim
    ), "w_uv.static_shape[1] should be equal to v_head_dim"

    var seq_len = Int(q.dim(0))
    if seq_len == 0:
        return

    # Unit-scale FP8 latent cache on SM100 stages Q in FP8 and runs the
    # native-FP8 sparse prefill kernel (QK^T and P*V in FP8, no dequant); the
    # kernel supports num_q_heads == 128 (cta_group=2) or a multiple of 8 in
    # (0, 64] (cta_group=1). Other configs keep bf16 Q.
    comptime native_fp8_sparse = (
        collection_t.CacheType.dtype == .float8_e4m3fn
        and not collection_t.CacheType.quantization_enabled
        and _is_sm10x_gpu(ctx.default_device_info)
        and (num_heads == 128 or (num_heads <= 64 and num_heads % 8 == 0))
    )
    comptime q_staging_dtype = (
        DType.float8_e4m3fn if native_fp8_sparse else DType.bfloat16
    )
    var mla_decode_input_buf = ctx.enqueue_create_buffer[q_staging_dtype](
        seq_len * num_heads * k_cache_dim
    )
    var mla_decode_input = TileTensor(
        mla_decode_input_buf,
        row_major(seq_len, Idx[num_heads], Idx[k_cache_dim]),
    )

    # Transposed view [num_heads, seq_len, qk_nope_head_dim] of the first
    # qk_nope_head_dim columns of each Q head.
    var q_nope_t = TileTensor(
        q.ptr,
        TileLayout(
            (Idx[num_heads], seq_len, Idx[qk_nope_head_dim]),
            (Idx[q_head_dim], Idx[num_heads * q_head_dim], Idx[1]),
        ),
    )
    var mla_decode_input_nope = TileTensor(
        mla_decode_input.ptr,
        TileLayout(
            (Idx[num_heads], seq_len, Idx[kv_latent_dim]),
            (Idx[k_cache_dim], Idx[num_heads * k_cache_dim], Idx[1]),
        ),
    )
    _batched_matmul_gpu[transpose_b=True](
        mla_decode_input_nope, q_nope_t, w_uk, ctx
    )

    var q_rope = TileTensor(
        q.ptr + qk_nope_head_dim,
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * q_head_dim], Idx[q_head_dim], Idx[1]),
        ),
    )
    var mla_decode_input_rope = TileTensor(
        mla_decode_input.ptr + kv_latent_dim,
        TileLayout(
            (seq_len, Idx[num_heads], Idx[qk_rope_head_dim]),
            (Idx[num_heads * k_cache_dim], Idx[k_cache_dim], Idx[1]),
        ),
    )
    mla_fused_rope_rmsnorm_quantization[kv_input_fn=kv_input_fn](
        mla_decode_input_rope,
        q_rope,
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        kv_collection,
        layer_idx,
        epsilon,
        ctx,
    )

    var raw_output_buf = ctx.enqueue_create_buffer[.bfloat16](
        seq_len * num_heads * kv_latent_dim
    )
    var raw_output = TileTensor(
        raw_output_buf,
        row_major(seq_len, Idx[num_heads], Idx[kv_latent_dim]),
    )

    # `d_indices` / `topk_lengths` are int32 buffers reinterpreted as uint32:
    # invalid `-1` slots become 0xFFFFFFFF and are rejected by the kernel's
    # `idx >= 0` gather producer.
    var indices_tt = TileTensor(
        d_indices.bitcast[UInt32](),
        row_major(seq_len * indices_stride),
    )
    var topk_lengths_tt = TileTensor(
        topk_lengths.bitcast[UInt32](),
        row_major(seq_len),
    )
    var attn_sink_opt = Optional[UnsafePointer[Float32, ImmutAnyOrigin]](None)
    if attn_sink_ptr:
        attn_sink_opt = UnsafePointer[Float32, ImmutAnyOrigin](
            attn_sink_ptr.value()
        )

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    comptime if collection_t.CacheType.dtype.is_float8():
        comptime if native_fp8_sparse:
            # Native-FP8 sparse prefill: Q was staged in FP8 above, and the
            # kernel reads the unit-scale FP8 latent cache with no dequant.
            # Config knobs follow the kernel's validated tactics: head=128
            # runs the 2-CTA f8f6f4 tile at b_topk=128/num_mbars=2; head<=64
            # runs the single-CTA shared-KV tile at b_topk=64 (its only
            # correct PV mn-major layout) with num_mbars=4.
            comptime cta_group = 2 if num_heads == 128 else 1
            comptime b_topk = 128 if cta_group == 2 else 64
            comptime num_mbars = 2 if cta_group == 2 else 4
            comptime config = MLASparseConfig[
                DType.bfloat16,
                b_topk_=b_topk,
                num_mbars_=num_mbars,
                cta_group_=cta_group,
            ](
                num_q_heads=num_heads,
                num_kv_heads=1,
                # Tile stays laid out for 576. The kernel zero-fills the NoPE tail.
                qk_depth=kv_latent_dim + 64,
                input_qk_depth=k_cache_dim,
                v_depth=kv_latent_dim,
                indices_stride=indices_stride,
                group=num_heads,
            )
            mla_prefill_sparse_qkv_fp8[
                config=config,
                group=num_heads,
                q_depth=k_cache_dim,
                scale_block_size=0,
            ](
                raw_output,
                mla_decode_input,
                k_cache,
                indices_tt,
                topk_lengths_tt,
                attn_sink_opt,
                scale,
                Int32(indices_stride),
                ctx,
            )
        else:
            # FP8 latent cache with bf16 Q: run the FP8 sparse-prefill kernel
            # directly over the quantized cache (no BF16 staging). Today the
            # cache carries no dequant scales (scale_dtype=int8 => quantization
            # disabled), so read at unit scale (scale_block_size=0), mirroring
            # the sparse-DECODE kernel's read. scales_ptr is unused at
            # scale_block_size=0; pass a non-null dummy (SnapMLA/SERVOPT-1094
            # will supply real scales + a positive scale_block_size here once
            # the cache carries them).
            var dummy_scales = (
                raw_output_buf.unsafe_ptr()
                .bitcast[Float32]()
                .as_unsafe_any_origin()
            )
            mla_sm100_prefill_sparse_fp8[
                num_q_heads=num_heads,
                qk_depth=k_cache_dim,
                v_depth=kv_latent_dim,
                indices_stride=indices_stride,
                scale_block_size=0,
            ](
                raw_output,
                mla_decode_input,
                k_cache,
                indices_tt,
                topk_lengths_tt,
                attn_sink_opt,
                dummy_scales,
                scale,
                ctx,
            )
    else:
        mla_sm100_prefill_sparse[
            num_q_heads=num_heads,
            qk_depth=k_cache_dim,
            v_depth=kv_latent_dim,
            indices_stride=indices_stride,
        ](
            raw_output,
            mla_decode_input,
            k_cache,
            indices_tt,
            topk_lengths_tt,
            attn_sink_opt,
            scale,
            ctx,
        )

    var raw_output_t = TileTensor(
        raw_output_buf,
        TileLayout(
            (Idx[num_heads], seq_len, Idx[kv_latent_dim]),
            (Idx[kv_latent_dim], Idx[num_heads * kv_latent_dim], Idx[1]),
        ),
    )
    var output_t = TileTensor(
        output.ptr,
        TileLayout(
            (Idx[num_heads], seq_len, Idx[v_head_dim]),
            (Idx[v_head_dim], Idx[num_heads * v_head_dim], Idx[1]),
        ),
    )
    _batched_matmul_gpu[transpose_b=True](output_t, raw_output_t, w_uv, ctx)


# ===-----------------------------------------------------------------------===#
# MLA prefill-decode graph (BF16)
# ===-----------------------------------------------------------------------===#


@always_inline
def mla_prefill_decode_graph_bf16[
    collection_t: KVCollectionT,
    //,
    mask_str: StaticString,
    kv_input_fn: def[width: Int](IndexList[2]) capturing -> SIMD[
        DType.bfloat16, width
    ],
    target: StaticString = "cpu",
    sparse_mla: Bool = False,
    sparse_indices_stride: Int = 0,
](
    output: TileTensor[mut=True, .bfloat16, address_space=.GENERIC, ...],
    q: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    freqs_cis: TileTensor[_, address_space=.GENERIC, ...],
    kv_norm_gamma: TileTensor[_, address_space=.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    epsilon: Float32,
    buffer_row_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    cache_offsets: TileTensor[mut=True, .uint32, address_space=.GENERIC, ...],
    buffer_length: Int,
    max_seq_len: Int,
    w_k: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    w_uk: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    w_uv: TileTensor[.bfloat16, address_space=.GENERIC, ...],
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    ctx: DeviceContext,
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    attn_sink_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    # Capturable-graph scalar forwarded from the MoGG op input list.
    num_partitions_in: Optional[Int] = None,
) raises:
    """BF16 MLA prefill/decode graph.

    Dispatches to prefill or decode based on max sequence length in the batch.

    Parameters:
        collection_t: Type of the KV collection (inferred).
        mask_str: Mask variant.
        kv_input_fn: Input lambda function to load the KV latent values. Shape:
            [tot_seq_len, cache_head_dim]. Where cache_head_dim = kv_lora_rank
            + qk_rope_head_dim.
        target: Target device (defaults to "cpu").
        sparse_mla: Whether to use sparse MLA (defaults to False).
        sparse_indices_stride: Row stride of the sparse decode index buffer
            (defaults to 0).

    Args:
        output: Output tensor of shape [tot_seq_len, num_heads, v_head_dim].
        q: Combined query tensor containing both nope and rope parts. Shape:
            [tot_seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim].
        input_row_offsets: Indicates where each request starts and ends in
            `q`. Shape: [num_batches + 1].
        freqs_cis: Precomputed RoPE frequency values for rotary position
            embeddings. Shape: [max_seq_len, qk_rope_head_dim].
        kv_norm_gamma: RMSNorm gamma weights for normalizing the KV cache.
            Shape: [kv_lora_rank].
        kv_collection: Paged KV Cache object.
        layer_idx: Layer index.
        scale: Scale for the attention calculation.
        epsilon: Small constant for numerical stability in RMSNorm.
        buffer_row_offsets: Indicates where each request's KV latent values
            should be stored in the contiguous K buffer. This is a 1D tensor
            of shape [num_batches + 1].
        cache_offsets: Indicates the starting token position in the KV cache
            from which to copy KV latent values for each request. This is a 1D
            tensor of shape [num_batches + 1].
        buffer_length: The total number of tokens in the KV cache. Scalar.
        max_seq_len: Maximum sequence length in the batch, used to select
            prefill versus decode.
        w_k: Weight matrix for up-projecting the latent cache to full K. Shape:
            [num_heads * qk_nope_head_dim, kv_latent_dim].
        w_uk: Weight matrix for projecting the non-rope part of each query head
            to KV latent space. Shape: [num_heads, kv_latent_dim,
            qk_nope_head_dim].
        w_uv: Weight matrix for projecting the output of the attention back to
            each head's original space. Shape: [num_heads, v_head_dim,
            kv_latent_dim].
        scalar_args_buf: Packed MLA dispatch metadata buffer.
        ctx: Device context.
        d_indices: Optional device pointer to packed int32 physical KV row
            indices for sparse decode (defaults to None).
        indices_stride: Stride between batch rows in `d_indices` (defaults
            to 0).
        topk_lengths: Optional per-batch valid top-k counts (defaults to
            None).
        attn_sink_ptr: Optional per-batch attention sink weights (defaults
            to None).
        num_partitions_in: Capturable-graph num_partitions override (defaults
            to None).
    """
    var seq_len = q.dim(0)

    if seq_len == 0:
        return

    # The fold runs in the cache dtype (the decode branch quantizes Q to
    # `collection_t.CacheType.dtype`), so the decode-vs-prefill threshold keys on
    # the cache dtype, not the BF16 compute dtype: an FP8 cache with
    # num_heads<=AMD_MLA_DECODE_FOLD_MAX_NUM_HEADS routes S>1 to the decode fold
    # (prefill can't serve MTP); a BF16 cache or num_heads>16 returns 1 and routes
    # S>1 to prefill (no large-head decode fold exists). Mirrors
    # `mla_prefill_decode_graph_fp8`.
    if (
        max_seq_len
        <= mla_decode_max_seq_len[
            collection_t.CacheType.dtype, q.static_shape[1]
        ]()
    ):
        mla_decode_branch_bf16[
            mask_str=mask_str,
            kv_input_fn=kv_input_fn,
            target=target,
            sparse_mla=sparse_mla,
        ](
            output,
            q,
            input_row_offsets,
            freqs_cis,
            kv_norm_gamma,
            kv_collection,
            layer_idx,
            scale,
            epsilon,
            w_uk,
            w_uv,
            scalar_args_buf,
            ctx,
            d_indices,
            indices_stride,
            topk_lengths,
            attn_sink_ptr,
            num_partitions_in,
        )
    else:
        comptime if sparse_mla:
            # Sparse MLA prefill for BOTH bf16 and fp8 latent caches: the
            # branch comptime-dispatches the attention kernel on the cache
            # dtype (fp8 cache => mla_sm100_prefill_sparse_fp8 read at unit
            # scale, mirroring sparse decode; bf16 cache =>
            # mla_sm100_prefill_sparse). Replaces the dense FP8-cache fallback
            # that previously ran here.
            mla_prefill_branch_sparse_bf16[
                kv_input_fn=kv_input_fn,
                indices_stride=sparse_indices_stride,
                target=target,
            ](
                output,
                q,
                input_row_offsets,
                freqs_cis,
                kv_norm_gamma,
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                w_uk,
                w_uv,
                ctx,
                d_indices.value(),
                topk_lengths.value(),
                attn_sink_ptr,
            )
        else:
            # Dense prefill for NON-sparse MLA only. Sparse MLA (both bf16 and
            # fp8 caches) is handled by the sparse branch above.
            mla_prefill_branch_bf16[
                mask_str=mask_str,
                kv_input_fn=kv_input_fn,
                target=target,
            ](
                output,
                q,
                input_row_offsets,
                freqs_cis,
                kv_norm_gamma,
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                buffer_row_offsets,
                cache_offsets,
                buffer_length,
                w_k,
                w_uv,
                ctx,
            )
