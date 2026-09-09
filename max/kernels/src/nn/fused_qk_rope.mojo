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
"""Provides fused query/key rotary position embedding (RoPE) kernels with integrated KV-cache writes."""

from std.collections import OptionalReg
from std.math import gcd
from std.sys.info import _current_target, align_of, simd_width_of

from max.algorithm.functional import elementwise
from std.utils.numerics import get_accum_type
from std.complex import ComplexSIMD
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.info import is_cpu
from internal_utils.fp8_utils import cast_saturating
from kv_cache.types import KVCacheT, KVCollectionT
from layout import (
    Coord,
    CoordLike,
    Idx,
    RowMajorLayout,
    TensorLayout,
    TileTensor,
    coord_to_index_list,
)
from nn._ragged_utils import get_batch_from_row_offsets

from std.utils import IndexList


@always_inline
def rope_value[
    dtype: DType,
    freq_dtype: DType,
    width: SIMDLength,
](val: SIMD[dtype, width], freq: SIMD[freq_dtype, width]) -> SIMD[dtype, width]:
    """Applies a rotary position embedding transformation to a SIMD vector.

    Deinterleaves the input into real and imaginary parts, multiplies them as
    complex numbers against the frequency coefficients, and reinterleaves the
    result back into the original dtype.

    Parameters:
        dtype: Element type of the input and output SIMD vector (inferred).
        freq_dtype: Element type of the frequency coefficients (inferred).
        width: Number of elements in the SIMD vector (inferred).

    Args:
        val: The input SIMD vector with interleaved real and imaginary parts.
        freq: The frequency coefficients with interleaved real and imaginary parts.

    Returns:
        The RoPE-transformed SIMD vector in the original dtype.
    """
    var x_re, x_im = val.cast[freq_dtype]().deinterleave()
    var f_re, f_im = freq.deinterleave()
    var r = ComplexSIMD(x_re, x_im) * ComplexSIMD(f_re, f_im)
    return rebind[SIMD[dtype, width]](r.re.interleave(r.im).cast[dtype]())


# In GGUF, weights are organized as real, imag, real, imag, real, imag, …,
# while in safetensors, the data is stored as real, …, real, imag, …, imag.
# This function return the indices for the real and imaginary part.
@always_inline
def get_safetensors_idx(head_dim_idx: Int, head_size: Int) -> Tuple[Int, Int]:
    """Returns the real and imaginary element indices for a safetensors layout.

    Safetensors stores RoPE weights as all real components followed by all
    imaginary components, so the imaginary index is offset by half the head
    size.

    Args:
        head_dim_idx: The dimension index within the head.
        head_size: The total size of the head dimension.

    Returns:
        A tuple of the real index and the imaginary index.
    """
    return (head_dim_idx // 2, head_dim_idx // 2 + head_size // 2)


@always_inline
def get_identity_rope_coeff[width: Int, dtype: DType]() -> SIMD[dtype, width]:
    """Returns a SIMD vector representing an identity RoPE coefficient.

    Creates a SIMD vector with real parts set to 1 and imaginary parts set to
    0, effectively making the RoPE transformation an identity operation.

    Parameters:
        width: Number of elements in the returned SIMD vector.
        dtype: Element type of the returned SIMD vector.

    Returns:
        A SIMD vector of interleaved 1.0 real and 0.0 imaginary coefficients.
    """
    # Creates a SIMD vector with real parts set to 1 and imaginary parts to
    # 0, effectively making the RoPE transformation an identity operation.
    return rebind[SIMD[dtype, width]](
        SIMD[dtype, width // 2](1).interleave(SIMD[dtype, width // 2](0))
    )


@always_inline
def rope_q_proj[
    dtype: DType,
    freq_dtype: DType,
    rank: Int,
    width: SIMDLength,
    output_dtype: DType,
    //,
    *,
    interleaved: Bool,
    has_nope_prefix: Bool = False,
    rope_dim: Int = 0,
    alignment: Int = align_of[SIMD[dtype, width]](),
](
    q_proj: TileTensor[dtype, ...],
    output: TileTensor[mut=True, output_dtype, ...],
    idx: IndexList[rank],
    freq_val: SIMD[freq_dtype, width],
    head_size: Int,
):
    """Applies RoPE to a query projection tile and stores the result.

    Loads the query projection values at the given coordinate, applies the
    rotary position embedding via `rope_value`, and writes the transformed
    result to the output tile. Supports both interleaved and split (real and
    imaginary stored separately) layouts, and optionally leaves a nope prefix
    region unrotated.

    Parameters:
        dtype: Element type of the `q_proj` tile tensor (inferred).
        freq_dtype: Element type of the `freq_val` frequency
            coefficients (inferred).
        rank: Rank of the index list and tile tensors (inferred).
        width: Number of elements per SIMD vector (inferred).
        output_dtype: Element type of the `output` tile tensor (inferred).
        interleaved: Whether the RoPE weights use interleaved real and
            imaginary layout.
        has_nope_prefix: Whether a leading prefix of the head dimension is
            left unrotated (defaults to `False`).
        rope_dim: Number of trailing head dimensions that undergo RoPE when
            `has_nope_prefix` is set (defaults to 0).
        alignment: Memory alignment in bytes for tile loads and stores
            (defaults to the natural alignment of `SIMD[dtype, width]`).

    Args:
        q_proj: The query projection tile tensor to read from.
        output: The mutable output tile tensor to write the rotated result.
        idx: The index list identifying the load and store coordinate.
        freq_val: The RoPE frequency coefficients for this position.
        head_size: The size of each attention head dimension.
    """
    comptime assert q_proj.flat_rank == rank
    comptime assert output.flat_rank == rank
    var coord = Coord(idx)
    comptime assert q_proj.flat_rank >= coord.flat_rank
    comptime assert output.flat_rank >= coord.flat_rank

    comptime width_2 = width // 2
    comptime half_alignment = align_of[
        SIMD[dtype, width_2]
    ]() if alignment == align_of[SIMD[dtype, width]]() else alignment

    comptime if interleaved:
        var val_inter = q_proj.load[width=width, alignment=alignment](coord)
        var res_inter = cast_saturating[output_dtype](
            rope_value(val_inter, freq_val)
        )
        output.store[alignment=alignment](coord, res_inter)
    else:
        comptime if has_nope_prefix:
            if idx[rank - 1] >= rope_dim:
                var val_pass = q_proj.load[width=width, alignment=alignment](
                    coord
                )
                output.store[alignment=alignment](
                    coord, cast_saturating[output_dtype](val_pass)
                )
                return

        var split_size: Int
        comptime if has_nope_prefix:
            split_size = rope_dim
        else:
            split_size = head_size

        var indices = get_safetensors_idx(idx[rank - 1], split_size)
        var pos_re = idx
        var pos_im = idx
        pos_re[rank - 1] = indices[0]
        pos_im[rank - 1] = indices[1]

        var coord_re = Coord(pos_re)
        var coord_im = Coord(pos_im)

        var val = rebind[SIMD[dtype, width]](
            q_proj.load[width=width_2, alignment=half_alignment](
                coord_re
            ).interleave(
                q_proj.load[width=width_2, alignment=half_alignment](coord_im)
            )
        )

        var res = cast_saturating[output_dtype](rope_value(val, freq_val))
        var output_re, output_im = res.deinterleave()
        output.store[alignment=half_alignment](coord_re, output_re)
        output.store[alignment=half_alignment](coord_im, output_im)


@always_inline
def rope_k_cache[
    freq_dtype: DType,
    cache_t: KVCacheT,
    width: SIMDLength,
    //,
    *,
    interleaved: Bool,
    has_nope_prefix: Bool = False,
    rope_prefix_dim: Int = 0,
](
    k_cache: cache_t,
    b_idx: Int,
    h_idx: Int,
    s_idx: Int,
    d_idx: Int,
    freq_val: SIMD[freq_dtype, width],
    head_size: Int,
):
    """Applies RoPE to a key cache entry and stores the result back in the cache.

    Loads the key cache values at the given batch, head, sequence, and
    dimension indices, applies the rotary position embedding via
    `rope_value`, and writes the transformed result back to the cache.
    Supports both interleaved and split layouts, and optionally leaves a nope
    prefix region unrotated.

    Parameters:
        freq_dtype: Element type of the `freq_val` frequency
            coefficients (inferred).
        cache_t: KV cache view type used to load and store key cache
            entries (inferred).
        width: Number of elements per SIMD vector (inferred).
        interleaved: Whether the RoPE weights use interleaved real and
            imaginary layout.
        has_nope_prefix: Whether a leading prefix of the head dimension is
            left unrotated (defaults to `False`).
        rope_prefix_dim: Number of trailing head dimensions that undergo
            RoPE when `has_nope_prefix` is set (defaults to 0).

    Args:
        k_cache: The KV cache key view to read from and write to.
        b_idx: The batch index into the cache.
        h_idx: The head index into the cache.
        s_idx: The sequence position index into the cache.
        d_idx: The head dimension index into the cache.
        freq_val: The RoPE frequency coefficients for this position.
        head_size: The size of each attention head dimension.
    """
    comptime width_2 = width // 2
    comptime cache_type = cache_t.dtype
    # TODO: Remove this once FP8 KVCache is supported (KERN-2394).
    comptime accum_type = get_accum_type[cache_type]()

    comptime if interleaved:
        var val_inter = k_cache.load[width=width](
            b_idx, h_idx, s_idx, d_idx
        ).cast[accum_type]()
        var res_inter = rope_value(val_inter, freq_val).cast[cache_type]()
        k_cache.store(b_idx, h_idx, s_idx, d_idx, res_inter)
    else:
        var split_size: Int
        comptime if has_nope_prefix:
            split_size = rope_prefix_dim
        else:
            split_size = head_size

        var h_re, h_im = get_safetensors_idx(d_idx, split_size)

        var val = rebind[SIMD[accum_type, width]](
            k_cache.load[width=width_2](b_idx, h_idx, s_idx, h_re)
            .cast[accum_type]()
            .interleave(
                k_cache.load[width=width_2](b_idx, h_idx, s_idx, h_im).cast[
                    accum_type
                ]()
            )
        )

        var res = rope_value(val, freq_val).cast[cache_type]()
        var output_re, output_im = res.deinterleave()
        k_cache.store(b_idx, h_idx, s_idx, h_re, output_re)
        k_cache.store(b_idx, h_idx, s_idx, h_im, output_im)


@always_inline
def fused_qk_rope[
    dtype: DType,
    collection_t: KVCollectionT,
    //,
    cache_t: KVCacheT,
    *,
    interleaved: Bool,
    target: StaticString,
](
    q_proj: TileTensor[dtype, ...],
    kv_collection: collection_t,
    freqs_cis: TileTensor[dtype, ...],
    layer_idx: UInt32,
    valid_lengths: TileTensor[.uint32, ...],
    output: TileTensor[mut=True, dtype, ...],
    context: DeviceContext,
) raises:
    """Applies RoPE to query and key tensors.

    Parameters:
        dtype: The element type of `q_proj`, `freqs_cis`, and
            `output` (inferred).
        collection_t: The KV cache collection type holding the key
            cache (inferred).
        cache_t: The KV cache type used to store and load key cache entries.
        interleaved: Whether the RoPE weights use interleaved real and
            imaginary layout.
        target: The compilation target string for the kernel.

    Args:
        q_proj: Query projection tensor of shape [batch, seq_len, n_heads, head_dim].
        kv_collection: The KV cache collection containing the key cache.
        freqs_cis: Frequency tensor for RoPE of shape [max_seq_len, head_dim].
        layer_idx: The layer index for accessing the correct cache.
        valid_lengths: Tensor of shape [batch] containing the valid length for each
            sequence. RoPE is only applied to positions within these lengths.
        output: Output tensor for Q with RoPE applied, same shape as q_proj.
        context: Optional device context for GPU execution.
    """
    comptime assert q_proj.flat_rank == 4
    comptime assert freqs_cis.flat_rank == 2
    comptime assert output.flat_rank == 4
    comptime assert valid_lengths.flat_rank == 1

    comptime kv_params = cache_t.kv_params

    var batch_size = Int(q_proj.dim[0]())
    var new_seq_len = Int(q_proj.dim[1]())
    comptime num_q_heads = Int(q_proj.static_shape[2])
    comptime num_k_heads = kv_params.num_heads
    comptime head_size = Int(q_proj.static_shape[3])

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))

    # TODO: This elementwise body captures a KV cache view (`CacheType`),
    # which fails codegen when stored into a unified closure ('pop.store'
    # pointer element-type verification). Keep using the deprecated
    # parameter-closure overload until cache captures in unified closures are
    # supported.
    @always_inline
    @__parameter
    @__copy_capture(k_cache, valid_lengths)
    def rope_fn[width: Int, alignment: Int = 1](idx: Coord):
        comptime assert idx.rank == 4, "Invalid rank passed to rope kernel"

        comptime if width == 1:
            return
        else:
            var bs_idx = Int(idx[0].value())
            var seq_idx = Int(idx[1].value())

            # Check if this position is within the valid length for this batch
            var valid_len = Int(valid_lengths[bs_idx])
            if seq_idx >= valid_len:
                return

            # post_seq_idx: sum of start_pos (cache_lengths[batch_idx]) and
            # seq_idx (idx[1]).
            var post_seq_idx = k_cache.cache_length(bs_idx) + seq_idx
            var head_idx = Int(idx[2].value())
            var head_dim_idx = Int(idx[3].value())

            # WARN assumes head_size % simd_width == 0
            # guarded by constrained statement below
            var is_q_proj = head_idx < num_q_heads
            comptime _alignment = 1 if is_cpu[target]() else align_of[
                SIMD[dtype, width]
            ]()
            var f_c_temp = freqs_cis.load[width=width, alignment=_alignment](
                (post_seq_idx, head_dim_idx)
            )

            if is_q_proj:
                rope_q_proj[interleaved=interleaved, alignment=_alignment](
                    q_proj,
                    output,
                    coord_to_index_list(idx),
                    f_c_temp,
                    head_size,
                )
            else:
                head_idx -= num_q_heads
                rope_k_cache[interleaved=interleaved](
                    k_cache,
                    bs_idx,
                    head_idx,
                    post_seq_idx,
                    head_dim_idx,
                    f_c_temp,
                    head_size,
                )

    var launch_shape = (
        batch_size,
        new_seq_len,
        num_q_heads + num_k_heads,  # concat q and k along head dim
        head_size,
    )
    comptime compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    comptime target_simd_width = simd_width_of[dtype, target=compile_target]()
    comptime kernel_simd_width = gcd(target_simd_width, kv_params.head_size)
    comptime assert kernel_simd_width >= 2, "invalid simd_width and head size"

    elementwise[
        func=rope_fn,
        simd_width=kernel_simd_width,
        target=target,
        _trace_description="fused_qk_rope",
    ](launch_shape, context)


@always_inline
def fused_qk_rope_ragged[
    dtype: DType,
    freq_dtype: DType,
    collection_t: KVCollectionT,
    //,
    cache_t: KVCacheT,
    *,
    interleaved: Bool,
    target: StaticString,
    mrope_types: TypeList[Trait=CoordLike, ...] = TypeList.of[
        Trait=CoordLike
    ](),
    mrope_section: Optional[Coord[*mrope_types]] = None,
    PositionIdsLayoutType: TensorLayout = RowMajorLayout[
        *Coord[Int64, Int64].element_types
    ],
](
    q_proj: TileTensor[dtype, ...],
    input_row_offsets: TileTensor[.uint32, ...],
    kv_collection: collection_t,
    freqs_cis: TileTensor[freq_dtype, ...],
    position_ids: OptionalReg[
        TileTensor[.uint32, PositionIdsLayoutType, ImmutAnyOrigin]
    ],
    layer_idx: UInt32,
    output: TileTensor[mut=True, dtype, ...],
    context: DeviceContext,
) raises:
    """Applies RoPE (Rotary Position Embedding) to query and key tensors.

    This function can applies RoPE only to the last `rope_dim` elements of each
    head, leaving the first `unroped_dim` elements unchanged. This is required
    for DeepSeek models where only part of each head undergoes rotary
    transformation.

    Parameters:
        dtype: The element type of `q_proj` and `output` (inferred).
        freq_dtype: The element type of `freqs_cis` (inferred).
        collection_t: The KV cache collection type holding the key
            cache (inferred).
        cache_t: The KV cache type used to store and load key cache entries.
        interleaved: Whether the RoPE weights use interleaved real and
            imaginary layout.
        target: The compilation target string for the kernel.
        mrope_types: The coordinate element types for multimodal RoPE
            sections (defaults to a single `CoordLike` type).
        mrope_section: Optional section boundaries splitting the head
            dimension into position-id groups for multimodal RoPE
            (defaults to `None`).
        PositionIdsLayoutType: The tensor layout of the `position_ids`
            tensor (defaults to `RowMajorLayout`).

    Args:
        q_proj: Query projection tensor of shape [total_tokens, n_heads, head_dim].
        input_row_offsets: Tensor of shape [batch + 1] marking where each
            batch's tokens start and end.
        kv_collection: The KV cache collection containing the key cache.
        freqs_cis: Frequency tensor for RoPE of shape [max_seq_len, rope_dim].
        position_ids: Optional position ids overriding cache-derived
            positions, of shape [n_sections, total_tokens].
        layer_idx: The layer index for accessing the correct cache.
        output: Output tensor for Q with RoPE applied, same shape as `q_proj`.
        context: Device context for GPU execution.
    """
    comptime assert q_proj.flat_rank == 3, "q_proj must be rank 3"
    comptime assert freqs_cis.flat_rank == 2, "freqs_cis must be rank 2"
    comptime assert output.flat_rank == 3, "output must be rank 3"
    comptime assert PositionIdsLayoutType.rank == 2
    comptime assert (
        input_row_offsets.flat_rank == 1
    ), "input_row_offsets must be rank 1"
    comptime kv_params = cache_t.kv_params
    comptime num_q_heads = Int(q_proj.static_shape[1])
    comptime num_k_heads = kv_params.num_heads
    comptime q_head_size = Int(q_proj.static_shape[2])
    comptime k_head_size = kv_params.head_size
    var batch_size = input_row_offsets.dim[0]() - 1

    # Add rope dimension parameters
    comptime rope_dim = Int(freqs_cis.static_shape[1])

    # Check if shape of freqs_cis matches head_size.
    # If not, we only rope the last `rope_dim` dimensions of each head.
    comptime unroped_dim = q_head_size - rope_dim
    comptime has_nope = unroped_dim > 0

    comptime assert freqs_cis.LayoutType._shape_types[
        1
    ].is_static_value, "Need static shape for freqs_cis"
    comptime assert rope_dim <= q_head_size and rope_dim <= k_head_size, (
        "rope_dim must be smaller or equal to head size, but got rope_dim = "
        + String(rope_dim)
        + " and head_size = "
        + String(k_head_size)
    )
    comptime has_nope_prefix = has_nope and not interleaved
    comptime if has_nope and not interleaved:
        comptime assert (
            rope_dim % 2 == 0
        ), "prefix partial RoPE rope_dim must be even for split layout"

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))

    # TODO: This elementwise body captures a KV cache view (`CacheType`),
    # which fails codegen when stored into a unified closure ('pop.store'
    # pointer element-type verification). Keep using the deprecated
    # parameter-closure overload until cache captures in unified closures are
    # supported.
    @always_inline
    @__parameter
    @__copy_capture(k_cache, batch_size, input_row_offsets, position_ids)
    def rope_fn[width: Int, alignment: Int = 1](idx: Coord):
        comptime assert idx.rank == 3, "Invalid rank passed to rope kernel"

        comptime if width == 1:
            return
        else:
            var global_token_idx = Int(idx[0].value())

            var batch_idx: Int = get_batch_from_row_offsets(
                input_row_offsets, global_token_idx
            )
            var token_idx = Int(
                UInt32(global_token_idx) - input_row_offsets[batch_idx]
            )
            var head_idx = Int(idx[1].value())
            var head_dim_idx = Int(idx[2].value())

            # Use position_ids if provided, otherwise fall back to cache calculation
            var post_seq_idx = k_cache.cache_length(batch_idx) + token_idx

            var position_ids_idx = post_seq_idx
            if position_ids:
                comptime PIdTensor = type_of(position_ids.value())
                comptime assert PIdTensor.flat_rank == 2

                comptime if mrope_section:
                    var section_idx = 0

                    comptime for i in range(len(mrope_section.value())):
                        comptime val = Int(mrope_section.value()[i].value())
                        if head_dim_idx < val:
                            section_idx = i
                            break
                    position_ids_idx = Int(
                        position_ids.value()[section_idx, global_token_idx]
                    )
                else:
                    position_ids_idx = Int(
                        position_ids.value()[0, global_token_idx]
                    )

            # WARN assumes head_size % simd_width == 0
            # guarded by constrained statement below
            var is_q_proj = head_idx < num_q_heads
            comptime _alignment = 1 if is_cpu[target]() else align_of[
                SIMD[freq_dtype, width]
            ]()

            var f_c_temp: SIMD[freq_dtype, width]

            comptime if has_nope_prefix:
                if head_dim_idx >= rope_dim:
                    f_c_temp = get_identity_rope_coeff[width, freq_dtype]()
                else:
                    f_c_temp = freqs_cis.load[
                        width=width, alignment=_alignment
                    ]((position_ids_idx, head_dim_idx))
            elif has_nope:
                if head_dim_idx < unroped_dim:
                    f_c_temp = get_identity_rope_coeff[width, freq_dtype]()
                else:
                    f_c_temp = freqs_cis.load[
                        width=width, alignment=_alignment
                    ]((position_ids_idx, head_dim_idx - unroped_dim))
            else:
                f_c_temp = freqs_cis.load[width=width, alignment=_alignment](
                    (position_ids_idx, head_dim_idx)
                )

            if is_q_proj:
                rope_q_proj[
                    interleaved=interleaved,
                    has_nope_prefix=has_nope_prefix,
                    rope_dim=rope_dim,
                    alignment=_alignment,
                ](
                    q_proj,
                    output,
                    coord_to_index_list(idx),
                    f_c_temp,
                    q_head_size,
                )
            else:
                comptime if has_nope_prefix:
                    if head_dim_idx >= rope_dim:
                        return
                elif has_nope:
                    if head_dim_idx < unroped_dim:
                        return

                head_idx -= num_q_heads
                # in case k_head_size != q_head_size
                head_dim_idx += k_head_size - q_head_size
                rope_k_cache[
                    interleaved=interleaved,
                    has_nope_prefix=has_nope_prefix,
                    rope_prefix_dim=rope_dim,
                ](
                    k_cache,
                    batch_idx,
                    head_idx,
                    post_seq_idx,
                    head_dim_idx,
                    f_c_temp,
                    k_head_size,
                )

    var launch_shape = (
        Int(q_proj.dim[0]()),
        num_q_heads + num_k_heads,  # concat q and k along head dim
        q_head_size,
    )
    comptime compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    comptime target_simd_width = simd_width_of[dtype, target=compile_target]()
    comptime kernel_simd_width = gcd(target_simd_width, rope_dim)

    comptime if mrope_section:
        comptime for i in range(len(mrope_section.value())):
            comptime assert (
                Int(mrope_section.value()[i].value()) % kernel_simd_width == 0
            ), "mrope_section must be divisible by rope kernel simd_width"

    comptime assert kernel_simd_width >= 2, "invalid simd_width and head size"

    elementwise[
        func=rope_fn,
        simd_width=kernel_simd_width,
        target=target,
        _trace_description="fused_qk_rope_ragged",
    ](launch_shape, context)
