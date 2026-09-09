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

"""Fused rope + split + KV store kernel.

Reads a flat QKV matmul output, applies RoPE to Q and K regions, stores
K/V to the paged KV cache, and writes roped Q to the output buffer, all
in a single GPU kernel to eliminate intermediate tensor round-trips.
"""

from max.algorithm.functional import elementwise
from std.collections import OptionalReg
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.info import is_cpu, is_gpu
from std.math import gcd
from std.sys import align_of
from std.sys.info import _current_target, simd_width_of

from internal_utils.fp8_utils import cast_saturating
from kv_cache.types import KVCacheT, PagedKVCacheCollection
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
from nn.fused_qk_rope import rope_value
from nn.rope import get_safetensors_idx
from std.utils.index import IndexList


# ===-----------------------------------------------------------------------===#
# Core kernel
# ===-----------------------------------------------------------------------===#


# HACK: `k_cache` and `v_cache` are the key/value halves (kv_idx 0 vs 1) of the
# same `blocks` buffer, so they share the collection's mutable `blocks_origin`
# (and `scales_origin`). They are only ever stored to at disjoint offsets, but
# the exclusivity checker cannot prove that and rejects capturing both as
# separately-writable values in the store closure. Disabling the nested-origin
# exclusivity check is a stopgap workaround; the proper fix is to give the k/v
# views provably-disjoint origins instead of sharing the collection's.
@__unsafe_nested_origins_read_only
@always_inline
def _rope_split_store_ragged_impl[
    dtype: DType,
    freq_dtype: DType,
    cache_t: KVCacheT,
    q_out_dtype: DType,
    FuncType: ImplicitlyCopyable & RegisterPassable & def(Int, Int, Int) -> Int,
    //,
    *,
    target: StaticString,
    interleaved: Bool = True,
](
    qkv: TileTensor[mut=False, dtype, ...],
    input_row_offsets: TileTensor[mut=False, .uint32, ...],
    freqs_cis: TileTensor[mut=False, freq_dtype, ...],
    k_cache: cache_t,
    v_cache: OptionalReg[cache_t],
    q_output: TileTensor[mut=True, q_out_dtype, ...],
    get_freq_pos: FuncType,
    context: Optional[DeviceContext],
) raises:
    """Read flat QKV buffer, apply RoPE to Q and K, store K/V to cache.

    The ``get_freq_pos`` closure resolves a head-dimension index and
    token position to the row of ``freqs_cis`` that supplies the RoPE
    coefficients. Callers swap in different closures to get
    cache-derived positions vs. explicit position-ID lookups.

    Parameters:
        FuncType: Type of the ``get_freq_pos`` callback.
        target: Compile target backend, for example ``"gpu"`` or ``"cpu"``.
        interleaved: Whether RoPE uses the interleaved real/imag layout.

    Args:
        qkv: Flat matmul output [total_seq_len, q_dim + k_dim + v_dim].
        input_row_offsets: [batch_size + 1] ragged offsets.
        freqs_cis: [max_seq_len, head_dim] interleaved rope frequencies.
        k_cache: Key cache to store roped K.
        v_cache: Value cache to store V.
        q_output: Output buffer for roped Q [total_seq_len, q_dim].
        get_freq_pos: Maps ``(dim_idx, global_token_idx, cache_pos)`` to
            the ``freqs_cis`` row used for RoPE coefficients.
        context: DeviceContext for GPU.
    """
    comptime kv_params = cache_t.kv_params
    comptime kv_type = cache_t.dtype
    comptime head_size = kv_params.head_size
    comptime num_kv_heads = kv_params.num_heads

    comptime assert qkv.flat_rank == 2, "qkv must be rank 2"
    comptime assert q_output.flat_rank == 2, "q_output must be rank 2"
    comptime assert freqs_cis.flat_rank == 2, "freqs_cis must be rank 2"

    comptime assert q_out_dtype == dtype or (
        kv_type.is_float8() and q_out_dtype.is_float8()
    ), (
        "Q output dtype must equal Q/K/V input dtype, or both Q output dtype"
        " and KV cache dtype must be float8"
    )

    var q_dim = Int(q_output.dim[1]())
    var k_dim = head_size * num_kv_heads
    var total_seq_len = Int(qkv.dim[0]())
    var batch_size = Int(input_row_offsets.dim[0]()) - 1

    if batch_size == 0:
        return

    var freqs_ptr = freqs_cis.ptr
    var freqs_stride0 = head_size
    var qkv_ptr = qkv.ptr
    var q_out_ptr = q_output.ptr

    var combined_dim = Int(qkv.dim[1]())
    var qk_offset = q_dim + k_dim

    def rope_split_store_fn[
        width: Int, alignment: Int = 1
    ](idx_arg: Coord) {
        var q_dim,
        var qk_offset,
        var combined_dim,
        var batch_size,
        var freqs_ptr,
        var freqs_stride0,
        var qkv_ptr,
        var q_out_ptr,
        var k_cache,
        var v_cache,
        var input_row_offsets,
        var get_freq_pos,
    }:
        comptime simd_width = width
        comptime assert idx_arg.rank == 2
        var idx = rebind[IndexList[2]](coord_to_index_list(idx_arg))
        var global_token_idx = idx[0]
        var col = idx[1]

        # Guard: simd_width >= 2 is guaranteed by the comptime assert after
        # this function. We wrap in comptime if because elementwise
        # instantiates at all simd widths and width_2 = simd_width // 2 = 0
        # causes rebind errors.
        comptime if simd_width >= 2:
            comptime align_qkv = align_of[SIMD[dtype, simd_width]]() if is_gpu[
                target
            ]() else align_of[SIMD[dtype, 1]]()
            comptime align_freq = align_of[
                SIMD[freq_dtype, simd_width]
            ]() if is_gpu[target]() else align_of[SIMD[freq_dtype, 1]]()
            comptime align_q_out = align_of[
                SIMD[q_out_dtype, simd_width]
            ]() if is_gpu[target]() else align_of[SIMD[q_out_dtype, 1]]()

            # Hoist batch index lookup (binary search) before Q/K/V branch
            # so each thread does it once instead of per-region.
            var bi: Int = get_batch_from_row_offsets(
                input_row_offsets, global_token_idx
            )
            var ti = Int(
                UInt32(global_token_idx) - input_row_offsets.raw_load(bi)
            )

            # Cache position: used for cache stores and as the default
            # freq_pos when the caller doesn't supply explicit position IDs.
            var cache_pos = k_cache.cache_length(bi) + ti

            if col < q_dim:
                # Q region: apply rope, write to q_output.
                var hdi = col % head_size
                var freq_pos = get_freq_pos(hdi, global_token_idx, cache_pos)

                var qkv_base = global_token_idx * combined_dim + col
                var q_base = global_token_idx * q_dim + col

                comptime if interleaved:
                    var val = (qkv_ptr + qkv_base).load[
                        width=simd_width,
                        alignment=align_qkv,
                    ]()
                    var freq = (
                        freqs_ptr + freq_pos * freqs_stride0 + hdi
                    ).load[
                        width=simd_width,
                        alignment=align_freq,
                    ]()
                    (q_out_ptr + q_base).store[alignment=align_q_out](
                        cast_saturating[q_out_dtype](rope_value(val, freq))
                    )
                else:
                    # Non-interleaved: gather re/im halves, rope, scatter.
                    comptime width_2 = SIMDLength(simd_width) / 2
                    comptime align_qkv_2 = align_of[
                        SIMD[dtype, width_2]
                    ]() if is_gpu[target]() else align_of[SIMD[dtype, 1]]()
                    comptime align_q_out_2 = align_of[
                        SIMD[q_out_dtype, width_2]
                    ]() if is_gpu[target]() else align_of[
                        SIMD[q_out_dtype, 1]
                    ]()
                    var head_start_qkv = qkv_base - hdi
                    var head_start_q = q_base - hdi
                    var re_idx, im_idx = get_safetensors_idx(hdi, head_size)
                    var val_re = (qkv_ptr + head_start_qkv + re_idx).load[
                        width=width_2,
                        alignment=align_qkv_2,
                    ]()
                    var val_im = (qkv_ptr + head_start_qkv + im_idx).load[
                        width=width_2,
                        alignment=align_qkv_2,
                    ]()
                    var val = rebind[SIMD[dtype, simd_width]](
                        val_re.interleave(val_im)
                    )
                    var freq = (
                        freqs_ptr + freq_pos * freqs_stride0 + hdi
                    ).load[
                        width=simd_width,
                        alignment=align_freq,
                    ]()
                    var res = rope_value(val, freq)
                    var res_re: SIMD[dtype, width_2]
                    var res_im: SIMD[dtype, width_2]
                    res_re, res_im = res.deinterleave()
                    (q_out_ptr + head_start_q + re_idx).store[
                        alignment=align_q_out_2
                    ](cast_saturating[q_out_dtype](res_re))
                    (q_out_ptr + head_start_q + im_idx).store[
                        alignment=align_q_out_2
                    ](cast_saturating[q_out_dtype](res_im))
                return

            if col < qk_offset:
                # K region: apply rope, store to k_cache.
                var kv_col = col - q_dim
                var hi, di = divmod(
                    UInt(kv_col),
                    UInt(kv_params.head_size),
                )
                var freq_pos = get_freq_pos(
                    Int(di), global_token_idx, cache_pos
                )

                comptime if interleaved:
                    var qkv_base = global_token_idx * combined_dim + col
                    var val = (qkv_ptr + qkv_base).load[
                        width=simd_width,
                        alignment=align_qkv,
                    ]()
                    var freq = (
                        freqs_ptr + freq_pos * freqs_stride0 + Int(di)
                    ).load[
                        width=simd_width,
                        alignment=align_freq,
                    ]()
                    k_cache.store(
                        bi,
                        Int(hi),
                        cache_pos,
                        Int(di),
                        cast_saturating[kv_type](rope_value(val, freq)),
                    )
                else:
                    # Non-interleaved K: gather re/im, rope, deinterleave,
                    # store.
                    comptime width_2 = SIMDLength(simd_width) / 2
                    comptime align_qkv_2 = align_of[
                        SIMD[dtype, width_2]
                    ]() if is_gpu[target]() else align_of[SIMD[dtype, 1]]()
                    var k_head_base = (
                        global_token_idx * combined_dim
                        + q_dim
                        + Int(hi) * head_size
                    )
                    var re_idx, im_idx = get_safetensors_idx(Int(di), head_size)
                    var val_re = (qkv_ptr + k_head_base + re_idx).load[
                        width=width_2,
                        alignment=align_qkv_2,
                    ]()
                    var val_im = (qkv_ptr + k_head_base + im_idx).load[
                        width=width_2,
                        alignment=align_qkv_2,
                    ]()
                    var val = rebind[SIMD[dtype, simd_width]](
                        val_re.interleave(val_im)
                    )
                    var freq = (
                        freqs_ptr + freq_pos * freqs_stride0 + Int(di)
                    ).load[
                        width=simd_width,
                        alignment=align_freq,
                    ]()
                    var roped = rope_value(val, freq)
                    var roped_re: SIMD[dtype, width_2]
                    var roped_im: SIMD[dtype, width_2]
                    roped_re, roped_im = roped.deinterleave()
                    k_cache.store(
                        bi,
                        Int(hi),
                        cache_pos,
                        re_idx,
                        cast_saturating[kv_type](roped_re),
                    )
                    k_cache.store(
                        bi,
                        Int(hi),
                        cache_pos,
                        im_idx,
                        cast_saturating[kv_type](roped_im),
                    )
                return

            # V region: store directly to v_cache (no rope).
            var qkv_base = global_token_idx * combined_dim + col
            var val = (qkv_ptr + qkv_base).load[
                width=simd_width,
                alignment=align_qkv,
            ]()
            var v_col = col - qk_offset
            var hi, di = divmod(
                UInt(v_col),
                UInt(kv_params.head_size),
            )
            var cl = v_cache.value().cache_length(bi)
            v_cache.value().store(
                bi,
                Int(hi),
                ti + cl,
                Int(di),
                cast_saturating[kv_type](val),
            )

    var launch_shape = (total_seq_len, combined_dim)
    comptime compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    comptime target_simd_width = simd_width_of[dtype, target=compile_target]()
    comptime kernel_simd_width = gcd(target_simd_width, head_size)
    comptime assert (
        kernel_simd_width >= 2
    ), "rope_split_store requires simd_width >= 2"
    comptime assert (
        head_size % kernel_simd_width == 0
    ), "head_size must be divisible by simd_width"

    var device_ctx = context.value() if context else DeviceContext(api="cpu")
    elementwise[simd_width=kernel_simd_width, target=target](
        rope_split_store_fn, launch_shape, device_ctx
    )


# ===-----------------------------------------------------------------------===#
# Without position IDs
# ===-----------------------------------------------------------------------===#


# HACK: forwards the `k_cache`/`v_cache` pair (disjoint k/v halves of one
# `blocks` buffer that share the collection's mutable origins) on to the store
# impl, so it inherits the same false-positive aliasing rejection. See
# `_rope_split_store_ragged_impl` for the full rationale. Stopgap workaround.
@__unsafe_nested_origins_read_only
@always_inline
def _rope_split_store_ragged[
    dtype: DType,
    freq_dtype: DType,
    cache_t: KVCacheT,
    q_out_dtype: DType = dtype,
    //,
    *,
    target: StaticString,
    interleaved: Bool = True,
](
    qkv: TileTensor[mut=False, dtype, ...],
    input_row_offsets: TileTensor[mut=False, .uint32, ...],
    freqs_cis: TileTensor[mut=False, freq_dtype, ...],
    k_cache: cache_t,
    v_cache: OptionalReg[cache_t],
    q_output: TileTensor[mut=True, q_out_dtype, ...],
    context: Optional[DeviceContext],
) raises:
    """Read flat QKV buffer, apply rope to Q and K, store K/V to cache.

    Args:
        qkv: Flat matmul output [total_seq_len, q_dim + k_dim + v_dim].
        input_row_offsets: [batch_size + 1] ragged offsets.
        freqs_cis: [max_seq_len, head_dim] interleaved rope frequencies.
        k_cache: Key cache to store roped K.
        v_cache: Value cache to store V.
        q_output: Output buffer for roped Q [total_seq_len, q_dim].
        context: DeviceContext for GPU.
    """

    def get_freq_pos(
        dim_idx: Int, global_token_idx: Int, cache_pos: Int
    ) -> Int:
        return cache_pos

    return _rope_split_store_ragged_impl[
        target=target,
        interleaved=interleaved,
    ](
        qkv,
        input_row_offsets,
        freqs_cis,
        k_cache,
        v_cache,
        q_output,
        get_freq_pos,
        context,
    )


@always_inline
def rope_split_store_paged_ragged[
    dtype: DType,
    freq_dtype: DType,
    q_out_dtype: DType = dtype,
    target: StaticString = "cpu",
    interleaved: Bool = True,
](
    qkv: TileTensor[mut=False, dtype, ...],
    input_row_offsets: TileTensor[mut=False, .uint32, ...],
    freqs_cis: TileTensor[mut=False, freq_dtype, ...],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    q_output: TileTensor[mut=True, q_out_dtype, ...],
    ctx: DeviceContext,
) raises:
    """Rope+split+store with paged KV cache collection.

    Parameters:
        dtype: Element type of the QKV input buffer.
        freq_dtype: Element type of the ``freqs_cis`` RoPE frequency
            table.
        q_out_dtype: Element type of the roped Q output buffer (defaults
            to ``dtype``).
        target: Compile target backend, for example ``"gpu"`` or
            ``"cpu"`` (defaults to ``"cpu"``).
        interleaved: Whether RoPE uses the interleaved real/imag layout
            (defaults to ``True``).

    Args:
        qkv: Flat matmul output
            [total_seq_len, q_dim + k_dim + v_dim].
        input_row_offsets: [batch_size + 1] ragged offsets.
        freqs_cis: [max_seq_len, head_dim] interleaved rope frequencies.
        kv_collection: Paged KV cache collection to store K and V into.
        layer_idx: Index of the layer whose K/V caches in
            ``kv_collection`` to store into.
        q_output: Output buffer for roped Q [total_seq_len, q_dim].
        ctx: DeviceContext for GPU.
    """
    var cuda_ctx: Optional[DeviceContext] = None
    var layer_idx_cast = Int(layer_idx)
    var k_cache = kv_collection.get_key_cache(layer_idx_cast)
    var v_cache: OptionalReg[type_of(k_cache)] = kv_collection.get_value_cache(
        layer_idx_cast
    )

    comptime if is_gpu[target]():
        cuda_ctx = ctx

    return _rope_split_store_ragged[
        target=target,
        interleaved=interleaved,
    ](qkv, input_row_offsets, freqs_cis, k_cache, v_cache, q_output, cuda_ctx)


# ===-----------------------------------------------------------------------===#
# With position IDs
# ===-----------------------------------------------------------------------===#


# HACK: forwards the `k_cache`/`v_cache` pair (disjoint k/v halves of one
# `blocks` buffer that share the collection's mutable origins) on to the store
# impl, so it inherits the same false-positive aliasing rejection. See
# `_rope_split_store_ragged_impl` for the full rationale. Stopgap workaround.
@__unsafe_nested_origins_read_only
@always_inline
def _rope_split_store_ragged_with_position_ids[
    dtype: DType,
    freq_dtype: DType,
    cache_t: KVCacheT,
    q_out_dtype: DType = dtype,
    //,
    *,
    target: StaticString,
    interleaved: Bool = True,
    mrope_types: TypeList[Trait=CoordLike, ...] = TypeList.of[
        Trait=CoordLike
    ](),
    mrope_section: Optional[Coord[*mrope_types]] = None,
    PositionIdsLayoutType: TensorLayout = RowMajorLayout[
        *Coord[Int64, Int64].element_types
    ],
](
    qkv: TileTensor[mut=False, dtype, ...],
    input_row_offsets: TileTensor[mut=False, .uint32, ...],
    freqs_cis: TileTensor[mut=False, freq_dtype, ...],
    k_cache: cache_t,
    v_cache: OptionalReg[cache_t],
    position_ids: TileTensor[mut=False, .uint32, PositionIdsLayoutType, ...],
    q_output: TileTensor[mut=True, q_out_dtype, ...],
    context: Optional[DeviceContext],
) raises:
    """Read flat QKV buffer, apply rope (with explicit position IDs) to Q and K,
    store K/V to cache.

    Like ``_rope_split_store_ragged`` but looks up RoPE frequencies using
    ``position_ids`` instead of ``cache_length + token_offset``. When
    ``mrope_section`` is provided, different head-dimension sections use
    different rows of ``position_ids`` (multi-axis RoPE for VL models).

    Args:
        qkv: Flat matmul output [total_seq_len, q_dim + k_dim + v_dim].
        input_row_offsets: [batch_size + 1] ragged offsets.
        freqs_cis: [max_seq_len, head_dim] interleaved rope frequencies.
        k_cache: Key cache to store roped K.
        v_cache: Value cache to store V.
        position_ids: [num_sections, total_seq_len] explicit position IDs.
        q_output: Output buffer for roped Q [total_seq_len, q_dim].
        context: DeviceContext for GPU.
    """
    comptime assert PositionIdsLayoutType.rank == 2

    # Validate mrope_section alignment with kernel SIMD width.
    comptime kv_params = cache_t.kv_params
    comptime head_size = kv_params.head_size
    comptime compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    comptime target_simd_width = simd_width_of[dtype, target=compile_target]()
    comptime kernel_simd_width = gcd(target_simd_width, head_size)
    comptime if mrope_section:
        comptime for i in range(len(mrope_section.value())):
            comptime assert (
                Int(mrope_section.value()[i].value()) % kernel_simd_width == 0
            ), "mrope_section must be divisible by rope kernel simd_width"

    var pos_ids_ptr = position_ids.ptr
    var pos_ids_stride = Int(position_ids.dim[1]())

    def get_freq_pos(
        dim_idx: Int, global_token_idx: Int, cache_pos: Int
    ) {var pos_ids_ptr, var pos_ids_stride} -> Int:
        comptime if mrope_section:
            var section_idx = 0
            comptime for i in range(len(mrope_section.value())):
                comptime val = Int(mrope_section.value()[i].value())
                if dim_idx < val:
                    section_idx = i
                    break
            return Int(
                pos_ids_ptr[section_idx * pos_ids_stride + global_token_idx]
            )
        else:
            return Int(pos_ids_ptr[global_token_idx])

    return _rope_split_store_ragged_impl[
        target=target,
        interleaved=interleaved,
    ](
        qkv,
        input_row_offsets,
        freqs_cis,
        k_cache,
        v_cache,
        q_output,
        get_freq_pos,
        context,
    )


@always_inline
def rope_split_store_paged_ragged_with_position_ids[
    dtype: DType,
    freq_dtype: DType,
    q_out_dtype: DType = dtype,
    target: StaticString = "cpu",
    interleaved: Bool = True,
    mrope_types: TypeList[Trait=CoordLike, ...] = TypeList.of[
        Trait=CoordLike
    ](),
    mrope_section: Optional[Coord[*mrope_types]] = None,
    PositionIdsLayoutType: TensorLayout = RowMajorLayout[
        *Coord[Int64, Int64].element_types
    ],
](
    qkv: TileTensor[mut=False, dtype, ...],
    input_row_offsets: TileTensor[mut=False, .uint32, ...],
    freqs_cis: TileTensor[mut=False, freq_dtype, ...],
    kv_collection: PagedKVCacheCollection,
    position_ids: TileTensor[mut=False, .uint32, PositionIdsLayoutType, ...],
    layer_idx: UInt32,
    q_output: TileTensor[mut=True, q_out_dtype, ...],
    ctx: DeviceContext,
) raises:
    """Rope+split+store with paged KV cache and explicit position IDs.

    Parameters:
        dtype: Element type of the QKV input buffer.
        freq_dtype: Element type of the ``freqs_cis`` RoPE frequency
            table.
        q_out_dtype: Element type of the roped Q output. Defaults to
            ``dtype``; an FP8 KV cache sets it to the cache dtype so
            flash attention sees a Q matching its cache.
        target: Compile target backend, for example ``"gpu"`` or
            ``"cpu"`` (defaults to ``"cpu"``).
        interleaved: Whether RoPE uses the interleaved real/imag layout
            (defaults to ``True``).
        mrope_types: Element types of the ``mrope_section`` coord
            (defaults to an empty `TypeList`).
        mrope_section: Optional head-dimension section boundaries for
            multi-axis RoPE; when set, each section indexes a different
            row of ``position_ids`` (defaults to `None`).
        PositionIdsLayoutType: Tensor layout of the ``position_ids``
            buffer (defaults to `RowMajorLayout`).

    Args:
        qkv: Flat matmul output
            [total_seq_len, q_dim + k_dim + v_dim].
        input_row_offsets: [batch_size + 1] ragged offsets.
        freqs_cis: [max_seq_len, head_dim] interleaved rope frequencies.
        kv_collection: Paged KV cache collection to store K and V into.
        position_ids: [num_sections, total_seq_len] explicit position
            IDs used to look up RoPE frequencies.
        layer_idx: Index of the layer whose K/V caches in
            ``kv_collection`` to store into.
        q_output: Output buffer for roped Q [total_seq_len, q_dim].
        ctx: DeviceContext for GPU.
    """
    var cuda_ctx: Optional[DeviceContext] = None
    var layer_idx_cast = Int(layer_idx)
    var k_cache = kv_collection.get_key_cache(layer_idx_cast)
    var v_cache: OptionalReg[type_of(k_cache)] = kv_collection.get_value_cache(
        layer_idx_cast
    )

    comptime if is_gpu[target]():
        cuda_ctx = ctx

    return _rope_split_store_ragged_with_position_ids[
        target=target,
        interleaved=interleaved,
        mrope_types=mrope_types,
        mrope_section=mrope_section,
        PositionIdsLayoutType=PositionIdsLayoutType,
    ](
        qkv,
        input_row_offsets,
        freqs_cis,
        k_cache,
        v_cache,
        position_ids,
        q_output,
        cuda_ctx,
    )
