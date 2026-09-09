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

"""
Implements a naive GPU multihead cross attention kernel supporting ragged
batched inputs and a paged KV cache.
"""

from std.math import ceildiv
from std.math.uutils import ufloordiv, udivmod
from std.sys import align_of, simd_width_of

from std.algorithm.functional import vectorize
from max.gpu import block_idx, global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from kv_cache.types import KVCacheT
from layout import Coord, Idx, TensorLayout, TileTensor, row_major
from layout.tile_layout import Layout
from nn.attention.gpu.mha import MHAConfig, _kernel_mask
from nn.attention.mha_mask import MHAMask
from nn.softmax import _softmax_gpu

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type


@always_inline
@__name(t"mha_cross_bmm0_{q_type}_{p_type}")
def _bmm0_bs[
    QLayoutType: TensorLayout,
    KVLayoutType: TensorLayout,
    //,
    cache_t: KVCacheT,
    mask_t: MHAMask,
    q_type: DType,
    p_type: DType,
](
    p_ptr: UnsafePointer[Scalar[p_type], MutAnyOrigin],
    q_ptr: UnsafePointer[Scalar[q_type], ImmutAnyOrigin],
    k_cache: cache_t,
    q_input_row_offsets: TileTensor[.uint32, QLayoutType, ImmutAnyOrigin],
    kv_input_row_offsets: TileTensor[.uint32, KVLayoutType, ImmutAnyOrigin],
    scale: Float32,
    batch_size: Int32,
    q_max_seq_len: Int32,
    # The maximum current sequence length in the KV cache.
    kv_max_seq_len: Int32,
    max_cache_size: Int32,
    num_heads: Int32,
    depth: Int32,
    group: Int32,
    mask_functor: mask_t,
):
    var _batch_size = Int(batch_size)
    var _q_max_seq_len = Int(q_max_seq_len)
    var _kv_max_seq_len = Int(kv_max_seq_len)
    var _max_cache_size = Int(max_cache_size)
    var _num_heads = Int(num_heads)
    var _depth = Int(depth)
    var _group = Int(group)
    comptime assert q_input_row_offsets.flat_rank == 1
    comptime assert kv_input_row_offsets.flat_rank == 1

    # total_context_length
    var x = global_idx.x
    # prompt_length
    var y = global_idx.y

    comptime k_type = cache_t.dtype
    comptime kv_num_heads = cache_t.kv_params.num_heads

    var batch_head = block_idx.z
    var batch, head = udivmod(batch_head, _num_heads)

    var cur_query_len: Int
    var cur_kv_len: Int
    var q_offset: Int
    var num_keys: Int
    var padded_num_keys = _kv_max_seq_len + _max_cache_size
    var p_offset = batch_head * _q_max_seq_len * padded_num_keys

    var q_seq_start = Int(q_input_row_offsets[batch])
    var q_seq_end = Int(q_input_row_offsets[batch + 1])
    cur_query_len = q_seq_end - q_seq_start
    q_offset = (q_seq_start * _num_heads + head) * _depth

    var kv_seq_start = Int(kv_input_row_offsets[batch])
    var kv_seq_end = Int(kv_input_row_offsets[batch + 1])
    cur_kv_len = kv_seq_end - kv_seq_start
    # _num_heads * _kv_max_seq_len * batch * _depth + _depth * head
    num_keys = cur_kv_len + k_cache.cache_length(batch)

    assert cur_kv_len <= _kv_max_seq_len, "Invalid cur_kv_len"
    assert num_keys <= padded_num_keys, "Invalid _max_cache_size"

    if x >= (_kv_max_seq_len + _max_cache_size) or y >= _q_max_seq_len:
        return

    var q = q_ptr + q_offset

    var kv_head = ufloordiv(head, _group)

    var p = p_ptr + p_offset

    var accum = Scalar[p_type](0.0)

    # Set total KV length: KV written previous to and during this forward.
    if x < num_keys and y < cur_query_len:
        var accum_vec = SIMD[p_type, simd_width_of[p_type]()](0)
        var k_ptr = k_cache.block_paged_ptr[tile_size=1](batch, x, kv_head, 0)

        def accum_fn[
            width: Int
        ](offset: Int) {q, y, _num_heads, _depth, k_ptr, mut}:
            comptime alignment = align_of[SIMD[p_type, width]]()
            var q_val = q.load[width=width, alignment=alignment](
                y * _num_heads * _depth + offset
            ).cast[k_type]()
            var k_val = k_ptr.load[width=width, alignment=alignment](offset)
            var qk_val = (q_val * k_val).cast[p_type]()

            comptime if width == 1:
                accum += rebind[type_of(accum)](qk_val)
            else:
                accum_vec += rebind[type_of(accum_vec)](qk_val)

        vectorize[simd_width_of[p_type]()](_depth, accum_fn)
        accum += accum_vec.reduce_add()

    var score_row = y
    var score_col = x
    p[y * padded_num_keys + x] = mask_functor.mask(
        Index(batch, head, Int(score_row), score_col),
        accum * scale.cast[p_type](),
    )
    p[y * padded_num_keys + x] = _kernel_mask(
        Index(score_row, score_col),
        Index(cur_query_len, num_keys),
        p[y * padded_num_keys + x],
    )


@always_inline
@__name(t"mha_cross_bmm1_{output_type}_{p_type}")
def _bmm1_bs[
    QLayoutType: TensorLayout,
    KVLayoutType: TensorLayout,
    //,
    cache_t: KVCacheT,
    p_type: DType,
    output_type: DType,
](
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    p_ptr: UnsafePointer[Scalar[p_type], ImmutAnyOrigin],
    v_cache: cache_t,
    q_input_row_offsets: TileTensor[.uint32, QLayoutType, ImmutAnyOrigin],
    kv_input_row_offsets: TileTensor[.uint32, KVLayoutType, ImmutAnyOrigin],
    q_max_seq_len: Int32,
    kv_max_seq_len: Int32,
    max_cache_size: Int32,
    num_heads: Int32,
    depth: Int32,
    group: Int32,
):
    var _q_max_seq_len = Int(q_max_seq_len)
    var _kv_max_seq_len = Int(kv_max_seq_len)
    var _max_cache_size = Int(max_cache_size)
    var _num_heads = Int(num_heads)
    var _depth = Int(depth)
    var _group = Int(group)
    comptime assert q_input_row_offsets.flat_rank == 1
    comptime assert kv_input_row_offsets.flat_rank == 1

    comptime v_type = cache_t.dtype
    comptime kv_num_heads = cache_t.kv_params.num_heads

    # head_size
    var x = global_idx.x
    # query seq_len
    var y = global_idx.y

    var batch_head = block_idx.z
    var batch, head = udivmod(batch_head, _num_heads)

    var cur_query_len: Int
    var cur_kv_len: Int
    var output_offset: Int
    var padded_num_keys = _kv_max_seq_len + _max_cache_size
    var p_offset = batch_head * _q_max_seq_len * padded_num_keys

    var q_seq_start = Int(q_input_row_offsets[batch])
    var q_seq_end = Int(q_input_row_offsets[batch + 1])
    cur_query_len = q_seq_end - q_seq_start

    output_offset = (q_seq_start * _num_heads + head) * _depth

    var kv_seq_start = Int(kv_input_row_offsets[batch])
    var kv_seq_end = Int(kv_input_row_offsets[batch + 1])
    cur_kv_len = kv_seq_end - kv_seq_start

    assert cur_query_len <= _q_max_seq_len, "Invalid cur_query_len"
    assert cur_kv_len <= _kv_max_seq_len, "Invalid cur_kv_len"

    if x >= _depth or y >= cur_query_len:
        return

    var p = p_ptr + p_offset

    var kv_head = ufloordiv(head, _group)
    var output = output_ptr + output_offset

    var accum = Float32(0.0)

    for i in range(cur_kv_len + v_cache.cache_length(batch)):
        var v_ptr = v_cache.block_paged_ptr[tile_size=1](batch, i, kv_head, x)
        accum += (p[y * padded_num_keys + i].cast[v_type]() * v_ptr[0]).cast[
            DType.float32
        ]()

    output[y * _num_heads * _depth + x] = accum.cast[output_type]()


# ===-----------------------------------------------------------------------===#
# Naive GPU multihead cross attention supporting flexible dimensions and
# batch_size > 1.
# ===-----------------------------------------------------------------------===#


def mha_cross_gpu_naive[
    cache_t: KVCacheT,
    mask_t: MHAMask,
    dtype: DType,
    //,
    rank: Int,
](
    output: TileTensor[address_space=.GENERIC, ...],
    q: TileTensor[mut=False, dtype, address_space=.GENERIC, ...],
    q_input_row_offsets: TileTensor[mut=False, .uint32, ...],
    q_max_seq_len: Int,
    k: cache_t,
    v: cache_t,
    kv_input_row_offsets: TileTensor[mut=False, .uint32, ...],
    mask_functor: mask_t,
    scale: Float32,
    ctx: DeviceContext,
) raises:
    """Naive cross attention on GPU.

    Note that this assumes ragged tensor inputs and uses a mask functor.

    Computes:
        (1) Transpose (Q) BSHD -> BHSD;
        (2) Transpose (K) BSHD -> BHSD;
        (3) Transpose (V) BSHD -> BHSD;
        (4) P = Bmm(Q, K), P is also called "score";
        (5) P = P * scale + mask;
        (6) P = softmax(P);
        (7) O = Bmm(P, V)
        (8) Output = Transpose(O).

    B, S, H, D denote batch size, sequence length, head count and depth, respectively.
    (1), (2), (3) happens while loading the data into shared memory.
    (8) happens when writing output to global memory.

    All inputs (query, key, and value) must have BSHD layout. The mask can be
    BSS or BHSS.

    This kernel also handles grouped attention optimization. In this case the shape of
    K and V are BShD where h = H / num_groups.

    Parameters:
        cache_t: The paged KV cache type used for `k` and `v` (inferred).
        mask_t: The mask functor type applied to attention scores (inferred).
        dtype: The element type of the query, key, value, and output tensors
            (inferred).
        rank: The number of dimensions of the input tensors. Must be 3 for
            ragged inputs.

    Args:
        output: The output tensor receiving the cross attention result. Same
            dtype as `q` and the KV cache, in BSHD layout.
        q: The query tensor in BSHD layout. The static shape's last two
            dimensions give the head count and depth.
        q_input_row_offsets: Per-batch start and end offsets into the ragged
            query tensor. Length is `batch_size + 1`.
        q_max_seq_len: The maximum query sequence length across the batch.
        k: The paged KV cache holding keys.
        v: The paged KV cache holding values.
        kv_input_row_offsets: Per-batch start and end offsets into the ragged
            KV input. Length is `batch_size + 1`.
        mask_functor: The mask instance applied to attention scores before
            softmax.
        scale: The scaling factor multiplied with the query-key scores.
        ctx: The device context used to enqueue GPU kernels and buffers.
    """
    comptime assert rank == 3, "only support rank 3 inputs for ragged inputs."
    comptime assert (
        q.dtype == cache_t.dtype == cache_t.dtype == output.dtype
    ), "Q, K, V, output should have same type."
    comptime assert (
        q.dtype == .float32 or q.dtype.is_half_float()
    ), "Only support single and half precision."

    comptime config = MHAConfig[dtype](
        Int(q.static_shape[rank - 2]),
        Int(q.static_shape[rank - 1]),
    )

    comptime num_heads = config.num_heads
    comptime depth = config.depth
    comptime kv_num_heads = cache_t.kv_params.num_heads
    comptime group = config.num_heads // kv_num_heads
    var kv_max_seq_len = Int(k.max_prompt_length())
    var batch_size = Int(q_input_row_offsets.dim[0]()) - 1
    var max_cache_size = Int(k.max_context_length())

    comptime q_type = q.dtype
    comptime k_type = cache_t.dtype
    comptime v_type = cache_t.dtype

    # Assume self attention if the query sequence length isn't passed.
    var num_keys = kv_max_seq_len + max_cache_size
    comptime p_type = get_accum_type[q_type]()
    var p_device = ctx.enqueue_create_buffer[p_type](
        batch_size * num_heads * q_max_seq_len * num_keys
    )

    # FIXME: RUNP-356 Direct access to CUDA within DeviceContext
    var p_buffer = TileTensor(
        # FIXME: GEX-4123 Force use of DefaultEngine until the
        # `input_fn_device` legacy closure is replaced.
        p_device.unsafe_ptr(),
        row_major((batch_size * num_heads, q_max_seq_len, num_keys)),
    )
    var q_device = DeviceBuffer[q_type](
        ctx, q.ptr, q.num_elements(), owning=False
    )

    comptime kernel_0 = _bmm0_bs[
        QLayoutType=q.LayoutType,
        KVLayoutType=kv_input_row_offsets.LayoutType,
        type_of(k),
        mask_t,
        q_type,
        p_type,
    ]
    ctx.enqueue_function[kernel_0](
        p_device,
        q_device,
        k,
        q_input_row_offsets,
        kv_input_row_offsets,
        scale,
        Int32(batch_size),
        Int32(q_max_seq_len),
        Int32(kv_max_seq_len),
        Int32(max_cache_size),
        Int32(num_heads),
        Int32(depth),
        Int32(group),
        mask_functor,
        grid_dim=(
            ceildiv(num_keys, 32),
            ceildiv(q_max_seq_len, 16),
            num_heads * batch_size,
        ),
        block_dim=(32, 16, 1),
    )

    @__parameter
    @__copy_capture(p_buffer)
    def input_fn_device[
        _simd_width: Int
    ](coords: Coord) -> SIMD[p_type, _simd_width]:
        comptime assert p_buffer.flat_rank >= coords.flat_rank
        return p_buffer.load[width=_simd_width](coords)

    _softmax_gpu[p_type, 1, 3, input_fn_device](
        Coord(batch_size * num_heads, q_max_seq_len, num_keys),
        p_buffer,
        2,
        ctx,
    )
    var output_device = DeviceBuffer[output.dtype](
        ctx, output.ptr, output.num_elements(), owning=False
    )

    comptime kernel_1 = _bmm1_bs[
        QLayoutType=q.LayoutType,
        KVLayoutType=kv_input_row_offsets.LayoutType,
        type_of(v),
        p_type,
        output.dtype,
    ]
    ctx.enqueue_function[kernel_1](
        output_device,
        p_device,
        v,
        q_input_row_offsets,
        kv_input_row_offsets,
        Int32(q_max_seq_len),
        Int32(kv_max_seq_len),
        Int32(max_cache_size),
        Int32(num_heads),
        Int32(depth),
        Int32(group),
        grid_dim=(
            ceildiv(depth, 32),
            ceildiv(q_max_seq_len, 16),
            num_heads * batch_size,
        ),
        block_dim=(32, 16, 1),
    )

    _ = p_device
