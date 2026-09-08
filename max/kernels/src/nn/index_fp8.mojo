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
"""Implements tensor indexing and gather kernels with FP8 quantization support on Blackwell GPUs."""
from std.math.uutils import ufloordiv, udivmod
from std.sys import size_of, simd_width_of
from std.sys.info import _has_blackwell_tcgen05
from std.math import ceildiv
from layout import (
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    DefaultEngine,
    RuntimeLayout,
    TensorLayout,
    TensorEngine,
    TileTensor,
    UNKNOWN_VALUE,
)
from layout.tile_layout import row_major
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.sync import barrier
from max.gpu.memory import external_memory
from nn.attention.mha_operand import RaggedMHAOperand, MHAOperand
from nn.attention.gpu.nvidia.common import q_tma
from nn.attention.gpu.sparse_index_fp8_sm100 import (
    _BM_KEY,
    SPEC_DECODE_N_TOKENS_ALT,
    fp8_index_score_sm100,
)
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple


struct IndexSmemStorage[
    dtype: DType,
    num_heads: Int,
    depth: Int,
    BN: Int,
]:
    """Holds shared-memory buffers for the query, key, and scratch tiles used by the FP8 index kernel.

    Parameters:
        dtype: Data type of the query and key tiles.
        num_heads: Number of attention heads stored in the query tile.
        depth: Per-head feature depth of the query and key tiles.
        BN: Number of key rows staged in shared memory per block tile.
    """

    var q_smem: Array[Scalar[Self.dtype], Self.num_heads * Self.depth]
    var k_smem: Array[Scalar[Self.dtype], Self.BN * Self.depth]
    var scratch: Array[Float32, Self.BN * 8]


@__name(t"fp8_index_{dtype}")
def fp8_index_kernel[
    dtype: DType,
    OutputLT: TensorLayout,
    QLT: TensorLayout,
    QSLT: TensorLayout,
    k_operand_type: MHAOperand,
    ks_operand_type: MHAOperand,
    block_tile_shape: Array[Int, 2],
    VLLT: TensorLayout,
    num_heads: Int,
    depth: Int,
    # When False: num_keys = cache_length + seq_len (cache_length excludes new tokens).
    # When True: num_keys = cache_length (cache_length already includes new tokens).
    _is_cache_length_accurate: Bool = False,
    *,
    OutputEngine: TensorEngine = DefaultEngine[element_width=1],
    QEngine: TensorEngine = DefaultEngine[element_width=1],
    QSEngine: TensorEngine = DefaultEngine[element_width=1],
    VLEngine: TensorEngine = DefaultEngine[element_width=1],
](
    output_tt: TileTensor[
        .float32, OutputLT, MutAnyOrigin, Engine=OutputEngine
    ],
    # [total_seq_len, num_heads, depth]
    q_tt: TileTensor[dtype, QLT, ImmutAnyOrigin, Engine=QEngine],
    # [total_seq_len, num_heads]
    q_s_tt: TileTensor[.float32, QSLT, MutAnyOrigin, Engine=QSEngine],
    # MHAOperand for K values
    k_operand: k_operand_type,
    # MHAOperand for K scales
    ks_operand: ks_operand_type,
    valid_length_tt: TileTensor[.uint32, VLLT, ImmutAnyOrigin, Engine=VLEngine],
):
    """Computes the scalar FP8 index/gather score kernel as a Blackwell tensor-core fallback.

    Each block computes a slice of the query sequence against a tile of the
    paged key cache, accumulating per-head logits in shared memory and writing
    the scale-weighted row sum to the output tensor.

    Parameters:
        dtype: Data type of the query and key tiles.
        OutputLT: Layout type of the output score tensor.
        QLT: Layout type of the query tensor.
        QSLT: Layout type of the query scale tensor.
        k_operand_type: MHAOperand type used to address the paged key cache.
        ks_operand_type: MHAOperand type used to address the key scales.
        block_tile_shape: Block tile shape as `[BM, BN]` rows of sequence and keys.
        VLLT: Layout type of the valid-length (sequence offset) tensor.
        num_heads: Number of attention heads.
        depth: Per-head feature depth.
        _is_cache_length_accurate: When True, `cache_length` already includes new tokens.
        OutputEngine: Engine of the `output_tt` tile.
        QEngine: Engine of the `q_tt` tile.
        QSEngine: Engine of the `q_s_tt` tile.
        VLEngine: Engine of the `valid_length_tt` tile.

    Args:
        output_tt: Output score tensor of shape `[total_seq_len, num_keys]`.
        q_tt: Query tensor of shape `[total_seq_len, num_heads, depth]`.
        q_s_tt: Per-query scale tensor of shape `[total_seq_len, num_heads]`.
        k_operand: Ragged paged operand providing key rows.
        ks_operand: Ragged paged operand providing per-key scales.
        valid_length_tt: Cumulative sequence offsets of shape `[batch_size + 1]`.
    """
    # Convert TileTensor inputs to LayoutTensor for internal use,
    # which relies on LayoutTensor-specific APIs (tile, indexing).
    var output = output_tt.to_layout_tensor()
    var q = q_tt.to_layout_tensor()
    var q_s = q_s_tt.to_layout_tensor()
    var valid_length = valid_length_tt.to_layout_tensor()

    comptime valid_length_layout = type_of(valid_length).layout
    comptime assert valid_length_layout.rank() == 1, "valid_length must be 1D"
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]

    comptime thread_dim_x = 16
    comptime thread_dim_y = 8

    comptime simd_width = simd_width_of[dtype]()

    var batch_idx = block_idx.x
    var seq_offset = block_idx.y
    var key_offset = block_idx.z * BM
    var tid = thread_idx.x * 8 + thread_idx.y

    var start_of_seq = valid_length[batch_idx][0]
    var end_of_seq = valid_length[batch_idx + 1][0]
    var seq_len = end_of_seq - start_of_seq

    var num_keys = k_operand.cache_length(batch_idx)

    comptime if not _is_cache_length_accurate:
        num_keys += Int(seq_len)

    if seq_offset >= Int(seq_len) or key_offset >= num_keys:
        return

    ref smem_ptr = external_memory[
        UInt8,
        address_space=.SHARED,
        alignment=128,
    ]().bitcast[IndexSmemStorage[dtype, num_heads, depth, BN]]()[]

    ref q_smem = smem_ptr.q_smem
    ref k_smem = smem_ptr.k_smem
    ref scratch_smem = smem_ptr.scratch

    var q_smem_tile = LayoutTensor[
        dtype,
        Layout.row_major(num_heads, depth),
        address_space=.SHARED,
    ](q_smem.unsafe_ptr())

    var k_smem_tile = LayoutTensor[
        dtype,
        Layout.row_major(BN, depth),
        address_space=.SHARED,
    ](k_smem.unsafe_ptr())

    var k_smem_ptr: UnsafePointer[
        Scalar[dtype], origin_of(k_smem), address_space=.SHARED
    ] = k_smem.unsafe_ptr()

    var q_ptr = q.ptr_at_offset(Index(start_of_seq + UInt32(seq_offset), 0, 0))
    var q_s_ptr = q_s.ptr_at_offset(Index(start_of_seq + UInt32(seq_offset), 0))
    var o_ptr = output.ptr_at_offset(
        Index(start_of_seq + UInt32(seq_offset), UInt32(key_offset))
    )

    comptime QTileType = LayoutTensor[
        dtype, Layout.row_major(num_heads, depth), ImmutAnyOrigin
    ]

    comptime QSTileType = LayoutTensor[
        .float32, Layout.row_major(1, num_heads), MutAnyOrigin
    ]

    comptime LogitsType = LayoutTensor[
        .float32,
        Layout.row_major(BN // thread_dim_x, num_heads // thread_dim_y),
        MutAnyOrigin,
        address_space=.LOCAL,
    ]

    comptime QSRegTileType = LayoutTensor[
        .float32,
        Layout.row_major(1, num_heads // thread_dim_y),
        MutAnyOrigin,
        address_space=.LOCAL,
    ]

    comptime LogitsSumType = LayoutTensor[
        .float32,
        Layout.row_major(BN // thread_dim_x, 1),
        MutAnyOrigin,
        address_space=.LOCAL,
    ]

    comptime ScratchType = LayoutTensor[
        .float32,
        Layout.row_major(BN, thread_dim_y),
        MutAnyOrigin,
        address_space=.SHARED,
    ]

    var q_tile = QTileType(q_ptr)
    var q_s_tile = QSTileType(q_s_ptr)
    var logits = LogitsType.stack_allocation()
    var q_s_reg_tile = QSRegTileType.stack_allocation()
    var logits_sum = LogitsSumType.stack_allocation()
    var scratch = ScratchType(scratch_smem.unsafe_ptr().as_unsafe_any_origin())

    var q_s_frag = q_s_tile.tile[1, num_heads // thread_dim_y](
        ufloordiv(thread_idx.x, thread_dim_x), thread_idx.y
    )

    comptime for q_frag_idx in range(num_heads // thread_dim_y):
        q_s_reg_tile[0, q_frag_idx] = q_s_frag[0, q_frag_idx][0]

    comptime num_threads = thread_dim_x * thread_dim_y
    comptime assert (
        depth % simd_width == 0
    ), "depth must be a multiple of the SIMD width"

    # Flat thread-strided copy of the contiguous [num_heads, depth] Q tile.
    # A layout-distributed copy over the [16, 8] thread shape floor-divides
    # the tile shape per axis and silently stages NOTHING whenever
    # num_heads < 16 or depth // simd_width < 8 (e.g. depth == 64).
    comptime q_vecs = num_heads * depth // simd_width
    var q_smem_dst: UnsafePointer[
        Scalar[dtype], origin_of(q_smem), address_space=.SHARED
    ] = q_smem.unsafe_ptr()
    for v in range(Int(tid), q_vecs, num_threads):
        q_smem_dst.store(
            v * simd_width, q_ptr.load[width=simd_width](v * simd_width)
        )

    for i in range(BM // BN):
        var current_key_offset = key_offset + i * BN
        if current_key_offset >= num_keys:
            break
        for k_row in range(tid, BN, num_threads):
            var row_base = k_row * depth
            if current_key_offset + k_row >= num_keys:
                comptime for d in range(0, depth, simd_width):
                    k_smem_ptr.store(row_base + d, SIMD[dtype, simd_width](0))
            else:
                var k_ptr = k_operand.block_paged_ptr[1](
                    UInt32(batch_idx),
                    UInt32(current_key_offset + k_row),
                    UInt32(0),  # head_idx = 0 for MLA (single head for K)
                    UInt32(0),
                )
                comptime for d in range(0, depth, simd_width):
                    k_smem_ptr.store(
                        row_base + d,
                        k_ptr.load[width=simd_width](d).cast[dtype](),
                    )

        barrier()

        # Load K scales for current tile
        var k_s_reg: Float32 = 0.0
        if current_key_offset + tid < num_keys:
            var ks_ptr = ks_operand.block_paged_ptr[1](
                UInt32(batch_idx),
                UInt32(current_key_offset + tid),
                UInt32(0),
                UInt32(0),
            )
            k_s_reg = ks_ptr[0].cast[.float32]()

        var q_smem_frag = q_smem_tile.tile[num_heads // thread_dim_y, depth](
            thread_idx.y, 0
        )
        var k_smem_frag = k_smem_tile.tile[BN // thread_dim_x, depth](
            thread_idx.x, 0
        )

        _ = logits.fill(0)
        _ = logits_sum.fill(0)

        for k in range(depth):
            comptime for mma_m in range(BN // thread_dim_x):
                comptime for mma_n in range(num_heads // thread_dim_y):
                    logits[mma_m, mma_n] += (
                        k_smem_frag[mma_m, k][0].cast[.float32]()
                        * q_smem_frag[mma_n, k][0].cast[.float32]()
                    )

        comptime for l_i in range(BN // thread_dim_x):
            comptime for l_j in range(num_heads // thread_dim_y):
                logits[l_i, l_j] = (
                    max(logits[l_i, l_j], 0) * q_s_reg_tile[0, l_j][0]
                )

                logits_sum[l_i, 0] += logits[l_i, l_j]

            scratch[
                thread_idx.x * (BN // thread_dim_x) + l_i,
                thread_idx.y,
            ] = logits_sum[l_i, 0]

        barrier()

        if current_key_offset + tid < num_keys:
            # Sum logits across heads
            var row_sum: Float32 = 0.0

            for col_idx in range(thread_dim_y):
                row_sum += scratch[tid, col_idx][0]

            o_ptr[i * BN + tid] = k_s_reg * row_sum


@always_inline
def fp8_index[
    dtype: DType,
    //,
    num_heads: Int,
    depth: Int,
](
    output: TileTensor[.float32, ...],
    q: TileTensor[mut=False, dtype, ...],
    q_s: TileTensor[.float32, ...],
    k: TileTensor[mut=False, dtype, ...],
    k_s: TileTensor[mut=False, .float32, ...],
    valid_length: TileTensor[mut=False, .uint32, ...],
    cache_row_offsets: TileTensor[mut=False, .uint32, ...],
    batch_size: Int,
    max_seq_len: Int,
    max_num_keys: Int,
    ctx: DeviceContext,
) raises:
    """Dispatches the FP8 index/gather scorer on the given device context.

    Selects the Blackwell tcgen05/TMA tensor-core scorer when the device and
    operand layout support it, otherwise falls back to the scalar
    `fp8_index_kernel` path.

    Parameters:
        dtype: Data type of the query and key tensors.
        num_heads: Number of attention heads.
        depth: Per-head feature depth.

    Args:
        output: Output score tensor of shape `[total_seq_len, max_num_keys]`.
        q: Query tensor of shape `[total_seq_len, num_heads, depth]`.
        q_s: Per-query scale tensor of shape `[total_seq_len, num_heads]`.
        k: Key tensor of shape `[total_keys, 1, depth]`.
        k_s: Per-key scale tensor of shape `[total_keys]`.
        valid_length: Cumulative sequence offsets of shape `[batch_size + 1]`.
        cache_row_offsets: Per-batch row offsets into the paged key cache.
        batch_size: Number of sequences in the batch.
        max_seq_len: Maximum sequence length across the batch.
        max_num_keys: Maximum key count across the batch.
        ctx: Device context used to enqueue the selected kernel.

    Raises:
        When the underlying kernel enqueue reports a device-side error.
    """
    var total_keys = Int(k.dim[0]())
    var cro_size = Int(cache_row_offsets.dim[0]())

    var cro_buf = TileTensor(
        rebind[UnsafePointer[UInt32, ImmutAnyOrigin]](cache_row_offsets.ptr),
        row_major(Coord(cro_size)),
    )

    var k_buf = TileTensor(
        rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](k.ptr),
        row_major(Coord(total_keys, Idx[1], Idx[depth])),
    )
    var k_operand = RaggedMHAOperand(k_buf, cro_buf)

    var ks_buf = TileTensor(
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](k_s.ptr),
        row_major(Coord(total_keys, Idx[1], Idx[1])),
    )
    var ks_operand = RaggedMHAOperand(ks_buf, cro_buf)

    comptime assert num_heads % 4 == 0, "num_heads must be a multiple of 4"

    # RaggedMHAOperand.cache_length() returns full key length directly, so the
    # SM100 tensor-core scorer and the scalar fallback both run with
    # _is_cache_length_accurate=True (skip adding seq_len in the kernel).
    # The scorer uses tcgen05/TMA (Blackwell-only), so gate on
    # _has_blackwell_tcgen05(): H100/A100/other NVIDIA and AMD take the scalar
    # fallback. The SM100 scorer stages a BM_key-row K tile with one TMA copy,
    # so a paged K cache must have page_size == 0 (contiguous, as this ragged
    # path is) or a multiple of BM_key; any other page_size falls back too.
    comptime if (
        _has_blackwell_tcgen05()
        and (
            num_heads == 64
            or num_heads == 32
            or num_heads == 8
            or num_heads == 4
        )
        and depth == 128
        and (
            type_of(k_operand).page_size == 0
            or type_of(k_operand).page_size % _BM_KEY == 0
        )
    ):
        fp8_index_score_sm100[
            dtype,
            type_of(k_operand),
            type_of(ks_operand),
            num_heads,
            depth,
            _is_cache_length_accurate=True,
            # GLM 5.x MTP decodes 6 tokens (num_draft_tokens + 1), which the
            # default 4-token N-tile at nh=32 covers with two blocks spending 256
            # MMA columns on 192 live ones. 3 divides 6, so it tiles the step
            # exactly at 96 columns -- and unlike 6, its TMEM footprint still
            # leaves room for two co-resident CTAs. Inert wherever 3 tokens are
            # not a legal UMMA N or the default already divides the step (nh=64).
            #
            # This is a speculative-decode tile and nothing else. The bound is
            # arithmetic, not a threshold: a 3-token tile needs `msl // 3` blocks
            # where the default needs `ceildiv(msl, 4)`, and
            # `msl // 3 <= ceildiv(msl, 4)` iff `msl <= 9`, so the gap grows
            # without bound and only max_seq_len in {3, 6, 9} survives at nh=32
            # (12 is excluded because the default tile already divides it). A
            # 2-token hint was tried first, back when only 64 columns could be
            # hoisted without spilling -- but it needs THREE blocks to cover the
            # step, so it pays 1.5x the CTA prologues to reach the same hoist. Once
            # the index arithmetic was narrowed to 32-bit, every width hoists at
            # zero spill and 3 tokens is simply the exact divisor.
            N_TOKENS_ALT=SPEC_DECODE_N_TOKENS_ALT,
        ](
            output,
            q,
            q_s.as_immut(),
            k_operand,
            ks_operand,
            valid_length,
            batch_size,
            max_seq_len,
            max_num_keys,
            False,
            ctx,
        )
    else:
        comptime assert num_heads % 16 == 0, (
            "the scalar fp8_index_kernel tiles heads by thread_dim_y == 8 and"
            " is unvalidated below 16 heads; num_heads in {4, 8} requires the"
            " SM100 tensor-core path"
        )
        comptime block_tile_shape: Array[Int, 2] = [512, 128]
        comptime BM = block_tile_shape[0]
        comptime BN = block_tile_shape[1]
        comptime smem_use = size_of[
            IndexSmemStorage[dtype, num_heads, depth, BN]
        ]()
        comptime smem_available = ctx.default_device_info.shared_memory_per_multiprocessor - 1024

        comptime kernel = fp8_index_kernel[
            dtype,
            type_of(output).LayoutType,
            type_of(q).LayoutType,
            type_of(q_s).LayoutType,
            type_of(k_operand),
            type_of(ks_operand),
            block_tile_shape,
            type_of(valid_length).LayoutType,
            num_heads,
            depth,
            _is_cache_length_accurate=True,
            OutputEngine=output.Engine,
            QEngine=q.Engine,
            QSEngine=q_s.Engine,
            VLEngine=valid_length.Engine,
        ]

        ctx.enqueue_function[kernel](
            output,
            q.as_immut(),
            q_s,
            k_operand,
            ks_operand,
            valid_length.as_immut(),
            grid_dim=(
                batch_size,
                max_seq_len,
                ceildiv(max_num_keys, BM),
            ),
            block_dim=(16, 8, 1),
            shared_mem_bytes=smem_use,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(smem_available)
            ),
        )


@__name(t"fp8_index_matmul_max_{dtype}")
def _index_matmul_max[
    dtype: DType,
    output_layout: Layout,
    q_layout: Layout,
    qs_layout: Layout,
    k_layout: Layout,
    k_type: MHAOperand,
](
    output: LayoutTensor[.float32, output_layout, MutAnyOrigin],
    q: LayoutTensor[dtype, q_layout, ImmutAnyOrigin],
    q_s: LayoutTensor[.float32, qs_layout, MutAnyOrigin],
    k: LayoutTensor[dtype, k_layout, ImmutAnyOrigin],
    valid_length: LayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    k_lut: k_type,
    max_seq_len: Int32,
    max_num_keys: Int32,
):
    comptime num_heads = q_layout.shape[1].value()
    comptime depth = q_layout.shape[2].value()

    var batch_idx, head_idx = udivmod(block_idx.z, num_heads)
    var seq_idx = block_idx.x * 16 + thread_idx.x
    var key_idx = block_idx.y * 16 + thread_idx.y

    var start_of_seq = valid_length[batch_idx][0]
    var end_of_seq = valid_length[batch_idx + 1][0]
    var seq_len = end_of_seq - start_of_seq

    var num_keys = k_lut.cache_length(batch_idx)
    var k_row_start = k_lut.row_idx(UInt32(batch_idx), 0)

    if key_idx >= num_keys or seq_idx >= Int(seq_len):
        return

    var q_ptr = q.ptr_at_offset(Index(start_of_seq, 0, 0))
    var k_ptr = k.ptr_at_offset(Index(k_row_start, 0, 0))
    var o_ptr = output.ptr_at_offset(Index(start_of_seq, 0, 0))

    var q_runtime_layout = RuntimeLayout[q_layout].row_major(
        Index(seq_len, num_heads, depth)
    )
    var q_batch = LayoutTensor[dtype, q_layout](q_ptr, q_runtime_layout)

    var k_runtime_layout = RuntimeLayout[k_layout].row_major(
        Index(num_keys, 1, depth)
    )
    var k_batch = LayoutTensor[dtype, k_layout, k.origin](
        k_ptr, k_runtime_layout
    )

    # The logits buffer is one dense `[., max_num_keys, num_heads]` allocation, so
    # its row stride is `max_num_keys` and NOT this entry's own key count. On a
    # ragged batch those differ, and striding by `num_keys` aliases every entry
    # after the first onto the wrong rows.
    var o_runtime_layout = RuntimeLayout[output_layout].row_major(
        Index(seq_len, Int(max_num_keys), num_heads)
    )
    var o_batch = LayoutTensor[.float32, output_layout, MutAnyOrigin](
        o_ptr, o_runtime_layout
    )

    # Cast each FP8 code to F32 before multiply so we match TileLang-style GEMM
    # (FP8×FP8 in low precision can saturate/widen differently than F32 products).
    var accum = Float32(0.0)
    for d in range(Int(depth)):
        var kd = k_batch[key_idx, 0, d][0].cast[.float32]()
        var qd = q_batch[seq_idx, head_idx, d][0].cast[.float32]()
        accum += kd * qd

    accum = max(accum, 0) * q_s[start_of_seq + UInt32(seq_idx), head_idx][0]
    o_batch[seq_idx, key_idx, head_idx] = accum


@__name(t"fp8_index_reduce_logits")
def _reduce_logits[
    logits_layout: Layout,
    output_layout: Layout,
    ks_layout: Layout,
    k_type: MHAOperand,
](
    logits: LayoutTensor[.float32, logits_layout, MutAnyOrigin],
    output: LayoutTensor[.float32, output_layout, MutAnyOrigin],
    k_s: LayoutTensor[.float32, ks_layout, MutAnyOrigin],
    valid_length: LayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    k_lut: k_type,
    max_num_keys: Int32,
):
    comptime num_heads = logits_layout.shape[2].value()
    var batch_idx = block_idx.z
    var seq_idx = block_idx.x * 16 + thread_idx.x
    var key_idx = block_idx.y * 16 + thread_idx.y

    var start_of_seq = valid_length[batch_idx][0]
    var end_of_seq = valid_length[batch_idx + 1][0]
    var seq_len = end_of_seq - start_of_seq

    var num_keys = k_lut.cache_length(batch_idx)
    var k_row_offset = k_lut.row_idx(UInt32(batch_idx), 0)

    if seq_idx >= Int(seq_len) or key_idx >= num_keys:
        return

    var o_ptr = output.ptr_at_offset(Index(start_of_seq, 0))
    var logits_ptr = logits.ptr_at_offset(Index(start_of_seq, 0, 0))
    var k_s_ptr = k_s.ptr_at_offset(Index(k_row_offset))

    # Both the score buffer and the logits buffer are dense over `max_num_keys`,
    # so both stride by it rather than by this entry's own key count -- see the
    # matmul kernel above. `k_s` is the exception: it is already offset to this
    # entry's rows, so it stays keyed on `num_keys`.
    var o_runtime_layout = RuntimeLayout[output_layout].row_major(
        Index(seq_len, Int(max_num_keys))
    )
    var o_batch = LayoutTensor[.float32, output_layout, MutAnyOrigin](
        o_ptr, o_runtime_layout
    )
    var k_s_runtime_layout = RuntimeLayout[ks_layout].row_major(Index(num_keys))

    var logits_runtime_layout = RuntimeLayout[logits_layout].row_major(
        Index(seq_len, Int(max_num_keys), num_heads)
    )
    var logits_batch = LayoutTensor[.float32, logits_layout, MutAnyOrigin](
        logits_ptr, logits_runtime_layout
    )
    var k_s_batch = LayoutTensor[.float32, ks_layout, MutAnyOrigin](
        k_s_ptr, k_s_runtime_layout
    )

    var sum = Float32(0.0)
    for head in range(num_heads):
        sum += logits_batch[seq_idx, key_idx, head][0]

    o_batch[seq_idx, key_idx] = sum * k_s_batch[key_idx][0]


@always_inline
def fp8_index_naive[
    dtype: DType,
    //,
    num_heads: Int,
    depth: Int,
](
    output: TileTensor[.float32, ...],
    q: TileTensor[mut=False, dtype, ...],
    q_s: TileTensor[.float32, ...],
    k: TileTensor[mut=False, dtype, ...],
    k_s: TileTensor[.float32, ...],
    valid_length: TileTensor[mut=False, .uint32, ...],
    cache_row_offsets: TileTensor[mut=False, .uint32, ...],
    batch_size: Int,
    max_seq_len: Int,
    max_num_keys: Int,
    ctx: DeviceContext,
) raises:
    """Computes the FP8 index/gather score via a two-pass matmul-then-reduce reference path.

    Enqueues `_index_matmul_max` to produce per-head logits followed by
    `_reduce_logits` to sum across heads and apply the per-key scale, serving
    as a correctness reference for the optimized tensor-core kernels.

    Parameters:
        dtype: Data type of the query and key tensors.
        num_heads: Number of attention heads.
        depth: Per-head feature depth.

    Args:
        output: Output score tensor of shape `[total_seq_len, max_num_keys]`.
        q: Query tensor of shape `[total_seq_len, num_heads, depth]`.
        q_s: Per-query scale tensor of shape `[total_seq_len, num_heads]`.
        k: Key tensor of shape `[total_keys, 1, depth]`.
        k_s: Per-key scale tensor of shape `[total_keys]`.
        valid_length: Cumulative sequence offsets of shape `[batch_size + 1]`.
        cache_row_offsets: Per-batch row offsets into the paged key cache.
        batch_size: Number of sequences in the batch.
        max_seq_len: Maximum sequence length across the batch.
        max_num_keys: Maximum key count across the batch.
        ctx: Device context used to enqueue the kernels.

    Raises:
        When the underlying kernel enqueue reports a device-side error.
    """
    # Construct LayoutTensors from TileTensor ptr + dimensions for the
    # internal GPU kernels (_index_matmul_max, _reduce_logits) and
    # RaggedMHAOperand, which all require LayoutTensor with specific
    # Layout.row_major() representations.
    comptime q_layout = Layout.row_major(UNKNOWN_VALUE, num_heads, depth)
    comptime qs_layout = Layout.row_major(UNKNOWN_VALUE, num_heads)
    comptime k_layout = Layout.row_major(UNKNOWN_VALUE, 1, depth)
    comptime ks_layout = Layout.row_major(UNKNOWN_VALUE)
    comptime output_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime vl_layout = Layout.row_major(UNKNOWN_VALUE)

    var total_seq_len = Int(q.dim[0]())
    var total_keys = Int(k.dim[0]())
    var vl_size = Int(valid_length.dim[0]())

    var q_lt = LayoutTensor[dtype, q_layout, ImmutAnyOrigin](
        rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](q.ptr),
        RuntimeLayout[q_layout].row_major(
            Index(total_seq_len, num_heads, depth)
        ),
    )
    var q_s_lt = LayoutTensor[.float32, qs_layout, MutAnyOrigin](
        rebind[UnsafePointer[Float32, MutAnyOrigin]](q_s.ptr),
        RuntimeLayout[qs_layout].row_major(Index(total_seq_len, num_heads)),
    )
    var k_lt = LayoutTensor[dtype, k_layout, ImmutAnyOrigin](
        rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](k.ptr),
        RuntimeLayout[k_layout].row_major(Index(total_keys, 1, depth)),
    )
    var k_s_lt = LayoutTensor[.float32, ks_layout, MutAnyOrigin](
        rebind[UnsafePointer[Float32, MutAnyOrigin]](k_s.ptr),
        RuntimeLayout[ks_layout].row_major(Index(total_keys)),
    )
    var output_lt = LayoutTensor[.float32, output_layout, MutAnyOrigin](
        rebind[UnsafePointer[Float32, MutAnyOrigin]](output.ptr),
        RuntimeLayout[output_layout].row_major(
            Index(total_seq_len, max_num_keys)
        ),
    )
    var valid_length_lt = LayoutTensor[.uint32, vl_layout, ImmutAnyOrigin](
        rebind[UnsafePointer[UInt32, ImmutAnyOrigin]](valid_length.ptr),
        RuntimeLayout[vl_layout].row_major(Index(vl_size)),
    )

    var cro_size = Int(cache_row_offsets.dim[0]())
    var cro_buf = TileTensor(
        rebind[UnsafePointer[UInt32, ImmutAnyOrigin]](cache_row_offsets.ptr),
        row_major(Coord(cro_size)),
    )
    var k_buf = TileTensor(
        rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](k.ptr),
        row_major(Coord(total_keys, Idx[1], Idx[depth])),
    )
    var k_operand = RaggedMHAOperand(k_buf, cro_buf)

    var logits_size = batch_size * max_seq_len * max_num_keys * num_heads

    var logits_dev = ctx.enqueue_create_buffer[.float32](logits_size)
    logits_dev.enqueue_fill(Float32(0.0))

    comptime logits_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads
    )

    var logits_runtime_layout = RuntimeLayout[logits_layout].row_major(
        Index(batch_size * max_seq_len, max_num_keys, num_heads)
    )

    var logits_tensor = LayoutTensor[
        .float32,
        logits_layout,
    ](logits_dev.unsafe_ptr(), logits_runtime_layout)

    comptime mm = _index_matmul_max[
        dtype,
        logits_layout,
        q_layout,
        qs_layout,
        k_layout,
        type_of(k_operand),
    ]

    ctx.enqueue_function[mm](
        logits_tensor,
        q_lt,
        q_s_lt,
        k_lt,
        valid_length_lt,
        k_operand,
        Int32(max_seq_len),
        Int32(max_num_keys),
        grid_dim=(
            ceildiv(max_seq_len, 16),
            ceildiv(max_num_keys, 16),
            batch_size * num_heads,
        ),
        block_dim=(16, 16, 1),
    )

    comptime reduce_logits = _reduce_logits[
        logits_layout,
        output_layout,
        ks_layout,
        type_of(k_operand),
    ]

    ctx.enqueue_function[reduce_logits](
        logits_tensor,
        output_lt,
        k_s_lt,
        valid_length_lt,
        k_operand,
        Int32(max_num_keys),
        grid_dim=(
            ceildiv(max_seq_len, 16),
            ceildiv(max_num_keys, 16),
            batch_size,
        ),
        block_dim=(16, 16, 1),
    )

    _ = logits_dev
