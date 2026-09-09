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

"""Implements CPU-based multi-head attention kernels, including flash attention and KV-cache-backed attention variants."""

from std.collections import OptionalReg
from std.math import align_down, align_up, ceildiv, exp

from std.os import abort
from std.sys import align_of, simd_width_of
from std.sys.info import CompilationTarget

from std.algorithm import tile, vectorize

from max.algorithm import sync_parallelize
from max.gpu.host import DeviceContext
from max.algorithm.reduction import (
    _simd_max,
    _simd_max_elementwise,
    _simd_sum,
    _simd_sum_elementwise,
    map_reduce,
)
from kv_cache.types import KVCacheT
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout.int_tuple import to_index_list
from layout.tile_tensor import stack_allocation as tt_stack_allocation
from linalg.accumulate import _Accumulator
from linalg.matmul.cpu.apple_accelerate import (
    _cblas_f32,
    use_apple_accelerate_lib,
)
from linalg.transpose import transpose_inplace
from linalg.utils import partition_work
from std.memory import (
    alloc,
    dealloc,
    unsafe_memset_zero,
    unsafe_stack_allocation,
)
from std.memory.alloc import ManagedAllocation, Layout as AllocLayout
from nn.attention.mha_mask import MHAMask
from max.runtime.asyncrt import parallelism_level
from max.runtime.tracing import Trace, TraceLevel, trace_arg

from std.utils import Index, IndexList


@fieldwise_init
struct _MatmulConfig:
    var col_sizes: List[Int]
    var row_sizes: List[Int]
    var gemv_sizes: List[Int]
    var pack_sizes: List[Int]

    @staticmethod
    def _get_config() -> _MatmulConfig:
        comptime if CompilationTarget.has_neon():
            return _MatmulConfig(
                col_sizes=[4, 3, 2, 1],
                row_sizes=[6, 4, 1],
                gemv_sizes=[32, 4, 1],
                pack_sizes=[32, 8, 4, 1],
            )
        elif CompilationTarget.has_avx512f():
            return _MatmulConfig(
                col_sizes=[4, 3, 2, 1],
                row_sizes=[6, 4, 1],
                gemv_sizes=[64, 16, 4, 1],
                pack_sizes=[64, 16, 8, 4, 1],
            )
        else:
            return _MatmulConfig(
                col_sizes=[3, 2, 1],
                row_sizes=[4, 1],
                gemv_sizes=[64, 16, 4, 1],
                pack_sizes=[64, 16, 8, 4, 1],
            )


struct _Matmul[dtype: DType, simd_width: Int]:
    comptime _matmul_config = _MatmulConfig._get_config()

    comptime _input_fn_type = def[simd_width: Int](
        x: Int, y: Int
    ) capturing -> SIMD[Self.dtype, simd_width]

    @staticmethod
    @always_inline
    def _inner_loop_a_lane[
        tile_m: Int, tile_n: Int
    ](
        K: Int,
        a_ptr: UnsafePointer[Scalar[Self.dtype], _],
        a_stride: Int,
        b_ptr: UnsafePointer[Scalar[Self.dtype], _],
        b_stride: Int,
        mut c_tile: _Accumulator[Self.dtype, tile_m, tile_n, Self.simd_width],
    ):
        var ak_ptr = a_ptr
        var bk_ptr = b_ptr

        @always_inline
        def loop_body[
            lane_count: Int
        ](k: Int) {mut ak_ptr, mut bk_ptr, mut c_tile, imm}:
            var a_tile = Array[_, tile_m](
                fill_with=lambda (m: Int) -> SIMD[
                    Self.dtype, lane_count
                ]: ak_ptr.load[width=lane_count](m * a_stride)
            )

            ak_ptr += lane_count

            comptime for k in range(lane_count):
                comptime for n in range(tile_n):
                    var b_data = bk_ptr.load[width=Self.simd_width](
                        n * Self.simd_width
                    )

                    comptime for m in range(tile_m):
                        c_tile.fma(m, n, a_tile[m][k], b_data)

                bk_ptr += b_stride

        tile[[Self.simd_width, 1]](0, K, loop_body)
        # TODO(MOCO-2074): Suppress false positive unused var warning.
        _ = ak_ptr
        _ = bk_ptr

    @staticmethod
    @always_inline
    def _inner_loop_a_broadcast[
        tile_m: Int, tile_n: Int
    ](
        K: Int,
        a_ptr: UnsafePointer[Scalar[Self.dtype], _],
        a_stride: Int,
        b_ptr: UnsafePointer[Scalar[Self.dtype], _],
        b_stride: Int,
        mut c_tile: _Accumulator[Self.dtype, tile_m, tile_n, Self.simd_width],
    ):
        var ak_ptr = a_ptr
        var bk_ptr = b_ptr

        @always_inline
        def loop_body[
            unroll_factor: Int
        ](k: Int) {mut ak_ptr, mut bk_ptr, mut c_tile, imm}:
            var b_tile = Array[SIMD[Self.dtype, Self.simd_width], tile_n](
                fill=0
            )

            comptime for k in range(unroll_factor):
                comptime for n in range(tile_n):
                    b_tile[n] = bk_ptr.load[width=Self.simd_width](
                        n * Self.simd_width
                    )

                comptime for m in range(tile_m):
                    var a_data = ak_ptr.load(m * a_stride)

                    comptime for n in range(tile_n):
                        c_tile.fma(m, n, a_data, b_tile[n])

                ak_ptr += 1
                bk_ptr += b_stride

        tile[[2, 1]](0, K, loop_body)
        # TODO(MOCO-2074): Suppress false positive unused var warning.
        _ = ak_ptr
        _ = bk_ptr

    @no_inline
    @staticmethod
    def _matmul_packed(
        M: Int,
        N: Int,
        K: Int,
        a_ptr: UnsafePointer[Scalar[Self.dtype], _],
        a_stride: Int,
        b_ptr: UnsafePointer[Scalar[Self.dtype], _],
        c_ptr: UnsafePointer[mut=True, Scalar[Self.dtype], _],
        c_stride: Int,
        accumulate: Bool = False,
    ):
        var am_ptr = a_ptr
        var cm_ptr = c_ptr

        def process_rows[tile_m: Int](m: Int) {mut am_ptr, mut cm_ptr, imm}:
            var bn_ptr = b_ptr
            var cn_ptr = cm_ptr

            def process_cols[
                tile_n: Int
            ](n_unscaled: Int) {mut bn_ptr, mut cn_ptr, imm}:
                var c_tile = _Accumulator[
                    Self.dtype, tile_m, tile_n, Self.simd_width
                ]()

                if accumulate:
                    c_tile.load(cn_ptr, c_stride)
                else:
                    c_tile.init(0.0)

                comptime if CompilationTarget.has_neon():
                    Self._inner_loop_a_lane(
                        K, am_ptr, a_stride, bn_ptr, N, c_tile
                    )
                else:
                    Self._inner_loop_a_broadcast(
                        K, am_ptr, a_stride, bn_ptr, N, c_tile
                    )

                c_tile.store(cn_ptr, c_stride)

                bn_ptr += tile_n * Self.simd_width
                cn_ptr += tile_n * Self.simd_width

            tile[Self._matmul_config.col_sizes](
                0, ceildiv(N, Self.simd_width), process_cols
            )

            am_ptr += tile_m * a_stride
            cm_ptr += tile_m * c_stride

            # TODO(MOCO-2074): Suppress false positive unused var warning.
            _ = bn_ptr
            _ = cn_ptr

        tile[Self._matmul_config.row_sizes](0, M, process_rows)
        # TODO(MOCO-2074): Suppress false positive unused var warning.
        _ = am_ptr
        _ = cm_ptr

    @no_inline
    @staticmethod
    def _pack_buffer_transposed[
        input_b_fn: Self._input_fn_type, static_k: Int
    ](
        packed_ptr: UnsafePointer[mut=True, Scalar[Self.dtype], _],
        N: Int,
        dynamic_k: Int,
    ):
        var K = static_k if static_k != UNKNOWN_VALUE else dynamic_k

        var aligned_n = align_up(N, Self.simd_width)

        # Use a conservative SIMD width for transposing. Using a wider native
        # SIMD width has not been observed to improve performance and causes
        # code size to unnecessarily increase.
        comptime transpose_width = 4
        comptime tile_sizes: List[Int] = [transpose_width, 1]

        var transpose_buffer = tt_stack_allocation[dtype=Self.dtype,](
            row_major[transpose_width, transpose_width]()
        )

        @always_inline
        def process_tile[
            tile_n: Int, tile_k: Int
        ](n: Int, k: Int) {mut transpose_buffer, imm}:
            comptime if transpose_width == tile_n == tile_k:
                # Use an optimized path to transpose a square tile of the
                # input tensor.
                comptime for i in range(transpose_width):
                    var val = input_b_fn[simd_width=transpose_width](n + i, k)
                    transpose_buffer.store_linear[width=transpose_width](
                        Index(i, 0), val
                    )

                transpose_inplace[4, 4](transpose_buffer)

                comptime for i in range(transpose_width):
                    var val = transpose_buffer.load_linear[
                        width=transpose_width
                    ](Index(i, 0))
                    packed_ptr.store((k + i) * aligned_n + n, val)

            else:
                # Fallback to strided loads and stores of the tensors.
                #
                # Note that in the common case, `K` is statically known and is
                # a multiple of `transpose_width`, so the case to optimize for
                # `tile_n=1` and `tile_k=transpose_width`.
                comptime for nn in range(tile_n):
                    var val = input_b_fn[simd_width=tile_k](n + nn, k)

                    comptime for kk in range(tile_k):
                        packed_ptr.store(
                            (k + kk) * aligned_n + (n + nn), val[kk]
                        )

        tile[tile_sizes, tile_sizes](0, 0, N, K, process_tile)
        _ = transpose_buffer

        if aligned_n != N:
            for k in range(K):
                unsafe_memset_zero(
                    packed_ptr + k * aligned_n + N, aligned_n - N
                )

    @no_inline
    @staticmethod
    def _pack_buffer[
        input_b_fn: Self._input_fn_type
    ](
        packed_ptr: UnsafePointer[mut=True, Scalar[Self.dtype], _],
        N: Int,
        K: Int,
    ):
        var output_ptr = packed_ptr
        var aligned_n = align_up(N, Self.simd_width)

        for _k in range(K):
            # TODO(MOCO-4664): `var _k` copy-captures the loop variable to work
            # around wrong debug-info scopes on implicit nested-scope captures.
            @always_inline
            def packed_copy[_simd_width: Int](idx: Int) {var _k, imm}:
                var val = input_b_fn[_simd_width](idx, _k)
                output_ptr.store(idx, val)

            tile[Self._matmul_config.pack_sizes](0, N, packed_copy)

            if aligned_n != N:
                unsafe_memset_zero(output_ptr + N, aligned_n - N)

            output_ptr += aligned_n

    @no_inline
    @staticmethod
    def _gemv_transposed[
        input_b_fn: Self._input_fn_type, static_k: Int
    ](
        N: Int,
        dynamic_k: Int,
        a_ptr: UnsafePointer[Scalar[Self.dtype], _],
        c_ptr: UnsafePointer[mut=True, Scalar[Self.dtype], _],
    ):
        var K = static_k if static_k != UNKNOWN_VALUE else dynamic_k
        var cn_ptr = c_ptr

        @always_inline
        def process_cols[tile_n: Int](n: Int) {mut cn_ptr, imm}:
            @always_inline
            def do_reduce[
                _simd_width: SIMDLength
            ](
                start: Int,
                end: Int,
                mut accum: Array[SIMD[Self.dtype, _simd_width], tile_n],
            ) {imm}:
                for _k in range(start, end, _simd_width):
                    var a_data = a_ptr.load[width=_simd_width](_k)

                    comptime for nn in range(tile_n):
                        var b_data = input_b_fn[_simd_width](n + nn, _k)
                        accum[nn] = b_data.fma(a_data, accum[nn])

            @always_inline
            def do_reduce_accum[
                target_width: Int, _simd_width: SIMDLength
            ](accum: Array[SIMD[Self.dtype, _simd_width], tile_n]) {
                imm
            } -> Array[SIMD[Self.dtype, target_width], tile_n]:
                var accum_reduce = Array[
                    SIMD[Self.dtype, target_width], tile_n
                ](fill=0)

                comptime for nn in range(tile_n):
                    accum_reduce[nn] = accum[nn].reduce_add[target_width]()
                return accum_reduce^

            comptime unroll_factor = 2
            comptime unroll_simd_width = Self.simd_width * unroll_factor

            var unroll_loop_end = align_down(K, unroll_simd_width)
            var unroll_accum = Array[
                SIMD[Self.dtype, unroll_simd_width], tile_n
            ](fill=0)
            do_reduce(0, unroll_loop_end, unroll_accum)

            var simd_loop_end = align_down(K, Self.simd_width)
            var simd_accum = do_reduce_accum[Self.simd_width](unroll_accum)
            do_reduce(unroll_loop_end, simd_loop_end, simd_accum)

            var scalar_accum = do_reduce_accum[1](simd_accum)
            do_reduce(simd_loop_end, K, scalar_accum)

            comptime for nn in range(tile_n):
                cn_ptr.store(nn, scalar_accum[nn])

            cn_ptr += tile_n

        tile[[4, 1]](0, N, process_cols)
        # TODO(MOCO-2074): Suppress false positive unused var warning.
        _ = K
        _ = cn_ptr

    @no_inline
    @staticmethod
    def _gemv[
        input_b_fn: Self._input_fn_type
    ](
        N: Int,
        K: Int,
        a_ptr: UnsafePointer[Scalar[Self.dtype], _],
        c_ptr: UnsafePointer[mut=True, Scalar[Self.dtype], _],
        accumulate: Bool = False,
    ):
        var cn_ptr = c_ptr

        @always_inline
        def process_cols[_simd_width: Int](n: Int) {mut cn_ptr, imm}:
            var accum = SIMD[Self.dtype, _simd_width]()

            for k in range(K):
                var b_data = input_b_fn[_simd_width](n, k)
                accum = b_data.fma(a_ptr[k], accum)

            if accumulate:
                accum += cn_ptr.load[width=_simd_width]()

            cn_ptr.store(accum)
            cn_ptr += _simd_width

        tile[Self._matmul_config.gemv_sizes](0, N, process_cols)
        # TODO(MOCO-2074): Suppress false positive unused var warning.
        _ = cn_ptr

    @no_inline
    @staticmethod
    def _matmul[
        input_b_fn: Self._input_fn_type,
        *,
        transpose_b: Bool = False,
        static_k: Int = UNKNOWN_VALUE,
    ](
        M: Int,
        N: Int,
        K: Int,
        a_ptr: UnsafePointer[Scalar[Self.dtype], _],
        a_stride: Int,
        packed_ptr: UnsafePointer[mut=True, Scalar[Self.dtype], _],
        c_ptr: UnsafePointer[mut=True, Scalar[Self.dtype], _],
        c_stride: Int,
        accumulate: Bool = False,
    ) raises:
        if M == 1:
            comptime if transpose_b:
                # Transpose is implemented for the K tensor and accumulation
                # is used with the V tensor, so simplify the implementation by
                # falling back to the general path.
                if not accumulate:
                    return Self._gemv_transposed[input_b_fn, static_k](
                        N, K, a_ptr, c_ptr
                    )
            else:
                return Self._gemv[input_b_fn](
                    N, K, a_ptr, c_ptr, accumulate=accumulate
                )

        comptime if transpose_b:
            Self._pack_buffer_transposed[input_b_fn, static_k](packed_ptr, N, K)
        else:
            Self._pack_buffer[input_b_fn](packed_ptr, N, K)

        comptime if use_apple_accelerate_lib[
            Self.dtype, Self.dtype, Self.dtype
        ]():
            return _cblas_f32(
                Int32(M),
                Int32(N),
                Int32(K),
                Int32(a_stride),
                Int32(align_up(N, Self.simd_width)),
                Int32(c_stride),
                Float32(1.0),
                Float32(1.0) if accumulate else Float32(0.0),
                c_ptr.bitcast[Float32](),
                a_ptr.bitcast[Float32](),
                packed_ptr.bitcast[Float32](),
            )

        Self._matmul_packed(
            M,
            align_up(N, Self.simd_width),
            K,
            a_ptr,
            a_stride,
            packed_ptr,
            c_ptr,
            c_stride,
            accumulate=accumulate,
        )


struct _FlashAttentionConfig[
    dtype: DType,
    rank: Int,
    simd_width: Int,
    output_static_shape: IndexList[rank],
](Defaultable):
    var block_m: Int
    var qk_block_n: Int
    var o_block_n: Int

    def __init__(out self):
        self.qk_block_n = 128
        self.o_block_n = 128

        # Set a target size for the output block array.
        comptime output_target_size = 8192

        comptime depth_static_dim = Self.output_static_shape[Self.rank - 1]

        comptime if depth_static_dim != UNKNOWN_VALUE:
            # Extract the static depth dimension with a guard against zero.
            var depth_dim = max(depth_static_dim, 1)

            # Compute the number of columns for the output block array. If the
            # count is too large, then use the default size.
            self.o_block_n = align_up(
                depth_dim if depth_dim <= 256 else self.o_block_n,
                Self.simd_width,
            )

        # Compute the number of rows per iteration, but constrain this number
        # as other buffers are allocated to this size too.
        self.block_m = align_down(output_target_size // self.o_block_n, 4)
        self.block_m = min(max(self.block_m, 1), 64)


struct _FlashAttention[
    dtype: DType,
    rank: Int,
    q_origin: ImmOrigin,
    output_origin: Origin[mut=True],
    //,
    input_q_ptr_fn: def(IndexList[rank]) capturing -> UnsafePointer[
        Scalar[dtype], q_origin
    ],
    input_k_fn: def[simd_width: Int, rank: Int](
        idx: IndexList[rank]
    ) capturing -> SIMD[dtype, simd_width],
    input_v_fn: def[simd_width: Int, rank: Int](
        idx: IndexList[rank]
    ) capturing -> SIMD[dtype, simd_width],
    mask_fn: def[simd_width: SIMDLength, mask_rank: Int](
        idx: IndexList[mask_rank],
        score_vec: SIMD[dtype, simd_width],
        kv_cache_length: Int,
    ) capturing -> SIMD[dtype, simd_width],
    mask_rank: Int,
    output_ptr_fn: def(IndexList[rank]) capturing -> UnsafePointer[
        Scalar[dtype], output_origin
    ],
    q_length_fn: def(batch: Int) capturing -> Int,
    kv_length_fn: def(batch: Int) capturing -> Int,
    kv_cache_length_fn: def(batch: Int) capturing -> Int,
    padded_output_shape: IndexList[rank],
    *,
    simd_width: Int = simd_width_of[dtype](),
]:
    comptime _matmul = _Matmul[Self.dtype, Self.simd_width]
    comptime _config = _FlashAttentionConfig[
        Self.dtype, Self.rank, Self.simd_width, Self.padded_output_shape
    ]()
    comptime _depth_static_dim = Self.padded_output_shape[Self.rank - 1]

    @staticmethod
    def _online_softmax[
        _mask_fn: def[simd_width: SIMDLength](
            m: Int, n: Int, score_vec: SIMD[Self.dtype, simd_width]
        ) capturing -> SIMD[Self.dtype, simd_width],
    ](
        qk_block_ptr: UnsafePointer[mut=True, Scalar[Self.dtype], _],
        o_block_ptr: UnsafePointer[mut=True, Scalar[Self.dtype], _],
        max_vals: UnsafePointer[mut=True, Scalar[Self.dtype], _],
        sum_vals: UnsafePointer[mut=True, Scalar[Self.dtype], _],
        count_m: Int,
        count_n: Int,
        kv_seq_cnt: Int,
        scale: Float32,
        sink_weight: Optional[Scalar[Self.dtype]] = None,
    ):
        comptime assert (
            Self.dtype.is_floating_point()
        ), "dtype must be floating point"
        var qk_row_ptr = qk_block_ptr
        var o_row_ptr = o_block_ptr

        var sink_logit: Scalar[Self.dtype] = 0
        var do_sink = sink_weight is not None
        if do_sink:
            sink_logit = sink_weight.value()

        comptime layout_1d = Layout.row_major(UNKNOWN_VALUE)
        for m in range(count_m):
            var qk_row = LayoutTensor[Self.dtype, layout_1d, _](
                qk_row_ptr,
                RuntimeLayout[layout_1d].row_major(IndexList[1](kv_seq_cnt)),
            )

            @__parameter
            @always_inline
            def pass1_input_gen_fn[
                _dtype: DType, _simd_width: Int
            ](idx: Int) -> SIMD[_dtype, _simd_width]:
                var val = qk_row_ptr.load[width=_simd_width](idx)
                return _mask_fn(m, idx, val * scale.cast[Self.dtype]()).cast[
                    _dtype
                ]()

            @always_inline
            @__parameter
            def output_fn[
                _dtype: DType, width: SIMDLength, rank: Int
            ](idx: Int, val: SIMD[_dtype, width]):
                qk_row.store(
                    IndexList[1](idx), rebind[SIMD[Self.dtype, width]](val)
                )

            # Update the row with the scale and mask. Find the maximum value
            # of the row to bias the exponential function below for numeric
            # stability.
            var max_val = map_reduce[
                Self.simd_width,
                Self.dtype,
                Self.dtype,
                origin_of()._mlir_origin,
                pass1_input_gen_fn,
                origin_of()._mlir_origin,
                _simd_max_elementwise,
                _simd_max,
                output_fn,
            ](qk_row.size(), max_vals[m])

            if do_sink:
                max_val = max(max_val, sink_logit)

            @__parameter
            @always_inline
            def pass2_input_gen_fn[
                _dtype: DType, _simd_width: Int
            ](idx: Int) -> SIMD[_dtype, _simd_width]:
                var val = qk_row_ptr.load[width=_simd_width](idx)
                return rebind[SIMD[_dtype, _simd_width]](exp(val - max_val))

            # Update the row with the exponential of each value and accumulate
            # the result.
            var accum_val = map_reduce[
                Self.simd_width,
                Self.dtype,
                Self.dtype,
                origin_of()._mlir_origin,
                pass2_input_gen_fn,
                origin_of()._mlir_origin,
                _simd_sum_elementwise,
                _simd_sum,
                output_fn,
            ](qk_row.size(), 0)

            if do_sink:
                accum_val += exp(sink_logit - max_val)

            var fixup_val = exp(max_vals[m] - max_val)

            # Update the running maximum and sum for the row.
            max_vals[m] = max_val
            sum_vals[m] = sum_vals[m] * fixup_val + accum_val

            @always_inline
            def do_correction[
                _simd_width: Int
            ](idx: Int) {o_row_ptr, fixup_val, mut}:
                var val = o_row_ptr.load[width=_simd_width](idx)
                o_row_ptr.store(idx, val * fixup_val)

            vectorize[Self.simd_width, unroll_factor=2](count_n, do_correction)

            qk_row_ptr += Self._config.qk_block_n
            o_row_ptr += Self._config.o_block_n

    @staticmethod
    def run(
        num_batches: Int,
        num_heads: Int,
        depth_dim: Int,
        num_kv_heads: Int,
        # Max sequence length of query states.
        max_seq_len: Int,
        scale: Float32,
        sink_weights: OptionalReg[
            LayoutTensor[
                Self.dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
            ]
        ] = None,
        ctx: Optional[DeviceContext] = None,
    ):
        var kv_group_count = num_heads // num_kv_heads

        # Compute the maximum size in elements for the common packed buffer.
        var packed_qk_size = Self._config.qk_block_n * depth_dim
        var packed_o_size = Self._config.o_block_n * Self._config.qk_block_n
        var packed_size = max(packed_qk_size, packed_o_size)

        var num_blocks_m = ceildiv(max_seq_len, Self._config.block_m)
        var num_blocks_n = ceildiv(depth_dim, Self._config.o_block_n)
        var work_count = num_batches * num_heads * num_blocks_m * num_blocks_n

        var num_threads = min(work_count, parallelism_level(ctx))

        def task_func(
            task_id: Int,
        ) {
            var num_threads,
            var work_count,
            var num_blocks_n,
            var num_blocks_m,
            var packed_size,
            var kv_group_count,
            var depth_dim,
            var max_seq_len,
            var num_heads,
            var sink_weights,
            imm,
        }:
            var qk_block_ptr = unsafe_stack_allocation[
                Self._config.block_m * Self._config.qk_block_n,
                Self.dtype,
                alignment=align_of[SIMD[Self.dtype, Self.simd_width]](),
            ]()
            var o_block_ptr = unsafe_stack_allocation[
                Self._config.block_m * Self._config.o_block_n,
                Self.dtype,
                alignment=align_of[SIMD[Self.dtype, Self.simd_width]](),
            ]()
            var max_vals_storage = Array[
                Scalar[Self.dtype], Self._config.block_m
            ](uninitialized=True)
            var max_vals = TileTensor(
                Span(max_vals_storage), row_major[Self._config.block_m]()
            )
            var sum_vals_storage = Array[
                Scalar[Self.dtype], Self._config.block_m
            ](uninitialized=True)
            var sum_vals = TileTensor(
                Span(sum_vals_storage), row_major[Self._config.block_m]()
            )

            var packed_alloc = Optional[ManagedAllocation[Scalar[Self.dtype]]]()
            var packed_ptr = type_of(
                packed_alloc.value().unsafe_ptr()
            ).unsafe_dangling()

            if max_seq_len != 1:
                packed_alloc = alloc(
                    AllocLayout[Scalar[Self.dtype]](
                        count=packed_size,
                        alignment=align_of[SIMD[Self.dtype, Self.simd_width]](),
                    )
                ).into_managed()
                packed_ptr = packed_alloc.unsafe_value().unsafe_ptr()

            var q_seq_stride = num_heads * depth_dim

            var block_range = partition_work(
                task_id, num_threads, work_count, 1
            )

            for i in range(block_range[0], block_range[0] + block_range[1]):
                var n = (i % num_blocks_n) * Self._config.o_block_n
                var j = i // num_blocks_n
                var m = (j % num_blocks_m) * Self._config.block_m
                var batch_head = j // num_blocks_m
                var head = batch_head % num_heads
                var batch = batch_head // num_heads
                var kv_head = head // kv_group_count
                var kv_cache_len = Self.kv_cache_length_fn(batch)
                var seq_len = Self.q_length_fn(batch)
                var kv_seq_len = kv_cache_len + Self.kv_length_fn(batch)

                # Exit early if there's no more work to do for this batch.
                if m >= seq_len:
                    continue

                @__parameter
                @__copy_capture(batch, batch_head, kv_head, head)
                @always_inline
                def get_nd_index[
                    is_kv: Bool = False
                ](x: Int, y: Int) -> IndexList[Self.rank]:
                    comptime if Self.rank == 4:
                        return IndexList[Self.rank](
                            batch, x, kv_head if is_kv else head, y
                        )
                    else:
                        return IndexList[Self.rank](batch, x, y)

                @__parameter
                @__copy_capture(batch, head)
                @always_inline
                def get_mask_nd_index(
                    x: Int, y: Int
                ) -> IndexList[Self.mask_rank]:
                    comptime if Self.mask_rank == 4:
                        return IndexList[Self.mask_rank](batch, head, x, y)
                    elif Self.mask_rank == 3:
                        return IndexList[Self.mask_rank](batch, x, y)
                    elif Self.mask_rank == 2:
                        return IndexList[Self.mask_rank](x, y)
                    else:
                        return IndexList[Self.mask_rank]()

                var count_m = min(Self._config.block_m, seq_len - m)
                var count_n = min(Self._config.o_block_n, depth_dim - n)

                var o_ptr = Self.output_ptr_fn(get_nd_index(m, n))
                var q_ptr = Self.input_q_ptr_fn(get_nd_index(m, 0))

                _ = max_vals.fill(Scalar[Self.dtype].MIN)
                _ = sum_vals.fill(0)

                for kv_seq_idx in range(0, kv_seq_len, Self._config.qk_block_n):
                    var kv_seq_cnt = min(
                        kv_seq_len - kv_seq_idx, Self._config.qk_block_n
                    )

                    @__parameter
                    @always_inline
                    def input_k_2d_fn[
                        _simd_width: Int
                    ](_n: Int, _k: Int) -> SIMD[Self.dtype, _simd_width]:
                        return Self.input_k_fn[_simd_width, Self.rank](
                            get_nd_index[is_kv=True](_n + kv_seq_idx, _k)
                        )

                    try:
                        Self._matmul._matmul[
                            input_k_2d_fn,
                            transpose_b=True,
                            static_k=Self._depth_static_dim,
                        ](
                            count_m,
                            kv_seq_cnt,
                            depth_dim,
                            q_ptr,
                            q_seq_stride,
                            packed_ptr,
                            qk_block_ptr,
                            Self._config.qk_block_n,
                        )
                    except e:
                        # This won't trigger in practice, but we want to keep
                        # this function non-raising.
                        abort(String(e))

                    @__parameter
                    @always_inline
                    def mask_2d_fn[
                        _simd_width: SIMDLength
                    ](
                        _m: Int,
                        _n: Int,
                        score_vec: SIMD[Self.dtype, _simd_width],
                    ) -> SIMD[Self.dtype, _simd_width]:
                        return Self.mask_fn[_simd_width, Self.mask_rank](
                            get_mask_nd_index(_m + m, _n + kv_seq_idx),
                            score_vec,
                            kv_cache_len,
                        )

                    var sink_weight: Optional[Scalar[Self.dtype]] = None
                    if sink_weights:
                        sink_weight = sink_weights.value()[head][0]

                    Self._online_softmax[mask_2d_fn](
                        qk_block_ptr,
                        o_block_ptr,
                        max_vals.ptr,
                        sum_vals.ptr,
                        count_m,
                        count_n,
                        kv_seq_cnt,
                        scale,
                        sink_weight,
                    )

                    @__parameter
                    @always_inline
                    def input_v_2d_fn[
                        _simd_width: Int
                    ](_n: Int, _k: Int) -> SIMD[Self.dtype, _simd_width]:
                        return Self.input_v_fn[_simd_width, Self.rank](
                            get_nd_index[is_kv=True](_k + kv_seq_idx, n + _n)
                        )

                    try:
                        Self._matmul._matmul[input_v_2d_fn](
                            count_m,
                            count_n,
                            kv_seq_cnt,
                            qk_block_ptr,
                            Self._config.qk_block_n,
                            packed_ptr,
                            o_block_ptr,
                            Self._config.o_block_n,
                            accumulate=(kv_seq_idx > 0),
                        )
                    except e:
                        abort(String(e))
                    _ = kv_seq_idx

                _ = m
                _ = n
                var oz_ptr = o_block_ptr

                for m in range(count_m):
                    var reciprocal = 1 / sum_vals[m][0]

                    @always_inline
                    def do_final[
                        _simd_width: Int
                    ](idx: Int) {oz_ptr, o_ptr, reciprocal, mut}:
                        var v = oz_ptr.load[width=_simd_width](idx)
                        o_ptr.store(idx, v * reciprocal)

                    vectorize[Self.simd_width, unroll_factor=4](
                        count_n, do_final
                    )

                    o_ptr += q_seq_stride
                    oz_ptr += Self._config.o_block_n

            # NOTE: passing `dealloc[Scalar[Self.dtype]]` directly crashes the
            # when the dtype is parametric; wrap it in a local function as a workaround.
            def _dealloc_packed(
                var packed: ManagedAllocation[Scalar[Self.dtype]],
            ):
                dealloc(packed^)

            packed_alloc^.deinit_with(_dealloc_packed)

        sync_parallelize(task_func, num_threads, ctx)


@always_inline
def _flash_attention[
    dtype: DType,
    rank: Int,
    mask_rank: Int,
    q_origin: ImmOrigin,
    output_origin: Origin[mut=True],
    //,
    input_k_fn: def[simd_width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, simd_width],
    input_v_fn: def[simd_width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, simd_width],
    input_mask_fn: def[simd_width: Int, mask_rank: Int](
        IndexList[mask_rank]
    ) capturing -> SIMD[dtype, simd_width],
](
    q: LayoutTensor[dtype, _, q_origin, address_space=.GENERIC, ...],
    k_shape: IndexList[rank],
    v_shape: IndexList[rank],
    mask_shape: IndexList[mask_rank],
    output: LayoutTensor[dtype, _, output_origin, address_space=.GENERIC, ...],
    scale: Float32,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
    ctx: Optional[DeviceContext] = None,
):
    var num_batches = output.dim[0]()
    var max_seq_len = output.dim[1]()
    var num_heads = output.dim[rank - 2]() if rank == 4 else 1
    var depth_dim = output.dim[rank - 1]()
    var kv_cache_len = v_shape[1] - max_seq_len
    var num_kv_heads = k_shape[rank - 2] if rank == 4 else 1

    @always_inline
    @__parameter
    def input_q_ptr_fn(
        coords: IndexList[rank],
    ) -> UnsafePointer[Scalar[dtype], q_origin]:
        var idx = q._offset(coords)
        return q.ptr + idx

    @always_inline
    @__parameter
    def output_ptr_fn(
        coords: IndexList[rank],
    ) -> UnsafePointer[Scalar[dtype], output_origin]:
        var idx = output._offset(coords)
        return output.ptr + idx

    @always_inline
    @__parameter
    def mask_fn[
        simd_width: SIMDLength, rank: Int
    ](
        idx: IndexList[rank],
        score_vec: SIMD[dtype, simd_width],
        kv_cache_len: Int,
    ) -> SIMD[dtype, simd_width]:
        return score_vec + input_mask_fn[simd_width, rank](idx)

    @always_inline
    @__copy_capture(kv_cache_len)
    @__parameter
    def kv_cache_length_fn(batch: Int) -> Int:
        return kv_cache_len

    @always_inline
    @__copy_capture(max_seq_len)
    @__parameter
    def q_length_fn(batch: Int) -> Int:
        return max_seq_len

    _FlashAttention[
        input_q_ptr_fn,
        input_k_fn,
        input_v_fn,
        mask_fn,
        mask_rank,
        output_ptr_fn,
        q_length_fn,
        # Use the `q_length_fn` also for the KV length for now.
        # Note that this is only correct for self attention and is broken for
        # cross attention, which has different KV lengths.
        q_length_fn,
        kv_cache_length_fn,
        rebind[IndexList[rank]](
            to_index_list[output.rank](output.layout.shape)
        ),
    ].run(
        num_batches,
        num_heads,
        depth_dim,
        num_kv_heads,
        max_seq_len,
        scale,
        sink_weights,
        ctx,
    )


def flash_attention[
    dtype: DType,
    rank: Int,
    mask_rank: Int,
    q_origin: ImmOrigin,
    output_origin: Origin[mut=True],
    //,
    input_k_fn: def[simd_width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, simd_width],
    input_v_fn: def[simd_width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, simd_width],
    input_mask_fn: def[simd_width: Int, mask_rank: Int](
        IndexList[mask_rank]
    ) capturing -> SIMD[dtype, simd_width],
](
    q: LayoutTensor[dtype, _, q_origin, address_space=.GENERIC, ...],
    k_shape: IndexList[rank],
    v_shape: IndexList[rank],
    mask_shape: IndexList[mask_rank],
    output: LayoutTensor[dtype, _, output_origin, address_space=.GENERIC, ...],
    scale: Float32,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
    ctx: Optional[DeviceContext] = None,
):
    """Computes scaled dot-product flash attention on CPU for the given query, key, value, and mask accessors.

    Parameters:
        dtype: The element type of the query, key, value, and output
            tensors (inferred).
        rank: The number of dimensions in the query, key, value, and output
            tensors, either 3 or 4 (inferred).
        mask_rank: The number of dimensions in the attention mask tensor
            (inferred).
        q_origin: The memory origin of the read-only query tensor
            (inferred).
        output_origin: The memory origin of the writable output tensor
            (inferred).
        input_k_fn: Compile-time function loading a `SIMD` vector of key
            elements at a given `IndexList` index.
        input_v_fn: Compile-time function loading a `SIMD` vector of value
            elements at a given `IndexList` index.
        input_mask_fn: Compile-time function loading a `SIMD` vector of
            additive mask values at a given `IndexList` index.

    Args:
        q: Query tensor in BSHD or BSD layout.
        k_shape: Shape of the key tensor.
        v_shape: Shape of the value tensor.
        mask_shape: Shape of the attention mask tensor.
        output: Output tensor to write the attention results into.
        scale: Scaling factor applied to the query-key dot products.
        sink_weights: Optional per-head attention sink weights.
        ctx: Optional device context for controlling parallelism.
    """
    _flash_attention[input_k_fn, input_v_fn, input_mask_fn](
        q,
        k_shape,
        v_shape,
        mask_shape,
        output,
        scale,
        sink_weights,
        ctx,
    )


def flash_attention_split_kv[
    dtype: DType,
    rank: Int,
    mask_rank: Int,
    //,
    input_k_fn: def[simd_width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, simd_width],
    input_v_fn: def[simd_width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, simd_width],
    input_k_cache_fn: def[simd_width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, simd_width],
    input_v_cache_fn: def[simd_width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, simd_width],
    input_mask_fn: def[simd_width: Int, mask_rank: Int](
        IndexList[mask_rank]
    ) capturing -> SIMD[dtype, simd_width],
](
    q: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    k_shape: IndexList[rank],
    v_shape: IndexList[rank],
    # {k,v}_cache_shape are rank + 1 because reshape in MO IR prevents fusion.
    k_cache_shape: IndexList[rank + 1],
    v_cache_shape: IndexList[rank + 1],
    mask_shape: IndexList[mask_rank],
    output: LayoutTensor[mut=True, dtype, address_space=.GENERIC, ...],
    scale: Float32,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Variant of flash attention that takes the previous KV cache
    `input_{k,v}_cache_fn` and the current KV tensors `input_k_fn` and
    `input_v_fn` as separate arguments.

    This works around the fact that fusion can't currently look through concat.
    So this kernel does an in-place concat fusion by changing the input lambdas
    `input_{k,v}_cache_fn_wrapper` to take previous sequence KV elements from
    the KV cache, and current KV elements from tensors `k` and `v`.

    Parameters:
        dtype: The element type of the query, key, value, and output
            tensors (inferred).
        rank: The number of dimensions in the query, key, value, and output
            tensors, either 3 or 4 (inferred).
        mask_rank: The number of dimensions in the attention mask tensor
            (inferred).
        input_k_fn: Compile-time function loading a `SIMD` vector of current
            key elements at a given `IndexList` index.
        input_v_fn: Compile-time function loading a `SIMD` vector of current
            value elements at a given `IndexList` index.
        input_k_cache_fn: Compile-time function loading a `SIMD` vector of
            cached key elements at a given `IndexList` index.
        input_v_cache_fn: Compile-time function loading a `SIMD` vector of
            cached value elements at a given `IndexList` index.
        input_mask_fn: Compile-time function loading a `SIMD` vector of
            additive mask values at a given `IndexList` index.

    Args:
        q: Query tensor in BSHD layout.
        k_shape: Shape of the current key tensor in BSHD layout.
        v_shape: Shape of the current value tensor in BSHD layout.
        k_cache_shape: Shape of the cached key tensor with one extra
            leading dimension.
        v_cache_shape: Shape of the cached value tensor with one extra
            leading dimension.
        mask_shape: Shape of the attention mask tensor.
        output: Output tensor to write the attention results into.
        scale: Scaling factor applied to the query-key dot products.
        ctx: Optional device context for controlling parallelism
            (defaults to `None`).
    """
    # This expects the following layouts:
    # q: BSHD
    # k (input_k_fn): BSHD
    # v (input_v_fn): BSHD
    # k_cache (input_k_cache_fn): 1BHS'D
    # v_cache (input_v_cache_fn): 1BHS'D
    comptime assert rank == 4

    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("q", q.runtime_layout.shape.value),
                    trace_arg("k", k_shape),
                    trace_arg("v", v_shape),
                    trace_arg("k_cache", k_cache_shape),
                    trace_arg("v_cache", v_cache_shape),
                    trace_arg("output", output.runtime_layout.shape.value),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=StaticString("cpu")](
        "flash_attention_split_kv",
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
    ):
        comptime kv_rank = rank + 1

        var kv_cache_len = v_cache_shape[3]

        @always_inline
        @__parameter
        def kv_index[rank: Int](idx: IndexList[rank]) -> IndexList[kv_rank]:
            # Index into the previous kv_cache by unsqueezing dim 0.
            return IndexList[kv_rank](0, idx[0], idx[2], idx[1], idx[3])

        @always_inline
        @__copy_capture(kv_cache_len)
        @__parameter
        def load_from_split_cache[
            curr_fn: def[simd_width: Int, rank: Int](
                IndexList[rank]
            ) capturing -> SIMD[dtype, simd_width],
            cache_fn: def[simd_width: Int, rank: Int](
                IndexList[rank]
            ) capturing -> SIMD[dtype, simd_width],
            rank: Int,
            simd_width: Int,
        ](idx: IndexList[rank]) -> SIMD[dtype, simd_width]:
            # Load directly from either `curr_fn` or `cache_fn` depending on the
            # sequence index.
            # Boundary condition handling is done by the caller since
            # the last dim `depth_dim` is contiguous.
            var seq_idx = idx[1]

            if seq_idx >= kv_cache_len:
                return curr_fn[simd_width, rank](
                    IndexList[rank](
                        idx[0], seq_idx - kv_cache_len, idx[2], idx[3]
                    )
                )

            return cache_fn[simd_width, kv_rank](kv_index(idx))

        @always_inline
        @__parameter
        def input_k_cache_fn_wrapper[
            simd_width: Int,
            rank: Int,
        ](idx: IndexList[rank]) -> SIMD[dtype, simd_width]:
            return load_from_split_cache[
                input_k_fn, input_k_cache_fn, rank, simd_width
            ](idx)

        @always_inline
        @__parameter
        def input_v_cache_fn_wrapper[
            simd_width: Int,
            rank: Int,
        ](idx: IndexList[rank]) -> SIMD[dtype, simd_width]:
            return load_from_split_cache[
                input_v_fn, input_v_cache_fn, rank, simd_width
            ](idx)

        var combined_k_shape = IndexList[rank](
            k_shape[0], k_shape[1] + k_cache_shape[3], k_shape[2], k_shape[3]
        )
        var combined_v_shape = IndexList[rank](
            v_shape[0], v_shape[1] + v_cache_shape[3], v_shape[2], v_shape[3]
        )
        _flash_attention[
            input_k_cache_fn_wrapper,
            input_v_cache_fn_wrapper,
            input_mask_fn,
        ](
            q,
            combined_k_shape,
            combined_v_shape,
            mask_shape,
            output,
            scale,
            ctx=ctx,
        )


@always_inline
def _flash_attention_kv_cache[
    dtype: DType,
    cache_t: KVCacheT,
    q_origin: ImmOrigin,
    output_origin: Origin[mut=True],
    //,
    mask_fn: def[simd_width: SIMDLength, mask_rank: Int](
        idx: IndexList[mask_rank],
        score_vec: SIMD[dtype, simd_width],
        kv_cache_length: Int,
    ) capturing -> SIMD[dtype, simd_width],
    mask_rank: Int,
](
    q: LayoutTensor[dtype, _, q_origin, address_space=.GENERIC, ...],
    k: cache_t,
    v: cache_t,
    scale: Float32,
    output: LayoutTensor[dtype, _, output_origin, address_space=.GENERIC, ...],
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
):
    comptime kv_params = cache_t.kv_params

    var max_seq_len = q.dim[1]()
    var num_batches = q.dim[0]()
    comptime num_heads = Int(q.layout.shape[2])
    comptime head_size = cache_t.kv_params.head_size
    comptime output_shape = IndexList[4](
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, head_size
    )

    @always_inline
    @__parameter
    def input_q_ptr_fn(
        coords: IndexList[4],
    ) -> UnsafePointer[Scalar[dtype], q_origin]:
        var idx = q._offset(coords)
        return q.ptr + idx

    @always_inline
    @__parameter
    def output_ptr_fn(
        coords: IndexList[4],
    ) -> UnsafePointer[Scalar[dtype], output_origin]:
        var idx = output._offset(coords)
        return output.ptr + idx

    @always_inline
    @__copy_capture(max_seq_len)
    @__parameter
    def q_length_fn(batch: Int) -> Int:
        return max_seq_len

    return _flash_attention_kv_cache[
        input_q_ptr_fn,
        output_ptr_fn,
        q_length_fn,
        # NOTE: kv_length_fn = q_length_fn is only correct for self attention.
        kv_length_fn=q_length_fn,
        mask_fn=mask_fn,
        mask_rank=mask_rank,
        output_shape=output_shape,
    ](k, v, num_batches, num_heads, max_seq_len, scale, sink_weights)


@always_inline
def _flash_attention_kv_cache[
    dtype: DType,
    cache_t: KVCacheT,
    q_origin: ImmOrigin,
    output_origin: Origin[mut=True],
    //,
    input_q_ptr_fn: def(IndexList[4]) capturing -> UnsafePointer[
        Scalar[dtype], q_origin
    ],
    output_ptr_fn: def(IndexList[4]) capturing -> UnsafePointer[
        Scalar[dtype], output_origin
    ],
    q_length_fn: def(batch: Int) capturing -> Int,
    kv_length_fn: def(batch: Int) capturing -> Int,
    mask_fn: def[simd_width: SIMDLength, mask_rank: Int](
        idx: IndexList[mask_rank],
        score_vec: SIMD[dtype, simd_width],
        kv_cache_length: Int,
    ) capturing -> SIMD[dtype, simd_width],
    mask_rank: Int,
    output_shape: IndexList[4],
](
    k: cache_t,
    v: cache_t,
    num_batches: Int,
    num_heads: Int,
    max_seq_len: Int,
    scale: Float32,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
):
    comptime num_kv_heads = cache_t.kv_params.num_heads
    comptime depth_dim = cache_t.kv_params.head_size
    comptime cache_type = cache_t.dtype

    comptime assert cache_type == dtype, (
        "Expected cache dtype ("
        + String(cache_type)
        + ") to match input dtype ("
        + String(dtype)
        + ")"
    )

    @__parameter
    def input_k_fn[
        width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[dtype, width]:
        # Unwrap BSHD->BHSD indices.
        return rebind[SIMD[dtype, width]](
            k.load[width=width](idx[0], idx[2], idx[1], idx[3])
        )

    @__parameter
    def input_v_fn[
        width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[dtype, width]:
        # Unwrap BSHD->BHSD indices.
        return rebind[SIMD[dtype, width]](
            v.load[width=width](idx[0], idx[2], idx[1], idx[3])
        )

    @always_inline
    @__parameter
    def kv_cache_length_fn(batch: Int) -> Int:
        return k.cache_length(batch)

    _FlashAttention[
        input_q_ptr_fn,
        input_k_fn,
        input_v_fn,
        mask_fn,
        mask_rank,
        output_ptr_fn,
        q_length_fn,
        kv_length_fn,
        kv_cache_length_fn,
        output_shape,
    ].run(
        num_batches,
        num_heads,
        depth_dim,
        num_kv_heads,
        max_seq_len,
        scale,
        sink_weights,
    )


# See the note on the ragged overload below: `k`/`v` alias the same `blocks`
# buffer and are read-only here, so disable the nested-origin exclusivity check.
@__unsafe_nested_origins_read_only
def flash_attention_kv_cache[
    dtype: DType,
    cache_t: KVCacheT,
    q_origin: ImmOrigin,
    output_origin: Origin[mut=True],
    //,
](
    q: LayoutTensor[dtype, _, q_origin, address_space=.GENERIC, ...],
    k: cache_t,
    v: cache_t,
    mask: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
    scale: Float32,
    output: LayoutTensor[dtype, _, output_origin, address_space=.GENERIC, ...],
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
):
    """Computes flash attention on CPU using a KV cache with an additive LayoutTensor mask.

    Args:
        q: Query tensor in BSHD layout.
        k: Key cache.
        v: Value cache.
        mask: Additive attention mask tensor.
        scale: Scaling factor applied to the query-key dot products.
        output: Output tensor to write the attention results into.
        sink_weights: Optional per-head attention sink weights."""

    @always_inline
    @__parameter
    def mask_fn[
        simd_width: SIMDLength, rank: Int
    ](
        idx: IndexList[rank],
        score_vec: SIMD[dtype, simd_width],
        kv_cache_len: Int,
    ) -> SIMD[dtype, simd_width]:
        return score_vec + mask.load[width=simd_width](idx)

    _flash_attention_kv_cache[mask_fn, mask.rank](
        q, k, v, scale, output, sink_weights
    )


# See the note on the ragged overload below: `k`/`v` alias the same `blocks`
# buffer and are read-only here, so disable the nested-origin exclusivity check.
@__unsafe_nested_origins_read_only
def flash_attention_kv_cache[
    dtype: DType,
    cache_t: KVCacheT,
    mask_t: MHAMask,
    q_origin: ImmOrigin,
    output_origin: Origin[mut=True],
    //,
](
    q: LayoutTensor[dtype, _, q_origin, address_space=.GENERIC, ...],
    k: cache_t,
    v: cache_t,
    mask: mask_t,
    scale: Float32,
    output: LayoutTensor[dtype, _, output_origin, address_space=.GENERIC, ...],
    sink_weights: OptionalReg[
        LayoutTensor[
            mut=False, dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
    ] = None,
):
    """Computes flash attention on CPU using a KV cache with an MHAMask-based mask.

    Args:
        q: Query tensor in BSHD layout.
        k: Key cache.
        v: Value cache.
        mask: MHAMask applied to the attention scores.
        scale: Scaling factor applied to the query-key dot products.
        output: Output tensor to write the attention results into.
        sink_weights: Optional per-head attention sink weights."""

    @always_inline
    @__parameter
    def mask_fn[
        simd_width: SIMDLength,
        rank: Int,
    ](
        idx: IndexList[rank],
        score_vec: SIMD[dtype, simd_width],
        kv_cache_len: Int,
    ) -> SIMD[dtype, simd_width]:
        # Shift the mask index from local->global space.
        return mask.mask(
            Index(idx[0], idx[1], idx[2] + kv_cache_len, idx[3]), score_vec
        )

    _flash_attention_kv_cache[mask_fn, 4](q, k, v, scale, output, sink_weights)


# `k` and `v` are disjoint views into the same `blocks` buffer, so they share the
# collection's mutable `blocks_origin`. Attention only READS them, but the
# exclusivity checker can't prove that and rejects passing both. Disabling the
# nested-origin exclusivity check is safe here (read-only) and lets direct
# callers pass `get_key_cache()`/`get_value_cache()` without a copy-capture shim.
@__unsafe_nested_origins_read_only
def flash_attention_kv_cache[
    dtype: DType,
    cache_t: KVCacheT,
    mask_t: MHAMask,
    q_origin: ImmOrigin,
    output_origin: Origin[mut=True],
    //,
](
    q: LayoutTensor[dtype, _, q_origin, address_space=.GENERIC, ...],
    q_input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    kv_input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    k: cache_t,
    v: cache_t,
    mask: mask_t,
    scale: Float32,
    output: LayoutTensor[dtype, _, output_origin, address_space=.GENERIC, ...],
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
):
    """Computes flash attention on CPU for ragged tensors using a KV cache with an `MHAMask`-based mask.

    Parameters:
        dtype: The element type of the query, key, value, and output
            tensors (inferred).
        cache_t: The KV cache type storing key and value states (inferred).
        mask_t: The `MHAMask` type applied to the attention scores
            (inferred).
        q_origin: The memory origin of the read-only query tensor
            (inferred).
        output_origin: The memory origin of the writable output tensor
            (inferred).

    Args:
        q: Flattened query tensor indexed by `(row_offset, head, depth)`.
        q_input_row_offsets: Per-batch start offsets into the flattened
            query tensor; batch `b` spans rows `[offsets[b], offsets[b + 1])`.
        kv_input_row_offsets: Per-batch start offsets into the flattened KV
            tensors; batch `b` spans rows `[offsets[b], offsets[b + 1])`.
        k: Key cache storing per-head key states.
        v: Value cache storing per-head value states.
        mask: `MHAMask` applied additively to the query-key attention
            scores.
        scale: Scaling factor applied to the query-key dot products.
        output: Output tensor to write the attention results into.
        sink_weights: Optional per-head attention sink weights.
    """

    @always_inline
    @__parameter
    def mask_fn[
        simd_width: SIMDLength,
        rank: Int,
    ](
        idx: IndexList[rank],
        score_vec: SIMD[dtype, simd_width],
        kv_cache_len: Int,
    ) -> SIMD[dtype, simd_width]:
        # Shift the mask index from local->global space.
        return mask.mask(
            Index(idx[0], idx[1], idx[2] + kv_cache_len, idx[3]), score_vec
        )

    @always_inline
    @__parameter
    def q_length_fn(batch: Int) -> Int:
        return Int(q_input_row_offsets[batch + 1] - q_input_row_offsets[batch])

    @always_inline
    @__parameter
    def kv_length_fn(batch: Int) -> Int:
        return Int(
            kv_input_row_offsets[batch + 1] - kv_input_row_offsets[batch]
        )

    @always_inline
    @__parameter
    def input_q_ptr_fn(
        idx: IndexList[4],
    ) -> UnsafePointer[Scalar[dtype], q_origin]:
        var bs = idx[0]
        var tok_idx = idx[1]
        var q_start = Int(q_input_row_offsets[bs]) + tok_idx
        var flat_idx = IndexList[3](q_start, idx[2], idx[3])
        var out_idx = q._offset(flat_idx)
        return q.ptr + out_idx

    @always_inline
    @__parameter
    def output_ptr_fn(
        idx: IndexList[4],
    ) -> UnsafePointer[Scalar[dtype], output_origin]:
        var bs = idx[0]
        var tok_idx = idx[1]
        var q_start = Int(q_input_row_offsets[bs]) + tok_idx
        var flat_idx = IndexList[3](q_start, idx[2], idx[3])
        var out_idx = output._offset(flat_idx)
        return output.ptr + out_idx

    comptime mask_rank = 4
    var num_batches = q_input_row_offsets.dim[0]() - 1
    var max_seq_len = k.max_prompt_length()
    comptime num_heads = Int(q.layout.shape[q.rank - 2])
    comptime head_size = cache_t.kv_params.head_size
    comptime output_shape = IndexList[4](
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, head_size
    )

    _flash_attention_kv_cache[
        input_q_ptr_fn,
        output_ptr_fn,
        q_length_fn,
        kv_length_fn,
        mask_fn,
        mask_rank,
        output_shape,
    ](k, v, num_batches, num_heads, Int(max_seq_len), scale, sink_weights)
