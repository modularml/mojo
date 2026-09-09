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
"""Grouped W4A8 matmul on SM100: E4M3 activations against E2M1 weights.

Both operands are block-scaled by E8M0 factors over 32-element groups, which
is what lets one `kind::mxf8f6f4` instruction consume them without
dequantizing the weights first. The weights stay nibble-packed in global
memory -- two E2M1 values per byte, so `K/2` bytes per row -- and the FP4 TMA
copy pads them into shared memory so a K extent spans one byte per element
(the values stay nibble-packed; see `PACKED_FP4_ALIGN16B`).

The reference is computed on the host in FP32 rather than against a vendor
BLAS call, since no vendor path takes this operand pair. Operand values are
drawn so that reference and kernel agree EXACTLY before the output store:
every product is a multiple of 0.25 and the largest partial sum stays far
inside FP32's 24-bit mantissa, so no reordering of the accumulation can
change the result. That leaves the BF16 store as the only source of error,
which bounds the comparison at one BF16 ulp -- a derived tolerance, not a
tuned one.

Exactness holds while `K <= 10922`: the largest achievable partial sum is
`384 * K`, and FP32 represents every multiple of 0.25 up to 2**24, i.e. while
`1536 * K <= 2**24`. The shapes below sit 3x or more inside that, but a larger
K would silently make the reference wrong rather than merely imprecise.
"""

from std.math import ceildiv
from std.random import random_ui64, seed

from max.gpu.host import DeviceContext
from std.builtin.simd import _convert_f32_to_float8_ue8m0
from layout import Coord, Idx, TileTensor, row_major
from std.testing import assert_almost_equal, assert_equal

from linalg.fp4_utils import (
    E2M1_TO_FLOAT32,
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
    W4A8_A_DTYPE,
    W4A8_B_DTYPE,
    get_scale_factor,
    set_scale_factor,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_block_scaled_sm100_dispatch,
)

comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE

# Exact in E4M3 and small enough that every partial sum stays exact in FP32.
comptime A_VALUES = SIMD[.float32, 8](0.0, 0.5, 1.0, 2.0, -0.5, -1.0, -2.0, 1.5)

# One BF16 ulp. The kernel accumulates in FP32 and the host reference is exact,
# so rounding the result to BF16 on the store is the whole error budget.
comptime BF16_RTOL = 1.0 / 256.0


def _e8m0_pow2(bits: Int) -> Scalar[MXFP8_SF_DTYPE]:
    """An E8M0 scale of 2**bits, exact for the small exponents used here."""
    return _convert_f32_to_float8_ue8m0[MXFP8_SF_DTYPE](Float32(1 << bits))


def _test_w4a8_grouped[
    N: Int,
    K: Int,
    num_experts: Int,
    c_type: DType = .bfloat16,
](
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    ctx: DeviceContext,
) raises:
    seed(1234)
    comptime a_type = W4A8_A_DTYPE
    comptime b_type = W4A8_B_DTYPE
    comptime K_PACKED = K // 2
    comptime k_groups = ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
    comptime n_groups = ceildiv(N, SF_MN_GROUP_SIZE)
    comptime b_elems = num_experts * N * K_PACKED
    comptime b_scales_expert_stride = (
        n_groups * k_groups * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    )
    comptime b_scale_elems = num_experts * b_scales_expert_stride

    var num_active_experts = len(expert_ids)
    var M = 0
    for i in range(len(num_tokens_by_expert)):
        M += num_tokens_by_expert[i]

    print(
        t"[w4a8] M={M} N={N} K={K} experts={num_experts}"
        t" active={num_active_experts}"
    )

    # ---- Host allocations ----------------------------------------------
    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](M * K)
    var a_host = TileTensor(a_host_ptr, row_major(Coord(M, Idx[K])))
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_elems)
    var b_host = TileTensor(
        b_host_ptr, row_major(Coord(Idx[num_experts], Idx[N], Idx[K_PACKED]))
    )
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](M * N)
    var c_host = TileTensor(c_host_ptr, row_major(Coord(M, Idx[N])))

    var a_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        num_active_experts + 1
    )
    var a_scale_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        num_active_experts
    )
    var expert_ids_host = ctx.enqueue_create_host_buffer[.int32](
        num_active_experts
    )
    var expert_scales_host = ctx.enqueue_create_host_buffer[.float32](
        num_experts
    )

    # Powers of two so the epilogue multiply is exact too.
    for e in range(num_experts):
        expert_scales_host[e] = Float32(1 << (e % 2))

    var a_scale_rows = 0
    a_offsets_host[0] = 0
    for i in range(num_active_experts):
        a_scale_offsets_host[i] = UInt32(
            a_scale_rows - Int(a_offsets_host[i] // UInt32(SF_MN_GROUP_SIZE))
        )
        a_offsets_host[i + 1] = a_offsets_host[i] + UInt32(
            num_tokens_by_expert[i]
        )
        a_scale_rows += ceildiv(num_tokens_by_expert[i], SF_MN_GROUP_SIZE)
        expert_ids_host[i] = Int32(expert_ids[i])

    var a_scales_shape = row_major(
        Coord(
            Int(max(a_scale_rows, 1)),
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[MXFP8_SF_DTYPE](
        a_scales_shape.product()
    )
    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[MXFP8_SF_DTYPE](
        b_scale_elems
    )
    var b_scales_host = TileTensor(
        b_scales_host_ptr,
        row_major(
            Coord(
                Idx[num_experts],
                Idx[n_groups],
                Idx[k_groups],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            )
        ),
    )

    # ---- Operand fill ---------------------------------------------------
    for m in range(M):
        for k in range(K):
            a_host[m, k] = A_VALUES[Int(random_ui64(0, 7))].cast[a_type]()

    # Two E2M1 codes per byte: element 2i in the low nibble, 2i+1 in the high.
    for e in range(num_experts):
        for n in range(N):
            for kb in range(K_PACKED):
                b_host[e, n, kb] = UInt8(random_ui64(0, 255))

    for i in range(a_scales_host.num_elements()):
        a_scales_host._storage[unsafe_offset=i] = Scalar[MXFP8_SF_DTYPE](0.0)
    for i in range(b_scales_host.num_elements()):
        b_scales_host._storage[unsafe_offset=i] = Scalar[MXFP8_SF_DTYPE](0.0)

    # Scale rows past an expert's token count are padding the kernel still
    # reads; leaving them nonzero corrupts neighbouring tiles.
    for i in range(num_active_experts):
        var start = Int(a_offsets_host[i])
        var local_m = num_tokens_by_expert[i]
        var scale_row = (
            start // SF_MN_GROUP_SIZE + Int(a_scale_offsets_host[i])
        ) * SF_MN_GROUP_SIZE
        for row in range(scale_row, scale_row + local_m):
            for col in range(0, K, SF_VECTOR_SIZE):
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    a_scales_host,
                    row,
                    col,
                    _e8m0_pow2(Int(random_ui64(0, 2))),
                )

    for e in range(num_experts):
        var expert_slice = TileTensor(
            b_scales_host_ptr.unsafe_ptr().unsafe_offset(
                e * b_scales_expert_stride
            ),
            row_major(
                Coord(
                    Idx[n_groups],
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        )
        for n in range(N):
            for col in range(0, K, SF_VECTOR_SIZE):
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    expert_slice,
                    n,
                    col,
                    _e8m0_pow2(Int(random_ui64(0, 2))),
                )

    # ---- Device copies --------------------------------------------------
    var a_dev = ctx.enqueue_create_buffer[a_type](max(M * K, 1))
    var b_dev = ctx.enqueue_create_buffer[b_type](b_elems)
    var c_dev = ctx.enqueue_create_buffer[c_type](max(M * N, 1))
    var a_scales_dev = ctx.enqueue_create_buffer[MXFP8_SF_DTYPE](
        a_scales_shape.product()
    )
    var b_scales_dev = ctx.enqueue_create_buffer[MXFP8_SF_DTYPE](b_scale_elems)
    var a_offsets_dev = ctx.enqueue_create_buffer[.uint32](
        num_active_experts + 1
    )
    var a_scale_offsets_dev = ctx.enqueue_create_buffer[.uint32](
        max(num_active_experts, 1)
    )
    var expert_ids_dev = ctx.enqueue_create_buffer[.int32](
        max(num_active_experts, 1)
    )
    var expert_scales_dev = ctx.enqueue_create_buffer[.float32](num_experts)

    ctx.enqueue_copy(a_dev, a_host_ptr)
    ctx.enqueue_copy(b_dev, b_host_ptr)
    ctx.enqueue_copy(a_scales_dev, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_dev, b_scales_host_ptr)
    ctx.enqueue_copy(a_offsets_dev, a_offsets_host)
    if num_active_experts > 0:
        ctx.enqueue_copy(a_scale_offsets_dev, a_scale_offsets_host)
        ctx.enqueue_copy(expert_ids_dev, expert_ids_host)
    ctx.enqueue_copy(expert_scales_dev, expert_scales_host)
    ctx.enqueue_memset(c_dev, Scalar[c_type](0))

    grouped_matmul_block_scaled_sm100_dispatch[transpose_b=True, target="gpu"](
        TileTensor(c_dev, row_major(Coord(M, Idx[N]))).as_unsafe_any_origin(),
        TileTensor(a_dev, row_major(Coord(M, Idx[K]))).as_unsafe_any_origin(),
        TileTensor(
            b_dev, row_major(Coord(Idx[num_experts], Idx[N], Idx[K_PACKED]))
        ).as_unsafe_any_origin(),
        TileTensor(a_scales_dev, a_scales_shape).as_unsafe_any_origin(),
        TileTensor(
            b_scales_dev,
            row_major(
                Coord(
                    Idx[num_experts],
                    Idx[n_groups],
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        ).as_unsafe_any_origin(),
        TileTensor(
            a_offsets_dev, row_major(Coord(Int(num_active_experts + 1)))
        ).as_unsafe_any_origin(),
        TileTensor(
            a_scale_offsets_dev, row_major(Coord(Int(num_active_experts)))
        ).as_unsafe_any_origin(),
        TileTensor(
            expert_ids_dev, row_major(Coord(Int(num_active_experts)))
        ).as_unsafe_any_origin(),
        TileTensor(
            expert_scales_dev, row_major(Coord(Idx[num_experts]))
        ).as_unsafe_any_origin(),
        num_active_experts,
        M,
        ctx,
    )

    ctx.enqueue_copy(c_host_ptr, c_dev)
    ctx.synchronize()
    print("  kernel returned; checking")

    # ---- Host reference -------------------------------------------------
    # Exhaustive checking is O(M*N*K) and these run as debug builds, so verify
    # a strided sample of the output instead. Every checked element is still
    # compared exactly; the sample bounds the cost, not the rigor.
    var n_stride = max(N * M // 4096, 1)
    for i in range(num_active_experts):
        var start = Int(a_offsets_host[i])
        var end = Int(a_offsets_host[i + 1])
        var expert_id = Int(expert_ids_host[i])
        var expert_scale = expert_scales_host[expert_id]
        var b_scale_slice = TileTensor(
            b_scales_host_ptr.unsafe_ptr().unsafe_offset(
                expert_id * b_scales_expert_stride
            ),
            row_major(
                Coord(
                    Idx[n_groups],
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        )
        var scale_row_base = (
            start // SF_MN_GROUP_SIZE + Int(a_scale_offsets_host[i])
        ) * SF_MN_GROUP_SIZE

        for m in range(start, end):
            var a_scale_row = scale_row_base + (m - start)
            for n in range(0, N, n_stride):
                var acc = Float32(0.0)
                for kb in range(0, K, SF_VECTOR_SIZE):
                    var sfa = get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        a_scales_host, a_scale_row, kb
                    ).cast[.float32]()
                    var sfb = get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        b_scale_slice, n, kb
                    ).cast[.float32]()
                    var block = Float32(0.0)
                    for k in range(kb, kb + SF_VECTOR_SIZE):
                        var av = a_host[m, k].cast[.float32]()
                        var packed = Int(b_host[expert_id, n, k // 2])
                        var nibble = (packed >> 4) if k % 2 else (packed & 0xF)
                        block += av * E2M1_TO_FLOAT32[nibble]
                    acc += block * sfa * sfb
                assert_almost_equal(
                    c_host[m, n].cast[.float32](),
                    acc * expert_scale,
                    atol=0.0,
                    rtol=BF16_RTOL,
                )

    # With no active experts the kernel returns early; the output must be
    # untouched rather than partially written.
    if num_active_experts == 0:
        for i in range(M * N):
            assert_equal(Float32(c_host_ptr[i]), Float32(0.0))

    print("=== W4A8 TEST PASSED ===")

    _ = a_dev^
    _ = b_dev^
    _ = c_dev^
    _ = a_scales_dev^
    _ = b_scales_dev^
    _ = a_offsets_dev^
    _ = a_scale_offsets_dev^
    _ = expert_ids_dev^
    _ = expert_scales_dev^


def main() raises:
    with DeviceContext() as ctx:
        # Single expert, one full M group.
        _test_w4a8_grouped[N=256, K=512, num_experts=1]([128], [0], ctx)

        # Ragged multi-expert, including a group shorter than the 128-row
        # scale-factor granule and a group whose tokens do not fill a tile.
        _test_w4a8_grouped[N=256, K=512, num_experts=4](
            [64, 200, 8, 129], [2, 0, 3, 1], ctx
        )

        # Decode-shaped: one token per expert exercises the mma_bn=8 regime.
        _test_w4a8_grouped[N=256, K=512, num_experts=4](
            [1, 1, 1, 1], [0, 1, 2, 3], ctx
        )

        # Small-prefill regime: the dispatch keys on avg_m per active expert,
        # and 8 < 32 <= 64 is the only band the shapes above skip.
        _test_w4a8_grouped[N=256, K=512, num_experts=4](
            [32, 32, 32, 32], [0, 1, 2, 3], ctx
        )

        # K3's routed-expert shapes.
        _test_w4a8_grouped[N=3072, K=3584, num_experts=2]([96, 96], [0, 1], ctx)
