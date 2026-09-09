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
"""Byte-exact correctness test for `grouped_matmul_swiglu_mxfp8_dispatch`.

Validates the fused MXFP8 in-tile epilogue (SwiGLU + per-block MXFP8
quant) byte-exact against the unfused reference chain:

    REF : grouped_matmul_mxfp8_dispatch[fuse_swiglu=False](...)
           -> BF16 scratch tensor (M_total, 2*H)
          fused_silu_mxfp8_interleaved_kernel(scratch, ...)
           -> packed MXFP8 (M_total, H) + 5D E8M0 SF tile

    TEST: grouped_matmul_swiglu_mxfp8_dispatch(A, W_perm, ..., sigma on B_scales)
           -> packed MXFP8 + 5D E8M0 SF tile

Test shapes follow a 128-expert MoE served at EP=8: 16 experts per
rank, hidden_size 6144, intermediate_size 3072, so N = 2 * 3072 =
6144 and K = 6144. The clamped swigluoai activation uses the
canonical HF config values alpha = 1.702, limit = 7.0.

`num_active_experts = len(num_tokens_by_expert)` is taken at runtime;
masked tail slots `[num_active_experts, num_experts)` are padded
`(0 tokens, -1 id)` so the buffer shapes stay fixed at `num_experts`.
B-side weights and scales are pre-built once in `_SharedB` and reused
across every test case.

Decode-sized cases (avg tokens/expert <= 8) route through the
dispatch's `(mma_bn=8, cta_group=1)` regime and, by default, the
in-place register epilogue; prefill cases take the cooperative
`(mma_bn=64 or 128, cta_group=2)` regimes. One case pins
`use_inplace=False` to cover the cooperative epilogue on the decode
config as well.
"""
from std.math import align_up, ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
from std.memory import alloc
from std.memory.unsafe import bitcast
from std.random import seed, rand

from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_mxfp8_dispatch,
    grouped_matmul_swiglu_mxfp8_dispatch,
)
from linalg.fp4_utils import (
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    get_scale_factor,
    set_scale_factor,
)
from shmem.ep_comm import fused_silu_mxfp8_interleaved_kernel


@always_inline
def e8m0(x: Float32) -> Float8_e8m0fnu:
    """Construct an E8M0 scalar from a Float32 via cast."""
    return x.cast[.float8_e8m0fnu]()


def _make_uniform(value: Int, count: Int) -> List[Int]:
    var r = List[Int]()
    for _ in range(count):
        r.append(value)
    return r^


def _make_range(count: Int) -> List[Int]:
    var r = List[Int]()
    for i in range(count):
        r.append(i)
    return r^


def _make_ragged(head: Int, tail_value: Int, tail_count: Int) -> List[Int]:
    var r = List[Int]()
    r.append(head)
    for _ in range(tail_count):
        r.append(tail_value)
    return r^


struct _SharedB(Movable):
    """Per-comptime-shape device buffers reused across all test cases:
    pre-permuted B weights, pre-permuted B scales, expert scales. The
    heavy host-side `rand(b_host)` (~600 MB at the EP=8 shape) and the
    per-expert N-axis sigma permutation run only once.
    """

    var b_perm: DeviceBuffer[.float8_e4m3fn]
    var b_scales_perm: DeviceBuffer[MXFP8_SF_DTYPE]
    var expert_scales: DeviceBuffer[.float32]

    def __init__(
        out self,
        var b_perm: DeviceBuffer[.float8_e4m3fn],
        var b_scales_perm: DeviceBuffer[MXFP8_SF_DTYPE],
        var expert_scales: DeviceBuffer[.float32],
    ):
        self.b_perm = b_perm^
        self.b_scales_perm = b_scales_perm^
        self.expert_scales = expert_scales^


def _build_shared_b[
    num_experts: Int,
    N: Int,
    K: Int,
](ctx: DeviceContext) raises -> _SharedB:
    """Build B-side weights and the permuted MXFP8 scale tile once."""
    seed(1234)
    comptime b_type = DType.float8_e4m3fn
    comptime scales_dtype = MXFP8_SF_DTYPE
    comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE
    comptime packed_K = K

    comptime assert N % 2 == 0, "N must be even for gate/up interleave."
    comptime H = N // 2

    comptime k_groups = ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
    comptime n_groups_b = ceildiv(N, SF_MN_GROUP_SIZE)

    var b_size = num_experts * N * packed_K

    var b_host_ptr = alloc[Scalar[b_type]](b_size)
    var b_perm_host_ptr = alloc[Scalar[b_type]](b_size)

    var b_scales_shape = row_major(
        Coord(
            Idx[num_experts],
            Idx[n_groups_b],
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var b_scales_total = b_scales_shape.product()
    var b_scales_host_ptr = alloc[Scalar[scales_dtype]](b_scales_total)
    var b_scales_perm_host_ptr = alloc[Scalar[scales_dtype]](b_scales_total)
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_shape)
    var b_scales_perm_host = TileTensor(b_scales_perm_host_ptr, b_scales_shape)

    var expert_scales_host_ptr = alloc[Float32](num_experts)
    for i in range(num_experts):
        expert_scales_host_ptr[i] = 1.0 + Float32(i + 1) / Float32(num_experts)

    # Deterministic fill on the raw byte stream; the kernel interprets
    # bytes per the E4M3 / E8M0 encodings respectively.
    rand(b_host_ptr.bitcast[UInt8](), b_size, min=0, max=255)
    # Small power-of-2 scale (2^-7) keeps matmul outputs inside the
    # +-448 E4M3 representable range so we don't saturate to NaN.
    var sf_small = e8m0(Float32(0.0078125))
    var sf_zero = e8m0(Float32(0.0))
    for i in range(b_scales_total):
        b_scales_host_ptr[i] = sf_small
    for i in range(b_scales_perm_host.num_elements()):
        b_scales_perm_host._storage[i] = sf_zero

    # sigma-permute W on the N axis: W_perm[e, 2i, :] = W[e, i, :],
    # W_perm[e, 2i+1, :] = W[e, H + i, :].
    for e in range(num_experts):
        for i in range(H):
            for kp in range(packed_K):
                b_perm_host_ptr[((e * N + 2 * i) * packed_K) + kp] = b_host_ptr[
                    ((e * N + i) * packed_K) + kp
                ]
                b_perm_host_ptr[
                    ((e * N + 2 * i + 1) * packed_K) + kp
                ] = b_host_ptr[((e * N + H + i) * packed_K) + kp]

    var b_expert_sf_size = (
        Int(b_scales_host.dim(1))
        * Int(b_scales_host.dim(2))
        * Int(b_scales_host.dim(3))
        * Int(b_scales_host.dim(4))
        * Int(b_scales_host.dim(5))
    )
    for e in range(num_experts):
        var src_view = TileTensor(
            b_scales_host_ptr + e * b_expert_sf_size,
            row_major(
                Coord(
                    Idx[n_groups_b],
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        )
        var dst_view = TileTensor(
            b_scales_perm_host_ptr + e * b_expert_sf_size,
            row_major(
                Coord(
                    Idx[n_groups_b],
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        )
        for i in range(H):
            for col in range(
                0, align_up(K, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
            ):
                var v_gate = get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    src_view, i, col
                )
                var v_up = get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    src_view, H + i, col
                )
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    dst_view, 2 * i, col, v_gate
                )
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    dst_view, 2 * i + 1, col, v_up
                )

    var b_perm_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_scales_perm_device = ctx.enqueue_create_buffer[scales_dtype](
        b_scales_total
    )
    var expert_scales_device = ctx.enqueue_create_buffer[.float32](num_experts)

    ctx.enqueue_copy(b_perm_device, b_perm_host_ptr)
    ctx.enqueue_copy(b_scales_perm_device, b_scales_perm_host_ptr)
    ctx.enqueue_copy(expert_scales_device, expert_scales_host_ptr)
    ctx.synchronize()

    b_host_ptr.free()
    b_perm_host_ptr.free()
    b_scales_host_ptr.free()
    b_scales_perm_host_ptr.free()
    expert_scales_host_ptr.free()

    return _SharedB(
        b_perm=b_perm_device^,
        b_scales_perm=b_scales_perm_device^,
        expert_scales=expert_scales_device^,
    )


def _test_swiglu_mxfp8_dispatch[
    num_experts: Int,
    N: Int,
    K: Int,
    match_bf16: Bool = True,
    clamp_activation: Bool = False,
    use_inplace: Bool = True,
](
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    shared: _SharedB,
    ctx: DeviceContext,
    alpha: Float32 = Float32(1.702),
    limit: Float32 = Float32(7.0),
    est_m_override: Int = -1,
) raises:
    """End-to-end byte-exact compare: fused MXFP8 dispatch vs chain.

    `num_active_experts = len(num_tokens_by_expert)`. Tail slots
    `[num_active_experts, num_experts)` are padded (0 tokens, -1 id).
    When `clamp_activation=True`, both REF and TEST paths apply the
    swigluoai clamped activation so the comparison remains byte-exact.
    `use_inplace` selects the fused decode epilogue flavor (register
    in-place vs cooperative); both must match the REF chain.
    """
    seed(1234)
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime fp8_dtype = DType.float8_e4m3fn
    comptime scales_dtype = MXFP8_SF_DTYPE
    comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE
    comptime transpose_b = True

    comptime assert N % 2 == 0, "N must be even for gate/up interleave."
    comptime H = N // 2
    # MXFP8 is one byte per element (no nibble packing).
    comptime packed_K = K
    comptime packed_H = H

    var num_active_experts = len(num_tokens_by_expert)

    var num_tokens = 0
    for i in range(num_active_experts):
        num_tokens += num_tokens_by_expert[i]
    var M = num_tokens
    # `estimated_total_m` is a tuning/buffer-sizing hint, not the true token
    # count. Default to the honest M; a test may override it (e.g. 0) to
    # exercise the floored-estimate EP-decode path (S*top_k//n_gpus == 0).
    var est_m = est_m_override if est_m_override >= 0 else M
    print(
        "  N=",
        N,
        " K=",
        K,
        " H=",
        H,
        " M=",
        M,
        " est_m=",
        est_m,
        " active=",
        num_active_experts,
        "/",
        num_experts,
        " match_bf16=",
        match_bf16,
        " clamp=",
        clamp_activation,
        " inplace=",
        use_inplace,
        sep="",
    )

    # ---- M-dependent A-side / output buffers ----
    var a_shape = row_major(Coord(Int(M), Idx[packed_K]))
    var b_shape = row_major(Coord(Idx[num_experts], Idx[N], Idx[packed_K]))
    var c_shape = row_major(Coord(Int(M), Idx[N]))

    var a_size = M * packed_K
    var c_size = M * N

    var a_host_ptr = alloc[Scalar[a_type]](a_size)
    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_tensor = TileTensor(a_device, a_shape)

    # Wrap shared pre-permuted B.
    var b_perm_tensor = TileTensor(shared.b_perm, b_shape)

    # REF path needs its own BF16 scratch.
    var c_ref_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_tensor = TileTensor(c_ref_device, c_shape)

    # ---- Per-expert offsets / IDs sized at comptime upper bound
    # `num_experts`. Tail slots [num_active_experts, num_experts) are
    # padded (0 tokens, -1 id); the kernel skips expert_id < 0.
    var a_offsets_host_ptr = alloc[UInt32](num_experts + 1)
    var a_scale_offsets_ptr = alloc[UInt32](num_experts)
    var expert_ids_host_ptr = alloc[Int32](num_experts)

    var a_offsets_device = ctx.enqueue_create_buffer[.uint32](num_experts + 1)
    var a_offsets_tensor = TileTensor(
        a_offsets_device, row_major(Coord(Idx[num_experts + 1]))
    )
    var a_scale_offsets_device = ctx.enqueue_create_buffer[.uint32](num_experts)
    var a_scale_offsets_tensor = TileTensor(
        a_scale_offsets_device, row_major(Coord(Idx[num_experts]))
    )
    var expert_ids_device = ctx.enqueue_create_buffer[.int32](num_experts)
    var expert_ids_tensor = TileTensor(
        expert_ids_device, row_major(Coord(Idx[num_experts]))
    )
    var expert_scales_tensor = TileTensor(
        shared.expert_scales, row_major(Coord(Idx[num_experts]))
    )

    var a_scale_dim0 = 0
    a_offsets_host_ptr[0] = 0
    for i in range(num_active_experts):
        a_scale_offsets_ptr[i] = UInt32(
            a_scale_dim0
            - Int(a_offsets_host_ptr[i] // UInt32(SF_MN_GROUP_SIZE))
        )
        var local_m = num_tokens_by_expert[i]
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[i] + UInt32(local_m)
        a_scale_dim0 += ceildiv(local_m, SF_MN_GROUP_SIZE)
        expert_ids_host_ptr[i] = Int32(expert_ids[i])

    # Pad tail slots: 0 tokens, -1 id.
    for i in range(num_active_experts, num_experts):
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[num_active_experts]
        a_scale_offsets_ptr[i] = UInt32(0)
        expert_ids_host_ptr[i] = Int32(-1)

    # ---- A scales (5D, E8M0). ----
    comptime k_groups = ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)

    var a_scales_shape = row_major(
        Coord(
            Int(a_scale_dim0),
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var a_scales_total = a_scales_shape.product()
    var a_scales_host_ptr = alloc[Scalar[scales_dtype]](a_scales_total)
    var a_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        a_scales_total
    )

    # ---- Output buffers (fused FP8 packed + 5D E8M0 SFs) ----
    comptime k_groups_swiglu = ceildiv(H, SF_VECTOR_SIZE * SF_ATOM_K)
    var O_shape = row_major(Coord(Int(M), Idx[packed_H]))
    var O_size = M * packed_H
    var swiglu_scales_shape = row_major(
        Coord(
            Int(a_scale_dim0),
            Idx[k_groups_swiglu],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var S_size = swiglu_scales_shape.product()

    var O_ref_device = ctx.enqueue_create_buffer[fp8_dtype](O_size)
    var O_test_device = ctx.enqueue_create_buffer[fp8_dtype](O_size)
    var S_ref_device = ctx.enqueue_create_buffer[scales_dtype](S_size)
    var S_test_device = ctx.enqueue_create_buffer[scales_dtype](S_size)
    var O_ref_host_ptr = alloc[Scalar[fp8_dtype]](O_size)
    var O_test_host_ptr = alloc[Scalar[fp8_dtype]](O_size)
    var S_ref_host_ptr = alloc[Scalar[scales_dtype]](S_size)
    var S_test_host_ptr = alloc[Scalar[scales_dtype]](S_size)
    var O_ref_tensor = TileTensor(O_ref_device, O_shape)
    var O_test_tensor = TileTensor(O_test_device, O_shape)
    var S_ref_tensor = TileTensor(S_ref_device, swiglu_scales_shape)
    var S_test_tensor = TileTensor(S_test_device, swiglu_scales_shape)

    # ---- Init A + A-scales ----
    rand(a_host_ptr.bitcast[UInt8](), a_size, min=0, max=255)

    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    var sf_zero = e8m0(Float32(0.0))
    var sf_small = e8m0(Float32(0.0078125))
    for i in range(a_scales_total):
        a_scales_host_ptr[i] = sf_zero

    # Set A scales (per-block) for the live token rows.
    for i in range(num_active_experts):
        var start = Int(a_offsets_host_ptr[i])
        var end = Int(a_offsets_host_ptr[i + 1])
        var local_m = end - start
        var actual_start = (
            start // SF_MN_GROUP_SIZE + Int(a_scale_offsets_ptr[i])
        ) * SF_MN_GROUP_SIZE
        var actual_end = actual_start + local_m
        for idx0 in range(actual_start, actual_end):
            for idx1 in range(0, K, SF_VECTOR_SIZE):
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    a_scales_host, idx0, idx1, sf_small
                )

    # ---- Copy A-side + offsets to device ----
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(a_offsets_device, a_offsets_host_ptr)
    ctx.enqueue_copy(a_scale_offsets_device, a_scale_offsets_ptr)
    ctx.enqueue_copy(expert_ids_device, expert_ids_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)

    var a_scales_tt = TileTensor(
        a_scales_device, a_scales_shape
    ).as_unsafe_any_origin()
    var b_scales_perm_shape = row_major(
        Coord(
            Idx[num_experts],
            Idx[ceildiv(N, SF_MN_GROUP_SIZE)],
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var b_scales_perm_tt = TileTensor(
        shared.b_scales_perm, b_scales_perm_shape
    ).as_unsafe_any_origin()
    var expert_scales_tt = TileTensor(
        shared.expert_scales, row_major(Coord(Int64(num_experts)))
    ).as_unsafe_any_origin()

    # ---- Path REF: non-fused matmul -> BF16 -> standalone SwiGLU. ----
    grouped_matmul_mxfp8_dispatch[transpose_b=transpose_b](
        c_ref_tensor,
        a_tensor,
        b_perm_tensor,
        a_scales_tt,
        b_scales_perm_tt,
        a_offsets_tensor,
        a_scale_offsets_tensor,
        expert_ids_tensor,
        expert_scales_tt,
        num_active_experts,
        M,
        ctx,
    )

    comptime hw_info = ctx.default_device_info
    var c_ref_immut = c_ref_tensor.as_immut()
    var a_offsets_immut = a_offsets_tensor.as_immut()
    var a_scale_offsets_immut = a_scale_offsets_tensor.as_immut()

    comptime ref_silu_mxfp8 = fused_silu_mxfp8_interleaved_kernel[
        fp8_dtype,
        scales_dtype,
        c_type,
        O_ref_tensor.LayoutType,
        S_ref_tensor.LayoutType,
        c_ref_immut.LayoutType,
        a_offsets_immut.LayoutType,
        a_scale_offsets_immut.LayoutType,
        hw_info.max_thread_block_size,
        hw_info.sm_count,
        clamp_activation=clamp_activation,
    ]
    ctx.enqueue_function[ref_silu_mxfp8](
        O_ref_tensor,
        S_ref_tensor,
        c_ref_immut,
        a_offsets_immut,
        a_scale_offsets_immut,
        alpha,
        limit,
        grid_dim=hw_info.sm_count,
        block_dim=hw_info.max_thread_block_size,
    )

    # ---- Path TEST: fused MXFP8 dispatch. ----
    grouped_matmul_swiglu_mxfp8_dispatch[
        transpose_b=transpose_b,
        match_bf16=match_bf16,
        use_inplace=use_inplace,
        clamp_activation=clamp_activation,
    ](
        O_test_tensor,
        S_test_tensor,
        a_tensor,
        b_perm_tensor,
        a_scales_tt,
        b_scales_perm_tt,
        a_offsets_tensor,
        a_scale_offsets_tensor,
        expert_ids_tensor,
        expert_scales_tt,
        num_active_experts,
        est_m,
        ctx,
        alpha,
        limit,
    )
    ctx.synchronize()

    # ---- Copy results back, byte-exact compare ----
    ctx.enqueue_copy(O_ref_host_ptr, O_ref_device)
    ctx.enqueue_copy(O_test_host_ptr, O_test_device)
    ctx.enqueue_copy(S_ref_host_ptr, S_ref_device)
    ctx.enqueue_copy(S_test_host_ptr, S_test_device)
    ctx.synchronize()

    var O_mismatch = 0
    var first_bad_O = -1
    for i in range(O_size):
        var ref_b = bitcast[.uint8, 1](SIMD[fp8_dtype, 1](O_ref_host_ptr[i]))[0]
        var test_b = bitcast[.uint8, 1](SIMD[fp8_dtype, 1](O_test_host_ptr[i]))[
            0
        ]
        if ref_b != test_b:
            if first_bad_O < 0:
                first_bad_O = i
            O_mismatch += 1

    var S_mismatch = 0
    var first_bad_S = -1
    for i in range(S_size):
        var ref_b = bitcast[.uint8, 1](
            SIMD[scales_dtype, 1](S_ref_host_ptr[i])
        )[0]
        var test_b = bitcast[.uint8, 1](
            SIMD[scales_dtype, 1](S_test_host_ptr[i])
        )[0]
        if ref_b != test_b:
            if first_bad_S < 0:
                first_bad_S = i
            S_mismatch += 1

    a_host_ptr.free()
    a_offsets_host_ptr.free()
    a_scale_offsets_ptr.free()
    expert_ids_host_ptr.free()
    a_scales_host_ptr.free()
    O_ref_host_ptr.free()
    O_test_host_ptr.free()
    S_ref_host_ptr.free()
    S_test_host_ptr.free()

    if O_mismatch != 0 or S_mismatch != 0:
        print(
            "    OUT mismatch =",
            O_mismatch,
            "/",
            O_size,
            " first @ idx=",
            first_bad_O,
            "  SF mismatch =",
            S_mismatch,
            "/",
            S_size,
            " first @ idx=",
            first_bad_S,
        )
        raise Error(
            "MXFP8 fused dispatch disagrees with non-fused reference chain"
        )

    print("    OK  (byte-exact: O and S match across fused vs reference)")


def main() raises:
    with DeviceContext() as ctx:
        # EP=8 GMM shape: 128 total experts / 8 EP ranks = 16 per
        # rank. hidden_size=6144, intermediate_size=3072, so N=2*3072
        # and K=hidden_size.
        comptime NUM_E = 16
        comptime N = 6144
        comptime K = 6144

        var shared = _build_shared_b[NUM_E, N, K](ctx)

        print("=== EP=8 grouped_matmul_swiglu_mxfp8_dispatch ===")

        # A: decode 1 expert.
        print("  A: 1 active, [2] tokens")
        _test_swiglu_mxfp8_dispatch[NUM_E, N, K](
            [2],
            [0],
            shared,
            ctx,
        )

        # A.est0: same single-token-decode layout as A, but with
        # estimated_total_m = 0 -- the value expert_parallel.py computes for an
        # EP8 decode step (S=1, top_k=4, n_gpus=8 -> 1*4//8 == 0). Pre-fix this
        # sized the unused fused-path C buffer as [0, N] and the launcher
        # encoded a zero-dim C TMA descriptor, crashing at tma.mojo:404. Output
        # must still be byte-exact to the reference.
        print("  A.est0: 1 active, [2] tokens, estimated_total_m=0")
        _test_swiglu_mxfp8_dispatch[NUM_E, N, K](
            [2],
            [0],
            shared,
            ctx,
            est_m_override=0,
        )

        # B: decode 8 experts, uniform.
        print("  B: 8 active, [2] x 8 tokens")
        _test_swiglu_mxfp8_dispatch[NUM_E, N, K](
            _make_uniform(2, 8),
            _make_range(8),
            shared,
            ctx,
        )

        # C: small prefill, all 16 experts x 16 tokens.
        print("  C: 16 active, [16] x 16 tokens")
        _test_swiglu_mxfp8_dispatch[NUM_E, N, K](
            _make_uniform(16, 16),
            _make_range(16),
            shared,
            ctx,
        )

        # D: large prefill, all 16 experts x 64 tokens.
        print("  D: 16 active, [64] x 16 tokens")
        _test_swiglu_mxfp8_dispatch[NUM_E, N, K](
            _make_uniform(64, 16),
            _make_range(16),
            shared,
            ctx,
        )

        # E: ragged unaligned, 4 experts [127, 257, 513, 1025].
        print("  E: 4 active, [127, 257, 513, 1025], ids=[0,3,2,4]")
        _test_swiglu_mxfp8_dispatch[NUM_E, N, K](
            [127, 257, 513, 1025],
            [0, 3, 2, 4],
            shared,
            ctx,
        )

        # E.masked: -1 ids interleaved.
        print("  E.masked: 5 active, [4,0,1,1,1], ids=[0,-1,1,2,3]")
        _test_swiglu_mxfp8_dispatch[NUM_E, N, K](
            [4, 0, 1, 1, 1],
            [0, -1, 1, 2, 3],
            shared,
            ctx,
        )

        # E.tail-pad: token counts at varied mod-128 positions force
        # kernel-side zero-fill of pad rows to byte-match the chain.
        print("  E.tail-pad: 3 active, [129, 50, 1], ids=[0,1,2]")
        _test_swiglu_mxfp8_dispatch[NUM_E, N, K](
            [129, 50, 1],
            [0, 1, 2],
            shared,
            ctx,
        )

        # Ragged decode sweep (mirrors NVFP4 decode-B4/B8/B16).
        print("  decode-B4: 5 active, [4,1,1,1,1]")
        _test_swiglu_mxfp8_dispatch[NUM_E, N, K](
            _make_ragged(4, 1, 4),
            _make_range(5),
            shared,
            ctx,
        )

        print("  decode-B8: 9 active, [8] + 8*[1]")
        _test_swiglu_mxfp8_dispatch[NUM_E, N, K](
            _make_ragged(8, 1, 8),
            _make_range(9),
            shared,
            ctx,
        )

        print("  decode-B16: 16 active, [16] + 15*[1]")
        _test_swiglu_mxfp8_dispatch[NUM_E, N, K](
            _make_ragged(16, 1, 15),
            _make_range(16),
            shared,
            ctx,
        )

        # Shared-expert-style prefill: one expert carries the full
        # batch (the folded shared expert) plus 15 lightly-loaded
        # routed experts. This is the shape family where the NVFP4
        # sibling has a known single-byte match_bf16 divergence, so
        # probe it byte-exact here.
        print("  prefill-shared-rt8: 16 active, [4096] + 15*[8]")
        _test_swiglu_mxfp8_dispatch[NUM_E, N, K](
            _make_ragged(4096, 8, 15),
            _make_range(16),
            shared,
            ctx,
        )

        print("  prefill-shared-rt32: 16 active, [4096] + 15*[32]")
        _test_swiglu_mxfp8_dispatch[NUM_E, N, K](
            _make_ragged(4096, 32, 15),
            _make_range(16),
            shared,
            ctx,
        )

        # Single-expert large M (shared expert alone at prefill).
        print("  prefill-single: 1 active, [4096]")
        _test_swiglu_mxfp8_dispatch[NUM_E, N, K](
            [4096],
            [0],
            shared,
            ctx,
        )

        # Decode through the cooperative path (use_inplace=False) on
        # the decode-B8 shape, isolating in-place vs cooperative on
        # the decode config.
        print("  decode-B8 cooperative: 9 active, [8] + 8*[1]")
        _test_swiglu_mxfp8_dispatch[NUM_E, N, K, use_inplace=False](
            _make_ragged(8, 1, 8),
            _make_range(9),
            shared,
            ctx,
        )

        # Clamped + decode (clamp through the in-place register path).
        print("  clamp decode: 8 active, [2] x 8")
        _test_swiglu_mxfp8_dispatch[NUM_E, N, K, clamp_activation=True](
            _make_uniform(2, 8),
            _make_range(8),
            shared,
            ctx,
        )

        # Clamped (swigluoai) activation with the canonical HF config
        # values alpha=1.702, limit=7.0.
        print("  clamp: 2 active, [64, 33], alpha=1.702 limit=7.0")
        _test_swiglu_mxfp8_dispatch[NUM_E, N, K, clamp_activation=True](
            [64, 33],
            [0, 1],
            shared,
            ctx,
        )

        # Clamped + ragged prefill mix.
        print("  clamp prefill: 4 active, [127, 257, 513, 1025]")
        _test_swiglu_mxfp8_dispatch[NUM_E, N, K, clamp_activation=True](
            [127, 257, 513, 1025],
            [0, 3, 2, 4],
            shared,
            ctx,
        )

        # match_bf16=False (fp32 across SMEM scatter).
        print("  match_bf16=False: 2 active, [64, 33]")
        _test_swiglu_mxfp8_dispatch[NUM_E, N, K, match_bf16=False](
            [64, 33],
            [0, 1],
            shared,
            ctx,
        )

        print("\nALL EP=8 MXFP8 SWIGLU DISPATCH TESTS PASSED!")

        _ = shared^
