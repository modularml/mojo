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
"""Byte-exact correctness test for `fused_silu_mxfp8_interleaved_kernel`.

The kernel takes an interleaved BF16 input `[g0, u0, g1, u1, ..., g_{H-1},
u_{H-1}]` per token, applies `silu(g) * u`, and quantizes per 32-element
block to FP8-E4M3 with FP8-UE8M0 block scales. This test:

1. Builds a deterministic BF16 input tensor on the host.
2. Computes the expected MXFP8 output + 5D E8M0 scale tile on the host
   using the same arithmetic as the kernel (`block_max / 448`, cast to
   E8M0 with round-to-nearest-even on the exponent, then quantize each
   element to FP8-E4M3 via the rounding the standard `cast` performs).
3. Launches the kernel and copies the result back.
4. Asserts byte-equal on both the output tensor and the scale tile.

Multiple expert shapes are exercised (single expert, multi-expert with
ragged token counts) to cover the per-expert offset / SF-block-id math.
"""
from std.math import align_up, ceildiv, exp, recip
from max.gpu.host import DeviceContext
from std.memory import alloc
from std.memory.unsafe import bitcast

from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
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


def _test_silu_mxfp8[
    num_active_experts: Int,
    H: Int,
    clamp_activation: Bool = False,
](
    num_tokens_by_expert: List[Int],
    ctx: DeviceContext,
    alpha: Float32 = Float32(1.702),
    limit: Float32 = Float32(7.0),
) raises:
    """Run the kernel on a `num_active_experts` × ragged-M workload and
    assert byte-equality with a host-computed reference. When
    `clamp_activation=True` both the kernel and the host reference
    apply the clamped (swigluoai) activation with the given
    alpha/limit.
    """
    comptime fp8_dtype = DType.float8_e4m3fn
    comptime scales_dtype = MXFP8_SF_DTYPE
    comptime in_dtype = DType.bfloat16
    comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE

    comptime assert (
        H % (SF_VECTOR_SIZE * SF_ATOM_K) == 0
    ), "H must be a multiple of (MXFP8_SF_VECTOR_SIZE * SF_ATOM_K) = 128"

    var total_num_tokens = 0
    for i in range(len(num_tokens_by_expert)):
        total_num_tokens += num_tokens_by_expert[i]
    var M = total_num_tokens

    print(
        "  H=",
        H,
        " M=",
        M,
        " active=",
        num_active_experts,
        " clamp=",
        clamp_activation,
        sep="",
    )

    # ---- Input / output / SF buffers ----
    comptime two_H = 2 * H
    var in_shape = row_major(Coord(Int(M), Idx[two_H]))
    var out_shape = row_major(Coord(Int(M), Idx[H]))
    var in_size = M * two_H
    var out_size = M * H

    var in_host_ptr = alloc[Scalar[in_dtype]](in_size)
    var out_host_ptr = alloc[Scalar[fp8_dtype]](out_size)
    var out_ref_host_ptr = alloc[Scalar[fp8_dtype]](out_size)

    var in_device = ctx.enqueue_create_buffer[in_dtype](in_size)
    var in_tensor = TileTensor(in_device, in_shape)
    var out_device = ctx.enqueue_create_buffer[fp8_dtype](out_size)
    var out_tensor = TileTensor(out_device, out_shape)

    # ---- Per-expert offsets ----
    var row_offsets_host_ptr = alloc[UInt32](num_active_experts + 1)
    var scales_offsets_host_ptr = alloc[UInt32](num_active_experts)

    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](
        num_active_experts + 1
    )
    var row_offsets_tensor = TileTensor(
        row_offsets_device,
        row_major(Coord(Idx[num_active_experts + 1])),
    )
    var scales_offsets_device = ctx.enqueue_create_buffer[.uint32](
        num_active_experts
    )
    var scales_offsets_tensor = TileTensor(
        scales_offsets_device,
        row_major(Coord(Idx[num_active_experts])),
    )

    var scales_dim0 = 0
    row_offsets_host_ptr[0] = 0
    for i in range(num_active_experts):
        scales_offsets_host_ptr[i] = UInt32(
            scales_dim0
            - Int(row_offsets_host_ptr[i] // UInt32(SF_MN_GROUP_SIZE))
        )
        var local_m = num_tokens_by_expert[i]
        row_offsets_host_ptr[i + 1] = row_offsets_host_ptr[i] + UInt32(local_m)
        scales_dim0 += ceildiv(local_m, SF_MN_GROUP_SIZE)

    comptime k_groups = ceildiv(H, SF_VECTOR_SIZE * SF_ATOM_K)
    var scales_shape = row_major(
        Coord(
            Int(scales_dim0),
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var scales_size = scales_shape.product()

    var scales_host_ptr = alloc[Scalar[scales_dtype]](scales_size)
    var scales_ref_host_ptr = alloc[Scalar[scales_dtype]](scales_size)
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](scales_size)
    var scales_tensor = TileTensor(scales_device, scales_shape)

    # Zero-initialize scale tiles so the trailing zero-pad path is
    # observed (matches the kernel's pad-with-zero contract for the
    # rows in [expert_m, SF_MN_GROUP_SIZE)).
    for i in range(scales_size):
        scales_host_ptr[i] = Scalar[scales_dtype](0.0)
        scales_ref_host_ptr[i] = Scalar[scales_dtype](0.0)

    # Deterministic [-3.5, +3.5] LCG fill so silu(g)*u stays inside
    # FP8-E4M3 range and exercises multiple E8M0 scale buckets.
    for i in range(in_size):
        var v_i = (i * 1103515245 + 12345) % 16
        var fv = Float32(Int(v_i) - 7) * 0.5
        in_host_ptr[i] = fv.cast[in_dtype]()

    # Host reference: per-token, per-32-elt block compute silu(g)*u,
    # derive E8M0 scale = block_max/448, quantize each element to FP8.
    var ref_full_sf_view = TileTensor(
        scales_ref_host_ptr,
        scales_shape,
    )
    for e in range(num_active_experts):
        var start = Int(row_offsets_host_ptr[e])
        var end = Int(row_offsets_host_ptr[e + 1])
        var local_m = end - start
        var scales_block_id = (start // SF_MN_GROUP_SIZE) + Int(
            scales_offsets_host_ptr[e]
        )
        for tok in range(local_m):
            var m = start + tok
            for k_blk in range(H // SF_VECTOR_SIZE):
                var k_base = k_blk * SF_VECTOR_SIZE
                var gate_block = SIMD[.float32, SF_VECTOR_SIZE]()
                var up_block = SIMD[.float32, SF_VECTOR_SIZE]()
                for j in range(SF_VECTOR_SIZE):
                    var col = k_base + j
                    var g_bf = in_host_ptr[m * two_H + 2 * col]
                    var u_bf = in_host_ptr[m * two_H + 2 * col + 1]
                    gate_block[j] = g_bf.cast[.float32]()
                    up_block[j] = u_bf.cast[.float32]()

                var z = SIMD[.float32, SF_VECTOR_SIZE]()
                var block_max = Float32(0.0)
                for j in range(SF_VECTOR_SIZE):
                    var g = gate_block[j]
                    var u = up_block[j]
                    var zi: Float32
                    comptime if clamp_activation:
                        var g_c = min(g, limit)
                        var u_c = max(min(u, limit), -limit)
                        var sigmoid = recip(Float32(1.0) + exp(-(g_c * alpha)))
                        zi = (u_c + Float32(1.0)) * g_c * sigmoid
                    else:
                        var silu_g = g * recip(Float32(1.0) + exp(-g))
                        zi = silu_g * u
                    z[j] = zi
                    var az = abs(zi)
                    if az > block_max:
                        block_max = az

                var scale_factor = block_max * recip(Float32(448.0))
                var sf = scale_factor.cast[scales_dtype]()
                var output_scale = Float32(0.0)
                if block_max != 0:
                    output_scale = recip(sf.cast[.float32]())

                var scaled = z * output_scale
                var out_vec = scaled.cast[fp8_dtype]()

                for j in range(SF_VECTOR_SIZE):
                    out_ref_host_ptr[m * H + k_base + j] = out_vec[j]

                # Store the per-block scale into the global 5D SF tile.
                # The row_idx is `scales_block_id * SF_MN_GROUP_SIZE +
                # tok` so `set_scale_factor`'s i0 = row // 128 lands
                # in this expert's reserved SF-block slab.
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    ref_full_sf_view,
                    scales_block_id * SF_MN_GROUP_SIZE + tok,
                    k_base,
                    sf,
                )

    # ---- Copy inputs to device, launch kernel ----
    ctx.enqueue_copy(in_device, in_host_ptr)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host_ptr)
    ctx.enqueue_copy(scales_offsets_device, scales_offsets_host_ptr)
    # Pre-fill device SF tile with zeros so the kernel's per-element
    # writes are the only non-zero values. (The reference's matching
    # zero-pad regions stay zero on both sides.)
    ctx.enqueue_copy(scales_device, scales_host_ptr)

    comptime hw_info = ctx.default_device_info

    var in_immut = in_tensor.as_immut()
    var row_offsets_immut = row_offsets_tensor.as_immut()
    var scales_offsets_immut = scales_offsets_tensor.as_immut()

    comptime kernel = fused_silu_mxfp8_interleaved_kernel[
        fp8_dtype,
        scales_dtype,
        in_dtype,
        out_tensor.LayoutType,
        scales_tensor.LayoutType,
        in_immut.LayoutType,
        row_offsets_immut.LayoutType,
        scales_offsets_immut.LayoutType,
        hw_info.max_thread_block_size,
        hw_info.sm_count,
        clamp_activation=clamp_activation,
    ]
    ctx.enqueue_function[kernel](
        out_tensor,
        scales_tensor,
        in_immut,
        row_offsets_immut,
        scales_offsets_immut,
        alpha,
        limit,
        grid_dim=hw_info.sm_count,
        block_dim=hw_info.max_thread_block_size,
    )
    ctx.synchronize()

    # ---- Copy device output back, byte-exact compare ----
    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.enqueue_copy(scales_host_ptr, scales_device)
    ctx.synchronize()

    # Byte-exact compare on the underlying fp8 byte stream. fp8_e4m3fn
    # and fp8_e8m0fnu are both 1-byte dtypes; bitcast each scalar to
    # UInt8 to get the raw byte for comparison.
    var out_mismatch = 0
    var first_bad_out = -1
    for i in range(out_size):
        var got_b = bitcast[.uint8, 1](SIMD[fp8_dtype, 1](out_host_ptr[i]))[0]
        var ref_b = bitcast[.uint8, 1](SIMD[fp8_dtype, 1](out_ref_host_ptr[i]))[
            0
        ]
        if got_b != ref_b:
            if first_bad_out < 0:
                first_bad_out = i
            out_mismatch += 1

    var sf_mismatch = 0
    var first_bad_sf = -1
    for i in range(scales_size):
        var got_b = bitcast[.uint8, 1](
            SIMD[scales_dtype, 1](scales_host_ptr[i])
        )[0]
        var ref_b = bitcast[.uint8, 1](
            SIMD[scales_dtype, 1](scales_ref_host_ptr[i])
        )[0]
        if got_b != ref_b:
            if first_bad_sf < 0:
                first_bad_sf = i
            sf_mismatch += 1

    if out_mismatch != 0 or sf_mismatch != 0:
        print(
            "    OUT mismatch count =",
            out_mismatch,
            " first @ idx=",
            first_bad_out,
            "  SF mismatch count =",
            sf_mismatch,
            " first @ idx=",
            first_bad_sf,
        )
        if first_bad_out >= 0:
            var m_bad = first_bad_out // H
            var k_bad = first_bad_out % H
            var ref_byte = bitcast[.uint8, 1](
                SIMD[fp8_dtype, 1](out_ref_host_ptr[first_bad_out])
            )[0]
            var got_byte = bitcast[.uint8, 1](
                SIMD[fp8_dtype, 1](out_host_ptr[first_bad_out])
            )[0]
            print(
                "    first OUT bad: m=",
                m_bad,
                " k=",
                k_bad,
                " ref(byte)=",
                ref_byte,
                " got(byte)=",
                got_byte,
            )
        raise Error("MXFP8 fused-silu reference disagrees with kernel output")

    print("    OK  (byte-exact: O and S match)")


def main() raises:
    var ctx = DeviceContext()

    # Single expert, exactly one SF-MN block (128 rows). H=128 is the
    # minimum H allowed by the SF-tile geometry constraint
    # `H % (32 * SF_ATOM_K = 128) == 0`.
    _test_silu_mxfp8[num_active_experts=1, H=128]([128], ctx)

    # Single expert, ragged M (forces the trailing zero-pad path).
    _test_silu_mxfp8[num_active_experts=1, H=128]([49], ctx)

    # Two experts, non-aligned token counts.
    _test_silu_mxfp8[num_active_experts=2, H=128]([64, 33], ctx)

    # Larger H (covers multiple SF k-groups per token).
    _test_silu_mxfp8[num_active_experts=1, H=256]([49], ctx)

    # Clamped (swigluoai) activation. limit=2.0 sits inside the
    # [-3.5, 3.5] input range so both clamp bounds actually bind.
    _test_silu_mxfp8[num_active_experts=1, H=128, clamp_activation=True](
        [49], ctx, alpha=Float32(1.702), limit=Float32(2.0)
    )

    # Wide-H case: single small expert at a model-scale hidden dim,
    # keeping test runtime manageable while exercising many SF
    # k-groups per token.
    _test_silu_mxfp8[num_active_experts=1, H=3072]([32], ctx)
