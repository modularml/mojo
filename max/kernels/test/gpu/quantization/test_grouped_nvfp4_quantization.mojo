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

from std.math import ceildiv
from max.gpu.host import DeviceContext
from std.testing import assert_equal
from layout import Coord, Idx, TileTensor, row_major
from layout._fillers import random
from linalg.fp4_utils import (
    NVFP4_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    NVFP4_SF_VECTOR_SIZE,
    cast_uint_to_fp4e2m1,
)
from linalg.block_scaled_quantization import (
    quantize_dynamic_scaled_fp4fp8,
    grouped_quantize_dynamic_scaled_fp4_async,
)


def test_grouped_nvfp4_quantization[
    dtype: DType,
    scales_dtype: DType,
    SF_VECTOR_SIZE: Int,
    N: Int,
    num_experts: Int,
](
    ctx: DeviceContext,
    expert_counts: List[Int],
    tensor_sfs: List[Float32],
) raises:
    assert_equal(len(expert_counts), num_experts)
    assert_equal(len(tensor_sfs), num_experts)
    comptime out_dtype = DType.uint8
    comptime K_tiles = ceildiv(N, SF_VECTOR_SIZE * SF_ATOM_K)
    comptime output_N = ceildiv(N, 2)
    comptime scales_per_m_tile = K_tiles * SF_MN_GROUP_SIZE * SF_ATOM_K
    comptime SENTINEL = 0xEE

    # Derive row_offsets, tile_starts, scales_offsets from expert_counts.
    var row_offsets_host = alloc[UInt32](num_experts + 1)
    var scales_offsets_host = alloc[UInt32](num_experts)
    var expert_ids_host = alloc[Int32](num_experts)
    var sf_tensor_host = alloc[Float32](num_experts)
    var tile_starts = alloc[Int](num_experts + 1)

    row_offsets_host[0] = 0
    tile_starts[0] = 0
    for i in range(num_experts):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(expert_counts[i])
        tile_starts[i + 1] = tile_starts[i] + ceildiv(
            expert_counts[i], SF_MN_GROUP_SIZE
        )
        scales_offsets_host[i] = UInt32(
            tile_starts[i] - Int(row_offsets_host[i]) // SF_MN_GROUP_SIZE
        )
        expert_ids_host[i] = Int32(i)
        sf_tensor_host[i] = tensor_sfs[i]

    var total_tokens = Int(row_offsets_host[num_experts])
    var total_m_tiles = tile_starts[num_experts]
    var total_scales = total_m_tiles * scales_per_m_tile

    # --- Grouped kernel buffers ---
    var host_input = alloc[Scalar[dtype]](total_tokens * N)
    var host_input_tensor = TileTensor(
        host_input, row_major(Coord(total_tokens, Idx[N]))
    )
    random(host_input_tensor, min=-1.0, max=1.0)

    var dev_input = ctx.enqueue_create_buffer[dtype](max(total_tokens * N, 1))
    # One slack row past the payload, sentinel-filled: any kernel write there
    # is an out-of-bounds payload store. The last k_idx block covers a full
    # SF_K_GROUP_SIZE column block, so an N that is not a multiple of 64 used
    # to spill the tail columns into the next row.
    var dev_output = ctx.enqueue_create_buffer[out_dtype](
        max(total_tokens * output_N, 1) + output_N
    )
    var sentinel_host = alloc[Scalar[out_dtype]](
        total_tokens * output_N + output_N
    )
    for i in range(total_tokens * output_N + output_N):
        sentinel_host[i] = SENTINEL
    ctx.enqueue_copy(dev_output, sentinel_host)
    var dev_scales = ctx.enqueue_create_buffer[scales_dtype](
        max(total_scales, 1)
    )
    var dev_row_offsets = ctx.enqueue_create_buffer[.uint32](num_experts + 1)
    var dev_scales_offsets = ctx.enqueue_create_buffer[.uint32](num_experts)
    var dev_expert_ids = ctx.enqueue_create_buffer[.int32](num_experts)
    var dev_sf_tensor = ctx.enqueue_create_buffer[.float32](num_experts)

    ctx.enqueue_copy(dev_input, host_input)
    ctx.enqueue_copy(dev_row_offsets, row_offsets_host)
    ctx.enqueue_copy(dev_scales_offsets, scales_offsets_host)
    ctx.enqueue_copy(dev_expert_ids, expert_ids_host)
    ctx.enqueue_copy(dev_sf_tensor, sf_tensor_host)

    var input_shape = Coord(total_tokens, Idx[N])
    var output_shape = Coord(total_tokens, Idx[output_N])
    var scales_shape = Coord(
        total_m_tiles,
        Idx[K_tiles],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )

    var input_tensor = TileTensor(dev_input, row_major(input_shape))
    var output_tensor = TileTensor(dev_output, row_major(output_shape))
    var scales_tensor = TileTensor(dev_scales, row_major(scales_shape))
    var row_offsets_tensor = TileTensor(
        dev_row_offsets, row_major(Coord(Idx[num_experts + 1]))
    )
    var scales_offsets_tensor = TileTensor(
        dev_scales_offsets, row_major(Coord(Idx[num_experts]))
    )
    var expert_ids_tensor = TileTensor(
        dev_expert_ids, row_major(Coord(Idx[num_experts]))
    )
    var sf_tensor_device = TileTensor(
        dev_sf_tensor, row_major(Coord(Idx[num_experts]))
    )

    grouped_quantize_dynamic_scaled_fp4_async(
        output_tensor,
        scales_tensor,
        input_tensor,
        row_offsets_tensor,
        scales_offsets_tensor,
        expert_ids_tensor,
        sf_tensor_device,
        ctx,
    )

    # --- Copy grouped results back to host ---
    var host_output = alloc[Scalar[out_dtype]](
        total_tokens * output_N + output_N
    )
    var host_scales = alloc[Scalar[scales_dtype]](max(total_scales, 1))

    ctx.enqueue_copy(host_output, dev_output)
    ctx.enqueue_copy(host_scales, dev_scales)
    ctx.synchronize()

    # The slack row past the last token must still hold the sentinel.
    for i in range(output_N):
        assert_equal(Int(host_output[total_tokens * output_N + i]), SENTINEL)

    # --- Per-expert reference and comparison ---
    for expert_i in range(num_experts):
        var count = expert_counts[expert_i]
        if count == 0:
            continue

        var row_start = Int(row_offsets_host[expert_i])
        var m_tiles_i = ceildiv(count, SF_MN_GROUP_SIZE)
        var scales_total_i = m_tiles_i * scales_per_m_tile

        var ref_input = ctx.enqueue_create_buffer[dtype](count * N)
        var ref_output = ctx.enqueue_create_buffer[out_dtype](count * output_N)
        var ref_scales = ctx.enqueue_create_buffer[scales_dtype](scales_total_i)

        ctx.enqueue_copy(ref_input, host_input + row_start * N)

        var ref_input_shape = Coord(count, Idx[N])
        var ref_output_shape = Coord(count, Idx[output_N])
        var ref_scales_shape = Coord(
            m_tiles_i,
            Idx[K_tiles],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )

        var ref_input_tensor = TileTensor(ref_input, row_major(ref_input_shape))
        var ref_output_tensor = TileTensor(
            ref_output, row_major(ref_output_shape)
        )
        var ref_scales_tensor = TileTensor(
            ref_scales, row_major(ref_scales_shape)
        )

        quantize_dynamic_scaled_fp4fp8[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
            ctx,
            ref_output_tensor,
            ref_scales_tensor,
            ref_input_tensor,
            num_cols=N,
            num_cols_padded=N,
            tensor_sf=tensor_sfs[expert_i],
        )

        var host_ref_output = alloc[Scalar[out_dtype]](count * output_N)
        var host_ref_scales = alloc[Scalar[scales_dtype]](scales_total_i)

        ctx.enqueue_copy(host_ref_output, ref_output)
        ctx.enqueue_copy(host_ref_scales, ref_scales)
        ctx.synchronize()

        # Compare output for this expert.
        var grouped_out = TileTensor(
            host_output + row_start * output_N,
            row_major(ref_output_shape),
        )
        var ref_out = TileTensor(host_ref_output, row_major(ref_output_shape))

        for row_idx in range(count):
            for col_idx in range(0, output_N, SF_VECTOR_SIZE // 2):
                var vec = grouped_out.load[width=SF_VECTOR_SIZE // 2](
                    Coord(row_idx, col_idx)
                )
                var vec_ref = ref_out.load[width=SF_VECTOR_SIZE // 2](
                    Coord(row_idx, col_idx)
                )
                var fp32 = cast_uint_to_fp4e2m1[
                    out_dtype=DType.float32, out_width=SF_VECTOR_SIZE
                ](vec)
                var fp32_ref = cast_uint_to_fp4e2m1[
                    out_dtype=DType.float32, out_width=SF_VECTOR_SIZE
                ](vec_ref)
                assert_equal(fp32, fp32_ref)

        # Compare scales for this expert.
        var tile_start_i = tile_starts[expert_i]
        var grouped_scales = TileTensor(
            host_scales + tile_start_i * scales_per_m_tile,
            row_major(ref_scales_shape),
        )
        var ref_sc = TileTensor(host_ref_scales, row_major(ref_scales_shape))

        for mi in range(m_tiles_i):
            for ki in range(K_tiles):
                for a0 in range(SF_ATOM_M[0]):
                    for a1 in range(SF_ATOM_M[1]):
                        for ak in range(SF_ATOM_K):
                            var c = Coord(
                                mi,
                                ki,
                                a0,
                                a1,
                                ak,
                            )
                            assert_equal(
                                grouped_scales[c].cast[.float64](),
                                ref_sc[c].cast[.float64](),
                            )

        host_ref_output.free()
        host_ref_scales.free()

    host_input.free()
    host_output.free()
    host_scales.free()
    sentinel_host.free()
    row_offsets_host.free()
    scales_offsets_host.free()
    expert_ids_host.free()
    sf_tensor_host.free()
    tile_starts.free()


def main() raises:
    with DeviceContext() as ctx:
        # Aligned expert counts.
        test_grouped_nvfp4_quantization[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
            N=11 * 64,
            num_experts=3,
        ](ctx, [256, 128, 256], [1.0, 1.0, 1.0])

        # Non-aligned expert counts.
        test_grouped_nvfp4_quantization[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
            N=11 * 64,
            num_experts=3,
        ](ctx, [100, 200, 50], [1.0, 1.0, 1.0])

        # Odd sizes.
        test_grouped_nvfp4_quantization[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
            N=11 * 64,
            num_experts=2,
        ](ctx, [13, 999], [1.0, 1.0])

        # Mixed scale factors.
        test_grouped_nvfp4_quantization[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
            N=23 * 128,
            num_experts=3,
        ](ctx, [256, 128, 256], [0.43, 1.0, 0.5])

        # Column count that is not a multiple of SF_VECTOR_SIZE * SF_ATOM_K
        # (64). Gemma 4 26B-A4B has moe_dim 704, so a 2-way tensor-parallel
        # shard quantizes the 352-column SiLU output; the last k_idx block
        # covers columns 320-383 and must mask the 352-383 tail.
        test_grouped_nvfp4_quantization[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
            N=352,
            num_experts=2,
        ](ctx, [100, 60], [1.0, 0.5])
