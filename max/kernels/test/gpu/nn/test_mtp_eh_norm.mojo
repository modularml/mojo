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
"""Tests for the MTP draft layer's fused input projection."""

from std.math import rsqrt, sqrt
from std.random import rand, seed
from std.testing import assert_equal, assert_true

from layout import TileTensor, row_major
from max.gpu import WARP_SIZE
from max.gpu.host import DeviceContext, HostBuffer

from nn.mtp_eh_norm import mtp_eh_norm_kernel


def _reference_row[
    hidden_size: Int
](
    src: HostBuffer[.bfloat16],
    weight: HostBuffer[.bfloat16],
    row: Int,
    col: Int,
    eps: Float64,
    multiply_before_cast: Bool,
) -> Float64:
    """One normalized channel, modelling the kernel's roundings exactly.

    Everything outside a rounding runs in f64, but each bf16 rounding the
    kernel performs is reproduced, so the expected value is exact rather than
    approximate. `multiply_before_cast` selects between the two orderings
    `ops.rms_norm` supports; the kernel implements the `False` arm.
    """
    var acc = Float64(0)
    for i in range(hidden_size):
        var v = Float64(src[row * hidden_size + i].cast[.float32]())
        acc += v * v
    var rrms = 1.0 / sqrt(acc / Float64(hidden_size) + eps)
    var x = Float64(src[row * hidden_size + col].cast[.float32]())
    var w = Float64(weight[col].cast[.float32]())
    if multiply_before_cast:
        return Float64(
            Scalar[.bfloat16]((x * rrms * w).cast[.float32]()).cast[.float32]()
        )
    var normed = Float64(
        Scalar[.bfloat16]((x * rrms).cast[.float32]()).cast[.float32]()
    )
    return Float64(
        Scalar[.bfloat16]((normed * w).cast[.float32]()).cast[.float32]()
    )


def test_mtp_eh_norm[
    hidden_size: Int, block_threads: Int
](num_tokens: Int, ctx: DeviceContext) raises:
    """Checks every output channel of every token against an f64 reference.

    Parameters:
        hidden_size: Channels per input.
        block_threads: Threads per block; the row is walked in strides of it.

    Args:
        num_tokens: Rows to normalize.
        ctx: Device context.
    """
    comptime eps32 = Float32(1e-5)
    # A power-of-two ceiling, the same value the graph op passes.
    comptime max_warps = (
        ctx.default_device_info.max_thread_block_size // WARP_SIZE
    )
    var eps = Float64(1e-5)

    print(
        "mtp_eh_norm: hidden=",
        hidden_size,
        " tokens=",
        num_tokens,
        " threads=",
        block_threads,
    )

    var embed_h = ctx.enqueue_create_host_buffer[.bfloat16](
        num_tokens * hidden_size
    )
    var prev_h = ctx.enqueue_create_host_buffer[.bfloat16](
        num_tokens * hidden_size
    )
    var ew_h = ctx.enqueue_create_host_buffer[.bfloat16](hidden_size)
    var hw_h = ctx.enqueue_create_host_buffer[.bfloat16](hidden_size)
    var out_h = ctx.enqueue_create_host_buffer[.bfloat16](
        num_tokens * 2 * hidden_size
    )
    ctx.synchronize()

    # The expectations below are exact, so the inputs have to be fixed.
    seed(0)
    rand(embed_h.unsafe_ptr(), num_tokens * hidden_size)
    rand(prev_h.unsafe_ptr(), num_tokens * hidden_size)
    rand(ew_h.unsafe_ptr(), hidden_size)
    rand(hw_h.unsafe_ptr(), hidden_size)
    # Different scales per half, so reusing one row's sum for both shows up.
    for i in range(num_tokens * hidden_size):
        embed_h[i] = (embed_h[i] - 0.5) * 2.0
        prev_h[i] = (prev_h[i] - 0.5) * 6.0
    for i in range(hidden_size):
        ew_h[i] = ew_h[i] + 0.5
        hw_h[i] = hw_h[i] + 0.5

    var embed_d = ctx.enqueue_create_buffer[.bfloat16](num_tokens * hidden_size)
    var prev_d = ctx.enqueue_create_buffer[.bfloat16](num_tokens * hidden_size)
    var ew_d = ctx.enqueue_create_buffer[.bfloat16](hidden_size)
    var hw_d = ctx.enqueue_create_buffer[.bfloat16](hidden_size)
    var out_d = ctx.enqueue_create_buffer[.bfloat16](
        num_tokens * 2 * hidden_size
    )
    ctx.enqueue_copy(embed_d, embed_h)
    ctx.enqueue_copy(prev_d, prev_h)
    ctx.enqueue_copy(ew_d, ew_h)
    ctx.enqueue_copy(hw_d, hw_h)
    ctx.synchronize()

    var out_t = TileTensor(out_d, row_major(num_tokens, 2 * hidden_size))
    var embed_t = TileTensor(embed_d, row_major(num_tokens, hidden_size))
    var prev_t = TileTensor(prev_d, row_major(num_tokens, hidden_size))
    var ew_t = TileTensor(ew_d, row_major(hidden_size))
    var hw_t = TileTensor(hw_d, row_major(hidden_size))

    comptime kernel = mtp_eh_norm_kernel[
        .bfloat16,
        out_t.LayoutType,
        out_t.origin,
        type_of(embed_t.as_immut()).LayoutType,
        ImmOrigin(embed_t.origin),
        type_of(prev_t.as_immut()).LayoutType,
        ImmOrigin(prev_t.origin),
        type_of(ew_t.as_immut()).LayoutType,
        ImmOrigin(ew_t.origin),
        type_of(hw_t.as_immut()).LayoutType,
        ImmOrigin(hw_t.origin),
        hidden_size,
        max_warps,
        out_t.Engine,
        type_of(embed_t.as_immut()).Engine,
        type_of(prev_t.as_immut()).Engine,
        type_of(ew_t.as_immut()).Engine,
        type_of(hw_t.as_immut()).Engine,
    ]
    ctx.enqueue_function[kernel](
        out_t,
        embed_t.as_immut(),
        prev_t.as_immut(),
        ew_t.as_immut(),
        hw_t.as_immut(),
        eps32,
        Int32(num_tokens),
        grid_dim=(num_tokens, 1, 1),
        block_dim=(block_threads, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_copy(out_h, out_d)
    ctx.synchronize()

    # The exact check above only pins the multiply ordering on channels where
    # the two orderings actually produce different bits, so count those.
    var orderings_differ = 0

    for t in range(num_tokens):
        for c in range(hidden_size):
            var want_e = _reference_row[hidden_size](
                embed_h, ew_h, t, c, eps, False
            )
            var got_e = Float64(out_h[t * 2 * hidden_size + c].cast[.float32]())
            assert_equal(
                got_e,
                want_e,
                String("embed half, token ", t, " channel ", c, " mismatch"),
            )

            var want_h = _reference_row[hidden_size](
                prev_h, hw_h, t, c, eps, False
            )
            var got_h = Float64(
                out_h[t * 2 * hidden_size + hidden_size + c].cast[.float32]()
            )
            assert_equal(
                got_h,
                want_h,
                String("hidden half, token ", t, " channel ", c, " mismatch"),
            )

            var other = _reference_row[hidden_size](
                embed_h, ew_h, t, c, eps, True
            )
            if other != want_e:
                orderings_differ += 1

    assert_true(
        orderings_differ > 0,
        (
            "the two multiply orderings agreed everywhere on this data, so"
            " matching one of them says nothing -- pick inputs that separate"
            " them"
        ),
    )
    print(
        "  matched the cast-before-multiply ordering on",
        num_tokens * 2 * hidden_size,
        "channels;",
        orderings_differ,
        "would have differed under the other ordering",
    )

    _ = embed_d
    _ = prev_d
    _ = ew_d
    _ = hw_d
    _ = out_d


def main() raises:
    with DeviceContext() as ctx:
        # GLM-5.3-Flash / DeepSeek-V3.2 draft width.
        test_mtp_eh_norm[hidden_size=4096, block_threads=256](
            num_tokens=3, ctx=ctx
        )
        # 384 is not a multiple of 256: the last strided pass is ragged.
        test_mtp_eh_norm[hidden_size=384, block_threads=256](
            num_tokens=2, ctx=ctx
        )
        # Three warps, stated in `WARP_SIZE` so it stays three on every
        # target. `block_reduce_dual_sum` reduces over a power-of-two lane
        # group, so a non-power-of-two warp count can drop a warp's sum. The
        # other shapes here all have power-of-two warp counts.
        test_mtp_eh_norm[hidden_size=384, block_threads=3 * WARP_SIZE](
            num_tokens=2, ctx=ctx
        )
        # Five and seven warps, for the same reason.
        test_mtp_eh_norm[hidden_size=1024, block_threads=5 * WARP_SIZE](
            num_tokens=2, ctx=ctx
        )
        test_mtp_eh_norm[hidden_size=2048, block_threads=7 * WARP_SIZE](
            num_tokens=2, ctx=ctx
        )
        # One warp per block: the cross-warp path is skipped.
        test_mtp_eh_norm[hidden_size=512, block_threads=WARP_SIZE](
            num_tokens=2, ctx=ctx
        )

        print("\nAll tests passed!")
