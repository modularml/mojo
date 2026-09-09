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

"""GPU regression test for `rope_ragged` with explicit `position_ids`.

Guards against the Metal closure-capture bug where the `position_ids`
`OptionalReg[TileTensor]` was not `@__copy_capture`'d into the device kernel.
The kernel then read a stale pointer -> every token resolved freqs row 0 ->
RoPE collapsed to the identity (output == input), scrambling FLUX latents into
noise. We run the GPU kernel with non-zero `position_ids` and an explicitly
non-identity freqs table and assert the rotated output matches a host fp32
reference (and differs from the input -- the identity-collapse signature).
"""

from std.math import cos, sin

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from nn.rope import rope_ragged
from std.testing import assert_almost_equal, assert_true

from std.utils import IndexList


def test_rope_ragged_position_ids[
    dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    comptime seq_len = 16
    comptime num_heads = 2
    comptime head_dim = 16
    comptime max_pos = 32

    comptime x_layout = row_major[seq_len, num_heads, head_dim]()
    comptime row_offsets_layout = row_major[2]()
    comptime start_pos_layout = row_major[1]()
    comptime freqs_layout = row_major[max_pos, head_dim]()
    comptime pid_layout = row_major[1, seq_len]()

    var x_dev = ctx.enqueue_create_buffer[dtype](x_layout.static_product)
    var out_dev = ctx.enqueue_create_buffer[dtype](x_layout.static_product)
    var freqs_dev = ctx.enqueue_create_buffer[dtype](
        freqs_layout.static_product
    )
    var ro_dev = ctx.enqueue_create_buffer[.uint32](2)
    var sp_dev = ctx.enqueue_create_buffer[.uint32](1)
    var pid_dev = ctx.enqueue_create_buffer[.uint32](seq_len)

    # x[token, head, dim] = deterministic small values.
    with x_dev.map_to_host() as h:
        for i in range(x_layout.static_product):
            h[i] = Scalar[dtype]((i % 7) - 3) * 0.5 + 0.1

    # Non-identity freqs: row p uses angle p*0.2 for every (re, im) pair.
    with freqs_dev.map_to_host() as h:
        for p in range(max_pos):
            var ang = Scalar[dtype](p) * 0.2
            for d in range(0, head_dim, 2):
                h[p * head_dim + d] = cos(ang)
                h[p * head_dim + d + 1] = sin(ang)

    with ro_dev.map_to_host() as h:
        h[0] = 0
        h[1] = seq_len
    with sp_dev.map_to_host() as h:
        h[0] = 0
    # position_ids: token s -> position (s + 3); even token 0 is non-identity.
    with pid_dev.map_to_host() as h:
        for s in range(seq_len):
            h[s] = UInt32(s + 3)

    var x_t = TileTensor(x_dev.unsafe_ptr(), x_layout)
    var out_t = TileTensor(out_dev.unsafe_ptr(), x_layout)
    var freqs_t = TileTensor(freqs_dev.unsafe_ptr(), freqs_layout)
    var ro_t = TileTensor(ro_dev.unsafe_ptr(), row_offsets_layout)
    var sp_t = TileTensor(sp_dev.unsafe_ptr(), start_pos_layout)
    var pid_t = TileTensor(pid_dev.unsafe_ptr(), pid_layout)

    # `out_t` must be captured by value (`{var out_t}`) into the device
    # closure: a by-ref capture leaves the kernel dereferencing the host-stack
    # tensor on the GPU (memory access fault). Same rule the kernel itself
    # follows for its captured tensors; mirrors `test_rope_ragged.mojo`'s
    # `output_fn`.
    @always_inline
    def output_fn[
        width: SIMDLength, alignment: Int
    ](idx: IndexList[3], val: SIMD[dtype, width]) {var out_t} -> None:
        out_t.store[width=width](Coord(idx), val)

    rope_ragged[
        dtype,
        dtype,
        interleaved=True,
        target=StaticString("gpu"),
    ](
        x=x_t,
        input_row_offsets=ro_t,
        start_pos=sp_t,
        freqs_cis=freqs_t,
        context=ctx,
        output_fn=output_fn,
        position_ids=pid_t.as_unsafe_any_origin().as_immut(),
    )
    ctx.synchronize()

    var total_abs_diff = Scalar[dtype](0)
    with x_dev.map_to_host() as xh, out_dev.map_to_host() as oh:
        for s in range(seq_len):
            var prow = s + 3  # position_ids[s]
            var ang = Scalar[dtype](prow) * 0.2
            var c = cos(ang)
            var sn = sin(ang)
            for hh in range(num_heads):
                var base = (s * num_heads + hh) * head_dim
                for d in range(0, head_dim, 2):
                    var re = xh[base + d]
                    var im = xh[base + d + 1]
                    var exp_re = re * c - im * sn
                    var exp_im = re * sn + im * c
                    # 1) GPU output matches the host reference rotation.
                    assert_almost_equal(
                        oh[base + d], exp_re, atol=1e-4, rtol=1e-4
                    )
                    assert_almost_equal(
                        oh[base + d + 1], exp_im, atol=1e-4, rtol=1e-4
                    )
                    total_abs_diff += abs(oh[base + d] - re)
                    total_abs_diff += abs(oh[base + d + 1] - im)

    # 2) Identity-collapse guard: non-zero position_ids + non-identity freqs
    #    must NOT leave the output equal to the input.
    assert_true(
        total_abs_diff > 1.0,
        "RoPE output equals input -> position_ids ignored (identity collapse)",
    )


def main() raises:
    with DeviceContext() as ctx:
        test_rope_ragged_position_ids[.float32](ctx)
        print("test_rope_ragged_position_ids: PASS")
