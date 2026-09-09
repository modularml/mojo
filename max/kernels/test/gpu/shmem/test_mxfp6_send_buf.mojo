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
"""`MXTokenFormat`'s MXFP6 send-buffer packing, byte for byte.

The EP dispatch quantizer and the standalone `quantize_mxfp6_amd` encode the
same bf16 activations to the same packed FP6 + E8M0 pair, so they must agree
exactly -- both round to nearest with even-mode scales, and neither has any
freedom left once the block maximum is fixed. `quantize_mxfp6_amd` is already
pinned against the OCP tables, which makes it a reference rather than a second
opinion.

Comparing bytes rather than dequantized values is deliberate: the wire layout
is where a format that packs four codes per three bytes goes wrong, and a
mis-strided write still dequantizes to plausible-looking numbers.
"""

from max.gpu.host import DeviceContext
from max.gpu import block_dim, thread_idx
from max.gpu.host.info import MI355X
from std.random import random_float64, seed
from std.memory import bitcast
from std.testing import assert_equal, assert_true

from layout import Coord, Idx, TileTensor, row_major
from linalg.fp6_quantization import quantize_mxfp6_amd
from linalg.fp6_utils import FP6Format
from shmem.ep_comm import MXTokenFormat
from linalg.mx_format import MXFormat


comptime HID = 512
comptime BLOCK = 256
comptime QUANT_BYTES = (HID * 6) // 8
comptime SCALE_K = HID // 32


comptime L2D = type_of(row_major(Int64(1), Int64(1)))

comptime TokenFmt = MXTokenFormat[
    quant_dtype=DType.uint8,
    scales_dtype=DType.float8_e8m0fnu,
    output_layout=L2D,
    scales_layout=L2D,
    HID,
    1,
    mx_format=MXFormat.FP6_E2M3,
]


@__name("mxfp6_send_buf_probe")
def _send_buf_probe_kernel(
    buf: Pointer[UInt8, MutAnyOrigin],
    src: Pointer[BFloat16, MutAnyOrigin],
):
    """Runs one token through the dispatch packer, as `ep_dispatch` would."""
    TokenFmt.copy_token_to_send_buf[.bfloat16, BLOCK](buf, src, Float32(1.0))


def main() raises:
    seed(0)
    var ctx = DeviceContext()
    comptime assert (
        ctx.default_device_info == MI355X
    ), "test_mxfp6_send_buf requires MI355X (CDNA4)"

    print("===> MXFP6 send buffer vs quantize_mxfp6_amd")

    var src_h = ctx.enqueue_create_host_buffer[.bfloat16](HID)
    ctx.synchronize()
    for i in range(HID):
        src_h[i] = BFloat16(random_float64(-6.0, 6.0))

    var src_d = ctx.enqueue_create_buffer[.bfloat16](HID)
    ctx.enqueue_copy(src_d, src_h)

    var token_bytes = TokenFmt.token_size()
    var buf_d = ctx.enqueue_create_buffer[.uint8](token_bytes)
    buf_d.enqueue_fill(UInt8(0))

    ctx.enqueue_function[_send_buf_probe_kernel](
        buf_d.unsafe_ptr(),
        src_d.unsafe_ptr(),
        grid_dim=1,
        block_dim=BLOCK,
    )

    # Reference: the standalone quantizer, already pinned to the OCP tables.
    var ref_out_d = ctx.enqueue_create_buffer[.uint8](QUANT_BYTES)
    var ref_scales_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](SCALE_K)
    quantize_mxfp6_amd[FP6Format.E2M3](
        ctx,
        TileTensor[origin=MutAnyOrigin](
            ref_out_d, row_major(Idx[1], Idx[QUANT_BYTES])
        ),
        TileTensor[origin=MutAnyOrigin](
            ref_scales_d, row_major(Idx[1], Idx[SCALE_K])
        ),
        TileTensor[origin=MutAnyOrigin](src_d, row_major(Idx[1], Idx[HID])),
    )

    var buf_h = ctx.enqueue_create_host_buffer[.uint8](token_bytes)
    var ref_out_h = ctx.enqueue_create_host_buffer[.uint8](QUANT_BYTES)
    var ref_scales_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](SCALE_K)
    ctx.enqueue_copy(buf_h, buf_d)
    ctx.enqueue_copy(ref_out_h, ref_out_d)
    ctx.enqueue_copy(ref_scales_h, ref_scales_d)
    ctx.synchronize()

    var quant_mismatch = 0
    var first_bad = -1
    for i in range(QUANT_BYTES):
        if buf_h[i] != ref_out_h[i]:
            quant_mismatch += 1
            if first_bad < 0:
                first_bad = i
    print("  packed bytes:", QUANT_BYTES, " mismatched:", quant_mismatch)
    if first_bad >= 0:
        print(
            "  first mismatch at byte ",
            first_bad,
            ": got ",
            buf_h[first_bad],
            " want ",
            ref_out_h[first_bad],
        )

    var scale_off = TokenFmt.scales_offset()
    print("  scales_offset:", scale_off, " (quant_size)")
    var scale_mismatch = 0
    for i in range(SCALE_K):
        var got = buf_h[scale_off + i]
        var want = bitcast[.uint8, 1](ref_scales_h[i])
        if got != want:
            scale_mismatch += 1
    print("  scales:", SCALE_K, " mismatched:", scale_mismatch)

    assert_equal(quant_mismatch, 0, "packed FP6 bytes disagree")
    assert_equal(scale_mismatch, 0, "E8M0 scales disagree")
    print("==== MXFP6 send buffer OK ====")
