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
"""Kernel-level perf benchmark: MXFP8 attention output projection on CDNA4.

Bare `block_scaled_matmul_amd` counterpart to the fused QKV bench. Default N/K
is MiniMax-M3 o_proj at TP=4; M is a runtime arg. Cache-busts operands and scales.
"""

from std.random import seed
from std.sys import get_defined_int, size_of
from std.utils import IndexList

from max.benchmark import bencher_iter_custom
from max.gpu.host import DeviceContext
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)

from internal_utils._cache_busting import CacheBustingBuffer
from internal_utils._utils import InitializationType, arg_parse
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout.tile_tensor import lt_to_tt
from linalg.matmul.gpu.amd import block_scaled_matmul_amd

comptime OPERAND_DTYPE = DType.float8_e4m3fn
comptime OUT_DTYPE = DType.bfloat16
comptime SCALE_DTYPE = DType.float8_e8m0fnu
comptime SF_VECTOR_SIZE = 32
comptime LANE_BYTES = 32

# M3 per-device (TP=4) o_proj: [M, 2048] x [6144, 2048]^T -> [M, 6144].
comptime N = get_defined_int["N", 6144]()
comptime K = get_defined_int["K", 2048]()
comptime K_SCALES = K // SF_VECTOR_SIZE

comptime A_LAYOUT = Layout.row_major(UNKNOWN_VALUE, K)
comptime ASF_LAYOUT = Layout.row_major(UNKNOWN_VALUE, K_SCALES)
comptime B_LAYOUT = Layout.row_major(N, K)
comptime BSF_LAYOUT = Layout.row_major(N, K_SCALES)
comptime C_LAYOUT = Layout.row_major(UNKNOWN_VALUE, N)


def bench_shape(ctx: DeviceContext, mut m: Bench, M: Int) raises:
    """Build cache-busted operands for `M` rows and register the entry."""
    comptime assert (
        K % SF_VECTOR_SIZE == 0
    ), "K must be a multiple of the MXFP8 scale vector"

    comptime simd_size = 4
    var cb_a = CacheBustingBuffer[OPERAND_DTYPE](M * K, simd_size, ctx)
    var cb_b = CacheBustingBuffer[OPERAND_DTYPE](N * K, simd_size, ctx)
    # E8M0 has no zero encoding, so CacheBustingBuffer of it trips APFloat;
    # hold scale bytes as uint8 and reinterpret at the tensor seam.
    var cb_asf = CacheBustingBuffer[DType.uint8](M * K_SCALES, simd_size, ctx)
    var cb_bsf = CacheBustingBuffer[DType.uint8](N * K_SCALES, simd_size, ctx)
    cb_a.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_b.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_asf.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_bsf.init_on_device(InitializationType.uniform_distribution, ctx)

    var c_dev = ctx.enqueue_create_buffer[OUT_DTYPE](M * N)
    var c = LayoutTensor[OUT_DTYPE, C_LAYOUT](
        c_dev.unsafe_ptr(),
        RuntimeLayout[C_LAYOUT].row_major(IndexList[2](M, N)),
    )

    @always_inline
    def launch(
        ctx: DeviceContext, iteration: Int
    ) raises {mut cb_a, mut cb_b, mut cb_asf, mut cb_bsf, mut c, imm}:
        var a = LayoutTensor[mut=False, OPERAND_DTYPE, A_LAYOUT](
            cb_a.offset_ptr(iteration),
            RuntimeLayout[A_LAYOUT].row_major(IndexList[2](M, K)),
        )
        var asf = LayoutTensor[mut=False, SCALE_DTYPE, ASF_LAYOUT](
            cb_asf.offset_ptr(iteration).bitcast[Scalar[SCALE_DTYPE]](),
            RuntimeLayout[ASF_LAYOUT].row_major(IndexList[2](M, K_SCALES)),
        )
        var b = LayoutTensor[mut=False, OPERAND_DTYPE, B_LAYOUT](
            cb_b.offset_ptr(iteration),
            RuntimeLayout[B_LAYOUT].row_major(IndexList[2](N, K)),
        )
        var bsf = LayoutTensor[mut=False, SCALE_DTYPE, BSF_LAYOUT](
            cb_bsf.offset_ptr(iteration).bitcast[Scalar[SCALE_DTYPE]](),
            RuntimeLayout[BSF_LAYOUT].row_major(IndexList[2](N, K_SCALES)),
        )

        block_scaled_matmul_amd[lane_bytes=LANE_BYTES](
            lt_to_tt(c),
            lt_to_tt(a).bitcast[DType.uint8](),
            lt_to_tt(b).bitcast[DType.uint8](),
            lt_to_tt(asf),
            lt_to_tt(bsf),
            ctx,
        )

    @always_inline
    def o_proj_bench(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, launch, ctx)

    var flops = 2 * M * N * K
    var bytes = (
        M * K
        + N * K
        + M * K_SCALES
        + N * K_SCALES
        + M * N * size_of[OUT_DTYPE]()
    )

    m.bench_function(
        o_proj_bench,
        BenchId(
            "o_proj mxfp8 M="
            + String(M)
            + " N="
            + String(N)
            + " K="
            + String(K)
        ),
        [
            ThroughputMeasure(BenchMetric.flops, flops),
            ThroughputMeasure(BenchMetric.bytes, bytes),
        ],
    )

    _ = cb_a^
    _ = cb_b^
    _ = cb_asf^
    _ = cb_bsf^
    _ = c_dev^


def main() raises:
    # M is the only runtime knob; N/K are comptime. Default is the 8k CE chunk.
    var M = Int(arg_parse("M", 8192))

    seed(0)
    var m = Bench()
    with DeviceContext() as ctx:
        bench_shape(ctx, m, M)
    m.dump_report()
