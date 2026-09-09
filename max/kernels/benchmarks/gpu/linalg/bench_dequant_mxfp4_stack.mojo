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
# Standalone bench: MXFP4-packed weight stack -> bf16 dequant, at MoE
# expert-stack scale. Measures the standalone dequant-to-bf16 step that
# W4A8 eliminates (config C's dequant path), driven the same way the
# in-tree AMD dequant+matmul bench drives `dequant_mxfp4`
# (bench_matmul_dequant_mxfp4_amd.mojo), but on its own and at
# num_rows = num_experts * N to cover a whole (or partial) expert stack.

from std.math import ceildiv
from std.sys import get_defined_int, size_of
from std.memory import bitcast
from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from internal_utils._utils import InitializationType, init_vector_launch
from layout import Idx, TileTensor, row_major
from linalg.mxfp4_dequant import dequant_mxfp4


def bench_dequant_stack[
    N: Int, K: Int, num_rows_total: Int, out_type: DType = .bfloat16
](ctx: DeviceContext, mut b: Bench) raises:
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)

    var b_packed_device = ctx.enqueue_create_buffer[.uint8](
        num_rows_total * packed_K
    )
    var b_scales_device = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        num_rows_total * scale_K
    )
    var b_out_device = ctx.enqueue_create_buffer[out_type](num_rows_total * K)

    init_vector_launch[.uint8](
        b_packed_device,
        num_rows_total * packed_K,
        InitializationType.uniform_distribution,
        ctx,
    )
    var bs_hbuf = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        num_rows_total * scale_K
    )
    for i in range(num_rows_total * scale_K):
        bs_hbuf[i] = bitcast[.float8_e8m0fnu](UInt8(127))
    ctx.enqueue_copy(b_scales_device, bs_hbuf)
    ctx.synchronize()

    var b_packed_tt = TileTensor(
        b_packed_device, row_major((num_rows_total, Idx[packed_K]))
    )
    var b_scales_tt = TileTensor(
        b_scales_device, row_major((num_rows_total, Idx[scale_K]))
    )
    var b_out_tt = TileTensor(b_out_device, row_major((num_rows_total, Idx[K])))

    @always_inline
    def kernel_launch(
        ctx: DeviceContext, iteration: Int
    ) raises {mut b_out_tt, imm}:
        dequant_mxfp4(
            ctx,
            b_out_tt,
            b_packed_tt,
            b_scales_tt,
            num_rows=num_rows_total,
            num_cols=K,
        )

    @always_inline
    def bench_func(mut bencher: Bencher) raises {imm}:
        bencher_iter_custom(bencher, kernel_launch, ctx)

    comptime total_bytes = (
        num_rows_total * packed_K
        + num_rows_total * scale_K
        + num_rows_total * K * size_of[out_type]()
    )
    var bandwidth = ThroughputMeasure(BenchMetric.bytes, total_bytes)

    b.bench_function(
        bench_func,
        BenchId(
            String(
                "dequant_mxfp4_to_bf16(N=",
                N,
                ",K=",
                K,
                ",rows=",
                num_rows_total,
                ")",
            )
        ),
        [bandwidth],
    )

    _ = b_packed_device^
    _ = b_scales_device^
    _ = b_out_device^
    _ = bs_hbuf^


def main() raises:
    comptime N = get_defined_int["N", 3072]()
    comptime K = get_defined_int["K", 3584]()
    comptime num_experts = get_defined_int["num_experts", 896]()
    comptime num_rows_total = N * num_experts

    var b = Bench()
    with DeviceContext() as ctx:
        bench_dequant_stack[N, K, num_rows_total](ctx, b)
    b.dump_report()
