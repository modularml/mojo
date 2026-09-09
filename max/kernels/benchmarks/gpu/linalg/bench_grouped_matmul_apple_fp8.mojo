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
"""Weight-read-bandwidth microbench for the Apple M5 grouped (MoE) W8A16 matmul.

Times the tiled grouped FP8 kernel (`enqueue_grouped_matmul2d_fp8`) against the
`naive_grouped_matmul` kernel the Nemotron-3-Nano-30B-A3B MoE runs today -- with
BOTH a bf16 weight stack (the bf16 checkpoint's decode baseline) and an fp8
weight stack (the FP8 checkpoint's slow scalar-cast path this replaces) -- at the
real 30B MoE expert dims, across the batch-1 and c32 concurrent-decode regimes.

Each active expert reads its full `[N, K]` weight slab once, so the aggregate
weight traffic is `num_active * N * K` bytes (fp8) or `2 * num_active * N * K`
(bf16); the reported aggregate GB/s uses that denominator. The two ratios answer
the shipping questions directly:

- `tiled-fp8 / naive-bf16`: does the FP8 checkpoint's grouped matmul beat the
  bf16 checkpoint's (the decode gate: FP8 >= bf16)?
- `tiled-fp8 / naive-fp8`: how much faster is the tiled kernel than the naive
  scalar-cast path it replaces?

Timing mirrors `bench_apple_fp8_gemv.mojo`: a per-iter `synchronize` in warmup
(so the GPU clocks ramp) and a single batched `synchronize` for the hot loop, so
the fixed per-launch overhead is amortized and the ratio is a conservative lower
bound on the pure kernel delta.
"""

from max.gpu.host import DeviceContext
from std.time import perf_counter

from layout import Coord, Idx, TileTensor, row_major

from linalg.grouped_matmul import naive_grouped_matmul
from linalg.matmul.gpu.apple.matmul2d_fp8 import enqueue_grouped_matmul2d_fp8


def _bench_grouped[
    N: Int, K: Int, num_experts: Int
](
    num_active: Int,
    tokens_per_expert: Int,
    name: String,
    ctx: DeviceContext,
    warmup: Int = 25,
    hot: Int = 50,
) raises:
    """Time naive-bf16 / naive-fp8 / tiled-fp8 grouped matmul at `[N, K]`.

    `num_active` distinct experts (ids `0..num_active-1`), each routed
    `tokens_per_expert` tokens. Timing is data-independent (no value branches),
    so the fill is arbitrary.
    """
    var total_M = num_active * tokens_per_expert

    var act_host = ctx.enqueue_create_host_buffer[.bfloat16](total_M * K)
    var b_fp8_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        num_experts * N * K
    )
    var b_bf16_host = ctx.enqueue_create_host_buffer[.bfloat16](
        num_experts * N * K
    )
    var off_host = ctx.enqueue_create_host_buffer[.uint32](num_experts + 1)
    var eid_host = ctx.enqueue_create_host_buffer[.int32](num_experts)

    for i in range(total_M * K):
        act_host[i] = BFloat16(Float32((i % 5) - 2))
    for i in range(num_experts * N * K):
        var v = Float32((i % 7) - 3) * Float32(0.5)
        b_fp8_host[i] = v.cast[.float8_e4m3fn]()
        b_bf16_host[i] = v.cast[.bfloat16]()
    for i in range(num_experts + 1):
        off_host[i] = 0
    for i in range(num_experts):
        eid_host[i] = 0
    for i in range(num_active):
        off_host[i + 1] = off_host[i] + UInt32(tokens_per_expert)
        eid_host[i] = Int32(i)

    var act_dev = ctx.enqueue_create_buffer[.bfloat16](total_M * K)
    var b_fp8_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](
        num_experts * N * K
    )
    var b_bf16_dev = ctx.enqueue_create_buffer[.bfloat16](num_experts * N * K)
    var out_dev = ctx.enqueue_create_buffer[.float32](total_M * N)
    var off_dev_buf = ctx.enqueue_create_buffer[.uint32](num_experts + 1)
    var eid_dev_buf = ctx.enqueue_create_buffer[.int32](num_experts)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(b_fp8_dev, b_fp8_host)
    ctx.enqueue_copy(b_bf16_dev, b_bf16_host)
    ctx.enqueue_copy(off_dev_buf, off_host)
    ctx.enqueue_copy(eid_dev_buf, eid_host)

    var a_tt = TileTensor(
        act_dev.unsafe_ptr(), row_major(Coord(total_M, Idx[K]))
    )
    var b_fp8_tt = TileTensor(
        b_fp8_dev.unsafe_ptr(), row_major[num_experts, N, K]()
    )
    var b_bf16_tt = TileTensor(
        b_bf16_dev.unsafe_ptr(), row_major[num_experts, N, K]()
    )
    var out_tt = TileTensor(
        out_dev.unsafe_ptr(), row_major(Coord(total_M, Idx[N]))
    )
    var off_tt = TileTensor(
        off_dev_buf.unsafe_ptr(), row_major(Coord(num_experts + 1))
    )
    var eid_tt = TileTensor(
        eid_dev_buf.unsafe_ptr(), row_major(Coord(Idx[num_experts]))
    )
    ctx.synchronize()

    # --- naive grouped, bf16 weight (2 bytes/weight): the bf16 decode baseline.
    for _ in range(warmup):
        naive_grouped_matmul(
            out_tt,
            a_tt,
            b_bf16_tt,
            off_tt,
            eid_tt,
            tokens_per_expert,
            num_active,
            ctx,
        )
        ctx.synchronize()
    var t_nb = perf_counter()
    for _ in range(hot):
        naive_grouped_matmul(
            out_tt,
            a_tt,
            b_bf16_tt,
            off_tt,
            eid_tt,
            tokens_per_expert,
            num_active,
            ctx,
        )
    ctx.synchronize()
    var nb_sec = (perf_counter() - t_nb) / Float64(hot)

    # --- naive grouped, fp8 weight (1 byte/weight): the scalar-cast path today.
    for _ in range(warmup):
        naive_grouped_matmul(
            out_tt,
            a_tt,
            b_fp8_tt,
            off_tt,
            eid_tt,
            tokens_per_expert,
            num_active,
            ctx,
        )
        ctx.synchronize()
    var t_nf = perf_counter()
    for _ in range(hot):
        naive_grouped_matmul(
            out_tt,
            a_tt,
            b_fp8_tt,
            off_tt,
            eid_tt,
            tokens_per_expert,
            num_active,
            ctx,
        )
    ctx.synchronize()
    var nf_sec = (perf_counter() - t_nf) / Float64(hot)

    # --- tiled grouped fp8 (direct DRAM fp8 B feed, native Apple MMA). ---
    for _ in range(warmup):
        enqueue_grouped_matmul2d_fp8[c_type=DType.float32](
            out_tt,
            a_tt,
            b_fp8_tt,
            off_tt,
            eid_tt,
            tokens_per_expert,
            num_active,
            ctx,
        )
        ctx.synchronize()
    var t_tf = perf_counter()
    for _ in range(hot):
        enqueue_grouped_matmul2d_fp8[c_type=DType.float32](
            out_tt,
            a_tt,
            b_fp8_tt,
            off_tt,
            eid_tt,
            tokens_per_expert,
            num_active,
            ctx,
        )
    ctx.synchronize()
    var tf_sec = (perf_counter() - t_tf) / Float64(hot)

    var wbytes = Float64(num_active) * Float64(N) * Float64(K)  # fp8: 1 B/wt
    print(
        "  ",
        name,
        " N=",
        N,
        " K=",
        K,
        " active=",
        num_active,
        " tok/exp=",
        tokens_per_expert,
        ":\n     naive-bf16 ",
        nb_sec * 1e6,
        "us (",
        (2.0 * wbytes) / (nb_sec * 1e9),
        "GB/s) | naive-fp8 ",
        nf_sec * 1e6,
        "us (",
        wbytes / (nf_sec * 1e9),
        "GB/s) | tiled-fp8 ",
        tf_sec * 1e6,
        "us (",
        wbytes / (tf_sec * 1e9),
        "GB/s)\n     tiled-fp8 vs naive-bf16: ",
        nb_sec / tf_sec,
        "x | tiled-fp8 vs naive-fp8: ",
        nf_sec / tf_sec,
        "x",
    )

    _ = act_dev^
    _ = b_fp8_dev^
    _ = b_bf16_dev^
    _ = out_dev^
    _ = off_dev_buf^
    _ = eid_dev_buf^


def main() raises:
    with DeviceContext() as ctx:
        if ctx.compute_capability() != 5:
            print("skip: grouped W8A16 tiled matmul bench requires Apple M5")
            return

        # Nemotron-3-Nano-30B-A3B MoE: 128 experts, top-6, hidden=2688,
        # moe_intermediate=1856. up-proj weight [E, 1856, 2688] (N=1856,
        # K=2688); down-proj [E, 2688, 1856] (N=2688, K=1856).
        print("== up-proj (N=1856, K=2688) ==")
        # batch-1 decode: 1 token, top-6 -> 6 active experts, 1 token each.
        _bench_grouped[N=1856, K=2688, num_experts=8](6, 1, "up batch-1", ctx)
        # c32 concurrent decode: 32 tokens x top-6 = 192 assignments; model as
        # 32 active experts x 6 tokens (dense) and 64 x 3 (sparser tail).
        _bench_grouped[N=1856, K=2688, num_experts=32](32, 6, "up c32-d", ctx)
        _bench_grouped[N=1856, K=2688, num_experts=64](64, 3, "up c32-s", ctx)

        print("== down-proj (N=2688, K=1856) ==")
        _bench_grouped[N=2688, K=1856, num_experts=8](6, 1, "down batch-1", ctx)
        _bench_grouped[N=2688, K=1856, num_experts=32](32, 6, "down c32-d", ctx)
        _bench_grouped[N=2688, K=1856, num_experts=64](64, 3, "down c32-s", ctx)
