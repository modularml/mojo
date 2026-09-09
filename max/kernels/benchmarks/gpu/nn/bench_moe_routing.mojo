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

from std.random import rand
from std.sys import get_defined_int

from max.benchmark import bencher_iter_custom
from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from max.gpu.host import DeviceContext

from layout import TileTensor, row_major
from nn.moe import (
    eplb_remap,
    moe_create_indices,
    router_group_limited,
    single_group_router,
    single_group_router_eplb,
)


def bench_single_group_router_eplb[
    num_tokens: Int,
    n_routed_experts: Int,
    n_experts_per_tok: Int,
    max_replicas: Int,
    num_layers: Int,
    hash_decorrelate: Bool,
](ctx: DeviceContext, mut b: Bench) raises:
    comptime dtype = DType.float32
    comptime num_log = n_routed_experts
    var scores_h = List(
        length=num_tokens * n_routed_experts, fill=Scalar[dtype](0)
    )
    var bias_h = List(length=n_routed_experts, fill=Scalar[dtype](0))
    rand[dtype](scores_h)
    rand[dtype](bias_h)
    var logcnt_h = List(length=num_layers * num_log, fill=Int32(0))
    var log2phy_h = List(
        length=num_layers * num_log * max_replicas, fill=Int32(0)
    )
    for L in range(num_layers):
        for log in range(num_log):
            var cnt = 2 if (log & 1) == 0 else 1
            logcnt_h[L * num_log + log] = Int32(cnt)
            for r in range(max_replicas):
                log2phy_h[(L * num_log + log) * max_replicas + r] = Int32(
                    (L * num_log + log + r) % (num_log * max_replicas)
                )
    var scores_d = ctx.enqueue_create_buffer[dtype](
        num_tokens * n_routed_experts
    )
    var bias_d = ctx.enqueue_create_buffer[dtype](n_routed_experts)
    var phy_d = ctx.enqueue_create_buffer[.int32](
        num_tokens * n_experts_per_tok
    )
    var log_d = ctx.enqueue_create_buffer[.int32](
        num_tokens * n_experts_per_tok
    )
    var w_d = ctx.enqueue_create_buffer[dtype](num_tokens * n_experts_per_tok)
    var logcnt_d = ctx.enqueue_create_buffer[.int32](num_layers * num_log)
    var log2phy_d = ctx.enqueue_create_buffer[.int32](
        num_layers * num_log * max_replicas
    )
    var layer_idx_d = ctx.enqueue_create_buffer[.int32](1)
    ctx.enqueue_copy(scores_d, scores_h)
    ctx.enqueue_copy(bias_d, bias_h)
    ctx.enqueue_copy(logcnt_d, logcnt_h)
    ctx.enqueue_copy(log2phy_d, log2phy_h)
    var layer_idx_h = ctx.enqueue_create_host_buffer[.int32](1)
    layer_idx_h[0] = Int32(num_layers // 2)
    ctx.enqueue_copy(layer_idx_d, layer_idx_h)
    var expert_indices = TileTensor(
        phy_d, row_major[num_tokens, n_experts_per_tok]()
    )
    var expert_indices_log = TileTensor(
        log_d, row_major[num_tokens, n_experts_per_tok]()
    )
    var expert_weights = TileTensor(
        w_d, row_major[num_tokens, n_experts_per_tok]()
    )
    var expert_scores = TileTensor(
        scores_d, row_major[num_tokens, n_routed_experts]()
    )
    var expert_bias = TileTensor(bias_d, row_major[n_routed_experts]())
    var logcnt = TileTensor(logcnt_d, row_major[num_layers, num_log]())
    var log2phy = TileTensor(
        log2phy_d, row_major[num_layers, num_log, max_replicas]()
    )
    var layer_idx = TileTensor(layer_idx_d, row_major[1]())
    var routed_scaling_factor = Float32(1.0)

    @always_inline
    def bench_fn(
        mut b: Bencher,
    ) raises {
        var expert_indices,
        var expert_indices_log,
        var expert_weights,
        var expert_scores,
        var expert_bias,
        var logcnt,
        var log2phy,
        var layer_idx,
        var routed_scaling_factor,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            single_group_router_eplb[
                scores_type=dtype,
                bias_type=dtype,
                n_routed_experts=n_routed_experts,
                n_experts_per_tok=n_experts_per_tok,
                norm_weights=True,
                num_log=num_log,
                max_replicas=max_replicas,
                hash_decorrelate=hash_decorrelate,
                target="gpu",
            ](
                expert_indices,
                expert_indices_log,
                expert_weights,
                expert_scores.as_immut(),
                expert_bias.as_immut(),
                logcnt.as_immut(),
                log2phy.as_immut(),
                layer_idx.as_immut(),
                routed_scaling_factor,
                ctx,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    b.bench_function(
        bench_fn,
        BenchId(
            "single_group_router_eplb",
            input_id=String(
                num_tokens,
                "tok/",
                n_routed_experts,
                "exp/",
                n_experts_per_tok,
                "per_tok/r",
                max_replicas,
                "/h",
                Int(hash_decorrelate),
            ),
        ),
    )
    ctx.synchronize()
    _ = scores_d
    _ = bias_d
    _ = phy_d
    _ = log_d
    _ = w_d
    _ = logcnt_d
    _ = log2phy_d
    _ = layer_idx_d
    _ = layer_idx_h
    _ = scores_h^
    _ = bias_h^
    _ = logcnt_h^
    _ = log2phy_h^


def bench_moe_create_indices[
    num_tokens: Int, num_experts: Int
](ctx: DeviceContext, mut b: Bench) raises:
    var topk_h = ctx.enqueue_create_host_buffer[.uint32](num_tokens)
    for i in range(num_tokens):
        topk_h[i] = UInt32(i % num_experts)

    var topk_d = ctx.enqueue_create_buffer[.uint32](num_tokens)
    ctx.enqueue_copy[.uint32](topk_d.unsafe_ptr(), topk_h)

    var token_expert_order_d = ctx.enqueue_create_buffer[.uint32](num_tokens)
    var expert_start_indices_d = ctx.enqueue_create_buffer[.uint32](
        num_experts + 1
    )
    var restore_token_order_d = ctx.enqueue_create_buffer[.uint32](num_tokens)
    var expert_ids_d = ctx.enqueue_create_buffer[.int32](num_experts)
    var expert_usage_stats_d = ctx.enqueue_create_buffer[.uint32](2)

    var token_expert_order = TileTensor(
        token_expert_order_d, row_major[num_tokens]()
    )
    var expert_start_indices = TileTensor(
        expert_start_indices_d, row_major[num_experts + 1]()
    )
    var restore_token_order = TileTensor(
        restore_token_order_d, row_major[num_tokens]()
    )
    var expert_ids = TileTensor(expert_ids_d, row_major[num_experts]())
    var expert_usage_stats = TileTensor(expert_usage_stats_d, row_major[2]())
    var topk_ids = TileTensor(topk_d, row_major[num_tokens]())

    @always_inline
    def bench_fn(
        mut b: Bencher,
    ) raises {
        var token_expert_order,
        var expert_start_indices,
        var restore_token_order,
        var expert_ids,
        var expert_usage_stats,
        var topk_ids,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            moe_create_indices[input_type=DType.uint32, target="gpu"](
                token_expert_order,
                expert_start_indices,
                restore_token_order,
                expert_ids,
                expert_usage_stats,
                topk_ids,
                ctx,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    b.bench_function(
        bench_fn,
        BenchId(
            "moe_create_indices",
            input_id=String(num_tokens, "tok/", num_experts, "exp"),
        ),
    )

    ctx.synchronize()

    _ = topk_d
    _ = topk_h
    _ = token_expert_order_d
    _ = expert_start_indices_d
    _ = restore_token_order_d
    _ = expert_ids_d
    _ = expert_usage_stats_d


def bench_router_group_limited[
    num_tokens: Int,
    n_routed_experts: Int,
    n_experts_per_tok: Int,
    n_groups: Int,
    topk_group: Int,
](ctx: DeviceContext, mut b: Bench) raises:
    comptime dtype = DType.float32

    var scores_h = List(
        length=num_tokens * n_routed_experts, fill=Scalar[dtype](0)
    )
    var bias_h = List(length=n_routed_experts, fill=Scalar[dtype](0))
    rand[dtype](scores_h)
    rand[dtype](bias_h)

    var scores_d = ctx.enqueue_create_buffer[dtype](
        num_tokens * n_routed_experts
    )
    var bias_d = ctx.enqueue_create_buffer[dtype](n_routed_experts)
    var indices_d = ctx.enqueue_create_buffer[.int32](
        num_tokens * n_experts_per_tok
    )
    var weights_d = ctx.enqueue_create_buffer[dtype](
        num_tokens * n_experts_per_tok
    )

    ctx.enqueue_copy(scores_d, scores_h)
    ctx.enqueue_copy(bias_d, bias_h)

    var expert_indices = TileTensor(
        indices_d,
        row_major[num_tokens, n_experts_per_tok](),
    )
    var expert_weights = TileTensor(
        weights_d,
        row_major[num_tokens, n_experts_per_tok](),
    )
    var expert_scores = TileTensor(
        scores_d,
        row_major[num_tokens, n_routed_experts](),
    )
    var expert_bias = TileTensor(bias_d, row_major[n_routed_experts]())
    var routed_scaling_factor = Float32(1.0)

    @always_inline
    def bench_fn(
        mut b: Bencher,
    ) raises {
        var expert_indices,
        var expert_weights,
        var expert_scores,
        var expert_bias,
        var routed_scaling_factor,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            router_group_limited[
                scores_type=dtype,
                bias_type=dtype,
                n_routed_experts=n_routed_experts,
                n_experts_per_tok=n_experts_per_tok,
                n_groups=n_groups,
                topk_group=topk_group,
                norm_weights=True,
                target="gpu",
            ](
                expert_indices,
                expert_weights,
                expert_scores.as_immut(),
                expert_bias.as_immut(),
                routed_scaling_factor,
                ctx,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    b.bench_function(
        bench_fn,
        BenchId(
            "router_group_limited",
            input_id=String(
                num_tokens,
                "tok/",
                n_routed_experts,
                "exp/",
                n_experts_per_tok,
                "per_tok",
            ),
        ),
    )

    ctx.synchronize()

    _ = scores_d
    _ = bias_d
    _ = indices_d
    _ = weights_d
    _ = bias_h^
    _ = scores_h^


def bench_single_group_router[
    num_tokens: Int,
    n_routed_experts: Int,
    n_experts_per_tok: Int,
](ctx: DeviceContext, mut b: Bench) raises:
    comptime dtype = DType.float32

    var scores_h = List(
        length=num_tokens * n_routed_experts, fill=Scalar[dtype](0)
    )
    var bias_h = List(length=n_routed_experts, fill=Scalar[dtype](0))
    rand[dtype](scores_h)
    rand[dtype](bias_h)

    var scores_d = ctx.enqueue_create_buffer[dtype](
        num_tokens * n_routed_experts
    )
    var bias_d = ctx.enqueue_create_buffer[dtype](n_routed_experts)
    var indices_d = ctx.enqueue_create_buffer[.int32](
        num_tokens * n_experts_per_tok
    )
    var weights_d = ctx.enqueue_create_buffer[dtype](
        num_tokens * n_experts_per_tok
    )

    ctx.enqueue_copy(scores_d, scores_h)
    ctx.enqueue_copy(bias_d, bias_h)

    var expert_indices = TileTensor(
        indices_d, row_major[num_tokens, n_experts_per_tok]()
    )
    var expert_weights = TileTensor(
        weights_d, row_major[num_tokens, n_experts_per_tok]()
    )
    var expert_scores = TileTensor(
        scores_d, row_major[num_tokens, n_routed_experts]()
    )
    var expert_bias = TileTensor(bias_d, row_major[n_routed_experts]())
    var routed_scaling_factor = Float32(1.0)

    @always_inline
    def bench_fn(
        mut b: Bencher,
    ) raises {
        var expert_indices,
        var expert_weights,
        var expert_scores,
        var expert_bias,
        var routed_scaling_factor,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            single_group_router[
                scores_type=dtype,
                bias_type=dtype,
                n_routed_experts=n_routed_experts,
                n_experts_per_tok=n_experts_per_tok,
                norm_weights=True,
                target="gpu",
            ](
                expert_indices,
                expert_weights,
                expert_scores.as_immut(),
                expert_bias.as_immut(),
                routed_scaling_factor,
                ctx,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    b.bench_function(
        bench_fn,
        BenchId(
            "single_group_router",
            input_id=String(
                num_tokens,
                "tok/",
                n_routed_experts,
                "exp/",
                n_experts_per_tok,
                "per_tok",
            ),
        ),
    )

    ctx.synchronize()

    _ = scores_d
    _ = bias_d
    _ = indices_d
    _ = weights_d
    _ = bias_h^
    _ = scores_h^


def bench_eplb_remap[
    num_tokens: Int,
    num_log: Int,
    K: Int,
    max_replicas: Int,
    num_layers: Int,
    hash_decorrelate: Bool,
](ctx: DeviceContext, mut b: Bench) raises:
    # Host buffers ---------------------------------------------------------
    var router_h = List(length=num_tokens * K, fill=Int32(0))
    var logcnt_h = List(length=num_layers * num_log, fill=Int32(0))
    var log2phy_h = List(
        length=num_layers * num_log * max_replicas, fill=Int32(0)
    )
    # Fill router_idx with a uniform random logical id in [0, num_log).
    for i in range(num_tokens * K):
        router_h[i] = Int32(i % num_log)
    # Realistic logcnt: half the experts have 2 replicas, rest have 1.
    for L in range(num_layers):
        for log in range(num_log):
            var cnt = 2 if (log & 1) == 0 else 1
            logcnt_h[L * num_log + log] = Int32(cnt)
            for r in range(max_replicas):
                log2phy_h[(L * num_log + log) * max_replicas + r] = Int32(
                    (L * num_log + log + r) % (num_log * max_replicas)
                )

    # Device buffers + TileTensors ----------------------------------------
    var router_d = ctx.enqueue_create_buffer[.int32](num_tokens * K)
    var phy_d = ctx.enqueue_create_buffer[.int32](num_tokens * K)
    var logcnt_d = ctx.enqueue_create_buffer[.int32](num_layers * num_log)
    var log2phy_d = ctx.enqueue_create_buffer[.int32](
        num_layers * num_log * max_replicas
    )
    var layer_idx_d = ctx.enqueue_create_buffer[.int32](1)

    ctx.enqueue_copy(router_d, router_h)
    ctx.enqueue_copy(logcnt_d, logcnt_h)
    ctx.enqueue_copy(log2phy_d, log2phy_h)

    # Pick a middle layer for the bench.
    var layer_idx_h = ctx.enqueue_create_host_buffer[.int32](1)
    layer_idx_h[0] = Int32(num_layers // 2)
    ctx.enqueue_copy(layer_idx_d, layer_idx_h)

    var phy = TileTensor(phy_d, row_major[num_tokens, K]())
    var router_idx = TileTensor(router_d, row_major[num_tokens, K]())
    var logcnt = TileTensor(logcnt_d, row_major[num_layers, num_log]())
    var log2phy = TileTensor(
        log2phy_d, row_major[num_layers, num_log, max_replicas]()
    )
    var layer_idx = TileTensor(layer_idx_d, row_major[1]())

    @always_inline
    def bench_fn(
        mut b: Bencher,
    ) raises {
        var phy,
        var router_idx,
        var logcnt,
        var log2phy,
        var layer_idx,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            eplb_remap[
                num_log=num_log,
                max_replicas=max_replicas,
                K=K,
                hash_decorrelate=hash_decorrelate,
                target="gpu",
            ](
                phy,
                router_idx.as_immut(),
                logcnt.as_immut(),
                log2phy.as_immut(),
                layer_idx.as_immut(),
                ctx,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    b.bench_function(
        bench_fn,
        BenchId(
            "eplb_remap",
            input_id=String(
                num_tokens,
                "tok/",
                num_log,
                "exp/",
                K,
                "topk/",
                max_replicas,
                "rep/h",
                Int(hash_decorrelate),
            ),
        ),
    )

    ctx.synchronize()
    _ = router_d
    _ = phy_d
    _ = logcnt_d
    _ = log2phy_d
    _ = layer_idx_d
    _ = layer_idx_h
    _ = router_h^
    _ = logcnt_h^
    _ = log2phy_h^


def main() raises:
    comptime num_tokens = get_defined_int["num_tokens", 4096]()
    comptime num_experts = get_defined_int["num_experts", 256]()
    comptime n_experts_per_tok = get_defined_int["n_experts_per_tok", 8]()
    comptime n_groups = get_defined_int["n_groups", 8]()
    comptime topk_group = get_defined_int["topk_group", 4]()

    # new params for single-group router (Kimi K2.5 defaults)
    comptime sg_n_experts = get_defined_int["sg_n_experts", 384]()
    comptime sg_n_experts_per_tok = get_defined_int["sg_n_experts_per_tok", 8]()

    var m = Bench(
        BenchConfig(
            num_repetitions=1,
            max_iters=10000,
        )
    )
    with DeviceContext() as ctx:
        bench_moe_create_indices[num_tokens, num_experts](ctx, m)
        bench_router_group_limited[
            num_tokens, num_experts, n_experts_per_tok, n_groups, topk_group
        ](ctx, m)

        bench_single_group_router[
            num_tokens, sg_n_experts, sg_n_experts_per_tok
        ](ctx, m)

        bench_eplb_remap[
            num_tokens=num_tokens,
            num_log=num_experts,
            K=n_experts_per_tok,
            max_replicas=4,
            num_layers=58,
            hash_decorrelate=False,
        ](ctx, m)

        bench_eplb_remap[
            num_tokens=num_tokens,
            num_log=num_experts,
            K=n_experts_per_tok,
            max_replicas=4,
            num_layers=58,
            hash_decorrelate=True,
        ](ctx, m)

        bench_single_group_router_eplb[
            num_tokens=num_tokens,
            n_routed_experts=sg_n_experts,
            n_experts_per_tok=sg_n_experts_per_tok,
            max_replicas=4,
            num_layers=58,
            hash_decorrelate=False,
        ](ctx, m)

    m.dump_report()
