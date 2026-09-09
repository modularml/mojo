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
#
# `sink_gate_router` against an independent host reference. The kernel selects
# the top `n_experts_per_tok` routed experts by `sigmoid(logit) + bias`, then
# softmax-normalizes the log-sigmoid of those experts' raw logits together with
# the always-on sink logits. A wrong index routes a token to the wrong expert,
# which is a large discrete error rather than a rounding one, so indices are
# compared exactly and only the weights carry a tolerance.

from std.math import exp
from std.random import seed

from internal_utils import assert_almost_equal, assert_equal
from layout import Coord, Idx, TileTensor, row_major
from layout._fillers import random
from max.gpu.host import DeviceContext
from nn.moe import sink_gate_router

comptime scores_type = DType.float32
comptime bias_type = DType.float32


def _sigmoid_ref(x: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-x))


def test_sink_gate_router[
    n_routed: Int, topk: Int, n_shared: Int
](num_tokens: Int, ctx: DeviceContext) raises:
    comptime n_total = n_routed + n_shared
    comptime k_total = topk + n_shared
    comptime route_scale = 8.0
    comptime global_scale_val = 1.3

    var logits_host = ctx.enqueue_create_host_buffer[scores_type](
        num_tokens * n_total
    )
    var bias_host = ctx.enqueue_create_host_buffer[bias_type](n_routed)
    var gscale_host = ctx.enqueue_create_host_buffer[scores_type](1)

    # Continuous random values keep the selection order unambiguous: ties would
    # be broken by lower index in the kernel but are measure-zero here.
    random(
        TileTensor(logits_host, row_major(Coord(num_tokens, Idx[n_total]))),
        min=-4.0,
        max=4.0,
    )
    random(TileTensor(bias_host, row_major(Idx[n_routed])), min=-0.1, max=0.1)
    gscale_host[0] = Float32(global_scale_val)

    var logits_dev = ctx.enqueue_create_buffer[scores_type](
        num_tokens * n_total
    )
    var bias_dev = ctx.enqueue_create_buffer[bias_type](n_routed)
    var gscale_dev = ctx.enqueue_create_buffer[scores_type](1)
    var idx_dev = ctx.enqueue_create_buffer[.int32](num_tokens * topk)
    var w_dev = ctx.enqueue_create_buffer[scores_type](num_tokens * topk)
    var sink_dev = ctx.enqueue_create_buffer[scores_type](num_tokens * n_shared)
    ctx.enqueue_copy(logits_dev, logits_host)
    ctx.enqueue_copy(bias_dev, bias_host)
    ctx.enqueue_copy(gscale_dev, gscale_host)

    sink_gate_router[n_routed, topk, n_shared, "gpu"](
        TileTensor(idx_dev, row_major(Coord(num_tokens, Idx[topk]))),
        TileTensor(w_dev, row_major(Coord(num_tokens, Idx[topk]))),
        TileTensor(sink_dev, row_major(Coord(num_tokens, Idx[n_shared]))),
        TileTensor(
            logits_dev, row_major(Coord(num_tokens, Idx[n_total]))
        ).as_immut(),
        TileTensor(bias_dev, row_major(Idx[n_routed])).as_immut(),
        TileTensor(gscale_dev, row_major(Idx[1])).as_immut(),
        Float32(route_scale),
        ctx,
    )

    var idx_out = ctx.enqueue_create_host_buffer[.int32](num_tokens * topk)
    var w_out = ctx.enqueue_create_host_buffer[scores_type](num_tokens * topk)
    var sink_out = ctx.enqueue_create_host_buffer[scores_type](
        num_tokens * n_shared
    )
    ctx.enqueue_copy(idx_out, idx_dev)
    ctx.enqueue_copy(w_out, w_dev)
    ctx.enqueue_copy(sink_out, sink_dev)
    ctx.synchronize()

    var idx_ref = ctx.enqueue_create_host_buffer[.int32](num_tokens * topk)
    var w_ref = ctx.enqueue_create_host_buffer[scores_type](num_tokens * topk)
    var sink_ref = ctx.enqueue_create_host_buffer[scores_type](
        num_tokens * n_shared
    )

    for t in range(num_tokens):
        var winners = List[Int]()
        for _ in range(topk):
            var best = -1
            var best_score = Float32(0)
            for e in range(n_routed):
                var taken = False
                for w in winners:
                    if w == e:
                        taken = True
                if taken:
                    continue
                var s = (
                    _sigmoid_ref(logits_host[t * n_total + e]) + bias_host[e]
                )
                if best == -1 or s > best_score:
                    best = e
                    best_score = s
            winners.append(best)

        for j in range(topk):
            idx_ref[t * topk + j] = Int32(winners[j])

        # The kernel's log-space softmax equals a plain sigmoid share.
        # Asserting the identity keeps this oracle independent of the
        # kernel's own expression.
        var sigmoids = List[Float32]()
        for j in range(topk):
            sigmoids.append(_sigmoid_ref(logits_host[t * n_total + winners[j]]))
        for s in range(n_shared):
            sigmoids.append(
                _sigmoid_ref(logits_host[t * n_total + n_routed + s])
            )

        var total = Float32(0)
        for j in range(k_total):
            total += sigmoids[j]
        var factor = Float32(route_scale) * Float32(global_scale_val) / total

        for j in range(topk):
            w_ref[t * topk + j] = sigmoids[j] * factor
        for s in range(n_shared):
            sink_ref[t * n_shared + s] = sigmoids[topk + s] * factor

    assert_equal(
        idx_out.as_span(),
        idx_ref.as_span(),
        "expert index mismatch",
        shape=[num_tokens, topk],
    )
    assert_almost_equal(
        w_out.as_span(),
        w_ref.as_span(),
        "expert weight mismatch",
        shape=[num_tokens, topk],
        rtol=1e-5,
        atol=1e-6,
    )
    assert_almost_equal(
        sink_out.as_span(),
        sink_ref.as_span(),
        "sink weight mismatch",
        shape=[num_tokens, n_shared],
        rtol=1e-5,
        atol=1e-6,
    )


def main() raises:
    seed(0)
    with DeviceContext() as ctx:
        # Inkling-Small geometry: 256 routed experts, 6 selected, 2 sinks.
        test_sink_gate_router[256, 6, 2](1, ctx)
        test_sink_gate_router[256, 6, 2](17, ctx)
        test_sink_gate_router[256, 6, 2](64, ctx)
        # k_total = 4, and a routed count of exactly one AMD wavefront.
        test_sink_gate_router[64, 2, 2](5, ctx)
