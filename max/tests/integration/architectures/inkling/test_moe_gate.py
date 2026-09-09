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
"""Inkling's MoE gate against the graph-op chain its fused router replaced.

The kernel's own shape coverage lives in the Mojo unit test
(``max/kernels/test/gpu/nn/test_sink_gate_router.mojo``); this one exists to
keep the Python binding and the ``mo.moe.sink.gate.router`` registration
honest, so one shape is enough. It evaluates the production path and its
unfused reference in one graph, so the two see identical weights and inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.kernels import moe_sink_gate_router
from max.pipelines.architectures.inkling.layers.moe import InklingGate
from torch.utils.dlpack import from_dlpack

_HIDDEN_DIM = 64
_ROUTE_SCALE = 8.0


def _assert_close(actual: np.ndarray, expected: np.ndarray) -> None:
    torch.testing.assert_close(
        torch.from_numpy(actual),
        torch.from_numpy(expected),
        rtol=1e-5,
        atol=1e-6,
    )


def _unfused_route(
    gate: InklingGate, logits: TensorValue
) -> tuple[TensorValue, TensorValue, TensorValue]:
    """The `top_k` / `gather_nd` / log-sigmoid-softmax chain that
    `moe_sink_gate_router` collapses into one kernel."""
    device = logits.device
    routed = logits[:, : gate.n_routed_experts]
    _, expert_ids = ops.top_k(
        ops.sigmoid(routed) + gate.bias.to(device),
        k=gate.num_experts_per_token,
        axis=-1,
    )

    # Weights renormalize the raw logits: the selection bias must not leak
    # into them.
    selected = ops.gather_nd(
        routed, ops.unsqueeze(expert_ids, axis=-1), batch_dims=1
    )
    sinks = logits[:, gate.n_routed_experts :]

    combined = ops.concat([selected, sinks], axis=-1)
    zero = ops.constant(0.0, combined.dtype, device=device)
    log_scores = ops.min(combined, zero) - ops.log1p(
        ops.exp(-ops.abs(combined))
    )
    scores = ops.exp(log_scores - ops.max(log_scores, axis=-1))
    factor = (
        ops.constant(gate.route_scale, logits.dtype, device=device)
        * gate.global_scale.to(device)
    ) / ops.sum(scores, axis=-1)
    weights = scores * factor
    k = gate.num_experts_per_token
    return expert_ids, weights[:, :k], weights[:, k:]


def test_inkling_gate_route_matches_graph_ops() -> None:
    """`InklingGate.route`'s fused kernel must match the graph-op chain."""
    num_tokens, n_routed_experts, n_experts_per_tok, n_shared = 17, 256, 6, 2
    device = Accelerator(0)
    rng = np.random.default_rng(1)
    hidden_states_np = rng.normal(size=(num_tokens, _HIDDEN_DIM)).astype(
        np.float32
    )
    n_total = n_routed_experts + n_shared
    weights = {
        "weight": rng.normal(size=(n_total, _HIDDEN_DIM)).astype(np.float32),
        "bias": rng.normal(size=(n_routed_experts,)).astype(np.float32) * 0.1,
        "global_scale": np.array([1.3], dtype=np.float32),
    }

    gate = InklingGate(
        devices=[DeviceRef.GPU()],
        hidden_dim=_HIDDEN_DIM,
        num_experts=n_routed_experts,
        num_experts_per_token=n_experts_per_tok,
        dtype=DType.float32,
        n_shared_experts=n_shared,
        route_scale=_ROUTE_SCALE,
    )
    gate.load_state_dict(weights)

    with Graph(
        "test_inkling_gate_route",
        input_types=(
            TensorType(
                DType.float32, hidden_states_np.shape, device=DeviceRef.GPU()
            ),
        ),
    ) as g:
        hidden_states = g.inputs[0].tensor
        routing = gate.route(hidden_states)
        # The same logits `route` builds, so only the routing math differs.
        weight = ops.cast(gate.weight, hidden_states.dtype).to(
            hidden_states.device
        )
        logits = ops.cast(hidden_states @ weight.T, DType.float32)
        g.output(
            routing.expert_ids,
            routing.expert_weights,
            routing.sink_weights,
            *_unfused_route(gate, logits),
        )

    session = InferenceSession(devices=[device])
    model = session.load(g, weights_registry=gate.state_dict())
    results = model.execute(Buffer.from_numpy(hidden_states_np).to(device))
    fused = [from_dlpack(r).cpu().numpy() for r in results[:3]]
    unfused = [from_dlpack(r).cpu().numpy() for r in results[3:]]

    np.testing.assert_array_equal(fused[0], unfused[0])
    _assert_close(fused[1], unfused[1])
    _assert_close(fused[2], unfused[2])


@pytest.mark.parametrize(
    ("n_routed_experts", "n_experts_per_tok", "n_shared"),
    [
        (1024, 30, 2),
        (512, 28, 4),
    ],
)
def test_moe_sink_gate_router_rejects_shapes_the_kernel_cannot_compile(
    n_routed_experts: int, n_experts_per_tok: int, n_shared: int
) -> None:
    """These clear the per-count bounds but leave more top-k survivors than a
    warp holds. The check that catches them re-derives the kernel's phase
    arithmetic in Python, so it needs pinning against the day that changes.

    Both overrun at 32 and 64 lanes, because the survivor bound scales with the
    warp width and this target runs on AMD too. Deriving the shapes from the
    same width the guard reads would only restate the arithmetic under test.
    """
    n_total = n_routed_experts + n_shared
    with Graph(
        "reject",
        input_types=(
            TensorType(DType.float32, [8, n_total], device=DeviceRef.GPU()),
        ),
    ) as g:
        with pytest.raises(ValueError, match="survivors"):
            moe_sink_gate_router(
                g.inputs[0].tensor,
                ops.constant(
                    0.0, DType.float32, device=DeviceRef.GPU()
                ).broadcast_to([n_routed_experts]),
                ops.constant(
                    1.0, DType.float32, device=DeviceRef.GPU()
                ).broadcast_to([1]),
                n_routed_experts=n_routed_experts,
                n_experts_per_tok=n_experts_per_tok,
                n_shared_experts=n_shared,
                route_scale=1.0,
            )
