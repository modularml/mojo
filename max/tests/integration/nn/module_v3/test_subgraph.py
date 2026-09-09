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
"""Subgraph compilation via :func:`max.experimental.nn.subgraphable`.

A repeated sub-module lowers to one shared subgraph called once per layer, so
the compiler processes the body once instead of once per repetition. This file
covers the feature:

* core — a repeated block becomes one shared subgraph (via the ``@subgraphable``
  class decorator or the ``subgraphable(layer)(x)`` form), each call threading
  its own weights, with correct numerics;
* the dedup key — two calls share a body iff their traced IR is identical, so the
  key tracks exactly what the body reads: an incidental field shares, a baked
  per-layer value (a constant or a distinct subclass) splits, and threading that
  value in as a Tensor operand shares again;
* outputs — a nested (tuple/dict) return round-trips through the call;
* nesting — a subgraph called inside a subgraph body inlines (any depth);
* execution modes — eager inlines (no subgraph), while graph-compile and lazy
  both share subgraphs;
* distributed — a tensor-parallel block with sharded weights and an all-reduce;
* guardrails — calling outside a capture raises, ``allow_subgraphs=False`` opts
  out, and a body cannot read a value realized in its parent graph.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import (
    Module,
    ModuleList,
    module_dataclass,
    subgraphable,
)
from max.experimental.sharding import (
    DeviceMesh,
    PlacementMapping,
    Replicated,
    Sharded,
)
from max.experimental.sharding.types import DistributedTensorType
from max.experimental.tensor import (
    Tensor,
    TensorType,
    current_realization_context,
)
from max.graph import DeviceRef

F32 = DType.float32
D = 4
H = 8


def default_type() -> TensorType:
    return TensorType(F32, ["batch", D], device=DeviceRef.CPU())


def zeros(*shape: int) -> Tensor:
    return Tensor.zeros(list(shape), dtype=F32, device=CPU())


def randn(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return rng.standard_normal(shape).astype(np.float32)


def count_graphs(mlir: str, name: str) -> int:
    return len(re.findall(rf"mo\.graph @{name}(?:_\d+)?\b", mlir))


def count_calls(mlir: str, name: str) -> int:
    return len(re.findall(rf"mo\.call @{name}(?:_\d+)?\b", mlir))


def externals(mlir: str) -> int:
    return mlir.count("mo.constant.external")


@module_dataclass
class Stack(Module[[Tensor], Tensor]):
    """An ordinary layer loop."""

    layers: ModuleList

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


@subgraphable
@module_dataclass
class Block(Module[[Tensor], Tensor]):
    """A residual MLP block; ``tag`` is incidental and never read in forward."""

    w_in: Tensor  # [D, hidden]
    w_out: Tensor  # [hidden, D]
    tag: int = 0

    def forward(self, x: Tensor) -> Tensor:
        return x + F.relu(x @ self.w_in) @ self.w_out


def _block(tag: int = 0) -> Block:
    return Block(zeros(D, H), zeros(H, D), tag=tag)


def _block_ref(
    x: np.ndarray, w_in: np.ndarray, w_out: np.ndarray
) -> np.ndarray:
    return x + np.maximum(x @ w_in, 0.0) @ w_out


# ─── Core: a repeated block becomes one shared subgraph ──────────────────────


def test_repeated_block_shares_one_subgraph() -> None:
    """The flagship case: decorate a block, stack it in a plain loop, compile.
    Three layers trace to one ``@Block`` body called three times, each threading
    its own weights, and the result matches the inlined reference."""
    rng = np.random.default_rng(0)
    stack = Stack(layers=ModuleList([_block() for _ in range(3)]))
    weights: dict[str, np.ndarray] = {}
    for i in range(3):
        weights[f"layers.{i}.w_in"] = randn(rng, D, H)
        weights[f"layers.{i}.w_out"] = randn(rng, H, D)

    mlir = str(stack.trace(default_type())._module)
    assert count_graphs(mlir, "Block") == 1
    assert count_calls(mlir, "Block") == 3

    compiled = stack.compile(default_type(), weights=weights)
    x = randn(rng, 2, D)
    expected = x
    for i in range(3):
        expected = _block_ref(
            expected, weights[f"layers.{i}.w_in"], weights[f"layers.{i}.w_out"]
        )
    np.testing.assert_allclose(
        compiled(Tensor(x, device=CPU())).to_numpy(),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )


def test_callable_form_shares_body() -> None:
    """``subgraphable(layer)(x)`` dedups exactly like the class decorator, so a
    model builder can subgraph selected instances at the call site instead of
    decorating the class."""

    @module_dataclass
    class CallableStack(Module[[Tensor], Tensor]):
        layers: ModuleList

        def forward(self, x: Tensor) -> Tensor:
            for layer in self.layers:
                x = subgraphable(layer)(x)
            return x

    stack = CallableStack(layers=ModuleList([_block() for _ in range(3)]))
    mlir = str(stack.trace(default_type())._module)
    assert count_graphs(mlir, "Block") == 1
    assert count_calls(mlir, "Block") == 3


# ─── The dedup key tracks exactly what the body reads ────────────────────────


def test_dedup_key_tracks_what_the_body_reads() -> None:
    """The dedup key is the body's traced IR, so sharing tracks what the body
    actually *reads*:

    * an incidental field the body never reads does not split (Dan Moldovan's
      "changing hyperparameters" caveat -- a differing field is safe as long as
      forward ignores it);
    * a per-layer ``int`` *baked* into the body (``F.constant(idx)``) makes each
      body's IR differ, so every layer becomes its own subgraph (Kathy Wu's
      ``layer_idx`` question);
    * passing that same index in as a *Tensor operand* threads it through the
      call site, so the bodies are identical again and share one subgraph --
      this is the gemma3 ``layer_idx`` fix.
    """
    # (a) Incidental field: ``tag`` differs per layer but forward never reads it.
    incidental = Stack(layers=ModuleList([_block(tag=i) for i in range(3)]))
    mlir = str(incidental.trace(default_type())._module)
    assert (count_graphs(mlir, "Block"), count_calls(mlir, "Block")) == (1, 3)

    # (b) Baked per-layer index: read inside forward as a Python int, so each
    # layer bakes a different ``ops.constant`` -> a distinct body per layer.
    @subgraphable
    @module_dataclass
    class BakedIndex(Module[[Tensor], Tensor]):
        w: Tensor
        idx: int = 0

        def forward(self, x: Tensor) -> Tensor:
            return x @ self.w + F.constant(self.idx, F32, device=CPU())

    baked = Stack(
        layers=ModuleList([BakedIndex(zeros(D, D), idx=i) for i in range(3)])
    )
    mlir = str(baked.trace(default_type())._module)
    assert (
        count_graphs(mlir, "BakedIndex"),
        count_calls(mlir, "BakedIndex"),
    ) == (3, 3)

    # (c) Threaded index: the same per-layer value flows in as a Tensor operand,
    # so the body reads a block argument (not a literal) and all layers share;
    # the differing values live at the call sites in the parent graph.
    @subgraphable
    @module_dataclass
    class ThreadedIndex(Module[..., Tensor]):
        w: Tensor

        def forward(self, x: Tensor, idx: Tensor) -> Tensor:
            return x @ self.w + idx

    @module_dataclass
    class ThreadedStack(Module[[Tensor], Tensor]):
        layers: ModuleList

        def forward(self, x: Tensor) -> Tensor:
            for i, layer in enumerate(self.layers):
                x = layer(x, F.constant(i, F32, device=CPU()))
            return x

    threaded = ThreadedStack(
        layers=ModuleList([ThreadedIndex(zeros(D, D)) for _ in range(3)])
    )
    mlir = str(threaded.trace(default_type())._module)
    assert (
        count_graphs(mlir, "ThreadedIndex"),
        count_calls(mlir, "ThreadedIndex"),
    ) == (
        1,
        3,
    )


def test_distinct_subclasses_split() -> None:
    """Separate subclasses (dense vs. expert) lower to separate bodies, and
    identical siblings within a class share one -- the class-per-group answer to
    grouping e.g. dense and MoE layers backed by different blocks."""

    @subgraphable
    @module_dataclass
    class Dense(Module[[Tensor], Tensor]):
        w: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + F.relu(x @ self.w)

    @subgraphable
    @module_dataclass
    class Expert(Module[[Tensor], Tensor]):
        w: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + (x @ self.w) * 2.0

    stack = Stack(
        layers=ModuleList(
            [Dense(zeros(D, D)), Dense(zeros(D, D))]
            + [Expert(zeros(D, D)) for _ in range(3)]
        )
    )
    mlir = str(stack.trace(default_type())._module)
    assert (count_graphs(mlir, "Dense"), count_calls(mlir, "Dense")) == (1, 2)
    assert (count_graphs(mlir, "Expert"), count_calls(mlir, "Expert")) == (1, 3)


def test_captured_constant_not_in_parameters() -> None:
    """A Tensor the body reads but that is not a module parameter (a captured
    *eager* constant) is re-materialized inside the subgraph rather than threaded
    as an operand; identical captures still share one body and numerics hold.
    Contrast :func:`test_subgraph_cannot_read_parent_graph_value`, where the
    captured value is a *parent-graph* value and cannot be re-materialized."""
    cap_np = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    cap = Tensor(cap_np, device=CPU())

    @subgraphable
    @module_dataclass
    class CapBlock(Module[[Tensor], Tensor]):
        w: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return F.relu(x @ self.w) + cap

    rng = np.random.default_rng(7)
    stack = Stack(layers=ModuleList([CapBlock(zeros(D, D)) for _ in range(3)]))
    weights = {f"layers.{i}.w": randn(rng, D, D) for i in range(3)}

    mlir = str(stack.trace(default_type())._module)
    assert count_graphs(mlir, "CapBlock") == 1
    assert count_calls(mlir, "CapBlock") == 3

    compiled = stack.compile(default_type(), weights=weights)
    x = randn(rng, 2, D)
    expected = x
    for i in range(3):
        expected = np.maximum(expected @ weights[f"layers.{i}.w"], 0.0) + cap_np
    np.testing.assert_allclose(
        compiled(Tensor(x, device=CPU())).to_numpy(),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )


# ─── Weight materialization: weights live once in the shared body ────────────


def test_subgraphed_weights_materialize_once() -> None:
    """Tests that the weight-prefix mechanism works correctly."""
    for n_layers in (3, 8):
        stack = Stack(layers=ModuleList([_block() for _ in range(n_layers)]))
        mlir = str(stack.trace(default_type())._module)
        assert count_calls(mlir, "Block") == n_layers
        # Two external constants (w_in, w_out) regardless of depth, not 2 x N.
        assert externals(mlir) == 2
        # They are placeholders resolved by the call prefix, and carry the
        # block's *relative* names -- per-layer full names are never emitted as
        # constants (the call's prefix resolves them at load time instead).
        assert mlir.count("isPlaceholder = true") == 2
        assert 'name = "w_in"' in mlir and 'name = "w_out"' in mlir
        assert 'name = "layers.0.w_in"' not in mlir


# ─── Explicit subgraph name controls dedup (name overrides the IR hash) ──────


def test_explicit_name_controls_sharing() -> None:
    """``subgraphable(layer, name=...)`` keys the subgraph on the name instead
    of the body's IR hash, so naming controls dedup in both directions:

    * a shared name collapses blocks whose bodies would *otherwise hash
      differently* into one subgraph (you vouch they match);
    * distinct names keep *otherwise-identical* blocks as separate subgraphs.
    """

    @module_dataclass
    class Baked(Module[[Tensor], Tensor]):
        """Bakes a per-layer constant into the body, so unnamed blocks hash
        differently and would not otherwise share."""

        w: Tensor
        idx: int = 0

        def forward(self, x: Tensor) -> Tensor:
            return x @ self.w + F.constant(self.idx, F32, device=CPU())

    # Unnamed: each baked constant differs, so the IR hash splits the three
    # layers into three subgraphs.
    unnamed = Stack(
        layers=ModuleList(
            [subgraphable(Baked(zeros(D, D), idx=i)) for i in range(3)]
        )
    )
    mlir = str(unnamed.trace(default_type())._module)
    assert (count_graphs(mlir, "Baked"), count_calls(mlir, "Baked")) == (3, 3)

    # Same name: the three differing bodies collapse to one shared subgraph.
    named = Stack(
        layers=ModuleList(
            [
                subgraphable(Baked(zeros(D, D), idx=i), name="shared")
                for i in range(3)
            ]
        )
    )
    mlir = str(named.trace(default_type())._module)
    assert (count_graphs(mlir, "shared"), count_calls(mlir, "shared")) == (1, 3)
    assert count_graphs(mlir, "Baked") == 0

    # Distinct names split two identical blocks that would otherwise share.
    distinct = Stack(
        layers=ModuleList(
            [
                subgraphable(_block(), name="a"),
                subgraphable(_block(), name="b"),
            ]
        )
    )
    mlir = str(distinct.trace(default_type())._module)
    assert (count_graphs(mlir, "a"), count_calls(mlir, "a")) == (1, 1)
    assert (count_graphs(mlir, "b"), count_calls(mlir, "b")) == (1, 1)


# ─── Nesting: a subgraph inside a subgraph body inlines (any depth) ──────────


def test_nested_subgraphs_inline() -> None:
    """Only the outermost call becomes a subgraph; subgraphs nested inside a
    body inline at every depth (here three levels, C -> B -> A). The graph
    compiler does not nest subgraphs -- only a depth-1 submodule is a subgraph,
    deeper ones flatten into it."""

    @subgraphable
    @module_dataclass
    class A(Module[[Tensor], Tensor]):
        w: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return F.relu(x @ self.w)

    @subgraphable
    @module_dataclass
    class B(Module[[Tensor], Tensor]):
        a: A
        w: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.a(x) @ self.w

    @subgraphable
    @module_dataclass
    class C(Module[[Tensor], Tensor]):
        b: B
        w: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x + self.b(x) @ self.w

    stack = Stack(
        layers=ModuleList(
            [C(B(A(zeros(D, D)), zeros(D, D)), zeros(D, D)) for _ in range(2)]
        )
    )
    mlir = str(stack.trace(default_type())._module)
    assert (count_graphs(mlir, "C"), count_calls(mlir, "C")) == (1, 2)
    # The two inner levels inline rather than nesting their own subgraphs.
    assert "mo.graph @B" not in mlir and "mo.graph @A" not in mlir


# ─── Tensor parallelism: sharded weights + an all-reduce inside the body ─────

# A two-way tensor-parallel mesh, simulated on CPU: the trace shape and numerics
# match a real two-GPU mesh, minus the hardware collectives.
MESH = DeviceMesh(devices=(CPU(), CPU()), mesh_shape=(2,), axis_names=("tp",))
REPLICATED = PlacementMapping(MESH, (Replicated(),))
COLUMN = PlacementMapping(MESH, (Sharded(1),))
ROW = PlacementMapping(MESH, (Sharded(0),))


@subgraphable
@module_dataclass
class TPBlock(Module[[Tensor], Tensor]):
    """Column-parallel up-projection, row-parallel down-projection, all-reduced
    back to a replicated residual."""

    w_in: Tensor  # [D, H], column-parallel
    w_out: Tensor  # [H, D], row-parallel

    def forward(self, x: Tensor) -> Tensor:  # x replicated [batch, D]
        hidden = F.relu(x @ self.w_in)  # sharded on H
        # Row-parallel matmul leaves partial sums; transfer to replicated
        # performs the all-reduce.
        return x + F.transfer_to(hidden @ self.w_out, REPLICATED)


def _tp_block() -> TPBlock:
    return TPBlock(
        w_in=F.transfer_to(zeros(D, H), COLUMN),
        w_out=F.transfer_to(zeros(H, D), ROW),
    )


def test_tensor_parallel_block_shares_one_subgraph() -> None:
    """A real distributed block: sharded weights thread in per device and an
    all-reduce runs inside the body. Two layers share one ``@TPBlock``, each
    weight registers one external constant per shard, and numerics hold."""
    rng = np.random.default_rng(1)
    input_type = DistributedTensorType(F32, ["batch", D], MESH, (Replicated(),))
    stack = Stack(layers=ModuleList([_tp_block() for _ in range(2)]))

    weights: dict[str, np.ndarray] = {}
    for i in range(2):
        weights[f"layers.{i}.w_in"] = randn(rng, D, H)
        weights[f"layers.{i}.w_out"] = randn(rng, H, D)

    mlir = str(stack.trace(input_type)._module)
    assert count_graphs(mlir, "TPBlock") == 1
    assert count_calls(mlir, "TPBlock") == 2
    # The shared body registers each sharded weight once under its *relative*
    # name (one external constant per device); each layer's ``mo.call`` carries
    # the per-layer prefix that resolves those names to its own weights.
    for weight in ("w_in", "w_out"):
        for shard in range(2):
            assert f'name = "{weight}._shard.{shard}"' in mlir
    for i in range(2):
        assert f'prefix = "layers.{i}."' in mlir

    compiled = stack.compile(input_type, weights=weights)
    x = randn(rng, 2, D)
    replicated = F.transfer_to(Tensor(x, device=CPU()), REPLICATED)
    result = compiled(replicated)
    assert result.placements == (Replicated(),)

    expected = x
    for i in range(2):
        w_in, w_out = weights[f"layers.{i}.w_in"], weights[f"layers.{i}.w_out"]
        expected = expected + np.maximum(expected @ w_in, 0.0) @ w_out
    np.testing.assert_allclose(
        result.to_numpy(), expected, rtol=1e-4, atol=1e-4
    )


# ─── Execution modes: eager inlines; lazy shares (compile shares, above) ─────


def _block_with(
    rng: np.random.Generator,
) -> tuple[Block, np.ndarray, np.ndarray]:
    w_in, w_out = randn(rng, D, H), randn(rng, H, D)
    block = Block(Tensor(w_in, device=CPU()), Tensor(w_out, device=CPU()))
    return block, w_in, w_out


def test_eager_call_passes_through() -> None:
    """In eager mode a ``@subgraphable`` layer runs inline (no subgraph): the
    call just computes, so the result matches the bare forward."""
    rng = np.random.default_rng(20)
    block, w_in, w_out = _block_with(rng)
    x = randn(rng, 2, D)
    out = block(Tensor(x, device=CPU())).to_numpy()
    np.testing.assert_allclose(
        out, _block_ref(x, w_in, w_out), rtol=1e-4, atol=1e-4
    )


def test_lazy_shares_one_subgraph_and_runs() -> None:
    """A lazy block builds one deferred graph, so repeated layers share a single
    subgraph (like compile); realizing it later yields the right numerics."""
    rng = np.random.default_rng(21)
    blocks = [_block_with(rng) for _ in range(3)]
    x = randn(rng, 2, D)

    with F.lazy():
        stack = Stack(layers=ModuleList([b for b, _, _ in blocks]))
        out = stack(Tensor(x, device=CPU()))
        # The deferred graph already holds the shared subgraph + per-layer calls.
        # (Lazy IR prints generic form, so check the registry + ``callee``.)
        graph = current_realization_context().graph
        assert list(graph._subgraphs) == ["Block"]
        assert str(graph._module).count("callee = @Block") == 3

    expected = x
    for _, w_in, w_out in blocks:
        expected = _block_ref(expected, w_in, w_out)
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-4, atol=1e-4)


# ─── Outputs: a nested (tuple/dict) return round-trips through the call ───────


def test_structured_output_round_trips() -> None:
    """A non-tensor return structure -- here a tuple nesting a dict -- round-trips
    through the ``mo.call`` via the same pytree walk used for inputs: one shared
    body, one call per layer, and correct numerics. Plain tuple and plain dict
    returns are the degenerate cases of this."""

    @subgraphable
    @module_dataclass
    class NestBlock(Module[..., tuple[Tensor, dict[str, Tensor]]]):
        w: Tensor

        def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
            h = F.relu(x @ self.w)
            return h, {"residual": x + h}

    @module_dataclass
    class NestStack(Module[[Tensor], Tensor]):
        layers: ModuleList

        def forward(self, x: Tensor) -> Tensor:
            for layer in self.layers:
                a, d = layer(x)
                x = a + d["residual"]
            return x

    rng = np.random.default_rng(27)
    stack = NestStack(
        layers=ModuleList([NestBlock(zeros(D, D)) for _ in range(3)])
    )
    weights = {f"layers.{i}.w": randn(rng, D, D) for i in range(3)}

    mlir = str(stack.trace(default_type())._module)
    assert (
        count_graphs(mlir, "NestBlock"),
        count_calls(mlir, "NestBlock"),
    ) == (1, 3)

    compiled = stack.compile(default_type(), weights=weights)
    x = randn(rng, 2, D)
    expected = x
    for i in range(3):
        h = np.maximum(expected @ weights[f"layers.{i}.w"], 0.0)
        expected = h + (expected + h)
    np.testing.assert_allclose(
        compiled(Tensor(x, device=CPU())).to_numpy(),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )


# ─── Guardrails ──────────────────────────────────────────────────────────────


def test_calling_outside_a_capture_runs_eager() -> None:
    """Marking a module has no effect outside a capture: calling a subgraphable
    instance eagerly just runs ``forward`` (no subgraph, no error) and matches
    the bare-forward result."""
    rng = np.random.default_rng(31)
    block, w_in, w_out = _block_with(rng)
    marked = subgraphable(block)
    x = randn(rng, 2, D)
    out = marked(Tensor(x, device=CPU())).to_numpy()
    np.testing.assert_allclose(
        out, _block_ref(x, w_in, w_out), rtol=1e-4, atol=1e-4
    )


def test_allow_subgraphs_false_inlines_everything() -> None:
    """``compile(allow_subgraphs=False)`` inlines every ``@subgraphable`` layer
    instead of emitting shared subgraphs, so the model traces into one flat
    graph (no ``mo.graph``/``mo.call``) and the numerics are unchanged. This is
    the Module-level equivalent of the pipelines ``use_subgraphs`` flag."""
    rng = np.random.default_rng(30)
    stack = Stack(layers=ModuleList([_block() for _ in range(3)]))
    weights: dict[str, np.ndarray] = {}
    for i in range(3):
        weights[f"layers.{i}.w_in"] = randn(rng, D, H)
        weights[f"layers.{i}.w_out"] = randn(rng, H, D)

    graph, *_ = stack._trace((default_type(),), allow_subgraphs=False)
    mlir = str(graph._module)
    assert count_graphs(mlir, "Block") == 0
    assert count_calls(mlir, "Block") == 0

    compiled = stack.compile(
        default_type(), weights=weights, allow_subgraphs=False
    )
    x = randn(rng, 2, D)
    expected = x
    for i in range(3):
        expected = _block_ref(
            expected, weights[f"layers.{i}.w_in"], weights[f"layers.{i}.w_out"]
        )
    np.testing.assert_allclose(
        compiled(Tensor(x, device=CPU())).to_numpy(),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )


def test_subgraph_cannot_read_parent_graph_value() -> None:
    """KNOWN LIMITATION: a subgraph body may only read its own operands (its
    forward arguments and its parameters) plus eager constants it can
    re-materialize. A value *realized in the enclosing (parent) graph* lives in
    a different region and is out of scope, so reading one across the subgraph
    boundary fails to compile.

    The fix is to thread that value in as a forward argument. This is exactly
    why the gemma3 attention takes the rope ``freqs_cis`` table as an operand
    instead of reading the rope's cached tensor across the boundary -- the
    capture below is the reduced form of that bug.
    """

    @module_dataclass
    class CapturingStack(Module[[Tensor], Tensor]):
        w: Tensor

        def forward(self, x: Tensor) -> Tensor:
            # Realized in THIS (parent) graph; the body below closes over it
            # rather than taking it as an argument, so it crosses the boundary.
            parent_value = F.relu(x)

            @subgraphable
            @module_dataclass
            class Reader(Module[[Tensor], Tensor]):
                w: Tensor

                def forward(self, h: Tensor) -> Tensor:
                    # self.w threads in as an operand (fine); parent_value is
                    # captured from the parent graph (the unsupported case).
                    return h @ self.w + parent_value

            block = Reader(self.w)
            for _ in range(2):
                x = block(x)
            return x

    stack = CapturingStack(zeros(D, D))
    with pytest.raises(TypeError, match="Can't realize from a graph context"):
        stack.compile(default_type())
