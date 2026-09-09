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
"""Generic multi-adapter LoRA wrapper with per-call adapter inputs.

:class:`LoRA` wraps any Linear-like module in place and adds the ragged
multi-adapter LoRA delta computed by the SGMV kernels. Everything the batch
needs arrives out of band, not through ``forward``:

* the stacked adapters (``lora_a``/``lora_b``) are per-layer per-call graph
  inputs delivered through a ``_LoRAWeight`` ContextVar side-car, and
* the ragged routing (``lora_ids`` / ``grouped_offsets`` / ``end_idx``) is the
  batch-wide side-car set once per layer with :meth:`LoRA.set_lora_batch_info`.

The enclosing model takes these as extra inputs and distributes them via
:func:`lora_parameters` (adapters) and :func:`lora_layers` (routing), so
intermediate layers keep their plain ``forward(x)`` signature. Adapters ride in
as inputs rather than mutable weights. The SGMV kernels are GPU-only.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.linear import Linear
from max.experimental.nn.module import Module
from max.experimental.nn.transparent_module import TransparentModule
from max.experimental.tensor import Tensor, TensorType

from .linear import QKVLinear


class _LoRAWeight:
    r"""A per-call adapter stack: a graph-input ``type`` plus a value holder.

    Not a ``Tensor``, so :class:`Module` parameter discovery ignores it -- the
    adapter is delivered as an ordinary graph input rather than a baked or
    mutable weight. The enclosing model :meth:`set`\ s the value for the call;
    the wrapper :meth:`get`\ s it during ``forward``. The value lives in a
    :class:`~contextvars.ContextVar` so concurrent traces do not clobber it.
    """

    def __init__(
        self, type: TensorType, *, part_shapes: list[list[int]] | None = None
    ) -> None:
        self.type = type
        # For a fused qkv slot, the per-call input is the axis-0 concatenation
        # of its per-projection PEFT weights; ``part_shapes`` records those
        # sub-shapes (q, k, v order) so serving can build and zero-fill the
        # fused buffer. ``None`` for a single-projection slot (the whole input
        # is one PEFT weight).
        self.part_shapes = part_shapes
        self._value: ContextVar[Tensor | None] = ContextVar(
            "lora_weight", default=None
        )

    def set(self, value: Tensor) -> None:
        """Binds the adapter stack for the current call."""
        self._value.set(value)

    def get(self) -> Tensor:
        """Returns the adapter stack bound for the current call.

        Raises:
            RuntimeError: If the enclosing model did not :meth:`set` it first.
        """
        value = self._value.get()
        if value is None:
            raise RuntimeError(
                "LoRA adapter was not set for this call; the enclosing model "
                "must set() each slot before running the wrapped module."
            )
        return value


@dataclass(frozen=True)
class _Routing:
    """Batch-wide ragged routing, shared across all LoRA layers (not a param)."""

    ids: Tensor
    grouped_offsets: Tensor
    end_idx: Tensor


def _segmented_gather_matmul(
    x: Tensor,
    lora: Tensor,
    routing: _Routing,
    max_lora_seq_len: int,
) -> Tensor:
    """Segmented grouped matmul: ``out[start:end] = x[start:end] @ lora[id].T``.

    Each ``[start, end)`` token group from ``routing.grouped_offsets`` is
    multiplied by the adapter selected by the matching ``routing.ids`` entry.
    Only the first ``end_idx`` rows are produced; the kernel is GPU-only.
    """
    return F.custom(
        "mo.lora_sgmv.ragged",
        device=x.device,
        values=[
            x,
            lora,
            routing.grouped_offsets,
            routing.ids,
            F.constant(max_lora_seq_len, DType.uint32, device=CPU()),
        ],
        out_types=[
            TensorType(
                x.dtype,
                [routing.end_idx.shape[0], lora.shape[1]],
                device=x.device,
            )
        ],
    )[0]


def _qkv_shrink(
    x: Tensor,
    lora_a: Tensor,
    routing: _Routing,
    max_lora_seq_len: int,
    max_rank: int,
) -> Tensor:
    """Fused q/k/v shrink in one launch: ``[M, K] @ [G, 3*rank, K]^T``.

    Selects the per-group adapter as :func:`_segmented_gather_matmul` does,
    then returns the result in the planar layout ``[3, M, rank]`` (plane
    0 = q, 1 = k, 2 = v) the fused expand consumes. ``lora_a`` stacks the three
    per-projection A weights on its rank axis (rows ``[t*rank, (t+1)*rank)`` =
    projection ``t``). GPU-only.
    """
    return F.custom(
        "mo.lora_sgmv.qkv_shrink.ragged",
        device=x.device,
        values=[
            x,
            lora_a,
            routing.grouped_offsets,
            routing.ids,
            F.constant(max_lora_seq_len, DType.uint32, device=CPU()),
        ],
        out_types=[
            TensorType(
                x.dtype,
                [3, routing.end_idx.shape[0], max_rank],
                device=x.device,
            )
        ],
    )[0]


def _qkv_expand(
    planar: Tensor,
    lora_b: Tensor,
    routing: _Routing,
    max_lora_seq_len: int,
    q_dim: int,
    kv_dim: int,
) -> tuple[Tensor, Tensor]:
    """Fused q/k/v expand in one launch over the fused B weight.

    Takes the planar shrink output ``[3, M, rank]`` and the fused B weight
    ``[G, q_dim + 2*kv_dim, rank]`` (output rows partitioned q, then k, then v)
    and returns ``(q_out, kv_out)``. ``kv_out`` is row-stacked ``[2*M, kv_dim]``:
    k into rows ``[0, M)``, v into rows ``[M, 2*M)`` (the layout
    ``expand_qkv_sm100`` produces). GPU-only.
    """
    m = routing.end_idx.shape[0]
    outputs = F.custom(
        "mo.lora_sgmv.qkv_expand.ragged",
        device=planar.device,
        values=[
            planar,
            lora_b,
            routing.grouped_offsets,
            routing.ids,
            F.constant(max_lora_seq_len, DType.uint32, device=CPU()),
        ],
        out_types=[
            TensorType(planar.dtype, [m, q_dim], device=planar.device),
            TensorType(planar.dtype, [2 * m, kv_dim], device=planar.device),
        ],
    )
    return outputs[0], outputs[1]


def _sliced_add(x: Tensor, y: Tensor, end_idx: Tensor) -> Tensor:
    """Adds ``y`` into the first ``end_idx`` rows of ``x``; the rest pass through."""
    return F.custom(
        "mo.sliced.add.ragged",
        device=x.device,
        values=[x, y, end_idx],
        out_types=[TensorType(x.dtype, x.shape, device=x.device)],
    )[0]


class LoRA(TransparentModule[[Tensor], Tensor]):
    """Wraps a projection module with an additive multi-adapter LoRA term.

    ``forward(x)`` keeps the wrapped module's signature and returns
    ``module(x)`` plus, per ragged token group, the SGMV delta
    ``x @ lora_a[id].T @ lora_b[id].T``. A fused q/k/v module runs the delta as
    one fused shrink + one fused expand (2 adapter slots) over a fused A/B
    weight that stacks the per-projection q/k/v adapters; a plain Linear uses
    the single-projection shrink/expand (also 2 slots). Adapters arrive through
    :attr:`lora_a`/:attr:`lora_b` (per-call inputs) and routing through
    :meth:`set_lora_batch_info`; neither is a module parameter. Adapters are
    expected pre-scaled (``lora_alpha / r``).

    It holds no Tensor parameters of its own, so as a
    :class:`~max.experimental.nn.TransparentModule` it keeps the wrapped
    module's native checkpoint paths.

    Wrap a projection in place; the enclosing model drives routing and adapters
    each step via :func:`lora_layers` / :func:`lora_parameters`:

    .. code-block:: python

        from max.experimental.nn import Linear, LoRA

        proj = LoRA(
            Linear(1024, 1024),
            max_num_loras=4,
            max_lora_rank=16,
            max_lora_seq_len=512,
        )
        # Per step, before calling proj(x):
        #   proj.set_lora_batch_info(lora_ids, grouped_offsets, end_idx)
        #   proj.lora_a.set(a_stack); proj.lora_b.set(b_stack)
    """

    module: Module[..., Any]

    def __init__(
        self,
        module: Module[..., Any],
        *,
        max_num_loras: int,
        max_lora_rank: int,
        max_lora_seq_len: int,
    ) -> None:
        # A fused q/k/v runs the delta as one shrink + one expand launch over a
        # single fused A/B weight (2 slots); a plain Linear keeps the
        # single-projection path (also 2 slots). Only the weight's
        # dtype/device/in_dim are read (not its value).
        if isinstance(module, QKVLinear):
            weight = module.fused_weight
            projection_dims: tuple[int, ...] = (
                module.q_dim,
                module.kv_dim,
                module.kv_dim,
            )
        elif isinstance(module, Linear):
            weight = module.weight
            projection_dims = (int(module.weight.shape[0]),)
        else:
            raise TypeError(
                f"LoRA wraps a Linear or QKVLinear; got {type(module).__name__}."
            )
        in_dim = int(weight.shape[1])
        self.module = module
        self.max_lora_seq_len = max_lora_seq_len
        self.max_lora_rank = max_lora_rank
        self.fused = len(projection_dims) == 3
        if self.fused:
            self.q_dim, self.kv_dim, _ = projection_dims
            a_type = TensorType(
                weight.dtype,
                [max_num_loras, 3 * max_lora_rank, in_dim],
                device=weight.device,
            )
            b_type = TensorType(
                weight.dtype,
                [max_num_loras, self.q_dim + 2 * self.kv_dim, max_lora_rank],
                device=weight.device,
            )
            # Fused A concatenates three [rank, in] projection weights on the
            # rank axis; fused B concatenates [q_dim, rank], [kv_dim, rank],
            # [kv_dim, rank] on the output axis (q, k, v order).
            self.lora_a = _LoRAWeight(
                a_type,
                part_shapes=[[max_lora_rank, in_dim]] * 3,
            )
            self.lora_b = _LoRAWeight(
                b_type,
                part_shapes=[
                    [proj_dim, max_lora_rank] for proj_dim in projection_dims
                ],
            )
        else:
            (out_dim,) = projection_dims
            self.lora_a = _LoRAWeight(
                TensorType(
                    weight.dtype,
                    [max_num_loras, max_lora_rank, in_dim],
                    device=weight.device,
                )
            )
            self.lora_b = _LoRAWeight(
                TensorType(
                    weight.dtype,
                    [max_num_loras, out_dim, max_lora_rank],
                    device=weight.device,
                )
            )
        # Shared batch routing, set per layer via set_lora_batch_info; a
        # non-Tensor container so it is not mistaken for a parameter.
        self._routing: _Routing | None = None

    def set_lora_batch_info(
        self,
        lora_ids: Tensor,
        grouped_offsets: Tensor,
        end_idx: Tensor,
    ) -> None:
        """Wires this batch's ragged routing in for the next ``forward``."""
        self._routing = _Routing(lora_ids, grouped_offsets, end_idx)

    def forward(self, x: Tensor) -> Tensor:
        """Returns the base projection plus the per-request LoRA delta."""
        base = self.module(x)
        routing = self._routing
        if routing is None:
            raise RuntimeError(
                "set_lora_batch_info(...) must be called before forward; the "
                "enclosing model wires the batch routing into each LoRA layer."
            )
        if self.fused:
            planar = _qkv_shrink(
                x,
                self.lora_a.get(),
                routing,
                self.max_lora_seq_len,
                self.max_lora_rank,
            )
            q_out, kv_out = _qkv_expand(
                planar,
                self.lora_b.get(),
                routing,
                self.max_lora_seq_len,
                self.q_dim,
                self.kv_dim,
            )
            # kv_out row-stacks k over v ([2M, kv_dim]); split it back so the
            # delta columns line up with the fused qkv output ([q | k | v]).
            kv = kv_out.reshape([2, -1, self.kv_dim])
            delta = F.concat([q_out, kv[0], kv[1]], axis=-1)
        else:
            shrunk = _segmented_gather_matmul(
                x, self.lora_a.get(), routing, self.max_lora_seq_len
            )
            delta = _segmented_gather_matmul(
                shrunk, self.lora_b.get(), routing, self.max_lora_seq_len
            )
        return _sliced_add(base, delta, routing.end_idx)

    def _qualify_name(self, prefix: str, name: str) -> str:
        """Names the wrapped module's params as if this LoRA were absent.

        The wrapper holds no parameters of its own, so it must stay invisible
        to naming: the base weights keep the exact checkpoint paths they'd have
        unwrapped. Drop the ``module`` holder segment this wrapper introduces,
        then delegate to the wrapped module's own (possibly name-transparent)
        qualification at this LoRA's position -- so an opaque leaf keeps its
        native leaf name (``o_proj.weight``, not ``module.weight``) and a
        transparent ``QKVLinear`` still exposes native ``q_proj``/``k_proj``/
        ``v_proj``.
        """
        return self.module._qualify_name(prefix, name.removeprefix("module."))


def lora_layers(model: Module[..., Any]) -> Iterator[tuple[str, LoRA]]:
    """Yields ``(qualified_name, layer)`` for every :class:`LoRA` in ``model``.

    Used to broadcast the batch routing (:meth:`LoRA.set_lora_batch_info`).
    """
    for name, child in model.descendants:
        if isinstance(child, LoRA):
            yield name, child


def lora_parameters(
    model: Module[..., Any],
) -> Iterator[tuple[str, _LoRAWeight]]:
    """Yields ``(qualified_name, slot)`` for every adapter stack in ``model``.

    Two slots per :class:`LoRA` layer (``lora_a`` then ``lora_b``): a plain
    Linear's A/B, or a fused qkv layer's fused A/B (each stacking the three
    q/k/v adapters). Serving uses this both to append the adapter input types at
    :meth:`~Module.compile` and to feed each call's adapters in order.
    """
    for name, layer in lora_layers(model):
        yield f"{name}.lora_a", layer.lora_a
        yield f"{name}.lora_b", layer.lora_b
