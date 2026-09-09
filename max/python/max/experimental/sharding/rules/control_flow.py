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

"""Placement rules for control-flow ops (``cond``, ``while_loop``)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from typing import Any

from max.experimental.sharding import (
    DeviceMapping,
    PlacementMapping,
)
from max.experimental.sharding.types import TensorLayout
from max.graph.type import Type
from max.graph.value import TensorValue, Value

from ..action import Action, ActionSet
from ..cost import force_replicated_action_set


def cond_rule(
    pred: TensorLayout,
    out_types: Iterable[Type[Any]] | None,
    then_fn: Callable[..., Any],
    else_fn: Callable[..., Any],
) -> ActionSet:
    """Auto-gathers the predicate; every device takes the same branch.

    ``out_types``, ``then_fn`` and ``else_fn`` ride along as extras so the
    dispatcher forwards them to the underlying op unchanged.
    """
    return force_replicated_action_set(
        pred, extras=(out_types, then_fn, else_fn)
    )


def _while_loop_finalize(
    items: list[Any],
    n: int,
    predicate: Callable[..., TensorValue],
    body: Callable[..., Value[Any] | Iterable[Value[Any]]],
    container_type: type,
) -> Callable[[Action], Action]:
    """Builds the ``while_loop`` finalize, pre-bound to its context.

    Args:
        items: User's original ``initial_values`` flattened to a list,
            preserving non-tensor :class:`Value` entries in position.
        n: Count of distributed :class:`TensorLayout` entries in ``items``.
        predicate: User-supplied loop predicate callable.
        body: User-supplied loop body callable.
        container_type: Original container type to restore.
    """

    def finalize(action: Action) -> Action:
        mappings = action.inputs[:n]
        suggested: list[DeviceMapping | Value[Any]] = []
        out_mappings: list[DeviceMapping] = []
        m_idx = 0
        for v in items:
            if isinstance(v, TensorLayout):
                m = mappings[m_idx]
                assert isinstance(m, PlacementMapping)
                suggested.append(m)
                out_mappings.append(m)
                m_idx += 1
            else:
                suggested.append(v)
        return Action(
            inputs=(container_type(suggested), predicate, body),
            outputs=tuple(out_mappings),
        )

    return finalize


def while_loop_rule(
    initial_values: (
        Iterable[TensorLayout | Value[Any]] | TensorLayout | Value[Any]
    ),
    predicate: Callable[..., TensorValue],
    body: Callable[..., Value[Any] | Iterable[Value[Any]]],
) -> ActionSet:
    """Auto-gathers distributed initial values to Replicated.

    The loop body is not yet distribution-aware. ``finalize`` repackages
    the picked mappings (and any non-tensor :class:`Value` entries) into
    the user's original list/tuple container, plus ``predicate`` /
    ``body`` as extras.
    """
    if isinstance(initial_values, TensorLayout):
        return force_replicated_action_set(
            initial_values, extras=(predicate, body)
        )

    if not isinstance(initial_values, (list, tuple)):
        raise TypeError(
            "while_loop_rule: initial_values must be a TensorLayout, "
            f"list, or tuple; got {type(initial_values).__name__}."
        )

    items = list(initial_values)
    tensor_layouts = tuple(v for v in items if isinstance(v, TensorLayout))
    n = len(tensor_layouts)
    if n == 0:
        raise ValueError(
            "while_loop_rule: no distributed TensorLayouts in initial_values."
        )

    pool = force_replicated_action_set(*tensor_layouts)
    return dataclasses.replace(
        pool,
        finalize=_while_loop_finalize(
            items=items,
            n=n,
            predicate=predicate,
            body=body,
            container_type=type(initial_values),
        ),
    )
