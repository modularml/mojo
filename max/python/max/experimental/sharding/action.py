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

"""Per-op decision data model: :class:`AxisAssignment`, :class:`Action`, :class:`ActionSet`."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from max.experimental.sharding.mappings import DeviceMapping
from max.experimental.sharding.mesh import DeviceMesh
from max.experimental.sharding.placements import Placement
from max.experimental.sharding.types import TensorLayout
from max.graph.dim import Dim


class AxisAssignment(NamedTuple):
    """One per-mesh-axis row in an :class:`ActionSet`.

    Reads as: given these per-axis input placements, the output along
    that axis is :attr:`output`. Picking one :class:`AxisAssignment`
    per mesh axis builds a multi-axis :class:`Action`.
    """

    needed_inputs: tuple[Placement, ...]
    output: Placement


class PerShard:
    """A distinct value per mesh shard.

    Wraps a sequence of per-shard values so the dispatcher can index
    into it by mesh shard when forwarding a non-tensor argument to a
    per-shard op call. Non-:class:`PerShard` values appearing in
    :attr:`ActionSet.extras` are treated as uniform across shards.
    """

    __slots__ = ("values",)

    values: tuple[Any, ...]

    def __init__(self, values: Iterable[Any]) -> None:
        object.__setattr__(self, "values", tuple(values))

    def __getitem__(self, i: int) -> Any:
        return self.values[i]

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return f"PerShard({self.values!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PerShard) and self.values == other.values

    def __hash__(self) -> int:
        return hash(("PerShard", self.values))


@dataclass(frozen=True)
class Action:
    """A rule's picked decision for one op call.

    :attr:`inputs` has one entry per op argument in positional order:
    :class:`DeviceMapping` at tensor positions; bare scalar (treated as
    uniform across ranks) or :class:`PerShard` at non-tensor positions.
    :attr:`outputs` has one :class:`DeviceMapping` per op output. The
    dispatcher inserts ``transfer_to`` collectives to match
    :attr:`inputs` before per-shard dispatch; :attr:`outputs` is the
    post-op mapping each result wears.
    """

    inputs: tuple[Any, ...]
    outputs: tuple[DeviceMapping, ...]

    def __iter__(self) -> Iterator[Any]:
        """Yields ``(inputs, outputs)``."""
        yield self.inputs
        yield self.outputs


@dataclass(frozen=True)
class ActionSet:
    """A rule's menu of per-axis sharding options for one op call.

    Shape-aware (it depends on operand layouts) but cost-blind: it
    lists what is possible, not what is cheapest. The dispatcher
    picks one entry per mesh axis.
    """

    axis_assignments: tuple[AxisAssignment, ...]
    """Per-axis rows, each pickable independently per mesh axis. The
    last entry is the universal ``(R,…,R) -> R`` fallback."""

    layouts: tuple[TensorLayout, ...]
    """Per-tensor input layouts this menu was built for."""

    mesh: DeviceMesh
    """The mesh both the picker and planner work over."""

    extras: tuple[Any, ...] = ()
    """Non-tensor positional args appended after tensor-input mappings
    in the picked :class:`Action`'s :attr:`inputs`. A bare value is
    treated as uniform across ranks; wrap in :class:`PerShard` to vary
    per rank."""

    result_shape: Sequence[Dim] | None = None
    """Output shape constraint, set by reshape-style rules. Lets the
    feasibility check reject output :class:`Sharded` rows whose result
    dim is too small."""

    finalize: Callable[[Action], Action] | None = None
    """Optional post-pick transform applied as ``finalize(action)``. Used
    by rules that need the picked placement to compute per-rank metadata
    or repack inputs into a user-facing container. Rules pre-bind any
    per-op context by closing over it, so no separate context field is
    needed."""
