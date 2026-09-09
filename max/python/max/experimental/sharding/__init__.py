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

"""Distributed-tensor sharding: how a tensor is laid out across a device mesh.

Describes, for every op, what redistribution to perform before the op runs.
The pipeline is deliberately local: per-op rules over a placement vocabulary
(:class:`Replicated`, :class:`Sharded`, :class:`Partial`), scored by a single
cost model, with one pluggable :class:`Solver` making the choice at each
dispatch. There is no whole-graph trace.

A ``mode(...)`` block selects the solver for the ops inside it:

.. code-block:: python

    import numpy as np
    from max.driver import CPU
    from max.dtype import DType
    from max.experimental.functional import full, matmul, relu, transfer_to
    from max.experimental.sharding import (
        DeviceMesh,
        GreedyReshard,
        PlacementMapping,
        Sharded,
        mode,
    )

    # A simulated two-device mesh (both slots are the same CPU).
    mesh = DeviceMesh(devices=(CPU(), CPU()), mesh_shape=(2,), axis_names=("tp",))

    # ``a`` is column-sharded, ``b`` is row-sharded: a @ b contracts the
    # sharded dimension, so the picker resolves the result back to replicated.
    a = transfer_to(
        full([4, 8], 1.0, dtype=DType.float32, device=mesh.devices[0]),
        PlacementMapping(mesh, (Sharded(1),)),
    )
    b = transfer_to(
        full([8, 2], 1.0, dtype=DType.float32, device=mesh.devices[0]),
        PlacementMapping(mesh, (Sharded(0),)),
    )

    with mode(GreedyReshard(on_reshard="warn")):
        y = relu(matmul(a, b))

.. invisible-code-block: python

    import numpy as np

    # full(4, 8) @ full(8, 2) = 8 * ones(4, 2), then relu is a no-op (positive).
    assert np.allclose(y.to_numpy(), np.full((4, 2), 8.0))

Shipped solvers: :class:`GreedyReshard` (cheapest feasible action),
:class:`NoReshard` (passthrough only; errors on any reshard), and
:class:`PartialsOnly` (only ``Partial -> Replicated`` resolutions).

This module avoids the overloaded word "rank". A *device* is one accelerator;
a *mesh axis* is one named dimension of the :class:`DeviceMesh` grid; a
*shard* is one device's piece of a tensor; a *tensor axis* is a dimension of
the tensor itself.
"""

from .action import (
    Action,
    ActionSet,
    AxisAssignment,
    PerShard,
)
from .cost import (
    P,
    R,
    build_action_set,
    force_replicated_action_set,
)
from .mappings import (
    ConversionError,
    DeviceMapping,
    NamedMapping,
    PlacementMapping,
)
from .mesh import DeviceMesh, get_active_mesh, mesh_context

# Re-export so ``sharding.mode(...)`` resolves to the function, not the submodule.
from .mode import ShardingError, isolated_solver, mode
from .per_shard_dim import PerShardDim
from .picker import (
    GreedyReshard,
    NoReshard,
    PartialsOnly,
    ReshardBehavior,
    Solver,
)
from .placements import (
    Collective,
    Partial,
    Placement,
    ReduceOp,
    Replicated,
    Sharded,
)
from .rules import *
from .types import (
    DistributedBufferType,
    DistributedTensorType,
    DistributedType,
    TensorLayout,
)

__all__ = [
    "Action",
    "ActionSet",
    "AxisAssignment",
    "Collective",
    "ConversionError",
    "DeviceMapping",
    "DeviceMesh",
    "DistributedBufferType",
    "DistributedTensorType",
    "DistributedType",
    "GreedyReshard",
    "NamedMapping",
    "NoReshard",
    "P",
    "Partial",
    "PartialsOnly",
    "PerShard",
    "PerShardDim",
    "Placement",
    "PlacementMapping",
    "R",
    "ReduceOp",
    "Replicated",
    "ReshardBehavior",
    "Sharded",
    "ShardingError",
    "Solver",
    "TensorLayout",
    "build_action_set",
    "force_replicated_action_set",
    "get_active_mesh",
    "isolated_solver",
    "mesh_context",
    "mode",
]
