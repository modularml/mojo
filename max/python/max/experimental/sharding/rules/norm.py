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

"""Placement rules for ``layer_norm`` and ``rms_norm``."""

from __future__ import annotations

from max.experimental.sharding.placements import Sharded
from max.experimental.sharding.types import TensorLayout

from ..action import ActionSet, AxisAssignment
from ..cost import R, build_action_set


def layer_norm_rule(
    input: TensorLayout,
    gamma: TensorLayout,
    beta: TensorLayout,
    epsilon: float,
) -> ActionSet:
    """Strategies for ``layer_norm``: shard only on leading (pre-norm) axes."""
    leading = range(input.rank - gamma.rank)
    rows = [
        AxisAssignment((R, R, R), R),
        *(AxisAssignment((Sharded(d), R, R), Sharded(d)) for d in leading),
    ]
    return build_action_set(
        rows, layouts=(input, gamma, beta), extras=(epsilon,)
    )


def rms_norm_rule(
    input: TensorLayout,
    weight: TensorLayout,
    epsilon: float,
    weight_offset: float = 0.0,
    multiply_before_cast: bool = False,
) -> ActionSet:
    """Strategies for ``rms_norm``: shard only on leading (pre-norm) axes."""
    leading = range(input.rank - weight.rank)
    rows = [
        AxisAssignment((R, R), R),
        *(AxisAssignment((Sharded(d), R), Sharded(d)) for d in leading),
    ]
    return build_action_set(
        rows,
        layouts=(input, weight),
        extras=(epsilon, weight_offset, multiply_before_cast),
    )
