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

"""Placement rules for ``max_pool`` and ``avg_pool``."""

from __future__ import annotations

from typing import Any

from max.experimental.sharding.placements import Sharded
from max.experimental.sharding.types import TensorLayout

from ..action import ActionSet, AxisAssignment
from ..cost import P, R, build_action_set


# NHWC: N=0, H=1, W=2, C=3. ``*extra`` absorbs kernel/stride/etc.
def pool_rule(input: TensorLayout, *extra: Any) -> ActionSet:
    """Strategies for pooling (max/avg): shard on N or C axis (NHWC)."""
    rows = [
        AxisAssignment((R,), R),
        AxisAssignment((Sharded(0),), Sharded(0)),
        AxisAssignment((Sharded(3),), Sharded(3)),
    ]
    return build_action_set(rows, layouts=(input,), extras=extra)


def linear_pool_rule(input: TensorLayout, *extra: Any) -> ActionSet:
    """Strategies for linear pooling (avg_pool2d): adds Partial passthrough."""
    rows = [
        AxisAssignment((R,), R),
        AxisAssignment((Sharded(0),), Sharded(0)),
        AxisAssignment((Sharded(3),), Sharded(3)),
        AxisAssignment((P,), P),
    ]
    return build_action_set(rows, layouts=(input,), extras=extra)
