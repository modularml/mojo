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
"""Shared test logic for custom op dispatch.

Demonstrates two ways to make a custom Mojo kernel distribution-aware:

1. **Explicit dispatch** via ``per_shard_dispatch`` — the recommended approach.
   The user writes the rule call and redistribution explicitly, then
   delegates per-shard execution to ``per_shard_dispatch``.
2. **Manual dispatch** — the user writes the per-shard loop explicitly
   (educational, full control).

Both approaches produce identical results and are tested here.

DO NOT run this file directly — it contains base classes that are
subclassed by test_custom_dispatch_simulated_cpu.py.

Subclasses must define:
    MESH_2: DeviceMesh  — 2 devices, shape (2,), axis_names=("tp",)
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from max.experimental import tensor as _tensor_mod
from max.experimental.functional import transfer_to
from max.experimental.functional.spmd_ops import (
    per_shard_dispatch,
)
from max.experimental.functional.spmd_ops import (
    tensor_to_layout as tl,
)
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    PlacementMapping,
    Replicated,
    Sharded,
    TensorLayout,
)
from max.experimental.tensor import Tensor
from max.graph import TensorValue, ops

# ═════════════════════════════════════════════════════════════════════════
#  Shared: placement rule + graph kernel
# ═════════════════════════════════════════════════════════════════════════
# The rule is a pure function on metadata — it could live in
# sharding/rules/ alongside the built-in rules.


def rms_norm_rule(
    x: TensorLayout,
    weight: TensorLayout,
    eps: float = 1e-6,
) -> tuple[
    tuple[DeviceMapping, DeviceMapping, float], tuple[DeviceMapping, ...]
]:
    """RMSNorm reduces over the last dim — cannot be sharded there."""
    placements = x.mapping.to_placements()
    ndim = x.rank
    for p in placements:
        if isinstance(p, Sharded) and p.axis == ndim - 1:
            raise ValueError(
                "rms_norm: cannot shard hidden dim. "
                "Gather first or shard a different axis."
            )
    out_mapping = PlacementMapping(x.mesh, placements)
    return (out_mapping, weight.mapping, eps), (out_mapping,)


def _rms_norm_kernel(
    x: TensorValue,
    weight: TensorValue,
    eps: float = 1e-6,
    weight_offset: float = 0.0,
) -> TensorValue:
    """Single-device rms_norm via the built-in graph op."""
    return ops.rms_norm(x, weight, epsilon=eps, weight_offset=weight_offset)


# ═════════════════════════════════════════════════════════════════════════
#  Approach 1: explicit dispatch via per_shard_dispatch (recommended)
# ═════════════════════════════════════════════════════════════════════════
# The user calls the rule explicitly, redistributes, then delegates
# per-shard execution to per_shard_dispatch.


def rms_norm(
    x: _tensor_mod.Tensor,
    weight: _tensor_mod.Tensor,
    eps: float = 1e-6,
) -> _tensor_mod.Tensor:
    (xm, wm, _eps), (out_m,) = rms_norm_rule(tl(x), tl(weight), eps)
    return per_shard_dispatch(
        _rms_norm_kernel,
        (transfer_to(x, xm), transfer_to(weight, wm), _eps),
        (out_m,),
    )


# ═════════════════════════════════════════════════════════════════════════
#  Approach 2: manual dispatch (per-shard loop written out)
# ═════════════════════════════════════════════════════════════════════════
# Same logic as per_shard_dispatch, but written out explicitly so the user
# can see every step.


def rms_norm_manual(
    x: _tensor_mod.Tensor,
    weight: _tensor_mod.Tensor,
    eps: float = 1e-6,
) -> _tensor_mod.Tensor:
    # 1. Rule
    (xm, wm, _eps), output_mappings = rms_norm_rule(tl(x), tl(weight), eps)

    # 2. Redistribute
    x_rd = transfer_to(x, xm)
    w_rd = transfer_to(weight, wm)

    # 3. Dispatch
    return per_shard_dispatch(
        _rms_norm_kernel,
        (x_rd, w_rd, _eps),
        output_mappings,
    )


# ═════════════════════════════════════════════════════════════════════════
#  Numpy reference
# ═════════════════════════════════════════════════════════════════════════


def _rms_norm_numpy(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    variance = np.mean(x**2, axis=-1, keepdims=True)
    return x / np.sqrt(variance + eps) * w


# ═════════════════════════════════════════════════════════════════════════
#  Test classes
# ═════════════════════════════════════════════════════════════════════════


class _CustomDispatchExplicit:
    """Tests for custom op dispatch via per_shard_dispatch (recommended path)."""

    MESH_2: ClassVar[DeviceMesh]

    def test_rms_norm_replicated(self) -> None:
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((4, 8)).astype(np.float32)
        w_np = np.ones(8, dtype=np.float32)

        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        w = transfer_to(
            Tensor(w_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = rms_norm(x, w, 1e-6)
        assert result.placements == (Replicated(),)
        expected = _rms_norm_numpy(x_np, w_np, 1e-6)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-4)

    def test_rms_norm_batch_sharded(self) -> None:
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((4, 8)).astype(np.float32)
        w_np = np.ones(8, dtype=np.float32)

        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        w = transfer_to(
            Tensor(w_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = rms_norm(x, w, 1e-6)
        assert result.placements == (Sharded(0),)
        expected = _rms_norm_numpy(x_np, w_np, 1e-6)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-4)


class _CustomDispatchManual:
    """Tests for custom op dispatch via manual per-shard loop."""

    MESH_2: ClassVar[DeviceMesh]

    def test_rms_norm_manual_replicated(self) -> None:
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((4, 8)).astype(np.float32)
        w_np = np.ones(8, dtype=np.float32)

        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        w = transfer_to(
            Tensor(w_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = rms_norm_manual(x, w, 1e-6)
        assert result.placements == (Replicated(),)
        expected = _rms_norm_numpy(x_np, w_np, 1e-6)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-4)

    def test_rms_norm_manual_batch_sharded(self) -> None:
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((4, 8)).astype(np.float32)
        w_np = np.ones(8, dtype=np.float32)

        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        w = transfer_to(
            Tensor(w_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = rms_norm_manual(x, w, 1e-6)
        assert result.placements == (Sharded(0),)
        expected = _rms_norm_numpy(x_np, w_np, 1e-6)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-4)


class CustomDispatchTests(_CustomDispatchExplicit, _CustomDispatchManual):
    """Aggregates all custom dispatch test classes."""
