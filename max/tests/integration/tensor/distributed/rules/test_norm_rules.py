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

"""Pure-metadata tests for normalization placement rules."""

from __future__ import annotations

from max.dtype import DType
from max.experimental.sharding import DeviceMapping, TensorLayout
from max.experimental.sharding.rules import (
    layer_norm_rule,
    rms_norm_rule,
)
from max.graph import Shape

from rules._fixtures import MESH_1D, M, R, S, pick


def _layout(
    mapping: DeviceMapping, shape: tuple[int, ...], dtype: DType = DType.float32
) -> TensorLayout:
    return TensorLayout(dtype, Shape(shape), mapping)


# ═════════════════════════════════════════════════════════════════════════
#  layer_norm_rule
# ═════════════════════════════════════════════════════════════════════════


class TestLayerNormRule:
    def test_replicated(self) -> None:
        x = _layout(M(MESH_1D, R), (4, 8))
        gamma = _layout(M(MESH_1D, R), (8,))
        beta = _layout(M(MESH_1D, R), (8,))
        _, (out,) = pick(layer_norm_rule, x, gamma, beta, 1e-5)
        assert out.to_placements() == (R,)

    def test_batch_sharded(self) -> None:
        """S(0) on [B, H] with weight shape [H]: norm dims start at 1."""
        x = _layout(M(MESH_1D, S(0)), (4, 8))
        gamma = _layout(M(MESH_1D, R), (8,))
        beta = _layout(M(MESH_1D, R), (8,))
        _, (out,) = pick(layer_norm_rule, x, gamma, beta, 1e-5)
        assert out.to_placements() == (S(0),)

    def test_norm_dim_sharded_falls_back(self) -> None:
        """Sharding the norm (hidden) dim falls back to Replicated.

        Consistent with all other rules: no raise, just cost-model
        fallback. The dispatcher will allgather to materialize the
        Replicated state.
        """
        x = _layout(M(MESH_1D, S(1)), (4, 8))
        gamma = _layout(M(MESH_1D, R), (8,))
        beta = _layout(M(MESH_1D, R), (8,))
        args, (out,) = pick(layer_norm_rule, x, gamma, beta, 1e-5)
        assert args[0].to_placements() == (R,)
        assert out.to_placements() == (R,)

    def test_3d_input_2d_weight(self) -> None:
        """[B, S, H] with weight [S, H]: norm dims start at 1."""
        x = _layout(M(MESH_1D, S(0)), (2, 4, 8))
        gamma = _layout(M(MESH_1D, R), (4, 8))
        beta = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = pick(layer_norm_rule, x, gamma, beta, 1e-5)
        assert out.to_placements() == (S(0),)

    def test_3d_input_sharded_seq_falls_back(self) -> None:
        """[B, S, H] with S(1) (seq dim, in norm range): falls back to R."""
        x = _layout(M(MESH_1D, S(1)), (2, 4, 8))
        gamma = _layout(M(MESH_1D, R), (4, 8))
        beta = _layout(M(MESH_1D, R), (4, 8))
        args, (out,) = pick(layer_norm_rule, x, gamma, beta, 1e-5)
        assert args[0].to_placements() == (R,)
        assert out.to_placements() == (R,)


# ═════════════════════════════════════════════════════════════════════════
#  rms_norm_rule
# ═════════════════════════════════════════════════════════════════════════


class TestRMSNormRule:
    def test_replicated(self) -> None:
        x = _layout(M(MESH_1D, R), (4, 8))
        weight = _layout(M(MESH_1D, R), (8,))
        _, (out,) = pick(rms_norm_rule, x, weight, 1e-6)
        assert out.to_placements() == (R,)

    def test_batch_sharded(self) -> None:
        x = _layout(M(MESH_1D, S(0)), (4, 8))
        weight = _layout(M(MESH_1D, R), (8,))
        _, (out,) = pick(rms_norm_rule, x, weight, 1e-6)
        assert out.to_placements() == (S(0),)

    def test_norm_dim_sharded_falls_back(self) -> None:
        x = _layout(M(MESH_1D, S(1)), (4, 8))
        weight = _layout(M(MESH_1D, R), (8,))
        args, (out,) = pick(rms_norm_rule, x, weight, 1e-6)
        assert args[0].to_placements() == (R,)
        assert out.to_placements() == (R,)
