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

"""Pure-metadata tests for pooling placement rules."""

from __future__ import annotations

from max.dtype import DType
from max.experimental.sharding import DeviceMapping, TensorLayout
from max.experimental.sharding.rules import pool_rule
from max.graph import Shape

from rules._fixtures import MESH_1D, M, R, S, pick


def _layout(
    mapping: DeviceMapping, shape: tuple[int, ...], dtype: DType = DType.float32
) -> TensorLayout:
    return TensorLayout(dtype, Shape(shape), mapping)


class TestPoolRule:
    """Tests for pool_rule (NHWC layout: batch=0, H=1, W=2, C=3)."""

    def test_replicated(self) -> None:
        layout = _layout(M(MESH_1D, R), (1, 8, 8, 3))
        _, (out,) = pick(pool_rule, layout)
        assert out.to_placements() == (R,)

    def test_batch_sharded(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8, 8, 3))
        _, (out,) = pick(pool_rule, layout)
        assert out.to_placements() == (S(0),)

    def test_channel_sharded(self) -> None:
        layout = _layout(M(MESH_1D, S(3)), (1, 8, 8, 16))
        _, (out,) = pick(pool_rule, layout)
        assert out.to_placements() == (S(3),)

    def test_spatial_h_sharded_falls_back(self) -> None:
        """Spatial H-sharded input: spatial axes excluded from catalogue, falls back to R."""
        layout = _layout(M(MESH_1D, S(1)), (1, 8, 8, 3))
        _, (out,) = pick(pool_rule, layout)
        assert out.to_placements() == (R,)

    def test_spatial_w_sharded_falls_back(self) -> None:
        """Spatial W-sharded input: spatial axes excluded from catalogue, falls back to R."""
        layout = _layout(M(MESH_1D, S(2)), (1, 8, 8, 3))
        _, (out,) = pick(pool_rule, layout)
        assert out.to_placements() == (R,)
