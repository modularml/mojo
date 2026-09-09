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
"""Construction tests for WeightNormConvTranspose1d in max.nn."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, Graph
from max.nn import WeightNormConvTranspose1d


def test_weight_norm_conv_transpose1d_default_no_bias() -> None:
    """Constructs with the default has_bias=False without raising.

    __init__ deleted self.conv.bias unconditionally, but the conv only sets
    that attribute when has_bias=True, so the default raised AttributeError.
    """
    with Graph("test"):
        conv = WeightNormConvTranspose1d(
            length=3,
            in_channels=16,
            out_channels=8,
            dtype=DType.float32,
            device=DeviceRef.CPU(),
        )
    assert conv.bias is None


def test_weight_norm_conv_transpose1d_with_bias() -> None:
    """has_bias=True still constructs and keeps a bias weight."""
    with Graph("test"):
        conv = WeightNormConvTranspose1d(
            length=3,
            in_channels=16,
            out_channels=8,
            dtype=DType.float32,
            device=DeviceRef.CPU(),
            has_bias=True,
        )
    assert conv.bias is not None
