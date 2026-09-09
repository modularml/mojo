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
"""Smoke tests for ops in `max.experimental.functional`.

These tests exercise each expected op at least once with real data and kernels.
They don't otherwise make any attempt at coverage, edge cases, or correctness.
"""

import pytest
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor

DEVICE = Accelerator() if accelerator_count() else CPU()

UNARY = [
    F.abs,
    F.argsort,
    F.atanh,
    F.ceil,
    F.cos,
    F.cumsum,
    F.erf,
    F.exp,
    F.floor,
    F.gelu,
    F.is_inf,
    F.is_nan,
    F.log,
    F.log1p,
    F.logsoftmax,
    F.negate,
    F.relu,
    F.round,
    F.rsqrt,
    F.sigmoid,
    F.silu,
    F.sin,
    F.softmax,
    F.sqrt,
    F.tanh,
    F.trunc,
]

LOGICAL_UNARY = [
    F.logical_not,
]


@pytest.mark.parametrize("op", UNARY)
def test_unary(op) -> None:  # noqa: ANN001
    tensor = Tensor.zeros([10], dtype=DType.float32, device=DEVICE)
    result = op(tensor)
    assert result.real
    assert list(result.driver_tensor.shape) == tensor.shape


@pytest.mark.parametrize("op", LOGICAL_UNARY)
def test_logical_unary(op) -> None:  # noqa: ANN001
    tensor = Tensor.full([10], False, dtype=DType.bool, device=DEVICE)
    result = op(tensor)
    assert result.real
    assert list(result.driver_tensor.shape) == tensor.shape
