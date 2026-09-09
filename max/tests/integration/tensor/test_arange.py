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

from __future__ import annotations

import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental.tensor import Tensor, default_dtype
from max.experimental.testing import assert_all_close


def test_range_like() -> None:
    t = Tensor.ones([3, 4, 5], dtype=DType.float32, device=CPU())
    t2 = Tensor.range_like(t.type)
    assert t.type == t2.type
    assert_all_close(range(5), t2[0, 0, :])
    assert_all_close(range(5), t2[1, 2, :])


def test_arange() -> None:
    t = Tensor.arange(10, dtype=DType.float32, device=CPU())
    assert_all_close(range(10), t)


def test_arange_defaults() -> None:
    with default_dtype(DType.float32):
        t = Tensor.arange(10)
        assert_all_close(range(10), t)


def test_arange_start_stop() -> None:
    t = Tensor.arange(5, 10, dtype=DType.float32, device=CPU())
    assert_all_close([5, 6, 7, 8, 9], t)


def test_arange_start_stop_step() -> None:
    t = Tensor.arange(0, 10, 2, dtype=DType.float32, device=CPU())
    assert_all_close([0, 2, 4, 6, 8], t)


def test_arange_float_step() -> None:
    # Note: Use values that divide evenly to avoid floating point precision issues
    # (1.0 / 0.2 = 5.0 exactly, but 1.0 // 0.2 = 4.0 due to floating point)
    t = Tensor.arange(0.0, 1.0, 0.25, dtype=DType.float32, device=CPU())
    assert_all_close([0.0, 0.25, 0.5, 0.75], t)


def test_arange_negative_step() -> None:
    t = Tensor.arange(5, 0, -1, dtype=DType.float32, device=CPU())
    assert_all_close([5, 4, 3, 2, 1], t)


def test_arange_float_start_stop() -> None:
    t = Tensor.arange(0.5, 3.5, 1.0, dtype=DType.float32, device=CPU())
    assert_all_close([0.5, 1.5, 2.5], t)


def test_invalid() -> None:
    t = Tensor.arange(10, dtype=DType.float32, device=CPU())
    with pytest.raises(
        AssertionError, match=r"atol: tensors not close at index 0, 2.0 > 1e-06"
    ):
        assert_all_close(range(2, 12), t)
