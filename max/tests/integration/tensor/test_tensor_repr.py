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
"""Smoke tests for methods on `max.experimental.tensor.Tensor`.

These tests exercise each expected op at least once with real data and kernels.
They don't otherwise make any attempt at coverage, edge cases, or correctness.
"""

from __future__ import annotations

import re

from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental.tensor import Tensor, default_dtype
from max.experimental.testing import assert_all_close


def test_ones_defaults() -> None:
    with default_dtype(DType.float32):
        t = Tensor.ones([10])
        assert_all_close([1] * 10, t)


def test_zeros_like() -> None:
    ref = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = Tensor.zeros_like(ref)
    assert result.real
    assert list(result.driver_tensor.shape) == [4, 6]
    assert result.dtype == DType.float32


def test_ones_like() -> None:
    ref = Tensor.zeros(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = Tensor.ones_like(ref)
    assert result.real
    assert list(result.driver_tensor.shape) == [4, 6]
    assert result.dtype == DType.float32


def test_full_like() -> None:
    ref = Tensor.zeros(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = Tensor.full_like(ref, value=42.0)
    assert result.real
    assert list(result.driver_tensor.shape) == [4, 6]
    assert result.dtype == DType.float32


def _strip_device(r: str) -> str:
    """Remove device info from repr for exact matching (device varies by system)."""
    # Pattern: ", device=Device(type=...,id=...)" - remove entire device clause
    return re.sub(r", device=Device\([^)]+\)", "", r)


def test_repr_scalar() -> None:
    """Test repr for scalar (0D) tensor - exact match."""
    t = Tensor(42.5, dtype=DType.float32)
    r = _strip_device(repr(t))
    assert r == "Tensor(42.5, dtype=DType.float32)"


def test_repr_1d() -> None:
    """Test repr for 1D tensor (vector) - exact match."""
    t = Tensor([1.0, 2.0, 3.0], dtype=DType.float32)
    r = _strip_device(repr(t))
    assert r == "Tensor([1 2 3], dtype=DType.float32)"


def test_repr_1d_integers() -> None:
    """Test repr for 1D integer tensor - exact match."""
    t = Tensor([10, 20, 30], dtype=DType.int32)
    r = _strip_device(repr(t))
    assert r == "Tensor([10 20 30], dtype=DType.int32)"


def test_repr_2d_integers() -> None:
    """Test repr for 2D integer tensor with zeros."""
    t = Tensor([[0, 1], [2, 0]], dtype=DType.int32)
    r = _strip_device(repr(t))
    expected = "Tensor([0 1\n 2 0], dtype=DType.int32)"
    assert r == expected, f"Got:\n{r}\nExpected:\n{expected}"


def test_repr_int64() -> None:
    """Test repr for int64 tensor - large values formatted correctly."""
    t = Tensor([1000000, 2000000, 3000000], dtype=DType.int64)
    r = _strip_device(repr(t))
    expected = "Tensor([1000000 2000000 3000000], dtype=DType.int64)"
    assert r == expected, f"Got:\n{r}\nExpected:\n{expected}"


def test_repr_2d() -> None:
    """Test repr for 2D tensor (matrix) - exact multi-line match."""
    t = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=DType.float32)
    r = _strip_device(repr(t))
    expected = "Tensor([1 2\n 3 4], dtype=DType.float32)"
    assert r == expected, f"Got:\n{r}\nExpected:\n{expected}"


def test_repr_2d_3x3() -> None:
    """Test repr for 3x3 matrix - exact multi-line match."""
    t = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=DType.int32)
    r = _strip_device(repr(t))
    expected = "Tensor([1 2 3\n 4 5 6\n 7 8 9], dtype=DType.int32)"
    assert r == expected, f"Got:\n{r}\nExpected:\n{expected}"


def test_repr_3d() -> None:
    """Test repr for 3D tensor - exact matrix-of-matrices match."""
    t = Tensor(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        dtype=DType.int32,
    )
    r = _strip_device(repr(t))
    # Two 2x2 matrices side by side with | separator
    expected = "Tensor([1 2 | 5 6\n 3 4 | 7 8], dtype=DType.int32)"
    assert r == expected, f"Got:\n{r}\nExpected:\n{expected}"


def test_repr_3d_shape_2x2x3() -> None:
    """Test repr for shape [2,2,3] tensor - exact match."""
    t = Tensor(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=DType.int32,
    )
    r = _strip_device(repr(t))
    # Two 2x3 matrices side by side
    expected = (
        "Tensor([ 1  2  3 |  7  8  9\n  4  5  6 | 10 11 12], dtype=DType.int32)"
    )
    assert r == expected, f"Got:\n{r}\nExpected:\n{expected}"


def test_repr_bool_tensor() -> None:
    """Test repr for boolean tensor - exact match."""
    t = Tensor.full([2, 2], value=True, dtype=DType.bool)
    r = _strip_device(repr(t))
    expected = "Tensor([True True\n True True], dtype=DType.bool)"
    assert r == expected, f"Got:\n{r}\nExpected:\n{expected}"


def test_repr_bool_mixed() -> None:
    """Test repr for mixed boolean tensor - exact match."""
    t = Tensor([[True, False], [False, True]], dtype=DType.bool)
    r = _strip_device(repr(t))
    expected = "Tensor([ True False\n False  True], dtype=DType.bool)"
    assert r == expected, f"Got:\n{r}\nExpected:\n{expected}"


def test_repr_empty_tensor() -> None:
    """Test repr for empty tensor - exact match."""
    t = Tensor.zeros([0], dtype=DType.float32)
    r = _strip_device(repr(t))
    assert r == "Tensor([], dtype=DType.float32)"


def test_repr_float_precision() -> None:
    """Test repr for float with precision - values formatted correctly."""
    t = Tensor([1.2345, 2.5, 100.0], dtype=DType.float32)
    r = _strip_device(repr(t))
    # Precision is 4 sig digits; cell width aligned to widest element
    expected = "Tensor([1.235   2.5   100], dtype=DType.float32)"
    assert r == expected, f"Got:\n{r}\nExpected:\n{expected}"


def test_repr_negative_numbers() -> None:
    """Test repr with negative numbers - alignment preserved."""
    t = Tensor([[-1, 2], [3, -4]], dtype=DType.int32)
    r = _strip_device(repr(t))
    expected = "Tensor([-1  2\n  3 -4], dtype=DType.int32)"
    assert r == expected, f"Got:\n{r}\nExpected:\n{expected}"


def test_repr_shows_device_for_accelerator() -> None:
    """Test that repr shows device info for accelerator tensors."""
    if not accelerator_count():
        return  # Skip if no accelerator
    t = Tensor.ones([2, 2], dtype=DType.float32, device=Accelerator())
    r = repr(t)
    assert "Tensor(" in r
    # Accelerator device should be shown
    assert "device=" in r
