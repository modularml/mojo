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
"""Tests for max.graph.buffer_utils casting functions."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph.buffer_utils import (
    cast_dlpack_to,
    cast_tensor_to,
    cast_tensors_to,
)


@pytest.mark.parametrize(
    "old_dtype,new_dtype",
    [
        (DType.float32, DType.int32),
        (DType.int32, DType.float32),
        (DType.float32, DType.float64),
        # f16 on host hops through numpy: aarch64 CPU targets reject f16
        # graphs, which broke Wan FP8 weight adaptation on macOS/Graviton.
        (DType.float16, DType.float32),
        (DType.float32, DType.float16),
    ],
)
def test_cast_tensor_to(
    session: InferenceSession,
    old_dtype: DType,
    new_dtype: DType,
) -> None:
    """Test that cast_tensor_to correctly converts between dtypes."""
    device = CPU()
    np_array = np.array([1.0, 2.0, 3.0], dtype=old_dtype.to_numpy())
    tensor = Buffer.from_numpy(np_array).to(device)

    result = cast_tensor_to(tensor, new_dtype, session=session)

    assert result.dtype == new_dtype
    assert result.shape == tensor.shape
    result_np = result.to_numpy()
    expected_np = np_array.astype(new_dtype.to_numpy())
    if new_dtype.is_integral():
        np.testing.assert_array_equal(result_np, expected_np)
    else:
        np.testing.assert_allclose(result_np, expected_np, rtol=1e-5, atol=1e-5)


def test_cast_tensor_to_same_dtype(session: InferenceSession) -> None:
    """Test that cast_tensor_to returns the same tensor when dtype matches."""
    device = CPU()
    tensor = Buffer.from_numpy(np.array([1.0, 2.0], dtype=np.float32)).to(
        device
    )
    result = cast_tensor_to(tensor, DType.float32, session=session)
    assert result is tensor


def test_cast_tensor_to_f16_to_bf16(session: InferenceSession) -> None:
    """f16 -> bf16 on host: the f16 leg uses numpy, the bf16 leg a graph."""
    device = CPU()
    np_array = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    tensor = Buffer.from_numpy(np_array).to(device)

    result = cast_tensor_to(tensor, DType.bfloat16, session=session)

    assert result.dtype == DType.bfloat16
    assert result.shape == tensor.shape
    # numpy has no bf16, so verify values by casting back to f32.
    roundtrip = cast_tensor_to(result, DType.float32, session=session)
    np.testing.assert_allclose(
        roundtrip.to_numpy(), np_array.astype(np.float32), rtol=1e-2
    )


def test_cast_dlpack_to(session: InferenceSession) -> None:
    """Test that cast_dlpack_to correctly wraps and casts DLPack arrays."""
    device = CPU()
    np_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    result = cast_dlpack_to(
        np_array, DType.float32, DType.int32, device, session=session
    )

    assert result.dtype == DType.int32
    assert result.shape == np_array.shape
    assert result.device == device
    np.testing.assert_array_equal(result.to_numpy(), np_array.astype(np.int32))


def test_cast_tensors_to(session: InferenceSession) -> None:
    """Test that cast_tensors_to correctly casts a sequence of tensors."""
    device = CPU()
    tensors = [
        Buffer.from_numpy(np.array([1.0], dtype=np.float32)).to(device),
        Buffer.from_numpy(np.array([2.0], dtype=np.float32)).to(device),
    ]

    results = cast_tensors_to(tensors, DType.int32, session=session)

    assert len(results) == len(tensors)
    for result in results:
        assert result.dtype == DType.int32
    # Test empty/None handling
    assert cast_tensors_to([], DType.int32, session=session) == []
    assert cast_tensors_to(None, DType.int32, session=session) == []
