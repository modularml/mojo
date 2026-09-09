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
"""End-to-end tests for MO interpreter op handlers.

These tests verify that the graph-compiler-backed op handlers produce
correct results by comparing against numpy reference implementations.
"""

from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from max import _core
from max._core.dialects import mo, rmo
from max._interpreter_ops import (
    data_movement_gc,
    elementwise_binary_gc,
    gather_gc,
    gc_compile,
    matmul_gc,
    nms_gc,
    nonzero_gc,
    random_gc,
    range_gc,
    scatter_nd_gc,
    select_gc,
    shape_rearrange_gc,
    unary_elementwise_gc,
    warm,
)
from max._interpreter_ops.handlers import (
    _handle_buffer_create,
    _handle_buffer_transfer,
    _handle_gather_sum,
    _handle_index_to_tensor,
    _handle_shape_from_tensor,
)
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental import random as max_random
from max.experimental import realization_context as rc
from max.experimental.executor import (
    CompilingExecutor,
    InterpreterExecutor,
    UnsupportedGraphError,
    set_default_executor,
)
from max.experimental.functional import (
    reduce_scatter,
)
from max.experimental.functional import (
    transfer_to as df_shard,
)
from max.experimental.realization_context import set_seed
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    PlacementMapping,
    Replicated,
    Sharded,
)
from max.experimental.tensor import Tensor, realization_context
from max.graph import BufferType, DeviceRef, Module


@pytest.fixture(autouse=True)
def _interpreter_only() -> Generator[None]:
    """All tests in this module run on the interpreter -- it is the unit
    under test.  Graphs it cannot execute raise UnsupportedGraphError."""
    with set_default_executor(InterpreterExecutor()):
        yield


# DTypes to test for elementwise operations
# Note: bfloat16 is excluded since NumPy doesn't support it natively
FLOAT_DTYPES = [DType.float32, DType.float64]
# Shape-rearrange ops are pure copies, so they also serve float16 (which has no
# typed CPU kernel) via the width-based uint bit-cast.
REARRANGE_FLOAT_DTYPES = FLOAT_DTYPES + [DType.float16]
INT_DTYPES = [DType.int8, DType.int16, DType.int32, DType.int64]
UINT_DTYPES = [DType.uint8, DType.uint16, DType.uint32, DType.uint64]
SIGNED_DTYPES = FLOAT_DTYPES + INT_DTYPES
ELEMENTWISE_DTYPES = SIGNED_DTYPES + UINT_DTYPES
# DTypes to test for matmul operations (float and integer)
MATMUL_DTYPES = FLOAT_DTYPES + INT_DTYPES

# Captured pre-migration; the shared kernel's counter mapping must not change them.
_RANDOM_NORMAL_SEED7_F32 = np.array(
    [
        -0.30484700202941895,
        1.0453333854675293,
        -0.27856412529945374,
        0.7178202867507935,
    ],
    dtype=np.float32,
)


class TestBinaryElementwiseOps:
    """Tests for binary elementwise Mojo ops."""

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_add(self, dtype: DType) -> None:
        """Test add op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        a_np = np.arange(12, dtype=np_dtype).reshape(shape)
        b_np = np.arange(12, 24, dtype=np_dtype).reshape(shape)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a + b

        expected = np.add(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_sub(self, dtype: DType) -> None:
        """Test sub op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        a_np = np.arange(12, 24, dtype=np_dtype).reshape(shape)
        b_np = np.arange(12, dtype=np_dtype).reshape(shape)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a - b

        expected = np.subtract(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_mul(self, dtype: DType) -> None:
        """Test mul op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        a_np = np.arange(1, 13, dtype=np_dtype).reshape(shape)
        b_np = np.arange(2, 14, dtype=np_dtype).reshape(shape)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a * b

        expected = np.multiply(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_div(self, dtype: DType) -> None:
        """Test div op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        a_np = np.arange(1, 13, dtype=np_dtype).reshape(shape)
        b_np = np.arange(1, 13, dtype=np_dtype).reshape(shape) + 0.5

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a / b

        expected = np.divide(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_pow(self, dtype: DType) -> None:
        """Test pow op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        # Use positive base values to avoid NaN from fractional exponents
        a_np = np.arange(1, 13, dtype=np_dtype).reshape(shape)
        b_np = np.full(shape, 2.0, dtype=np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a**b

        expected = np.power(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", INT_DTYPES + UINT_DTYPES)
    def test_pow_integer(self, dtype: DType) -> None:
        """Int ``pow`` matches numpy; Pow sweeps integer dtypes (``div`` does not)."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        # Small bases / exponent 2 keep the result within every int width.
        a_np = np.arange(1, 13, dtype=np_dtype).reshape(shape)
        b_np = np.full(shape, 2, dtype=np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a**b

        expected = np.power(a_np, b_np)
        np.testing.assert_array_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_max(self, dtype: DType) -> None:
        """Test elementwise max op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        a_np = np.arange(12, dtype=np_dtype).reshape(shape)
        b_np = np.arange(11, -1, -1, dtype=np_dtype).reshape(shape)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = F.max(a, b)

        expected = np.maximum(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_min(self, dtype: DType) -> None:
        """Test elementwise min op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        a_np = np.arange(12, dtype=np_dtype).reshape(shape)
        b_np = np.arange(11, -1, -1, dtype=np_dtype).reshape(shape)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = F.min(a, b)

        expected = np.minimum(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_clamp(self, dtype: DType) -> None:
        """Test elementwise clamp op matches numpy."""
        shape = [3]
        np_dtype = dtype.to_numpy()
        a_np = np.array([12, 12, 12], dtype=np_dtype).reshape(shape)
        lower_bound_np = np.array([0, 2, 12], dtype=np_dtype).reshape(shape)
        upper_bound_np = np.array([13, 3, 15], dtype=np_dtype).reshape(shape)

        a = Tensor.from_dlpack(a_np)
        lower_bound = Tensor.from_dlpack(lower_bound_np)
        upper_bound = Tensor.from_dlpack(upper_bound_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = F.clamp(a, lower_bound, upper_bound)

        expected = np.clip(a_np, lower_bound_np, upper_bound_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)


class TestBinaryComparisonOps:
    """Tests for binary comparison Mojo ops (output is bool)."""

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_equal(self, dtype: DType) -> None:
        """Test equal op returns bool and matches numpy."""
        np_dtype = dtype.to_numpy()
        a_np = np.array([1, 2, 3, 4], dtype=np_dtype)
        b_np = np.array([1, 5, 3, 6], dtype=np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a == b

        result_np = np.from_dlpack(c)
        expected = np.equal(a_np, b_np)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.bool_

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_not_equal(self, dtype: DType) -> None:
        """Test not_equal op returns bool and matches numpy."""
        np_dtype = dtype.to_numpy()
        a_np = np.array([1, 2, 3, 4], dtype=np_dtype)
        b_np = np.array([1, 5, 3, 6], dtype=np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a != b

        result_np = np.from_dlpack(c)
        expected = np.not_equal(a_np, b_np)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.bool_

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_greater(self, dtype: DType) -> None:
        """Test greater op returns bool and matches numpy."""
        np_dtype = dtype.to_numpy()
        a_np = np.array([1, 5, 3, 6], dtype=np_dtype)
        b_np = np.array([2, 3, 3, 4], dtype=np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a > b

        result_np = np.from_dlpack(c)
        expected = np.greater(a_np, b_np)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.bool_

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_greater_equal(self, dtype: DType) -> None:
        """Test greater_equal op returns bool and matches numpy."""
        np_dtype = dtype.to_numpy()
        a_np = np.array([1, 5, 3, 6], dtype=np_dtype)
        b_np = np.array([2, 3, 3, 4], dtype=np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a >= b

        result_np = np.from_dlpack(c)
        expected = np.greater_equal(a_np, b_np)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.bool_


class TestUnaryElementwiseOps:
    """Tests for unary elementwise Mojo ops."""

    @pytest.mark.parametrize("dtype", SIGNED_DTYPES)
    def test_negative(self, dtype: DType) -> None:
        """Test negative op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(-6, 6, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = -x

        expected = np.negative(x_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", SIGNED_DTYPES)
    def test_abs(self, dtype: DType) -> None:
        """Test abs op matches numpy for signed types."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(-6, 6, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = abs(x)

        expected = np.abs(x_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", UINT_DTYPES)
    def test_abs_unsigned(self, dtype: DType) -> None:
        """Test abs op matches numpy for unsigned types."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        # Use non-negative values for unsigned types
        x_np = np.arange(0, 12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = abs(x)

        expected = np.abs(x_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_exp(self, dtype: DType) -> None:
        """Test exp op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.linspace(-2, 2, 12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.exp(x)

        expected = np.exp(x_np)
        np.testing.assert_array_almost_equal(
            np.from_dlpack(y), expected, decimal=5
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_log(self, dtype: DType) -> None:
        """Test log op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.linspace(0.1, 10, 12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.log(x)

        expected = np.log(x_np)
        np.testing.assert_array_almost_equal(
            np.from_dlpack(y), expected, decimal=5
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_sqrt(self, dtype: DType) -> None:
        """Test sqrt op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.linspace(0, 10, 12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.sqrt(x)

        expected = np.sqrt(x_np)
        np.testing.assert_array_almost_equal(
            np.from_dlpack(y), expected, decimal=5
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_tanh(self, dtype: DType) -> None:
        """Test tanh op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.linspace(-3, 3, 12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.tanh(x)

        expected = np.tanh(x_np)
        np.testing.assert_array_almost_equal(
            np.from_dlpack(y), expected, decimal=5
        )

    @pytest.mark.xfail(
        raises=UnsupportedGraphError,
        reason="mo.relu has no interpreter handler",
        strict=True,
    )
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_relu(self, dtype: DType) -> None:
        """Test relu op matches numpy maximum(x, 0)."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.linspace(-3, 3, 12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.relu(x)

        expected = np.maximum(x_np, 0)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_sin(self, dtype: DType) -> None:
        """Test sin op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.linspace(-np.pi, np.pi, 12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.sin(x)

        expected = np.sin(x_np)
        np.testing.assert_array_almost_equal(
            np.from_dlpack(y), expected, decimal=5
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cos(self, dtype: DType) -> None:
        """Test cos op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.linspace(-np.pi, np.pi, 12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.cos(x)

        expected = np.cos(x_np)
        np.testing.assert_array_almost_equal(
            np.from_dlpack(y), expected, decimal=5
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_floor(self, dtype: DType) -> None:
        """Test floor op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.linspace(-2.5, 2.5, 12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.floor(x)

        expected = np.floor(x_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)


class TestUnaryMixedOps:
    """Tests for unary mixed-dtype Mojo ops (cast, is_nan, is_inf)."""

    @pytest.mark.parametrize(
        "in_dtype,out_dtype",
        [
            (DType.float32, DType.int32),
            (DType.float64, DType.float32),
            (DType.int32, DType.float64),
            (DType.int32, DType.float32),
            (DType.float32, DType.float64),
            (DType.int8, DType.int32),
            (DType.uint8, DType.float32),
            (DType.float32, DType.int64),
            (DType.int64, DType.float32),
            (DType.uint8, DType.uint16),
            (DType.int32, DType.bool),
            (DType.bool, DType.int32),
            (DType.float32, DType.bool),
        ],
    )
    def test_cast(self, in_dtype: DType, out_dtype: DType) -> None:
        """Test cast op converts dtype correctly."""
        in_np_dtype = in_dtype.to_numpy()
        out_np_dtype = out_dtype.to_numpy()
        # numpy rejects arange with an explicit bool dtype past 2 elements, so
        # build the ramp in the default dtype and cast it to the input dtype.
        x_np = np.arange(12).astype(in_np_dtype).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.cast(out_dtype)

        result_np = np.from_dlpack(y)
        expected = x_np.astype(out_np_dtype)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == out_np_dtype

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_is_nan(self, dtype: DType) -> None:
        """Test is_nan op detects NaN values."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([1.0, np.nan, 3.0, np.nan, np.inf, 0.0], dtype=np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.is_nan(x)

        result_np = np.from_dlpack(y)
        expected = np.isnan(x_np)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.bool_

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_is_inf(self, dtype: DType) -> None:
        """Test is_inf op detects Inf values."""
        np_dtype = dtype.to_numpy()
        x_np = np.array(
            [1.0, np.inf, -np.inf, np.nan, 0.0, 42.0], dtype=np_dtype
        )

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.is_inf(x)

        result_np = np.from_dlpack(y)
        expected = np.isinf(x_np)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.bool_

    def test_cast_identity(self) -> None:
        """Test cast to same dtype is identity."""
        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.cast(DType.float32)

        np.testing.assert_array_equal(np.from_dlpack(y), x_np)

    def test_cast_float_to_int_truncation(self) -> None:
        """Test cast from float to int truncates toward zero."""
        x_np = np.array([1.7, -2.3, 3.9, -4.1], dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.cast(DType.int32)

        expected = x_np.astype(np.int32)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_is_nan_all_normal(self) -> None:
        """Test is_nan returns all False for normal values."""
        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.is_nan(x)

        np.testing.assert_array_equal(
            np.from_dlpack(y), np.array([False, False, False, False])
        )

    @pytest.mark.parametrize(
        "in_dtype,out_dtype",
        [
            # Signed integer narrowing: values exceed target range
            (DType.int32, DType.int8),
            (DType.int64, DType.int16),
            (DType.int32, DType.int16),
            (DType.int64, DType.int8),
        ],
    )
    def test_cast_signed_integer_overflow(
        self, in_dtype: DType, out_dtype: DType
    ) -> None:
        """Test cast with signed integer values that overflow the target type."""
        in_np_dtype = in_dtype.to_numpy()
        out_np_dtype = out_dtype.to_numpy()
        # Values that exceed target range (e.g., 200 overflows int8 [-128,127])
        x_np = np.array(
            [200, -200, 1000, -1000, 0, 127, 128], dtype=in_np_dtype
        )

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.cast(out_dtype)

        result_np = np.from_dlpack(y)
        expected = x_np.astype(out_np_dtype)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == out_np_dtype

    @pytest.mark.parametrize(
        "in_dtype,out_dtype",
        [
            (DType.uint32, DType.int8),
            (DType.uint16, DType.int8),
            (DType.uint32, DType.uint8),
        ],
    )
    def test_cast_unsigned_integer_overflow(
        self, in_dtype: DType, out_dtype: DType
    ) -> None:
        """Test cast with unsigned integer values that overflow the target."""
        in_np_dtype = in_dtype.to_numpy()
        out_np_dtype = out_dtype.to_numpy()
        # Positive values that exceed target range
        x_np = np.array(
            [200, 300, 1000, 65535, 0, 127, 128, 255, 256],
            dtype=in_np_dtype,
        )

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.cast(out_dtype)

        result_np = np.from_dlpack(y)
        expected = x_np.astype(out_np_dtype)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == out_np_dtype

    def test_cast_float64_to_float32_precision_loss(self) -> None:
        """Test cast from float64 to float32 loses precision."""
        # Use values that have more precision than float32 can represent.
        # Avoid subnormal float32 values (< ~1.18e-38) which may be
        # flushed to zero by SIMD worker threads with FTZ enabled.
        x_np = np.array(
            [1.0000000000000002, 1.23456789012345678, 1e-30, 1e38],
            dtype=np.float64,
        )

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.cast(DType.float32)

        result_np = np.from_dlpack(y)
        expected = x_np.astype(np.float32)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.float32

    def test_cast_float_to_int_narrowing(self) -> None:
        """Test cast from float to narrow int with truncation and wrapping."""
        # Use float32→int32 with fractional values to test truncation
        x_np = np.array(
            [1e9, -1e9, 1.5e9, -1.5e9, 0.0, 1.0, -1.0], dtype=np.float32
        )

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.cast(DType.int32)

        result_np = np.from_dlpack(y)
        expected = x_np.astype(np.int32)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.int32


class TestBooleanLogicOps:
    """Tests for boolean logic Mojo ops."""

    def test_and(self) -> None:
        """Test logical and op."""
        a_np = np.array([True, True, False, False], dtype=np.bool_)
        b_np = np.array([True, False, True, False], dtype=np.bool_)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a & b

        expected = np.logical_and(a_np, b_np)
        np.testing.assert_array_equal(np.from_dlpack(c), expected)

    def test_or(self) -> None:
        """Test logical or op."""
        a_np = np.array([True, True, False, False], dtype=np.bool_)
        b_np = np.array([True, False, True, False], dtype=np.bool_)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a | b

        expected = np.logical_or(a_np, b_np)
        np.testing.assert_array_equal(np.from_dlpack(c), expected)

    def test_xor(self) -> None:
        """Test logical xor op."""
        a_np = np.array([True, True, False, False], dtype=np.bool_)
        b_np = np.array([True, False, True, False], dtype=np.bool_)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a ^ b

        expected = np.logical_xor(a_np, b_np)
        np.testing.assert_array_equal(np.from_dlpack(c), expected)

    def test_not(self) -> None:
        """Test logical not op."""
        x_np = np.array([True, False, True, False], dtype=np.bool_)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = ~x

        expected = np.logical_not(x_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)


class TestChainedOperations:
    """Tests for chained operations using Mojo ops."""

    def test_chained_arithmetic(self) -> None:
        """Test chained add/sub/mul operations."""
        shape = [3, 4]
        x_np = np.arange(12, dtype=np.float32).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            # (x + 1) * 2 - 3
            one = Tensor.from_dlpack(np.ones(shape, dtype=np.float32))
            two = Tensor.from_dlpack(np.full(shape, 2.0, dtype=np.float32))
            three = Tensor.from_dlpack(np.full(shape, 3.0, dtype=np.float32))
            result = (x + one) * two - three

        expected = (x_np + 1) * 2 - 3
        np.testing.assert_array_almost_equal(np.from_dlpack(result), expected)

    def test_comparison_with_arithmetic(self) -> None:
        """Test combining comparisons with arithmetic operations."""
        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            # Compare (x * 2) > 5
            two = Tensor.from_dlpack(np.full([4], 2.0, dtype=np.float32))
            five = Tensor.from_dlpack(np.full([4], 5.0, dtype=np.float32))
            result = (x * two) > five

        result_np = np.from_dlpack(result)
        expected = (x_np * 2) > 5
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.bool_


class TestBasicOpExecution:
    """Tests for basic op execution through the interpreter."""

    def test_add_two_constants(self) -> None:
        """Test adding two constants."""
        a = Tensor.from_dlpack(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        )
        b = Tensor.from_dlpack(
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        )
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a + b

        expected = np.array([[6.0, 8.0], [10.0, 12.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    def test_mul_two_constants(self) -> None:
        """Test multiplying two constants."""
        a = Tensor.from_dlpack(np.array([2.0, 3.0, 4.0], dtype=np.float32))
        b = Tensor.from_dlpack(np.array([5.0, 6.0, 7.0], dtype=np.float32))
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a * b

        expected = np.array([10.0, 18.0, 28.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    def test_unary_operations(self) -> None:
        """Test unary operations like exp, sqrt, tanh."""
        x = Tensor.from_dlpack(np.array([0.0, 1.0, 2.0], dtype=np.float32))
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.exp(x)

        expected = np.exp(np.array([0.0, 1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(
            np.from_dlpack(result), expected, decimal=5
        )


class TestDataPassthrough:
    """Tests for data passthrough via the interpreter."""

    def test_passthrough_basic(self) -> None:
        """Test that data passes through correctly via interpreter."""
        input_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        zeros_np = np.zeros((3, 4), dtype=np.float32)

        x = Tensor.from_dlpack(input_np)
        z = Tensor.from_dlpack(zeros_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = x + z

        np.testing.assert_array_almost_equal(np.from_dlpack(result), input_np)

    def test_passthrough_multiple_dtypes(self) -> None:
        """Test data passes through correctly with different dtypes."""
        for np_dtype in [np.float32, np.float64, np.int32, np.int64]:
            input_np = np.array([1, 2, 3, 4], dtype=np_dtype)
            zeros_np = np.zeros([4], dtype=np_dtype)

            x = Tensor.from_dlpack(input_np)
            z = Tensor.from_dlpack(zeros_np)
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                result = x + z

            np.testing.assert_array_equal(np.from_dlpack(result), input_np)

    def test_passthrough_preserves_shape(self) -> None:
        """Test that operations preserve tensor shape."""
        for shape in [[4], [2, 3], [2, 3, 4], [1, 2, 3, 4]]:
            size = 1
            for dim in shape:
                size *= dim
            input_np = np.arange(size, dtype=np.float32).reshape(shape)
            zeros_np = np.zeros(shape, dtype=np.float32)

            x = Tensor.from_dlpack(input_np)
            z = Tensor.from_dlpack(zeros_np)
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                result = x + z

            result_np = np.from_dlpack(result)
            assert result_np.shape == tuple(shape)
            np.testing.assert_array_almost_equal(result_np, input_np)


class TestShapeOps:
    """Tests for shape operations (rebind, broadcast_to) in the interpreter."""

    def test_broadcast_to_static_shape(self) -> None:
        """Test that broadcast_to correctly broadcasts to a static target shape."""
        input_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[2, 3])

        expected = np.array(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32
        )
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_broadcast_to_higher_rank(self) -> None:
        """Test broadcasting to a higher rank tensor."""
        input_np = np.array([5.0], dtype=np.float32)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[2, 3, 4])

        expected = np.full((2, 3, 4), 5.0, dtype=np.float32)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_broadcast_to_2d_to_3d(self) -> None:
        """Test broadcasting a 2D tensor to 3D."""
        input_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[3, 2, 2])

        expected = np.broadcast_to(input_np, (3, 2, 2))
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_broadcast_then_add(self) -> None:
        """Test broadcasting followed by element-wise operation."""
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y_np = np.array(
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32
        )

        x = Tensor.from_dlpack(x_np)
        y = Tensor.from_dlpack(y_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            x_broadcast = x.broadcast_to(shape=[2, 3])
            z = x_broadcast + y

        expected = np.array(
            [[11.0, 22.0, 33.0], [41.0, 52.0, 63.0]], dtype=np.float32
        )
        np.testing.assert_array_almost_equal(np.from_dlpack(z), expected)


class TestNonMaximumSuppressionOp:
    """Tests for NMS interpreter op (mo.non_maximum_suppression).

    Routes through F.non_maximum_suppression -> ops.non_maximum_suppression ->
    rmo.MoNonMaximumSuppressionOp -> mo.NonMaximumSuppressionOp ->
    _handle_non_maximum_suppression -> nms_gc.nms_model. CPU-only (MO_HostOnly).

    The reference is a pure-NumPy greedy NMS implementation applied
    independently per (batch, class) pair.
    """

    @staticmethod
    def _nms_numpy_reference(
        boxes_np: np.ndarray,
        scores_np: np.ndarray,
        max_output_boxes_per_class: int,
        iou_threshold: float,
        score_threshold: float,
    ) -> np.ndarray:
        """Pure-NumPy greedy NMS reference.

        Args:
            boxes_np: [batch, num_boxes, 4] float array (y1, x1, y2, x2).
            scores_np: [batch, num_classes, num_boxes] float array.
            max_output_boxes_per_class: Max selections per (batch, class).
            iou_threshold: IoU suppression threshold.
            score_threshold: Minimum score threshold.

        Returns:
            [num_selected, 3] int64 array with rows
            [batch_idx, class_idx, box_idx].
        """
        batch_size, num_boxes, _ = boxes_np.shape
        num_classes = scores_np.shape[1]
        results: list[list[int]] = []

        for b in range(batch_size):
            for c in range(num_classes):
                sc = scores_np[b, c]
                # Filter by score threshold.
                candidates = [
                    i for i in range(num_boxes) if sc[i] > score_threshold
                ]
                # Sort by score descending.
                candidates.sort(key=lambda i: -sc[i])

                selected: list[int] = []
                suppressed = set()
                for idx in candidates:
                    if idx in suppressed:
                        continue
                    if len(selected) >= max_output_boxes_per_class:
                        break
                    selected.append(idx)
                    # Suppress overlapping boxes.
                    bx = boxes_np[b]
                    y1_a, x1_a, y2_a, x2_a = bx[idx]
                    ay1, ay2 = min(y1_a, y2_a), max(y1_a, y2_a)
                    ax1, ax2 = min(x1_a, x2_a), max(x1_a, x2_a)
                    area_a = (ay2 - ay1) * (ax2 - ax1)

                    for other in candidates:
                        if other in suppressed or other == idx:
                            continue
                        y1_b, x1_b, y2_b, x2_b = bx[other]
                        by1, by2 = min(y1_b, y2_b), max(y1_b, y2_b)
                        bx1, bx2 = min(x1_b, x2_b), max(x1_b, x2_b)
                        area_b = (by2 - by1) * (bx2 - bx1)

                        iy1 = max(ay1, by1)
                        ix1 = max(ax1, bx1)
                        iy2 = min(ay2, by2)
                        ix2 = min(ax2, bx2)
                        inter = max(0.0, iy2 - iy1) * max(0.0, ix2 - ix1)
                        union = area_a + area_b - inter
                        if union > 0 and inter / union > iou_threshold:
                            suppressed.add(other)

                for box_idx in selected:
                    results.append([b, c, box_idx])

        if not results:
            return np.zeros((0, 3), dtype=np.int64)
        return np.array(results, dtype=np.int64)

    @pytest.mark.parametrize("dtype", [DType.float32, DType.float64])
    def test_nms_basic(self, dtype: DType) -> None:
        """Test NMS with 1 batch, 1 class, 6 overlapping boxes.

        Parametrized over boxes/scores dtype; thresholds stay float32.
        """
        np_dtype = dtype.to_numpy()
        boxes_np = np.array(
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.05, 0.05, 1.05, 1.05],
                    [2.0, 2.0, 3.0, 3.0],
                    [2.1, 2.1, 3.1, 3.1],
                    [5.0, 5.0, 6.0, 6.0],
                    [8.0, 8.0, 9.0, 9.0],
                ]
            ],
            dtype=np_dtype,
        )
        scores_np = np.array(
            [[[0.9, 0.75, 0.6, 0.5, 0.4, 0.3]]],
            dtype=np_dtype,
        )

        max_out = 10
        iou_thresh = 0.5
        score_thresh = 0.0

        boxes = Tensor.from_dlpack(boxes_np)
        scores = Tensor.from_dlpack(scores_np)
        max_out_t = Tensor.from_dlpack(np.array(max_out, dtype=np.int64))
        iou_t = Tensor.from_dlpack(np.array(iou_thresh, dtype=np.float32))
        score_t = Tensor.from_dlpack(np.array(score_thresh, dtype=np.float32))

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.non_maximum_suppression(
                boxes, scores, max_out_t, iou_t, score_t
            )

        result_np = np.from_dlpack(result)
        ref = self._nms_numpy_reference(
            boxes_np, scores_np, max_out, iou_thresh, score_thresh
        )
        np.testing.assert_array_equal(result_np, ref)

    def test_nms_multi_class(self) -> None:
        """Test NMS with 1 batch, 2 classes (NMS runs per class)."""
        boxes_np = np.array(
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.05, 0.05, 1.05, 1.05],
                    [5.0, 5.0, 6.0, 6.0],
                ]
            ],
            dtype=np.float32,
        )
        scores_np = np.array(
            [[[0.9, 0.8, 0.3], [0.1, 0.95, 0.5]]],
            dtype=np.float32,
        )

        max_out = 10
        iou_thresh = 0.5
        score_thresh = 0.0

        boxes = Tensor.from_dlpack(boxes_np)
        scores = Tensor.from_dlpack(scores_np)
        max_out_t = Tensor.from_dlpack(np.array(max_out, dtype=np.int64))
        iou_t = Tensor.from_dlpack(np.array(iou_thresh, dtype=np.float32))
        score_t = Tensor.from_dlpack(np.array(score_thresh, dtype=np.float32))

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.non_maximum_suppression(
                boxes, scores, max_out_t, iou_t, score_t
            )

        result_np = np.from_dlpack(result)
        ref = self._nms_numpy_reference(
            boxes_np, scores_np, max_out, iou_thresh, score_thresh
        )
        np.testing.assert_array_equal(result_np, ref)

    def test_nms_multi_batch(self) -> None:
        """Test NMS with 2 batches, 1 class each."""
        boxes_np = np.array(
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.05, 0.05, 1.05, 1.05],
                ],
                [
                    [2.0, 2.0, 3.0, 3.0],
                    [5.0, 5.0, 6.0, 6.0],
                ],
            ],
            dtype=np.float32,
        )
        scores_np = np.array(
            [[[0.9, 0.8]], [[0.7, 0.6]]],
            dtype=np.float32,
        )

        max_out = 10
        iou_thresh = 0.5
        score_thresh = 0.0

        boxes = Tensor.from_dlpack(boxes_np)
        scores = Tensor.from_dlpack(scores_np)
        max_out_t = Tensor.from_dlpack(np.array(max_out, dtype=np.int64))
        iou_t = Tensor.from_dlpack(np.array(iou_thresh, dtype=np.float32))
        score_t = Tensor.from_dlpack(np.array(score_thresh, dtype=np.float32))

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.non_maximum_suppression(
                boxes, scores, max_out_t, iou_t, score_t
            )

        result_np = np.from_dlpack(result)
        ref = self._nms_numpy_reference(
            boxes_np, scores_np, max_out, iou_thresh, score_thresh
        )
        np.testing.assert_array_equal(result_np, ref)

    def test_nms_score_threshold(self) -> None:
        """Test that boxes below score_threshold are excluded."""
        boxes_np = np.array(
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [5.0, 5.0, 6.0, 6.0],
                    [10.0, 10.0, 11.0, 11.0],
                ]
            ],
            dtype=np.float32,
        )
        scores_np = np.array(
            [[[0.9, 0.3, 0.1]]],
            dtype=np.float32,
        )

        max_out = 10
        iou_thresh = 0.5
        score_thresh = 0.5

        boxes = Tensor.from_dlpack(boxes_np)
        scores = Tensor.from_dlpack(scores_np)
        max_out_t = Tensor.from_dlpack(np.array(max_out, dtype=np.int64))
        iou_t = Tensor.from_dlpack(np.array(iou_thresh, dtype=np.float32))
        score_t = Tensor.from_dlpack(np.array(score_thresh, dtype=np.float32))

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.non_maximum_suppression(
                boxes, scores, max_out_t, iou_t, score_t
            )

        result_np = np.from_dlpack(result)
        ref = self._nms_numpy_reference(
            boxes_np, scores_np, max_out, iou_thresh, score_thresh
        )
        np.testing.assert_array_equal(result_np, ref)
        assert result_np.shape[0] == 1

    def test_nms_max_output_boxes(self) -> None:
        """Test that max_output_boxes_per_class caps selections."""
        boxes_np = np.array(
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [5.0, 5.0, 6.0, 6.0],
                    [10.0, 10.0, 11.0, 11.0],
                ]
            ],
            dtype=np.float32,
        )
        scores_np = np.array(
            [[[0.9, 0.8, 0.7]]],
            dtype=np.float32,
        )

        max_out = 2
        iou_thresh = 0.5
        score_thresh = 0.0

        boxes = Tensor.from_dlpack(boxes_np)
        scores = Tensor.from_dlpack(scores_np)
        max_out_t = Tensor.from_dlpack(np.array(max_out, dtype=np.int64))
        iou_t = Tensor.from_dlpack(np.array(iou_thresh, dtype=np.float32))
        score_t = Tensor.from_dlpack(np.array(score_thresh, dtype=np.float32))

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.non_maximum_suppression(
                boxes, scores, max_out_t, iou_t, score_t
            )

        result_np = np.from_dlpack(result)
        ref = self._nms_numpy_reference(
            boxes_np, scores_np, max_out, iou_thresh, score_thresh
        )
        np.testing.assert_array_equal(result_np, ref)
        assert result_np.shape[0] == 2

    def test_nms_all_suppressed(self) -> None:
        """Test heavy overlap — only 1 box survives."""
        boxes_np = np.array(
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.01, 0.01, 1.01, 1.01],
                    [0.02, 0.02, 1.02, 1.02],
                ]
            ],
            dtype=np.float32,
        )
        scores_np = np.array(
            [[[0.9, 0.8, 0.7]]],
            dtype=np.float32,
        )

        max_out = 10
        iou_thresh = 0.5
        score_thresh = 0.0

        boxes = Tensor.from_dlpack(boxes_np)
        scores = Tensor.from_dlpack(scores_np)
        max_out_t = Tensor.from_dlpack(np.array(max_out, dtype=np.int64))
        iou_t = Tensor.from_dlpack(np.array(iou_thresh, dtype=np.float32))
        score_t = Tensor.from_dlpack(np.array(score_thresh, dtype=np.float32))

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.non_maximum_suppression(
                boxes, scores, max_out_t, iou_t, score_t
            )

        result_np = np.from_dlpack(result)
        ref = self._nms_numpy_reference(
            boxes_np, scores_np, max_out, iou_thresh, score_thresh
        )
        np.testing.assert_array_equal(result_np, ref)
        assert result_np.shape[0] == 1

    def test_nms_empty_output(self) -> None:
        """Test all scores below threshold — empty output."""
        boxes_np = np.array(
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [5.0, 5.0, 6.0, 6.0],
                ]
            ],
            dtype=np.float32,
        )
        scores_np = np.array(
            [[[0.1, 0.2]]],
            dtype=np.float32,
        )

        max_out = 10
        iou_thresh = 0.5
        score_thresh = 0.5

        boxes = Tensor.from_dlpack(boxes_np)
        scores = Tensor.from_dlpack(scores_np)
        max_out_t = Tensor.from_dlpack(np.array(max_out, dtype=np.int64))
        iou_t = Tensor.from_dlpack(np.array(iou_thresh, dtype=np.float32))
        score_t = Tensor.from_dlpack(np.array(score_thresh, dtype=np.float32))

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.non_maximum_suppression(
                boxes, scores, max_out_t, iou_t, score_t
            )

        result_np = np.from_dlpack(result)
        assert result_np.shape == (0, 3)

    def test_nms_gc_model_returns_exact_selection_count(self) -> None:
        """A compiled NMS model sizes its own data-dependent output."""
        # Two heavily-overlapping boxes in one batch and one class; with
        # max_output_boxes_per_class=1 exactly one survives.
        boxes = Buffer.from_numpy(
            np.array(
                [[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1]]],
                dtype=np.float32,
            )
        )
        scores = Buffer.from_numpy(np.array([[[0.9, 0.8]]], dtype=np.float32))
        model = nms_gc.nms_model(DType.float32)
        (out,) = model(
            boxes,
            scores,
            Buffer.from_numpy(np.asarray(1, dtype=np.int64)),
            Buffer.from_numpy(np.asarray(0.5, dtype=np.float32)),
            Buffer.from_numpy(np.asarray(0.0, dtype=np.float32)),
        )
        # One row: [batch_index, class_index, box_index] for the winner.
        np.testing.assert_array_equal(out.to_numpy(), np.array([[0, 0, 0]]))

    def test_float16_unsupported(self) -> None:
        """float16 isn't in this family's CPU dtype set."""
        with pytest.raises(KeyError, match="float16"):
            nms_gc.nms_model(DType.float16)


class TestInterpreterVsCompiled:
    """Tests comparing interpreter results to compiled execution."""

    def test_interpreter_matches_compiled_add(self) -> None:
        """Test that interpreter add matches compiled add."""
        a_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b_np = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)

        # Execute via interpreter
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            interp_result = a + b

        # Execute via compiled path
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            compiled_result = a + b

        # Results should match
        np.testing.assert_array_almost_equal(
            np.from_dlpack(interp_result), np.from_dlpack(compiled_result)
        )

    def test_interpreter_matches_compiled_mul(self) -> None:
        """Test that interpreter mul matches compiled mul."""
        a_np = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        b_np = np.array([5.0, 6.0, 7.0], dtype=np.float32)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            interp_result = a * b

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            compiled_result = a * b

        np.testing.assert_array_almost_equal(
            np.from_dlpack(interp_result), np.from_dlpack(compiled_result)
        )

    def test_interpreter_matches_compiled_chained(self) -> None:
        """Test that interpreter matches compiled for chained operations."""
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        two_np = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        one_np = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        two = Tensor.from_dlpack(two_np)
        one = Tensor.from_dlpack(one_np)

        # x * 2 + 1
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            interp_result = x * two + one

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            compiled_result = x * two + one

        np.testing.assert_array_almost_equal(
            np.from_dlpack(interp_result), np.from_dlpack(compiled_result)
        )

    def test_interpreter_matches_compiled_nms(self) -> None:
        """Test that interpreter NMS matches compiled NMS."""
        boxes_np = np.array(
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.05, 0.05, 1.05, 1.05],
                    [5.0, 5.0, 6.0, 6.0],
                ]
            ],
            dtype=np.float32,
        )
        scores_np = np.array(
            [[[0.9, 0.75, 0.4]]],
            dtype=np.float32,
        )

        boxes = Tensor.from_dlpack(boxes_np)
        scores = Tensor.from_dlpack(scores_np)
        max_out_t = Tensor.from_dlpack(np.array(10, dtype=np.int64))
        iou_t = Tensor.from_dlpack(np.array(0.5, dtype=np.float32))
        score_t = Tensor.from_dlpack(np.array(0.0, dtype=np.float32))

        # Execute via interpreter path
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            interp_result = F.non_maximum_suppression(
                boxes, scores, max_out_t, iou_t, score_t
            )

        # Execute via a compiling executor: deterministic, where the jit
        # may interpreter-serve while its background compile is pending.
        with (
            set_default_executor(CompilingExecutor()),
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            compiled_result = F.non_maximum_suppression(
                boxes, scores, max_out_t, iou_t, score_t
            )

        # Results should match
        np.testing.assert_array_equal(
            np.from_dlpack(interp_result), np.from_dlpack(compiled_result)
        )


class TestStaticBroadcastToOp:
    """Tests for StaticBroadcastTo using the Tensor API with MO interpreter."""

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_broadcast_1d_to_2d(self, dtype: DType) -> None:
        """Test broadcasting 1D tensor to 2D."""
        np_dtype = dtype.to_numpy()
        input_np = np.array([1, 2, 3], dtype=np_dtype)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[2, 3])

        result = np.from_dlpack(y)
        expected = np.broadcast_to(input_np, (2, 3))
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_broadcast_1d_to_3d(self, dtype: DType) -> None:
        """Test broadcasting 1D tensor to 3D."""
        np_dtype = dtype.to_numpy()
        input_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np_dtype)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[2, 3, 4])

        result = np.from_dlpack(y)
        expected = np.broadcast_to(input_np, (2, 3, 4))
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_broadcast_2d_to_3d(self, dtype: DType) -> None:
        """Test broadcasting 2D tensor to 3D."""
        np_dtype = dtype.to_numpy()
        input_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np_dtype)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[3, 2, 2])

        result = np.from_dlpack(y)
        expected = np.broadcast_to(input_np, (3, 2, 2))
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_broadcast_size1_dim(self, dtype: DType) -> None:
        """Test broadcasting with size-1 dimension."""
        np_dtype = dtype.to_numpy()
        # Shape [1, 3] -> [4, 3]
        input_np = np.array([[1.0, 2.0, 3.0]], dtype=np_dtype)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[4, 3])

        result = np.from_dlpack(y)
        expected = np.broadcast_to(input_np, (4, 3))
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_broadcast_multiple_size1_dims(self, dtype: DType) -> None:
        """Test broadcasting with multiple size-1 dimensions."""
        np_dtype = dtype.to_numpy()
        # Shape [1, 3, 1] -> [2, 3, 4]
        input_np = np.array([[[1.0], [2.0], [3.0]]], dtype=np_dtype)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[2, 3, 4])

        result = np.from_dlpack(y)
        expected = np.broadcast_to(input_np, (2, 3, 4))
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_broadcast_scalar_like(self, dtype: DType) -> None:
        """Test broadcasting a scalar-like tensor [1] to higher dimensions."""
        np_dtype = dtype.to_numpy()
        input_np = np.array([42.0], dtype=np_dtype)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[2, 3, 4])

        result = np.from_dlpack(y)
        expected = np.full((2, 3, 4), 42.0, dtype=np_dtype)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_broadcast_same_shape(self, dtype: DType) -> None:
        """Test broadcasting when shapes are already compatible (no-op)."""
        np_dtype = dtype.to_numpy()
        input_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np_dtype)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[2, 2])

        result = np.from_dlpack(y)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np_dtype)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_broadcast_to_4d(self, dtype: DType) -> None:
        """Test broadcasting to 4D tensor."""
        np_dtype = dtype.to_numpy()
        input_np = np.array([[1.0, 2.0]], dtype=np_dtype)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[2, 3, 1, 2])

        result = np.from_dlpack(y)
        expected = np.broadcast_to(input_np, (2, 3, 1, 2))
        np.testing.assert_array_equal(result, expected)

    def test_broadcast_integer_types(self) -> None:
        """Test broadcasting with integer types."""
        for dtype in INT_DTYPES:
            np_dtype = dtype.to_numpy()
            input_np = np.array([1, 2, 3], dtype=np_dtype)

            x = Tensor.from_dlpack(input_np)
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                y = x.broadcast_to(shape=[2, 3])

            result = np.from_dlpack(y)
            expected = np.broadcast_to(input_np, (2, 3))
            np.testing.assert_array_equal(result, expected)

    def test_broadcast_preserves_values(self) -> None:
        """Test that broadcast preserves exact values during chained operations."""
        input_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ones_np = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            # Broadcast then add
            x_broadcast = x.broadcast_to(shape=[2, 3])
            ones = Tensor.from_dlpack(ones_np)
            y = x_broadcast + ones

        result = np.from_dlpack(y)
        expected = np.array(
            [[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]], dtype=np.float32
        )
        np.testing.assert_array_equal(result, expected)

    def test_broadcast_rank0_to_rank0(self) -> None:
        """Broadcasting a rank-0 (scalar) tensor to a rank-0 target shape.

        Exercises ``data_movement_gc.broadcast_to``'s rank-0 special case: the
        fixed-rank data_movement_gc graph needs at least one axis, so a
        rank-0 target must fall back to a driver copy rather than a
        compiled model call.
        """
        input_np = np.array(42.0, dtype=np.float32)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[])

        result = np.from_dlpack(y)
        assert result.shape == ()
        np.testing.assert_array_equal(result, input_np)

    def test_broadcast_zero_extent_dim(self) -> None:
        """Broadcasting to a zero-extent target dim must produce an empty
        result, not fail in the padded fixed-rank graph."""
        input_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[0, 3])

        result = np.from_dlpack(y)
        assert result.shape == (0, 3)
        np.testing.assert_array_equal(result, np.broadcast_to(input_np, (0, 3)))


class TestTransposeOp:
    """Tests for TransposeOp using the Tensor API with MO interpreter."""

    @pytest.mark.parametrize(
        "dtype", REARRANGE_FLOAT_DTYPES + INT_DTYPES + UINT_DTYPES
    )
    def test_permute_rank3_non_adjacent(self, dtype: DType) -> None:
        """Test permute with a non-adjacent rank-3 permutation (2, 0, 1)."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(24, dtype=np_dtype).reshape(2, 3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.permute([2, 0, 1])

        result = np.from_dlpack(y)
        expected = np.transpose(x_np, (2, 0, 1))
        np.testing.assert_array_equal(result, expected)

    def test_transpose_rank0_scalar(self) -> None:
        """Transposing a scalar (rank-0) tensor via ``Tensor.transpose``.

        ``ops.transpose`` accepts axes ``-1``/``0`` on a rank-0 tensor and
        emits an empty perm, so ``_handle_transpose`` must special-case
        rank 0 rather than reaching the fixed-rank ``data_movement_gc``
        graph (which needs at least one axis).
        """
        x_np = np.array(3.5, dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.transpose(-1, 0)

        result = np.from_dlpack(y)
        assert result.shape == ()
        np.testing.assert_array_equal(result, x_np)

    def test_transpose_zero_extent_dim(self) -> None:
        """Transposing a tensor with a zero-extent dim must produce the
        permuted empty shape, not fail in the padded fixed-rank graph."""
        x_np = np.empty((0, 3), dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.permute([1, 0])

        result = np.from_dlpack(y)
        assert result.shape == (3, 0)
        np.testing.assert_array_equal(result, np.transpose(x_np, (1, 0)))


class TestMatmulOp:
    """Tests for matmul Mojo op."""

    @pytest.mark.parametrize("dtype", MATMUL_DTYPES)
    def test_matmul_basic(self, dtype: DType) -> None:
        """Test basic 2D matmul matches numpy."""
        np_dtype = dtype.to_numpy()
        # Use small values to avoid overflow for integer types
        a_np = np.arange(12, dtype=np_dtype).reshape(3, 4) % 10
        b_np = np.arange(20, dtype=np_dtype).reshape(4, 5) % 10

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a @ b

        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", MATMUL_DTYPES)
    def test_matmul_square(self, dtype: DType) -> None:
        """Test square matrix matmul."""
        np_dtype = dtype.to_numpy()
        a_np = np.arange(16, dtype=np_dtype).reshape(4, 4) % 5
        b_np = np.arange(16, dtype=np_dtype).reshape(4, 4) % 5

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a @ b

        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_matmul_float_precision(self, dtype: DType) -> None:
        """Test matmul with random floats for precision."""
        np_dtype = dtype.to_numpy()
        np.random.seed(42)
        a_np = np.random.randn(8, 16).astype(np_dtype)
        b_np = np.random.randn(16, 8).astype(np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a @ b

        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_almost_equal(
            np.from_dlpack(c), expected, decimal=5
        )

    def test_matmul_vector(self) -> None:
        """Test matmul with vector-like shapes."""
        a_np = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        b_np = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a @ b

        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)


class TestBatchMatmulOp:
    """Tests for batch matmul Mojo op."""

    @pytest.mark.parametrize("dtype", MATMUL_DTYPES)
    def test_batch_matmul_3d(self, dtype: DType) -> None:
        """Test 3D batch matmul: (2, 3, 4) @ (2, 4, 5)."""
        np_dtype = dtype.to_numpy()
        a_np = np.arange(24, dtype=np_dtype).reshape(2, 3, 4) % 10
        b_np = np.arange(40, dtype=np_dtype).reshape(2, 4, 5) % 10

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a @ b

        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_batch_matmul_4d(self, dtype: DType) -> None:
        """Test 4D batch matmul: (2, 3, 4, 5) @ (2, 3, 5, 6)."""
        np_dtype = dtype.to_numpy()
        a_np = np.arange(120, dtype=np_dtype).reshape(2, 3, 4, 5) % 10
        b_np = np.arange(180, dtype=np_dtype).reshape(2, 3, 5, 6) % 10

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a @ b

        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_batch_matmul_float_precision(self, dtype: DType) -> None:
        """Test batch matmul with random floats for precision."""
        np_dtype = dtype.to_numpy()
        np.random.seed(42)
        a_np = np.random.randn(2, 8, 16).astype(np_dtype)
        b_np = np.random.randn(2, 16, 8).astype(np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a @ b

        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_almost_equal(
            np.from_dlpack(c), expected, decimal=5
        )

    def test_batch_matmul_single_batch(self) -> None:
        """Test batch matmul with single batch dimension."""
        a_np = np.arange(12, dtype=np.float32).reshape(1, 3, 4) % 10
        b_np = np.arange(20, dtype=np.float32).reshape(1, 4, 5) % 10

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a @ b

        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_equal(np.from_dlpack(c), expected)


class TestRangeOp:
    """Tests for range Mojo op via Tensor.arange with typed tensor inputs."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_range_basic(self, dtype: DType) -> None:
        """Test basic range op matches numpy arange."""
        np_dtype = dtype.to_numpy()
        start_t = Tensor.from_dlpack(np.array(0, dtype=np_dtype))
        stop_t = Tensor.from_dlpack(np.array(10, dtype=np_dtype))
        step_t = Tensor.from_dlpack(np.array(1, dtype=np_dtype))

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(
                start_t,
                stop_t,
                step_t,
                out_dim=10,
                dtype=dtype,
                device=CPU(),
            )

        expected = np.arange(0, 10, 1, dtype=np_dtype)
        np.testing.assert_array_almost_equal(np.from_dlpack(t), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_range_with_step(self, dtype: DType) -> None:
        """Test range op with custom step size."""
        np_dtype = dtype.to_numpy()
        start_t = Tensor.from_dlpack(np.array(0, dtype=np_dtype))
        stop_t = Tensor.from_dlpack(np.array(10, dtype=np_dtype))
        step_t = Tensor.from_dlpack(np.array(2, dtype=np_dtype))

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(
                start_t,
                stop_t,
                step_t,
                out_dim=5,
                dtype=dtype,
                device=CPU(),
            )

        expected = np.arange(0, 10, 2, dtype=np_dtype)
        np.testing.assert_array_almost_equal(np.from_dlpack(t), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_range_float_step(self, dtype: DType) -> None:
        """Test range op with float step size."""
        np_dtype = dtype.to_numpy()
        start_t = Tensor.from_dlpack(np.array(0.0, dtype=np_dtype))
        stop_t = Tensor.from_dlpack(np.array(1.0, dtype=np_dtype))
        step_t = Tensor.from_dlpack(np.array(0.25, dtype=np_dtype))

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(
                start_t,
                stop_t,
                step_t,
                out_dim=4,
                dtype=dtype,
                device=CPU(),
            )

        expected = np.arange(0.0, 1.0, 0.25, dtype=np_dtype)
        np.testing.assert_array_almost_equal(np.from_dlpack(t), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_range_negative_step(self, dtype: DType) -> None:
        """Test range op with negative step."""
        np_dtype = dtype.to_numpy()
        start_t = Tensor.from_dlpack(np.array(5, dtype=np_dtype))
        stop_t = Tensor.from_dlpack(np.array(0, dtype=np_dtype))
        step_t = Tensor.from_dlpack(np.array(-1, dtype=np_dtype))

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(
                start_t,
                stop_t,
                step_t,
                out_dim=5,
                dtype=dtype,
                device=CPU(),
            )

        expected = np.arange(5, 0, -1, dtype=np_dtype)
        np.testing.assert_array_almost_equal(np.from_dlpack(t), expected)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_range_int(self, dtype: DType) -> None:
        """Test range op with integer dtypes."""
        np_dtype = dtype.to_numpy()
        start_t = Tensor.from_dlpack(np.array(0, dtype=np_dtype))
        stop_t = Tensor.from_dlpack(np.array(10, dtype=np_dtype))
        step_t = Tensor.from_dlpack(np.array(1, dtype=np_dtype))

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(
                start_t,
                stop_t,
                step_t,
                out_dim=10,
                dtype=dtype,
                device=CPU(),
            )

        expected = np.arange(0, 10, 1, dtype=np_dtype)
        np.testing.assert_array_equal(np.from_dlpack(t), expected)

    def test_range_nonzero_start(self) -> None:
        """Test range op with nonzero start value."""
        start_t = Tensor.from_dlpack(np.array(5, dtype=np.float32))
        stop_t = Tensor.from_dlpack(np.array(15, dtype=np.float32))
        step_t = Tensor.from_dlpack(np.array(2, dtype=np.float32))

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(
                start_t,
                stop_t,
                step_t,
                out_dim=5,
                dtype=DType.float32,
                device=CPU(),
            )

        expected = np.arange(5, 15, 2, dtype=np.float32)
        np.testing.assert_array_almost_equal(np.from_dlpack(t), expected)

    def test_range_gc_model_returns_runtime_length(self) -> None:
        """The compiled model sizes its own output from the host scalars."""
        model = range_gc.range_model(CPU(), DType.float32)
        (out,) = model(
            Buffer.from_numpy(np.asarray(0.0, dtype=np.float32)),
            Buffer.from_numpy(np.asarray(5.0, dtype=np.float32)),
            Buffer.from_numpy(np.asarray(1.0, dtype=np.float32)),
        )
        np.testing.assert_array_equal(
            out.to_numpy(), np.arange(0.0, 5.0, 1.0, dtype=np.float32)
        )

    def test_range_non_divisible_interval_rounds_up(self) -> None:
        """``arange(0, 5, 2)`` now yields 3 elements, not 2.

        The shared shape function rounds up like Python's ``range`` where the
        old handler floored, so callers must now declare ``out_dim=3``.
        """
        start_t = Tensor.from_dlpack(np.array(0, dtype=np.float32))
        stop_t = Tensor.from_dlpack(np.array(5, dtype=np.float32))
        step_t = Tensor.from_dlpack(np.array(2, dtype=np.float32))

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(
                start_t,
                stop_t,
                step_t,
                out_dim=3,
                dtype=DType.float32,
                device=CPU(),
            )

        expected = np.array([0.0, 2.0, 4.0], dtype=np.float32)
        np.testing.assert_array_equal(np.from_dlpack(t), expected)

    def test_range_declared_length_mismatch_raises(self) -> None:
        """A declared length that disagrees with the model's output raises.

        Uses buffer-backed scalars with a deliberately floored ``out_dim``: a
        fully-literal range is constant-folded before the handler runs.
        """
        start_t = Tensor.from_dlpack(np.array(0, dtype=np.float32))
        stop_t = Tensor.from_dlpack(np.array(5, dtype=np.float32))
        step_t = Tensor.from_dlpack(np.array(2, dtype=np.float32))

        with pytest.raises(RuntimeError, match=r"declared.*length.*2.*3"):
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                # Keep a reference: an unreferenced result tensor can be
                # elided by realization as dead code, along with the error.
                t = Tensor.arange(
                    start_t,
                    stop_t,
                    step_t,
                    out_dim=2,
                    dtype=DType.float32,
                    device=CPU(),
                )
                assert t is not None

    def test_float16_unsupported(self) -> None:
        """float16 isn't in this family's CPU dtype set."""
        with pytest.raises(KeyError, match="float16"):
            range_gc.range_model(CPU(), DType.float16)


class TestReduceOps:
    """Tests for reduction Mojo ops."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
    def test_reduce_max_last_axis(self, dtype: DType) -> None:
        """Test reduce_max on the last axis matches numpy."""
        shape = [3, 4, 5]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(60, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.max(axis=-1)

        expected = np.max(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_max_first_axis(self, dtype: DType) -> None:
        """Test reduce_max on the first axis."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3, 4, 5)).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.max(axis=0)

        expected = np.max(x_np, axis=0, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_max_middle_axis(self, dtype: DType) -> None:
        """Test reduce_max on a middle axis."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 3, 4)).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.max(axis=1)

        expected = np.max(x_np, axis=1, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_max_2d(self, dtype: DType) -> None:
        """Test reduce_max on 2D tensor."""
        shape = [4, 5]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(20, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.max(axis=-1)

        expected = np.max(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    # --- ReduceMin tests ---

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
    def test_reduce_min_last_axis(self, dtype: DType) -> None:
        """Test reduce_min on the last axis matches numpy."""
        shape = [3, 4, 5]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(60, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.min(axis=-1)

        expected = np.min(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_min_first_axis(self, dtype: DType) -> None:
        """Test reduce_min on the first axis."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3, 4, 5)).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.min(axis=0)

        expected = np.min(x_np, axis=0, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_min_middle_axis(self, dtype: DType) -> None:
        """Test reduce_min on a middle axis."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 3, 4)).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.min(axis=1)

        expected = np.min(x_np, axis=1, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_min_2d(self, dtype: DType) -> None:
        """Test reduce_min on 2D tensor."""
        shape = [4, 5]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(20, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.min(axis=-1)

        expected = np.min(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    # --- ReduceAdd (sum) tests ---

    @pytest.mark.parametrize(
        "dtype",
        FLOAT_DTYPES + [DType.int32, DType.int64],
    )
    def test_reduce_sum_last_axis(self, dtype: DType) -> None:
        """Test reduce_sum on the last axis matches numpy."""
        shape = [3, 4, 5]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(60, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.sum(axis=-1)

        expected = np.sum(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_sum_first_axis(self, dtype: DType) -> None:
        """Test reduce_sum on the first axis."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3, 4, 5)).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.sum(axis=0)

        expected = np.sum(x_np, axis=0, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_sum_middle_axis(self, dtype: DType) -> None:
        """Test reduce_sum on a middle axis."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 3, 4)).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.sum(axis=1)

        expected = np.sum(x_np, axis=1, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_sum_2d(self, dtype: DType) -> None:
        """Test reduce_sum on 2D tensor."""
        shape = [4, 5]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(20, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.sum(axis=-1)

        expected = np.sum(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    # --- Mean tests ---

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_mean_last_axis(self, dtype: DType) -> None:
        """Test mean on the last axis matches numpy."""
        shape = [3, 4, 5]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(60, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.mean(axis=-1)

        expected = np.mean(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_mean_first_axis(self, dtype: DType) -> None:
        """Test mean on the first axis."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3, 4, 5)).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.mean(axis=0)

        expected = np.mean(x_np, axis=0, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_mean_middle_axis(self, dtype: DType) -> None:
        """Test mean on a middle axis."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 3, 4)).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.mean(axis=1)

        expected = np.mean(x_np, axis=1, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_mean_2d(self, dtype: DType) -> None:
        """Test mean on 2D tensor."""
        shape = [4, 5]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(20, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.mean(axis=-1)

        expected = np.mean(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    # --- ReduceMul (prod) tests ---

    @pytest.mark.parametrize(
        "dtype",
        FLOAT_DTYPES + [DType.int32, DType.int64],
    )
    def test_reduce_mul_last_axis(self, dtype: DType) -> None:
        """Test reduce_mul on the last axis matches numpy."""
        shape = [3, 4, 5]
        np_dtype = dtype.to_numpy()
        # Use small values to avoid overflow
        x_np = np.arange(1, 61, dtype=np_dtype).reshape(shape) * 0.1 + 1
        x_np = x_np.astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.prod(axis=-1)

        expected = np.prod(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_mul_first_axis(self, dtype: DType) -> None:
        """Test reduce_mul on the first axis."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = (rng.standard_normal((3, 4, 5)) * 0.5 + 1).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.prod(axis=0)

        expected = np.prod(x_np, axis=0, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_mul_middle_axis(self, dtype: DType) -> None:
        """Test reduce_mul on a middle axis."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = (rng.standard_normal((2, 3, 4)) * 0.5 + 1).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.prod(axis=1)

        expected = np.prod(x_np, axis=1, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_mul_2d(self, dtype: DType) -> None:
        """Test reduce_mul on 2D tensor."""
        shape = [4, 5]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(1, 21, dtype=np_dtype).reshape(shape) * 0.1 + 1
        x_np = x_np.astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.prod(axis=-1)

        expected = np.prod(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    # --- Bool reduce tests (issue #6067) ---

    def test_reduce_max_bool(self) -> None:
        """Test reduce_max on a bool tensor (logical OR)."""
        x_np = np.array(
            [[True, False, True], [False, False, False]], dtype=np.bool_
        )

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.max(axis=-1)

        expected = np.max(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_reduce_min_bool(self) -> None:
        """Test reduce_min on a bool tensor (logical AND)."""
        x_np = np.array(
            [[True, False, True], [True, True, True]], dtype=np.bool_
        )

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.min(axis=-1)

        expected = np.min(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_reduce_max_bool_first_axis(self) -> None:
        """Test reduce_max on a bool tensor along the first axis."""
        x_np = np.array(
            [[False, True, False], [False, False, True]], dtype=np.bool_
        )

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.max(axis=0)

        expected = np.max(x_np, axis=0, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_reduce_max_bool_all_false(self) -> None:
        """Test reduce_max on an all-False bool tensor."""
        x_np = np.zeros((2, 3), dtype=np.bool_)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.max(axis=-1)

        expected = np.max(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_reduce_max_bool_all_true(self) -> None:
        """Test reduce_max on an all-True bool tensor."""
        x_np = np.ones((2, 3), dtype=np.bool_)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.max(axis=-1)

        expected = np.max(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    # --- Additional tests requested in PR review ---

    def test_reduce_min_bool_all_false(self) -> None:
        """Test reduce_min on an all-False bool tensor."""
        x_np = np.zeros((2, 3), dtype=np.bool_)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.min(axis=-1)

        expected = np.min(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_reduce_min_bool_all_true(self) -> None:
        """Test reduce_min on an all-True bool tensor."""
        x_np = np.ones((2, 3), dtype=np.bool_)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.min(axis=-1)

        expected = np.min(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_reduce_bool_3d(self) -> None:
        """Test reduce max/min on a 3D bool tensor across each axis."""
        x_np = np.array(
            [
                [[True, False], [False, True], [True, True]],
                [[False, False], [True, False], [False, True]],
            ],
            dtype=np.bool_,
        )  # shape (2, 3, 2)

        for axis in range(3):
            for op_fn, np_fn in [
                (Tensor.max, np.max),
                (Tensor.min, np.min),
            ]:
                x = Tensor.from_dlpack(x_np)
                with (
                    rc.EagerRealizationContext() as ctx,
                    realization_context(ctx),
                ):
                    y = op_fn(x, axis=axis)

                expected = np_fn(x_np, axis=axis, keepdims=True)
                np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_reduce_bool_4d(self) -> None:
        """Test reduce max/min on a 4D bool tensor across each axis."""
        x_np = (
            np.random.default_rng(42)
            .choice([True, False], size=(2, 2, 3, 2))
            .astype(np.bool_)
        )

        for axis in range(4):
            for op_fn, np_fn in [
                (Tensor.max, np.max),
                (Tensor.min, np.min),
            ]:
                x = Tensor.from_dlpack(x_np)
                with (
                    rc.EagerRealizationContext() as ctx,
                    realization_context(ctx),
                ):
                    y = op_fn(x, axis=axis)

                expected = np_fn(x_np, axis=axis, keepdims=True)
                np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_reduce_bool_single_element_axis(self) -> None:
        """Test reduce max/min on a bool tensor with a size-1 axis."""
        x_np = np.array([[True], [False], [True]], dtype=np.bool_)  # (3, 1)

        for op_fn, np_fn in [
            (Tensor.max, np.max),
            (Tensor.min, np.min),
        ]:
            x = Tensor.from_dlpack(x_np)
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                y = op_fn(x, axis=1)

            expected = np_fn(x_np, axis=1, keepdims=True)
            np.testing.assert_array_equal(np.from_dlpack(y), expected)


def _numpy_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax reference implementation."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x_shifted)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _numpy_logsoftmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable logsoftmax reference implementation."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    return x_shifted - np.log(
        np.sum(np.exp(x_shifted), axis=axis, keepdims=True)
    )


class TestSoftmaxOps:
    """Tests for softmax and logsoftmax Mojo ops."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_softmax_last_axis_3d(self, dtype: DType) -> None:
        """Test softmax on the last axis of a 3D tensor."""
        shape = [3, 4, 5]
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal(shape).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.softmax(x, axis=-1)

        expected = _numpy_softmax(x_np, axis=-1)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_softmax_2d(self, dtype: DType) -> None:
        """Test softmax on a 2D tensor."""
        shape = [4, 5]
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal(shape).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.softmax(x, axis=-1)

        expected = _numpy_softmax(x_np, axis=-1)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_logsoftmax_last_axis_3d(self, dtype: DType) -> None:
        """Test logsoftmax on the last axis of a 3D tensor."""
        shape = [3, 4, 5]
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal(shape).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.logsoftmax(x, axis=-1)

        expected = _numpy_logsoftmax(x_np, axis=-1)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_logsoftmax_2d(self, dtype: DType) -> None:
        """Test logsoftmax on a 2D tensor."""
        shape = [4, 5]
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal(shape).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.logsoftmax(x, axis=-1)

        expected = _numpy_logsoftmax(x_np, axis=-1)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)


class TestBroadcastBinaryOps:
    """Tests for implicit broadcasting in binary ops on CPU.

    These tests exercise the ShapeOfOp -> BroadcastShapeOp -> BroadcastToOp
    chain that gets generated when binary elementwise ops have operands with
    different shapes.
    """

    def test_add_broadcast_1d_2d(self) -> None:
        """Test add with broadcasting: [3] + [2,3] -> [2,3]."""
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b_np = np.array(
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32
        )

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a + b

        expected = np.add(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    def test_mul_broadcast_scalar_like(self) -> None:
        """Test mul with broadcasting: [1] * [3,4] -> [3,4]."""
        a_np = np.array([2.0], dtype=np.float32)
        b_np = np.arange(12, dtype=np.float32).reshape(3, 4)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a * b

        expected = np.multiply(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    def test_sub_broadcast_different_ranks(self) -> None:
        """Test sub with broadcasting: [4] - [2,3,4] -> [2,3,4]."""
        a_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a - b

        expected = np.subtract(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_add_broadcast_size1_dim(self, dtype: DType) -> None:
        """Test add with broadcasting: [1,4] + [3,4] -> [3,4]."""
        np_dtype = dtype.to_numpy()
        a_np = np.arange(4, dtype=np_dtype).reshape(1, 4)
        b_np = np.arange(12, dtype=np_dtype).reshape(3, 4)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a + b

        expected = np.add(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)


class TestRandomNormalOp:
    """Tests for random normal op via max.random.gaussian with interpreter."""

    def test_random_normal_shape_and_dtype(self) -> None:
        """Test that random normal produces correct shape and dtype."""

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result = max_random.gaussian(
                (3, 4), dtype=DType.float32, device=CPU()
            )

        result_np = np.from_dlpack(result)
        assert result_np.shape == (3, 4)
        assert result_np.dtype == np.float32

    def test_random_normal_deterministic(self) -> None:
        """Test that same seed produces identical results."""

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result1 = max_random.gaussian(
                (5, 5), dtype=DType.float32, device=CPU()
            )

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result2 = max_random.gaussian(
                (5, 5), dtype=DType.float32, device=CPU()
            )

        np.testing.assert_array_equal(
            np.from_dlpack(result1), np.from_dlpack(result2)
        )

    def test_random_normal_statistics(self) -> None:
        """Test that random normal has approximately correct mean and std."""

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            set_seed(123)
            result = max_random.gaussian(
                (1000, 1000),
                mean=5.0,
                std=2.0,
                dtype=DType.float32,
                device=CPU(),
            )

        result_np = np.from_dlpack(result)
        # With 1M samples, statistics should be quite close
        np.testing.assert_allclose(result_np.mean(), 5.0, atol=0.1)
        np.testing.assert_allclose(result_np.std(), 2.0, atol=0.1)

    @pytest.mark.parametrize("dtype", [DType.float32, DType.float64])
    def test_random_normal_dtypes(self, dtype: DType) -> None:
        """Test random normal with different float dtypes."""

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result = max_random.gaussian((10, 10), dtype=dtype, device=CPU())

        result_np = np.from_dlpack(result)
        assert result_np.shape == (10, 10)
        assert result_np.dtype == dtype.to_numpy()

    def test_random_normal_matches_recorded_baseline(self) -> None:
        """Migrating to the shared GC kernel must not change sampled values.

        A mismatch means the counter mapping diverged, not that the baseline
        is stale.
        """
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            set_seed(7)
            result = max_random.gaussian(
                (4,), dtype=DType.float32, device=CPU()
            )

        np.testing.assert_array_equal(
            np.from_dlpack(result), _RANDOM_NORMAL_SEED7_F32
        )


class TestRandomUniformOp:
    """Tests for random uniform op via max.random.uniform with interpreter."""

    def test_random_uniform_shape_and_dtype(self) -> None:
        """Test that random uniform produces correct shape and dtype."""

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result = max_random.uniform(
                (3, 4), dtype=DType.float32, device=CPU()
            )

        result_np = np.from_dlpack(result)
        assert result_np.shape == (3, 4)
        assert result_np.dtype == np.float32

    def test_random_uniform_deterministic(self) -> None:
        """Test that same seed produces identical results."""

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result1 = max_random.uniform(
                (5, 5), dtype=DType.float32, device=CPU()
            )

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result2 = max_random.uniform(
                (5, 5), dtype=DType.float32, device=CPU()
            )

        np.testing.assert_array_equal(
            np.from_dlpack(result1), np.from_dlpack(result2)
        )

    def test_random_uniform_statistics(self) -> None:
        """Test that random uniform has approximately correct statistics."""

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            set_seed(123)
            result = max_random.uniform(
                (1000, 1000),
                range=(2.0, 5.0),
                dtype=DType.float32,
                device=CPU(),
            )

        result_np = np.from_dlpack(result)
        # With 1M samples, statistics should be quite close
        np.testing.assert_allclose(result_np.mean(), 3.5, atol=0.1)
        assert result_np.min() >= 2.0
        assert result_np.max() <= 5.0

    @pytest.mark.parametrize("dtype", [DType.float32, DType.float64])
    def test_random_uniform_dtypes(self, dtype: DType) -> None:
        """Test random uniform with different float dtypes."""

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result = max_random.uniform((10, 10), dtype=dtype, device=CPU())

        result_np = np.from_dlpack(result)
        assert result_np.shape == (10, 10)
        assert result_np.dtype == dtype.to_numpy()

    def test_float16_unsupported(self) -> None:
        """float16 isn't in this family's CPU dtype set."""
        with pytest.raises(KeyError, match="float16"):
            random_gc.random_model(mo.RandomUniformOp, CPU(), DType.float16)


class TestShapeChangeOps:
    """Tests for shape change operations (squeeze, unsqueeze, reshape variants).

    These test the reshape semantics that SqueezeShapeOp, UnsqueezeShapeOp,
    AddSingletonDimOp, SplitDimOp, and MergeDimOp implement. Since these ops
    are emitted by MLIR lowering passes rather than the Python API directly,
    we test through the Tensor API methods that produce equivalent reshapes.
    """

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_squeeze_single_dim(self, dtype: DType) -> None:
        """Test squeeze removes a size-1 dimension."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 1, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.squeeze(axis=1)

        result = np.from_dlpack(y)
        expected = np.squeeze(x_np, axis=1)
        assert result.shape == (3, 4)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_squeeze_first_dim(self, dtype: DType) -> None:
        """Test squeeze on the first dimension."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(1, 3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.squeeze(axis=0)

        result = np.from_dlpack(y)
        expected = np.squeeze(x_np, axis=0)
        assert result.shape == (3, 4)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_squeeze_last_dim(self, dtype: DType) -> None:
        """Test squeeze on the last dimension."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4, 1)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.squeeze(axis=-1)

        result = np.from_dlpack(y)
        expected = np.squeeze(x_np, axis=-1)
        assert result.shape == (3, 4)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_unsqueeze_beginning(self, dtype: DType) -> None:
        """Test unsqueeze adds a dimension at the beginning."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.unsqueeze(axis=0)

        result = np.from_dlpack(y)
        expected = np.expand_dims(x_np, axis=0)
        assert result.shape == (1, 3, 4)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_unsqueeze_middle(self, dtype: DType) -> None:
        """Test unsqueeze adds a dimension in the middle."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.unsqueeze(axis=1)

        result = np.from_dlpack(y)
        expected = np.expand_dims(x_np, axis=1)
        assert result.shape == (3, 1, 4)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_unsqueeze_end(self, dtype: DType) -> None:
        """Test unsqueeze adds a dimension at the end."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.unsqueeze(axis=-1)

        result = np.from_dlpack(y)
        expected = np.expand_dims(x_np, axis=-1)
        assert result.shape == (3, 4, 1)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reshape_split_dim(self, dtype: DType) -> None:
        """Test reshape that splits a dimension (equivalent to SplitDimOp).

        E.g., [12, 3] -> [3, 4, 3] splits dimension 0 into (3, 4).
        """
        np_dtype = dtype.to_numpy()
        x_np = np.arange(36, dtype=np_dtype).reshape(12, 3)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.reshape([3, 4, 3])

        result = np.from_dlpack(y)
        expected = x_np.reshape(3, 4, 3)
        assert result.shape == (3, 4, 3)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reshape_merge_dims(self, dtype: DType) -> None:
        """Test reshape that merges adjacent dimensions (equivalent to MergeDimOp).

        E.g., [2, 3, 4] -> [6, 4] merges dimensions 0 and 1.
        """
        np_dtype = dtype.to_numpy()
        x_np = np.arange(24, dtype=np_dtype).reshape(2, 3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.reshape([6, 4])

        result = np.from_dlpack(y)
        expected = x_np.reshape(6, 4)
        assert result.shape == (6, 4)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reshape_add_singleton(self, dtype: DType) -> None:
        """Test reshape that adds a singleton dimension (equiv to AddSingletonDimOp).

        E.g., [3, 4] -> [3, 1, 4] adds a dimension of size 1.
        """
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.reshape([3, 1, 4])

        result = np.from_dlpack(y)
        expected = x_np.reshape(3, 1, 4)
        assert result.shape == (3, 1, 4)
        np.testing.assert_array_equal(result, expected)

    def test_squeeze_then_unsqueeze_roundtrip(self) -> None:
        """Test that squeeze then unsqueeze returns to original shape."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 1, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            squeezed = x.squeeze(axis=1)
            unsqueezed = squeezed.unsqueeze(axis=1)

        result = np.from_dlpack(unsqueezed)
        assert result.shape == (3, 1, 4)
        np.testing.assert_array_equal(result, x_np)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_squeeze_integer_types(self, dtype: DType) -> None:
        """Test squeeze with integer dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(6, dtype=np_dtype).reshape(1, 2, 3)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.squeeze(axis=0)

        result = np.from_dlpack(y)
        expected = np.squeeze(x_np, axis=0)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_unsqueeze_integer_types(self, dtype: DType) -> None:
        """Test unsqueeze with integer dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(6, dtype=np_dtype).reshape(2, 3)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.unsqueeze(axis=0)

        result = np.from_dlpack(y)
        expected = np.expand_dims(x_np, axis=0)
        assert result.shape == (1, 2, 3)
        np.testing.assert_array_equal(result, expected)


class TestSelectOp:
    """Tests for select (where) op via F.where with interpreter."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_select_basic(self, dtype: DType) -> None:
        """Test basic select op matches numpy.where."""
        np_dtype = dtype.to_numpy()
        cond_np = np.array(
            [True, False, True, False, True, False], dtype=np.bool_
        )
        x_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np_dtype)
        y_np = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np_dtype)

        cond = Tensor.from_dlpack(cond_np)
        x = Tensor.from_dlpack(x_np)
        y = Tensor.from_dlpack(y_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.where(cond, x, y)

        expected = np.where(cond_np, x_np, y_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(result), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_select_2d(self, dtype: DType) -> None:
        """Test select op with 2D tensors."""
        np_dtype = dtype.to_numpy()
        cond_np = np.array(
            [[True, False, True], [False, True, False]], dtype=np.bool_
        )
        x_np = np.arange(1, 7, dtype=np_dtype).reshape(2, 3)
        y_np = np.arange(10, 70, 10, dtype=np_dtype).reshape(2, 3)

        cond = Tensor.from_dlpack(cond_np)
        x = Tensor.from_dlpack(x_np)
        y = Tensor.from_dlpack(y_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.where(cond, x, y)

        expected = np.where(cond_np, x_np, y_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(result), expected)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_select_int(self, dtype: DType) -> None:
        """Test select op with integer dtypes."""
        np_dtype = dtype.to_numpy()
        cond_np = np.array([True, False, True, False], dtype=np.bool_)
        x_np = np.array([1, 2, 3, 4], dtype=np_dtype)
        y_np = np.array([10, 20, 30, 40], dtype=np_dtype)

        cond = Tensor.from_dlpack(cond_np)
        x = Tensor.from_dlpack(x_np)
        y = Tensor.from_dlpack(y_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.where(cond, x, y)

        expected = np.where(cond_np, x_np, y_np)
        np.testing.assert_array_equal(np.from_dlpack(result), expected)

    @pytest.mark.parametrize("dtype", UINT_DTYPES)
    def test_select_uint(self, dtype: DType) -> None:
        """Test select op with unsigned integer dtypes."""
        np_dtype = dtype.to_numpy()
        cond_np = np.array([True, False, True, False], dtype=np.bool_)
        x_np = np.array([1, 2, 3, 4], dtype=np_dtype)
        y_np = np.array([10, 20, 30, 40], dtype=np_dtype)

        cond = Tensor.from_dlpack(cond_np)
        x = Tensor.from_dlpack(x_np)
        y = Tensor.from_dlpack(y_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.where(cond, x, y)

        expected = np.where(cond_np, x_np, y_np)
        np.testing.assert_array_equal(np.from_dlpack(result), expected)

    def test_select_bool_values(self) -> None:
        """Test select op where x/y (not just cond) are bool tensors."""
        cond_np = np.array([True, False, True, False], dtype=np.bool_)
        x_np = np.array([True, True, False, False], dtype=np.bool_)
        y_np = np.array([False, False, True, True], dtype=np.bool_)

        cond = Tensor.from_dlpack(cond_np)
        x = Tensor.from_dlpack(x_np)
        y = Tensor.from_dlpack(y_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.where(cond, x, y)

        expected = np.where(cond_np, x_np, y_np)
        np.testing.assert_array_equal(np.from_dlpack(result), expected)

    def test_select_all_true(self) -> None:
        """Test select with all-true condition returns x."""
        cond_np = np.ones(4, dtype=np.bool_)
        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

        cond = Tensor.from_dlpack(cond_np)
        x = Tensor.from_dlpack(x_np)
        y = Tensor.from_dlpack(y_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.where(cond, x, y)

        np.testing.assert_array_equal(np.from_dlpack(result), x_np)

    def test_select_all_false(self) -> None:
        """Test select with all-false condition returns y."""
        cond_np = np.zeros(4, dtype=np.bool_)
        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

        cond = Tensor.from_dlpack(cond_np)
        x = Tensor.from_dlpack(x_np)
        y = Tensor.from_dlpack(y_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.where(cond, x, y)

        np.testing.assert_array_equal(np.from_dlpack(result), y_np)


class TestConcatOp:
    """Tests for concat op via F.concat with interpreter."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_concat_axis0(self, dtype: DType) -> None:
        """Test concat along axis 0."""
        np_dtype = dtype.to_numpy()
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np_dtype)
        b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=0)

        expected = np.concatenate([a_np, b_np], axis=0)
        np.testing.assert_array_almost_equal(np.from_dlpack(result), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_concat_axis1(self, dtype: DType) -> None:
        """Test concat along axis 1."""
        np_dtype = dtype.to_numpy()
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np_dtype)
        b_np = np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=1)

        expected = np.concatenate([a_np, b_np], axis=1)
        np.testing.assert_array_almost_equal(np.from_dlpack(result), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_concat_negative_axis(self, dtype: DType) -> None:
        """Test concat along negative axis (-1 = last dim)."""
        np_dtype = dtype.to_numpy()
        a_np = np.arange(6, dtype=np_dtype).reshape(2, 3)
        b_np = np.arange(4, dtype=np_dtype).reshape(2, 2)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=-1)

        expected = np.concatenate([a_np, b_np], axis=-1)
        np.testing.assert_array_almost_equal(np.from_dlpack(result), expected)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_concat_int_dtypes(self, dtype: DType) -> None:
        """Test concat with integer dtypes."""
        np_dtype = dtype.to_numpy()
        a_np = np.array([1, 2, 3], dtype=np_dtype)
        b_np = np.array([4, 5, 6], dtype=np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=0)

        expected = np.concatenate([a_np, b_np], axis=0)
        np.testing.assert_array_equal(np.from_dlpack(result), expected)

    def test_concat_multiple_tensors(self) -> None:
        """Test concat with more than two tensors."""
        a_np = np.array([[1.0, 2.0]], dtype=np.float32)
        b_np = np.array([[3.0, 4.0]], dtype=np.float32)
        c_np = np.array([[5.0, 6.0]], dtype=np.float32)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        c = Tensor.from_dlpack(c_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b, c], axis=0)

        expected = np.concatenate([a_np, b_np, c_np], axis=0)
        np.testing.assert_array_almost_equal(np.from_dlpack(result), expected)

    def test_concat_single_tensor(self) -> None:
        """Test concat with a single tensor is a no-op."""
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        a = Tensor.from_dlpack(a_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a], axis=0)

        np.testing.assert_array_almost_equal(np.from_dlpack(result), a_np)

    def test_concat_3d(self) -> None:
        """Test concat with 3D tensors."""
        a_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        b_np = np.arange(24, 48, dtype=np.float32).reshape(2, 3, 4)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=0)

        expected = np.concatenate([a_np, b_np], axis=0)
        np.testing.assert_array_almost_equal(np.from_dlpack(result), expected)

    @pytest.mark.parametrize("axis", [0, 1, -1])
    @pytest.mark.parametrize(
        "dtype",
        [DType.uint8, DType.int16, DType.bool, DType.float16],
    )
    def test_concat_extra_dtypes(self, dtype: DType, axis: int) -> None:
        """Concat over dtypes the old Mojo path also accepted.

        float16 has no typed CPU kernel; it exercises the width-16 uint bit-cast.
        """
        a_np = np.array([[1, 0, 1], [0, 1, 1]]).astype(dtype.to_numpy())
        b_np = np.array([[1, 1, 0], [0, 0, 1]]).astype(dtype.to_numpy())

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=axis)

        expected = np.concatenate([a_np, b_np], axis=axis)
        np.testing.assert_array_equal(np.from_dlpack(result), expected)


def _numpy_layer_norm(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Numerically stable layer normalization reference implementation."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


class TestLayerNormOps:
    """Tests for layer_norm Mojo op."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_layer_norm_2d(self, dtype: DType) -> None:
        """Test layer_norm on a 2D tensor."""
        shape = [4, 5]
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal(shape).astype(np_dtype)
        gamma_np = rng.standard_normal(shape[-1]).astype(np_dtype)
        beta_np = rng.standard_normal(shape[-1]).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        gamma = Tensor.from_dlpack(gamma_np)
        beta = Tensor.from_dlpack(beta_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.layer_norm(x, gamma, beta, epsilon=1e-5)

        expected = _numpy_layer_norm(x_np, gamma_np, beta_np, eps=1e-5)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_layer_norm_3d(self, dtype: DType) -> None:
        """Test layer_norm on a 3D tensor."""
        shape = [3, 4, 5]
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal(shape).astype(np_dtype)
        gamma_np = rng.standard_normal(shape[-1]).astype(np_dtype)
        beta_np = rng.standard_normal(shape[-1]).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        gamma = Tensor.from_dlpack(gamma_np)
        beta = Tensor.from_dlpack(beta_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.layer_norm(x, gamma, beta, epsilon=1e-5)

        expected = _numpy_layer_norm(x_np, gamma_np, beta_np, eps=1e-5)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_layer_norm_4d(self, dtype: DType) -> None:
        """Test layer_norm on a 4D tensor."""
        shape = [2, 3, 4, 5]
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal(shape).astype(np_dtype)
        gamma_np = rng.standard_normal(shape[-1]).astype(np_dtype)
        beta_np = rng.standard_normal(shape[-1]).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        gamma = Tensor.from_dlpack(gamma_np)
        beta = Tensor.from_dlpack(beta_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.layer_norm(x, gamma, beta, epsilon=1e-5)

        expected = _numpy_layer_norm(x_np, gamma_np, beta_np, eps=1e-5)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_layer_norm_large_feature_dim(self) -> None:
        """Test layer_norm with a large feature dimension."""
        shape = [8, 128]
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal(shape).astype(np.float32)
        gamma_np = rng.standard_normal(shape[-1]).astype(np.float32)
        beta_np = rng.standard_normal(shape[-1]).astype(np.float32)

        x = Tensor.from_dlpack(x_np)
        gamma = Tensor.from_dlpack(gamma_np)
        beta = Tensor.from_dlpack(beta_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.layer_norm(x, gamma, beta, epsilon=1e-5)

        expected = _numpy_layer_norm(x_np, gamma_np, beta_np, eps=1e-5)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_layer_norm_single_element_feature(self) -> None:
        """Test layer_norm with a single-element feature dimension."""
        shape = [4, 1]
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal(shape).astype(np.float32)
        gamma_np = rng.standard_normal(shape[-1]).astype(np.float32)
        beta_np = rng.standard_normal(shape[-1]).astype(np.float32)

        x = Tensor.from_dlpack(x_np)
        gamma = Tensor.from_dlpack(gamma_np)
        beta = Tensor.from_dlpack(beta_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.layer_norm(x, gamma, beta, epsilon=1e-5)

        expected = _numpy_layer_norm(x_np, gamma_np, beta_np, eps=1e-5)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)


class TestRmsNormOp:
    """Tests for rms_norm interpreter op via F.rms_norm.

    Routes through F.rms_norm -> ops.rms_norm -> mo.ReduceRmsNormOp ->
    _handle_rms_norm -> rms_norm_gc.rms_norm_model.
    """

    @staticmethod
    def _rms_norm_ref(
        x: np.ndarray,
        weight: np.ndarray,
        eps: float,
        weight_offset: float = 0.0,
        multiply_before_cast: bool = False,
    ) -> np.ndarray:
        """Pure-numpy RMS norm reference.

        output = x / rms(x) * (weight + weight_offset)
        rms(x) = sqrt(mean(x^2) + eps)
        """
        x_f64 = x.astype(np.float64)
        rms = np.sqrt(np.mean(x_f64**2, axis=-1, keepdims=True) + eps)
        normed = x_f64 / rms
        w = weight.astype(np.float64) + weight_offset
        return (normed * w).astype(x.dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic_2d(self, dtype: DType) -> None:
        """Test rms_norm on a 2D tensor."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((4, 5)).astype(np_dtype)
        w_np = rng.standard_normal(5).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        w = Tensor.from_dlpack(w_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.rms_norm(x, w, epsilon=1e-5)

        expected = self._rms_norm_ref(x_np, w_np, 1e-5)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-4, atol=1e-4
        )

    def test_3d_input(self) -> None:
        """Test rms_norm on a 3D tensor (batch + sequence + feature)."""
        rng = np.random.default_rng(43)
        x_np = rng.standard_normal((2, 3, 8)).astype(np.float32)
        w_np = rng.standard_normal(8).astype(np.float32)

        x = Tensor.from_dlpack(x_np)
        w = Tensor.from_dlpack(w_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.rms_norm(x, w, epsilon=1e-5)

        expected = self._rms_norm_ref(x_np, w_np, 1e-5)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_multiply_before_cast(self) -> None:
        """Test Gemma-style multiply_before_cast=True."""
        rng = np.random.default_rng(44)
        x_np = rng.standard_normal((4, 6)).astype(np.float32)
        w_np = rng.standard_normal(6).astype(np.float32)

        x = Tensor.from_dlpack(x_np)
        w = Tensor.from_dlpack(w_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.rms_norm(x, w, epsilon=1e-5, multiply_before_cast=True)

        expected = self._rms_norm_ref(
            x_np, w_np, 1e-5, multiply_before_cast=True
        )
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_weight_offset(self) -> None:
        """Test rms_norm with non-zero weight_offset (Gemma-style +1)."""
        rng = np.random.default_rng(45)
        x_np = rng.standard_normal((3, 4)).astype(np.float32)
        w_np = rng.standard_normal(4).astype(np.float32)

        x = Tensor.from_dlpack(x_np)
        w = Tensor.from_dlpack(w_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.rms_norm(x, w, epsilon=1e-5, weight_offset=1.0)

        expected = self._rms_norm_ref(x_np, w_np, 1e-5, weight_offset=1.0)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_large_feature_dim(self) -> None:
        """Test rms_norm with a large feature dimension."""
        rng = np.random.default_rng(46)
        x_np = rng.standard_normal((8, 128)).astype(np.float32)
        w_np = rng.standard_normal(128).astype(np.float32)

        x = Tensor.from_dlpack(x_np)
        w = Tensor.from_dlpack(w_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.rms_norm(x, w, epsilon=1e-6)

        expected = self._rms_norm_ref(x_np, w_np, 1e-6)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )


def _numpy_group_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    num_groups: int,
    eps: float = 1e-5,
) -> np.ndarray:
    """Numerically stable group normalization reference implementation."""
    n, c, *spatial = x.shape
    x_grouped = x.astype(np.float64).reshape(
        n, num_groups, c // num_groups, *spatial
    )
    axes = tuple(range(2, x_grouped.ndim))
    mean = x_grouped.mean(axis=axes, keepdims=True)
    var = x_grouped.var(axis=axes, keepdims=True)
    normed = ((x_grouped - mean) / np.sqrt(var + eps)).reshape(x.shape)
    broadcast_shape = (1, c) + (1,) * len(spatial)
    result = normed * gamma.astype(np.float64).reshape(
        broadcast_shape
    ) + beta.astype(np.float64).reshape(broadcast_shape)
    return result.astype(x.dtype)


class TestGroupNormOp:
    """Tests for group_norm interpreter op via F.group_norm.

    Routes through F.group_norm -> ops.group_norm -> mo.ReduceGroupNormOp ->
    _handle_group_norm -> group_norm_gc.group_norm_model. GPU correctness is
    covered by TestGroupNormGPU in test_interpreter_ops_gpu.py.
    """

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_group_norm_4d(self, dtype: DType) -> None:
        """Test group_norm on a 4D (N, C, H, W) tensor."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(50)
        x_np = rng.standard_normal((2, 4, 3, 3)).astype(np_dtype)
        gamma_np = rng.standard_normal(4).astype(np_dtype)
        beta_np = rng.standard_normal(4).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        gamma = Tensor.from_dlpack(gamma_np)
        beta = Tensor.from_dlpack(beta_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.group_norm(x, gamma, beta, num_groups=2, epsilon=1e-5)

        expected = _numpy_group_norm(
            x_np, gamma_np, beta_np, num_groups=2, eps=1e-5
        )
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-4, atol=1e-4
        )

    def test_group_norm_3d(self) -> None:
        """Test group_norm on a 3D (N, C, L) tensor."""
        rng = np.random.default_rng(51)
        x_np = rng.standard_normal((2, 8, 5)).astype(np.float32)
        gamma_np = rng.standard_normal(8).astype(np.float32)
        beta_np = rng.standard_normal(8).astype(np.float32)

        x = Tensor.from_dlpack(x_np)
        gamma = Tensor.from_dlpack(gamma_np)
        beta = Tensor.from_dlpack(beta_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.group_norm(x, gamma, beta, num_groups=4, epsilon=1e-5)

        expected = _numpy_group_norm(
            x_np, gamma_np, beta_np, num_groups=4, eps=1e-5
        )
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-4, atol=1e-4
        )


class TestSliceOp:
    """Tests for slice operations."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_slice_1d(self, dtype: DType) -> None:
        """Test basic 1D slice."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(10, dtype=np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.slice_tensor(x, [slice(2, 7)])

        expected = x_np[2:7]
        np.testing.assert_array_equal(np.from_dlpack(result), expected)

    @pytest.mark.parametrize("dtype", REARRANGE_FLOAT_DTYPES)
    def test_slice_2d(self, dtype: DType) -> None:
        """Test 2D slice across both dimensions."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.slice_tensor(x, [slice(0, 2), slice(1, 3)])

        expected = x_np[0:2, 1:3]
        np.testing.assert_array_equal(np.from_dlpack(result), expected)

    def test_slice_with_step(self) -> None:
        """Test slice with step > 1."""
        x_np = np.arange(10, dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.slice_tensor(x, [slice(0, 10, 2)])

        expected = x_np[0:10:2]
        np.testing.assert_array_equal(np.from_dlpack(result), expected)

    def test_slice_3d(self) -> None:
        """Test slice with 3D tensor."""
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.slice_tensor(x, [slice(0, 2), slice(1, 3), slice(0, 2)])

        expected = x_np[0:2, 1:3, 0:2]
        np.testing.assert_array_equal(np.from_dlpack(result), expected)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_slice_int_dtypes(self, dtype: DType) -> None:
        """Test slice with integer dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.slice_tensor(x, [slice(1, 3), slice(0, 2)])

        expected = x_np[1:3, 0:2]
        np.testing.assert_array_equal(np.from_dlpack(result), expected)

    def test_slice_single_element(self) -> None:
        """Test slice extracting a single element along each dim."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.slice_tensor(x, [slice(1, 2), slice(2, 3)])

        expected = x_np[1:2, 2:3]
        np.testing.assert_array_equal(np.from_dlpack(result), expected)

    def test_slice_full_dim(self) -> None:
        """Test slice that takes the full range of a dimension."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.slice_tensor(x, [slice(0, 3), slice(0, 4)])

        expected = x_np[0:3, 0:4]
        np.testing.assert_array_equal(np.from_dlpack(result), expected)


class TestCumsumOps:
    """Tests for cumsum Mojo ops."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cumsum_1d(self, dtype: DType) -> None:
        """Test cumsum on a 1D tensor."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.cumsum(x, axis=0)

        expected = np.cumsum(x_np, axis=0)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cumsum_2d_last_axis(self, dtype: DType) -> None:
        """Test cumsum on the last axis of a 2D tensor."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.cumsum(x, axis=-1)

        expected = np.cumsum(x_np, axis=-1)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cumsum_2d_first_axis(self, dtype: DType) -> None:
        """Test cumsum on the first axis of a 2D tensor."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.cumsum(x, axis=0)

        expected = np.cumsum(x_np, axis=0)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cumsum_3d_middle_axis(self, dtype: DType) -> None:
        """Test cumsum on the middle axis of a 3D tensor."""
        shape = [2, 3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(24, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.cumsum(x, axis=1)

        expected = np.cumsum(x_np, axis=1)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cumsum_4d(self, dtype: DType) -> None:
        """Test cumsum on a 4D tensor along axis 2."""
        shape = [2, 3, 4, 5]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(120, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.cumsum(x, axis=2)

        expected = np.cumsum(x_np, axis=2)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cumsum_5d(self, dtype: DType) -> None:
        """Test cumsum on a 5D tensor along axis 3."""
        shape = [2, 3, 2, 4, 2]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(96, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.cumsum(x, axis=3)

        expected = np.cumsum(x_np, axis=3)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cumsum_exclusive(self, dtype: DType) -> None:
        """Test cumsum with exclusive=True."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([1, 2, 3], dtype=np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.cumsum(x, axis=0, exclusive=True)

        # exclusive cumsum: [0, 1, 3]
        expected = np.array([0, 1, 3], dtype=np_dtype)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cumsum_reverse(self, dtype: DType) -> None:
        """Test cumsum with reverse=True."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([1, 2, 3], dtype=np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.cumsum(x, axis=0, reverse=True)

        # reverse cumsum: [6, 5, 3]
        expected = np.array([6, 5, 3], dtype=np_dtype)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cumsum_exclusive_reverse(self, dtype: DType) -> None:
        """Test cumsum with both exclusive=True and reverse=True."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([1, 2, 3], dtype=np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.cumsum(x, axis=0, exclusive=True, reverse=True)

        # exclusive reverse cumsum: [5, 3, 0]
        expected = np.array([5, 3, 0], dtype=np_dtype)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", [DType.int32, DType.int64])
    def test_cumsum_integer(self, dtype: DType) -> None:
        """Test cumsum with integer dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([1, 2, 3, 4], dtype=np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.cumsum(x, axis=0)

        expected = np.cumsum(x_np, axis=0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)


def test_flat_index_strides() -> None:
    """Row-major strides used to flatten a multi-dim index into a scalar."""
    assert gc_compile.flat_index_strides((4, 3, 2)) == [6, 2, 1]
    assert gc_compile.flat_index_strides((5,)) == [1]
    assert gc_compile.flat_index_strides(()) == []


class TestGatherOp:
    """Tests for gather op via MO interpreter."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_gather_axis0(self, dtype: DType) -> None:
        """Test gather along axis 0 on a 2D tensor."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)
        idx_np = np.array([2, 0], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather(x, idx, axis=0)

        expected = np.take(x_np, idx_np, axis=0)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_gather_axis1(self, dtype: DType) -> None:
        """Test gather along axis 1 on a 2D tensor."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)
        idx_np = np.array([3, 1, 0], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather(x, idx, axis=1)

        expected = np.take(x_np, idx_np, axis=1)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_gather_negative_axis(self) -> None:
        """Test gather with negative axis."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        idx_np = np.array([0, 2], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather(x, idx, axis=-1)

        expected = np.take(x_np, idx_np, axis=-1)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_gather_3d(self) -> None:
        """Test gather on a 3D tensor along axis 1."""
        x_np = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        idx_np = np.array([1, 3], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather(x, idx, axis=1)

        expected = np.take(x_np, idx_np, axis=1)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_gather_3d_axis2(self) -> None:
        """Test gather on a 3D tensor along last axis."""
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        idx_np = np.array([0, 3], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather(x, idx, axis=2)

        expected = np.take(x_np, idx_np, axis=2)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_gather_multidim_indices(self) -> None:
        """Test gather with 2D indices tensor."""
        x_np = np.arange(20, dtype=np.float32).reshape(4, 5)
        idx_np = np.array([[0, 2], [1, 3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather(x, idx, axis=0)

        expected = np.take(x_np, idx_np, axis=0)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_gather_int32_indices(self) -> None:
        """Test gather with int32 index dtype."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        idx_np = np.array([2, 0], dtype=np.int32)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather(x, idx, axis=0)

        expected = np.take(x_np, idx_np, axis=0)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_gather_integer_data(self, dtype: DType) -> None:
        """Test gather with integer data dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)
        idx_np = np.array([1, 0, 2], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather(x, idx, axis=0)

        expected = np.take(x_np, idx_np, axis=0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)


class TestGatherNdOp:
    """Tests for gather_nd op via MO interpreter."""

    def test_gather_nd_basic(self) -> None:
        """Test basic 2D gather_nd."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        idx_np = np.array([[0, 1], [2, 3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather_nd(x, idx)

        # idx has shape (2, 2), index_depth=2 indexes fully into (3,4)
        # output shape: idx[:-1] + input[2:] = (2,) + () = (2,)
        expected = np.array([x_np[0, 1], x_np[2, 3]], dtype=np.float32)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_gather_nd_3d(self) -> None:
        """Test gather_nd on a 3D input with partial indexing."""
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        # index_depth=1: index into first dim only, slicing remaining
        idx_np = np.array([[0], [1]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather_nd(x, idx)

        # output shape: idx[:-1] + input[1:] = (2,) + (3,4) = (2,3,4)
        expected = x_np[idx_np.flatten()]
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_gather_nd_batch_dims(self) -> None:
        """Test gather_nd with batch_dims > 0."""
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        # batch_dims=1, index_depth=1: per-batch index into dim 1
        idx_np = np.array([[1], [0]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather_nd(x, idx, batch_dims=1)

        # output shape: input[:1] + idx[1:-1] + input[1+1:] = (2,) + () + (4,) = (2,4)
        expected = np.array([x_np[0, 1], x_np[1, 0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_gather_nd_single_index(self) -> None:
        """Test gather_nd where index_depth=1 (single dimension indexing)."""
        x_np = np.arange(20, dtype=np.float32).reshape(4, 5)
        idx_np = np.array([[0], [2], [3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather_nd(x, idx)

        # output shape: idx[:-1] + input[1:] = (3,) + (5,) = (3,5)
        expected = x_np[[0, 2, 3]]
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_gather_nd_full_index(self) -> None:
        """Test gather_nd where index_depth = input rank (full indexing)."""
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        idx_np = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather_nd(x, idx)

        # output shape: idx[:-1] + input[3:] = (2,) + () = (2,)
        expected = np.array([x_np[0, 1, 2], x_np[1, 2, 3]], dtype=np.float32)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_gather_nd_dtype(self, dtype: DType) -> None:
        """Test gather_nd with various float dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)
        idx_np = np.array([[1, 2], [0, 0]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather_nd(x, idx)

        expected = np.array([x_np[1, 2], x_np[0, 0]], dtype=np_dtype)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_gather_nd_int32_indices(self) -> None:
        """Test gather_nd with int32 index dtype."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        idx_np = np.array([[2, 1], [0, 3]], dtype=np.int32)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather_nd(x, idx)

        expected = np.array([x_np[2, 1], x_np[0, 3]], dtype=np.float32)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)


class TestGatherNdFusedModel:
    """Tests for gather_gc's fused flatten-index + real gather_nd graph.

    ``handlers.py`` isn't updated until Task 4R, so these call
    :func:`gather_gc.gather_nd_model`/:func:`gather_gc.gather_nd_operand_views`
    directly rather than through ``F.gather_nd`` (see ``TestGatherNdOp`` for
    the handler-level suite, which stays red until Task 4R lands).
    """

    def test_batch_dims_0(self) -> None:
        """batch_dims=0: single flat batch, multi-dim index fully flattened."""
        device = CPU()
        x_np = np.arange(12, dtype=np.uint32).reshape(3, 4)
        idx_np = np.array([[0, 1], [2, 3]], dtype=np.int64)  # m=2, k=2

        data_view, idx_view, strides = gather_gc.gather_nd_operand_views(
            x_np.shape, idx_np.shape, batch_dims=0
        )
        strides_np = np.array(strides, dtype=np.int64)

        model = gather_gc.gather_nd_model(device, DType.uint32, DType.int64)
        (out,) = model(
            Buffer.from_numpy(x_np.reshape(data_view)),
            Buffer.from_numpy(idx_np.reshape(idx_view)),
            Buffer.from_numpy(strides_np),
        )

        # flat indices: 0*4+1=1, 2*4+3=11 -> x.flat[[1, 11]]
        expected = np.array(
            [[[x_np.flat[1]], [x_np.flat[11]]]], dtype=np.uint32
        )
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_batch_dims_gt_0_multi_index_per_batch(self) -> None:
        """batch_dims=1, batch>1, m>1: per-batch-distinct indices via the
        real ``batch_dims=1`` handling -- no batch-folding hack needed."""
        device = CPU()
        x_np = np.arange(24, dtype=np.uint32).reshape(2, 3, 4)
        # Per batch, 2 index vectors of depth 1 (index into the size-3 dim).
        idx_np = np.array([[[1], [0]], [[2], [1]]], dtype=np.int64)

        data_view, idx_view, strides = gather_gc.gather_nd_operand_views(
            x_np.shape, idx_np.shape, batch_dims=1
        )
        strides_np = np.array(strides, dtype=np.int64)

        model = gather_gc.gather_nd_model(device, DType.uint32, DType.int64)
        (out,) = model(
            Buffer.from_numpy(x_np.reshape(data_view)),
            Buffer.from_numpy(idx_np.reshape(idx_view)),
            Buffer.from_numpy(strides_np),
        )

        expected = np.stack(
            [
                np.stack([x_np[0, 1], x_np[0, 0]]),
                np.stack([x_np[1, 2], x_np[1, 1]]),
            ]
        )
        np.testing.assert_array_equal(out.to_numpy(), expected)


class TestGatherSumOp:
    """Tests for the gather_sum interpreter handler.

    ``mo.GatherSumOp`` fuses a gather (axis 0) with a sum reduction
    (axis 1).  It is ``MO_HostOnly`` and used by DLRM-style multi-hot
    embeddings.  Tests call the handler directly since no user-facing
    graph API produces this op.
    """

    def test_gather_sum_basic(self) -> None:
        """Gather rows then sum over the multi-hot dimension."""
        input_np = np.arange(12, dtype=np.float32).reshape(4, 3)
        indices_np = np.array([[0, 2], [1, 3]], dtype=np.int32)

        mock_op = MagicMock()
        mock_result = MagicMock()
        mock_result.type = MagicMock()
        mock_result.type.device_ref = MagicMock()

        mock_result.type.device_ref = DeviceRef.CPU().to_mlir()
        mock_op.results = [mock_result]

        input_buf = Buffer.from_numpy(input_np)
        indices_buf = Buffer.from_numpy(indices_np)

        result = _handle_gather_sum(mock_op, [input_buf, indices_buf])

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, Buffer)

        gathered = np.take(input_np, indices_np, axis=0)
        expected = gathered.sum(axis=1, keepdims=True)
        np.testing.assert_array_almost_equal(out.to_numpy(), expected)

    def test_gather_sum_single_index(self) -> None:
        """Single index per row — sum is a no-op."""
        input_np = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        indices_np = np.array([[2], [0]], dtype=np.int32)

        mock_op = MagicMock()
        mock_result = MagicMock()

        mock_result.type = MagicMock()
        mock_result.type.device_ref = DeviceRef.CPU().to_mlir()
        mock_op.results = [mock_result]

        result = _handle_gather_sum(
            mock_op,
            [Buffer.from_numpy(input_np), Buffer.from_numpy(indices_np)],
        )

        gathered = np.take(input_np, indices_np, axis=0)
        expected = gathered.sum(axis=1, keepdims=True)
        assert isinstance(result[0], Buffer)
        np.testing.assert_array_almost_equal(result[0].to_numpy(), expected)

    def test_gather_sum_int_data(self) -> None:
        """Integer input dtype."""
        input_np = np.arange(8, dtype=np.int64).reshape(4, 2)
        indices_np = np.array([[0, 1], [2, 3]], dtype=np.int32)

        mock_op = MagicMock()
        mock_result = MagicMock()

        mock_result.type = MagicMock()
        mock_result.type.device_ref = DeviceRef.CPU().to_mlir()
        mock_op.results = [mock_result]

        result = _handle_gather_sum(
            mock_op,
            [Buffer.from_numpy(input_np), Buffer.from_numpy(indices_np)],
        )

        gathered = np.take(input_np, indices_np, axis=0)
        expected = gathered.sum(axis=1, keepdims=True)
        assert isinstance(result[0], Buffer)
        np.testing.assert_array_equal(result[0].to_numpy(), expected)


class TestArgMaxMinOp:
    """Tests for ArgMax and ArgMin interpreter ops.

    Parameterized on op (argmax/argmin) and axis to avoid duplication.
    """

    @pytest.mark.parametrize("op_name", ["argmax", "argmin"])
    @pytest.mark.parametrize("axis", [0, 1, -1])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_2d_axes(self, op_name: str, axis: int, dtype: DType) -> None:
        """Test argmax/argmin on a 2D tensor along each axis."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([[1, 5, 3], [4, 2, 6]], dtype=np_dtype)
        np_op = getattr(np, op_name)
        f_op = getattr(F, op_name)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = f_op(x, axis=axis)

        expected = np_op(x_np, axis=axis, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("op_name", ["argmax", "argmin"])
    def test_3d_middle_axis(self, op_name: str) -> None:
        """Test on a 3D tensor along the middle axis."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3, 4, 5)).astype(np.float32)
        np_op = getattr(np, op_name)
        f_op = getattr(F, op_name)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = f_op(x, axis=1)

        expected = np_op(x_np, axis=1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize(
        "op_name,tie_data",
        [
            ("argmax", [5.0, 5.0, 3.0, 5.0]),
            ("argmin", [1.0, 3.0, 1.0, 5.0]),
        ],
    )
    def test_ties_lowest_index(
        self, op_name: str, tie_data: list[float]
    ) -> None:
        """Test that ties return the lowest index."""
        x_np = np.array(tie_data, dtype=np.float32)
        f_op = getattr(F, op_name)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = f_op(x, axis=0)

        result = np.from_dlpack(y)
        assert result.item() == 0, (
            f"Expected index 0 for tie, got {result.item()}"
        )

    @pytest.mark.parametrize("op_name", ["argmax", "argmin"])
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_integer_dtypes(self, op_name: str, dtype: DType) -> None:
        """Test with integer input dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([[10, 3, 7], [1, 8, 4]], dtype=np_dtype)
        np_op = getattr(np, op_name)
        f_op = getattr(F, op_name)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = f_op(x, axis=1)

        expected = np_op(x_np, axis=1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("op_name", ["argmax", "argmin"])
    def test_1d(self, op_name: str) -> None:
        """Test on a 1D tensor."""
        x_np = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0], dtype=np.float32)
        np_op = getattr(np, op_name)
        f_op = getattr(F, op_name)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = f_op(x, axis=0)

        expected = np_op(x_np, axis=0, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("op_name", ["argmax", "argmin"])
    def test_4d(self, op_name: str) -> None:
        """Test on a 4D tensor along axis 2."""
        rng = np.random.default_rng(99)
        x_np = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
        np_op = getattr(np, op_name)
        f_op = getattr(F, op_name)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = f_op(x, axis=2)

        expected = np_op(x_np, axis=2, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)


class TestSplitOp:
    """Tests for split op via MO interpreter."""

    @staticmethod
    def _assert_split_equal(
        results: Sequence[object],
        expected: list[np.ndarray],
    ) -> None:
        assert len(results) == len(expected)
        for result, exp in zip(results, expected, strict=True):
            assert isinstance(result, Tensor)
            np.testing.assert_array_equal(np.from_dlpack(result), exp)

    @pytest.mark.parametrize("dtype", REARRANGE_FLOAT_DTYPES)
    @pytest.mark.parametrize("axis", [0, 1])
    def test_2d_axes(self, dtype: DType, axis: int) -> None:
        """Test split on a 2D tensor along each axis."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(24, dtype=np_dtype).reshape(6, 4)
        split_sizes = [2, 4] if axis == 0 else [1, 3]

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            results = F.split(x, split_sizes, axis=axis)

        indices = np.cumsum(split_sizes[:-1])
        expected = np.split(x_np, indices, axis=axis)
        self._assert_split_equal(results, expected)

    def test_3d_middle_axis(self) -> None:
        """Test split on a 3D tensor along the middle axis."""
        x_np = np.arange(60, dtype=np.float32).reshape(3, 4, 5)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            results = F.split(x, [1, 2, 1], axis=1)

        expected = np.split(x_np, [1, 3], axis=1)
        self._assert_split_equal(results, expected)

    def test_negative_axis(self) -> None:
        """Test split with a negative axis value."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            results = F.split(x, [1, 3], axis=-1)

        expected = np.split(x_np, [1], axis=-1)
        self._assert_split_equal(results, expected)

    def test_three_way_split(self) -> None:
        """Test splitting into three uneven parts."""
        x_np = np.arange(30, dtype=np.float32).reshape(5, 6)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            results = F.split(x, [1, 2, 3], axis=1)

        expected = np.split(x_np, [1, 3], axis=1)
        self._assert_split_equal(results, expected)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_integer_dtypes(self, dtype: DType) -> None:
        """Test split with integer input dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            results = F.split(x, [1, 2], axis=0)

        expected = np.split(x_np, [1], axis=0)
        self._assert_split_equal(results, expected)

    def test_4d(self) -> None:
        """Test split on a 4D tensor."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 6, 4, 3)).astype(np.float32)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            results = F.split(x, [2, 1, 3], axis=1)

        expected = np.split(x_np, [2, 3], axis=1)
        self._assert_split_equal(results, expected)

    def test_equal_split(self) -> None:
        """Test splitting into equal-size chunks via int split_size."""
        x_np = np.arange(12, dtype=np.float32).reshape(4, 3)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            results = F.split(x, 2, axis=0)

        expected = np.split(x_np, 2, axis=0)
        self._assert_split_equal(results, expected)


class TestScatterOp:
    """Tests for scatter op via MO interpreter (CPU-only, MO_HostOnly)."""

    @staticmethod
    def _scatter_ref(
        x: np.ndarray, updates: np.ndarray, indices: np.ndarray, axis: int
    ) -> np.ndarray:
        """Numpy reference: copy x, then put_along_axis."""
        out = x.copy()
        np.put_along_axis(out, indices, updates, axis=axis)
        return out

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_2d(self, axis: int, dtype: DType) -> None:
        """Test scatter on a 2D tensor along axis 0 and 1."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)
        if axis == 0:
            updates_np = np.array(
                [[90, 91, 92, 93], [94, 95, 96, 97]], dtype=np_dtype
            )
            indices_np = np.array([[2, 1, 0, 2], [0, 2, 1, 0]], dtype=np.int64)
        else:
            updates_np = np.array(
                [[90, 91], [92, 93], [94, 95]], dtype=np_dtype
            )
            indices_np = np.array([[3, 0], [2, 1], [0, 3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter(x, updates, indices, axis=axis)

        expected = self._scatter_ref(x_np, updates_np, indices_np, axis)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_negative_axis(self) -> None:
        """Test scatter with negative axis (-1 == last axis)."""
        x_np = np.zeros((3, 4), dtype=np.float32)
        updates_np = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            dtype=np.float32,
        )
        indices_np = np.array(
            [[0, 3, 1, 2], [2, 1, 3, 0], [3, 0, 2, 1]], dtype=np.int64
        )

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter(x, updates, indices, axis=-1)

        expected = self._scatter_ref(x_np, updates_np, indices_np, axis=-1)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_3d_middle_axis(self) -> None:
        """Test scatter on a 3D tensor along axis 1."""
        x_np = np.zeros((2, 4, 3), dtype=np.float32)
        updates_np = np.ones((2, 2, 3), dtype=np.float32) * 7.0
        indices_np = np.array(
            [[[0, 0, 0], [3, 3, 3]], [[1, 1, 1], [2, 2, 2]]], dtype=np.int64
        )

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter(x, updates, indices, axis=1)

        expected = self._scatter_ref(x_np, updates_np, indices_np, axis=1)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_4d(self) -> None:
        """Test scatter on a 4D tensor along axis 2."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 3, 5, 4)).astype(np.float32)
        indices_np = rng.integers(0, 5, size=(2, 3, 2, 4)).astype(np.int64)
        updates_np = np.ones((2, 3, 2, 4), dtype=np.float32) * 99.0

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter(x, updates, indices, axis=2)

        expected = self._scatter_ref(x_np, updates_np, indices_np, axis=2)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_integer_dtypes(self, dtype: DType) -> None:
        """Test scatter with integer data dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)
        updates_np = np.array(
            [[50, 60, 70, 80], [90, 100, 110, 120], [10, 20, 30, 40]],
            dtype=np_dtype,
        )
        indices_np = np.array(
            [[1, 0, 3, 2], [3, 2, 0, 1], [0, 1, 2, 3]], dtype=np.int64
        )

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter(x, updates, indices, axis=1)

        expected = self._scatter_ref(x_np, updates_np, indices_np, axis=1)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_duplicate_indices(self) -> None:
        """Test scatter with duplicate indices (last write wins)."""
        x_np = np.zeros((4,), dtype=np.float32)
        updates_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        indices_np = np.array([1, 1, 1, 2], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter(x, updates, indices, axis=0)

        expected = self._scatter_ref(x_np, updates_np, indices_np, axis=0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_partial_update_along_axis(self) -> None:
        """Test scatter where updates are smaller than input along axis."""
        x_np = np.arange(20, dtype=np.float32).reshape(4, 5)
        updates_np = np.array(
            [[-1.0, -2.0, -3.0, -4.0, -5.0], [-6.0, -7.0, -8.0, -9.0, -10.0]],
            dtype=np.float32,
        )
        indices_np = np.array(
            [[2, 0, 3, 1, 0], [0, 3, 1, 2, 3]], dtype=np.int64
        )

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter(x, updates, indices, axis=0)

        expected = self._scatter_ref(x_np, updates_np, indices_np, axis=0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_int32_indices(self) -> None:
        """Test scatter with int32 index dtype."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        updates_np = np.array(
            [
                [99.0, 88.0, 77.0, 66.0],
                [55.0, 44.0, 33.0, 22.0],
                [11.0, 0.0, -1.0, -2.0],
            ],
            dtype=np.float32,
        )
        indices_np = np.array(
            [[2, 0, 3, 1], [1, 3, 0, 2], [0, 1, 2, 3]], dtype=np.int32
        )

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter(x, updates, indices, axis=1)

        expected = self._scatter_ref(x_np, updates_np, indices_np, axis=1)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)


class TestScatterAddOp:
    """Tests for scatter_add op via MO interpreter (CPU-only, MO_HostOnly).

    Uses ``F.scatter_add`` which routes through ``ops.scatter_add`` ->
    ``rmo.MoScatterAddOp`` -> ``mo.scatter.add`` -> interpreter handler.
    The reference is ``numpy.add.at``, which accumulates duplicate indices.
    """

    @staticmethod
    def _scatter_add_ref(
        x_np: np.ndarray,
        updates_np: np.ndarray,
        indices_np: np.ndarray,
        axis: int,
    ) -> np.ndarray:
        """NumPy reference: copy input then accumulate updates at indices.

        Mirrors the kernel's exact semantics:
        ``out[upd_idx[:axis], indices_np[upd_idx], upd_idx[axis+1:]] += updates``
        Duplicate indices are summed.
        """
        out = x_np.copy()
        ndim = x_np.ndim
        if axis < 0:
            axis += ndim
        for upd_idx in np.ndindex(updates_np.shape):
            out_idx: list[Any] = list(upd_idx)
            out_idx[axis] = int(indices_np[upd_idx])
            out[tuple(out_idx)] += updates_np[upd_idx]
        return out

    @pytest.mark.parametrize("axis", [0, 1])
    def test_basic_2d(self, axis: int) -> None:
        """Test scatter_add on a 2D float32 tensor along axis 0 and 1."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        if axis == 0:
            updates_np = np.ones((2, 4), dtype=np.float32) * 10.0
            indices_np = np.array([[0, 1, 2, 0], [2, 0, 1, 2]], dtype=np.int64)
        else:
            updates_np = np.ones((3, 2), dtype=np.float32) * 5.0
            indices_np = np.array([[0, 3], [1, 2], [0, 1]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_add(x, updates, indices, axis=axis)

        expected = self._scatter_add_ref(x_np, updates_np, indices_np, axis)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_negative_axis(self) -> None:
        """Test scatter_add with a negative axis (-1 == last axis)."""
        x_np = np.zeros((3, 4), dtype=np.float32)
        updates_np = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        )
        indices_np = np.array([[0, 1, 0], [2, 3, 2], [1, 0, 1]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_add(x, updates, indices, axis=-1)

        expected = self._scatter_add_ref(x_np, updates_np, indices_np, axis=-1)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_3d(self) -> None:
        """Test scatter_add on a 3D tensor along axis 1."""
        x_np = np.zeros((2, 4, 3), dtype=np.float32)
        updates_np = np.ones((2, 2, 3), dtype=np.float32) * 2.0
        indices_np = np.array(
            [[[0, 1, 2], [2, 3, 0]], [[1, 2, 3], [3, 0, 1]]], dtype=np.int64
        )

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_add(x, updates, indices, axis=1)

        expected = self._scatter_add_ref(x_np, updates_np, indices_np, axis=1)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_duplicate_indices(self) -> None:
        """Duplicate indices must sum, not overwrite."""
        x_np = np.zeros((4,), dtype=np.float32)
        updates_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        # indices[0] and indices[1] both point to slot 1 → slot 1 = 10+20=30
        indices_np = np.array([1, 1, 2, 3], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_add(x, updates, indices, axis=0)

        expected = self._scatter_add_ref(x_np, updates_np, indices_np, axis=0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)
        # Explicit check: slot 1 must be 30.0, not 20.0.
        assert float(np.from_dlpack(y)[1]) == 30.0

    @pytest.mark.parametrize("dtype", [DType.float32, DType.int32])
    def test_dtypes(self, dtype: DType) -> None:
        """Test scatter_add with numeric dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)
        updates_np = np.ones((2, 4), dtype=np_dtype)
        indices_np = np.array([[0, 1, 2, 0], [2, 1, 0, 2]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_add(x, updates, indices, axis=0)

        expected = self._scatter_add_ref(x_np, updates_np, indices_np, axis=0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_int32_indices(self) -> None:
        """Test scatter_add with int32 index dtype."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        updates_np = np.array([[100.0, 200.0, 300.0, 400.0]], dtype=np.float32)
        indices_np = np.array([[2, 0, 1, 2]], dtype=np.int32)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_add(x, updates, indices, axis=0)

        expected = self._scatter_add_ref(x_np, updates_np, indices_np, axis=0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)


class TestScatterMaxOp:
    """Tests for scatter_max op via MO interpreter (CPU-only, MO_HostOnly).

    Uses ``F.scatter_max`` which routes through ``ops.scatter_max`` ->
    ``rmo.MoScatterMaxOp`` -> ``mo.scatter.max`` -> interpreter handler.
    The reference keeps ``max(existing, update)`` at duplicate indices.
    """

    @staticmethod
    def _scatter_max_ref(
        x_np: np.ndarray,
        updates_np: np.ndarray,
        indices_np: np.ndarray,
        axis: int,
    ) -> np.ndarray:
        out = x_np.copy()
        ndim = x_np.ndim
        if axis < 0:
            axis += ndim
        for upd_idx in np.ndindex(updates_np.shape):
            out_idx: list[Any] = list(upd_idx)
            out_idx[axis] = int(indices_np[upd_idx])
            t = tuple(out_idx)
            out[t] = max(out[t], updates_np[upd_idx])
        return out

    @pytest.mark.parametrize("axis", [0, 1])
    def test_basic_2d(self, axis: int) -> None:
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        if axis == 0:
            updates_np = np.ones((2, 4), dtype=np.float32) * 100.0
            indices_np = np.array([[0, 1, 2, 0], [2, 0, 1, 2]], dtype=np.int64)
        else:
            updates_np = np.ones((3, 2), dtype=np.float32) * 50.0
            indices_np = np.array([[0, 3], [1, 2], [0, 1]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_max(x, updates, indices, axis=axis)

        expected = self._scatter_max_ref(x_np, updates_np, indices_np, axis)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_duplicate_indices(self) -> None:
        """Duplicate indices keep the maximum."""
        x_np = np.zeros((4,), dtype=np.float32)
        updates_np = np.array([10.0, 20.0, 5.0, 40.0], dtype=np.float32)
        indices_np = np.array([1, 1, 2, 3], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_max(x, updates, indices, axis=0)

        expected = self._scatter_max_ref(x_np, updates_np, indices_np, axis=0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)
        assert float(np.from_dlpack(y)[1]) == 20.0

    def test_negative_axis(self) -> None:
        x_np = np.zeros((3, 4), dtype=np.float32)
        updates_np = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        )
        indices_np = np.array([[0, 1, 0], [2, 3, 2], [1, 0, 1]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_max(x, updates, indices, axis=-1)

        expected = self._scatter_max_ref(x_np, updates_np, indices_np, axis=-1)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", [DType.float32, DType.int32])
    def test_dtypes(self, dtype: DType) -> None:
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)
        updates_np = (np.ones((2, 4), dtype=np_dtype) * 100).astype(np_dtype)
        indices_np = np.array([[0, 1, 2, 0], [2, 1, 0, 2]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_max(x, updates, indices, axis=0)

        expected = self._scatter_max_ref(x_np, updates_np, indices_np, axis=0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)


class TestScatterMinOp:
    """Tests for scatter_min op via MO interpreter (CPU-only, MO_HostOnly).

    Uses ``F.scatter_min`` which routes through ``ops.scatter_min`` ->
    ``rmo.MoScatterMinOp`` -> ``mo.scatter.min`` -> interpreter handler.
    The reference keeps ``min(existing, update)`` at duplicate indices.
    """

    @staticmethod
    def _scatter_min_ref(
        x_np: np.ndarray,
        updates_np: np.ndarray,
        indices_np: np.ndarray,
        axis: int,
    ) -> np.ndarray:
        out = x_np.copy()
        ndim = x_np.ndim
        if axis < 0:
            axis += ndim
        for upd_idx in np.ndindex(updates_np.shape):
            out_idx: list[Any] = list(upd_idx)
            out_idx[axis] = int(indices_np[upd_idx])
            t = tuple(out_idx)
            out[t] = min(out[t], updates_np[upd_idx])
        return out

    @pytest.mark.parametrize("axis", [0, 1])
    def test_basic_2d(self, axis: int) -> None:
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4) + 100.0
        if axis == 0:
            updates_np = np.ones((2, 4), dtype=np.float32) * 5.0
            indices_np = np.array([[0, 1, 2, 0], [2, 0, 1, 2]], dtype=np.int64)
        else:
            updates_np = np.ones((3, 2), dtype=np.float32) * 5.0
            indices_np = np.array([[0, 3], [1, 2], [0, 1]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_min(x, updates, indices, axis=axis)

        expected = self._scatter_min_ref(x_np, updates_np, indices_np, axis)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_duplicate_indices(self) -> None:
        """Duplicate indices keep the minimum."""
        x_np = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32)
        updates_np = np.array([10.0, 20.0, 5.0, 40.0], dtype=np.float32)
        indices_np = np.array([1, 1, 2, 3], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_min(x, updates, indices, axis=0)

        expected = self._scatter_min_ref(x_np, updates_np, indices_np, axis=0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)
        assert float(np.from_dlpack(y)[1]) == 10.0

    def test_negative_axis(self) -> None:
        x_np = np.full((3, 4), 999.0, dtype=np.float32)
        updates_np = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        )
        indices_np = np.array([[0, 1, 0], [2, 3, 2], [1, 0, 1]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_min(x, updates, indices, axis=-1)

        expected = self._scatter_min_ref(x_np, updates_np, indices_np, axis=-1)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", [DType.float32, DType.int32])
    def test_dtypes(self, dtype: DType) -> None:
        np_dtype = dtype.to_numpy()
        x_np = (np.arange(12, dtype=np.float32).reshape(3, 4) + 100).astype(
            np_dtype
        )
        updates_np = np.ones((2, 4), dtype=np_dtype)
        indices_np = np.array([[0, 1, 2, 0], [2, 1, 0, 2]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_min(x, updates, indices, axis=0)

        expected = self._scatter_min_ref(x_np, updates_np, indices_np, axis=0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)


class TestScatterMulOp:
    """Tests for scatter_mul op via MO interpreter (CPU-only, MO_HostOnly).

    Uses ``F.scatter_mul`` which routes through ``ops.scatter_mul`` ->
    ``rmo.MoScatterMulOp`` -> ``mo.scatter.mul`` -> interpreter handler.
    The reference multiplies ``output[...][idx] *= update`` at each index.
    """

    @staticmethod
    def _scatter_mul_ref(
        x_np: np.ndarray,
        updates_np: np.ndarray,
        indices_np: np.ndarray,
        axis: int,
    ) -> np.ndarray:
        out = x_np.copy()
        ndim = x_np.ndim
        if axis < 0:
            axis += ndim
        for upd_idx in np.ndindex(updates_np.shape):
            out_idx: list[Any] = list(upd_idx)
            out_idx[axis] = int(indices_np[upd_idx])
            t = tuple(out_idx)
            out[t] *= updates_np[upd_idx]
        return out

    @pytest.mark.parametrize("axis", [0, 1])
    def test_basic_2d(self, axis: int) -> None:
        x_np = np.ones((3, 4), dtype=np.float32) * 2.0
        if axis == 0:
            updates_np = np.ones((2, 4), dtype=np.float32) * 3.0
            indices_np = np.array([[0, 1, 2, 0], [2, 0, 1, 2]], dtype=np.int64)
        else:
            updates_np = np.ones((3, 2), dtype=np.float32) * 5.0
            indices_np = np.array([[0, 3], [1, 2], [0, 1]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_mul(x, updates, indices, axis=axis)

        expected = self._scatter_mul_ref(x_np, updates_np, indices_np, axis)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_duplicate_indices(self) -> None:
        """Duplicate indices multiply: 2 * 10 * 20 = 400."""
        x_np = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        updates_np = np.array([10.0, 20.0, 5.0, 40.0], dtype=np.float32)
        indices_np = np.array([1, 1, 2, 3], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_mul(x, updates, indices, axis=0)

        expected = self._scatter_mul_ref(x_np, updates_np, indices_np, axis=0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)
        assert float(np.from_dlpack(y)[1]) == 400.0

    def test_negative_axis(self) -> None:
        x_np = np.ones((3, 4), dtype=np.float32) * 10.0
        updates_np = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        )
        indices_np = np.array([[0, 1, 0], [2, 3, 2], [1, 0, 1]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_mul(x, updates, indices, axis=-1)

        expected = self._scatter_mul_ref(x_np, updates_np, indices_np, axis=-1)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", [DType.float32, DType.int32])
    def test_dtypes(self, dtype: DType) -> None:
        np_dtype = dtype.to_numpy()
        x_np = np.ones((3, 4), dtype=np_dtype) * 2
        x_np = x_np.astype(np_dtype)
        updates_np = (np.ones((2, 4), dtype=np_dtype) * 3).astype(np_dtype)
        indices_np = np.array([[0, 1, 2, 0], [2, 1, 0, 2]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_mul(x, updates, indices, axis=0)

        expected = self._scatter_mul_ref(x_np, updates_np, indices_np, axis=0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)


class TestScatterNdFusedModel:
    """Tests for scatter_nd_gc's fused flatten-index + real scatter_nd graph.

    ``handlers.py`` isn't updated until Task 4R, so these call
    :func:`scatter_nd_gc.scatter_nd_model`/
    :func:`scatter_nd_gc.scatter_nd_operand_views` directly rather than
    through ``F.scatter_nd``/``F.scatter_nd_add`` (see ``TestScatterNdOp``/
    ``TestScatterNdAddOp`` for the handler-level suites, which stay red until
    Task 4R lands).
    """

    def test_overwrite(self) -> None:
        """ScatterNdOp: last-write-wins overwrite through the fused graph."""
        device = CPU()
        x_np = np.arange(12, dtype=np.uint32).reshape(3, 4)
        updates_np = np.array([100, 200], dtype=np.uint32)
        idx_np = np.array([[0, 1], [2, 3]], dtype=np.int64)  # m=2, k=2

        input_view, updates_view, indices_view, strides = (
            scatter_nd_gc.scatter_nd_operand_views(x_np.shape, idx_np.shape)
        )
        strides_np = np.array(strides, dtype=np.int64)

        model = scatter_nd_gc.scatter_nd_model(
            mo.ScatterNdOp, device, DType.uint32, DType.int64
        )
        (out,) = model(
            Buffer.from_numpy(x_np.reshape(input_view)),
            Buffer.from_numpy(updates_np.reshape(updates_view)),
            Buffer.from_numpy(idx_np.reshape(indices_view)),
            Buffer.from_numpy(strides_np),
        )

        expected = x_np.copy()
        expected[0, 1] = 100
        expected[2, 3] = 200
        np.testing.assert_array_equal(
            out.to_numpy().reshape(x_np.shape), expected
        )

    def test_reduce_add_with_duplicate_indices(self) -> None:
        """ScatterNdAddOp: duplicate index vectors accumulate through the
        fused graph, exercising the real reduce kernel (not just overwrite)."""
        device = CPU()
        x_np = np.zeros((3, 4), dtype=np.float32)
        # Two update rows target the same index vector -> must accumulate.
        updates_np = np.array([1.0, 2.0, 5.0], dtype=np.float32)
        idx_np = np.array([[0, 1], [0, 1], [2, 3]], dtype=np.int64)

        input_view, updates_view, indices_view, strides = (
            scatter_nd_gc.scatter_nd_operand_views(x_np.shape, idx_np.shape)
        )
        strides_np = np.array(strides, dtype=np.int64)

        model = scatter_nd_gc.scatter_nd_model(
            mo.ScatterNdAddOp, device, DType.float32, DType.int64
        )
        (out,) = model(
            Buffer.from_numpy(x_np.reshape(input_view)),
            Buffer.from_numpy(updates_np.reshape(updates_view)),
            Buffer.from_numpy(idx_np.reshape(indices_view)),
            Buffer.from_numpy(strides_np),
        )

        expected = x_np.copy()
        expected[0, 1] = 1.0 + 2.0
        expected[2, 3] = 5.0
        np.testing.assert_array_equal(
            out.to_numpy().reshape(x_np.shape), expected
        )

    def test_lazy_default_compiles_on_miss(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """By default (env var unset) a supported miss compiles lazily."""
        monkeypatch.delenv(
            gc_compile.EAGER_OP_PRECOMPILE_ENV_VAR, raising=False
        )
        monkeypatch.setattr(
            scatter_nd_gc._FAMILIES[mo.ScatterNdOp], "cache", {}
        )
        model = scatter_nd_gc.scatter_nd_model(
            mo.ScatterNdOp, CPU(), DType.uint32, DType.int64
        )
        assert model is not None


class TestScatterNdOp:
    """Tests for scatter_nd op via the MO interpreter (CPU + GPU capable).

    Uses ``F.scatter_nd`` which routes through ``ops.scatter_nd`` ->
    ``rmo.MoScatterNdOp`` -> ``mo.scatter_nd`` -> interpreter handler.
    The NumPy reference iterates over update positions via ``np.ndindex``
    to mirror the kernel's exact flat-index semantics.
    """

    @staticmethod
    def _scatter_nd_ref(
        x_np: np.ndarray,
        updates_np: np.ndarray,
        indices_np: np.ndarray,
    ) -> np.ndarray:
        """NumPy reference: copy input then overwrite at N-D index positions."""
        out = x_np.copy()
        index_depth = indices_np.shape[-1]
        batch_shape = indices_np.shape[:-1]
        for batch_idx in np.ndindex(batch_shape):
            idx_vec = tuple(
                int(indices_np[batch_idx + (k,)]) for k in range(index_depth)
            )
            out[idx_vec] = updates_np[batch_idx]
        return out

    def test_1d_full_index(self) -> None:
        """Test scatter_nd on a 1D tensor with full indexing (depth=1)."""
        x_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        updates_np = np.array([10.0, 20.0], dtype=np.float32)
        indices_np = np.array([[1], [3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd(x, updates, indices)

        expected = self._scatter_nd_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_2d_row_scatter(self) -> None:
        """Test scatter_nd on a 2D tensor with 1-D partial indexing (row write)."""
        x_np = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        )
        updates_np = np.array(
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]], dtype=np.float32
        )
        indices_np = np.array([[0], [2]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd(x, updates, indices)

        expected = self._scatter_nd_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_2d_element_scatter(self) -> None:
        """Test scatter_nd on a 2D tensor with full 2-D indexing (scalar write)."""
        x_np = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        )
        updates_np = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        indices_np = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd(x, updates, indices)

        expected = self._scatter_nd_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_3d_partial_index(self) -> None:
        """Test scatter_nd on a 3D tensor with 1-D partial indexing (plane write)."""
        x_np = np.zeros((3, 2, 4), dtype=np.float32)
        updates_np = np.ones((2, 2, 4), dtype=np.float32) * 7.0
        indices_np = np.array([[0], [2]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd(x, updates, indices)

        expected = self._scatter_nd_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_empty_updates(self) -> None:
        """Test scatter_nd with zero update vectors (output equals input)."""
        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        updates_np = np.empty((0,), dtype=np.float32)
        indices_np = np.empty((0, 1), dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd(x, updates, indices)

        np.testing.assert_array_equal(np.from_dlpack(y), x_np)

    def test_int32_indices(self) -> None:
        """Test scatter_nd with int32 index dtype."""
        x_np = np.arange(6, dtype=np.float32).reshape(2, 3)
        updates_np = np.array([99.0, 88.0, 77.0], dtype=np.float32)
        indices_np = np.array([[0, 0], [1, 1], [0, 2]], dtype=np.int32)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd(x, updates, indices)

        expected = self._scatter_nd_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)


class TestScatterNdAddOp:
    """Tests for scatter_nd_add op via the MO interpreter (CPU path).

    Uses ``F.scatter_nd_add`` which routes through ``ops.scatter_nd_add`` ->
    ``rmo.MoScatterNdAddOp`` -> ``mo.scatter_nd.add`` -> interpreter handler.
    Duplicate index vectors are accumulated (summed).
    """

    @staticmethod
    def _scatter_nd_add_ref(
        x_np: np.ndarray,
        updates_np: np.ndarray,
        indices_np: np.ndarray,
    ) -> np.ndarray:
        """NumPy reference: copy input then accumulate at N-D index positions."""
        out = x_np.copy()
        index_depth = indices_np.shape[-1]
        batch_shape = indices_np.shape[:-1]
        for batch_idx in np.ndindex(batch_shape):
            idx_vec = tuple(
                int(indices_np[batch_idx + (k,)]) for k in range(index_depth)
            )
            out[idx_vec] += updates_np[batch_idx]
        return out

    def test_1d_full_index(self) -> None:
        """Test scatter_nd_add on a 1D tensor with full indexing."""
        x_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        updates_np = np.array([10.0, 20.0], dtype=np.float32)
        indices_np = np.array([[1], [3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_add(x, updates, indices)

        expected = self._scatter_nd_add_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_2d_row_scatter(self) -> None:
        """Test scatter_nd_add on a 2D tensor with 1-D partial indexing."""
        x_np = np.zeros((3, 3), dtype=np.float32)
        updates_np = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )
        indices_np = np.array([[0], [2]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_add(x, updates, indices)

        expected = self._scatter_nd_add_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_2d_element_scatter(self) -> None:
        """Test scatter_nd_add on a 2D tensor with full 2-D indexing."""
        x_np = np.zeros((3, 3), dtype=np.float32)
        updates_np = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        indices_np = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_add(x, updates, indices)

        expected = self._scatter_nd_add_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_duplicate_indices(self) -> None:
        """Duplicate index vectors must be summed, not overwritten."""
        x_np = np.zeros((4,), dtype=np.float32)
        updates_np = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        # indices[0] and indices[1] both point to slot 1 → slot 1 = 10+20=30
        indices_np = np.array([[1], [1], [3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_add(x, updates, indices)

        expected = self._scatter_nd_add_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)
        # Explicit check: slot 1 must be 30.0, not 20.0.
        assert float(np.from_dlpack(y)[1]) == 30.0

    @pytest.mark.parametrize("dtype", [DType.float32, DType.int32])
    def test_dtypes(self, dtype: DType) -> None:
        """Test scatter_nd_add with various numeric dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.zeros((4,), dtype=np_dtype)
        updates_np = np.array([5, 10, 15], dtype=np_dtype)
        indices_np = np.array([[0], [2], [3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_add(x, updates, indices)

        expected = self._scatter_nd_add_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_int32_indices(self) -> None:
        """Test scatter_nd_add with int32 index dtype (full 2-D indexing)."""
        x_np = np.arange(6, dtype=np.float32).reshape(2, 3)
        # Full depth=2 indexing: each index selects a scalar element.
        updates_np = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        indices_np = np.array([[0, 0], [1, 2], [0, 0]], dtype=np.int32)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_add(x, updates, indices)

        expected = self._scatter_nd_add_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)


class TestScatterNdMaxOp:
    """Tests for scatter_nd_max op via the MO interpreter (CPU path).

    Uses ``F.scatter_nd_max`` which routes through ``ops.scatter_nd_max`` ->
    ``rmo.MoScatterNdMaxOp`` -> ``mo.scatter_nd.max`` -> interpreter handler.
    Duplicate index vectors keep the maximum.
    """

    @staticmethod
    def _scatter_nd_max_ref(
        x_np: np.ndarray,
        updates_np: np.ndarray,
        indices_np: np.ndarray,
    ) -> np.ndarray:
        """NumPy reference: copy input then keep max at N-D index positions."""
        out = x_np.copy()
        index_depth = indices_np.shape[-1]
        batch_shape = indices_np.shape[:-1]
        for batch_idx in np.ndindex(batch_shape):
            idx_vec = tuple(
                int(indices_np[batch_idx + (k,)]) for k in range(index_depth)
            )
            out[idx_vec] = np.maximum(out[idx_vec], updates_np[batch_idx])
        return out

    def test_1d_full_index(self) -> None:
        """Test scatter_nd_max on a 1D tensor with full indexing."""
        x_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        updates_np = np.array([10.0, 0.5], dtype=np.float32)
        indices_np = np.array([[1], [3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_max(x, updates, indices)

        expected = self._scatter_nd_max_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_2d_row_scatter(self) -> None:
        """Test scatter_nd_max on a 2D tensor with 1-D partial indexing."""
        x_np = np.zeros((3, 3), dtype=np.float32)
        updates_np = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )
        indices_np = np.array([[0], [2]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_max(x, updates, indices)

        expected = self._scatter_nd_max_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_2d_element_scatter(self) -> None:
        """Test scatter_nd_max on a 2D tensor with full 2-D indexing."""
        x_np = np.zeros((3, 3), dtype=np.float32)
        updates_np = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        indices_np = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_max(x, updates, indices)

        expected = self._scatter_nd_max_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_duplicate_indices(self) -> None:
        """Duplicate index vectors must keep the maximum, not overwrite."""
        x_np = np.zeros((4,), dtype=np.float32)
        updates_np = np.array([10.0, 20.0, 5.0], dtype=np.float32)
        indices_np = np.array([[1], [1], [3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_max(x, updates, indices)

        expected = self._scatter_nd_max_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)
        assert float(np.from_dlpack(y)[1]) == 20.0

    @pytest.mark.parametrize("dtype", [DType.float32, DType.int32])
    def test_dtypes(self, dtype: DType) -> None:
        """Test scatter_nd_max with various numeric dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.zeros((4,), dtype=np_dtype)
        updates_np = np.array([5, 10, 15], dtype=np_dtype)
        indices_np = np.array([[0], [2], [3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_max(x, updates, indices)

        expected = self._scatter_nd_max_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)


class TestScatterNdMinOp:
    """Tests for scatter_nd_min op via the MO interpreter (CPU path).

    Uses ``F.scatter_nd_min`` which routes through ``ops.scatter_nd_min`` ->
    ``rmo.MoScatterNdMinOp`` -> ``mo.scatter_nd.min`` -> interpreter handler.
    Duplicate index vectors keep the minimum.
    """

    @staticmethod
    def _scatter_nd_min_ref(
        x_np: np.ndarray,
        updates_np: np.ndarray,
        indices_np: np.ndarray,
    ) -> np.ndarray:
        """NumPy reference: copy input then keep min at N-D index positions."""
        out = x_np.copy()
        index_depth = indices_np.shape[-1]
        batch_shape = indices_np.shape[:-1]
        for batch_idx in np.ndindex(batch_shape):
            idx_vec = tuple(
                int(indices_np[batch_idx + (k,)]) for k in range(index_depth)
            )
            out[idx_vec] = np.minimum(out[idx_vec], updates_np[batch_idx])
        return out

    def test_1d_full_index(self) -> None:
        """Test scatter_nd_min on a 1D tensor with full indexing."""
        x_np = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
        updates_np = np.array([5.0, 100.0], dtype=np.float32)
        indices_np = np.array([[1], [3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_min(x, updates, indices)

        expected = self._scatter_nd_min_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_2d_row_scatter(self) -> None:
        """Test scatter_nd_min on a 2D tensor with 1-D partial indexing."""
        x_np = np.full((3, 3), 100.0, dtype=np.float32)
        updates_np = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )
        indices_np = np.array([[0], [2]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_min(x, updates, indices)

        expected = self._scatter_nd_min_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_2d_element_scatter(self) -> None:
        """Test scatter_nd_min on a 2D tensor with full 2-D indexing."""
        x_np = np.full((3, 3), 100.0, dtype=np.float32)
        updates_np = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        indices_np = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_min(x, updates, indices)

        expected = self._scatter_nd_min_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_duplicate_indices(self) -> None:
        """Duplicate index vectors must keep the minimum."""
        x_np = np.full((4,), 100.0, dtype=np.float32)
        updates_np = np.array([30.0, 10.0, 5.0], dtype=np.float32)
        indices_np = np.array([[1], [1], [3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_min(x, updates, indices)

        expected = self._scatter_nd_min_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)
        assert float(np.from_dlpack(y)[1]) == 10.0

    @pytest.mark.parametrize("dtype", [DType.float32, DType.int32])
    def test_dtypes(self, dtype: DType) -> None:
        """Test scatter_nd_min with various numeric dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.full((4,), 100, dtype=np_dtype)
        updates_np = np.array([5, 10, 15], dtype=np_dtype)
        indices_np = np.array([[0], [2], [3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_min(x, updates, indices)

        expected = self._scatter_nd_min_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)


class TestScatterNdMulOp:
    """Tests for scatter_nd_mul op via the MO interpreter (CPU path).

    Uses ``F.scatter_nd_mul`` which routes through ``ops.scatter_nd_mul`` ->
    ``rmo.MoScatterNdMulOp`` -> ``mo.scatter_nd.mul`` -> interpreter handler.
    Duplicate index vectors multiply.
    """

    @staticmethod
    def _scatter_nd_mul_ref(
        x_np: np.ndarray,
        updates_np: np.ndarray,
        indices_np: np.ndarray,
    ) -> np.ndarray:
        """NumPy reference: copy input then multiply at N-D index positions."""
        out = x_np.copy()
        index_depth = indices_np.shape[-1]
        batch_shape = indices_np.shape[:-1]
        for batch_idx in np.ndindex(batch_shape):
            idx_vec = tuple(
                int(indices_np[batch_idx + (k,)]) for k in range(index_depth)
            )
            out[idx_vec] *= updates_np[batch_idx]
        return out

    def test_1d_full_index(self) -> None:
        """Test scatter_nd_mul on a 1D tensor with full indexing."""
        x_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        updates_np = np.array([10.0, 2.0], dtype=np.float32)
        indices_np = np.array([[1], [3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_mul(x, updates, indices)

        expected = self._scatter_nd_mul_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_2d_row_scatter(self) -> None:
        """Test scatter_nd_mul on a 2D tensor with 1-D partial indexing."""
        x_np = np.ones((3, 3), dtype=np.float32) * 2
        updates_np = np.array(
            [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]], dtype=np.float32
        )
        indices_np = np.array([[0], [2]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_mul(x, updates, indices)

        expected = self._scatter_nd_mul_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_2d_element_scatter(self) -> None:
        """Test scatter_nd_mul on a 2D tensor with full 2-D indexing."""
        x_np = np.ones((3, 3), dtype=np.float32) * 5
        updates_np = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        indices_np = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_mul(x, updates, indices)

        expected = self._scatter_nd_mul_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_duplicate_indices(self) -> None:
        """Duplicate index vectors must chain multiplications."""
        x_np = np.ones((4,), dtype=np.float32)
        updates_np = np.array([3.0, 5.0, 2.0], dtype=np.float32)
        indices_np = np.array([[1], [1], [3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_mul(x, updates, indices)

        expected = self._scatter_nd_mul_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)
        assert float(np.from_dlpack(y)[1]) == 15.0

    @pytest.mark.parametrize("dtype", [DType.float32, DType.int32])
    def test_dtypes(self, dtype: DType) -> None:
        """Test scatter_nd_mul with various numeric dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.ones((4,), dtype=np_dtype) * 2
        updates_np = np.array([5, 10, 15], dtype=np_dtype)
        indices_np = np.array([[0], [2], [3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd_mul(x, updates, indices)

        expected = self._scatter_nd_mul_ref(x_np, updates_np, indices_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)


class TestConv2dOp:
    """Tests for the mo.conv (2D forward convolution) interpreter handler."""

    @staticmethod
    def _conv2d_ref(
        x: np.ndarray,
        filt: np.ndarray,
        stride: tuple[int, int],
        dilation: tuple[int, int],
        padding: tuple[int, int, int, int],
        groups: int,
    ) -> np.ndarray:
        """Pure-numpy 2D convolution reference (NHWC input, RSCF filter)."""
        n, in_h, in_w, _in_c = x.shape
        kh, kw, ic_pg, out_c = filt.shape
        sh, sw = stride
        dh, dw = dilation
        ph0, ph1, pw0, pw1 = padding

        oh = 1 + (in_h + ph0 + ph1 - (1 + dh * (kh - 1))) // sh
        ow = 1 + (in_w + pw0 + pw1 - (1 + dw * (kw - 1))) // sw
        oc_pg = out_c // groups

        out = np.zeros((n, oh, ow, out_c), dtype=x.dtype)
        for b in range(n):
            for g in range(groups):
                for ohi in range(oh):
                    for owi in range(ow):
                        for oci in range(oc_pg):
                            acc = np.float64(0)
                            for fi in range(kh):
                                ih = ohi * sh - ph0 + fi * dh
                                if ih < 0 or ih >= in_h:
                                    continue
                                for fj in range(kw):
                                    iw = owi * sw - pw0 + fj * dw
                                    if iw < 0 or iw >= in_w:
                                        continue
                                    for ic in range(ic_pg):
                                        acc += float(
                                            x[b, ih, iw, g * ic_pg + ic]
                                        ) * float(
                                            filt[fi, fj, ic, g * oc_pg + oci]
                                        )
                            out[b, ohi, owi, g * oc_pg + oci] = acc
        return out

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic_3x3(self, dtype: DType) -> None:
        """Test basic 3x3 conv, stride 1, no padding."""
        np_dt = dtype.to_numpy()
        x_np = np.arange(1 * 5 * 5 * 1, dtype=np_dt).reshape(1, 5, 5, 1)
        f_np = np.ones((3, 3, 1, 1), dtype=np_dt)

        x = Tensor.from_dlpack(x_np)
        f = Tensor.from_dlpack(f_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.conv2d(x, f)

        expected = self._conv2d_ref(x_np, f_np, (1, 1), (1, 1), (0, 0, 0, 0), 1)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_stride_2(self) -> None:
        """Test 2x2 conv with stride 2."""
        x_np = np.arange(1 * 6 * 6 * 1, dtype=np.float32).reshape(1, 6, 6, 1)
        f_np = np.ones((2, 2, 1, 1), dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        f = Tensor.from_dlpack(f_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.conv2d(x, f, stride=(2, 2))

        expected = self._conv2d_ref(x_np, f_np, (2, 2), (1, 1), (0, 0, 0, 0), 1)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_padding(self) -> None:
        """Test 3x3 conv with symmetric padding."""
        x_np = np.arange(1 * 4 * 4 * 1, dtype=np.float32).reshape(1, 4, 4, 1)
        f_np = np.ones((3, 3, 1, 1), dtype=np.float32)
        padding = (1, 1, 1, 1)

        x = Tensor.from_dlpack(x_np)
        f = Tensor.from_dlpack(f_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.conv2d(x, f, padding=padding)

        expected = self._conv2d_ref(x_np, f_np, (1, 1), (1, 1), padding, 1)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("dilation", [(2, 2), (2, 3), (1, 2)])
    def test_dilation(self, dilation: tuple[int, int]) -> None:
        """Non-unit dilation produces correct output (MXF-529 / KERN-3238)."""
        x_np = np.arange(1 * 9 * 9 * 1, dtype=np.float32).reshape(1, 9, 9, 1)
        f_np = np.ones((3, 3, 1, 1), dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        f = Tensor.from_dlpack(f_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.conv2d(x, f, dilation=dilation)

        expected = self._conv2d_ref(
            x_np, f_np, (1, 1), dilation, (0, 0, 0, 0), 1
        )
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_groups(self) -> None:
        """Grouped conv (groups > 1) is not supported after the GC migration
        (MXF-529): it needs a pre-packed filter layout the eager
        interpreter never has, and there's no way to pack it from Python."""
        x_np = np.arange(1 * 4 * 4 * 4, dtype=np.float32).reshape(1, 4, 4, 4)
        f_np = np.ones((3, 3, 2, 4), dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        f = Tensor.from_dlpack(f_np)
        with pytest.raises(NotImplementedError, match="groups"):
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                y = F.conv2d(x, f, groups=2)
                assert y is not None

    def test_1x1_conv(self) -> None:
        """Test pointwise (1x1) convolution."""
        x_np = np.arange(1 * 3 * 3 * 2, dtype=np.float32).reshape(1, 3, 3, 2)
        f_np = np.array([[[[1, -1], [0, 1]]]], dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        f = Tensor.from_dlpack(f_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.conv2d(x, f)

        expected = self._conv2d_ref(x_np, f_np, (1, 1), (1, 1), (0, 0, 0, 0), 1)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_non_square_kernel(self) -> None:
        """Test non-square (2, 3) kernel."""
        x_np = np.arange(1 * 5 * 6 * 1, dtype=np.float32).reshape(1, 5, 6, 1)
        f_np = np.ones((2, 3, 1, 1), dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        f = Tensor.from_dlpack(f_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.conv2d(x, f)

        expected = self._conv2d_ref(x_np, f_np, (1, 1), (1, 1), (0, 0, 0, 0), 1)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_float16_unsupported(self) -> None:
        """float16 doesn't compile through the GC conv path on CPU.

        conv_gc._supported_dtypes excludes float16 on CPU (fails to compile
        there; compiles fine on GPU), matching pooling_gc's and resize_gc's
        own CPU+float16 exclusion.
        """
        x_np = np.arange(1 * 5 * 5 * 1, dtype=np.float16).reshape(1, 5, 5, 1)
        f_np = np.ones((3, 3, 1, 1), dtype=np.float16)

        x = Tensor.from_dlpack(x_np)
        f = Tensor.from_dlpack(f_np)
        with pytest.raises(KeyError, match="float16"):
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                y = F.conv2d(x, f)
                assert y is not None


class TestConvTranspose2dOp:
    """Tests for the mo.conv_transpose (2D transposed conv) handler."""

    def test_unsupported(self) -> None:
        """conv_transpose has no supported kernel (KERN-3233): the old Mojo
        binding's GPU path crashed on Apple/CUDA and it was never migrated
        to the graph compiler, so the handler now raises rather than
        falling back to a broken kernel."""
        x_np = np.arange(1 * 3 * 3 * 1, dtype=np.float32).reshape(1, 3, 3, 1)
        f_np = np.ones((3, 3, 1, 1), dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        f = Tensor.from_dlpack(f_np)
        with pytest.raises(NotImplementedError, match="conv_transpose"):
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                y = F.conv2d_transpose(x, f)
                assert y is not None


class TestMaxPoolOp:
    """Tests for max_pool2d op via MO interpreter (CPU+GPU)."""

    @staticmethod
    def _max_pool_ref(
        x_nhwc: np.ndarray,
        kernel: tuple[int, int],
        stride: tuple[int, int],
        dilation: tuple[int, int],
        padding: tuple[int, int, int, int],
        ceil_mode: bool = False,
    ) -> np.ndarray:
        """Pure-numpy NHWC max_pool2d reference implementation."""
        n, h, w, c = x_nhwc.shape
        kh, kw = kernel
        sh, sw = stride
        dh, dw = dilation
        ph_b, ph_a, pw_b, pw_a = padding

        def _out_dim(in_dim: int, k: int, s: int, d: int, pad: int) -> int:
            num = in_dim + pad - (d * (k - 1) + 1)
            if ceil_mode:
                return 1 + -(-num // s)
            return 1 + num // s

        oh = _out_dim(h, kh, sh, dh, ph_b + ph_a)
        ow = _out_dim(w, kw, sw, dw, pw_b + pw_a)

        out = np.full((n, oh, ow, c), -np.inf, dtype=x_nhwc.dtype)
        for bi in range(n):
            for ohi in range(oh):
                for owi in range(ow):
                    for ki in range(kh):
                        ih = ohi * sh - ph_b + ki * dh
                        if ih < 0 or ih >= h:
                            continue
                        for kj in range(kw):
                            iw = owi * sw - pw_b + kj * dw
                            if iw < 0 or iw >= w:
                                continue
                            out[bi, ohi, owi, :] = np.maximum(
                                out[bi, ohi, owi, :],
                                x_nhwc[bi, ih, iw, :],
                            )
        return out

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic_2x2(self, dtype: DType) -> None:
        """Test 2x2 max pool, stride 1, no padding."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(16, dtype=np_dtype).reshape(1, 4, 4, 1)
        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.max_pool2d(x, kernel_size=(2, 2))

        expected = self._max_pool_ref(
            x_np, (2, 2), (1, 1), (1, 1), (0, 0, 0, 0)
        )
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_stride_2(self) -> None:
        """Test 2x2 max pool with stride 2."""
        x_np = np.arange(36, dtype=np.float32).reshape(1, 6, 6, 1)
        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        expected = self._max_pool_ref(
            x_np, (2, 2), (2, 2), (1, 1), (0, 0, 0, 0)
        )
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_padding(self) -> None:
        """Test 3x3 max pool with padding=1."""
        x_np = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.max_pool2d(x, kernel_size=(3, 3), padding=1)

        expected = self._max_pool_ref(
            x_np, (3, 3), (1, 1), (1, 1), (1, 1, 1, 1)
        )
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_dilation(self) -> None:
        """Test 3x3 max pool with dilation=2."""
        x_np = np.arange(49, dtype=np.float32).reshape(1, 7, 7, 1)
        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.max_pool2d(x, kernel_size=(3, 3), dilation=2)

        expected = self._max_pool_ref(
            x_np, (3, 3), (1, 1), (2, 2), (0, 0, 0, 0)
        )
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_ceil_mode(self) -> None:
        """Test ceil_mode produces larger output than floor mode."""
        x_np = np.arange(25, dtype=np.float32).reshape(1, 5, 5, 1)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y_floor = F.max_pool2d(
                x, kernel_size=(2, 2), stride=3, ceil_mode=False
            )
            y_ceil = F.max_pool2d(
                x, kernel_size=(2, 2), stride=3, ceil_mode=True
            )

        floor_shape = np.from_dlpack(y_floor).shape
        ceil_shape = np.from_dlpack(y_ceil).shape
        assert ceil_shape[1] >= floor_shape[1]
        assert ceil_shape[2] >= floor_shape[2]

        expected_floor = self._max_pool_ref(
            x_np, (2, 2), (3, 3), (1, 1), (0, 0, 0, 0), ceil_mode=False
        )
        expected_ceil = self._max_pool_ref(
            x_np, (2, 2), (3, 3), (1, 1), (0, 0, 0, 0), ceil_mode=True
        )
        np.testing.assert_array_equal(np.from_dlpack(y_floor), expected_floor)
        np.testing.assert_array_equal(np.from_dlpack(y_ceil), expected_ceil)

    def test_non_square_kernel(self) -> None:
        """Test max pool with non-square kernel (2, 3)."""
        x_np = np.arange(24, dtype=np.float32).reshape(1, 4, 6, 1)
        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.max_pool2d(x, kernel_size=(2, 3), stride=(1, 2))

        expected = self._max_pool_ref(
            x_np, (2, 3), (1, 2), (1, 1), (0, 0, 0, 0)
        )
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_multi_channel_batch(self) -> None:
        """Test max pool with multiple channels and batch size > 1."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 8, 8, 3)).astype(np.float32)
        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1)

        expected = self._max_pool_ref(
            x_np, (3, 3), (2, 2), (1, 1), (1, 1, 1, 1)
        )
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", [DType.int32, DType.int64])
    def test_integer_dtypes(self, dtype: DType) -> None:
        """Test max pool with integer data dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(16, dtype=np_dtype).reshape(1, 4, 4, 1)
        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        expected = self._max_pool_ref(
            x_np, (2, 2), (2, 2), (1, 1), (0, 0, 0, 0)
        )
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_float16_unsupported(self) -> None:
        """float16 doesn't compile through the GC pooling path on CPU.

        pooling_gc._supported_dtypes excludes float16 on CPU (fails to
        compile there; compiles fine on GPU), matching resize_gc's
        CPU+float16 exclusion.
        """
        x_np = np.arange(16, dtype=np.float16).reshape(1, 4, 4, 1)
        x = Tensor.from_dlpack(x_np)

        with pytest.raises(KeyError, match="float16"):
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                y = F.max_pool2d(x, kernel_size=(2, 2))
                assert y is not None


class TestTileOp:
    """Tests for the mo.tile interpreter handler."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_1d(self, dtype: DType) -> None:
        """Test tiling a 1-D tensor."""
        x_np = np.array([1, 2, 3], dtype=dtype.to_numpy())
        reps = (3,)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.tile(x, reps)

        np.testing.assert_array_equal(np.from_dlpack(y), np.tile(x_np, reps))

    @pytest.mark.parametrize("dtype", REARRANGE_FLOAT_DTYPES)
    @pytest.mark.parametrize(
        "reps", [(2, 3), (1, 4), (3, 1)], ids=["2x3", "1x4", "3x1"]
    )
    def test_2d(self, dtype: DType, reps: tuple[int, ...]) -> None:
        """Test tiling a 2-D tensor with various repeat patterns."""
        x_np = np.arange(6, dtype=dtype.to_numpy()).reshape(2, 3)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.tile(x, reps)

        np.testing.assert_array_equal(np.from_dlpack(y), np.tile(x_np, reps))

    def test_3d(self) -> None:
        """Test tiling a 3-D tensor along all axes."""
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        reps = (2, 3, 2)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.tile(x, reps)

        np.testing.assert_array_equal(np.from_dlpack(y), np.tile(x_np, reps))

    def test_repeat_one(self) -> None:
        """Test tile with all repeats = 1 (identity)."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        reps = (1, 1)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.tile(x, reps)

        np.testing.assert_array_equal(np.from_dlpack(y), x_np)

    def test_large_repeat(self) -> None:
        """Test tile with a large repeat factor on one axis."""
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        reps = (1, 10)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.tile(x, reps)

        np.testing.assert_array_equal(np.from_dlpack(y), np.tile(x_np, reps))

    @pytest.mark.parametrize(
        "dtype",
        [DType.int32, DType.int64],
        ids=["int32", "int64"],
    )
    def test_integer_dtypes(self, dtype: DType) -> None:
        """Test tile with integer dtypes."""
        x_np = np.arange(6, dtype=dtype.to_numpy()).reshape(2, 3)
        reps = (2, 2)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.tile(x, reps)

        np.testing.assert_array_equal(np.from_dlpack(y), np.tile(x_np, reps))


class TestBandPartOp:
    """Tests for the mo.linalg.band_part interpreter handler."""

    @staticmethod
    def _band_part_ref(
        x: np.ndarray,
        num_lower: int,
        num_upper: int,
        exclude: bool = False,
    ) -> np.ndarray:
        """Pure-numpy band_part reference."""
        shape = x.shape
        M, N = shape[-2], shape[-1]
        m = np.arange(M)[:, None]
        n = np.arange(N)[None, :]
        lower_ok = (num_lower < 0) | ((m - n) <= num_lower)
        upper_ok = (num_upper < 0) | ((n - m) <= num_upper)
        in_band = lower_ok & upper_ok
        if exclude:
            in_band = ~in_band
        mask = np.broadcast_to(in_band, shape)
        return np.where(mask, x, np.zeros_like(x))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_lower_triangle(self, dtype: DType) -> None:
        """Test lower triangle: num_lower=None (-1), num_upper=0."""
        np_dt = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dt).reshape(3, 4) + 1

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.band_part(x, num_lower=None, num_upper=0)

        expected = self._band_part_ref(x_np, -1, 0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_upper_triangle(self) -> None:
        """Test upper triangle: num_lower=0, num_upper=None (-1)."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4) + 1

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.band_part(x, num_lower=0, num_upper=None)

        expected = self._band_part_ref(x_np, 0, -1)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_diagonal(self) -> None:
        """Test diagonal only: num_lower=0, num_upper=0."""
        x_np = np.arange(9, dtype=np.float32).reshape(3, 3) + 1

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.band_part(x, num_lower=0, num_upper=0)

        expected = self._band_part_ref(x_np, 0, 0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_band(self) -> None:
        """Test tridiagonal band: num_lower=1, num_upper=1."""
        x_np = np.arange(20, dtype=np.float32).reshape(4, 5) + 1

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.band_part(x, num_lower=1, num_upper=1)

        expected = self._band_part_ref(x_np, 1, 1)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_exclude(self) -> None:
        """Test inverted mask: exclude=True zeroes the band."""
        x_np = np.arange(9, dtype=np.float32).reshape(3, 3) + 1

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.band_part(x, num_lower=None, num_upper=0, exclude=True)

        expected = self._band_part_ref(x_np, -1, 0, exclude=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_batched(self) -> None:
        """Test batched input [B, M, N]."""
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 1

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.band_part(x, num_lower=1, num_upper=0)

        expected = self._band_part_ref(x_np, 1, 0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_full_matrix(self, dtype: DType) -> None:
        """Test keeping entire matrix: num_lower=None, num_upper=None."""
        np_dt = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dt).reshape(3, 4) + 1

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.band_part(x, num_lower=None, num_upper=None)

        np.testing.assert_array_equal(np.from_dlpack(y), x_np)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float16,
            DType.float32,
            DType.float64,
            DType.int32,
            DType.int64,
            DType.uint8,
            DType.bool,
        ],
    )
    def test_band_part_dtype_width_coverage(self, dtype: DType) -> None:
        """Every byte-aligned dtype rides the same width-keyed GC model."""
        np_dt = dtype.to_numpy()
        x_np = np.ones((2, 3, 4), dtype=np_dt)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            x = Tensor.from_dlpack(x_np).to(CPU())
            out = F.band_part(x, num_lower=0, num_upper=0)

        np.testing.assert_array_equal(
            np.from_dlpack(out),
            self._band_part_ref(x_np, num_lower=0, num_upper=0),
        )


class TestAvgPool2dOp:
    """Tests for the mo.avg_pool / mo.avg_pool_ceil_mode_true interpreter
    handler."""

    @staticmethod
    def _avg_pool2d_ref(
        x: np.ndarray,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        dilation: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        ceil_mode: bool = False,
        count_boundary: bool = True,
    ) -> np.ndarray:
        """Pure-numpy avg_pool2d reference (NHWC layout)."""
        N, H, W, C = x.shape
        kH, kW = kernel_size
        sH, sW = stride
        dH, dW = dilation
        pH, pW = padding

        eff_kH = dH * (kH - 1) + 1
        eff_kW = dW * (kW - 1) + 1
        if ceil_mode:
            oH = int(np.ceil((H + 2 * pH - eff_kH + 1) / sH))
            oW = int(np.ceil((W + 2 * pW - eff_kW + 1) / sW))
        else:
            oH = (H + 2 * pH - eff_kH) // sH + 1
            oW = (W + 2 * pW - eff_kW) // sW + 1

        out = np.zeros((N, oH, oW, C), dtype=x.dtype)
        for n in range(N):
            for oh in range(oH):
                for ow in range(oW):
                    for c in range(C):
                        s = 0.0
                        cnt = 0
                        for fh in range(kH):
                            ih = oh * sH - pH + fh * dH
                            if ih < 0 or ih >= H:
                                if count_boundary:
                                    cnt += kW
                                continue
                            for fw in range(kW):
                                iw = ow * sW - pW + fw * dW
                                if iw < 0 or iw >= W:
                                    if count_boundary:
                                        cnt += 1
                                    continue
                                s += float(x[n, ih, iw, c])
                                cnt += 1
                        out[n, oh, ow, c] = s / cnt if cnt > 0 else 0.0
        return out

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic_2x2(self, dtype: DType) -> None:
        """Test 2x2 kernel, stride 1, no padding."""
        np_dt = dtype.to_numpy()
        x_np = np.arange(16, dtype=np_dt).reshape(1, 4, 4, 1)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.avg_pool2d(x, kernel_size=(2, 2))

        expected = self._avg_pool2d_ref(x_np, (2, 2))
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_stride_and_padding(self) -> None:
        """Test stride 2 with padding 1."""
        x_np = np.arange(25, dtype=np.float32).reshape(1, 5, 5, 1)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.avg_pool2d(
                x, kernel_size=(3, 3), stride=2, padding=1, count_boundary=True
            )

        expected = self._avg_pool2d_ref(
            x_np, (3, 3), stride=(2, 2), padding=(1, 1), count_boundary=True
        )
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_dilation(self) -> None:
        """Test dilated average pooling."""
        x_np = np.arange(36, dtype=np.float32).reshape(1, 6, 6, 1)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.avg_pool2d(x, kernel_size=(2, 2), dilation=2)

        expected = self._avg_pool2d_ref(x_np, (2, 2), dilation=(2, 2))
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_ceil_mode(self) -> None:
        """Test ceil mode output shape."""
        x_np = np.arange(25, dtype=np.float32).reshape(1, 5, 5, 1)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.avg_pool2d(x, kernel_size=(3, 3), stride=2, ceil_mode=True)

        expected = self._avg_pool2d_ref(
            x_np, (3, 3), stride=(2, 2), ceil_mode=True
        )
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_count_boundary_false(self) -> None:
        """Test excluding padding from divisor."""
        x_np = np.ones((1, 3, 3, 1), dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.avg_pool2d(
                x,
                kernel_size=(3, 3),
                padding=1,
                count_boundary=False,
            )

        expected = self._avg_pool2d_ref(
            x_np, (3, 3), padding=(1, 1), count_boundary=False
        )
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_float16_unsupported(self) -> None:
        """float16 doesn't compile through the GC pooling path on CPU.

        pooling_gc._supported_dtypes excludes float16 on CPU (fails to
        compile there; compiles fine on GPU), matching resize_gc's
        CPU+float16 exclusion.
        """
        x_np = np.arange(16, dtype=np.float16).reshape(1, 4, 4, 1)
        x = Tensor.from_dlpack(x_np)

        with pytest.raises(KeyError, match="float16"):
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                y = F.avg_pool2d(x, kernel_size=(2, 2))
                assert y is not None


class TestRoiAlignOp:
    """Tests for roi_align interpreter op via F.roi_align.

    Routes through F.roi_align -> ops.roi_align -> rmo.MoRoiAlignOp ->
    mo.RoiAlignOp -> _handle_roi_align -> roi_align_gc.roi_align_model.
    CPU-only (MO_HostOnly).
    The reference is a pure-numpy bilinear-interpolation ROI pooler.
    """

    @staticmethod
    def _roi_align_ref(
        x_nhwc: np.ndarray,
        rois: np.ndarray,
        out_h: int,
        out_w: int,
        spatial_scale: float = 1.0,
        sampling_ratio: float = 0.0,
        aligned: bool = False,
        mode: str = "AVG",
    ) -> np.ndarray:
        """Pure-numpy ROI Align reference (NHWC layout).

        ROIs have shape [M, 5]: [batch_idx, x0, y0, x1, y1].
        """
        n_regions = rois.shape[0]
        height, width, channels = (
            x_nhwc.shape[1],
            x_nhwc.shape[2],
            x_nhwc.shape[3],
        )
        offset = 0.5 if aligned else 0.0
        output = np.zeros(
            (n_regions, out_h, out_w, channels), dtype=x_nhwc.dtype
        )

        for ri in range(n_regions):
            batch_idx = int(rois[ri, 0])
            roi_start_w = rois[ri, 1] * spatial_scale - offset
            roi_start_h = rois[ri, 2] * spatial_scale - offset
            roi_end_w = rois[ri, 3] * spatial_scale - offset
            roi_end_h = rois[ri, 4] * spatial_scale - offset

            if aligned:
                roi_h = roi_end_h - roi_start_h
                roi_w = roi_end_w - roi_start_w
            else:
                roi_h = max(roi_end_h - roi_start_h, 1.0)
                roi_w = max(roi_end_w - roi_start_w, 1.0)

            bin_h = roi_h / out_h
            bin_w = roi_w / out_w

            grid_h = int(
                sampling_ratio if sampling_ratio > 0 else np.ceil(bin_h)
            )
            grid_w = int(
                sampling_ratio if sampling_ratio > 0 else np.ceil(bin_w)
            )
            pool_count = max(grid_h * grid_w, 1)

            for ph in range(out_h):
                for pw in range(out_w):
                    for c in range(channels):
                        if mode == "AVG":
                            pool_val = 0.0
                        else:
                            pool_val = -np.inf
                        for iy in range(grid_h):
                            for ix in range(grid_w):
                                y = (
                                    roi_start_h
                                    + ph * bin_h
                                    + (iy + 0.5) * bin_h / grid_h
                                )
                                x = (
                                    roi_start_w
                                    + pw * bin_w
                                    + (ix + 0.5) * bin_w / grid_w
                                )
                                if (
                                    y < -1.0
                                    or y > height
                                    or x < -1.0
                                    or x > width
                                ):
                                    continue
                                y = max(y, 0.0)
                                x = max(x, 0.0)
                                y_low = min(int(y), height - 1)
                                x_low = min(int(x), width - 1)
                                y_high = min(y_low + 1, height - 1)
                                x_high = min(x_low + 1, width - 1)
                                ly = y - y_low
                                lx = x - x_low
                                hy = 1.0 - ly
                                hx = 1.0 - lx
                                v = (
                                    hy * hx * x_nhwc[batch_idx, y_low, x_low, c]
                                    + hy
                                    * lx
                                    * x_nhwc[batch_idx, y_low, x_high, c]
                                    + ly
                                    * hx
                                    * x_nhwc[batch_idx, y_high, x_low, c]
                                    + ly
                                    * lx
                                    * x_nhwc[batch_idx, y_high, x_high, c]
                                )
                                if mode == "AVG":
                                    pool_val += v
                                else:
                                    pool_val = max(
                                        pool_val,
                                        hy
                                        * hx
                                        * x_nhwc[batch_idx, y_low, x_low, c],
                                    )
                                    pool_val = max(
                                        pool_val,
                                        hy
                                        * lx
                                        * x_nhwc[batch_idx, y_low, x_high, c],
                                    )
                                    pool_val = max(
                                        pool_val,
                                        ly
                                        * hx
                                        * x_nhwc[batch_idx, y_high, x_low, c],
                                    )
                                    pool_val = max(
                                        pool_val,
                                        ly
                                        * lx
                                        * x_nhwc[batch_idx, y_high, x_high, c],
                                    )
                        if mode == "AVG":
                            output[ri, ph, pw, c] = pool_val / pool_count
                        else:
                            output[ri, ph, pw, c] = pool_val
        return output

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_avg_mode(self, dtype: DType) -> None:
        """Test ROI Align with AVG pooling mode."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(100, dtype=np_dtype).reshape(1, 10, 10, 1)
        rois_np = np.array([[0, 1, 1, 5, 5]], dtype=np_dtype)

        x = Tensor.from_dlpack(x_np)
        rois = Tensor.from_dlpack(rois_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.roi_align(x, rois, output_height=3, output_width=3)

        expected = self._roi_align_ref(x_np, rois_np, 3, 3)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_max_mode(self) -> None:
        """Test ROI Align with MAX pooling mode."""
        x_np = np.arange(100, dtype=np.float32).reshape(1, 10, 10, 1)
        rois_np = np.array([[0, 2, 2, 8, 8]], dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        rois = Tensor.from_dlpack(rois_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.roi_align(
                x, rois, output_height=3, output_width=3, mode="MAX"
            )

        expected = self._roi_align_ref(x_np, rois_np, 3, 3, mode="MAX")
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_aligned(self) -> None:
        """Test ROI Align with aligned=True (half-pixel offset)."""
        x_np = np.arange(100, dtype=np.float32).reshape(1, 10, 10, 1)
        rois_np = np.array([[0, 1, 1, 5, 5]], dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        rois = Tensor.from_dlpack(rois_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.roi_align(
                x, rois, output_height=3, output_width=3, aligned=True
            )

        expected = self._roi_align_ref(x_np, rois_np, 3, 3, aligned=True)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_spatial_scale(self) -> None:
        """Test ROI Align with non-unit spatial_scale."""
        x_np = np.arange(100, dtype=np.float32).reshape(1, 10, 10, 1)
        rois_np = np.array([[0, 2, 2, 10, 10]], dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        rois = Tensor.from_dlpack(rois_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.roi_align(
                x, rois, output_height=3, output_width=3, spatial_scale=0.5
            )

        expected = self._roi_align_ref(x_np, rois_np, 3, 3, spatial_scale=0.5)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_sampling_ratio(self) -> None:
        """Test ROI Align with explicit sampling_ratio."""
        x_np = np.arange(100, dtype=np.float32).reshape(1, 10, 10, 1)
        rois_np = np.array([[0, 1, 1, 8, 8]], dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        rois = Tensor.from_dlpack(rois_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.roi_align(
                x, rois, output_height=3, output_width=3, sampling_ratio=3.0
            )

        expected = self._roi_align_ref(x_np, rois_np, 3, 3, sampling_ratio=3.0)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_multiple_rois(self) -> None:
        """Test ROI Align with multiple ROIs."""
        x_np = np.arange(200, dtype=np.float32).reshape(2, 10, 10, 1)
        rois_np = np.array(
            [
                [0, 0, 0, 5, 5],
                [1, 2, 2, 8, 8],
                [0, 3, 3, 9, 9],
            ],
            dtype=np.float32,
        )

        x = Tensor.from_dlpack(x_np)
        rois = Tensor.from_dlpack(rois_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.roi_align(
                x, rois, output_height=2, output_width=2, sampling_ratio=2.0
            )

        expected = self._roi_align_ref(x_np, rois_np, 2, 2, sampling_ratio=2.0)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_multichannel(self) -> None:
        """Test ROI Align with multiple channels."""
        x_np = np.arange(300, dtype=np.float32).reshape(1, 10, 10, 3)
        rois_np = np.array([[0, 1, 1, 6, 6]], dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        rois = Tensor.from_dlpack(rois_np)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.roi_align(x, rois, output_height=3, output_width=3)

        expected = self._roi_align_ref(x_np, rois_np, 3, 3)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )


class TestTopKOp:
    """Tests for TopK interpreter op (mo.top_k).

    Uses F.top_k which routes through ops.top_k -> rmo.top_k -> mo.top_k ->
    interpreter handler.  The reference implementation uses numpy stable
    argsort to match the kernel's selection-sort ordering.
    """

    @staticmethod
    def _top_k_ref(
        x_np: np.ndarray, k: int, axis: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """NumPy reference: top-k values and original indices, descending."""
        sorted_idx = np.argsort(
            -x_np.astype(np.float64), axis=axis, stable=True
        )
        idx = np.take(sorted_idx, np.arange(k), axis=axis)
        vals = np.take_along_axis(x_np, idx, axis=axis)
        return vals, idx

    @pytest.mark.parametrize("axis", [-1, 0])
    def test_basic_2d(self, axis: int) -> None:
        """Test top-2 on a 2D tensor along axis 0 and -1."""
        x_np = np.array([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]], dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            vals, idxs = F.top_k(x, k=2, axis=axis)

        ref_vals, ref_idxs = self._top_k_ref(x_np, k=2, axis=axis)
        np.testing.assert_array_equal(np.from_dlpack(vals), ref_vals)
        np.testing.assert_array_equal(np.from_dlpack(idxs), ref_idxs)

    def test_3d_middle_axis(self) -> None:
        """Test top-3 on a 3D tensor along axis 1."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 6, 4)).astype(np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            vals, idxs = F.top_k(x, k=3, axis=1)

        ref_vals, ref_idxs = self._top_k_ref(x_np, k=3, axis=1)
        np.testing.assert_allclose(
            np.from_dlpack(vals), ref_vals, rtol=1e-6, atol=0
        )
        np.testing.assert_array_equal(np.from_dlpack(idxs), ref_idxs)

    def test_k_equals_1(self) -> None:
        """k=1 must return the same element as argmax."""
        x_np = np.array([3.0, 7.0, 1.0, 5.0], dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            vals, idxs = F.top_k(x, k=1, axis=0)
            argmax_result = F.argmax(x, axis=0)

        np.testing.assert_array_equal(
            np.from_dlpack(idxs), np.from_dlpack(argmax_result)
        )
        np.testing.assert_allclose(np.from_dlpack(vals), np.array([7.0]))

    def test_k_equals_dim(self) -> None:
        """k equal to the axis size returns a full sorted permutation."""
        x_np = np.array([[4.0, 2.0, 1.0, 3.0]], dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            vals, idxs = F.top_k(x, k=4, axis=1)

        np.testing.assert_array_equal(
            np.from_dlpack(vals), np.array([[4.0, 3.0, 2.0, 1.0]])
        )
        np.testing.assert_array_equal(
            np.from_dlpack(idxs), np.array([[0, 3, 1, 2]])
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
    def test_dtypes(self, dtype: DType) -> None:
        """Test top-2 with numeric dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([[5, 1, 8, 3], [9, 2, 7, 4]], dtype=np_dtype)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            vals, idxs = F.top_k(x, k=2, axis=1)

        ref_vals, ref_idxs = self._top_k_ref(x_np, k=2, axis=1)
        np.testing.assert_array_equal(np.from_dlpack(vals), ref_vals)
        np.testing.assert_array_equal(np.from_dlpack(idxs), ref_idxs)


class TestBottomKOp:
    """Tests for BottomK interpreter op (mo.bottom_k).

    Uses F.bottom_k which routes through ops.bottom_k -> rmo.MoBottomKOp ->
    mo.bottom_k -> interpreter handler.  The reference implementation uses
    numpy stable argsort (ascending) to match the kernel's selection-sort
    ordering.
    """

    @staticmethod
    def _bottom_k_ref(
        x_np: np.ndarray, k: int, axis: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """NumPy reference: bottom-k values and original indices, ascending."""
        sorted_idx = np.argsort(x_np.astype(np.float64), axis=axis, stable=True)
        idx = np.take(sorted_idx, np.arange(k), axis=axis)
        vals = np.take_along_axis(x_np, idx, axis=axis)
        return vals, idx

    @pytest.mark.parametrize("axis", [-1, 0])
    def test_basic_2d(self, axis: int) -> None:
        """Test bottom-2 on a 2D tensor along axis 0 and -1."""
        x_np = np.array([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]], dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            vals, idxs = F.bottom_k(x, k=2, axis=axis)

        ref_vals, ref_idxs = self._bottom_k_ref(x_np, k=2, axis=axis)
        np.testing.assert_array_equal(np.from_dlpack(vals), ref_vals)
        np.testing.assert_array_equal(np.from_dlpack(idxs), ref_idxs)

    def test_3d_middle_axis(self) -> None:
        """Test bottom-3 on a 3D tensor along axis 1."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 6, 4)).astype(np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            vals, idxs = F.bottom_k(x, k=3, axis=1)

        ref_vals, ref_idxs = self._bottom_k_ref(x_np, k=3, axis=1)
        np.testing.assert_allclose(
            np.from_dlpack(vals), ref_vals, rtol=1e-6, atol=0
        )
        np.testing.assert_array_equal(np.from_dlpack(idxs), ref_idxs)

    def test_k_equals_1(self) -> None:
        """k=1 must return the same element as argmin."""
        x_np = np.array([3.0, 7.0, 1.0, 5.0], dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            vals, idxs = F.bottom_k(x, k=1, axis=0)
            argmin_result = F.argmin(x, axis=0)

        np.testing.assert_array_equal(
            np.from_dlpack(idxs), np.from_dlpack(argmin_result)
        )
        np.testing.assert_allclose(np.from_dlpack(vals), np.array([1.0]))

    def test_k_equals_dim(self) -> None:
        """k equal to the axis size returns a full sorted permutation."""
        x_np = np.array([[4.0, 2.0, 1.0, 3.0]], dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            vals, idxs = F.bottom_k(x, k=4, axis=1)

        np.testing.assert_array_equal(
            np.from_dlpack(vals), np.array([[1.0, 2.0, 3.0, 4.0]])
        )
        np.testing.assert_array_equal(
            np.from_dlpack(idxs), np.array([[2, 1, 3, 0]])
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
    def test_dtypes(self, dtype: DType) -> None:
        """Test bottom-2 with numeric dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([[5, 1, 8, 3], [9, 2, 7, 4]], dtype=np_dtype)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            vals, idxs = F.bottom_k(x, k=2, axis=1)

        ref_vals, ref_idxs = self._bottom_k_ref(x_np, k=2, axis=1)
        np.testing.assert_array_equal(np.from_dlpack(vals), ref_vals)
        np.testing.assert_array_equal(np.from_dlpack(idxs), ref_idxs)


class TestArgNonzeroOp:
    """Tests for ArgNonzero interpreter op (mo.arg_nonzero).

    Uses F.nonzero which routes through ops.nonzero -> rmo.MoArgNonzeroOp ->
    mo.ArgNonzeroOp -> interpreter handler. The reference is
    ``np.argwhere(x != 0).astype(np.int64)``, which returns row-major
    coordinates in the same order as the kernel.

    ``mo.arg_nonzero`` is MO_HostOnly so no GPU path exists; no GPU tests
    are needed.
    """

    @staticmethod
    def _nonzero_ref(x_np: np.ndarray) -> np.ndarray:
        """NumPy reference: row-major coordinates of nonzero elements."""
        return np.argwhere(x_np != 0).astype(np.int64)

    def test_1d_basic(self) -> None:
        """Test nonzero on a 1-D tensor with a mix of zeros and nonzeros."""
        x_np = np.array([0, 1, 0, 3, 0, 5], dtype=np.int32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.nonzero(x, out_dim="nnz")

        ref = self._nonzero_ref(x_np)
        np.testing.assert_array_equal(np.from_dlpack(result), ref)

    def test_2d_basic(self) -> None:
        """Test nonzero on a 2-D tensor."""
        x_np = np.array([[0, 1, 2], [0, 0, 3]], dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.nonzero(x, out_dim="nnz")

        ref = self._nonzero_ref(x_np)
        np.testing.assert_array_equal(np.from_dlpack(result), ref)

    def test_3d_basic(self) -> None:
        """Test nonzero on a 3-D tensor."""
        x_np = np.zeros((2, 3, 4), dtype=np.float32)
        x_np[0, 1, 2] = 1.0
        x_np[1, 0, 3] = -2.0
        x_np[1, 2, 0] = 5.0
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.nonzero(x, out_dim="nnz")

        ref = self._nonzero_ref(x_np)
        np.testing.assert_array_equal(np.from_dlpack(result), ref)

    def test_6d_basic(self) -> None:
        """Test nonzero on a rank-6 tensor."""
        x_np = np.zeros((2, 2, 2, 2, 2, 2), dtype=np.float32)
        x_np[0, 1, 0, 1, 0, 1] = 1.0
        x_np[1, 0, 1, 0, 1, 0] = -2.0
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.nonzero(x, out_dim="nnz")

        ref = self._nonzero_ref(x_np)
        np.testing.assert_array_equal(np.from_dlpack(result), ref)

    def test_all_zeros(self) -> None:
        """All-zero input must return an empty [0, rank] tensor."""
        x_np = np.zeros((3, 4), dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.nonzero(x, out_dim="nnz")

        ref = self._nonzero_ref(x_np)
        assert ref.shape == (0, 2)
        np.testing.assert_array_equal(np.from_dlpack(result), ref)

    def test_all_nonzero(self) -> None:
        """All-nonzero input must return coordinates for every element."""
        x_np = np.ones((2, 3), dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.nonzero(x, out_dim="nnz")

        ref = self._nonzero_ref(x_np)
        np.testing.assert_array_equal(np.from_dlpack(result), ref)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float64,
            DType.int8,
            DType.int16,
            DType.int32,
            DType.int64,
            DType.uint8,
            DType.uint32,
            DType.bool,
        ],
    )
    def test_dtype_parametrize(self, dtype: DType) -> None:
        """Nonzero must work for all common numeric dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([0, 1, 0, 2, 3, 0], dtype=np_dtype)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.nonzero(x, out_dim="nnz")

        ref = self._nonzero_ref(x_np)
        np.testing.assert_array_equal(np.from_dlpack(result), ref)

    def test_nonzero_gc_model_returns_exact_nnz(self) -> None:
        """A compiled arg_nonzero model sizes its own data-dependent output.

        Uses a flattened input directly; the [nnz, rank] coordinate decode is
        covered by the handler-level tests above via F.nonzero.
        """
        x = Buffer.from_numpy(
            np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0], dtype=np.float32)
        )
        model = nonzero_gc.nonzero_model(DType.float32)
        (out,) = model(x)
        np.testing.assert_array_equal(out.to_numpy(), np.array([[1], [3], [5]]))

    def test_float16_unsupported(self) -> None:
        """float16 isn't in this family's CPU dtype set."""
        with pytest.raises(KeyError, match="float16"):
            nonzero_gc.nonzero_model(DType.float16)


class TestPadConstantOp:
    """Tests for mo.PadConstantOp via F.pad(mode='constant')."""

    @staticmethod
    def _ref(
        x: np.ndarray, paddings: list[int], constant_value: float = 0.0
    ) -> np.ndarray:
        """NumPy reference for constant padding.

        Converts the flat [pre0, post0, pre1, post1, ...] paddings format
        used by mo.PadConstantOp into NumPy's per-axis tuple format.
        """
        rank = x.ndim
        pad_widths = [
            (paddings[2 * d], paddings[2 * d + 1]) for d in range(rank)
        ]
        return np.pad(
            x, pad_widths, mode="constant", constant_values=constant_value
        )

    def test_1d_zero_pad(self) -> None:
        """1-D tensor padded with zeros on both sides."""
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.pad(x, [1, 2])

        np.testing.assert_array_equal(
            np.from_dlpack(out), self._ref(x_np, [1, 2])
        )

    def test_2d_constant_value(self) -> None:
        """2-D tensor padded with a non-zero constant."""
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.pad(x, [1, 1, 2, 0], value=7.0)

        np.testing.assert_array_equal(
            np.from_dlpack(out),
            self._ref(x_np, [1, 1, 2, 0], constant_value=7.0),
        )

    def test_3d_asymmetric(self) -> None:
        """3-D tensor with asymmetric paddings."""
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.pad(x, [0, 1, 2, 0, 1, 3])

        np.testing.assert_array_equal(
            np.from_dlpack(out), self._ref(x_np, [0, 1, 2, 0, 1, 3])
        )

    def test_zero_padding(self) -> None:
        """All-zero paddings should produce an identical tensor."""
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.pad(x, [0, 0, 0, 0])

        np.testing.assert_array_equal(np.from_dlpack(out), x_np)

    @pytest.mark.parametrize(
        "dtype", [DType.float32, DType.float16, DType.int32]
    )
    def test_dtypes(self, dtype: DType) -> None:
        """Constant padding with multiple numeric dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np_dtype)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.pad(x, [1, 1, 0, 2])

        np.testing.assert_array_equal(
            np.from_dlpack(out), self._ref(x_np, [1, 1, 0, 2])
        )


class TestPadReflectOp:
    """Tests for mo.PadReflectOp via F.pad(mode='reflect')."""

    @staticmethod
    def _ref(x: np.ndarray, paddings: list[int]) -> np.ndarray:
        """NumPy reference for reflect padding."""
        rank = x.ndim
        pad_widths = [
            (paddings[2 * d], paddings[2 * d + 1]) for d in range(rank)
        ]
        return np.pad(x, pad_widths, mode="reflect")

    def test_2d_basic(self) -> None:
        """Basic 2-D reflect pad."""
        x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.pad(x, [1, 1, 0, 1], mode="reflect")

        np.testing.assert_array_equal(
            np.from_dlpack(out), self._ref(x_np, [1, 1, 0, 1])
        )

    def test_2d_symmetric(self) -> None:
        """Symmetric reflect pad on all sides."""
        x_np = np.arange(9, dtype=np.float32).reshape(3, 3)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.pad(x, [2, 2, 1, 1], mode="reflect")

        np.testing.assert_array_equal(
            np.from_dlpack(out), self._ref(x_np, [2, 2, 1, 1])
        )

    def test_3d_reflect(self) -> None:
        """3-D reflect pad along all axes."""
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.pad(x, [1, 0, 1, 1, 0, 1], mode="reflect")

        np.testing.assert_array_equal(
            np.from_dlpack(out), self._ref(x_np, [1, 0, 1, 1, 0, 1])
        )

    def test_zero_padding(self) -> None:
        """All-zero paddings should produce an identical tensor."""
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.pad(x, [0, 0, 0, 0], mode="reflect")

        np.testing.assert_array_equal(np.from_dlpack(out), x_np)


class TestPadRepeatOp:
    """Tests for mo.PadRepeatOp via F.pad(mode='edge')."""

    @staticmethod
    def _ref(x: np.ndarray, paddings: list[int]) -> np.ndarray:
        """NumPy reference for edge (repeat) padding."""
        rank = x.ndim
        pad_widths = [
            (paddings[2 * d], paddings[2 * d + 1]) for d in range(rank)
        ]
        return np.pad(x, pad_widths, mode="edge")

    def test_2d_basic(self) -> None:
        """Basic 2-D edge pad."""
        x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.pad(x, [2, 1, 1, 0], mode="edge")

        np.testing.assert_array_equal(
            np.from_dlpack(out), self._ref(x_np, [2, 1, 1, 0])
        )

    def test_2d_only_pre(self) -> None:
        """Edge pad only before each dimension (no post padding)."""
        x_np = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.pad(x, [3, 0, 2, 0], mode="edge")

        np.testing.assert_array_equal(
            np.from_dlpack(out), self._ref(x_np, [3, 0, 2, 0])
        )

    def test_3d_edge(self) -> None:
        """3-D edge pad along all axes."""
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.pad(x, [1, 2, 0, 1, 2, 0], mode="edge")

        np.testing.assert_array_equal(
            np.from_dlpack(out), self._ref(x_np, [1, 2, 0, 1, 2, 0])
        )

    def test_zero_padding(self) -> None:
        """All-zero paddings should produce an identical tensor."""
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.pad(x, [0, 0, 0, 0], mode="edge")

        np.testing.assert_array_equal(np.from_dlpack(out), x_np)

    @pytest.mark.parametrize(
        "dtype", [DType.float32, DType.float16, DType.int32]
    )
    def test_dtypes(self, dtype: DType) -> None:
        """Edge padding with multiple numeric dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np_dtype)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.pad(x, [1, 1, 2, 0], mode="edge")

        np.testing.assert_array_equal(
            np.from_dlpack(out), self._ref(x_np, [1, 1, 2, 0])
        )


class TestShapeIndexOps:
    """Tests for internal shape/index interpreter handlers.

    ``mo.ShapeFromTensorOp`` and ``mo.IndexToTensorOp`` are internal MO
    dialect ops that are usually folded away by canonicalization.  The
    handlers are defensive — they prevent ``NotImplementedError`` if these
    ops survive into the interpreter.  Tests call the handler functions
    directly with constructed buffers since no user-facing graph API
    produces these ops.
    """

    def test_index_to_tensor(self) -> None:
        """IndexToTensorOp wraps a scalar int64 into a rank-0 tensor."""
        input_np = np.array([42], dtype=np.int64)
        input_buf = Buffer.from_numpy(input_np)

        mock_op = MagicMock()
        result = _handle_index_to_tensor(mock_op, [input_buf])

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, Buffer)
        out_np = out.to_numpy()
        assert out_np.shape == ()
        assert int(out_np.item()) == 42

    def test_index_to_tensor_negative(self) -> None:
        """IndexToTensorOp handles negative integers."""
        input_np = np.array([-7], dtype=np.int64)
        input_buf = Buffer.from_numpy(input_np)

        mock_op = MagicMock()
        result = _handle_index_to_tensor(mock_op, [input_buf])

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, Buffer)
        assert int(out.to_numpy().item()) == -7

    def test_index_to_tensor_zero(self) -> None:
        """IndexToTensorOp handles zero."""
        input_np = np.array([0], dtype=np.int64)
        input_buf = Buffer.from_numpy(input_np)

        mock_op = MagicMock()
        result = _handle_index_to_tensor(mock_op, [input_buf])

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, Buffer)
        assert int(out.to_numpy().item()) == 0
        assert out.to_numpy().shape == ()

    def test_shape_from_tensor_passthrough(self) -> None:
        """ShapeFromTensorOp passes through the input buffer."""
        shape_np = np.array([2, 3, 4], dtype=np.int64)
        shape_buf = Buffer.from_numpy(shape_np)

        mock_op = MagicMock()
        result = _handle_shape_from_tensor(mock_op, [shape_buf])

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, Buffer)
        np.testing.assert_array_equal(out.to_numpy(), shape_np)

    def test_shape_from_tensor_single_dim(self) -> None:
        """ShapeFromTensorOp handles single-dimension shapes."""
        shape_np = np.array([10], dtype=np.int64)
        shape_buf = Buffer.from_numpy(shape_np)

        mock_op = MagicMock()
        result = _handle_shape_from_tensor(mock_op, [shape_buf])

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, Buffer)
        np.testing.assert_array_equal(out.to_numpy(), [10])


class TestBufferOps:
    """Tests for buffer create/transfer interpreter handlers.

    ``mo.BufferCreateOp`` and ``mo.BufferTransferOp`` are internal MO dialect
    ops that are normally lowered by the graph compiler.  The handlers are
    defensive — they prevent ``NotImplementedError`` if these ops survive into
    the interpreter.  Tests call the handler functions directly with
    constructed buffers since no user-facing graph API produces these ops.
    """

    def test_buffer_create_shape_and_dtype(self) -> None:
        """BufferCreateOp allocates a buffer with the requested shape/dtype."""
        mock_op = MagicMock()
        mock_result = MagicMock()

        buf_type = BufferType(DType.float32, [2, 3], DeviceRef.CPU())
        mock_result.type = buf_type.to_mlir()
        mock_op.results = [mock_result]

        result = _handle_buffer_create(mock_op, [])

        assert len(result) == 1
        buf = result[0]
        assert isinstance(buf, Buffer)
        assert buf.shape == (2, 3)
        assert buf.dtype == DType.float32

    def test_buffer_create_scalar(self) -> None:
        """BufferCreateOp handles rank-0 (scalar) buffers."""
        mock_op = MagicMock()
        mock_result = MagicMock()

        buf_type = BufferType(DType.int32, [], DeviceRef.CPU())
        mock_result.type = buf_type.to_mlir()
        mock_op.results = [mock_result]

        result = _handle_buffer_create(mock_op, [])

        assert len(result) == 1
        buf = result[0]
        assert isinstance(buf, Buffer)
        assert buf.shape == ()
        assert buf.dtype == DType.int32

    def test_buffer_transfer_copies_data(self) -> None:
        """BufferTransferOp copies src contents into dst."""
        src_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        src = Buffer.from_numpy(src_np)
        dst = Buffer(dtype=DType.float32, shape=[2, 2])

        mock_op = MagicMock()
        result = _handle_buffer_transfer(mock_op, [src, dst, None])

        assert result == [None]
        np.testing.assert_array_equal(dst.to_numpy(), src_np)

    def test_buffer_transfer_independent(self) -> None:
        """After transfer, modifying src does not affect dst."""
        src_np = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        src = Buffer.from_numpy(src_np)
        dst = Buffer(dtype=DType.float32, shape=[3])

        mock_op = MagicMock()
        _handle_buffer_transfer(mock_op, [src, dst, None])

        dst_snapshot = dst.to_numpy().copy()

        new_src = Buffer.from_numpy(
            np.array([99.0, 99.0, 99.0], dtype=np.float32)
        )
        src.inplace_copy_from(new_src)

        np.testing.assert_array_equal(dst.to_numpy(), dst_snapshot)


class TestMutableStoreOps:
    """End-to-end CPU tests for the mutable-tensor write interpreter handlers.

    ``F.buffer_store`` emits ``mo.MutableStoreOp``; ``F.buffer_store_slice``
    emits ``mo.MutableStoreSliceOp``. Both are dispatched by the interpreter
    to the respective handlers. Buffers are CPU-resident to exercise the
    handler's host fast path.
    """

    def test_buffer_store(self) -> None:
        """F.buffer_store writes a full tensor into the buffer."""
        buf = Buffer.zeros([4], DType.float32, CPU())
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            a = Tensor(storage=buf)
            b = Tensor.from_dlpack(np.ones(4, dtype=np.float32))
            F.buffer_store(a, b)

        np.testing.assert_array_equal(
            buf.to_numpy(), np.ones(4, dtype=np.float32)
        )

    def test_buffer_store_slice_unit_steps(self) -> None:
        """F.buffer_store_slice writes a contiguous 2D sub-region."""
        buf = Buffer.zeros([4, 4], DType.float32, CPU())
        slice_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            a = Tensor(storage=buf)
            b = Tensor.from_dlpack(slice_np)
            F.buffer_store_slice(a, b, [slice(1, 3), slice(1, 3)])

        expected = np.zeros((4, 4), dtype=np.float32)
        expected[1:3, 1:3] = slice_np
        np.testing.assert_array_equal(buf.to_numpy(), expected)

    def test_buffer_store_slice_stepped(self) -> None:
        """F.buffer_store_slice honors non-unit steps."""
        buf = Buffer.zeros([8], DType.float32, CPU())
        slice_np = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            a = Tensor(storage=buf)
            b = Tensor.from_dlpack(slice_np)
            F.buffer_store_slice(a, b, [slice(0, 8, 2)])

        expected = np.zeros(8, dtype=np.float32)
        expected[0:8:2] = slice_np
        np.testing.assert_array_equal(buf.to_numpy(), expected)

    def test_buffer_store_slice_heterogeneous_steps(self) -> None:
        """F.buffer_store_slice handles different steps per axis."""
        buf = Buffer.zeros([6, 6], DType.float32, CPU())
        slice_np = np.arange(6, dtype=np.float32).reshape(3, 2) + 1.0

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            a = Tensor(storage=buf)
            b = Tensor.from_dlpack(slice_np)
            F.buffer_store_slice(a, b, [slice(0, 6, 2), slice(1, 6, 3)])

        expected = np.zeros((6, 6), dtype=np.float32)
        expected[0:6:2, 1:6:3] = slice_np
        np.testing.assert_array_equal(buf.to_numpy(), expected)

    def test_buffer_store_slice_negative_indices(self) -> None:
        """F.buffer_store_slice supports negative start/stop."""
        buf = Buffer.from_numpy(np.arange(10, dtype=np.float32))
        slice_np = np.array([100.0, 200.0, 300.0], dtype=np.float32)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            a = Tensor(storage=buf)
            b = Tensor.from_dlpack(slice_np)
            F.buffer_store_slice(a, b, [slice(-4, -1)])

        expected = np.arange(10, dtype=np.float32)
        expected[-4:-1] = slice_np
        np.testing.assert_array_equal(buf.to_numpy(), expected)

    def test_buffer_store_slice_bfloat16_cpu(self) -> None:
        """F.buffer_store_slice works end-to-end on a bf16 CPU buffer."""
        # bf16 is 2 bytes/element. Build source/dest via uint16 bytes then
        # view as bf16 to avoid DLPack entirely.
        dst_u16 = np.zeros((4, 4), dtype=np.uint16)
        src_u16 = np.array(
            [[0x3F80, 0x4000], [0x4040, 0x4080]], dtype=np.uint16
        )

        buf = Buffer.from_numpy(dst_u16).view(DType.bfloat16)
        src_buf = Buffer.from_numpy(src_u16).view(DType.bfloat16)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            a = Tensor(storage=buf)
            b = Tensor(storage=src_buf)
            F.buffer_store_slice(a, b, [slice(1, 3), slice(1, 3)])

        # Compare via uint16 view: slice coords and element bytes match.
        expected = np.zeros((4, 4), dtype=np.uint16)
        expected[1:3, 1:3] = src_u16
        np.testing.assert_array_equal(
            buf.view(DType.uint16).to_numpy(), expected
        )

    def test_buffer_store_slice_float8_e4m3fn_cpu(self) -> None:
        """F.buffer_store_slice works end-to-end on a float8_e4m3fn CPU buffer."""
        # fp8 is 1 byte per element; build source/dest via uint8 bytes then
        # view as fp8 to avoid DLPack entirely.
        dst_bytes = np.zeros((4, 4), dtype=np.uint8)
        src_bytes = np.array([[0x11, 0x22], [0x33, 0x44]], dtype=np.uint8)

        buf = Buffer.from_numpy(dst_bytes).view(DType.float8_e4m3fn)
        src_buf = Buffer.from_numpy(src_bytes).view(DType.float8_e4m3fn)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            a = Tensor(storage=buf)
            b = Tensor(storage=src_buf)
            F.buffer_store_slice(a, b, [slice(1, 3), slice(1, 3)])

        # Compare via uint8 view: slice coords and element bytes match.
        expected = np.zeros((4, 4), dtype=np.uint8)
        expected[1:3, 1:3] = src_bytes
        np.testing.assert_array_equal(
            buf.view(DType.uint8).to_numpy(), expected
        )

    def test_buffer_store_slice_float8_e5m2_cpu(self) -> None:
        """F.buffer_store_slice works end-to-end on a float8_e5m2 CPU buffer."""
        dst_bytes = np.zeros((4, 4), dtype=np.uint8)
        src_bytes = np.array([[0x55, 0x66], [0x77, 0x88]], dtype=np.uint8)

        buf = Buffer.from_numpy(dst_bytes).view(DType.float8_e5m2)
        src_buf = Buffer.from_numpy(src_bytes).view(DType.float8_e5m2)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            a = Tensor(storage=buf)
            b = Tensor(storage=src_buf)
            F.buffer_store_slice(a, b, [slice(1, 3), slice(1, 3)])

        expected = np.zeros((4, 4), dtype=np.uint8)
        expected[1:3, 1:3] = src_bytes
        np.testing.assert_array_equal(
            buf.view(DType.uint8).to_numpy(), expected
        )

    def test_buffer_store_slice_float4_e2m1fn_raises(self) -> None:
        """Slice writes on float4_e2m1fn still raise (packed 4-bit unsupported)."""
        buf = Buffer.zeros([4, 4], DType.float4_e2m1fn, CPU())
        src = Buffer.zeros([2, 2], DType.float4_e2m1fn, CPU())

        # Realization happens on EagerRealizationContext exit, so the raise
        # must wrap the whole context.
        with pytest.raises(NotImplementedError, match="float4_e2m1fn"):
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                a = Tensor(storage=buf)
                b = Tensor(storage=src)
                F.buffer_store_slice(a, b, [slice(1, 3), slice(1, 3)])


class TestResizeLinearOp:
    """Tests for linear (bilinear) resize interpreter op (mo.resize.linear).

    Routes through F.resize_linear -> ops.resize_linear ->
    rmo.MoResizeLinearOp -> mo.ResizeLinearOp -> _handle_resize_linear ->
    resize_gc.resize_model.  CPU-only (MO_HostOnly).

    The reference is a pure-numpy separable 1-D linear interpolation along
    each spatial dimension (dimensions 2 and beyond).  Four coordinate
    transformation modes are supported -- see ``_resize_linear_ref``.
    For ``antialias=True``, a tent-filtered reference is computed by
    widening the linear kernel support by ``1/scale`` when downscaling.
    """

    @staticmethod
    def _coord(
        x_out: int,
        in_size: int,
        out_size: int,
        mode: int,
    ) -> float:
        """Map output coordinate to input coordinate.

        Args:
            x_out: Output pixel index.
            in_size: Input dimension size.
            out_size: Output dimension size.
            mode: 0=half_pixel, 1=align_corners, 2=asymmetric, 3=half_pixel_1D.

        Returns:
            Corresponding input coordinate (may be fractional).
        """
        scale = in_size / out_size
        if mode == 1:  # align_corners
            return (
                float(x_out) * (in_size - 1) / (out_size - 1)
                if out_size > 1
                else 0.0
            )
        if mode == 2:  # asymmetric
            return float(x_out) * scale
        # half_pixel (0) and half_pixel_1D (3)
        return (float(x_out) + 0.5) * scale - 0.5

    @staticmethod
    def _resize_1d(
        arr: np.ndarray,
        out_size: int,
        dim: int,
        mode: int,
        antialias: bool,
    ) -> np.ndarray:
        """Resize ``arr`` along a single spatial axis using linear interpolation.

        Args:
            arr: Input array (float64 working precision).
            out_size: Desired output size along ``dim``.
            dim: The axis to resize.
            mode: Coordinate transformation mode (0-3).
            antialias: Widen the tent filter by ``1/scale`` when downscaling.

        Returns:
            Array with ``arr.shape[dim]`` replaced by ``out_size``.
        """
        in_size = arr.shape[dim]
        if in_size == out_size:
            return arr

        rank = arr.ndim
        new_shape = list(arr.shape)
        new_shape[dim] = out_size
        out = np.zeros(new_shape, dtype=np.float64)
        scale = in_size / out_size

        for x_out in range(out_size):
            x_in = TestResizeLinearOp._coord(x_out, in_size, out_size, mode)

            if antialias and scale > 1.0:
                # Widened tent filter: sample in [x_in - r, x_in + r], r = scale
                r = scale
                x0 = int(np.floor(x_in - r + 1e-6))
                x1 = int(np.ceil(x_in + r - 1e-6))
                total_w = 0.0
                acc_idx: list[Any] = [slice(None)] * rank
                out_idx: list[Any] = [slice(None)] * rank
                out_idx[dim] = x_out
                out[tuple(out_idx)] = 0.0
                for xi in range(max(0, x0), min(in_size, x1 + 1)):
                    w = max(0.0, 1.0 - abs((xi - x_in) / r))
                    acc_idx[dim] = xi
                    out[tuple(out_idx)] += w * arr[tuple(acc_idx)]
                    total_w += w
                if total_w > 0.0:
                    out[tuple(out_idx)] /= total_w
            else:
                x_in_c = float(np.clip(x_in, 0.0, in_size - 1))
                x0_i = int(np.floor(x_in_c))
                x1_i = min(x0_i + 1, in_size - 1)
                w = x_in_c - x0_i
                idx0: list[Any] = [slice(None)] * rank
                idx1: list[Any] = [slice(None)] * rank
                idx_out: list[Any] = [slice(None)] * rank
                idx0[dim] = x0_i
                idx1[dim] = x1_i
                idx_out[dim] = x_out
                out[tuple(idx_out)] = (1.0 - w) * arr[tuple(idx0)] + w * arr[
                    tuple(idx1)
                ]

        return out

    @staticmethod
    def _resize_linear_ref(
        x_np: np.ndarray,
        out_shape: list[int],
        coordinate_transform_mode: int = 0,
        antialias: bool = False,
    ) -> np.ndarray:
        """Separable numpy reference for linear (bilinear) resize.

        Applies ``_resize_1d`` sequentially along each spatial dimension
        (dims 2 and beyond) in float64 working precision.

        Args:
            x_np: Input array.
            out_shape: Full output shape (same rank as ``x_np``).
            coordinate_transform_mode: 0=half_pixel (default),
                1=align_corners, 2=asymmetric, 3=half_pixel_1D.
            antialias: Widen tent filter when downscaling.

        Returns:
            Output array cast back to ``x_np.dtype``.
        """
        out = x_np.astype(np.float64)
        rank = x_np.ndim
        for dim in range(2, rank):
            out = TestResizeLinearOp._resize_1d(
                out, out_shape[dim], dim, coordinate_transform_mode, antialias
            )
        return out.astype(x_np.dtype)

    def test_2d_upsample(self) -> None:
        """Upsample 4x4 spatial to 8x8 with half_pixel (coord_mode=0)."""
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((1, 1, 4, 4)).astype(np.float32)
        x = Tensor.from_dlpack(x_np)
        out_shape = [1, 1, 8, 8]

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.resize_linear(x, out_shape, coordinate_transform_mode=0)

        ref = self._resize_linear_ref(
            x_np, out_shape, coordinate_transform_mode=0
        )
        np.testing.assert_allclose(
            np.from_dlpack(out), ref, rtol=1e-4, atol=1e-4
        )

    def test_2d_downscale(self) -> None:
        """Downscale 8x8 spatial to 4x4 with half_pixel (coord_mode=0)."""
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
        x = Tensor.from_dlpack(x_np)
        out_shape = [1, 3, 4, 4]

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.resize_linear(x, out_shape, coordinate_transform_mode=0)

        ref = self._resize_linear_ref(
            x_np, out_shape, coordinate_transform_mode=0
        )
        np.testing.assert_allclose(
            np.from_dlpack(out), ref, rtol=1e-4, atol=1e-4
        )

    def test_align_corners(self) -> None:
        """Resize with align_corners coordinate mode (coord_mode=1)."""
        rng = np.random.default_rng(2)
        x_np = rng.standard_normal((1, 2, 4, 4)).astype(np.float32)
        x = Tensor.from_dlpack(x_np)
        out_shape = [1, 2, 7, 7]

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.resize_linear(x, out_shape, coordinate_transform_mode=1)

        ref = self._resize_linear_ref(
            x_np, out_shape, coordinate_transform_mode=1
        )
        np.testing.assert_allclose(
            np.from_dlpack(out), ref, rtol=1e-4, atol=1e-4
        )

    def test_antialias_downscale(self) -> None:
        """Downscaling with antialias=True produces finite values."""
        rng = np.random.default_rng(3)
        x_np = rng.standard_normal((1, 1, 8, 8)).astype(np.float32)
        x = Tensor.from_dlpack(x_np)
        out_shape = [1, 1, 3, 3]

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.resize_linear(
                x, out_shape, coordinate_transform_mode=0, antialias=True
            )

        out_np = np.from_dlpack(out)
        assert out_np.shape == tuple(out_shape)
        assert np.all(np.isfinite(out_np)), (
            "antialias output contains NaN or Inf"
        )

        ref = self._resize_linear_ref(
            x_np, out_shape, coordinate_transform_mode=0, antialias=True
        )
        np.testing.assert_allclose(out_np, ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype: DType) -> None:
        """Resize works for every dtype resize_gc supports on CPU
        (resize_gc._RESIZE_DTYPES == FLOAT_DTYPES)."""
        rng = np.random.default_rng(4)
        np_dtype = dtype.to_numpy()
        x_np = rng.standard_normal((1, 2, 4, 4)).astype(np_dtype)
        x = Tensor.from_dlpack(x_np)
        out_shape = [1, 2, 6, 6]

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.resize_linear(x, out_shape)

        ref = self._resize_linear_ref(x_np, out_shape)
        np.testing.assert_allclose(
            np.from_dlpack(out), ref, rtol=1e-4, atol=1e-4
        )

    def test_float16_unsupported(self) -> None:
        """float16 doesn't compile through the GC resize path on CPU.

        Unlike the old hand-written Mojo kernel, the graph-compiler CPU
        backend has no float16 kernel (resize_gc._RESIZE_DTYPES excludes it,
        matching gc_compile.CPU_FLOAT_DTYPES's exclusion for other families).
        """
        rng = np.random.default_rng(4)
        x_np = rng.standard_normal((1, 2, 4, 4)).astype(np.float16)
        x = Tensor.from_dlpack(x_np)
        out_shape = [1, 2, 6, 6]

        with pytest.raises(KeyError, match="float16"):
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                # Keep a reference alive through context exit (where
                # realize_all -> the handler -> resize_model actually raises)
                # -- an unassigned call's Tensor can get GC'd first.
                out = F.resize_linear(x, out_shape)
                assert out is not None

    def test_3d_input(self) -> None:
        """Resize a rank-3 (NCW) input using 1-D linear interpolation."""
        rng = np.random.default_rng(5)
        x_np = rng.standard_normal((1, 4, 8)).astype(np.float32)
        x = Tensor.from_dlpack(x_np)
        out_shape = [1, 4, 16]

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.resize_linear(x, out_shape)

        ref = self._resize_linear_ref(x_np, out_shape)
        np.testing.assert_allclose(
            np.from_dlpack(out), ref, rtol=1e-4, atol=1e-4
        )


class TestResizeNearestOp:
    """Tests for nearest-neighbor resize interpreter op (mo.resize.nearest).

    Routes through F.resize_nearest -> ops.resize_nearest ->
    rmo.MoResizeNearestOp -> mo.ResizeNearestOp -> _handle_resize_nearest ->
    resize_gc.resize_model.  CPU-only (MO_HostOnly).

    The reference is a pure-numpy nearest-neighbor lookup applied to every
    dimension.  Four coordinate transformation modes and four rounding modes
    are supported -- see ``_resize_nearest_ref``.
    """

    @staticmethod
    def _coord(
        x_out: int,
        in_size: int,
        out_size: int,
        mode: int,
    ) -> float:
        """Map output coordinate to input coordinate.

        Args:
            x_out: Output pixel index.
            in_size: Input dimension size.
            out_size: Output dimension size.
            mode: 0=half_pixel, 1=align_corners, 2=asymmetric, 3=half_pixel_1D.

        Returns:
            Corresponding input coordinate (may be fractional).
        """
        scale = out_size / in_size
        if mode == 1:  # align_corners
            return (
                float(x_out) * (in_size - 1) / (out_size - 1)
                if out_size > 1
                else 0.0
            )
        if mode == 2:  # asymmetric
            return float(x_out) / scale
        if mode == 3:  # half_pixel_1D
            if out_size == 1:
                return 0.0
            return (float(x_out) + 0.5) / scale - 0.5
        # half_pixel (0)
        return (float(x_out) + 0.5) / scale - 0.5

    @staticmethod
    def _round(val: float, round_mode: int) -> int:
        """Round a coordinate using the specified rounding mode.

        Args:
            val: Fractional input coordinate.
            round_mode: 0=HalfDown, 1=HalfUp, 2=Floor, 3=Ceil.

        Returns:
            Rounded integer coordinate.
        """
        if round_mode == 0:  # HalfDown: ceil(x - 0.5)
            return int(np.ceil(val - 0.5))
        if round_mode == 1:  # HalfUp: floor(x + 0.5)
            return int(np.floor(val + 0.5))
        if round_mode == 2:  # Floor
            return int(np.floor(val))
        # Ceil (3)
        return int(np.ceil(val))

    @staticmethod
    def _resize_nearest_ref(
        x_np: np.ndarray,
        out_shape: list[int],
        coordinate_transform_mode: int = 0,
        round_mode: int = 0,
    ) -> np.ndarray:
        """Numpy reference for nearest-neighbor resize across all dimensions.

        Args:
            x_np: Input array.
            out_shape: Full output shape (same rank as ``x_np``).
            coordinate_transform_mode: 0=half_pixel (default),
                1=align_corners, 2=asymmetric, 3=half_pixel_1D.
            round_mode: 0=HalfDown (default), 1=HalfUp, 2=Floor, 3=Ceil.

        Returns:
            Output array with ``out_shape`` and ``x_np.dtype``.
        """
        rank = x_np.ndim
        out = np.empty(out_shape, dtype=x_np.dtype)

        for out_idx in np.ndindex(*out_shape):
            in_idx = []
            for d in range(rank):
                mapped = TestResizeNearestOp._coord(
                    out_idx[d],
                    x_np.shape[d],
                    out_shape[d],
                    coordinate_transform_mode,
                )
                rounded = TestResizeNearestOp._round(mapped, round_mode)
                clamped = min(rounded, x_np.shape[d] - 1)
                in_idx.append(clamped)
            out[out_idx] = x_np[tuple(in_idx)]

        return out

    def test_2d_upsample(self) -> None:
        """Upsample 4x4 spatial to 8x8 with half_pixel (coord_mode=0)."""
        rng = np.random.default_rng(100)
        x_np = rng.standard_normal((1, 1, 4, 4)).astype(np.float32)
        x = Tensor.from_dlpack(x_np)
        out_shape = [1, 1, 8, 8]

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.resize_nearest(x, out_shape, coordinate_transform_mode=0)

        ref = self._resize_nearest_ref(
            x_np, out_shape, coordinate_transform_mode=0
        )
        np.testing.assert_allclose(np.from_dlpack(out), ref)

    def test_2d_downscale(self) -> None:
        """Downscale 8x8 spatial to 4x4 with half_pixel (coord_mode=0)."""
        rng = np.random.default_rng(101)
        x_np = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
        x = Tensor.from_dlpack(x_np)
        out_shape = [1, 3, 4, 4]

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.resize_nearest(x, out_shape, coordinate_transform_mode=0)

        ref = self._resize_nearest_ref(
            x_np, out_shape, coordinate_transform_mode=0
        )
        np.testing.assert_allclose(np.from_dlpack(out), ref)

    def test_floor_round_mode(self) -> None:
        """Upsample with round_mode=2 (Floor), asymmetric coord mode.

        Uses asymmetric (mode=2) bc half_pixel + floor can produce negative
        input coordinates at the boundary, which the kernel doesn't clamp.
        """
        rng = np.random.default_rng(102)
        x_np = rng.standard_normal((1, 1, 3, 3)).astype(np.float32)
        x = Tensor.from_dlpack(x_np)
        out_shape = [1, 1, 6, 6]

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.resize_nearest(
                x, out_shape, coordinate_transform_mode=2, round_mode=2
            )

        ref = self._resize_nearest_ref(
            x_np, out_shape, coordinate_transform_mode=2, round_mode=2
        )
        np.testing.assert_allclose(np.from_dlpack(out), ref)

    def test_align_corners(self) -> None:
        """Resize with align_corners coordinate mode (coord_mode=1)."""
        rng = np.random.default_rng(103)
        x_np = rng.standard_normal((1, 2, 4, 4)).astype(np.float32)
        x = Tensor.from_dlpack(x_np)
        out_shape = [1, 2, 7, 7]

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.resize_nearest(x, out_shape, coordinate_transform_mode=1)

        ref = self._resize_nearest_ref(
            x_np, out_shape, coordinate_transform_mode=1
        )
        np.testing.assert_allclose(np.from_dlpack(out), ref)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype: DType) -> None:
        """Resize works for every dtype resize_gc supports on CPU
        (resize_gc._RESIZE_DTYPES == FLOAT_DTYPES)."""
        rng = np.random.default_rng(104)
        np_dtype = dtype.to_numpy()
        x_np = rng.standard_normal((1, 2, 4, 4)).astype(np_dtype)
        x = Tensor.from_dlpack(x_np)
        out_shape = [1, 2, 6, 6]

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.resize_nearest(x, out_shape)

        ref = self._resize_nearest_ref(x_np, out_shape)
        np.testing.assert_allclose(np.from_dlpack(out), ref)

    def test_float16_unsupported(self) -> None:
        """float16 doesn't compile through the GC resize path on CPU.

        Unlike the old hand-written Mojo kernel, the graph-compiler CPU
        backend has no float16 kernel (resize_gc._RESIZE_DTYPES excludes it,
        matching gc_compile.CPU_FLOAT_DTYPES's exclusion for other families).
        """
        rng = np.random.default_rng(104)
        x_np = rng.standard_normal((1, 2, 4, 4)).astype(np.float16)
        x = Tensor.from_dlpack(x_np)
        out_shape = [1, 2, 6, 6]

        with pytest.raises(KeyError, match="float16"):
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                # Keep a reference alive through context exit (where
                # realize_all -> the handler -> resize_model actually raises)
                # -- an unassigned call's Tensor can get GC'd first.
                out = F.resize_nearest(x, out_shape)
                assert out is not None

    def test_3d_input(self) -> None:
        """Resize a rank-3 (NCW) input using nearest-neighbor."""
        rng = np.random.default_rng(105)
        x_np = rng.standard_normal((1, 4, 8)).astype(np.float32)
        x = Tensor.from_dlpack(x_np)
        out_shape = [1, 4, 16]

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.resize_nearest(x, out_shape)

        ref = self._resize_nearest_ref(x_np, out_shape)
        np.testing.assert_allclose(np.from_dlpack(out), ref)


class TestResizeBicubicOp:
    """Tests for bicubic resize interpreter op (mo.resize.bicubic)."""

    def test_unsupported(self) -> None:
        """resize_bicubic has no supported kernel (GEX-3990): its old Mojo
        binding has been deleted and the graph compiler has no
        shape-fallback registration for MO::ResizeBicubicOp, so the handler
        now raises rather than falling back to a Mojo kernel."""
        rng = np.random.default_rng(200)
        x_np = rng.standard_normal((1, 1, 4, 4)).astype(np.float32)
        x = Tensor.from_dlpack(x_np)
        out_shape = [1, 1, 8, 8]

        with pytest.raises(NotImplementedError, match="resize_bicubic"):
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                out = F.resize_bicubic(x, out_shape)
                assert out is not None


class TestDistributedScatterSimulated:
    """Test distributed_scatter on a simulated CPU mesh."""

    def test_scatter_simulated_fallback(self) -> None:
        """Simulated mesh: distributed_scatter falls back to transfer_to."""
        cpu = CPU()
        mesh = DeviceMesh(
            devices=(cpu, cpu), mesh_shape=(2,), axis_names=("dp",)
        )

        data = np.arange(8, dtype=np.float32).reshape(4, 2)
        mapping = PlacementMapping(mesh, (Sharded(0),))

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = df_shard(Tensor(data), mapping)

        assert result.placements == (Sharded(0),)
        assert len(result.local_shards) == 2
        np.testing.assert_allclose(result.local_shards[0].to_numpy(), data[:2])
        np.testing.assert_allclose(result.local_shards[1].to_numpy(), data[2:])


class TestDistributedBroadcastSimulated:
    """Test distributed_broadcast on a simulated CPU mesh."""

    def test_broadcast_simulated_fallback(self) -> None:
        """Simulated mesh: distributed_broadcast falls back to transfer_to."""
        cpu = CPU()
        mesh = DeviceMesh(
            devices=(cpu, cpu), mesh_shape=(2,), axis_names=("dp",)
        )

        data = np.arange(8, dtype=np.float32).reshape(4, 2)
        t = Tensor.from_dlpack(data)

        mapping = PlacementMapping(mesh, (Replicated(),))

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = df_shard(t, mapping)

        assert result.placements == (Replicated(),)
        assert len(result.local_shards) == 2
        np.testing.assert_allclose(result.local_shards[0].to_numpy(), data)
        np.testing.assert_allclose(result.local_shards[1].to_numpy(), data)


class TestDistributedReducescatterSumSimulated:
    """Test distributed_reducescatter_sum on a simulated CPU mesh."""

    def test_reducescatter_sum_simulated_fallback(self) -> None:
        """Simulated mesh: reduce-scatter falls back to add + split."""
        cpu = CPU()
        mesh = DeviceMesh(
            devices=(cpu, cpu), mesh_shape=(2,), axis_names=("tp",)
        )

        # Two [4, 2] partial contributions — one per device.
        data_a = np.arange(8, dtype=np.float32).reshape(4, 2)
        data_b = np.arange(8, 16, dtype=np.float32).reshape(4, 2)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            # Build a Partial tensor with different data per shard.
            shard_a = Tensor(data_a).__tensorvalue__()
            shard_b = Tensor(data_b).__tensorvalue__()
            partial_t = Tensor.from_shard_values(
                [shard_a, shard_b],
                PlacementMapping(mesh, (Partial(),)),
            )
            result = reduce_scatter(partial_t, scatter_axis=0, mesh_axis=0)

        assert result.placements == (Sharded(0),)
        assert len(result.local_shards) == 2
        total = data_a + data_b
        np.testing.assert_allclose(result.local_shards[0].to_numpy(), total[:2])
        np.testing.assert_allclose(result.local_shards[1].to_numpy(), total[2:])


class TestLazyGCModelCompilation:
    """Direct tests for the per-(op, device, dtype) GC model caches.

    Covers behavior the dispatch handlers rely on but don't assert: compile-once
    reuse, unsupported unary targets raising, and ``MAX_EAGER_OP_PRECOMPILE``
    selecting lazy (default) vs the opt-in precompile sweep.
    """

    @pytest.fixture(autouse=True)
    def _isolate_from_session_warm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run these decision-logic tests free of any session-wide warm cache.

        The shared conftest points ``MODULAR_DERIVED_PATH`` at a build-time warm
        dir and this target sets ``MODULAR_EAGER_WARM_ADOPT_ASSERTED=1``, so the
        import-time precompile force-loads it. This class instead unit-tests the
        lazy/stamp/per-target decision logic against a mocked cache dir, so an
        always-adoptable process-wide manifest would short-circuit the very path
        under test. Clear both; tests that need a manifest set their own env (or
        patch ``read_manifest``) in-body, after this fixture.
        """
        monkeypatch.delenv("MODULAR_DERIVED_PATH", raising=False)
        monkeypatch.delenv("MODULAR_EAGER_WARM_ADOPT_ASSERTED", raising=False)

    def test_matmul_model_compiles_once_and_reuses(self) -> None:
        """A second call for the same (device, dtype) returns the cached model."""
        cpu = CPU()
        first = matmul_gc.matmul_model(cpu, DType.float32)
        second = matmul_gc.matmul_model(cpu, DType.float32)
        assert first is second

    def test_unary_model_compiles_once_and_reuses(self) -> None:
        """A second call for the same unary target returns the cached model."""
        cpu = CPU()
        first = unary_elementwise_gc.unary_model(mo.ExpOp, cpu, DType.float32)
        second = unary_elementwise_gc.unary_model(mo.ExpOp, cpu, DType.float32)
        assert first is second

    def test_unary_model_unsupported_dtype_raises(self) -> None:
        """A transcendental op on an int dtype is outside the supported set."""
        # Exp only sweeps float dtypes; int32 is unsupported and must not be
        # handed to load_all as an uncompilable graph.
        with pytest.raises(KeyError, match="Unsupported unary op/device/dtype"):
            unary_elementwise_gc.unary_model(mo.ExpOp, CPU(), DType.int32)

    def test_unary_model_canonicalizes_rmo_alias(self) -> None:
        """An rmo (Mo-prefixed) unary op resolves like its mo form.

        Canonicalizing the op name keys ``rmo.MoExpOp`` to the same cache
        entry as ``mo.ExpOp``; without it a supported op misses and raises
        KeyError (see gc_compile.canonical_op_name).
        """
        rmo_exp = rmo.MoExpOp
        cpu = CPU()
        u = unary_elementwise_gc
        assert u._graph_name(rmo_exp, cpu, DType.float32) == u._graph_name(
            mo.ExpOp, cpu, DType.float32
        )
        assert u._is_supported(rmo_exp, cpu, DType.float32)
        # Resolves to the same cached model rather than raising KeyError.
        assert u.unary_model(rmo_exp, cpu, DType.float32) is u.unary_model(
            mo.ExpOp, cpu, DType.float32
        )

    # MXF-510: a residual precompile-mode cache miss used to raise KeyError.
    @pytest.mark.parametrize(
        ("family", "call_model"),
        [
            (
                matmul_gc._FAMILY,
                lambda: matmul_gc.matmul_model(CPU(), DType.float32),
            ),
            (
                select_gc._FAMILY,
                lambda: select_gc.select_model(CPU(), DType.float32),
            ),
            (
                unary_elementwise_gc._FAMILY,
                lambda: unary_elementwise_gc.unary_model(
                    mo.ExpOp, CPU(), DType.float32
                ),
            ),
            (
                elementwise_binary_gc._FAMILY,
                lambda: elementwise_binary_gc.binary_model(
                    mo.AddOp, CPU(), DType.float32
                ),
            ),
            (
                shape_rearrange_gc._FAMILY,
                lambda: shape_rearrange_gc.model(
                    mo.ConcatOp, CPU(), DType.uint32
                ),
            ),
            # Op-keyed: each of scatter_nd/_add/_max/_min/_mul dispatches
            # through its own registered family object on a miss.
            (
                scatter_nd_gc._FAMILIES[mo.ScatterNdOp],
                lambda: scatter_nd_gc.scatter_nd_model(
                    mo.ScatterNdOp, CPU(), DType.uint32, DType.int64
                ),
            ),
        ],
        ids=[
            "matmul",
            "select",
            "unary",
            "binary",
            "shape_rearrange",
            "scatter_nd",
        ],
    )
    def test_precompile_miss_compiles(
        self,
        monkeypatch: pytest.MonkeyPatch,
        family: gc_compile.GCOpFamily,
        call_model: Callable[[], object],
    ) -> None:
        """With MAX_EAGER_OP_PRECOMPILE=1, a residual cache miss compiles."""
        monkeypatch.setenv(gc_compile.EAGER_OP_PRECOMPILE_ENV_VAR, "1")
        monkeypatch.setattr(family, "cache", {})
        assert call_model() is not None

    def test_matmul_model_lazy_default_compiles_on_miss(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """By default (env var unset) a miss compiles the target lazily."""
        monkeypatch.delenv(
            gc_compile.EAGER_OP_PRECOMPILE_ENV_VAR, raising=False
        )
        monkeypatch.setattr(matmul_gc._FAMILY, "cache", {})
        model = matmul_gc.matmul_model(CPU(), DType.float32)
        assert model is not None

    def test_select_model_compiles_once_and_reuses(self) -> None:
        """A second call for the same (device, dtype) returns the cached model."""
        cpu = CPU()
        first = select_gc.select_model(cpu, DType.float32)
        second = select_gc.select_model(cpu, DType.float32)
        assert first is second

    def test_select_model_lazy_default_compiles_on_miss(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """By default (env var unset) a miss compiles the target lazily."""
        monkeypatch.delenv(
            gc_compile.EAGER_OP_PRECOMPILE_ENV_VAR, raising=False
        )
        monkeypatch.setattr(select_gc._FAMILY, "cache", {})
        model = select_gc.select_model(CPU(), DType.float32)
        assert model is not None

    def test_unary_model_lazy_default_compiles_on_miss(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """By default (env var unset) a supported miss compiles lazily."""
        monkeypatch.delenv(
            gc_compile.EAGER_OP_PRECOMPILE_ENV_VAR, raising=False
        )
        monkeypatch.setattr(unary_elementwise_gc._FAMILY, "cache", {})
        model = unary_elementwise_gc.unary_model(mo.ExpOp, CPU(), DType.float32)
        assert model is not None

    def test_warm_stamp_roundtrip(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """write_warm_stamp then warm_stamp_matches for the same context."""
        monkeypatch.setattr(gc_compile, "_cache_dir", lambda: tmp_path)
        assert not gc_compile.warm_stamp_matches()
        gc_compile.write_warm_stamp()
        assert gc_compile.warm_stamp_matches()

    def test_gc_op_family_prefers_manifest_over_stamp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ensure_swept force-loads an adoptable manifest, skipping the stamp sweep."""

        class _FakeFamily(gc_compile.GCFamilySpec):
            name = "test"

            def build_module(self) -> Module:
                return Module()

            def build_module_for_device(
                self, device: Device, module: Module | None = None
            ) -> Module:
                return Module()

            def sweep_devices(self) -> list[Device]:
                return []

        family = gc_compile.GCOpFamily(_FakeFamily())
        calls: list[str] = []

        def fake_adopt(manifest: object) -> bool:
            calls.append("manifest")
            return True

        monkeypatch.setattr(gc_compile, "read_manifest", lambda: {"x": 1})
        monkeypatch.setattr(gc_compile, "manifest_adoptable", lambda m: True)
        monkeypatch.setattr(family, "_adopt_from_manifest", fake_adopt)
        monkeypatch.setattr(
            family, "compile_sweep", lambda: calls.append("stamp_sweep")
        )
        family.ensure_swept()
        assert calls == ["manifest"]
        assert family.swept

    def test_gc_op_family_sweeps_on_warm_stamp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no manifest but a matching warm stamp, ensure_swept batch-sweeps."""

        class _FakeFamily(gc_compile.GCFamilySpec):
            name = "test"

            def build_module(self) -> Module:
                return Module()

            def build_module_for_device(
                self, device: Device, module: Module | None = None
            ) -> Module:
                return Module()

            def sweep_devices(self) -> list[Device]:
                return []

        family = gc_compile.GCOpFamily(_FakeFamily())
        calls: list[str] = []
        monkeypatch.setattr(gc_compile, "read_manifest", lambda: None)
        monkeypatch.setattr(gc_compile, "warm_stamp_matches", lambda: True)
        monkeypatch.setattr(
            family, "compile_sweep", lambda: calls.append("sweep")
        )
        family.ensure_swept()
        assert calls == ["sweep"]
        assert family.swept

    def test_warm_families_skips_family_with_no_sweep_devices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A family that builds for none of this machine's devices (e.g. a
        GPU-only family on a CPU-only host) warms as zero ops instead of
        crashing the whole warm."""

        # MXF-599: GPU-only group_norm crashed the warm on CPU-only machines.
        class _GpuOnlyFamily(gc_compile.GCFamilySpec):
            name = "test_gpu_only"

            def build_module(self) -> Module:
                return Module()

            def build_module_for_device(
                self, device: Device, module: Module | None = None
            ) -> Module:
                return module if module is not None else Module()

            def sweep_devices(self) -> list[Device]:
                return []

        family = gc_compile.GCOpFamily(_GpuOnlyFamily())
        monkeypatch.setattr(gc_compile, "registered_families", lambda: [family])
        monkeypatch.setattr(gc_compile, "read_manifest", lambda: None)

        assert list(warm.warm_families(jobs=2)) == [("test_gpu_only", 0)]
        assert list(warm.warm_families(jobs=1)) == [("test_gpu_only", 0)]
        assert family.cache == {}

        # The manifest-adoption path must also no-op without ever building a
        # session (InferenceSession(devices=[]) silently defaults to CPU).
        monkeypatch.setattr(gc_compile, "read_manifest", lambda: {"x": 1})
        monkeypatch.setattr(gc_compile, "manifest_adoptable", lambda m: True)

        def _no_session(*args: object, **kwargs: object) -> None:
            raise AssertionError("InferenceSession must not be constructed")

        monkeypatch.setattr(gc_compile.engine, "InferenceSession", _no_session)
        family.swept = False
        family.ensure_swept()
        assert family.swept
        assert family.cache == {}

    def test_cache_dir_from_derived_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The stamp dir is MODULAR_DERIVED_PATH/cache/.max_cache, else the
        engine's own MEF cache dir."""
        monkeypatch.setenv("MODULAR_DERIVED_PATH", str(tmp_path))
        assert gc_compile._cache_dir() == tmp_path / "cache" / ".max_cache"

        monkeypatch.delenv("MODULAR_DERIVED_PATH")
        assert gc_compile._cache_dir() == _core.engine.max_cache_dir()

    def test_context_signature_stable(self) -> None:
        """The signature is stable and pins count + host/accelerator arch.

        Also a regression guard: it must not raise on a CPU-only host
        (accelerator_architecture_name raises for a CPU device).
        """
        sig = gc_compile._context_signature()
        assert sig == gc_compile._context_signature()
        assert "accelerators=" in sig and "cpu=" in sig

    def test_manifest_roundtrip_and_adoptable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("MODULAR_DERIVED_PATH", str(tmp_path))
        monkeypatch.setenv("MODULAR_EAGER_WARM_ADOPT_ASSERTED", "1")
        monkeypatch.setattr(gc_compile, "accelerator_count", lambda: 2)
        monkeypatch.setattr(
            gc_compile, "accelerator_architecture_name", lambda: "sm_100a"
        )
        envelope = {
            "host_arch": gc_compile.platform.machine(),
            "gpu": {"arch": "sm_100a"},
            "device_count": 2,
            "toolchain": {"mode": "asserted"},
        }
        entries = [
            {
                "family": "matmul",
                "device_class": "cpu",
                "mef": "matmul_cpu.mef",
            },
            {
                "family": "matmul",
                "device_class": "gpu:0",
                "mef": "matmul_slot_0.mef",
            },
        ]
        assert gc_compile.write_manifest(envelope, entries)
        m = gc_compile.read_manifest()
        assert m is not None
        assert gc_compile.manifest_adoptable(m)
        assert (
            gc_compile.manifest_entry_path(m, "matmul", "gpu:0")
            == tmp_path / "matmul_slot_0.mef"
        )
        # An unknown (family, device_class) pair resolves to None.
        assert gc_compile.manifest_entry_path(m, "unary", "cpu") is None

    def test_manifest_not_adoptable_without_optin(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("MODULAR_DERIVED_PATH", str(tmp_path))
        monkeypatch.delenv("MODULAR_EAGER_WARM_ADOPT_ASSERTED", raising=False)
        monkeypatch.setattr(gc_compile, "accelerator_count", lambda: 2)
        monkeypatch.setattr(
            gc_compile, "accelerator_architecture_name", lambda: "sm_100a"
        )
        gc_compile.write_manifest(
            {
                "gpu": {"arch": "sm_100a"},
                "device_count": 2,
                "toolchain": {"mode": "asserted"},
            },
            [],
        )
        manifest = gc_compile.read_manifest()
        assert manifest is not None
        assert not gc_compile.manifest_adoptable(manifest)

    def test_manifest_device_count_is_adoption_ceiling(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Per-slot MEFs make device_count a ceiling, not an equality: a warm
        adopts iff it has at least as many slots as this box needs (which
        force-loads slots 0..k-1); fewer means missing slots, so it can't."""
        monkeypatch.setenv("MODULAR_DERIVED_PATH", str(tmp_path))
        monkeypatch.setenv("MODULAR_EAGER_WARM_ADOPT_ASSERTED", "1")
        monkeypatch.setattr(gc_compile, "accelerator_count", lambda: 4)
        monkeypatch.setattr(
            gc_compile, "accelerator_architecture_name", lambda: "sm_100a"
        )

        def write(device_count: int) -> dict[str, object]:
            gc_compile.write_manifest(
                {
                    "host_arch": gc_compile.platform.machine(),
                    "gpu": {"arch": "sm_100a"},
                    "device_count": device_count,
                    "toolchain": {"mode": "asserted"},
                },
                [],
            )
            manifest = gc_compile.read_manifest()
            assert manifest is not None
            return manifest

        # Warmed 2 < this box's 4: slots 2,3 were never compiled -> reject.
        assert not gc_compile.manifest_adoptable(write(2))
        # Warmed 8 >= this box's 4: force-loads slots 0..3, extras unused -> adopt.
        assert gc_compile.manifest_adoptable(write(8))

    def test_manifest_not_adoptable_on_host_arch_mismatch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("MODULAR_DERIVED_PATH", str(tmp_path))
        monkeypatch.setenv("MODULAR_EAGER_WARM_ADOPT_ASSERTED", "1")
        monkeypatch.setattr(gc_compile, "accelerator_count", lambda: 2)
        monkeypatch.setattr(
            gc_compile, "accelerator_architecture_name", lambda: "sm_100a"
        )
        # Count + GPU arch match, opt-in on, only the host arch differs (the
        # per-slot CPU MEF embeds host-ELF kernels, so a foreign host can't
        # adopt).
        gc_compile.write_manifest(
            {
                "host_arch": "not-this-host",
                "gpu": {"arch": "sm_100a"},
                "device_count": 2,
                "toolchain": {"mode": "asserted"},
            },
            [],
        )
        manifest = gc_compile.read_manifest()
        assert manifest is not None
        assert not gc_compile.manifest_adoptable(manifest)

    def test_cpu_only_manifest_adoptable_when_no_accelerator(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A CPU-only manifest (no ``gpu`` key) adopts on a box with no
        accelerator, there is no GPU slot to mismatch."""
        monkeypatch.setenv("MODULAR_DERIVED_PATH", str(tmp_path))
        monkeypatch.setenv("MODULAR_EAGER_WARM_ADOPT_ASSERTED", "1")
        monkeypatch.setattr(gc_compile, "accelerator_count", lambda: 0)
        gc_compile.write_manifest(
            {
                "host_arch": gc_compile.platform.machine(),
                "cpu_target": "triple=x86_64-unknown-linux-gnu;cpu=x86-64-v3",
                "toolchain": {"mode": "asserted"},
            },
            [],
        )
        manifest = gc_compile.read_manifest()
        assert manifest is not None
        assert gc_compile.manifest_adoptable(manifest)

    def test_cpu_only_manifest_not_adoptable_with_accelerator(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A CPU-only manifest is rejected on a box that has an accelerator: its
        GPU slots were never warmed, so a GPU box must not adopt it."""
        monkeypatch.setenv("MODULAR_DERIVED_PATH", str(tmp_path))
        monkeypatch.setenv("MODULAR_EAGER_WARM_ADOPT_ASSERTED", "1")
        monkeypatch.setattr(gc_compile, "accelerator_count", lambda: 2)
        # The CPU-only branch returns before consulting
        # accelerator_architecture_name; stub it anyway so a stray call can't
        # raise on a CPU host.
        monkeypatch.setattr(
            gc_compile, "accelerator_architecture_name", lambda: "sm_100a"
        )
        gc_compile.write_manifest(
            {
                "host_arch": gc_compile.platform.machine(),
                "cpu_target": "triple=x86_64-unknown-linux-gnu;cpu=x86-64-v3",
                "toolchain": {"mode": "asserted"},
            },
            [],
        )
        manifest = gc_compile.read_manifest()
        assert manifest is not None
        assert not gc_compile.manifest_adoptable(manifest)

    def test_read_manifest_absent_is_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("MODULAR_DERIVED_PATH", str(tmp_path))
        assert gc_compile.read_manifest() is None

    def test_binary_model_compiles_once_and_reuses(self) -> None:
        """A second call for the same binary target returns the cached model."""
        cpu = CPU()
        first = elementwise_binary_gc.binary_model(mo.AddOp, cpu, DType.float32)
        second = elementwise_binary_gc.binary_model(
            mo.AddOp, cpu, DType.float32
        )
        assert first is second

    def test_binary_comparison_model_compiles(self) -> None:
        """A comparison op (bool output) compiles like an arithmetic one."""
        model = elementwise_binary_gc.binary_model(
            mo.GreaterOp, CPU(), DType.float32
        )
        assert model is not None

    def test_binary_model_unsupported_dtype_raises(self) -> None:
        """Div sweeps floats only; an int dtype is outside the supported set."""
        with pytest.raises(
            KeyError, match="Unsupported binary op/device/dtype"
        ):
            elementwise_binary_gc.binary_model(mo.DivOp, CPU(), DType.int32)

    def test_binary_model_pow_integer_supported(self) -> None:
        """Pow sweeps NUMERIC, so an int Pow compiles (not the Div case)."""
        model = elementwise_binary_gc.binary_model(mo.PowOp, CPU(), DType.int32)
        assert model is not None

    def test_binary_model_lazy_default_compiles_on_miss(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """By default (env var unset) a supported miss compiles lazily."""
        monkeypatch.delenv(
            gc_compile.EAGER_OP_PRECOMPILE_ENV_VAR, raising=False
        )
        monkeypatch.setattr(elementwise_binary_gc._FAMILY, "cache", {})
        model = elementwise_binary_gc.binary_model(
            mo.AddOp, CPU(), DType.float32
        )
        assert model is not None

    def test_uint_view_dtype_rejects_sub_byte(self) -> None:
        """Byte-aligned dtypes bit-cast to a same-width uint; sub-byte raises."""
        assert gc_compile.uint_view_dtype(DType.float16) == DType.uint16
        assert gc_compile.uint_view_dtype(DType.float64) == DType.uint64
        assert gc_compile.uint_view_dtype(DType.bool) == DType.uint8
        assert gc_compile.uint_view_dtype(DType.float8_e4m3fn) == DType.uint8
        with pytest.raises(NotImplementedError, match="sub-byte"):
            gc_compile.uint_view_dtype(DType.float4_e2m1fn)

    def test_tile_rank_over_cap_raises(self) -> None:
        """Rank beyond an op's cap raises cleanly, not a kernel comptime assert.

        Uses the same KeyError "Unsupported" signal as an unsupported dtype.
        """
        # tile's GC kernel supports up to rank 4; rank 5 must be rejected here.
        with pytest.raises(
            KeyError, match="Unsupported shape-rearrange rank 5"
        ):
            shape_rearrange_gc.model(mo.TileOp, CPU(), DType.uint8, rank=5)


class TestDataMovementGCFamily:
    """Direct family-level tests for data_movement_gc graphs.

    Every graph is built at ``data_movement_gc.MAX_RANK``; the handlers
    pad lower-rank shapes to it with leading 1s, so these direct model
    calls use rank-5 inputs (the padded form the handlers produce).
    """

    def test_family_transpose_model(self) -> None:
        x = np.arange(24, dtype=np.uint32).reshape(1, 1, 2, 3, 4)
        model = data_movement_gc.model(mo.TransposeOp, CPU(), DType.uint32)
        (out,) = model(
            Buffer.from_numpy(x),
            Buffer.from_numpy(np.array([0, 1, 4, 2, 3], dtype=np.int64)),
        )
        np.testing.assert_array_equal(
            out.to_numpy(), np.transpose(x, (0, 1, 4, 2, 3))
        )

    def test_family_broadcast_model(self) -> None:
        x = np.array([[1], [2]], dtype=np.uint64).reshape(1, 1, 1, 2, 1)
        model = data_movement_gc.model(mo.BroadcastToOp, CPU(), DType.uint64)
        # Output dims bind from the shapes of the host carrier inputs.
        (out,) = model(
            Buffer.from_numpy(x),
            *(
                Buffer.from_numpy(np.zeros(dim, dtype=np.uint8))
                for dim in (1, 1, 1, 2, 3)
            ),
        )
        np.testing.assert_array_equal(
            out.to_numpy(), np.broadcast_to(x, (1, 1, 1, 2, 3))
        )

    def test_family_store_slice_model(self) -> None:
        dst = Buffer.from_numpy(np.zeros((1, 1, 1, 4, 4), dtype=np.uint8))
        src = Buffer.from_numpy(np.ones((1, 1, 1, 2, 2), dtype=np.uint8) * 5)
        model = data_movement_gc.model(
            mo.MutableStoreSliceOp, CPU(), DType.uint8
        )
        model(
            dst,
            src,
            Buffer.from_numpy(np.array([0, 0, 0, 1, 1], dtype=np.int64)),
            Buffer.from_numpy(np.array([1, 1, 1, 3, 3], dtype=np.int64)),
            Buffer.from_numpy(np.array([1, 1, 1, 1, 1], dtype=np.int64)),
        )
        expected = np.zeros((1, 1, 1, 4, 4), dtype=np.uint8)
        expected[..., 1:3, 1:3] = 5
        np.testing.assert_array_equal(dst.to_numpy(), expected)

    def test_broadcast_gc_rejects_rank_over_max(self) -> None:
        """The fixed-rank graphs cap at MAX_RANK; a rank-6 target must
        raise the handler's clear error, not an opaque compile failure."""
        input_buf = Buffer.from_numpy(np.zeros((1,) * 6, dtype=np.float32))
        with pytest.raises(ValueError, match="at most rank-5"):
            data_movement_gc.broadcast_to(input_buf, [1] * 6, CPU())

    def test_broadcast_gc_rank0_is_a_copy(self) -> None:
        """Rank 0 bypasses the compiled model (ranks 1..MAX_RANK only) and
        falls back to a driver copy in ``data_movement_gc.broadcast_to``;
        result is a distinct buffer, not an alias of the input."""
        input_buf = Buffer.from_numpy(np.array(7.0, dtype=np.float32))
        out = data_movement_gc.broadcast_to(input_buf, [], CPU())

        assert out is not input_buf
        np.testing.assert_array_equal(out.to_numpy(), input_buf.to_numpy())

        # Mutating the output must not affect the input: proves a real
        # copy rather than an aliased view.
        out.to_numpy()[...] = 99.0
        assert float(input_buf.to_numpy()) == 7.0

    def test_broadcast_gc_rejects_illegal_shape(self) -> None:
        """An input dim that is neither 1 nor equal to the target dim is an
        illegal broadcast; ``data_movement_gc.broadcast_to`` must raise rather
        than silently dispatching a model that would misbehave."""
        input_buf = Buffer.from_numpy(np.zeros(3, dtype=np.float32))

        with pytest.raises(ValueError, match="dim 0"):
            data_movement_gc.broadcast_to(input_buf, [5], CPU())
