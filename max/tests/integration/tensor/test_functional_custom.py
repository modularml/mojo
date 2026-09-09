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

import os
from pathlib import Path

import pytest
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.executor import (
    InterpreterExecutor,
    JitExecutor,
    set_default_executor,
)
from max.experimental.tensor import Tensor
from max.nn import kernels

DEVICE = Accelerator() if accelerator_count() else CPU()


moe_create_indices = F.functional(kernels.moe_create_indices)
scatter_set_constant = F.functional(kernels.scatter_set_constant)


@pytest.fixture
def kernel_verification_ops_path() -> Path:
    return Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])


@pytest.mark.skipif(
    DEVICE.is_host, reason="moe_create_indices only supports GPU devices"
)
def test_custom() -> None:
    indices = Tensor.ones([4], dtype=DType.int32, device=DEVICE)
    token_expert_order, *_rest = moe_create_indices(indices, 8)
    assert token_expert_order.real


@pytest.mark.skipif(
    DEVICE.is_host, reason="scatter_set_constant only supports GPU devices"
)
def test_inplace_custom() -> None:
    # Prior allocate-then-free perturbs the heap next to the buffers below;
    # the mutation must still land at exactly the requested coordinate.
    junk = Tensor.ones([4])
    del junk

    values = Tensor.zeros([2, 2])
    # Each indices row is a (row, col) coordinate into values.
    indices = Tensor([[1, 0]], dtype=DType.int32, device=DEVICE)
    scatter_set_constant(values, indices, 5.0)
    assert values[1, 0].item() == 5.0
    assert values.real
    scatter_set_constant(values, indices, 4.0)
    assert values[1, 0].item() == 4.0
    assert values.real


def test_inplace_custom_invalid_indices_shape() -> None:
    values = Tensor.zeros([2, 2])
    indices = Tensor.ones([1, 1], dtype=DType.int32)
    with pytest.raises(ValueError, match=r"\[num_indices, 2\]"):
        scatter_set_constant(values, indices, 5.0)


def test_custom_with_custom_extensions(
    kernel_verification_ops_path: Path,
) -> None:
    """Test F.custom with inline custom_extensions loading."""
    x = Tensor.ones([64], dtype=DType.float32, device=CPU())
    y = Tensor.ones([64], dtype=DType.float32, device=CPU())

    # Call custom op with custom_extensions - kernels loaded automatically
    result = F.custom(
        "my_add",
        device=CPU(),
        values=[x, y],
        out_types=[x.type],
        custom_extensions=kernel_verification_ops_path,
    )

    assert len(result) == 1
    output = result[0]
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert output.real


def test_custom_with_custom_extensions_list(
    kernel_verification_ops_path: Path,
) -> None:
    """Test F.custom with custom_extensions as a list."""
    x = Tensor.ones([64], dtype=DType.float32, device=CPU())
    y = Tensor.ones([64], dtype=DType.float32, device=CPU())

    result = F.custom(
        "my_add",
        device=CPU(),
        values=[x, y],
        out_types=[x.type],
        custom_extensions=[kernel_verification_ops_path],
    )

    assert len(result) == 1
    assert result[0].real


def test_custom_with_string_path(kernel_verification_ops_path: Path) -> None:
    """Test F.custom with custom_extensions as a string path."""
    x = Tensor.ones([64], dtype=DType.float32, device=CPU())
    y = Tensor.ones([64], dtype=DType.float32, device=CPU())

    result = F.custom(
        "my_add",
        device=CPU(),
        values=[x, y],
        out_types=[x.type],
        custom_extensions=str(kernel_verification_ops_path),
    )

    assert len(result) == 1
    assert result[0].real


def test_custom_extensions_cached_across_calls(
    kernel_verification_ops_path: Path,
) -> None:
    """Repeated identical custom op calls compile once."""
    # Realized out here so only the custom op's graph reaches the executor
    # under test: the JIT submits every graph it executes, tensor creation
    # included.
    x = Tensor.ones([65], dtype=DType.float32, device=CPU())
    y = Tensor.ones([65], dtype=DType.float32, device=CPU())
    assert x.real and y.real

    executor = JitExecutor(interpreter=InterpreterExecutor())
    with set_default_executor(executor):
        for _ in range(2):
            (result,) = F.custom(
                "my_add",
                device=CPU(),
                values=[x, y],
                out_types=[x.type],
                custom_extensions=kernel_verification_ops_path,
            )
            assert result.real

    assert len(executor.cache) == 1, (
        f"Expected 1 compilation but got {len(executor.cache)} — cache miss"
    )


def test_custom_extensions_cache_miss_on_different_shapes(
    kernel_verification_ops_path: Path,
) -> None:
    """Different input shapes are distinct graphs, so each compiles."""
    inputs = [
        (
            Tensor.ones([size], dtype=DType.float32, device=CPU()),
            Tensor.ones([size], dtype=DType.float32, device=CPU()),
        )
        for size in (65, 66)
    ]
    assert all(x.real and y.real for x, y in inputs)

    executor = JitExecutor(interpreter=InterpreterExecutor())
    with set_default_executor(executor):
        for x, y in inputs:
            (result,) = F.custom(
                "my_add",
                device=CPU(),
                values=[x, y],
                out_types=[x.type],
                custom_extensions=kernel_verification_ops_path,
            )
            assert result.real

    assert len(executor.cache) == 2, (
        "Expected 2 compilations for different shapes but got"
        f" {len(executor.cache)}"
    )


def test_custom_helper_function_pattern(
    kernel_verification_ops_path: Path,
) -> None:
    """Test the recommended pattern for creating reusable custom op wrappers."""

    def my_add(a: Tensor, b: Tensor) -> Tensor:
        """Element-wise addition using custom Mojo kernel."""
        return F.custom(
            "my_add",
            device=a.device,
            values=[a, b],
            out_types=[a.type],
            custom_extensions=kernel_verification_ops_path,
        )[0]

    x = Tensor.ones([64], dtype=DType.float32, device=CPU())
    y = Tensor.full([64], 2.0, dtype=DType.float32, device=CPU())

    result = my_add(x, y)

    assert result.real
    assert result.shape == x.shape
