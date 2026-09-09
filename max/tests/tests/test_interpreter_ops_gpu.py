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
"""GPU tests for MO interpreter operations.

These tests verify that the Mojo op implementations produce correct results
on GPU by comparing against PyTorch reference implementations.
"""

import operator
import os
from collections.abc import Generator, Sequence
from typing import Any

import numpy as np
import pytest
import torch
from max import _interpreter
from max._interpreter_ops import GC_FAMILIES, adopted_from_manifest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.experimental import functional as F
from max.experimental import random as max_random
from max.experimental import realization_context as rc
from max.experimental.executor import (
    InterpreterExecutor,
    set_default_executor,
)
from max.experimental.functional import (
    allgather as all_gather,
)
from max.experimental.functional import (
    allreduce_sum as all_reduce_sum,
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
from max.graph import BufferType as GBufferType
from max.graph import DeviceRef, Graph, TensorType
from max.graph import Type as GType
from max.graph import ops as graph_ops


@pytest.fixture(autouse=True)
def _interpreter_only() -> Generator[None]:
    """All tests in this module run on the interpreter -- it is the unit
    under test.  Graphs it cannot execute raise UnsupportedGraphError."""
    with set_default_executor(InterpreterExecutor()):
        yield


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    torch.manual_seed(42)


@pytest.fixture(scope="session", autouse=True)
def _require_warm_adoption() -> None:
    """Fail loudly unless the build-warmed eager-GC cache actually took effect.

    The numeric tests below pass whether the warm adopts or silently cold-
    compiles, so this asserts adoption directly. ``XARCH_WARM_RLOCATION`` is set
    only on a warmed lane; its absence with an accelerator present means a new
    NVIDIA lane wasn't wired (checked at runtime, not in the BUILD select, since
    the pydeps lint aspect evaluates that select on non-NVIDIA lanes too). When
    wired, ``adopted_from_manifest`` must hold for every family, else
    adoption fell through to a cold recompile.
    """
    if not os.environ.get("XARCH_WARM_RLOCATION"):
        if accelerator_count() > 0:
            raise RuntimeError(
                "test_interpreter_ops_gpu ran on a GPU with no eager-GC warm"
                " wired, add its SKU to _GPU_TARGET/_GPU_LANE_ONLY in"
                " max/tests/integration/xarch_warm/BUILD.bazel and the selects"
                " in max/tests/tests/BUILD.bazel."
            )
        return
    # Derive from the registry, not a hand-maintained tuple, so this adoption
    # gate can't drift from the families the warm actually covers (MXF-533).
    unadopted = [
        f.name for f in GC_FAMILIES if not adopted_from_manifest(f.name)
    ]
    if unadopted:
        raise RuntimeError(
            f"eager-GC warm is wired but {unadopted} did not adopt from the"
            " manifest, a silent cold-compile fallback (arch/key drift)."
        )


# Mapping from MAX DType to torch dtype
DTYPE_TO_TORCH = {
    DType.float32: torch.float32,
    DType.float16: torch.float16,
    DType.bfloat16: torch.bfloat16,
    DType.int8: torch.int8,
    DType.int16: torch.int16,
    DType.int32: torch.int32,
    DType.int64: torch.int64,
    DType.uint8: torch.uint8,
    DType.uint16: torch.uint16,
    DType.uint32: torch.uint32,
    DType.uint64: torch.uint64,
    DType.bool: torch.bool,
}


class TestBasicGPUExecution:
    """Tests for basic GPU execution through the interpreter."""

    def test_add_on_gpu(self) -> None:
        """Test that basic add works on GPU tensors."""
        a_torch = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float32, device="cuda"
        )
        b_torch = torch.tensor(
            [4.0, 5.0, 6.0], dtype=torch.float32, device="cuda"
        )

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a + b

        expected = a_torch + b_torch
        torch.testing.assert_close(torch.from_dlpack(c), expected)

    def test_add_on_gpu_2d(self) -> None:
        """Test 2D add on GPU."""
        a_torch = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device="cuda"
        )
        b_torch = torch.tensor(
            [[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32, device="cuda"
        )

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a + b

        expected = a_torch + b_torch
        torch.testing.assert_close(torch.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", [DType.float32, DType.int32, DType.int64])
    def test_add_on_gpu_dtypes(self, dtype: DType) -> None:
        """Test GPU add with various dtypes."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        a_torch = torch.tensor([1, 2, 3, 4], dtype=torch_dtype, device="cuda")
        b_torch = torch.tensor([5, 6, 7, 8], dtype=torch_dtype, device="cuda")

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a + b

        expected = a_torch + b_torch
        torch.testing.assert_close(torch.from_dlpack(c), expected)


class TestPowGPU:
    """Tests for GPU pow operation."""

    def test_pow_on_gpu(self) -> None:
        """Test that pow works on GPU tensors."""
        a_torch = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda"
        )
        b_torch = torch.tensor(
            [2.0, 3.0, 2.0, 0.5], dtype=torch.float32, device="cuda"
        )

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a**b

        expected = torch.pow(a_torch, b_torch)
        torch.testing.assert_close(torch.from_dlpack(c), expected)

    @pytest.mark.parametrize(
        "dtype", [DType.float32, DType.float16, DType.bfloat16]
    )
    def test_pow_on_gpu_dtypes(self, dtype: DType) -> None:
        """Test GPU pow with various float dtypes."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        a_torch = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch_dtype, device="cuda"
        )
        b_torch = torch.tensor(
            [2.0, 2.0, 2.0, 2.0], dtype=torch_dtype, device="cuda"
        )

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a**b

        expected = torch.pow(a_torch, b_torch)
        torch.testing.assert_close(torch.from_dlpack(c), expected)


class TestBinaryComparisonOpsGPU:
    """Tests for GPU binary comparison operations."""

    @pytest.mark.parametrize(
        "op,torch_func",
        [
            (operator.eq, torch.eq),
            (operator.ne, torch.ne),
            (operator.gt, torch.gt),
            (operator.ge, torch.ge),
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
            DType.int32,
            DType.int64,
        ],
    )
    def test_comparison_ops_gpu(
        self, op: Any, torch_func: Any, dtype: DType
    ) -> None:
        """Test comparison ops on GPU with various dtypes."""
        torch_dtype = DTYPE_TO_TORCH[dtype]

        # Use test data that exercises both equal and unequal cases
        if op in (operator.gt, operator.ge):
            a_torch = torch.tensor(
                [1, 5, 3, 6], dtype=torch_dtype, device="cuda"
            )
            b_torch = torch.tensor(
                [2, 3, 3, 4], dtype=torch_dtype, device="cuda"
            )
        else:
            a_torch = torch.tensor(
                [1, 2, 3, 4], dtype=torch_dtype, device="cuda"
            )
            b_torch = torch.tensor(
                [1, 5, 3, 6], dtype=torch_dtype, device="cuda"
            )

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = op(a, b)

        expected = torch_func(a_torch, b_torch)
        result_torch = torch.from_dlpack(c)
        torch.testing.assert_close(result_torch, expected)


class TestBooleanLogicOpsGPU:
    """Tests for GPU boolean logic operations."""

    @pytest.mark.parametrize(
        "op,torch_func",
        [
            (operator.and_, torch.logical_and),
            (operator.or_, torch.logical_or),
            (operator.xor, torch.logical_xor),
        ],
    )
    def test_binary_logical_ops_gpu(self, op: Any, torch_func: Any) -> None:
        """Test binary logical ops on GPU."""
        a_torch = torch.tensor(
            [True, True, False, False], dtype=torch.bool, device="cuda"
        )
        b_torch = torch.tensor(
            [True, False, True, False], dtype=torch.bool, device="cuda"
        )

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = op(a, b)

        expected = torch_func(a_torch, b_torch)
        result_torch = torch.from_dlpack(c)
        torch.testing.assert_close(result_torch, expected)

    def test_logical_not_gpu(self) -> None:
        """Test logical not op on GPU."""
        a_torch = torch.tensor(
            [True, False, True, False], dtype=torch.bool, device="cuda"
        )

        a = Tensor.from_dlpack(a_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = ~a

        expected = torch.logical_not(a_torch)
        result_torch = torch.from_dlpack(c)
        torch.testing.assert_close(result_torch, expected)


class TestElementwiseGPU:
    """Tests for GPU elementwise operations."""

    def test_mixed_device_inputs_raises_error(self) -> None:
        """Test that mixed CPU/GPU inputs raise an error."""
        a_torch_cpu = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cpu"
        )
        b_torch_gpu = torch.tensor(
            [5.0, 6.0, 7.0, 8.0], dtype=torch.float32, device="cuda"
        )

        a = Tensor.from_dlpack(a_torch_cpu)
        b = Tensor.from_dlpack(b_torch_gpu)

        with pytest.raises(Exception):
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                a + b

    def test_unsupported_op_raises_on_gpu(self) -> None:
        """Test that unsupported GPU ops raise an error."""
        # atanh uses libm and is not supported on GPU
        a_torch = torch.tensor(
            [0.1, 0.2, 0.3, 0.4], dtype=torch.float32, device="cuda"
        )

        a = Tensor.from_dlpack(a_torch)

        with pytest.raises(
            Exception, match="Unsupported unary op/device/dtype"
        ):
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                _ = F.atanh(a)  # atanh is lazy, needs a live reference

    @pytest.mark.parametrize(
        "op,torch_func",
        [
            (operator.add, torch.add),
            (operator.sub, torch.sub),
            (operator.mul, torch.mul),
            (operator.truediv, torch.div),
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
            DType.int32,
            DType.int64,
        ],
    )
    def test_binary_ops_gpu(
        self, op: Any, torch_func: Any, dtype: DType
    ) -> None:
        """Test binary ops on GPU with various dtypes."""
        # Skip div for integer types (different semantics)
        if op == operator.truediv and dtype in (DType.int32, DType.int64):
            pytest.skip("Division not tested for integer types")

        torch_dtype = DTYPE_TO_TORCH[dtype]
        a_torch = torch.tensor([1, 2, 3, 4], dtype=torch_dtype, device="cuda")
        b_torch = torch.tensor([5, 6, 7, 8], dtype=torch_dtype, device="cuda")

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = op(a, b)

        expected = torch_func(a_torch, b_torch)
        result_torch = torch.from_dlpack(c)
        torch.testing.assert_close(result_torch, expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize(
        "op,torch_func",
        [
            (operator.neg, torch.neg),
            (abs, torch.abs),
            (F.exp, torch.exp),
            (F.log, torch.log),
            (F.sqrt, torch.sqrt),
            (F.sin, torch.sin),
            (F.cos, torch.cos),
            (F.tanh, torch.tanh),
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
            DType.int32,
            DType.int64,
        ],
    )
    def test_unary_ops_gpu(
        self, op: Any, torch_func: Any, dtype: DType
    ) -> None:
        """Test unary ops on GPU with various dtypes."""
        # Float-only ops: skip for integer dtypes
        float_only_ops = (F.exp, F.log, F.sqrt, F.sin, F.cos, F.tanh)
        if op in float_only_ops and dtype in (DType.int32, DType.int64):
            pytest.skip("Op not tested for integer types")

        torch_dtype = DTYPE_TO_TORCH[dtype]

        # Use positive values to avoid domain issues with log/sqrt
        # Use values with negatives for negate/abs with integers
        if dtype in (DType.int32, DType.int64):
            a_torch = torch.tensor(
                [-1, 2, -3, 4], dtype=torch_dtype, device="cuda"
            )
        else:
            a_torch = torch.tensor(
                [1.0, 2.0, 3.0, 4.0], dtype=torch_dtype, device="cuda"
            )

        a = Tensor.from_dlpack(a_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = op(a)

        expected = torch_func(a_torch)
        result_torch = torch.from_dlpack(c)
        # Use relaxed tolerance for lower precision dtypes
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)


class TestMatmulGPU:
    """Tests for GPU matmul operations."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_matmul_gpu(self, dtype: DType) -> None:
        """Test matmul on GPU with various dtypes."""
        torch_dtype = DTYPE_TO_TORCH[dtype]

        m, k, n = 3, 4, 5
        lhs_torch = torch.randn(m, k, dtype=torch_dtype, device="cuda")
        rhs_torch = torch.randn(k, n, dtype=torch_dtype, device="cuda")

        lhs = Tensor.from_dlpack(lhs_torch)
        rhs = Tensor.from_dlpack(rhs_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = lhs @ rhs

        expected = torch.matmul(lhs_torch, rhs_torch)
        result_torch = torch.from_dlpack(c)
        # Use relaxed tolerance for lower precision dtypes
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    def test_matmul_gpu_mixed_device_raises_error(self) -> None:
        """Test that mixed CPU/GPU inputs raise an error for matmul."""
        m, k, n = 3, 4, 5
        lhs_torch_cpu = torch.randn(m, k, dtype=torch.float32, device="cpu")
        rhs_torch_gpu = torch.randn(k, n, dtype=torch.float32, device="cuda")

        lhs = Tensor.from_dlpack(lhs_torch_cpu)
        rhs = Tensor.from_dlpack(rhs_torch_gpu)

        with pytest.raises(Exception):
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                lhs @ rhs


class TestBatchMatmulGPU:
    """Tests for GPU batch matmul operations."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_batch_matmul_gpu(self, dtype: DType) -> None:
        """Test batch matmul on GPU with various dtypes."""
        torch_dtype = DTYPE_TO_TORCH[dtype]

        batch, m, k, n = 2, 3, 4, 5
        lhs_torch = torch.randn(batch, m, k, dtype=torch_dtype, device="cuda")
        rhs_torch = torch.randn(batch, k, n, dtype=torch_dtype, device="cuda")

        lhs = Tensor.from_dlpack(lhs_torch)
        rhs = Tensor.from_dlpack(rhs_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = lhs @ rhs

        expected = torch.matmul(lhs_torch, rhs_torch)
        result_torch = torch.from_dlpack(c)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    def test_batch_matmul_gpu_4d(self) -> None:
        """Test 4D batch matmul on GPU."""
        lhs_torch = torch.randn(2, 3, 4, 5, dtype=torch.float32, device="cuda")
        rhs_torch = torch.randn(2, 3, 5, 6, dtype=torch.float32, device="cuda")

        lhs = Tensor.from_dlpack(lhs_torch)
        rhs = Tensor.from_dlpack(rhs_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = lhs @ rhs

        expected = torch.matmul(lhs_torch, rhs_torch)
        result_torch = torch.from_dlpack(c)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    def test_batch_matmul_gpu_mixed_device_raises_error(self) -> None:
        """Test that mixed CPU/GPU inputs raise an error for batch matmul."""
        lhs_torch_cpu = torch.randn(2, 3, 4, dtype=torch.float32, device="cpu")
        rhs_torch_gpu = torch.randn(2, 4, 5, dtype=torch.float32, device="cuda")

        lhs = Tensor.from_dlpack(lhs_torch_cpu)
        rhs = Tensor.from_dlpack(rhs_torch_gpu)

        with pytest.raises(Exception):
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                lhs @ rhs


class TestStaticBroadcastToGPU:
    """Tests for GPU static broadcast_to operations in the interpreter."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
            DType.int32,
            DType.int64,
        ],
    )
    def test_broadcast_1d_to_2d(self, dtype: DType) -> None:
        """Test broadcasting 1D tensor to 2D on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        target_shape = [2, 3]

        x_torch = torch.tensor([1, 2, 3], dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=target_shape)

        result_torch = torch.from_dlpack(y)
        expected = torch.broadcast_to(x_torch, target_shape)
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_broadcast_2d_to_3d(self, dtype: DType) -> None:
        """Test broadcasting 2D tensor with size-1 dim to 3D on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        target_shape = [2, 4, 3]

        x_torch = torch.tensor(
            [[1.0, 2.0, 3.0]], dtype=torch_dtype, device="cuda"
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=target_shape)

        result_torch = torch.from_dlpack(y)
        expected = torch.broadcast_to(x_torch, target_shape)
        torch.testing.assert_close(result_torch, expected)

    def test_broadcast_scalar_like(self) -> None:
        """Test broadcasting scalar-like tensor [1] to higher rank on GPU."""
        target_shape = [2, 3, 4]

        x_torch = torch.tensor([5.0], dtype=torch.float32, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=target_shape)

        result_torch = torch.from_dlpack(y)
        expected = torch.broadcast_to(x_torch, target_shape)
        torch.testing.assert_close(result_torch, expected)

    def test_broadcast_same_shape(self) -> None:
        """Test broadcasting to same shape (no-op) on GPU."""
        shape = [2, 3]

        x_torch = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
            device="cuda",
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=shape)

        result_torch = torch.from_dlpack(y)
        expected = torch.broadcast_to(x_torch, shape)
        torch.testing.assert_close(result_torch, expected)


class TestShapeRearrangeGPU:
    """Minimal GPU numeric coverage for the GPU-capable shape-rearrange ops.

    ``concat``/``split``/``slice``/``pad`` (constant) are ``DeviceClass.ALL``;
    ``tile`` and the reflect/repeat pads are CPU-only (see ``shape_rearrange_gc``),
    so they need no GPU test. Inputs are placed on the accelerator via a CUDA
    torch tensor; references use NumPy, matching the CPU tests.
    """

    def test_concat_gpu(self) -> None:
        """F.concat along axis 0 on GPU."""
        a_torch = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device="cuda"
        )
        b_torch = torch.tensor(
            [[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32, device="cuda"
        )
        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=0)
        expected = np.concatenate(
            [a_torch.cpu().numpy(), b_torch.cpu().numpy()], axis=0
        )
        np.testing.assert_array_equal(
            torch.from_dlpack(result).cpu().numpy(), expected
        )

    def test_slice_gpu(self) -> None:
        """F.slice_tensor across both dims on GPU."""
        x_torch = torch.arange(12, dtype=torch.float32, device="cuda").reshape(
            3, 4
        )
        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.slice_tensor(x, [slice(0, 2), slice(1, 3)])
        expected = x_torch.cpu().numpy()[0:2, 1:3]
        np.testing.assert_array_equal(
            torch.from_dlpack(result).cpu().numpy(), expected
        )

    def test_split_gpu(self) -> None:
        """F.split into size-[1, 3] chunks on GPU."""
        x_torch = torch.arange(12, dtype=torch.float32, device="cuda").reshape(
            3, 4
        )
        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            parts = F.split(x, [1, 3], axis=1)
        # np.split takes cut indices; sizes [1, 3] -> a single cut at index 1.
        expected = np.split(x_torch.cpu().numpy(), [1], axis=1)
        for got, exp in zip(parts, expected, strict=False):
            np.testing.assert_array_equal(
                torch.from_dlpack(got).cpu().numpy(), exp
            )

    def test_pad_constant_gpu(self) -> None:
        """F.pad constant mode on GPU; paddings are flat [pre0, post0, ...]."""
        x_torch = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device="cuda"
        )
        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            out = F.pad(x, [1, 1, 2, 0], value=7.0)
        expected = np.pad(
            x_torch.cpu().numpy(),
            [(1, 1), (2, 0)],
            mode="constant",
            constant_values=7.0,
        )
        np.testing.assert_array_equal(
            torch.from_dlpack(out).cpu().numpy(), expected
        )


class TestRangeGPU:
    """Tests for GPU range operations via Tensor.arange with typed inputs."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_range_basic_gpu(self, dtype: DType) -> None:
        """Test basic range op on GPU with float dtypes."""
        gpu = Accelerator()
        torch_dtype = DTYPE_TO_TORCH[dtype]
        start_t = Tensor.from_dlpack(torch.tensor(0, dtype=torch_dtype))
        stop_t = Tensor.from_dlpack(torch.tensor(10, dtype=torch_dtype))
        step_t = Tensor.from_dlpack(torch.tensor(1, dtype=torch_dtype))

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
                device=gpu,
            )

        result_torch = torch.from_dlpack(t)
        expected = torch.arange(0, 10, 1, dtype=torch_dtype, device="cuda")
        torch.testing.assert_close(result_torch, expected)

    def test_range_with_step_gpu(self) -> None:
        """Test range op with custom step on GPU."""
        gpu = Accelerator()
        start_t = Tensor.from_dlpack(torch.tensor(0, dtype=torch.float32))
        stop_t = Tensor.from_dlpack(torch.tensor(10, dtype=torch.float32))
        step_t = Tensor.from_dlpack(torch.tensor(2, dtype=torch.float32))

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
                device=gpu,
            )

        result_torch = torch.from_dlpack(t)
        expected = torch.arange(0, 10, 2, dtype=torch.float32, device="cuda")
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.int32,
            DType.int64,
        ],
    )
    def test_range_int_gpu(self, dtype: DType) -> None:
        """Test range op with integer dtypes on GPU."""
        gpu = Accelerator()
        torch_dtype = DTYPE_TO_TORCH[dtype]
        start_t = Tensor.from_dlpack(torch.tensor(0, dtype=torch_dtype))
        stop_t = Tensor.from_dlpack(torch.tensor(10, dtype=torch_dtype))
        step_t = Tensor.from_dlpack(torch.tensor(1, dtype=torch_dtype))

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
                device=gpu,
            )

        result_torch = torch.from_dlpack(t)
        expected = torch.arange(0, 10, 1, dtype=torch_dtype, device="cuda")
        torch.testing.assert_close(result_torch, expected)


class TestReduceMaxGPU:
    """Tests for GPU reduce_max operations via Tensor.max with interpreter."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_max_last_axis(self, dtype: DType) -> None:
        """Test reduce_max on the last axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.max(axis=-1)

        result_torch = torch.from_dlpack(y)
        expected = torch.amax(x_torch, dim=-1, keepdim=True)
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_max_first_axis(self, dtype: DType) -> None:
        """Test reduce_max on the first axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.max(axis=0)

        result_torch = torch.from_dlpack(y)
        expected = torch.amax(x_torch, dim=0, keepdim=True)
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_max_middle_axis(self, dtype: DType) -> None:
        """Test reduce_max on a middle axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.max(axis=1)

        result_torch = torch.from_dlpack(y)
        expected = torch.amax(x_torch, dim=1, keepdim=True)
        torch.testing.assert_close(result_torch, expected)

    def test_reduce_max_2d(self) -> None:
        """Test reduce_max on a 2D tensor on GPU."""
        shape = [4, 6]

        x_torch = torch.randn(shape, dtype=torch.float32, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.max(axis=-1)

        result_torch = torch.from_dlpack(y)
        expected = torch.amax(x_torch, dim=-1, keepdim=True)
        torch.testing.assert_close(result_torch, expected)


class TestBroadcastBinaryOpsGPU:
    """Tests for implicit broadcasting in binary ops on GPU.

    These tests exercise the ShapeOfOp -> BroadcastShapeOp -> BroadcastToOp
    chain that gets generated when binary elementwise ops have operands with
    different shapes.
    """

    def test_add_broadcast_1d_2d(self) -> None:
        """Test add with broadcasting: [3] + [2,3] -> [2,3] on GPU."""
        a_torch = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float32, device="cuda"
        )
        b_torch = torch.tensor(
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
            dtype=torch.float32,
            device="cuda",
        )

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a + b

        expected = a_torch + b_torch
        torch.testing.assert_close(torch.from_dlpack(c), expected)

    def test_mul_broadcast_scalar_like(self) -> None:
        """Test mul with broadcasting: [1] * [3,4] -> [3,4] on GPU."""
        a_torch = torch.tensor([2.0], dtype=torch.float32, device="cuda")
        b_torch = torch.randn(3, 4, dtype=torch.float32, device="cuda")

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a * b

        expected = a_torch * b_torch
        torch.testing.assert_close(torch.from_dlpack(c), expected)

    def test_sub_broadcast_different_ranks(self) -> None:
        """Test sub with broadcasting: [4] - [2,3,4] -> [2,3,4] on GPU."""
        a_torch = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda"
        )
        b_torch = torch.randn(2, 3, 4, dtype=torch.float32, device="cuda")

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a - b

        expected = a_torch - b_torch
        torch.testing.assert_close(
            torch.from_dlpack(c), expected, rtol=1e-3, atol=1e-3
        )

    @pytest.mark.parametrize(
        "dtype", [DType.float32, DType.float16, DType.bfloat16]
    )
    def test_add_broadcast_size1_dim(self, dtype: DType) -> None:
        """Test add with broadcasting: [1,4] + [3,4] -> [3,4] on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        a_torch = torch.randn(1, 4, dtype=torch_dtype, device="cuda")
        b_torch = torch.randn(3, 4, dtype=torch_dtype, device="cuda")

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            c = a + b

        expected = a_torch + b_torch
        torch.testing.assert_close(
            torch.from_dlpack(c), expected, rtol=1e-2, atol=1e-2
        )


class TestReduceMinGPU:
    """Tests for GPU reduce_min operations via Tensor.min with interpreter."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_min_last_axis(self, dtype: DType) -> None:
        """Test reduce_min on the last axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.min(axis=-1)

        result_torch = torch.from_dlpack(y)
        expected = torch.amin(x_torch, dim=-1, keepdim=True)
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_min_first_axis(self, dtype: DType) -> None:
        """Test reduce_min on the first axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.min(axis=0)

        result_torch = torch.from_dlpack(y)
        expected = torch.amin(x_torch, dim=0, keepdim=True)
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_min_middle_axis(self, dtype: DType) -> None:
        """Test reduce_min on a middle axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.min(axis=1)

        result_torch = torch.from_dlpack(y)
        expected = torch.amin(x_torch, dim=1, keepdim=True)
        torch.testing.assert_close(result_torch, expected)

    def test_reduce_min_2d(self) -> None:
        """Test reduce_min on a 2D tensor on GPU."""
        shape = [4, 6]

        x_torch = torch.randn(shape, dtype=torch.float32, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.min(axis=-1)

        result_torch = torch.from_dlpack(y)
        expected = torch.amin(x_torch, dim=-1, keepdim=True)
        torch.testing.assert_close(result_torch, expected)


class TestReduceSumGPU:
    """Tests for GPU reduce_sum operations via Tensor.sum with interpreter."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_sum_last_axis(self, dtype: DType) -> None:
        """Test reduce_sum on the last axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.sum(axis=-1)

        result_torch = torch.from_dlpack(y)
        expected = torch.sum(x_torch, dim=-1, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_sum_first_axis(self, dtype: DType) -> None:
        """Test reduce_sum on the first axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.sum(axis=0)

        result_torch = torch.from_dlpack(y)
        expected = torch.sum(x_torch, dim=0, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_sum_middle_axis(self, dtype: DType) -> None:
        """Test reduce_sum on a middle axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.sum(axis=1)

        result_torch = torch.from_dlpack(y)
        expected = torch.sum(x_torch, dim=1, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    def test_reduce_sum_2d(self) -> None:
        """Test reduce_sum on a 2D tensor on GPU."""
        shape = [4, 6]

        x_torch = torch.randn(shape, dtype=torch.float32, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.sum(axis=-1)

        result_torch = torch.from_dlpack(y)
        expected = torch.sum(x_torch, dim=-1, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)


class TestMeanGPU:
    """Tests for GPU mean operations via Tensor.mean with interpreter."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_mean_last_axis(self, dtype: DType) -> None:
        """Test mean on the last axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.mean(axis=-1)

        result_torch = torch.from_dlpack(y)
        expected = torch.mean(x_torch, dim=-1, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_mean_first_axis(self, dtype: DType) -> None:
        """Test mean on the first axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.mean(axis=0)

        result_torch = torch.from_dlpack(y)
        expected = torch.mean(x_torch, dim=0, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_mean_middle_axis(self, dtype: DType) -> None:
        """Test mean on a middle axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.mean(axis=1)

        result_torch = torch.from_dlpack(y)
        expected = torch.mean(x_torch, dim=1, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    def test_mean_2d(self) -> None:
        """Test mean on a 2D tensor on GPU."""
        shape = [4, 6]

        x_torch = torch.randn(shape, dtype=torch.float32, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.mean(axis=-1)

        result_torch = torch.from_dlpack(y)
        expected = torch.mean(x_torch, dim=-1, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)


class TestReduceMulGPU:
    """Tests for GPU reduce_mul operations via Tensor.prod with interpreter."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_mul_last_axis(self, dtype: DType) -> None:
        """Test reduce_mul on the last axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        # Use values close to 1 to avoid overflow
        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda") * 0.3 + 1
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.prod(axis=-1)

        result_torch = torch.from_dlpack(y)
        expected = torch.prod(x_torch, dim=-1, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_mul_first_axis(self, dtype: DType) -> None:
        """Test reduce_mul on the first axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda") * 0.3 + 1
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.prod(axis=0)

        result_torch = torch.from_dlpack(y)
        expected = torch.prod(x_torch, dim=0, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_mul_middle_axis(self, dtype: DType) -> None:
        """Test reduce_mul on a middle axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda") * 0.3 + 1
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.prod(axis=1)

        result_torch = torch.from_dlpack(y)
        expected = torch.prod(x_torch, dim=1, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    def test_reduce_mul_2d(self) -> None:
        """Test reduce_mul on a 2D tensor on GPU."""
        shape = [4, 6]

        x_torch = (
            torch.randn(shape, dtype=torch.float32, device="cuda") * 0.3 + 1
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.prod(axis=-1)

        result_torch = torch.from_dlpack(y)
        expected = torch.prod(x_torch, dim=-1, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)


class TestUnaryMixedOpsGPU:
    """Tests for GPU unary mixed-dtype ops (cast, is_nan, is_inf)."""

    @pytest.mark.parametrize(
        "in_dtype,out_dtype",
        [
            (DType.float32, DType.int32),
            (DType.float32, DType.float16),
            (DType.float16, DType.float32),
            (DType.float32, DType.bfloat16),
            (DType.bfloat16, DType.float32),
            (DType.int32, DType.float32),
            (DType.int64, DType.float32),
            (DType.float32, DType.int64),
            (DType.int32, DType.bool),
            (DType.bool, DType.int32),
        ],
    )
    def test_cast(self, in_dtype: DType, out_dtype: DType) -> None:
        """Test cast op on GPU converts dtype correctly."""
        in_torch_dtype = DTYPE_TO_TORCH[in_dtype]
        out_torch_dtype = DTYPE_TO_TORCH[out_dtype]

        if in_dtype in (DType.int32, DType.int64):
            x_torch = torch.arange(12, dtype=in_torch_dtype, device="cuda")
        else:
            x_torch = torch.tensor(
                [0.0, 1.0, 2.5, -1.5, 3.0, -3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                dtype=in_torch_dtype,
                device="cuda",
            )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.cast(out_dtype)

        result_torch = torch.from_dlpack(y)
        expected = x_torch.to(out_torch_dtype)
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16],
    )
    def test_is_nan(self, dtype: DType) -> None:
        """Test is_nan op on GPU detects NaN values."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.tensor(
            [1.0, float("nan"), 3.0, float("nan"), float("inf"), 0.0],
            dtype=torch_dtype,
            device="cuda",
        )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.is_nan(x)

        result_torch = torch.from_dlpack(y)
        expected = torch.isnan(x_torch)
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16],
    )
    def test_is_inf(self, dtype: DType) -> None:
        """Test is_inf op on GPU detects Inf values."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.tensor(
            [1.0, float("inf"), float("-inf"), float("nan"), 0.0, 42.0],
            dtype=torch_dtype,
            device="cuda",
        )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.is_inf(x)

        result_torch = torch.from_dlpack(y)
        expected = torch.isinf(x_torch)
        torch.testing.assert_close(result_torch, expected)

    def test_cast_identity(self) -> None:
        """Test cast to same dtype on GPU is identity."""
        x_torch = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda"
        )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.cast(DType.float32)

        result_torch = torch.from_dlpack(y)
        torch.testing.assert_close(result_torch, x_torch)

    def test_cast_float_to_int_truncation(self) -> None:
        """Test cast from float to int on GPU truncates toward zero."""
        x_torch = torch.tensor(
            [1.7, -2.3, 3.9, -4.1], dtype=torch.float32, device="cuda"
        )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.cast(DType.int32)

        result_torch = torch.from_dlpack(y)
        expected = x_torch.to(torch.int32)
        torch.testing.assert_close(result_torch, expected)


class TestRandomNormalGPU:
    """Tests for random normal op on GPU with interpreter."""

    def test_random_normal_gpu_shape_and_device(self) -> None:
        """Test random normal on GPU produces correct shape and device."""

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result = max_random.gaussian(
                (3, 4), dtype=DType.float32, device=Accelerator()
            )

        result_torch = torch.from_dlpack(result)
        assert result_torch.shape == (3, 4)
        assert result_torch.dtype == torch.float32
        assert result_torch.is_cuda

    def test_random_normal_gpu_statistics(self) -> None:
        """Test random normal on GPU has approximately correct statistics."""

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            set_seed(123)
            result = max_random.gaussian(
                (1000, 1000),
                mean=3.0,
                std=1.5,
                dtype=DType.float32,
                device=Accelerator(),
            )

        result_torch = torch.from_dlpack(result).float()
        torch.testing.assert_close(
            result_torch.mean(),
            torch.tensor(3.0, device="cuda"),
            atol=0.1,
            rtol=0.1,
        )
        torch.testing.assert_close(
            result_torch.std(),
            torch.tensor(1.5, device="cuda"),
            atol=0.1,
            rtol=0.1,
        )

    def test_random_normal_gpu_deterministic(self) -> None:
        """Test that same seed produces identical results on GPU."""

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result1 = max_random.gaussian(
                (5, 5), dtype=DType.float32, device=Accelerator()
            )

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result2 = max_random.gaussian(
                (5, 5), dtype=DType.float32, device=Accelerator()
            )

        torch.testing.assert_close(
            torch.from_dlpack(result1), torch.from_dlpack(result2)
        )


class TestRandomUniformGPU:
    """Tests for random uniform op on GPU with interpreter."""

    def test_random_uniform_gpu_shape_and_device(self) -> None:
        """Test random uniform on GPU produces correct shape and device."""

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result = max_random.uniform(
                (3, 4), dtype=DType.float32, device=Accelerator()
            )

        result_torch = torch.from_dlpack(result)
        assert result_torch.shape == (3, 4)
        assert result_torch.dtype == torch.float32
        assert result_torch.is_cuda

    def test_random_uniform_gpu_statistics(self) -> None:
        """Test random uniform on GPU has approximately correct statistics."""

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            set_seed(123)
            result = max_random.uniform(
                (1000, 1000),
                range=(2.0, 5.0),
                dtype=DType.float32,
                device=Accelerator(),
            )

        result_torch = torch.from_dlpack(result).float()
        torch.testing.assert_close(
            result_torch.mean(),
            torch.tensor(3.5, device="cuda"),
            atol=0.1,
            rtol=0.1,
        )
        assert result_torch.min().item() >= 2.0
        assert result_torch.max().item() <= 5.0

    def test_random_uniform_gpu_deterministic(self) -> None:
        """Test that same seed produces identical results on GPU."""

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result1 = max_random.uniform(
                (5, 5), dtype=DType.float32, device=Accelerator()
            )

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result2 = max_random.uniform(
                (5, 5), dtype=DType.float32, device=Accelerator()
            )

        torch.testing.assert_close(
            torch.from_dlpack(result1), torch.from_dlpack(result2)
        )


class TestTransferOpsGPU:
    """Tests for GPU transfer operations via Tensor.to() with interpreter."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
            DType.int32,
            DType.int64,
        ],
    )
    def test_cpu_to_gpu(self, dtype: DType) -> None:
        """Test transferring a tensor from CPU to GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        gpu = Accelerator()

        x_torch_cpu = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch_dtype, device="cpu"
        )
        x = Tensor.from_dlpack(x_torch_cpu)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.to(gpu)

        result_torch = torch.from_dlpack(y)
        expected = x_torch_cpu.to("cuda")
        assert torch.equal(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
            DType.int32,
            DType.int64,
        ],
    )
    def test_gpu_to_cpu(self, dtype: DType) -> None:
        """Test transferring a tensor from GPU to CPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]

        x_torch_gpu = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch_dtype, device="cuda"
        )
        x = Tensor.from_dlpack(x_torch_gpu)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.to(CPU())

        result_torch = torch.from_dlpack(y)
        expected = x_torch_gpu.to("cpu")
        assert torch.equal(result_torch, expected)

    def test_cpu_to_gpu_2d(self) -> None:
        """Test transferring a 2D tensor from CPU to GPU."""
        gpu = Accelerator()

        x_torch_cpu = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
            device="cpu",
        )
        x = Tensor.from_dlpack(x_torch_cpu)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.to(gpu)

        result_torch = torch.from_dlpack(y)
        expected = x_torch_cpu.to("cuda")
        assert torch.equal(result_torch, expected)

    def test_gpu_to_cpu_2d(self) -> None:
        """Test transferring a 2D tensor from GPU to CPU."""
        x_torch_gpu = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
            device="cuda",
        )
        x = Tensor.from_dlpack(x_torch_gpu)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.to(CPU())

        result_torch = torch.from_dlpack(y)
        expected = x_torch_gpu.to("cpu")
        assert torch.equal(result_torch, expected)

    def test_cpu_to_gpu_large_tensor(self) -> None:
        """Test transferring a larger tensor from CPU to GPU."""
        gpu = Accelerator()

        x_torch_cpu = torch.randn(64, 128, dtype=torch.float32, device="cpu")
        x = Tensor.from_dlpack(x_torch_cpu)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.to(gpu)

        result_torch = torch.from_dlpack(y)
        expected = x_torch_cpu.to("cuda")
        assert torch.equal(result_torch, expected)

    def test_transfer_preserves_data_roundtrip(self) -> None:
        """Test CPU -> GPU -> CPU preserves data exactly."""
        gpu = Accelerator()

        x_torch = torch.tensor(
            [1.0, -2.5, 3.14, 0.0, -100.0],
            dtype=torch.float32,
            device="cpu",
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y_gpu = x.to(gpu)
            y_cpu = y_gpu.to(CPU())

        result_torch = torch.from_dlpack(y_cpu)
        assert torch.equal(result_torch, x_torch)

    def test_same_device_elide_true_aliases(self) -> None:
        """Test same-device transfer with alwaysElideSameDeviceCopy=True aliases."""
        from max._core.dialects import m, mo

        gpu_ref = DeviceRef.GPU(0)
        input_type = TensorType(DType.float32, [4], gpu_ref)

        with Graph("same_device_transfer", input_types=[input_type]) as g:
            x = g.inputs[0]
            in_chain = g.always_ready_chain
            (result, _) = g._add_op_generated(
                mo.TransferOp,
                x,
                m.DeviceRefAttr("gpu", 0),
                in_chain,
                always_elide_same_device_copy=True,
            )
            g.output(result)

        x_torch = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda"
        )
        input_buf = Buffer.from_dlpack(x_torch)

        outputs = _interpreter.execute(g, [input_buf])

        out_buf = outputs[0]
        assert isinstance(out_buf, Buffer)
        result_torch = torch.from_dlpack(out_buf)
        assert torch.equal(result_torch, x_torch)
        # alwaysElideSameDeviceCopy=True should alias, not copy.
        assert out_buf._data_ptr() == input_buf._data_ptr()

    def test_same_device_elide_false_copies(self) -> None:
        """Test same-device transfer with alwaysElideSameDeviceCopy=False copies."""
        from max._core.dialects import m, mo

        gpu_ref = DeviceRef.GPU(0)
        input_type = TensorType(DType.float32, [4], gpu_ref)

        with Graph("same_device_elide_false", input_types=[input_type]) as g:
            x = g.inputs[0]
            in_chain = g.always_ready_chain
            (result, _) = g._add_op_generated(
                mo.TransferOp,
                x,
                m.DeviceRefAttr("gpu", 0),
                in_chain,
                always_elide_same_device_copy=False,
            )
            g.output(result)

        x_torch = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda"
        )
        input_buf = Buffer.from_dlpack(x_torch)

        outputs = _interpreter.execute(g, [input_buf])

        out_buf = outputs[0]
        assert isinstance(out_buf, Buffer)
        result_torch = torch.from_dlpack(out_buf)
        assert torch.equal(result_torch, x_torch)
        # alwaysElideSameDeviceCopy=False should produce a copy, not alias.
        assert out_buf._data_ptr() != input_buf._data_ptr()


class TestReshapeOpsGPU:
    """Tests for reshape operations (squeeze, unsqueeze, reshape) on GPU."""

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16, DType.int32],
    )
    def test_squeeze_on_gpu(self, dtype: DType) -> None:
        """Test squeeze removes a size-1 dimension on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.arange(12, dtype=torch_dtype, device="cuda").reshape(
            3, 1, 4
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.squeeze(axis=1)

        result_torch = torch.from_dlpack(y)
        expected = x_torch.squeeze(1)
        assert result_torch.shape == expected.shape
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16, DType.int32],
    )
    def test_unsqueeze_on_gpu(self, dtype: DType) -> None:
        """Test unsqueeze adds a dimension on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.arange(12, dtype=torch_dtype, device="cuda").reshape(
            3, 4
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.unsqueeze(axis=1)

        result_torch = torch.from_dlpack(y)
        expected = x_torch.unsqueeze(1)
        assert result_torch.shape == expected.shape
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16],
    )
    def test_reshape_split_dim_on_gpu(self, dtype: DType) -> None:
        """Test reshape splitting a dimension on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.arange(36, dtype=torch_dtype, device="cuda").reshape(
            12, 3
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.reshape([3, 4, 3])

        result_torch = torch.from_dlpack(y)
        expected = x_torch.reshape(3, 4, 3)
        assert result_torch.shape == expected.shape
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16],
    )
    def test_reshape_merge_dims_on_gpu(self, dtype: DType) -> None:
        """Test reshape merging adjacent dimensions on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.arange(24, dtype=torch_dtype, device="cuda").reshape(
            2, 3, 4
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.reshape([6, 4])

        result_torch = torch.from_dlpack(y)
        expected = x_torch.reshape(6, 4)
        assert result_torch.shape == expected.shape
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16],
    )
    def test_reshape_add_singleton_on_gpu(self, dtype: DType) -> None:
        """Test reshape adding a singleton dimension on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.arange(12, dtype=torch_dtype, device="cuda").reshape(
            3, 4
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.reshape([3, 1, 4])

        result_torch = torch.from_dlpack(y)
        expected = x_torch.reshape(3, 1, 4)
        assert result_torch.shape == expected.shape
        torch.testing.assert_close(result_torch, expected)

    def test_squeeze_then_unsqueeze_roundtrip_on_gpu(self) -> None:
        """Test squeeze then unsqueeze roundtrip on GPU."""
        x_torch = torch.arange(12, dtype=torch.float32, device="cuda").reshape(
            3, 1, 4
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            squeezed = x.squeeze(axis=1)
            unsqueezed = squeezed.unsqueeze(axis=1)

        result_torch = torch.from_dlpack(unsqueezed)
        assert result_torch.shape == x_torch.shape
        torch.testing.assert_close(result_torch, x_torch)


class TestSoftmaxGPU:
    """Tests for softmax and logsoftmax on GPU."""

    @pytest.mark.parametrize(
        "dtype", [DType.float32, DType.float16, DType.bfloat16]
    )
    def test_softmax_gpu(self, dtype: DType) -> None:
        """Test softmax on GPU matches torch reference."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.randn(3, 4, 5, dtype=torch_dtype, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.softmax(x, axis=-1)

        expected = torch.softmax(x_torch, dim=-1)
        tol = 1e-2 if dtype == DType.bfloat16 else 1e-3
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, atol=tol, rtol=tol
        )

    @pytest.mark.parametrize(
        "dtype", [DType.float32, DType.float16, DType.bfloat16]
    )
    def test_logsoftmax_gpu(self, dtype: DType) -> None:
        """Test logsoftmax on GPU matches torch reference."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.randn(3, 4, 5, dtype=torch_dtype, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.logsoftmax(x, axis=-1)

        expected = torch.log_softmax(x_torch, dim=-1)
        tol = 1e-2 if dtype == DType.bfloat16 else 1e-3
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, atol=tol, rtol=tol
        )


class TestSelectGPU:
    """Tests for GPU select (where) operations with interpreter."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_select_basic_gpu(self, dtype: DType) -> None:
        """Test basic select op on GPU with float dtypes."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        cond_torch = torch.tensor(
            [True, False, True, False, True, False],
            dtype=torch.bool,
            device="cuda",
        )
        x_torch = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            dtype=torch_dtype,
            device="cuda",
        )
        y_torch = torch.tensor(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            dtype=torch_dtype,
            device="cuda",
        )

        cond = Tensor.from_dlpack(cond_torch)
        x = Tensor.from_dlpack(x_torch)
        y = Tensor.from_dlpack(y_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.where(cond, x, y)

        expected = torch.where(cond_torch, x_torch, y_torch)
        torch.testing.assert_close(torch.from_dlpack(result), expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.int8,
            DType.int16,
            DType.int32,
            DType.int64,
        ],
    )
    def test_select_int_gpu(self, dtype: DType) -> None:
        """Test select op with integer dtypes on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        cond_torch = torch.tensor(
            [True, False, True, False],
            dtype=torch.bool,
            device="cuda",
        )
        x_torch = torch.tensor([1, 2, 3, 4], dtype=torch_dtype, device="cuda")
        y_torch = torch.tensor(
            [10, 20, 30, 40], dtype=torch_dtype, device="cuda"
        )

        cond = Tensor.from_dlpack(cond_torch)
        x = Tensor.from_dlpack(x_torch)
        y = Tensor.from_dlpack(y_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.where(cond, x, y)

        expected = torch.where(cond_torch, x_torch, y_torch)
        torch.testing.assert_close(torch.from_dlpack(result), expected)

    def test_select_2d_gpu(self) -> None:
        """Test select with 2D tensors on GPU."""
        cond_torch = torch.tensor(
            [[True, False, True], [False, True, False]],
            dtype=torch.bool,
            device="cuda",
        )
        x_torch = torch.arange(
            1, 7, dtype=torch.float32, device="cuda"
        ).reshape(2, 3)
        y_torch = torch.arange(
            10, 70, 10, dtype=torch.float32, device="cuda"
        ).reshape(2, 3)

        cond = Tensor.from_dlpack(cond_torch)
        x = Tensor.from_dlpack(x_torch)
        y = Tensor.from_dlpack(y_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.where(cond, x, y)

        expected = torch.where(cond_torch, x_torch, y_torch)
        torch.testing.assert_close(torch.from_dlpack(result), expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.uint8,
            DType.uint16,
            DType.uint32,
            DType.uint64,
        ],
    )
    def test_select_uint_gpu(self, dtype: DType) -> None:
        """Test select op with unsigned integer dtypes on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        cond_torch = torch.tensor(
            [True, False, True, False],
            dtype=torch.bool,
            device="cuda",
        )
        x_torch = torch.tensor([1, 2, 3, 4], dtype=torch_dtype, device="cuda")
        y_torch = torch.tensor(
            [10, 20, 30, 40], dtype=torch_dtype, device="cuda"
        )

        cond = Tensor.from_dlpack(cond_torch)
        x = Tensor.from_dlpack(x_torch)
        y = Tensor.from_dlpack(y_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.where(cond, x, y)

        # torch.where has no uint16/32/64 kernel (only uint8), so compute the
        # oracle in int64 and cast back; F.where above already ran natively.
        expected = torch.where(
            cond_torch, x_torch.to(torch.int64), y_torch.to(torch.int64)
        ).to(torch_dtype)
        torch.testing.assert_close(torch.from_dlpack(result), expected)

    def test_select_bool_values_gpu(self) -> None:
        """Test select op where x/y (not just cond) are bool tensors on GPU."""
        cond_torch = torch.tensor(
            [True, False, True, False], dtype=torch.bool, device="cuda"
        )
        x_torch = torch.tensor(
            [True, True, False, False], dtype=torch.bool, device="cuda"
        )
        y_torch = torch.tensor(
            [False, False, True, True], dtype=torch.bool, device="cuda"
        )

        cond = Tensor.from_dlpack(cond_torch)
        x = Tensor.from_dlpack(x_torch)
        y = Tensor.from_dlpack(y_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.where(cond, x, y)

        expected = torch.where(cond_torch, x_torch, y_torch)
        torch.testing.assert_close(torch.from_dlpack(result), expected)


class TestConcatGPU:
    """Tests for GPU concat operations with interpreter."""

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16],
    )
    def test_concat_axis0_gpu(self, dtype: DType) -> None:
        """Test concat along axis 0 on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        a_torch = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch_dtype, device="cuda"
        )
        b_torch = torch.tensor(
            [[5.0, 6.0], [7.0, 8.0]], dtype=torch_dtype, device="cuda"
        )

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=0)

        expected = torch.cat([a_torch, b_torch], dim=0)
        torch.testing.assert_close(torch.from_dlpack(result), expected)

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16],
    )
    def test_concat_axis1_gpu(self, dtype: DType) -> None:
        """Test concat along axis 1 on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        a_torch = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch_dtype, device="cuda"
        )
        b_torch = torch.tensor(
            [[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]],
            dtype=torch_dtype,
            device="cuda",
        )

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=1)

        expected = torch.cat([a_torch, b_torch], dim=1)
        torch.testing.assert_close(torch.from_dlpack(result), expected)

    def test_concat_negative_axis_gpu(self) -> None:
        """Test concat with negative axis on GPU."""
        a_torch = torch.arange(6, dtype=torch.float32, device="cuda").reshape(
            2, 3
        )
        b_torch = torch.arange(
            6, 10, dtype=torch.float32, device="cuda"
        ).reshape(2, 2)

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=-1)

        expected = torch.cat([a_torch, b_torch], dim=-1)
        torch.testing.assert_close(torch.from_dlpack(result), expected)

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16],
    )
    def test_concat_multiple_tensors_gpu(self, dtype: DType) -> None:
        """Test concat with more than two tensors on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        a_torch = torch.tensor([[1.0, 2.0]], dtype=torch_dtype, device="cuda")
        b_torch = torch.tensor([[3.0, 4.0]], dtype=torch_dtype, device="cuda")
        c_torch = torch.tensor([[5.0, 6.0]], dtype=torch_dtype, device="cuda")

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        c = Tensor.from_dlpack(c_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b, c], axis=0)

        expected = torch.cat([a_torch, b_torch, c_torch], dim=0)
        torch.testing.assert_close(torch.from_dlpack(result), expected)

    def test_concat_3d_gpu(self) -> None:
        """Test concat with 3D tensors on GPU."""
        a_torch = torch.arange(24, dtype=torch.float32, device="cuda").reshape(
            2, 3, 4
        )
        b_torch = torch.arange(
            24, 48, dtype=torch.float32, device="cuda"
        ).reshape(2, 3, 4)

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=0)

        expected = torch.cat([a_torch, b_torch], dim=0)
        torch.testing.assert_close(torch.from_dlpack(result), expected)

    @pytest.mark.parametrize("dtype", [DType.int32, DType.int64])
    def test_concat_int_dtypes_gpu(self, dtype: DType) -> None:
        """Test concat with integer dtypes on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        a_torch = torch.tensor([1, 2, 3], dtype=torch_dtype, device="cuda")
        b_torch = torch.tensor([4, 5, 6], dtype=torch_dtype, device="cuda")

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=0)

        expected = torch.cat([a_torch, b_torch], dim=0)
        torch.testing.assert_close(torch.from_dlpack(result), expected)


class TestLayerNormGPU:
    """Tests for layer_norm on GPU."""

    @pytest.mark.parametrize(
        "dtype", [DType.float32, DType.float16, DType.bfloat16]
    )
    def test_layer_norm_gpu(self, dtype: DType) -> None:
        """Test layer_norm on GPU matches torch reference."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]
        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        gamma_torch = torch.randn(shape[-1], dtype=torch_dtype, device="cuda")
        beta_torch = torch.randn(shape[-1], dtype=torch_dtype, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        gamma = Tensor.from_dlpack(gamma_torch)
        beta = Tensor.from_dlpack(beta_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.layer_norm(x, gamma, beta, epsilon=1e-5)

        expected = torch.nn.functional.layer_norm(
            x_torch, [shape[-1]], weight=gamma_torch, bias=beta_torch, eps=1e-5
        )
        tol = 1e-2 if dtype == DType.bfloat16 else 1e-3
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, atol=tol, rtol=tol
        )

    @pytest.mark.parametrize(
        "dtype", [DType.float32, DType.float16, DType.bfloat16]
    )
    def test_layer_norm_gpu_2d(self, dtype: DType) -> None:
        """Test layer_norm on a 2D tensor on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [4, 8]
        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        gamma_torch = torch.randn(shape[-1], dtype=torch_dtype, device="cuda")
        beta_torch = torch.randn(shape[-1], dtype=torch_dtype, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        gamma = Tensor.from_dlpack(gamma_torch)
        beta = Tensor.from_dlpack(beta_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.layer_norm(x, gamma, beta, epsilon=1e-5)

        expected = torch.nn.functional.layer_norm(
            x_torch, [shape[-1]], weight=gamma_torch, bias=beta_torch, eps=1e-5
        )
        tol = 1e-2 if dtype == DType.bfloat16 else 1e-3
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, atol=tol, rtol=tol
        )

    def test_layer_norm_gpu_large(self) -> None:
        """Test layer_norm with a large feature dimension on GPU."""
        shape = [8, 128]
        x_torch = torch.randn(shape, dtype=torch.float32, device="cuda")
        gamma_torch = torch.randn(shape[-1], dtype=torch.float32, device="cuda")
        beta_torch = torch.randn(shape[-1], dtype=torch.float32, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        gamma = Tensor.from_dlpack(gamma_torch)
        beta = Tensor.from_dlpack(beta_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.layer_norm(x, gamma, beta, epsilon=1e-5)

        expected = torch.nn.functional.layer_norm(
            x_torch,
            [shape[-1]],
            weight=gamma_torch,
            bias=beta_torch,
            eps=1e-5,
        )
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, atol=1e-5, rtol=1e-5
        )


class TestRmsNormGPU:
    """Tests for rms_norm on GPU."""

    @pytest.mark.parametrize(
        "dtype", [DType.float32, DType.float16, DType.bfloat16]
    )
    def test_rms_norm_gpu(self, dtype: DType) -> None:
        """Test rms_norm (Llama-style) on GPU matches torch reference."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]
        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        weight_torch = torch.randn(shape[-1], dtype=torch_dtype, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        weight = Tensor.from_dlpack(weight_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.rms_norm(x, weight, epsilon=1e-5)

        variance = x_torch.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normed = x_torch.to(torch.float32) * torch.rsqrt(variance + 1e-5)
        expected = (weight_torch.to(torch.float32) * normed).to(torch_dtype)
        tol = 1e-2 if dtype == DType.bfloat16 else 1e-3
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, atol=tol, rtol=tol
        )

    def test_rms_norm_gpu_multiply_before_cast(self) -> None:
        """Test rms_norm (Gemma-style multiply_before_cast=True) on GPU."""
        shape = [4, 6]
        x_torch = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        weight_torch = torch.randn(
            shape[-1], dtype=torch.bfloat16, device="cuda"
        )

        x = Tensor.from_dlpack(x_torch)
        weight = Tensor.from_dlpack(weight_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.rms_norm(
                x,
                weight,
                epsilon=1e-6,
                weight_offset=1.0,
                multiply_before_cast=True,
            )

        x_f32 = x_torch.to(torch.float32)
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        normed = x_f32 * torch.rsqrt(variance + 1e-6)
        w_f32 = weight_torch.to(torch.float32) + 1.0
        expected = (normed * w_f32).to(torch.bfloat16)
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, atol=1e-2, rtol=1e-2
        )


class TestGroupNormGPU:
    """Tests for group_norm on GPU."""

    @staticmethod
    def _group_norm_ref(
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        num_groups: int,
        eps: float,
    ) -> torch.Tensor:
        """Torch reference via ``torch.nn.functional.group_norm``."""
        return torch.nn.functional.group_norm(
            x, num_groups, weight=gamma, bias=beta, eps=eps
        )

    @pytest.mark.parametrize(
        "dtype", [DType.float32, DType.float16, DType.bfloat16]
    )
    def test_group_norm_gpu_4d(self, dtype: DType) -> None:
        """Test group_norm on a 4D NCHW tensor on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.randn(2, 4, 3, 3, dtype=torch_dtype, device="cuda")
        gamma_torch = torch.randn(4, dtype=torch_dtype, device="cuda")
        beta_torch = torch.randn(4, dtype=torch_dtype, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        gamma = Tensor.from_dlpack(gamma_torch)
        beta = Tensor.from_dlpack(beta_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.group_norm(x, gamma, beta, num_groups=2, epsilon=1e-5)

        expected = self._group_norm_ref(
            x_torch, gamma_torch, beta_torch, 2, 1e-5
        )
        tol = 1e-2 if dtype == DType.bfloat16 else 1e-3
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, atol=tol, rtol=tol
        )

    def test_group_norm_gpu_3d_input(self) -> None:
        """Test group_norm on a 3D [N, C, L] tensor on GPU."""
        x_torch = torch.randn(2, 6, 8, dtype=torch.float32, device="cuda")
        gamma_torch = torch.randn(6, dtype=torch.float32, device="cuda")
        beta_torch = torch.randn(6, dtype=torch.float32, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        gamma = Tensor.from_dlpack(gamma_torch)
        beta = Tensor.from_dlpack(beta_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.group_norm(x, gamma, beta, num_groups=3, epsilon=1e-5)

        expected = self._group_norm_ref(
            x_torch, gamma_torch, beta_torch, 3, 1e-5
        )
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, atol=1e-4, rtol=1e-4
        )

    def test_group_norm_gpu_single_group(self) -> None:
        """Test group_norm with num_groups=1 on GPU."""
        x_torch = torch.randn(2, 4, 3, 3, dtype=torch.float32, device="cuda")
        gamma_torch = torch.randn(4, dtype=torch.float32, device="cuda")
        beta_torch = torch.randn(4, dtype=torch.float32, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        gamma = Tensor.from_dlpack(gamma_torch)
        beta = Tensor.from_dlpack(beta_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.group_norm(x, gamma, beta, num_groups=1, epsilon=1e-5)

        expected = self._group_norm_ref(
            x_torch, gamma_torch, beta_torch, 1, 1e-5
        )
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, atol=1e-4, rtol=1e-4
        )


class TestTransposeGPU:
    """Tests for transpose operations on GPU."""

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16, DType.int32],
    )
    def test_transpose_2d_gpu(self, dtype: DType) -> None:
        """Test basic 2D transpose on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.arange(6, dtype=torch_dtype, device="cuda").reshape(
            2, 3
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.transpose(0, 1)

        result_torch = torch.from_dlpack(y)
        expected = x_torch.transpose(0, 1).contiguous()
        assert result_torch.shape == expected.shape
        torch.testing.assert_close(result_torch, expected)

    def test_transpose_3d_swap_first_last_gpu(self) -> None:
        """Test 3D transpose swapping first and last dims on GPU."""
        x_torch = torch.arange(24, dtype=torch.float32, device="cuda").reshape(
            2, 3, 4
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.transpose(0, 2)

        result_torch = torch.from_dlpack(y)
        expected = x_torch.transpose(0, 2).contiguous()
        assert result_torch.shape == expected.shape
        torch.testing.assert_close(result_torch, expected)

    def test_transpose_3d_swap_adjacent_gpu(self) -> None:
        """Test 3D transpose swapping adjacent dims on GPU."""
        x_torch = torch.arange(24, dtype=torch.float32, device="cuda").reshape(
            2, 3, 4
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.transpose(1, 2)

        result_torch = torch.from_dlpack(y)
        expected = x_torch.transpose(1, 2).contiguous()
        assert result_torch.shape == expected.shape
        torch.testing.assert_close(result_torch, expected)

    def test_transpose_4d_gpu(self) -> None:
        """Test 4D transpose on GPU (e.g., NCHW to NHWC-like)."""
        x_torch = torch.arange(120, dtype=torch.float32, device="cuda").reshape(
            2, 3, 4, 5
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = x.transpose(1, 3)

        result_torch = torch.from_dlpack(y)
        expected = x_torch.transpose(1, 3).contiguous()
        assert result_torch.shape == expected.shape
        torch.testing.assert_close(result_torch, expected)


class TestSliceGPU:
    """Tests for GPU slice operations with interpreter."""

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16],
    )
    def test_slice_1d_gpu(self, dtype: DType) -> None:
        """Test basic 1D slice on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch_dtype, device="cuda"
        )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.slice_tensor(x, [slice(1, 4)])

        expected = x_torch[1:4]
        torch.testing.assert_close(torch.from_dlpack(result), expected)

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16],
    )
    def test_slice_2d_gpu(self, dtype: DType) -> None:
        """Test 2D slice across both dimensions on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.arange(12, dtype=torch_dtype, device="cuda").reshape(
            3, 4
        )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.slice_tensor(x, [slice(0, 2), slice(1, 3)])

        expected = x_torch[0:2, 1:3]
        torch.testing.assert_close(torch.from_dlpack(result), expected)

    def test_slice_with_step_gpu(self) -> None:
        """Test slice with step > 1 on GPU."""
        x_torch = torch.arange(10, dtype=torch.float32, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.slice_tensor(x, [slice(0, 10, 2)])

        expected = x_torch[0:10:2]
        torch.testing.assert_close(torch.from_dlpack(result), expected)

    def test_slice_3d_gpu(self) -> None:
        """Test slice with 3D tensor on GPU."""
        x_torch = torch.arange(24, dtype=torch.float32, device="cuda").reshape(
            2, 3, 4
        )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = F.slice_tensor(x, [slice(0, 2), slice(1, 3), slice(0, 2)])

        expected = x_torch[0:2, 1:3, 0:2]
        torch.testing.assert_close(torch.from_dlpack(result), expected)


class TestGatherGPU:
    """Tests for GPU gather operations with interpreter."""

    def test_gather_gpu_axis0(self) -> None:
        """Test gather along axis 0 on GPU."""
        x_torch = torch.arange(12, dtype=torch.float32, device="cuda").reshape(
            3, 4
        )
        idx_torch = torch.tensor([2, 0], dtype=torch.int64, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        idx = Tensor.from_dlpack(idx_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather(x, idx, axis=0)

        expected = torch.index_select(x_torch, 0, idx_torch)
        torch.testing.assert_close(torch.from_dlpack(y), expected)

    def test_gather_gpu_axis1(self) -> None:
        """Test gather along axis 1 on GPU."""
        x_torch = torch.arange(12, dtype=torch.float32, device="cuda").reshape(
            3, 4
        )
        idx_torch = torch.tensor([3, 1, 0], dtype=torch.int64, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        idx = Tensor.from_dlpack(idx_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather(x, idx, axis=1)

        expected = torch.index_select(x_torch, 1, idx_torch)
        torch.testing.assert_close(torch.from_dlpack(y), expected)

    def test_gather_gpu_3d(self) -> None:
        """Test gather on 3D GPU tensor."""
        x_torch = torch.arange(60, dtype=torch.float32, device="cuda").reshape(
            3, 4, 5
        )
        idx_torch = torch.tensor([1, 3], dtype=torch.int64, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        idx = Tensor.from_dlpack(idx_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather(x, idx, axis=1)

        expected = torch.index_select(x_torch, 1, idx_torch)
        torch.testing.assert_close(torch.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", [DType.float32, DType.float16])
    def test_gather_gpu_dtypes(self, dtype: DType) -> None:
        """Test gather with various dtypes on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.arange(12, dtype=torch_dtype, device="cuda").reshape(
            3, 4
        )
        idx_torch = torch.tensor([1, 0, 2], dtype=torch.int64, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        idx = Tensor.from_dlpack(idx_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather(x, idx, axis=0)

        expected = torch.index_select(x_torch, 0, idx_torch)
        torch.testing.assert_close(torch.from_dlpack(y), expected)


class TestGatherNdGPU:
    """Tests for GPU gather_nd operations with interpreter."""

    def test_gather_nd_gpu_basic(self) -> None:
        """Test basic gather_nd on GPU (full indexing)."""
        x_torch = torch.arange(12, dtype=torch.float32, device="cuda").reshape(
            3, 4
        )
        idx_torch = torch.tensor(
            [[0, 1], [2, 3]], dtype=torch.int64, device="cuda"
        )

        x = Tensor.from_dlpack(x_torch)
        idx = Tensor.from_dlpack(idx_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather_nd(x, idx)

        expected = torch.tensor(
            [x_torch[0, 1].item(), x_torch[2, 3].item()],
            dtype=torch.float32,
            device="cuda",
        )
        torch.testing.assert_close(torch.from_dlpack(y), expected)

    def test_gather_nd_gpu_partial(self) -> None:
        """Test gather_nd on GPU with partial indexing (slicing)."""
        x_torch = torch.arange(24, dtype=torch.float32, device="cuda").reshape(
            2, 3, 4
        )
        idx_torch = torch.tensor([[0], [1]], dtype=torch.int64, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        idx = Tensor.from_dlpack(idx_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather_nd(x, idx)

        # output shape: (2, 3, 4) - gather slices along first dim
        expected = x_torch[torch.tensor([0, 1], device="cuda")]
        torch.testing.assert_close(torch.from_dlpack(y), expected)

    def test_gather_nd_gpu_batch_dims(self) -> None:
        """Test gather_nd on GPU with batch_dims > 0."""
        x_torch = torch.arange(24, dtype=torch.float32, device="cuda").reshape(
            2, 3, 4
        )
        idx_torch = torch.tensor([[1], [0]], dtype=torch.int64, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        idx = Tensor.from_dlpack(idx_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.gather_nd(x, idx, batch_dims=1)

        # output shape: (2, 4) - per-batch gather into dim 1
        expected = torch.stack([x_torch[0, 1], x_torch[1, 0]])
        torch.testing.assert_close(torch.from_dlpack(y), expected)


class TestArgMaxMinGPU:
    """Tests for GPU argmax/argmin operations with interpreter.

    Parameterized on op and axis to avoid duplication.
    """

    @pytest.mark.parametrize("op_name", ["argmax", "argmin"])
    @pytest.mark.parametrize("axis", [0, 1, -1])
    @pytest.mark.parametrize("dtype", [DType.float32, DType.float16])
    def test_3d_axes(self, op_name: str, axis: int, dtype: DType) -> None:
        """Test argmax/argmin on a 3D GPU tensor along each axis."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        torch_op = getattr(torch, op_name)
        f_op = getattr(F, op_name)
        x_torch = torch.randn(3, 4, 5, dtype=torch_dtype, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = f_op(x, axis=axis)

        result_torch = torch.from_dlpack(y)
        expected = torch_op(x_torch, dim=axis, keepdim=True)
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize("op_name", ["argmax", "argmin"])
    def test_2d(self, op_name: str) -> None:
        """Test on a small 2D GPU tensor along axis 1."""
        torch_op = getattr(torch, op_name)
        f_op = getattr(F, op_name)
        x_torch = torch.tensor(
            [[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]],
            dtype=torch.float32,
            device="cuda",
        )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = f_op(x, axis=1)

        result_torch = torch.from_dlpack(y)
        expected = torch_op(x_torch, dim=1, keepdim=True)
        torch.testing.assert_close(result_torch, expected)


class TestSplitGPU:
    """Tests for GPU split operations with interpreter."""

    @staticmethod
    def _assert_split_close(
        results: Sequence[object],
        expected: tuple[torch.Tensor, ...],
    ) -> None:
        assert len(results) == len(expected)
        for result, exp in zip(results, expected, strict=True):
            assert isinstance(result, Tensor)
            torch.testing.assert_close(torch.from_dlpack(result), exp)

    @pytest.mark.parametrize("dtype", [DType.float32, DType.float16])
    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_2d_axes(self, dtype: DType, axis: int) -> None:
        """Test split on a 2D GPU tensor along different axes."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.arange(24, dtype=torch_dtype, device="cuda").reshape(
            6, 4
        )
        split_sizes = [2, 4] if axis == 0 else [1, 3]

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            results = F.split(x, split_sizes, axis=axis)

        canonical = axis if axis >= 0 else axis + 2
        expected = torch.split(x_torch, split_sizes, dim=canonical)
        self._assert_split_close(results, expected)

    @pytest.mark.parametrize("dtype", [DType.float32, DType.float16])
    def test_3d(self, dtype: DType) -> None:
        """Test split on a 3D GPU tensor along the middle axis."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.arange(60, dtype=torch_dtype, device="cuda").reshape(
            3, 4, 5
        )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            results = F.split(x, [1, 2, 1], axis=1)

        expected = torch.split(x_torch, [1, 2, 1], dim=1)
        self._assert_split_close(results, expected)

    def test_equal_split(self) -> None:
        """Test equal-size split on GPU."""
        x_torch = torch.arange(12, dtype=torch.float32, device="cuda").reshape(
            4, 3
        )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            results = F.split(x, 2, axis=0)

        expected = torch.split(x_torch, 2, dim=0)
        self._assert_split_close(results, expected)


class TestConv2dGPU:
    """Tests for GPU conv2d operations with interpreter."""

    @pytest.mark.parametrize("dtype", [DType.float32, DType.float16])
    def test_basic_3x3(self, dtype: DType) -> None:
        """Test basic 3x3 conv on GPU, stride 1, no padding."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.arange(
            1 * 5 * 5 * 1, dtype=torch_dtype, device="cuda"
        ).reshape(1, 5, 5, 1)
        f_torch = torch.ones(3, 3, 1, 1, dtype=torch_dtype, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        f = Tensor.from_dlpack(f_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.conv2d(x, f)

        x_nchw = x_torch.permute(0, 3, 1, 2)
        f_nchw = f_torch.permute(3, 2, 0, 1)
        expected_nchw = torch.nn.functional.conv2d(x_nchw, f_nchw)
        expected = expected_nchw.permute(0, 2, 3, 1)
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, rtol=1e-3, atol=1e-3
        )

    def test_stride_and_padding(self) -> None:
        """Test conv2d with stride 2 and padding on GPU."""
        x_torch = torch.arange(
            1 * 6 * 6 * 1, dtype=torch.float32, device="cuda"
        ).reshape(1, 6, 6, 1)
        f_torch = torch.ones(3, 3, 1, 1, dtype=torch.float32, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        f = Tensor.from_dlpack(f_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.conv2d(x, f, stride=(2, 2), padding=(1, 1, 1, 1))

        x_nchw = x_torch.permute(0, 3, 1, 2)
        f_nchw = f_torch.permute(3, 2, 0, 1)
        expected_nchw = torch.nn.functional.conv2d(
            x_nchw, f_nchw, stride=2, padding=1
        )
        expected = expected_nchw.permute(0, 2, 3, 1)
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, rtol=1e-3, atol=1e-3
        )

    def test_groups(self) -> None:
        """Grouped conv (groups > 1) is not supported after the GC migration
        (MXF-529): it needs a pre-packed filter layout the eager
        interpreter never has, and there's no way to pack it from Python."""
        x_torch = torch.arange(
            1 * 4 * 4 * 4, dtype=torch.float32, device="cuda"
        ).reshape(1, 4, 4, 4)
        f_torch = torch.ones(3, 3, 2, 4, dtype=torch.float32, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        f = Tensor.from_dlpack(f_torch)
        with pytest.raises(NotImplementedError, match="groups"):
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                y = F.conv2d(x, f, groups=2)
                assert y is not None


class TestConvTranspose2dGPU:
    """Tests for GPU conv2d_transpose operations with interpreter."""

    def test_unsupported(self) -> None:
        """conv_transpose has no supported kernel (KERN-3233) -- see
        TestConvTranspose2dOp.test_unsupported for the full rationale."""
        x_torch = torch.arange(
            1 * 3 * 3 * 1, dtype=torch.float32, device="cuda"
        ).reshape(1, 3, 3, 1)
        f_torch = torch.ones(3, 3, 1, 1, dtype=torch.float32, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        f = Tensor.from_dlpack(f_torch)
        with pytest.raises(NotImplementedError, match="conv_transpose"):
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                y = F.conv2d_transpose(x, f)
                assert y is not None


class TestMaxPool2dGPU:
    """Tests for GPU max_pool2d operations with interpreter."""

    @pytest.mark.parametrize("dtype", [DType.float32, DType.float16])
    def test_basic_2x2(self, dtype: DType) -> None:
        """Test basic 2x2 max pool on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.arange(
            1 * 4 * 4 * 1, dtype=torch_dtype, device="cuda"
        ).reshape(1, 4, 4, 1)

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.max_pool2d(x, kernel_size=(2, 2))

        x_nchw = x_torch.permute(0, 3, 1, 2)
        expected_nchw = torch.nn.functional.max_pool2d(
            x_nchw, kernel_size=2, stride=1
        )
        expected = expected_nchw.permute(0, 2, 3, 1)
        torch.testing.assert_close(torch.from_dlpack(y), expected)

    def test_stride_and_padding(self) -> None:
        """Test max pool with stride 2 and padding on GPU."""
        x_torch = torch.arange(
            1 * 6 * 6 * 2, dtype=torch.float32, device="cuda"
        ).reshape(1, 6, 6, 2)

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1)

        x_nchw = x_torch.permute(0, 3, 1, 2)
        expected_nchw = torch.nn.functional.max_pool2d(
            x_nchw, kernel_size=3, stride=2, padding=1
        )
        expected = expected_nchw.permute(0, 2, 3, 1)
        torch.testing.assert_close(torch.from_dlpack(y), expected)

    def test_ceil_mode(self) -> None:
        """Test max pool with ceil_mode on GPU."""
        x_torch = torch.arange(
            1 * 5 * 5 * 1, dtype=torch.float32, device="cuda"
        ).reshape(1, 5, 5, 1)

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.max_pool2d(x, kernel_size=(2, 2), stride=2, ceil_mode=True)

        x_nchw = x_torch.permute(0, 3, 1, 2)
        expected_nchw = torch.nn.functional.max_pool2d(
            x_nchw, kernel_size=2, stride=2, ceil_mode=True
        )
        expected = expected_nchw.permute(0, 2, 3, 1)
        torch.testing.assert_close(torch.from_dlpack(y), expected)


class TestBandPartGPU:
    """GPU tests for mo.linalg.band_part interpreter handler."""

    @staticmethod
    def _band_part_ref_torch(
        x: torch.Tensor,
        num_lower: int,
        num_upper: int,
        exclude: bool = False,
    ) -> torch.Tensor:
        """PyTorch band_part reference."""
        M, N = x.shape[-2], x.shape[-1]
        m = torch.arange(M, device=x.device).unsqueeze(1)
        n = torch.arange(N, device=x.device).unsqueeze(0)
        lower_ok = (num_lower < 0) | ((m - n) <= num_lower)
        upper_ok = (num_upper < 0) | ((n - m) <= num_upper)
        in_band = lower_ok & upper_ok
        if exclude:
            in_band = ~in_band
        return torch.where(in_band, x, torch.zeros_like(x))

    @pytest.mark.parametrize("dtype", [DType.float32, DType.float16])
    def test_lower_triangle(self, dtype: DType) -> None:
        """Test causal mask (lower triangle) on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = (
            torch.arange(12, dtype=torch_dtype, device="cuda").reshape(3, 4) + 1
        )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.band_part(x, num_lower=None, num_upper=0)

        expected = self._band_part_ref_torch(x_torch, -1, 0)
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, rtol=0, atol=0
        )

    @pytest.mark.parametrize("dtype", [DType.float32, DType.float16])
    def test_band(self, dtype: DType) -> None:
        """Test tridiagonal band on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = (
            torch.arange(20, dtype=torch_dtype, device="cuda").reshape(4, 5) + 1
        )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.band_part(x, num_lower=1, num_upper=1)

        expected = self._band_part_ref_torch(x_torch, 1, 1)
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, rtol=0, atol=0
        )

    @pytest.mark.parametrize("dtype", [DType.float32, DType.float16])
    def test_exclude(self, dtype: DType) -> None:
        """Test inverted mask on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = (
            torch.arange(9, dtype=torch_dtype, device="cuda").reshape(3, 3) + 1
        )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.band_part(x, num_lower=None, num_upper=0, exclude=True)

        expected = self._band_part_ref_torch(x_torch, -1, 0, exclude=True)
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, rtol=0, atol=0
        )


class TestAvgPool2dGPU:
    """GPU tests for mo.avg_pool interpreter handler."""

    @pytest.mark.parametrize("dtype", [DType.float32, DType.float16])
    def test_basic_2x2(self, dtype: DType) -> None:
        """Test 2x2 avg pool on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_nchw = torch.arange(16, dtype=torch_dtype, device="cuda").reshape(
            1, 1, 4, 4
        )
        x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()

        x = Tensor.from_dlpack(x_nhwc)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.avg_pool2d(x, kernel_size=(2, 2))

        expected_nchw = torch.nn.functional.avg_pool2d(
            x_nchw, kernel_size=2, stride=1
        )
        expected_nhwc = expected_nchw.permute(0, 2, 3, 1).contiguous()
        torch.testing.assert_close(
            torch.from_dlpack(y), expected_nhwc, rtol=1e-3, atol=1e-3
        )

    def test_stride_and_padding(self) -> None:
        """Test stride 2, padding 1 on GPU."""
        x_nchw = torch.arange(25, dtype=torch.float32, device="cuda").reshape(
            1, 1, 5, 5
        )
        x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()

        x = Tensor.from_dlpack(x_nhwc)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.avg_pool2d(
                x,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                count_boundary=True,
            )

        expected_nchw = torch.nn.functional.avg_pool2d(
            x_nchw,
            kernel_size=3,
            stride=2,
            padding=1,
            count_include_pad=True,
        )
        expected_nhwc = expected_nchw.permute(0, 2, 3, 1).contiguous()
        torch.testing.assert_close(
            torch.from_dlpack(y), expected_nhwc, rtol=1e-5, atol=1e-5
        )

    def test_ceil_mode(self) -> None:
        """Test ceil mode on GPU."""
        x_nchw = torch.arange(25, dtype=torch.float32, device="cuda").reshape(
            1, 1, 5, 5
        )
        x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()

        x = Tensor.from_dlpack(x_nhwc)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.avg_pool2d(x, kernel_size=(3, 3), stride=2, ceil_mode=True)

        expected_nchw = torch.nn.functional.avg_pool2d(
            x_nchw, kernel_size=3, stride=2, ceil_mode=True
        )
        expected_nhwc = expected_nchw.permute(0, 2, 3, 1).contiguous()
        torch.testing.assert_close(
            torch.from_dlpack(y), expected_nhwc, rtol=1e-5, atol=1e-5
        )


class TestTopKGPU:
    """GPU tests for mo.top_k interpreter handler."""

    @pytest.mark.parametrize("dtype", [DType.float32, DType.float16])
    @pytest.mark.parametrize("axis", [-1, 0])
    def test_basic_2d(self, dtype: DType, axis: int) -> None:
        """Test top-2 on a 2D GPU tensor along axis 0 and -1."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.tensor(
            [[3.0, 1.0, 4.0, 1.0, 5.0], [9.0, 2.0, 6.0, 5.0, 3.0]],
            dtype=torch_dtype,
            device="cuda",
        )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            vals, idxs = F.top_k(x, k=2, axis=axis)

        ref_vals, ref_idxs = torch.topk(
            x_torch, k=2, dim=axis, largest=True, sorted=True
        )
        torch.testing.assert_close(
            torch.from_dlpack(vals), ref_vals, rtol=0, atol=0
        )
        torch.testing.assert_close(
            torch.from_dlpack(idxs),
            ref_idxs.to(torch.int64),
            rtol=0,
            atol=0,
        )

    def test_3d(self) -> None:
        """Test top-3 on a 3D float32 GPU tensor along axis 1."""
        x_torch = torch.randn(2, 8, 4, dtype=torch.float32, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            vals, idxs = F.top_k(x, k=3, axis=1)

        ref_vals, ref_idxs = torch.topk(
            x_torch, k=3, dim=1, largest=True, sorted=True
        )
        torch.testing.assert_close(
            torch.from_dlpack(vals), ref_vals, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            torch.from_dlpack(idxs),
            ref_idxs.to(torch.int64),
            rtol=0,
            atol=0,
        )


class TestBottomKGPU:
    """GPU tests for mo.bottom_k interpreter handler."""

    @pytest.mark.parametrize("dtype", [DType.float32, DType.float16])
    @pytest.mark.parametrize("axis", [-1, 0])
    def test_basic_2d(self, dtype: DType, axis: int) -> None:
        """Test bottom-2 on a 2D GPU tensor along axis 0 and -1."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        # Use strictly distinct values to avoid tie-breaking ambiguity:
        # the kernel and torch may return tied indices in different orders.
        x_torch = torch.tensor(
            [[3.0, 1.0, 4.0, 2.0, 5.0], [9.0, 2.0, 6.0, 5.0, 3.0]],
            dtype=torch_dtype,
            device="cuda",
        )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            vals, idxs = F.bottom_k(x, k=2, axis=axis)

        ref_vals, ref_idxs = torch.topk(
            x_torch, k=2, dim=axis, largest=False, sorted=True
        )
        torch.testing.assert_close(
            torch.from_dlpack(vals), ref_vals, rtol=0, atol=0
        )
        torch.testing.assert_close(
            torch.from_dlpack(idxs),
            ref_idxs.to(torch.int64),
            rtol=0,
            atol=0,
        )

    def test_3d(self) -> None:
        """Test bottom-3 on a 3D float32 GPU tensor along axis 1."""
        x_torch = torch.randn(2, 8, 4, dtype=torch.float32, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            vals, idxs = F.bottom_k(x, k=3, axis=1)

        ref_vals, ref_idxs = torch.topk(
            x_torch, k=3, dim=1, largest=False, sorted=True
        )
        torch.testing.assert_close(
            torch.from_dlpack(vals), ref_vals, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            torch.from_dlpack(idxs),
            ref_idxs.to(torch.int64),
            rtol=0,
            atol=0,
        )


class TestScatterNdGPU:
    """GPU tests for scatter_nd op (overwrite) via MO interpreter.

    ``scatter_nd`` uses ``MO_SingleDevice`` so it supports GPU.
    The PyTorch reference manually applies the index vectors.
    """

    @staticmethod
    def _scatter_nd_ref_torch(
        x_torch: "torch.Tensor",
        updates_torch: "torch.Tensor",
        indices_np: "Any",
    ) -> "torch.Tensor":
        """PyTorch reference: copy input then overwrite at N-D index positions."""
        import numpy as np

        out = x_torch.clone()
        index_depth = indices_np.shape[-1]
        batch_shape = indices_np.shape[:-1]
        for batch_idx in np.ndindex(batch_shape):
            idx_vec = tuple(
                int(indices_np[batch_idx + (k,)]) for k in range(index_depth)
            )
            out[idx_vec] = updates_torch[batch_idx]
        return out

    def test_2d_row_scatter(self) -> None:
        """Test scatter_nd on a 2D GPU tensor with 1-D partial indexing."""
        import numpy as np
        from max.experimental.tensor import Tensor, realization_context

        x_torch = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=torch.float32,
            device="cuda",
        )
        updates_torch = torch.tensor(
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]],
            dtype=torch.float32,
            device="cuda",
        )
        indices_np = np.array([[0], [2]], dtype=np.int64)
        indices_torch = torch.from_numpy(indices_np).to("cuda")

        x = Tensor.from_dlpack(x_torch)
        updates = Tensor.from_dlpack(updates_torch)
        indices = Tensor.from_dlpack(indices_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd(x, updates, indices)

        expected = self._scatter_nd_ref_torch(
            x_torch, updates_torch, indices_np
        )
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, rtol=0, atol=0
        )

    def test_2d_element_scatter(self) -> None:
        """Test scatter_nd on a 2D GPU tensor with full 2-D indexing."""
        import numpy as np
        from max.experimental.tensor import Tensor, realization_context

        x_torch = torch.zeros(3, 3, dtype=torch.float32, device="cuda")
        updates_torch = torch.tensor(
            [10.0, 20.0, 30.0], dtype=torch.float32, device="cuda"
        )
        indices_np = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
        indices_torch = torch.from_numpy(indices_np).to("cuda")

        x = Tensor.from_dlpack(x_torch)
        updates = Tensor.from_dlpack(updates_torch)
        indices = Tensor.from_dlpack(indices_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd(x, updates, indices)

        expected = self._scatter_nd_ref_torch(
            x_torch, updates_torch, indices_np
        )
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, rtol=0, atol=0
        )

    @pytest.mark.parametrize("dtype", [DType.float32, DType.float16])
    def test_dtypes(self, dtype: DType) -> None:
        """Test scatter_nd on GPU with float32 and float16 dtypes."""
        import numpy as np
        from max.experimental.tensor import Tensor, realization_context

        torch_dtype = {
            DType.float32: torch.float32,
            DType.float16: torch.float16,
        }[dtype]
        x_torch = torch.zeros(4, 3, dtype=torch_dtype, device="cuda")
        updates_torch = torch.ones(2, 3, dtype=torch_dtype, device="cuda") * 5.0
        indices_np = np.array([[1], [3]], dtype=np.int64)
        indices_torch = torch.from_numpy(indices_np).to("cuda")

        x = Tensor.from_dlpack(x_torch)
        updates = Tensor.from_dlpack(updates_torch)
        indices = Tensor.from_dlpack(indices_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = F.scatter_nd(x, updates, indices)

        expected = self._scatter_nd_ref_torch(
            x_torch, updates_torch, indices_np
        )
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, rtol=1e-3, atol=1e-3
        )


class TestScatterNdReduceGPU:
    """GPU tests for the scatter_nd reduce variants (add/max/min/mul).

    The GPU reduce kernels fold duplicate index vectors with a compare-and-swap
    loop (KERN-3231), so the result is order-independent and matches the CPU
    fold; ``test_duplicate_indices`` is the direct check that no update races.
    """

    _REDUCERS = {
        "add": F.scatter_nd_add,
        "max": F.scatter_nd_max,
        "min": F.scatter_nd_min,
        "mul": F.scatter_nd_mul,
    }

    @staticmethod
    def _ref_torch(
        reduce_op: str,
        x_torch: "torch.Tensor",
        updates_torch: "torch.Tensor",
        indices_np: "Any",
    ) -> "torch.Tensor":
        """Sequential fold matching the kernel's per-index-vector reduction."""
        import numpy as np

        out = x_torch.clone()
        index_depth = indices_np.shape[-1]
        for batch_idx in np.ndindex(indices_np.shape[:-1]):
            idx_vec = tuple(
                int(indices_np[batch_idx + (k,)]) for k in range(index_depth)
            )
            update = updates_torch[batch_idx]
            if reduce_op == "add":
                out[idx_vec] = out[idx_vec] + update
            elif reduce_op == "mul":
                out[idx_vec] = out[idx_vec] * update
            elif reduce_op == "max":
                out[idx_vec] = torch.maximum(out[idx_vec], update)
            else:
                out[idx_vec] = torch.minimum(out[idx_vec], update)
        return out

    @pytest.mark.parametrize("reduce_op", ["add", "max", "min", "mul"])
    @pytest.mark.parametrize(
        "dtype", [DType.float32, DType.bfloat16, DType.int32]
    )
    def test_reduce_dtypes(self, reduce_op: str, dtype: DType) -> None:
        """scatter_nd reduce on GPU across float and int dtypes."""
        import numpy as np
        from max.experimental.tensor import Tensor, realization_context

        torch_dtype = {
            DType.float32: torch.float32,
            DType.bfloat16: torch.bfloat16,
            DType.int32: torch.int32,
        }[dtype]
        x_torch = torch.full((4, 3), 4, dtype=torch_dtype, device="cuda")
        updates_torch = (
            torch.arange(1, 7, dtype=torch.float32, device="cuda")
            .reshape(2, 3)
            .to(torch_dtype)
        )
        indices_np = np.array([[0], [2]], dtype=np.int64)
        indices_torch = torch.from_numpy(indices_np).to("cuda")

        x = Tensor.from_dlpack(x_torch)
        updates = Tensor.from_dlpack(updates_torch)
        indices = Tensor.from_dlpack(indices_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = self._REDUCERS[reduce_op](x, updates, indices)

        expected = self._ref_torch(
            reduce_op, x_torch, updates_torch, indices_np
        )
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, rtol=1e-2, atol=1e-2
        )

    @pytest.mark.parametrize("reduce_op", ["add", "max", "min", "mul"])
    def test_duplicate_indices(self, reduce_op: str) -> None:
        """Two updates hit slot 1; the reducer must fold both, not race."""
        import numpy as np
        from max.experimental.tensor import Tensor, realization_context

        # Base + update values chosen so folding both duplicates is observable
        # for each reducer (dropping either changes slot 1).
        base, update_vals = {
            "add": (0.0, [10.0, 20.0, 30.0]),
            "mul": (1.0, [2.0, 3.0, 4.0]),
            "max": (0.0, [10.0, 20.0, 5.0]),
            "min": (100.0, [30.0, 20.0, 50.0]),
        }[reduce_op]

        x_torch = torch.full((4,), base, dtype=torch.float32, device="cuda")
        updates_torch = torch.tensor(
            update_vals, dtype=torch.float32, device="cuda"
        )
        indices_np = np.array([[1], [1], [3]], dtype=np.int64)
        indices_torch = torch.from_numpy(indices_np).to("cuda")

        x = Tensor.from_dlpack(x_torch)
        updates = Tensor.from_dlpack(updates_torch)
        indices = Tensor.from_dlpack(indices_torch)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            y = self._REDUCERS[reduce_op](x, updates, indices)

        expected = self._ref_torch(
            reduce_op, x_torch, updates_torch, indices_np
        )
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, rtol=0, atol=0
        )


_NEED_2_GPUS = pytest.mark.skipif(
    accelerator_count() < 2,
    reason="Requires at least 2 accelerator devices",
)


class TestDistributedAllreduceSumHandler:
    """Tests for the DistributedAllreduceSumOp interpreter handler."""

    @_NEED_2_GPUS
    def test_allreduce_sum_low_level(self) -> None:
        """Directly build a graph with ops.allreduce.sum and run via the MO interpreter."""
        shape = [4, 4]
        gpu0, gpu1 = DeviceRef.GPU(0), DeviceRef.GPU(1)

        input_types: list[GType[Any]] = [
            TensorType(DType.float32, shape, gpu0),
            TensorType(DType.float32, shape, gpu1),
            GBufferType(DType.uint8, [1024], gpu0),
            GBufferType(DType.uint8, [1024], gpu1),
        ]

        with Graph("allreduce_test", input_types=input_types) as graph:
            t0, t1, b0, b1 = graph.inputs
            results = graph_ops.allreduce.sum(
                [t0.tensor, t1.tensor], [b0.buffer, b1.buffer]
            )
            graph.output(*results)

        a_np = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        )
        b_np = np.array(
            [
                [10, 20, 30, 40],
                [50, 60, 70, 80],
                [90, 100, 110, 120],
                [130, 140, 150, 160],
            ],
            dtype=np.float32,
        )
        expected = a_np + b_np

        dev0, dev1 = Accelerator(0), Accelerator(1)
        input_bufs = [
            Buffer.from_numpy(a_np).to(dev0),
            Buffer.from_numpy(b_np).to(dev1),
            Buffer(shape=[1024], dtype=DType.uint8, device=dev0),
            Buffer(shape=[1024], dtype=DType.uint8, device=dev1),
        ]

        assert _interpreter.can_execute(graph)
        outputs = _interpreter.execute(graph, input_bufs)

        assert len(outputs) == 2
        for out in outputs:
            assert isinstance(out, Buffer)
            np.testing.assert_allclose(
                out.to(CPU()).to_numpy(), expected, rtol=1e-5
            )

    @_NEED_2_GPUS
    def test_allreduce_sum_distributed_tensor_e2e(self) -> None:
        """End-to-end: all_reduce_sum on a Partial distributed Tensor via interpreter."""
        num_gpus = min(accelerator_count(), 2)
        devices = tuple(Accelerator(i) for i in range(num_gpus))
        mesh = DeviceMesh(
            devices=devices, mesh_shape=(num_gpus,), axis_names=("tp",)
        )

        data = np.ones((4, 4), dtype=np.float32)

        # Replicate data on every device, then re-label as Partial.
        rep_mapping = PlacementMapping(mesh, (Replicated(),))

        try:
            replicated = df_shard(
                Tensor.from_dlpack(np.ascontiguousarray(data)), rep_mapping
            )
        except ValueError as exc:
            if "Memory manager cannot satisfy allocation" in str(exc):
                pytest.skip(
                    "Signal buffer allocation requires more GPU memory "
                    "than available on this executor"
                )
            raise

        partial_t = Tensor._from_shards(
            tuple(s.driver_tensor for s in replicated.local_shards),
            mesh,
            (Partial(),),
            data.shape,
        )

        try:
            with (
                rc.EagerRealizationContext() as ctx,
                realization_context(ctx),
            ):
                result = all_reduce_sum(partial_t)
        except ValueError as exc:
            if "Memory manager cannot satisfy allocation" in str(exc):
                pytest.skip(
                    "Signal buffer allocation requires more GPU memory "
                    "than available on this executor"
                )
            raise

        assert result.placements == (Replicated(),)
        expected = data * num_gpus
        for shard in result.local_shards:
            np.testing.assert_allclose(shard.to_numpy(), expected, rtol=1e-5)


class TestDistributedAllgatherHandler:
    """Tests for the DistributedAllgatherOp interpreter handler."""

    @_NEED_2_GPUS
    def test_allgather_eager(self) -> None:
        """all_gather on a Sharded distributed Tensor via eager interpreter."""
        num_gpus = min(accelerator_count(), 2)
        devices = tuple(Accelerator(i) for i in range(num_gpus))
        mesh = DeviceMesh(
            devices=devices, mesh_shape=(num_gpus,), axis_names=("tp",)
        )

        data = np.arange(num_gpus * 8, dtype=np.float32).reshape(
            num_gpus * 2, 4
        )

        sharded_t = df_shard(
            Tensor.from_dlpack(np.ascontiguousarray(data)),
            PlacementMapping(mesh, (Sharded(0),)),
        )

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = all_gather(sharded_t, tensor_axis=0)

        assert result.placements == (Replicated(),)
        for shard in result.local_shards:
            np.testing.assert_allclose(shard.to_numpy(), data, rtol=1e-5)


class TestDistributedScatterHandler:
    """Tests for the DistributedScatterOp interpreter handler."""

    @_NEED_2_GPUS
    def test_scatter_via_graph_ops(self) -> None:
        """Build a graph using ops.distributed_scatter and run via the MO interpreter."""
        shape = [4]
        gpu0, gpu1 = DeviceRef.GPU(0), DeviceRef.GPU(1)

        input_types: list[GType[Any]] = [
            TensorType(DType.float32, shape, gpu0),
            TensorType(DType.float32, shape, gpu0),
            GBufferType(DType.uint8, [1024], gpu0),
            GBufferType(DType.uint8, [1024], gpu1),
        ]

        with Graph("scatter_e2e", input_types=input_types) as graph:
            t0, t1, b0, b1 = graph.inputs
            scatter_out = graph_ops.distributed_scatter(
                [t0.tensor, t1.tensor], [b0.buffer, b1.buffer]
            )
            graph.output(*scatter_out)

        a_np = np.array([1, 2, 3, 4], dtype=np.float32)
        b_np = np.array([10, 20, 30, 40], dtype=np.float32)

        dev0, dev1 = Accelerator(0), Accelerator(1)
        input_bufs = [
            Buffer.from_numpy(a_np).to(dev0),
            Buffer.from_numpy(b_np).to(dev0),
            Buffer(shape=[1024], dtype=DType.uint8, device=dev0),
            Buffer(shape=[1024], dtype=DType.uint8, device=dev1),
        ]

        assert _interpreter.can_execute(graph)
        outputs = _interpreter.execute(graph, input_bufs)

        assert len(outputs) == 2
        out0, out1 = outputs[0], outputs[1]
        assert isinstance(out0, Buffer)
        assert isinstance(out1, Buffer)
        np.testing.assert_allclose(out0.to(CPU()).to_numpy(), a_np, rtol=1e-5)
        np.testing.assert_allclose(out1.to(CPU()).to_numpy(), b_np, rtol=1e-5)

    @_NEED_2_GPUS
    def test_scatter_distributed_tensor_e2e(self) -> None:
        """E2E: scatter pre-split chunks to Sharded via interpreter."""
        num_gpus = min(accelerator_count(), 2)
        devices = tuple(Accelerator(i) for i in range(num_gpus))
        mesh = DeviceMesh(
            devices=devices, mesh_shape=(num_gpus,), axis_names=("dp",)
        )

        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        rows_per_chunk = 4 // num_gpus

        mapping = PlacementMapping(mesh, (Sharded(0),))

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = df_shard(Tensor(data), mapping)

        assert result.placements == (Sharded(0),)
        for i, shard in enumerate(result.local_shards):
            expected = data[i * rows_per_chunk : (i + 1) * rows_per_chunk]
            np.testing.assert_allclose(shard.to_numpy(), expected, rtol=1e-5)


class TestDistributedBroadcastHandler:
    """Tests for the DistributedBroadcastOp interpreter handler."""

    @_NEED_2_GPUS
    def test_broadcast_via_graph_ops(self) -> None:
        """Build a graph using ops.distributed_broadcast and run via the MO interpreter."""
        shape = [4]
        gpu0, gpu1 = DeviceRef.GPU(0), DeviceRef.GPU(1)

        input_types: list[GType[Any]] = [
            TensorType(DType.float32, shape, gpu0),
            GBufferType(DType.uint8, [1024], gpu0),
            GBufferType(DType.uint8, [1024], gpu1),
        ]

        with Graph("broadcast_e2e", input_types=input_types) as graph:
            t0, b0, b1 = graph.inputs
            broadcast_out = graph_ops.distributed_broadcast(
                t0.tensor, [b0.buffer, b1.buffer]
            )
            graph.output(*broadcast_out)

        a_np = np.array([1, 2, 3, 4], dtype=np.float32)

        dev0, dev1 = Accelerator(0), Accelerator(1)
        input_bufs = [
            Buffer.from_numpy(a_np).to(dev0),
            Buffer(shape=[1024], dtype=DType.uint8, device=dev0),
            Buffer(shape=[1024], dtype=DType.uint8, device=dev1),
        ]

        assert _interpreter.can_execute(graph)
        outputs = _interpreter.execute(graph, input_bufs)

        assert len(outputs) == 2
        out0, out1 = outputs[0], outputs[1]
        assert isinstance(out0, Buffer)
        assert isinstance(out1, Buffer)
        np.testing.assert_allclose(out0.to(CPU()).to_numpy(), a_np, rtol=1e-5)
        np.testing.assert_allclose(out1.to(CPU()).to_numpy(), a_np, rtol=1e-5)

    @_NEED_2_GPUS
    def test_broadcast_distributed_tensor_e2e(self) -> None:
        """E2E: broadcast a tensor to Replicated via interpreter."""
        num_gpus = min(accelerator_count(), 2)
        devices = tuple(Accelerator(i) for i in range(num_gpus))
        mesh = DeviceMesh(
            devices=devices, mesh_shape=(num_gpus,), axis_names=("dp",)
        )

        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        t = Tensor(storage=Buffer.from_numpy(data).to(devices[0]))

        mapping = PlacementMapping(mesh, (Replicated(),))

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            result = df_shard(t, mapping)

        assert result.placements == (Replicated(),)
        for shard in result.local_shards:
            np.testing.assert_allclose(shard.to_numpy(), data, rtol=1e-5)


class TestDistributedReducescatterSumHandler:
    """Tests for the DistributedReducescatterSumOp interpreter handler."""

    @_NEED_2_GPUS
    def test_reducescatter_sum_via_graph_ops(self) -> None:
        """Build a graph with ops.reducescatter.sum and run via the MO interpreter."""
        shape = [4, 4]
        gpu0, gpu1 = DeviceRef.GPU(0), DeviceRef.GPU(1)

        input_types: list[GType[Any]] = [
            TensorType(DType.float32, shape, gpu0),
            TensorType(DType.float32, shape, gpu1),
            GBufferType(DType.uint8, [1024], gpu0),
            GBufferType(DType.uint8, [1024], gpu1),
        ]

        with Graph("reducescatter_test", input_types=input_types) as graph:
            t0, t1, b0, b1 = graph.inputs
            results = graph_ops.reducescatter.sum(
                [t0.tensor, t1.tensor], [b0.buffer, b1.buffer], axis=0
            )
            graph.output(*results)

        a_np = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        )
        b_np = np.array(
            [
                [10, 20, 30, 40],
                [50, 60, 70, 80],
                [90, 100, 110, 120],
                [130, 140, 150, 160],
            ],
            dtype=np.float32,
        )
        total = a_np + b_np
        # axis=0, 2 devices: each gets 2 rows of the 4-row sum.
        expected_0 = total[:2]  # rows 0-1
        expected_1 = total[2:]  # rows 2-3

        dev0, dev1 = Accelerator(0), Accelerator(1)
        input_bufs = [
            Buffer.from_numpy(a_np).to(dev0),
            Buffer.from_numpy(b_np).to(dev1),
            Buffer(shape=[1024], dtype=DType.uint8, device=dev0),
            Buffer(shape=[1024], dtype=DType.uint8, device=dev1),
        ]

        assert _interpreter.can_execute(graph)
        outputs = _interpreter.execute(graph, input_bufs)

        assert len(outputs) == 2
        out0, out1 = outputs[0], outputs[1]
        assert isinstance(out0, Buffer)
        assert isinstance(out1, Buffer)
        np.testing.assert_allclose(
            out0.to(CPU()).to_numpy(), expected_0, rtol=1e-5
        )
        np.testing.assert_allclose(
            out1.to(CPU()).to_numpy(), expected_1, rtol=1e-5
        )

    @_NEED_2_GPUS
    def test_reducescatter_sum_distributed_tensor_e2e(self) -> None:
        """E2E: reduce-scatter per-device tensors to Sharded via interpreter."""
        num_gpus = min(accelerator_count(), 2)
        devices = tuple(Accelerator(i) for i in range(num_gpus))
        mesh = DeviceMesh(
            devices=devices, mesh_shape=(num_gpus,), axis_names=("tp",)
        )

        # Each device contributes a [4, 4] tensor of ones (Partial).
        data = np.ones((4, 4), dtype=np.float32)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            # Build a Partial tensor with one shard per device.
            shard_tvs = [
                Tensor(
                    storage=Buffer.from_numpy(data).to(devices[i])
                ).__tensorvalue__()
                for i in range(num_gpus)
            ]
            partial_t = Tensor.from_shard_values(
                shard_tvs, PlacementMapping(mesh, (Partial(),))
            )
            result = reduce_scatter(partial_t, scatter_axis=0, mesh_axis=0)

        assert result.placements == (Sharded(0),)
        # Sum of num_gpus copies of ones, split along axis 0.
        total = data * num_gpus
        rows_per_chunk = total.shape[0] // num_gpus
        for i, shard in enumerate(result.local_shards):
            expected = total[i * rows_per_chunk : (i + 1) * rows_per_chunk]
            np.testing.assert_allclose(shard.to_numpy(), expected, rtol=1e-5)


class TestMutableStoreOpsGPU:
    """End-to-end GPU tests for the mutable-tensor write interpreter handlers.

    GPU-resident buffers exercise the handler's device round-trip path
    (as opposed to the host fast path tested on CPU).
    """

    def test_buffer_store_gpu(self) -> None:
        """F.buffer_store writes into a GPU-resident buffer."""
        gpu = Accelerator()
        buf = Buffer.from_numpy(np.zeros(4, dtype=np.float32)).to(gpu)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            a = Tensor(storage=buf)
            b_torch = torch.ones(4, dtype=torch.float32, device="cuda")
            b = Tensor.from_dlpack(b_torch)
            F.buffer_store(a, b)

        np.testing.assert_array_equal(
            buf.to_numpy(), np.ones(4, dtype=np.float32)
        )

    def test_buffer_store_slice_gpu_unit_steps(self) -> None:
        """F.buffer_store_slice writes a contiguous region into a GPU buffer."""
        gpu = Accelerator()
        buf = Buffer.from_numpy(np.zeros((4, 4), dtype=np.float32)).to(gpu)
        slice_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            a = Tensor(storage=buf)
            b_torch = torch.from_numpy(slice_np).to("cuda")
            b = Tensor.from_dlpack(b_torch)
            F.buffer_store_slice(a, b, [slice(1, 3), slice(1, 3)])

        expected = np.zeros((4, 4), dtype=np.float32)
        expected[1:3, 1:3] = slice_np
        np.testing.assert_array_equal(buf.to_numpy(), expected)

    def test_buffer_store_slice_gpu_stepped(self) -> None:
        """F.buffer_store_slice honors steps on a GPU buffer."""
        gpu = Accelerator()
        buf = Buffer.from_numpy(np.zeros(8, dtype=np.float32)).to(gpu)
        slice_np = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            a = Tensor(storage=buf)
            b_torch = torch.from_numpy(slice_np).to("cuda")
            b = Tensor.from_dlpack(b_torch)
            F.buffer_store_slice(a, b, [slice(0, 8, 2)])

        expected = np.zeros(8, dtype=np.float32)
        expected[0:8:2] = slice_np
        np.testing.assert_array_equal(buf.to_numpy(), expected)

    def test_buffer_store_slice_bfloat16_gpu(self) -> None:
        """F.buffer_store_slice on a GPU bf16 buffer."""
        gpu = Accelerator()

        # Represent bf16 values via uint16 bytes so numpy can hold them.
        # bf16 encoding of 1.0=0x3F80, 2.0=0x4000, 3.0=0x4040, 4.0=0x4080.
        dst_u16 = np.zeros((4, 4), dtype=np.uint16)
        src_u16 = np.array(
            [[0x3F80, 0x4000], [0x4040, 0x4080]], dtype=np.uint16
        )

        buf = Buffer.from_numpy(dst_u16).view(DType.bfloat16).to(gpu)
        src_buf = Buffer.from_numpy(src_u16).view(DType.bfloat16).to(gpu)

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            a = Tensor(storage=buf)
            b = Tensor(storage=src_buf)
            F.buffer_store_slice(a, b, [slice(1, 3), slice(1, 3)])

        expected = np.zeros((4, 4), dtype=np.uint16)
        expected[1:3, 1:3] = src_u16
        np.testing.assert_array_equal(
            buf.view(DType.uint16).to_numpy(), expected
        )

    def test_buffer_store_slice_float8_e4m3fn_gpu(self) -> None:
        """F.buffer_store_slice on a GPU float8_e4m3fn buffer."""
        gpu = Accelerator()

        dst_bytes = np.zeros((4, 4), dtype=np.uint8)
        src_bytes = np.array([[0x11, 0x22], [0x33, 0x44]], dtype=np.uint8)

        buf = Buffer.from_numpy(dst_bytes).view(DType.float8_e4m3fn).to(gpu)
        src_buf = Buffer.from_numpy(src_bytes).view(DType.float8_e4m3fn).to(gpu)

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
