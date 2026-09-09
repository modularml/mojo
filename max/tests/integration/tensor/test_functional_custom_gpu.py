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
"""Smoke tests for GPU custom ops in `max.experimental.functional`.

These tests exercise each expected op at least once with real data and kernels.
They don't otherwise make any attempt at coverage, edge cases, or correctness.
"""

import os
from pathlib import Path

import pytest
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.module import Module, module_dataclass
from max.experimental.tensor import Tensor, TensorType
from max.experimental.testing import assert_all_close


@pytest.fixture
def kernel_verification_ops_path() -> Path:
    return Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])


def test_custom_external_cubin(kernel_verification_ops_path: Path) -> None:
    # Kernel expects float32
    x = Tensor.zeros([64], dtype=DType.float32)
    y = Tensor.ones_like(x)

    assert x.device.api == "cuda"

    result = F.custom(
        "op_with_external_cubin",
        device=x.device,
        values=[x, y],
        out_types=[x.type],
        custom_extensions=[kernel_verification_ops_path],
    )[0]

    assert result.real
    assert result.type == x.type
    assert_all_close(result, y)


def test_compile_with_custom_extensions_gpu(
    kernel_verification_ops_path: Path,
) -> None:
    """Module.compile() loads GPU custom kernels for graph tracing."""

    @module_dataclass
    class GpuVecAddModule(Module[[Tensor, Tensor], Tensor]):
        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return F.custom(
                "op_with_external_cubin",
                device=x.device,
                values=[x, y],
                out_types=[x.type],
            )[0]

    x = Tensor.zeros([64], dtype=DType.float32)
    assert x.device.api == "cuda"

    input_type = TensorType(DType.float32, [64], device=x.device)
    module = GpuVecAddModule()

    compiled = module.compile(
        input_type,
        input_type,
        custom_extensions=[kernel_verification_ops_path],
    )

    y = Tensor.ones([64], dtype=DType.float32, device=x.device)
    result = compiled(x, y)

    assert result.shape == [64]
    assert result.dtype == DType.float32
    assert_all_close(result, y)


def test_compile_external_cubin_then_matmul_gpu(
    kernel_verification_ops_path: Path,
) -> None:
    """GEX-3806 repro: external-cubin custom op composed with a matmul."""

    @module_dataclass
    class VecAddThenMatmul(Module[[Tensor, Tensor, Tensor], Tensor]):
        def forward(self, x: Tensor, y: Tensor, w: Tensor) -> Tensor:
            z = F.custom(
                "op_with_external_cubin",
                device=x.device,
                values=[x, y],
                out_types=[x.type],
            )[0]
            return z.reshape([1, 64]) @ w

    x = Tensor.zeros([64], dtype=DType.float32)
    assert x.device.api == "cuda"

    vec_type = TensorType(DType.float32, [64], device=x.device)
    w_type = TensorType(DType.float32, [64, 16], device=x.device)
    module = VecAddThenMatmul()

    compiled = module.compile(
        vec_type,
        vec_type,
        w_type,
        custom_extensions=[kernel_verification_ops_path],
    )

    y = Tensor.ones([64], dtype=DType.float32, device=x.device)
    w = Tensor.ones([64, 16], dtype=DType.float32, device=x.device)
    result = compiled(x, y, w)

    assert result.shape == [1, 16]
    assert result.dtype == DType.float32
    # vec_add(0, 1) = 1 everywhere, so reshape([1, 64]) @ ones([64, 16])
    # sums 64 ones into each output element.
    assert_all_close(
        result,
        Tensor.full([1, 16], 64.0, dtype=DType.float32, device=result.device),
    )
