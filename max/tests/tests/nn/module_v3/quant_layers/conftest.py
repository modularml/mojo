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

"""Shared fixtures for the quant_layers lazy-trace tests.

The quant layers are traced under :func:`max.experimental.functional.lazy`
against a mocked accelerator, so they exercise GPU kernel dispatch without a
real device. ``mock_accelerator`` provides that stand-in device;
``fp8_quant_config`` and ``nvfp4_quant_config`` provide the block-scaled FP8
and NVFP4 configs used to drive the quantized code paths.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from max.driver import Device
from max.dtype import DType
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

# FP8 block-scaled weight block size, matching the deepseekV3 FP8 checkpoint.
# Dimensions in the tests are multiples of this so the weight-scale grid
# divides evenly.
FP8_BLOCK_SIZE = (128, 128)

# NVFP4 packs two float4-e2m1 values per byte with one float8_e4m3fn scale per
# 16-element block along K.
NVFP4_BLOCK_SIZE = (1, 16)


def _make_fake_gpu(id: int = 0) -> MagicMock:
    """A stand-in accelerator so lazy tracing routes to GPU kernels.

    The mock reports ``label == "gpu"`` so device dispatch picks the
    accelerator path without instantiating a real ``Accelerator``.
    """
    label = "gpu"
    fake = MagicMock(spec=Device)
    fake.id = id
    fake.label = label
    fake.__eq__ = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda other: (
            getattr(other, "id", None) == id
            and getattr(other, "label", None) == label
        )
    )
    fake.__hash__ = MagicMock(return_value=hash((id, label)))  # type: ignore[method-assign]
    return fake


@pytest.fixture
def mock_accelerator() -> Iterator[MagicMock]:
    """Patches ``Accelerator`` with a fake-GPU factory for the test body.

    Call the yielded mock to mint devices: ``mock_accelerator()`` for a single
    device, or ``mock_accelerator(0)`` / ``mock_accelerator(1)`` for distinct
    ids.
    """
    with patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _make_fake_gpu
        yield mock


@pytest.fixture
def sm100_arch() -> Iterator[None]:
    """Reports an SM100 architecture for the duration of the test.

    The NVFP4 kernel wrappers pick their scale layout from the *host's*
    accelerator architecture (rank-5 TCGEN scale tiles on SM100/SM120, rank-2
    elsewhere), so tracing the Blackwell path takes more than a fake device.
    """
    with patch(
        "max.nn.kernels.accelerator_architecture_name", return_value="sm_100a"
    ):
        yield


@pytest.fixture
def fp8_quant_config() -> QuantConfig:
    """A block-scaled FP8 config matching the deepseekV3 FP8 checkpoint."""
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, 128),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float32,
            block_size=FP8_BLOCK_SIZE,
        ),
        mlp_quantized_layers={0},
        attn_quantized_layers={0},
        format=QuantFormat.BLOCKSCALED_FP8,
    )


@pytest.fixture
def nvfp4_quant_config() -> QuantConfig:
    """A modelopt NVFP4 config, mirroring ``parse_quant_config``'s output.

    The activation scale is *static* (loaded from the checkpoint) and drives
    the dynamic block quantization of activations before the FP4 matmul.
    """
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.STATIC,
            dtype=DType.float32,
            block_size=NVFP4_BLOCK_SIZE,
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e4m3fn,
            block_size=NVFP4_BLOCK_SIZE,
        ),
        mlp_quantized_layers={0},
        attn_quantized_layers={0},
        format=QuantFormat.NVFP4,
    )
