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

from collections.abc import Callable, Sequence

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.graph.ops import conv2d
from test_common.modular_graph_test import modular_graph_test

# Avoid using TF32 for F32 accuracy tests
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def torch_conv2d(  # noqa: ANN201
    x: TensorValue,
    filter: TensorValue,
    stride: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    groups: int = 1,
):
    x = torch.permute(x, (0, 3, 1, 2))
    filter = torch.permute(filter, (3, 2, 0, 1))
    out = F.conv2d(
        x,
        filter,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    return torch.permute(out, (0, 2, 3, 1))


@pytest.mark.parametrize("device", [DeviceRef.CPU(), DeviceRef.GPU()])
@pytest.mark.parametrize(
    "input_shape, filter_shape",
    [
        ([1, 16, 16, 4], [16, 16, 4, 5]),
    ],
)
def test_conv2d(
    session: InferenceSession,
    input_shape: list[int],
    filter_shape: list[int],
    device: DeviceRef,
) -> None:
    if device.device_type == "gpu" and accelerator_count() == 0:
        pytest.skip("No GPU available")

    input_type = TensorType(DType.float32, input_shape, device=device)
    filter_type = TensorType(DType.float32, filter_shape, device=device)

    with Graph("conv2d", input_types=[input_type, filter_type]) as graph:
        x, filter = graph.inputs
        stride = (16, 16)
        padding = (0, 0)
        dilation = (1, 1)

        conv = conv2d(x.tensor, filter.tensor, stride, dilation, (0, 0, 0, 0))
        graph.output(conv)

        @modular_graph_test(session, graph)
        def test_correctness(
            execute: Callable[[Sequence[Buffer]], Buffer],
            inputs: Sequence[Buffer],
            torch_inputs: Sequence[torch.Tensor],
        ) -> None:
            result = execute(inputs).to_numpy()
            x, w = torch_inputs
            expected = (
                torch_conv2d(x, w, stride, dilation, padding)
                .detach()
                .cpu()
                .numpy()
            )
            ACCURACY_RTOL = 1e-4
            ACCURACY_ATOL = 1e-6
            np.testing.assert_allclose(
                result,
                expected,
                equal_nan=True,
                rtol=ACCURACY_RTOL,
                atol=ACCURACY_ATOL,
            )


@pytest.mark.parametrize(
    "input_shape, filter_shape, groups",
    [
        ([1, 16, 16, 4], [16, 16, 2, 6], 2),
    ],
)
def test_conv2d_grouped(
    session: InferenceSession,
    input_shape: list[int],
    filter_shape: list[int],
    groups: int,
) -> None:
    # Grouped conv on CPU requires the filter to be pre-packed; the filter
    # here is a graph input (not a compile-time constant), which used to
    # crash with "grouped conv requires packed filter" (KERN-3239).
    device = DeviceRef.CPU()
    input_type = TensorType(DType.float32, input_shape, device=device)
    filter_type = TensorType(DType.float32, filter_shape, device=device)

    with Graph(
        "conv2d_grouped", input_types=[input_type, filter_type]
    ) as graph:
        x, filter = graph.inputs
        stride = (1, 1)
        padding = (0, 0)
        dilation = (1, 1)

        conv = conv2d(
            x.tensor,
            filter.tensor,
            stride,
            dilation,
            (0, 0, 0, 0),
            groups=groups,
        )
        graph.output(conv)

        @modular_graph_test(session, graph)
        def test_correctness(
            execute: Callable[[Sequence[Buffer]], Buffer],
            inputs: Sequence[Buffer],
            torch_inputs: Sequence[torch.Tensor],
        ) -> None:
            result = execute(inputs).to_numpy()
            x, w = torch_inputs
            expected = (
                torch_conv2d(x, w, stride, dilation, padding, groups=groups)
                .detach()
                .cpu()
                .numpy()
            )
            ACCURACY_RTOL = 1e-4
            ACCURACY_ATOL = 1e-6
            np.testing.assert_allclose(
                result,
                expected,
                equal_nan=True,
                rtol=ACCURACY_RTOL,
                atol=ACCURACY_ATOL,
            )
