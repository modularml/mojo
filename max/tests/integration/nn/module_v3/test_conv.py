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
"""Tests for max.experimental.nn.Conv2d."""

from __future__ import annotations

import pytest
from max.driver import Accelerator, accelerator_count
from max.experimental.nn import Conv2d
from max.experimental.tensor import Tensor


@pytest.mark.skipif(not accelerator_count(), reason="requires an accelerator")
def test_conv2d_forward_moves_bias_to_input_device() -> None:
    """forward must move the bias to the input's device, not just the weight.

    The weight and bias default to CPU; running with the input on an
    accelerator previously moved only the weight, leaving the bias on CPU and
    failing with a device mismatch.
    """
    conv = Conv2d(
        kernel_size=3,
        in_channels=3,
        out_channels=8,
        has_bias=True,
        permute=True,
    )

    x = Tensor.ones([1, 3, 8, 8]).to(Accelerator())
    result = conv(x)

    assert result.device == Accelerator()
