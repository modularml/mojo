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
"""MinP Sampler custom ops."""

from max.dtype import DType
from max.graph import (
    DeviceRef,
    Shape,
    ShapeLike,
    TensorType,
    TensorValue,
    TensorValueLike,
    ops,
)
from max.nn.layer import Module


class MinPSampler(Module):
    """A min_p sampler."""

    dtype: DType
    shape: Shape
    min_p: float
    temperature: float

    def __init__(
        self,
        dtype: DType,
        shape: ShapeLike,
        temperature: float = 1,
    ) -> None:
        self.dtype = dtype
        self.shape = Shape(shape)
        self.temperature = temperature

    def __call__(
        self, input: TensorValue, min_p: TensorValueLike = 0.0
    ) -> TensorValue:
        batch_size = input.shape[0]
        # Handle min_p parameter - can be scalar or tensor
        if isinstance(min_p, float | int):
            if float(min_p) < 0.0 or float(min_p) > 1.0:
                raise ValueError(f"expected min_p to be in [0, 1], got {min_p}")
            min_p_tensor = ops.broadcast_to(
                ops.constant(min_p, dtype=DType.float32, device=input.device),
                [batch_size],
            )
        else:
            min_p_tensor = TensorValue(min_p)
            if min_p_tensor.shape[0] != batch_size:
                raise ValueError(
                    f"top_p tensor shape {min_p_tensor.shape} does not match batch_size {batch_size}"
                )

        return ops.custom(
            "min_p_sampling",
            input.device,
            [
                min_p_tensor,
                input,
                ops.constant(
                    self.temperature, dtype=self.dtype, device=DeviceRef.CPU()
                ),
            ],
            [TensorType(self.dtype, self.shape, device=input.device)],
        )[0].tensor
