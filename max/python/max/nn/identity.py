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
"""Identity layer that passes through input unchanged."""

from __future__ import annotations

from max.graph import TensorValue

from .layer import Module


class Identity(Module):
    """Identity layer that passes through input unchanged.

    This layer is useful for skipping certain operations (like normalization)
    in specific architectures such as EAGLE speculative decoding, where the
    draft model receives already-normalized hidden states from the target model.

    When called, ``Identity`` accepts a :class:`~max.graph.TensorValue` and returns the
    same tensor unchanged.

    .. code-block:: python

        from max.driver import Accelerator, CPU, accelerator_count
        from max.dtype import DType
        from max.graph import DeviceRef, Graph, TensorType
        from max.nn import Identity

        device = Accelerator() if accelerator_count() > 0 else CPU()
        device_ref = DeviceRef.from_device(device)

        identity = Identity()
        input_type = TensorType(DType.float32, [1, 256], device=device_ref)
        with Graph("identity", input_types=[input_type]) as graph:
            output = identity(graph.inputs[0])  # output == input_tensor
            graph.output(output)
    """

    def __call__(self, x: TensorValue) -> TensorValue:
        """Pass through the input unchanged.

        Args:
            x: Input tensor value.

        Returns:
            The same tensor value as input.
        """
        return x
