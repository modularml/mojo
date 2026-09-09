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
"""Module framework for :mod:`max.experimental`.

Provides the :class:`Module` base class, the :func:`module_dataclass`
decorator, and :meth:`Module.compile`, which traces ``forward`` against
symbolic inputs and returns an ahead-of-time-compiled ``CompiledModel``. The
same ``forward`` body runs eagerly, traces to a :class:`~max.graph.Graph`, or
compiles, depending on the active realization context.

Example:

.. code-block:: python

    from max.driver import CPU
    from max.dtype import DType
    from max.experimental.tensor import Tensor, default_device
    from max.experimental.nn import Module, module_dataclass
    from max.graph import TensorType

    @module_dataclass
    class MyLayer(Module):
        weight: Tensor
        bias: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x @ self.weight.T + self.bias

    with default_device(CPU()):
        model = MyLayer(weight=Tensor.zeros([10, 5]), bias=Tensor.zeros([10]))
        y = model(Tensor.ones([3, 5]))

        input_type = TensorType(DType.float32, ["batch", 5], device=model.device)
        compiled = model.compile(input_type)
        result = compiled(Tensor.ones([3, 5]))

.. invisible-code-block: python

    import numpy as np

    assert tuple(int(d) for d in y.shape) == (3, 10)
    assert tuple(int(d) for d in result.shape) == (3, 10)
    assert np.allclose(y.to_numpy(), result.to_numpy())
    assert np.allclose(result.to_numpy(), np.zeros((3, 10)))
"""

from .common_layers.lora_wrapper import LoRA, lora_layers, lora_parameters
from .conv import Conv2d
from .embedding import Embedding
from .linear import Linear
from .module import (
    CompiledModel,
    Module,
    PinnedDeviceTensor,
    module_dataclass,
    subgraphable,
)
from .norm import GemmaRMSNorm, GroupNorm, LayerNorm, RMSNorm
from .rope import RotaryEmbedding, TransposedRotaryEmbedding
from .sequential import ModuleList, Sequential
from .transparent_module import TransparentModule

__all__ = [
    "CompiledModel",
    "Conv2d",
    "Embedding",
    "GemmaRMSNorm",
    "GroupNorm",
    "LayerNorm",
    "Linear",
    "LoRA",
    "Module",
    "ModuleList",
    "PinnedDeviceTensor",
    "RMSNorm",
    "RotaryEmbedding",
    "Sequential",
    "TransparentModule",
    "TransposedRotaryEmbedding",
    "lora_layers",
    "lora_parameters",
    "module_dataclass",
    "subgraphable",
]
