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

"""Capture a MEF from a graph built with the new ``max.experimental`` API.

This is the new-API counterpart to ``test_capi.py``. It builds the same
vector-add graph as a :class:`~max.experimental.nn.Module`, compiles it, and
exports the compiled artifact to a MEF file via the public
:meth:`~max.experimental.nn.CompiledModel.export_mef`. The resulting
``graph.mef`` is consumed by the same ``example.c`` C-API executor, since both
APIs name graph inputs ``input0``/``input1`` and the output ``output0``.

``export_mef`` serializes straight from the compiled artifact, so it does not
require the model to be initialized on a live device. That makes it usable in
the cross-compilation / virtual-device scenarios that production serving via the
MAX C API relies on.
"""

import os

from max.driver import Accelerator
from max.dtype import DType
from max.experimental.nn import Module, module_dataclass
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType


@module_dataclass
class VectorAdd(Module[[Tensor, Tensor], Tensor]):
    """Adds two vectors elementwise."""

    def forward(self, vector1: Tensor, vector2: Tensor) -> Tensor:
        return vector1 + vector2


def build_graph() -> None:
    # Build and place the module on the accelerator. Calling `to` moves both
    # weights (none here) and computation to the device.
    device = Accelerator()
    model = VectorAdd().to(device)

    # Input tensors are expected on the accelerator. `vector_width` is a
    # symbolic dimension allowing for dynamic shapes on the vector inputs.
    #
    # Use float32 explicitly: the C executor (`example.c`) reads the output as
    # float32 and checks the sums, whereas the new API would otherwise default
    # to bfloat16 on an accelerator.
    input_type = TensorType(
        dtype=DType.float32,
        shape=("vector_width",),
        device=DeviceRef.from_device(device),
    )

    # Compile the module for the accelerator. The input types must match the
    # positional arguments of `forward`.
    compiled = model.compile(input_type, input_type)

    # Save the compiled artifact to a MEF file.
    compiled.export_mef("graph.mef")


def test_capi_v3() -> None:
    build_graph()

    path = os.environ["GRAPH_EXECUTOR"]
    os.execv(path, [path])


if __name__ == "__main__":
    test_capi_v3()
