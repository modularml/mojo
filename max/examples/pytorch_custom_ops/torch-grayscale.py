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

# DOC: max/develop/custom-kernels-pytorch.mdx

import max.experimental.torch
import numpy as np
import torch
from max.dtype import DType
from max.graph import ops


@max.experimental.torch.graph_op
def max_grayscale(pic: max.graph.TensorValue):  # noqa: ANN201
    scaled = pic.cast(DType.float32) * np.array([0.21, 0.71, 0.07])
    grayscaled = ops.sum(scaled, axis=-1).cast(pic.dtype)
    # max reductions don't remove the dimension, need to squeeze
    return ops.squeeze(grayscaled, axis=-1)


@torch.compile
def grayscale(pic: torch.Tensor):  # noqa: ANN201
    output = pic.new_empty(pic.shape[:-1])  # Remove color channel dimension
    max_grayscale(output, pic)  # Call as destination-passing style
    return output


# Define device first
device = "cuda" if torch.cuda.is_available() else "cpu"
img = (torch.rand(64, 64, 3, device=device) * 255).to(torch.uint8)
result = grayscale(img)
print(f"Input shape: {img.shape}")
print(f"Output shape: {result.shape}")
print("Grayscale conversion completed successfully!")
