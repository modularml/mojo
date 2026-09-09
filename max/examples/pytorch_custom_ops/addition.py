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


from pathlib import Path

import torch
from max.experimental.torch import CustomOpLibrary

# Load the Mojo custom operations from the `operations` directory.
mojo_kernels = Path(__file__).parent / "operations"
op_library = CustomOpLibrary(mojo_kernels)

# Register a custom operation that adds a constant value of 10 to a tensor.
# The `value` parameter is a compile-time constant that we specify when
# registering this operation
add_const_op = op_library.add_constant_custom[{"value": 10}]


def add_const(x: torch.Tensor) -> torch.Tensor:
    """A wrapper PyTorch function that calls the Mojo custom operation."""
    result = torch.zeros_like(x)
    add_const_op(result, x)
    return result


if __name__ == "__main__":
    # Initialize a tensor of ones.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.ones(10, device=device)

    # Call the custom operation, and print the result.
    print(add_const(x))
