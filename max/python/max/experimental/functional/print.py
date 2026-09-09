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

"""Debug print op that runs in both eager and graph modes."""

import builtins

from max.experimental.tensor import Tensor
from max.graph import TensorValue


def print(value: Tensor, name: str) -> None:
    """Prints a tensor to the console.

    Args:
        value: The tensor to print.
        name: The name of the tensor.
    """
    if value.real:
        # Realized tensors have concrete storage and no graph value to emit, so
        # print eagerly. Use ``builtins.print`` because this module shadows the
        # builtin with its own ``print``.
        builtins.print(f"Tensor {name}=", value)
        return
    if value.num_shards > 1:
        for shard in value.local_shards:
            TensorValue(shard).print(f"{name} ({shard.device})")
    else:
        TensorValue(value).print(name)
