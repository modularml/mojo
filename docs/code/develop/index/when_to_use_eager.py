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
# DOC: max/develop/eager-execution.mdx

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor


def debug_forward_pass(x: Tensor) -> Tensor:
    """Forward pass with intermediate inspection."""
    # Can print/inspect at any point
    print(f"Input: {x}")

    z = x * 2
    print(f"After multiply: {z}")

    h = F.relu(z)
    print(f"After ReLU: {h}")

    return h


x = Tensor([-1.0, 0.0, 1.0, 2.0], dtype=DType.float32, device=CPU())
result = debug_forward_pass(x)
