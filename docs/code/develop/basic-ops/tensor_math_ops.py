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
# DOC: max/develop/basic-ops.mdx

import max.experimental.functional as F
from max.experimental.tensor import Tensor

x = Tensor([1.0, -4.0, 9.0, -16.0])

absolute = abs(x)
power = x**2
square_root = F.sqrt(abs(x))

print(f"Absolute value: {absolute}")
print(f"Power (x**2): {power}")
print(f"Square root: {square_root}")
