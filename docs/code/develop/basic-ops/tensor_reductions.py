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

x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

sum_axis0 = x.sum(axis=0)  # sum each column across rows
sum_axis1 = x.sum(axis=1)  # sum each row across columns

print(f"Sum along axis 0: {sum_axis0}")
print(f"Sum along axis 1: {sum_axis1}")

mean_val = x.mean(axis=0)
max_val = x.max(axis=0)
min_val = F.min(x, axis=0)

print(f"Mean along axis 0: {mean_val}")
print(f"Max along axis 0: {max_val}")
print(f"Min along axis 0: {min_val}")
