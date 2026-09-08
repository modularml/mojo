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

from max.experimental.tensor import Tensor

a = Tensor([1.0, 2.0, 3.0])
b = Tensor([4.0, 5.0, 6.0])

c = a + b  # Addition
d = a * b  # Element-wise multiplication

print(c)
print(d)
