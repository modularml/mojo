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

x = Tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]])

relu_output = F.relu(x)
sigmoid_output = F.sigmoid(x)
tanh_output = F.tanh(x)

print(f"ReLU: {relu_output}")
print(f"Sigmoid: {sigmoid_output}")
print(f"Tanh: {tanh_output}")
