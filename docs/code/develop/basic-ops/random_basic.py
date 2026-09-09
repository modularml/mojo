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

uniform_tensor = F.uniform([3, 3], range=(0.0, 1.0))
normal_tensor = F.normal([3, 3], mean=0.0, std=1.0)

print("Uniform distribution:")
print(uniform_tensor)

print("\nNormal distribution:")
print(normal_tensor)
