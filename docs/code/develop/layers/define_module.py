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
# DOC: max/develop/layers.mdx

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight
from max.nn import Module


class LearnedScale(Module):
    """A layer that owns one learnable per-channel scale weight."""

    def __init__(self, dim: int, dtype: DType = DType.float32) -> None:
        super().__init__()
        self.weight = Weight("weight", dtype, [dim], device=DeviceRef.CPU())

    def __call__(self, x: TensorValue) -> TensorValue:
        weight = self.weight.cast(x.dtype)
        if x.device:
            weight = weight.to(x.device)
        return x * weight


layer = LearnedScale(576)
print(sorted(layer.state_dict().keys()))
