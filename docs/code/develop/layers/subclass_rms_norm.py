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
from max.nn.norm.rms_norm import RMSNorm


class ScaledRMSNorm(RMSNorm):
    """RMSNorm that scales by (1 + weight), the Gemma 3 convention."""

    def __init__(self, dim: int, dtype: DType, eps: float = 1e-6) -> None:
        super().__init__(dim=dim, dtype=dtype, eps=eps, weight_offset=1.0)


layer = ScaledRMSNorm(576, DType.float32)
print(sorted(layer.state_dict().keys()))
