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

from max import nn
from max.driver import accelerator_count
from max.dtype import DType
from max.graph import DeviceRef, TensorValue

# Use the GPU when one is available. DeviceRef labels the weights and inputs
# inside the graph.
device = DeviceRef.GPU() if accelerator_count() > 0 else DeviceRef.CPU()


class NormalizedProjection(nn.Module):
    """A linear projection whose output is normalized with RMSNorm."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, DType.float32, device)
        self.norm = nn.RMSNorm(out_dim, dtype=DType.float32)

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.norm(self.proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.qk_proj = NormalizedProjection(dim, dim)  # your custom layer
        self.o_proj = nn.Linear(dim, dim, DType.float32, device)

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.o_proj(self.qk_proj(x))


class MiniDecoder(nn.Module):
    """Two decoder blocks and an output projection, all on the same device."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block_0 = DecoderBlock(dim)
        self.block_1 = DecoderBlock(dim)
        self.lm_head = nn.Linear(dim, dim, DType.float32, device)

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.lm_head(self.block_1(self.block_0(x)))


model = MiniDecoder(dim=512)

# The composed weight names mirror the attribute hierarchy.
print(sorted(model.state_dict().keys()))
