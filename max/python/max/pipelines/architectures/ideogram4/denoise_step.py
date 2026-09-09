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
"""Compiled glue graphs for the Ideogram 4 denoise loop.

Following the FLUX.2 philosophy -- every numeric op lives inside a compiled
graph, the Python loop only orchestrates tensor-in / tensor-out executions --
each denoise step runs four compiled graphs:

* :class:`Ideogram4PackStep` -- ``concat([text_z_padding, z])`` for the
  conditional branch.
* the conditional ``Ideogram4Transformer2DModel`` (compiled directly).
* the unconditional ``Ideogram4Transformer2DModel`` (compiled directly).
* :class:`Ideogram4CombineStep` -- slice the conditional image-token velocity,
  apply the asymmetric dual-branch CFG combine
  ``v = gw * pos_v + (1 - gw) * neg_v``, and take the flow-match Euler step
  ``z = z + v * dt``.

The concat and the slice are deliberately kept in their *own* tiny graphs
rather than folded into the transformer graphs. Folding them in makes the
transformer's sequence dimension a symbolic sum / difference
(``num_text + num_image``), which triggers super-linear compiler cost (a single
transformer ballooned from ~1.3 s to >6 min). With the concat/slice isolated,
each transformer keeps the clean single ``seq`` contract it compiles in ~1.3 s,
and the attention-free pack/combine graphs compile near-instantly -- while the
loop still carries ``z`` as a realized float32 tensor with no eager glue.
"""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType


class Ideogram4PackStep(Module[..., Tensor]):
    """Concat the zero text padding and the image latents: the cond sequence.

    ``[text_z_padding (B, num_text, C)] ++ [z (B, num_image, C)]`` along the
    sequence axis. Kept attention-free so the resulting symbolic ``num_text +
    num_image`` sequence dim never reaches a transformer graph.
    """

    def __init__(
        self, in_channels: int, dtype: DType, device: DeviceRef
    ) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._dtype = dtype
        self._device = device

    def input_types(self) -> tuple[TensorType, ...]:
        c = self._in_channels
        d = self._dtype
        dev = self._device
        return (
            TensorType(d, shape=["batch", "num_text", c], device=dev),
            TensorType(d, shape=["batch", "num_image", c], device=dev),
        )

    def forward(self, text_z_padding: Tensor, z: Tensor) -> Tensor:
        return F.concat([text_z_padding, z], axis=1)


class Ideogram4CombineStep(Module[..., Tensor]):
    """Slice cond image velocity, CFG combine, and flow-match Euler step.

    The conditional transformer output spans the full ``[text][image]``
    sequence; its trailing ``num_image`` rows are the image-token velocity.
    Then ``v = gw * pos_v + (1 - gw) * neg_v`` and ``z = z + v * dt``, computed
    in float32 for numerical stability (the transformer branches already return
    float32) and cast back to the bf16 latent dtype on output. Attention-free,
    so the symbolic slice/rebind compiles near-instantly.
    """

    def __init__(
        self, in_channels: int, dtype: DType, device: DeviceRef
    ) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._dtype = dtype
        self._device = device

    def input_types(self) -> tuple[TensorType, ...]:
        c = self._in_channels
        d = self._dtype
        dev = self._device
        return (
            # z: current image-token latents (B, num_image, 128), bf16.
            TensorType(d, shape=["batch", "num_image", c], device=dev),
            # pos_out: full-sequence cond velocity (B, seq, 128), float32.
            TensorType(DType.float32, shape=["batch", "seq", c], device=dev),
            # neg_v: image-only uncond velocity (B, num_image, 128), float32.
            TensorType(
                DType.float32, shape=["batch", "num_image", c], device=dev
            ),
            # num_text_carrier: 1-D CPU tensor whose *length* is num_text.
            TensorType(
                DType.float32, shape=["num_text"], device=DeviceRef.CPU()
            ),
            # dt: step delta sigma[i+1]-sigma[i] (1,), float32.
            TensorType(DType.float32, shape=[1], device=dev),
            # guidance: CFG scale (1,), float32.
            TensorType(DType.float32, shape=[1], device=dev),
        )

    def forward(
        self,
        z: Tensor,
        pos_out: Tensor,
        neg_v: Tensor,
        num_text_carrier: Tensor,
        dt: Tensor,
        guidance: Tensor,
    ) -> Tensor:
        # Image-token velocity = trailing num_image rows of the cond output.
        # ``num_text`` comes from the carrier length; rebind the sliced
        # ``seq - num_text`` span back to z's ``num_image`` so the combine is
        # shape-consistent.
        num_text = num_text_carrier.shape[0]
        num_image = z.shape[1]
        pos_v = pos_out[:, num_text:, :]
        pos_v = F.rebind(pos_v, [pos_v.shape[0], num_image, pos_v.shape[2]])
        v = guidance * pos_v + (1.0 - guidance) * neg_v
        updated = F.cast(z, DType.float32) + dt * v
        return F.cast(updated, self._dtype)
