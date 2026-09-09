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
"""Projects autoregressive frame hidden states onto the Flow-VAE latent timeline.

Each generated frame carries eight hidden states -- one from the language model
and one per residual codebook step. This mixes them with learned softmax weights,
projects 4096 to 2048 channels through a 3-tap convolution, and resamples from the
25 Hz frame rate to the 86.13 Hz latent rate.

Runs in float32. The reference pipeline runs it in bfloat16 only because it has to
share a card with everything else; the projection's weights are 0.10 GiB, so the
accuracy is nearly free here.
"""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import TensorType

from ..layers.conv import ShiftedConv1d
from ..model_config import ConditionEncoderConfig


class ConditionEncoder(Module[[Tensor], Tensor]):
    """Mixes per-layer frame hidden states and resamples them to latent frames."""

    layer_weight_logits: Tensor
    layer_scale: Tensor

    def __init__(self, config: ConditionEncoderConfig) -> None:
        self.config = config
        self.layer_weight_logits = Tensor.zeros([config.num_condition_layers])
        self.layer_scale = Tensor.zeros([1])
        # Channels-last, so the reference's two transposes around the convolution
        # disappear and the resample becomes a gather on the time axis.
        self.proj = ShiftedConv1d(
            kernel_size=3,
            in_channels=config.condition_hidden_dim,
            out_channels=config.out_dim,
        )

    @property
    def dtype(self) -> DType:
        """The dtype the encoder computes in, taken from its parameters."""
        return self.layer_scale.dtype

    def input_types(self, frames: int) -> tuple[TensorType, ...]:
        """The graph input for a window of ``frames`` autoregressive frames."""
        config = self.config
        return (
            TensorType(
                self.dtype,
                [
                    1,
                    frames,
                    config.num_condition_layers * config.condition_hidden_dim,
                ],
                device=self.device,
            ),
        )

    def forward(self, frame_hiddens: Tensor) -> Tensor:
        """Builds the latent-aligned conditioning sequence.

        Args:
            frame_hiddens: Shape
                ``(batch, frames, num_condition_layers * condition_hidden_dim)``,
                with the layer index outermost on the last axis.

        Returns:
            Shape ``(batch, latent_length, out_dim)``.
        """
        config = self.config
        batch, frames = frame_hiddens.shape[0], frame_hiddens.shape[1]
        x = frame_hiddens.cast(self.dtype).reshape(
            [
                batch,
                frames,
                config.num_condition_layers,
                config.condition_hidden_dim,
            ]
        )
        weights = F.softmax(self.layer_weight_logits).reshape(
            [1, 1, config.num_condition_layers, 1]
        )
        # F.sum keeps the reduced axis, hence the squeeze.
        mixed = F.squeeze(F.sum(x * weights, axis=2), 2)
        mixed = self.layer_scale * mixed

        return self._resample(self.proj(mixed), int(frames))

    def _resample(self, x: Tensor, frames: int) -> Tensor:
        """Nearest-neighbor resample along time, to the latent frame rate.

        Torch's ``interpolate(mode="nearest")`` takes source index
        ``floor(destination * in / out)``, which is a plain gather. Both lengths
        are known when the graph is built, so the index vector folds.
        """
        latent_length = self.config.latent_length(frames)
        if latent_length == frames:
            return x
        indices = F.floor_div(
            F.arange(0, latent_length, dtype=DType.int64, device=self.device)
            * frames,
            latent_length,
        )
        return F.gather(x, indices, axis=1)
