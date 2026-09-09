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
"""Drives the condition encoder and the DiT over a song's denoising windows.

The Euler loop never touches the host. Each step is two graph invocations: the
DiT over a batch of two, and a small update graph that combines the two guidance
branches, takes the Euler increment, and re-blends the overlap.

Two structural choices worth stating. Batching the conditional and
unconditional branches is exact rather than an approximation -- nothing in this
architecture mixes across the batch axis -- and it halves the launch count for
the step that dominates the runtime. And the blend logically precedes the DiT
call, but running it at the *end* of the previous update computes the same value,
which lets the blend live in the cheap graph: the update recompiles at every
window geometry, and the DiT takes minutes to compile.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Accelerator, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import CompiledModel, Module
from max.experimental.tensor import Tensor, default_dtype
from max.graph import TensorType
from max.graph.weights import SafetensorWeights, Weights
from max.pipelines.lib.bfloat16_utils import float32_array_to_buffer

from .components.condition_encoder import ConditionEncoder
from .components.transformer import Transformer
from .denoise import (
    BLEND_EPSILON,
    CARRY_BACK_LATENTS,
    GUIDANCE_SCALE,
    NUM_INFERENCE_STEPS,
    OVERLAP_LATENTS,
    Window,
    euler_schedule,
    plan,
)
from .model_config import ConditionEncoderConfig, TransformerConfig
from .weight_adapters import (
    convert_condition_encoder_state,
    convert_transformer_state,
)

_HOST = CPU()


def _widen(tensor: Tensor) -> npt.NDArray[np.float32]:
    """Read a device tensor to the host as float32.

    bfloat16 is float32 with the low mantissa bits cleared, so widening it is a
    shift rather than a conversion. numpy has no bfloat16, so the read goes
    through the driver buffer rather than through :meth:`Tensor.to_numpy`.
    """
    host = tensor.to(_HOST).driver_tensor
    if host.dtype == DType.float32:
        return host.to_numpy().copy()
    packed = host.view(DType.uint16).to_numpy()
    return (packed.astype(np.uint32) << 16).view(np.float32)


class GuidedTransformer(Transformer):
    """The DiT evaluating both classifier-free-guidance branches in one call.

    Nothing in the architecture mixes across the batch axis, so stacking the two
    branches is exact rather than an approximation, and it halves the launch
    count of the step that dominates the runtime.

    Duplicating the latents happens *inside* the graph so the Euler loop can keep
    them on the device between steps: the two branches differ only in their
    conditioning, which is staged once per window. Subclassing rather than
    wrapping keeps the parameter paths -- and so the checkpoint keys -- identical
    to :class:`Transformer`'s.
    """

    def input_types(self, latent_length: int) -> tuple[TensorType, ...]:
        """Graph inputs for one guided step over ``latent_length`` latents."""
        config = self.config
        return (
            TensorType(
                self.dtype,
                [1, config.in_channels, latent_length],
                device=self.device,
            ),
            # Scalar, and float32 regardless of the model's dtype: it indexes the
            # schedule rather than participating in the arithmetic.
            TensorType(DType.float32, [], device=self.device),
            TensorType(
                self.dtype,
                [2, latent_length, config.condition_dim],
                device=self.device,
            ),
        )

    def forward(
        self, latents: Tensor, time: Tensor, condition: Tensor
    ) -> Tensor:
        """Predicts both branches' velocities, stacked on the batch axis."""
        return super().forward(
            F.concat([latents, latents], axis=0),
            time.cast(self.dtype).broadcast_to([2]),
            condition,
        )


class EulerUpdate(Module[..., Tensor]):
    """Guidance, the Euler increment, and the overlap blend, in one graph.

    Parameterless: it exists as a module only so the Euler loop's cheap step
    compiles and runs the same way as the DiT. It recompiles per window geometry,
    which is why the blend lives here rather than beside the DiT -- see this
    module's docstring.
    """

    def __init__(
        self, channels: int, overlap: int, dtype: DType, device: Device
    ) -> None:
        self.channels = channels
        self.overlap = overlap
        self.dtype = dtype
        # Assigned rather than reached through `to()`: there are no parameters
        # to place, but `input_types` still needs the device.
        self.device = device

    def input_types(self, latent_length: int) -> tuple[TensorType, ...]:
        """Graph inputs for one update over ``latent_length`` latents."""
        # Zero-width inputs are not expressible, so an unblended window still
        # takes a single dummy frame it never reads.
        width = max(self.overlap, 1)

        def tensor(*shape: int, dtype: DType = self.dtype) -> TensorType:
            return TensorType(dtype, list(shape), device=self.device)

        return (
            tensor(1, self.channels, latent_length),
            tensor(2, self.channels, latent_length),
            # Guidance, increment and time: coefficients rather than
            # activations, so float32 regardless of the model's dtype.
            tensor(dtype=DType.float32),
            tensor(dtype=DType.float32),
            tensor(dtype=DType.float32),
            tensor(1, self.channels, width),
            tensor(1, self.channels, width),
        )

    def forward(
        self,
        latents: Tensor,
        velocity: Tensor,
        guidance: Tensor,
        increment: Tensor,
        time: Tensor,
        previous: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """Steps ``latents`` once and re-blends the overlap.

        With a zero ``velocity`` and ``increment`` this reduces to the blend
        alone, which is how a window's first blend is applied before the DiT
        ever sees the overlap.
        """
        conditional, unconditional = velocity[0:1], velocity[1:2]
        # The reference differences the two branches in the model's dtype and
        # applies the scale in float32, so the scale reaches the arithmetic as
        # itself rather than as its bfloat16 neighbour.
        shift = (conditional - unconditional).cast(DType.float32)
        guided = (unconditional.cast(DType.float32) + guidance * shift).cast(
            self.dtype
        )
        # The reference upcasts the sample for the increment and rounds the
        # result back, so the latents carry only bfloat16 between steps.
        stepped = (
            latents.cast(DType.float32) + increment * guided.cast(DType.float32)
        ).cast(self.dtype)
        if self.overlap:
            weight = time.cast(self.dtype)
            head = (1.0 - (1.0 - BLEND_EPSILON) * weight) * noise + (
                weight * previous
            )
            stepped = F.concat([head, stepped[:, :, self.overlap :]], axis=-1)
        return stepped


class DiffusionStage:
    """The condition encoder and DiT, and the Euler loop that drives them.

    Graphs are cached per window geometry, which for a song is one shape plus
    possibly a ragged last window.
    """

    def __init__(
        self,
        device: Accelerator,
        condition_weights: Weights,
        transformer_weights: Weights,
        *,
        condition_config: ConditionEncoderConfig | None = None,
        transformer_config: TransformerConfig | None = None,
        dtype: DType = DType.bfloat16,
    ) -> None:
        """Declares both modules, compiling neither.

        Args:
            device: Where the weights and activations live.
            condition_weights: The opened ``condition_encoder`` checkpoint.
            transformer_weights: The opened ``transformer`` checkpoint.
            condition_config: Defaults to the released checkpoint's values.
            transformer_config: Likewise.
            dtype: The DiT's dtype. The condition encoder is always float32:
                it is small, and its output is what the DiT's rounding is
                measured against.
        """
        self.device = device
        self.condition_weights = condition_weights
        self.transformer_weights = transformer_weights
        self.dtype = dtype
        self.condition_config = condition_config or ConditionEncoderConfig()
        self.transformer_config = transformer_config or TransformerConfig()
        # One module each, compiled at as many window geometries as the song
        # needs. Parameters are declared but never materialized under
        # `F.lazy()`; `compile` binds the real values.
        with F.lazy():
            with default_dtype(DType.float32):
                self.condition_encoder = ConditionEncoder(
                    self.condition_config
                ).to(device)
            with default_dtype(dtype):
                self.dit = GuidedTransformer(self.transformer_config).to(device)
        self._encoders: dict[int, CompiledModel[..., Any]] = {}
        self._transformers: dict[int, CompiledModel[..., Any]] = {}
        self._updates: dict[tuple[int, int], CompiledModel[..., Any]] = {}

    @classmethod
    def from_model_dir(
        cls,
        device: Accelerator,
        model_dir: str | Path,
        *,
        dtype: DType = DType.bfloat16,
    ) -> DiffusionStage:
        """Build from a checkpoint directory, for callers without a manifest.

        Raises:
            FileNotFoundError: If either component's subfolder holds no
                safetensors.
        """
        root = Path(model_dir)

        def weights(subfolder: str) -> SafetensorWeights:
            paths = sorted((root / subfolder).glob("*.safetensors"))
            if not paths:
                raise FileNotFoundError(
                    f"no safetensors under {root / subfolder}"
                )
            return SafetensorWeights(paths)

        return cls(
            device,
            weights("condition_encoder"),
            weights("transformer"),
            dtype=dtype,
        )

    # -- staging -----------------------------------------------------------

    def _stage(self, array: npt.NDArray[np.float32]) -> Tensor:
        """Move a float32 host array to the device in the model's dtype.

        Routed through the driver rather than a cast-only graph, which is what
        ``float32_array_to_buffer`` exists for.
        """
        return Tensor(
            storage=float32_array_to_buffer(
                array, dtype=self.dtype, device=self.device
            )
        )

    def _scalar(self, value: float) -> Tensor:
        return Tensor.from_dlpack(np.array(value, np.float32)).to(self.device)

    # -- graphs ------------------------------------------------------------

    def encoder(self, frames: int) -> CompiledModel[..., Any]:
        """The condition encoder for a window of ``frames`` frames."""
        if frames not in self._encoders:
            module = self.condition_encoder
            self._encoders[frames] = module.compile(
                *module.input_types(frames),
                weights=convert_condition_encoder_state(self.condition_weights),
            )
        return self._encoders[frames]

    def transformer(self, length: int) -> CompiledModel[..., Any]:
        """The DiT for one window, evaluating both guidance branches at once."""
        if length not in self._transformers:
            module = self.dit
            self._transformers[length] = module.compile(
                *module.input_types(length),
                weights=convert_transformer_state(
                    self.transformer_weights, self.dtype
                ),
            )
        return self._transformers[length]

    def update(self, length: int, overlap: int) -> CompiledModel[..., Any]:
        """Guidance, Euler increment, and overlap blend for one step."""
        key = (length, overlap)
        if key not in self._updates:
            module = EulerUpdate(
                self.transformer_config.in_channels,
                overlap,
                self.dtype,
                self.device,
            )
            self._updates[key] = module.compile(*module.input_types(length))
        return self._updates[key]

    # -- the two stages ----------------------------------------------------

    def conditions(
        self, frame_hiddens: npt.NDArray[np.float32]
    ) -> tuple[list[Window], list[npt.NDArray[np.float32]]]:
        """Encode each window's conditioning, splicing the overlap forward.

        A window's leading latents take their conditioning from the previous
        window rather than from a re-encode: consecutive windows' latent grids
        are offset by half a latent frame, so re-encoding would denoise the
        shared latents against slightly shifted conditioning.
        """
        windows = plan(
            frame_hiddens.shape[1], self.condition_config.latent_length
        )
        spliced: list[npt.NDArray[np.float32]] = []
        for window in windows:
            frames = frame_hiddens[:, window.frame_start : window.frame_end]
            condition = _widen(
                self.encoder(int(window.frame_end - window.frame_start))(
                    Tensor.from_dlpack(
                        np.ascontiguousarray(frames, np.float32)
                    ).to(self.device)
                )
            )
            if window.overlap:
                back = max(0, spliced[-1].shape[1] - CARRY_BACK_LATENTS)
                condition[:, : window.overlap] = spliced[-1][
                    :, back : back + window.overlap
                ]
            spliced.append(condition)
        return windows, spliced

    def denoise(
        self,
        windows: list[Window],
        conditions: list[npt.NDArray[np.float32]],
        noise: list[npt.NDArray[np.float32]],
        *,
        num_steps: int = NUM_INFERENCE_STEPS,
        guidance_scale: float = GUIDANCE_SCALE,
    ) -> list[npt.NDArray[np.float32]]:
        """Flow-match each window from its noise to latents.

        Args:
            windows: The plan returned by :meth:`conditions`.
            conditions: Per-window spliced conditioning.
            noise: Per-window initial noise, ``(1, channels, latents)``. Passed
                in rather than drawn, so a run is reproducible across frameworks.
            num_steps: Euler steps per window.
            guidance_scale: How far each step moves from the unconditional
                velocity towards the conditional one.

        Returns:
            One uncropped latent tensor per window.
        """
        times, increments = euler_schedule(num_steps)
        guidance = self._scalar(guidance_scale)
        chunks: list[npt.NDArray[np.float32]] = []
        carry: npt.NDArray[np.float32] | None = None

        for window, condition, seed in zip(
            windows, conditions, noise, strict=False
        ):
            overlap = window.overlap
            channels = seed.shape[1]
            dit = self.transformer(window.latents)
            update = self.update(window.latents, overlap)

            # Constant for the whole window: the unconditional branch is a zero
            # conditioning, not a re-encoded empty prompt.
            branches = self._stage(
                np.concatenate([condition, np.zeros_like(condition)], axis=0)
            )
            width = max(overlap, 1)
            previous = self._stage(
                carry[:, :, :overlap]
                if overlap and carry is not None
                else np.zeros((1, channels, width), np.float32)
            )
            prompt = self._stage(
                seed[:, :, :overlap]
                if overlap
                else np.zeros((1, channels, width), np.float32)
            )
            no_velocity = self._stage(
                np.zeros((2, channels, window.latents), np.float32)
            )

            latents = self._stage(seed)
            # Blend the overlap at the first time before the DiT ever sees it.
            latents = update(
                latents,
                no_velocity,
                guidance,
                self._scalar(0.0),
                self._scalar(times[0]),
                previous,
                prompt,
            )

            for index, time in enumerate(times):
                velocity = dit(latents, self._scalar(time), branches)
                # The blend belongs to the next step; on the last iteration it
                # runs against a time the DiT never sees, and the overlap is
                # overwritten below in any case.
                next_time = times[min(index + 1, num_steps - 1)]
                latents = update(
                    latents,
                    velocity,
                    guidance,
                    self._scalar(increments[index]),
                    self._scalar(next_time),
                    previous,
                    prompt,
                )

            chunk = _widen(latents)
            if overlap and carry is not None:
                chunk[:, :, :overlap] = carry[:, :, :overlap]
            back = max(0, window.latents - CARRY_BACK_LATENTS)
            carry = chunk[
                :, :, back : max(back, window.latents - OVERLAP_LATENTS)
            ].copy()
            chunks.append(chunk)
        return chunks
