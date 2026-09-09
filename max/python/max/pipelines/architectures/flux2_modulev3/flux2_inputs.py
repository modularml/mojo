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
"""Input struct for the Flux2 ModuleV3 executor.

Mirrors :class:`~max.pipelines.architectures.flux2.Flux2ExecutorInputs`
directly: noise is pre-generated on CPU in ``prepare_inputs`` using
``np.random.RandomState(seed).standard_normal()`` so V3 inherits bit-
identical noise from V2.  Earlier revisions of this struct exposed a
``seed`` scalar so the graph could sample noise in-graph via
``ops.random``, but MAX's in-graph RNG diverges from numpy's
``RandomState`` for the same seed, which broke V2/V3 accuracy parity.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

from max.driver import Buffer, Device
from max.experimental.tensor import Tensor
from max.pipelines.modeling.base import TensorStruct
from typing_extensions import Self


@dataclass(frozen=True)
class Flux2ModuleV3Inputs(TensorStruct):
    """Structured inputs for the Flux2 ModuleV3 executor.

    Mirrors :class:`Flux2ExecutorInputs` directly: ``latents`` carries
    the patchified + packed initial noise tensor.  ``prepare_inputs``
    populates it from ``np.random.RandomState(seed).standard_normal()``
    so V3 inherits bit-identical noise from V2.
    """

    # -- Core (always present) ------------------------------------------------

    tokens: Buffer
    """Token IDs for the text encoder, shape ``(S,)``."""

    text_ids: Buffer
    """Text position IDs for the transformer, shape ``(1, S, 4)`` int64."""

    latents: Buffer
    """Pre-generated packed initial noise on the transformer device,
    shape ``(B, image_seq, num_channels)`` in the model dtype.  Sourced
    from :meth:`PixelContext.latents` (or fallback CPU sampling) and
    patchified + packed in :meth:`FLUXModule.prepare_inputs`."""

    latent_image_ids: Buffer
    """Latent positional identifiers, shape ``(B, seq, 4)`` int64."""

    timesteps: Buffer
    """Precomputed timesteps from the sigma schedule, shape ``(num_steps,)``
    (model dtype)."""

    dts: Buffer
    """Precomputed step deltas ``sigma[i+1] - sigma[i]``, shape
    ``(num_steps,)`` (float32)."""

    guidance: Buffer
    """Guidance scale broadcast tensor, shape ``(B,)``."""

    image_seq_len: Buffer
    """Packed image sequence length as a 1-element int64 tensor."""

    h_carrier: Buffer
    """Shape carrier of length ``packed_h``; content is never read."""

    w_carrier: Buffer
    """Shape carrier of length ``packed_w``; content is never read."""

    height: Buffer
    """Output image height in pixels as a 1-element int64 tensor."""

    width: Buffer
    """Output image width in pixels as a 1-element int64 tensor."""

    num_inference_steps: Buffer
    """Number of denoising steps as a 1-element int64 tensor."""

    num_images_per_prompt: Buffer
    """Number of images to generate per prompt as a 1-element int64 tensor."""

    # -- Image-conditioning input ---------------------------------------------

    input_image: Buffer
    """Input image, shape ``(H, W, C)`` uint8.  Always populated by
    :meth:`FLUXModule.prepare_inputs`: a real image for image-to-image
    requests, or a ``(0, 0, 3)`` placeholder for text-to-image so the
    VAE encoder can run unconditionally inside the compiled graph."""

    # -- Device transfer -------------------------------------------------------

    _CPU_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "num_inference_steps",
            "num_images_per_prompt",
            "height",
            "width",
            "image_seq_len",
            "h_carrier",
            "w_carrier",
        }
    )

    def to(self, device: Device) -> Self:
        """Transfer GPU-bound tensors to *device*, keeping metadata on CPU."""
        updates: dict[str, Any] = {}
        for f in fields(self):
            if f.name in self._CPU_FIELDS:
                continue
            val = getattr(self, f.name)
            if isinstance(val, (Tensor, Buffer)):
                updates[f.name] = val.to(device)
        return replace(self, **updates)
