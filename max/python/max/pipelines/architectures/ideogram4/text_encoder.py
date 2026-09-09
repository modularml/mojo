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
"""Ideogram 4 text encoder.

Ideogram 4 uses Qwen3-VL-8B in *text-only* mode and concatenates the hidden
states of 13 evenly-spaced layers into a ``13 * 4096 = 53248`` feature vector
that conditions the DiT.

The dense Qwen3 text-encoder component already implements multi-layer
hidden-state selection and concatenation (see
:class:`Qwen3TextEncoderKleinModel`, which selects layers ``(9, 18, 27)``).
For text-only input, Qwen3-VL's 3D MRoPE collapses to standard 1D RoPE because
the temporal/height/width position indices are identical for sequential text
tokens, so the dense encoder reproduces the reference features. We therefore
reuse it directly and only pin the Ideogram activation-layer set.
"""

from __future__ import annotations

from typing import Any

from max.driver import Device
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding

from ..qwen3_modulev3.text_encoder.model import Qwen3TextEncoderModel
from .model_config import QWEN3_VL_ACTIVATION_LAYERS
from .weight_adapters import dequantize_fp8_state_dict


class Ideogram4TextEncoderModel(Qwen3TextEncoderModel):
    """Qwen3-VL text-only encoder emitting the Ideogram 13-layer concat.

    The reference taps the *output of decoder block* ``k`` for ``k`` in
    ``QWEN3_VL_ACTIVATION_LAYERS`` (0-based block index). The dense Qwen3
    encoder selects hidden states against the HuggingFace
    ``output_hidden_states`` contract, where ``hidden_states[0]`` is the token
    embedding and ``hidden_states[i + 1]`` is the output of block ``i``. The
    reference block index ``k`` therefore maps to HF index ``k + 1``, so the
    activation layers are shifted by ``+1`` here.

    The dense encoder concatenates the selected layers tap-major
    (``[block0_dims][block1_dims]...``); the Ideogram reference is interleaved
    (``[dim0_taps][dim1_taps]...``). The pipeline re-orders tap-major to
    interleaved after encoding (see ``Ideogram4Pipeline.prepare_inputs``).
    """

    default_hidden_state_layers = tuple(
        layer + 1 for layer in QWEN3_VL_ACTIVATION_LAYERS
    )

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        **kwargs: Any,
    ) -> None:
        # The Qwen3-VL ``text_config`` stores ``rope_theta`` under
        # ``rope_parameters`` (5e6); the dense Qwen3 encoder config reads a
        # top-level ``rope_theta`` and would otherwise default to 1e6, breaking
        # RoPE. Surface it so the config picks up the correct value.
        config = dict(config)
        text_config = dict(config.get("text_config", config))
        rope_params = text_config.get("rope_parameters") or {}
        if "rope_theta" not in text_config and "rope_theta" in rope_params:
            text_config["rope_theta"] = rope_params["rope_theta"]
            config["text_config"] = text_config

        # Weights are fp8 and dequantized to bf16 at load; run in bf16
        # regardless of the manifest's fp8 encoding.
        super().__init__(config, "bfloat16", devices, weights, **kwargs)

    def _state_dict(self) -> dict[str, Any]:
        """Adapt Qwen3-VL checkpoint keys, then dequantize fp8 weights.

        The base ``Qwen3TextEncoderModel._state_dict`` is written for the
        ``model.`` prefix used by Z-Image/Klein and would mangle the Qwen3-VL
        ``language_model.`` prefix (its ``str.replace("model.", "")`` also
        strikes the substring inside ``language_model.``). The Qwen3-VL
        checkpoint additionally ships a ``visual.`` tower (unused in text-only
        mode) and a final ``norm.weight`` (the encoder taps intermediate
        hidden states and has no final norm). Adapt keys directly here:
        drop the vision tower, strip the ``language_model.`` prefix, skip the
        final norm / lm head, then fp8-dequantize the remaining Linear weights.
        """
        raw: dict[str, Any] = {}
        for key, value in self.weights.items():
            if key.startswith("visual."):
                continue
            adapted_key = key.removeprefix("language_model.")
            if adapted_key in ("norm.weight", "lm_head.weight"):
                continue
            raw[adapted_key] = value.data()

        deq = dequantize_fp8_state_dict(raw)
        target = self.config.dtype
        for key, weight in deq.items():
            if (
                weight.dtype != target
                and weight.dtype.is_float()
                and target.is_float()
            ):
                deq[key] = weight.astype(target)
        return deq
