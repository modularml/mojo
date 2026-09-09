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
"""Mistral3 text encoder sub-``Module`` for the FLUX.2 ModuleV3 executor."""

from __future__ import annotations

from typing import Any

from max.driver import DeviceSpec, load_devices
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.pipelines.architectures.flux2.arch import FLUX2_TEXT_SEQ_LEN
from max.pipelines.architectures.mistral3_modulev3.text_encoder.mistral3 import (
    Mistral3TextEncoderTransformer,
)
from max.pipelines.architectures.mistral3_modulev3.text_encoder.model_config import (
    Mistral3TextEncoderConfig,
)
from max.pipelines.lib.weight_loader import WeightLoader, rename
from max.pipelines.modeling.config_enums import SupportedEncoding


class TextEncoder(Module[[Tensor], Tensor]):
    """ModuleV3 text encoder: Mistral3 ``tokens`` -> fused prompt embeddings.

    Built directly from a HuggingFace config dict, quantization encoding,
    and device specs.  Constructs the underlying
    :class:`Mistral3TextEncoderTransformer` and places itself on
    ``device_specs[0]`` so the caller only needs to wrap construction
    in :func:`max.experimental.functional.lazy` and drive compilation.

    Weight loading is decoupled from construction.  :meth:`adapt_loader`
    wraps a source :class:`~max.pipelines.lib.weight_loader.WeightLoader`
    (typically composed by the
    :func:`~max.pipelines.lib.weight_loader.adapt_module_loader` walker
    over a parent :class:`~max.experimental.nn.Module`) so the encoder's
    parameter queries resolve against the HF Mistral3 checkpoint
    namespace.
    """

    def __init__(
        self,
        huggingface_config: dict[str, Any],
        quantization_encoding: SupportedEncoding | None,
        device_specs: list[DeviceSpec],
    ) -> None:
        encoding: SupportedEncoding = quantization_encoding or "bfloat16"
        devices = load_devices(device_specs)

        mistral_config = Mistral3TextEncoderConfig.initialize_from_config(
            huggingface_config, encoding, devices
        )
        # FLUX.2 trains the joint-attention with a static padded text
        # stream of ``FLUX2_TEXT_SEQ_LEN`` tokens; without this, image
        # tokens index the wrong rows of the joint rope table and the
        # generated image comes out spatially scrambled.  Mirrors V2's
        # ``flux2/components/text_encoder.py``.
        mistral_config.output_seq_len = FLUX2_TEXT_SEQ_LEN

        self.encoder = Mistral3TextEncoderTransformer(mistral_config)
        self.to(devices[0])

    def forward(self, tokens: Tensor) -> Tensor:
        """Encode token IDs into fused prompt embeddings.

        Returns shape ``(1, S, L*D)`` where ``L`` is the number of selected
        hidden-state layers and ``D`` is the hidden size.
        """
        return self.encoder(tokens)

    def input_types(self) -> tuple[TensorType, ...]:
        """Input tensor types for compilation; delegates to the encoder."""
        return self.encoder.input_types()

    @staticmethod
    def adapt_loader(loader: WeightLoader) -> WeightLoader:
        """Translate this Module's parameter queries to HF Mistral3 keys.

        The encoder asks for ``encoder.*`` names (e.g.
        ``"encoder.layers.0.input_layernorm.weight"``); the source
        checkpoint stores them under one of several HF prefixes
        (``language_model.model.``, ``model.``, or none).  Each query is
        translated lazily -- the matching source prefix is discovered
        from the loader's namespace -- so only the parameters the Module
        actually declares are ever resolved, and checkpoint-only keys
        (for example ``lm_head``) are never queried.
        """

        def to_source(name: str) -> str:
            inner = name.removeprefix("encoder.")
            for src in ("language_model.model.", "model.", ""):
                candidate = f"{src}{inner}"
                if any(k == candidate for k in loader.keys(src)):
                    return candidate
            raise KeyError(name)

        return rename(loader, to_source)
