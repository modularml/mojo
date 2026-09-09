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

"""Graph 1: Mistral3 text encoder component for Flux2Executor."""

from __future__ import annotations

from typing import Any

from max.driver import Buffer, load_devices
from max.engine import InferenceSession, Model
from max.graph import Graph
from max.graph import Module as GraphModule
from max.graph.weights import Weights, load_weights
from max.pipelines.lib.compiled_component import CompiledComponent
from max.pipelines.lib.config.model_config import (
    _resolve_component_encoding_and_weights,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.profiler import traced

# ModuleV2 imports only.
from ...mistral3.text_encoder.mistral3 import Mistral3TextEncoderTransformer
from ...mistral3.text_encoder.model_config import Mistral3TextEncoderConfig
from ..arch import FLUX2_TEXT_SEQ_LEN


class TextEncoder(CompiledComponent):
    """Graph 1: Mistral3 text encoder -> prompt_embeds.

    Encapsulates the full lifecycle: config extraction from the manifest,
    weight loading and key adaptation, Module construction, graph
    compilation, and runtime execution.

    Output shape:
        ``prompt_embeds``: ``(1, S, L*D)`` where L = number of hidden
        state layers, D = hidden_size.
    """

    _model: Model

    @traced(message="TextEncoder.__init__")
    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
        *,
        graphs_module: GraphModule | None = None,
    ) -> None:
        super().__init__(manifest, session, graphs_module=graphs_module)

        config = manifest["text_encoder"]
        config_dict = config.huggingface_config.to_dict()
        resolved_encoding, resolved_weight_path = (
            _resolve_component_encoding_and_weights(config)
        )
        encoding = resolved_encoding or "bfloat16"
        devices = load_devices(config.device_specs)

        mistral_config = Mistral3TextEncoderConfig.initialize_from_config(
            config_dict, encoding, devices
        )
        # Pad outputs to static sequence length expected by denoiser.
        mistral_config.output_seq_len = FLUX2_TEXT_SEQ_LEN

        paths = config.resolved_weight_paths(resolved_weight_path)
        weights = load_weights(paths)
        state_dict = self._adapt_state_dict(weights)

        module = Mistral3TextEncoderTransformer(mistral_config)
        module.load_state_dict(state_dict, weight_alignment=1, strict=True)

        with Graph(
            "text_encode",
            input_types=module.input_types(),
            module=self._graphs_module,
        ) as graph:
            outputs = module(*(v.tensor for v in graph.inputs))
            graph.output(outputs)

        self._load_graph(graph, weights_registry=module.state_dict())

    @traced(message="TextEncoder.__call__")
    def __call__(self, tokens: Buffer) -> Buffer:
        """Encode token IDs into prompt embeddings.

        Args:
            tokens: Token IDs, shape ``(S,)`` int64.

        Returns:
            ``prompt_embeds`` Buffer of shape ``(1, S, L*D)``.
        """
        result = self._model.execute(tokens)
        return result[0] if isinstance(result, (list, tuple)) else result

    @staticmethod
    def _adapt_state_dict(weights: Weights) -> dict[str, Any]:
        """Strip HuggingFace prefixes to match the Module hierarchy.

        Adapts keys like ``language_model.model.layers.0.…`` or
        ``model.layers.0.…`` to ``layers.0.…``, keeping only
        ``embed_tokens.*`` and ``layers.*`` entries.
        """
        state_dict: dict[str, Any] = {}
        for key, value in weights.items():
            if key.startswith("language_model.model."):
                adapted = key.removeprefix("language_model.model.")
            elif key.startswith("model."):
                adapted = key.removeprefix("model.")
            elif key.startswith(("embed_tokens.", "layers.")):
                adapted = key
            else:
                continue
            if not adapted.startswith(("embed_tokens.", "layers.")):
                continue
            state_dict[adapted] = value.data()
        return state_dict
