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
"""Llama4 (text-only) pipeline model."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from max.engine import InferenceSession
from max.graph import Graph
from max.pipelines.architectures.llama3.model import LlamaModelBase
from typing_extensions import override

from .llama4 import Llama4
from .model_config import Llama4Config


class Llama4Model(LlamaModelBase):
    """Llama4 text-only pipeline model.

    Reuses :class:`~max.pipelines.architectures.llama3.model.LlamaModelBase`
    for input preparation, execution, and KV-cache management, and overrides
    only graph construction to build the Llama4 ``nn.Module``.
    """

    model_config_cls: ClassVar[type[Llama4Config]] = Llama4Config
    norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm"
    attention_bias: bool = False

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> Llama4Config:
        model_config = Llama4Config.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
        )
        return model_config

    @override
    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: Llama4Config,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        if len(self.devices) > 1:
            raise ValueError(
                "Llama4 currently supports single-device execution only."
            )

        nn_model = Llama4(model_config)
        nn_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,
        )
        weights_registry = nn_model.state_dict()

        with Graph(
            "llama4",
            input_types=nn_model.input_types(self.kv_params),
        ) as graph:
            tokens, input_row_offsets, return_n_logits, *kv_cache_inputs = (
                graph.inputs
            )
            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)
            outputs = nn_model(
                tokens.tensor,
                kv_collections[0],
                return_n_logits.tensor,
                input_row_offsets.tensor,
            )
            graph.output(*outputs)

        return graph, weights_registry
