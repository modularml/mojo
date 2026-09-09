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

from __future__ import annotations

from typing import Any, Literal

from max.driver import Buffer
from max.engine import InferenceSession, Model
from max.graph import Graph
from typing_extensions import override

from ..llama3.model import LlamaModelBase
from .model_config import Olmo2Config
from .olmo2 import Olmo2


class Olmo2Model(LlamaModelBase):
    """OLMo2 pipeline model implementation."""

    model: Model
    """Compiled and initialized model ready for inference."""

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm"
    """Normalization layer."""

    attention_bias: bool = False
    """Whether to use attention bias."""

    state_dict: dict[str, Any]
    """Weights to load into the model."""

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> Olmo2Config:
        model_config = Olmo2Config.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
        )
        return model_config

    @override
    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: Olmo2Config,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        if len(self.devices) > 1:
            raise NotImplementedError("Multi-GPU OLMo2 is not implemented yet")

        nn_model = Olmo2(model_config)
        graph_inputs = nn_model.input_types(self.kv_params)

        nn_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
        )
        weights_registry = nn_model.state_dict()

        with Graph(
            "olmo2",
            input_types=graph_inputs,
        ) as graph:
            tokens, input_row_offsets, return_n_logits, *kv_cache_inputs = (
                graph.inputs
            )
            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)
            outputs = nn_model(
                tokens.tensor,
                kv_collections[0],
                input_row_offsets=input_row_offsets.tensor,
                return_n_logits=return_n_logits.tensor,
            )
            graph.output(*outputs)
            return graph, weights_registry
