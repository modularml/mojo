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

import logging
from typing import Any, ClassVar

from typing_extensions import override

from ..llama3_modulev3.batch_processor import Llama3ModuleV3BatchProcessor
from ..llama3_modulev3.model import Llama3Model
from .model_config import Olmo2Config
from .olmo2 import Olmo2

logger = logging.getLogger("max.pipelines")


class Olmo2Model(Llama3Model):
    """An Olmo2 pipeline model for text generation."""

    model_config_cls: ClassVar[type[Any]] = Olmo2Config
    batch_processor_cls: ClassVar[type[Llama3ModuleV3BatchProcessor]] = (
        Llama3ModuleV3BatchProcessor
    )
    config_class: type[Any] = Olmo2Config

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        model_config = Olmo2Config.initialize(
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
    def _instantiate_module(self, model_config: Any) -> Any:
        nn_model = Olmo2(model_config, self.kv_params)
        nn_model.to(self.devices[0])
        return nn_model
