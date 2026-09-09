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

"""Mistral3 text encoder ComponentModel wrapper.

This module provides an eager ComponentModel wrapper for the Mistral3 text
encoder.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from max.driver import Device
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.modeling.base.component_model import ComponentModel
from max.pipelines.weights.weight_loading import (
    auto_cast_weights_from_env,
)
from max.profiler import traced

from .mistral3 import Mistral3TextEncoderTransformer
from .model_config import Mistral3TextEncoderConfig


class Mistral3TextEncoderModel(ComponentModel):
    """Mistral3 text encoder Module V3."""

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, encoding, devices, weights, **kwargs)
        self.config = Mistral3TextEncoderConfig.initialize_from_config(
            config,
            encoding,
            devices,
        )
        self.load_model()

    @traced(message="Mistral3TextEncoderModel.load_model")
    def load_model(self) -> Callable[..., Any]:
        """Load and compile the Mistral3 text encoder."""
        state_dict = {}
        for key, value in self.weights.items():
            if key.startswith("language_model.model."):
                adapted_key = key.removeprefix("language_model.model.")
            elif key.startswith("model."):
                adapted_key = key.removeprefix("model.")
            elif key.startswith(("embed_tokens.", "layers.")):
                adapted_key = key
            else:
                continue
            if not adapted_key.startswith(("embed_tokens.", "layers.")):
                continue
            state_dict[adapted_key] = value.data()

        with F.lazy():
            model = Mistral3TextEncoderTransformer(self.config)
            model.to(self.devices[0])

        self.model = model.compile(
            *model.input_types(),
            weights=state_dict,
            auto_cast=auto_cast_weights_from_env(),
        )
        return self.model

    def __call__(self, tokens: Tensor) -> Tensor:
        """Run the compiled text encoder."""
        return self.model(tokens)
