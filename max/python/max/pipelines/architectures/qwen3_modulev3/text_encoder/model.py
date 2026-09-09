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

"""Qwen3 text encoder ComponentModel wrapper.

This module provides a ComponentModel wrapper for Qwen3 text encoder.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.modeling.base.component_model import ComponentModel
from max.pipelines.modeling.dataprocessing.causal_attention_mask import (
    causal_attention_mask_with_token_mask,
)
from max.pipelines.weights.weight_loading import (
    auto_cast_weights_from_env,
)

from .model_config import Qwen3TextEncoderConfig
from .qwen3 import Qwen3TextEncoderTransformer
from .weight_adapters import QWEN3_TEXT_ENCODER_SAFETENSOR_MAP


class Qwen3TextEncoderModel(ComponentModel):
    """Qwen3 text encoder ComponentModel wrapper."""

    default_hidden_state_layers: tuple[int, ...] | None = None

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        **kwargs: Any,
    ) -> None:
        """Initialize Qwen3TextEncoderModel.

        Args:
            config: Configuration dictionary from model config file.
            encoding: Supported encoding for the model.
            devices: List of devices to use.
            weights: Model weights.
            **kwargs: Additional keyword arguments forwarded to ComponentModel.
        """
        super().__init__(config, encoding, devices, weights, **kwargs)
        self.config = Qwen3TextEncoderConfig.initialize_from_config(
            config,
            encoding,
            devices,
        )
        self.config.hidden_state_layers = self._resolve_hidden_state_layers()
        self.load_model()

    def _resolve_hidden_state_layers(self) -> list[int]:
        raw_layers = list(self.config.hidden_state_layers)
        if not raw_layers:
            if self.default_hidden_state_layers is not None:
                raw_layers = list(self.default_hidden_state_layers)
            else:
                raw_layers = list(range(1, self.config.num_hidden_layers + 1))
        return self._normalize_hidden_state_layers(
            raw_layers,
            self.config.num_hidden_layers,
        )

    @staticmethod
    def _normalize_hidden_state_layers(
        layers: list[int], num_hidden_layers: int
    ) -> list[int]:
        """Normalize HF-style hidden-state indices.

        Contract:
        - 0 = token embeddings
        - i (1 <= i < num_hidden_layers) = output after transformer block i-1
        - num_hidden_layers = final normalized hidden state
        - negative indices follow standard Python indexing over the available
          hidden-states tuple of length ``num_hidden_layers + 1``
        """
        normalized: list[int] = []
        seen: set[int] = set()
        total_hidden_states = num_hidden_layers + 1
        for layer in layers:
            idx = int(layer)
            if idx < 0:
                idx += total_hidden_states
            if idx < 0 or idx >= total_hidden_states:
                raise ValueError(
                    "Invalid `hidden_state_layers` index "
                    f"{layer} for available_hidden_states="
                    f"{total_hidden_states} (num_hidden_layers={num_hidden_layers})."
                )
            if idx not in seen:
                normalized.append(idx)
                seen.add(idx)

        if not normalized:
            raise ValueError("`hidden_state_layers` cannot be empty.")

        return normalized

    def _state_dict(self) -> dict[str, Any]:
        state_dict = {}
        for key, value in self.weights.items():
            adapted_key = key
            for before, after in QWEN3_TEXT_ENCODER_SAFETENSOR_MAP.items():
                adapted_key = adapted_key.replace(before, after)
            # The text-encoder module uses local names without language_model prefix.
            adapted_key = adapted_key.removeprefix("language_model.")
            # Skip checkpoint keys the encoder-only module doesn't use.
            if adapted_key in ("norm.weight", "lm_head.weight"):
                continue
            state_dict[adapted_key] = value.data()
        return state_dict

    def load_model(self) -> Callable[..., Any]:
        """Load and compile the Qwen3 text encoder.

        Returns:
            Compiled model callable.
        """
        state_dict = self._state_dict()

        with F.lazy():
            model = Qwen3TextEncoderTransformer(self.config)
            model.to(self.devices[0])

        self.model = model.compile(
            *model.input_types(),
            weights=state_dict,
            auto_cast=auto_cast_weights_from_env(),
        )
        return self.model

    @staticmethod
    def attention_bias_from_attention_mask_array(
        attention_mask: np.ndarray,
        *,
        expected_seq_len: int | None = None,
    ) -> np.ndarray:
        additive_mask = causal_attention_mask_with_token_mask(
            [0],
            attention_mask,
        )
        # TODO: Lift this batch_size=1 restriction if the Klein text-encoder
        # path needs batched prompt-mask handling.
        if additive_mask.shape[0] != 1:
            raise ValueError(
                f"batch size must be 1, got {additive_mask.shape[0]}."
            )
        if (
            expected_seq_len is not None
            and additive_mask.shape[1] != expected_seq_len
        ):
            raise ValueError(
                "seq_len must match tokens "
                f"({additive_mask.shape[1]} != {expected_seq_len})."
            )
        return additive_mask[:, np.newaxis, :, :].astype(np.float32, copy=False)

    def __call__(
        self,
        tokens: Tensor,
        attention_mask: npt.ArrayLike | None = None,
        *,
        hidden_state_index: int | None = None,
    ):
        if tokens.rank == 2:
            if int(tokens.shape[0]) != 1:
                raise ValueError(
                    "Qwen3TextEncoderModel expects batch_size=1 for 2D token input."
                )
            tokens = tokens[0]

        if attention_mask is not None:
            attention_mask_np = np.asarray(attention_mask)
            attention_bias_np = self.attention_bias_from_attention_mask_array(
                attention_mask_np,
                expected_seq_len=int(tokens.shape[0]),
            )
        else:
            attention_bias_np = self.attention_bias_from_attention_mask_array(
                np.ones((int(tokens.shape[0]),), dtype=np.bool_),
                expected_seq_len=int(tokens.shape[0]),
            )

        attention_bias = Tensor(
            storage=Buffer.from_numpy(attention_bias_np).to(self.devices[0])
        )

        outputs = self.model(tokens, attention_bias)

        if hidden_state_index is None:
            if isinstance(outputs, tuple) and len(outputs) == 1:
                return outputs[0]
            return outputs

        if not isinstance(outputs, tuple):
            raise ValueError(
                "`hidden_state_index` requires model outputs to be tuple/list "
                f"of hidden states, got {type(outputs).__name__}."
            )

        num_layers = len(outputs)
        if hidden_state_index < -num_layers or hidden_state_index >= num_layers:
            raise ValueError(
                f"`hidden_state_index` out of range: {hidden_state_index}. "
                f"Valid range is [{-num_layers}, {num_layers - 1}]."
            )

        return outputs[hidden_state_index]


class Qwen3TextEncoderKleinModel(Qwen3TextEncoderModel):
    """Qwen3 text encoder tuned for Flux2 Klein prompt layers."""

    default_hidden_state_layers = (9, 18, 27)


class Qwen3TextEncoderZImageModel(Qwen3TextEncoderModel):
    """Qwen3 text encoder tuned for Z-Image prompt layers."""

    default_hidden_state_layers = (-2,)
