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

"""Qwen2.5-VL encoder ComponentModel wrapper (module v2)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from max.driver import Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import Weights
from max.nn.embedding import Embedding
from max.nn.layer import Module
from max.pipelines.architectures.llama3.weight_adapters import (
    LLAMA_SAFETENSOR_MAPPING as QWEN_SAFETENSOR_MAP,
)
from max.pipelines.lib import SupportedEncoding
from max.pipelines.modeling.base.component_model import ComponentModel

from .model_config import Qwen25VLTextEncoderConfig
from .qwen25vl import Qwen25VLTextEncoderTransformer


class _EmbedOnly(Module):
    """Token embedding only (module v2)."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.embed_tokens = Embedding(
            vocab_size,
            hidden_size,
            dtype=dtype,
            device=device,
        )

    def __call__(self, tokens: Any) -> Any:
        return self.embed_tokens(tokens)


class Qwen25VLEncoderModel(ComponentModel):
    """Qwen2.5-VL language-side encoder ComponentModel wrapper (module v2)."""

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        session: InferenceSession | None = None,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.config = Qwen25VLTextEncoderConfig.generate(
            config,
            encoding,
            devices,
        )
        self.session = session
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        embed_state: dict[str, Any] = {}
        transform_state: dict[str, Any] = {}

        for key, value in self.weights.items():
            wd = value.data()

            # Normalize floating-point weights to bf16
            if wd.dtype.is_float() and not wd.dtype.is_float8():
                is_scale = key.endswith((".weight_scale", ".input_scale"))
                if not is_scale:
                    wd = wd.astype(DType.bfloat16)

            # Key mapping
            adapted_key = key
            if adapted_key.startswith("model.language_model."):
                adapted_key = adapted_key[len("model.language_model.") :]
            else:
                for before, after in QWEN_SAFETENSOR_MAP.items():
                    adapted_key = adapted_key.replace(before, after)

            # Skip vision weights
            if adapted_key.startswith(("visual.", "vision_encoder.")):
                continue

            # Strip "model." prefix
            adapted_key = adapted_key.removeprefix("model.")

            if adapted_key.startswith("embed_tokens."):
                embed_state[adapted_key] = wd
            elif adapted_key.startswith(("layers.", "norm.", "rope.")):
                transform_state[adapted_key] = wd

        lc = self.config
        device_ref = DeviceRef.from_device(self.devices[0])

        # --- Compile embed_tokens ---
        embed_model = _EmbedOnly(
            lc.vocab_size,
            lc.hidden_size,
            dtype=lc.dtype,
            device=device_ref,
        )
        embed_model.load_state_dict(
            embed_state, weight_alignment=1, strict=True
        )
        embed_input_types = [
            TensorType(DType.int64, shape=["total_seq_len"], device=device_ref),
        ]
        with Graph("qwen_te_embed", input_types=embed_input_types) as g:
            out = embed_model(*(v.tensor for v in g.inputs))
            g.output(out)

        session = self.session
        if session is None:
            session = InferenceSession(devices=self.devices)

        self._embed_model: Model = session.load(
            g,
            weights_registry=embed_model.state_dict(),
        )

        # --- Compile transformer layers + norm ---
        transform_model = Qwen25VLTextEncoderTransformer(lc)
        transform_model.load_state_dict(
            transform_state,
            weight_alignment=1,
            strict=True,
        )
        transform_input_types = [
            TensorType(
                lc.dtype,
                shape=["total_seq_len", lc.hidden_size],
                device=device_ref,
            ),
        ]
        with Graph("qwen_te_transform", input_types=transform_input_types) as g:
            out = transform_model(*(v.tensor for v in g.inputs))
            g.output(out)
        self._transform_model: Model = session.load(
            g,
            weights_registry=transform_model.state_dict(),
        )

        return self._embed_model

    def __call__(self, token_input: Any) -> tuple[Any]:
        """Run text encoder: embed_tokens → transformer → normed output.

        Accepts both Buffer (v2) and experimental Tensor (v3 compat).
        Returns a tuple wrapping the result in the same type as input.
        """
        # Extract Buffer from _Tensor if needed
        is_tensor = hasattr(token_input, "driver_tensor")
        buf = token_input.driver_tensor if is_tensor else token_input

        embed_result = self._embed_model.execute(buf)
        transform_result = self._transform_model.execute(embed_result[0])
        result_buf = transform_result[0]

        if is_tensor:
            from max.experimental.tensor import Tensor as _Tensor

            return (_Tensor(storage=result_buf),)

        return (result_buf,)
