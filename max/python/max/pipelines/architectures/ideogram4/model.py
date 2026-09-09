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
"""Component holding one Ideogram 4 DiT's adapted weights.

The cond and uncond branches are *not* compiled standalone. Instead each
``Ideogram4TransformerModel`` loads + adapts its own ``transformer/`` (or
``unconditional_transformer/``) checkpoint into a bf16 state dict, and the
pipeline fuses both DiTs together with the asymmetric-CFG combine and the
Euler step into a single compiled graph (see ``denoise_step.py``). Building
one graph for the whole step keeps every numeric op (concat, both transformer
forwards, CFG combine, ``z += v * dt``) inside the compiled region, matching
the FLUX.2 ``DenoiseStep`` best practice and avoiding eager-mode dtype
re-tracing between steps.
"""

from __future__ import annotations

from typing import Any

from max.driver import Device
from max.dtype import DType
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.modeling.base.component_model import ComponentModel
from max.profiler import traced

from .model_config import Ideogram4Config
from .weight_adapters import (
    FP8_SCALE_SUFFIX,
    convert_ideogram4_transformer_state_dict,
)

# FP8 weights stay packed and their float32 rowwise scales stay float32; only
# the genuinely bf16/float32 tensors are cast to the compute dtype.
_FP8_DTYPES = (DType.float8_e4m3fn, DType.float8_e4m3fnuz)


class Ideogram4TransformerModel(ComponentModel):
    """Loads + adapts one Ideogram 4 DiT checkpoint (cond or uncond branch)."""

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        # The repo ships fp8 weights, so the manifest reports an fp8 encoding,
        # but we dequantize fp8 -> bf16 at load (see ``weight_adapters``) and
        # run the DiT in bf16. Pin the compute dtype to bf16 regardless of the
        # manifest encoding.
        self.config = Ideogram4Config.initialize_from_config(
            config, "bfloat16", devices
        )
        self.state_dict: dict[str, Any] = {}
        self.load_model()

    @traced(message="Ideogram4TransformerModel.load_model")
    def load_model(self) -> None:
        """Adapt the checkpoint into a bf16 state dict (no standalone compile).

        The actual graph compilation happens once in the pipeline, where both
        branches are fused into a single denoise-step graph.
        """
        target_dtype = self.config.dtype
        # Dequantize FP8 (-> float32) and pass other tensors through.
        raw: dict[str, Any] = {
            key: value.data() for key, value in self.weights.items()
        }
        state_dict = convert_ideogram4_transformer_state_dict(raw)
        # Cast every float tensor to the compute dtype, but leave native-FP8
        # weights packed and their rowwise scales in float32.
        for key, weight in state_dict.items():
            if key.endswith(FP8_SCALE_SUFFIX) or weight.dtype in _FP8_DTYPES:
                continue
            if (
                weight.dtype != target_dtype
                and weight.dtype.is_float()
                and target_dtype.is_float()
            ):
                state_dict[key] = weight.astype(target_dtype)
        self.state_dict = state_dict
