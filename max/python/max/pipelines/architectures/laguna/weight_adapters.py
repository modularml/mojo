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
"""Weight adapter for Laguna-M.1-NVFP4 (``poolside/Laguna-M.1-NVFP4``).

The checkpoint is compressed-tensors NVFP4: MLP/MoE ``gate_proj``/``up_proj``/
``down_proj`` (dense layer 0 + the routed experts + the shared expert) are
4-bit, each quantized Linear carrying four tensors:

.. code-block:: text

    weight_packed        packed FP4 (uint8, two fp4 per byte)
    weight_scale         per-group (16) FP8-e4m3 scale
    weight_global_scale  FP32 global scale
    input_global_scale   FP32 activation scale

MAX's ``MoEQuantized`` / ``Linear(quant_config=...)`` declare each expert's
projection as a per-expert quantized Linear expecting ``.weight`` (packed),
``.weight_scale``, ``.weight_scale_2`` (global), ``.input_scale``. So the NVFP4
mapping is a pure suffix rename:

.. code-block:: text

    .weight_packed        -> .weight
    .weight_global_scale  -> .weight_scale_2
    .input_global_scale   -> .input_scale
    .weight_scale         kept

Attention projections (q/k/v/o/g_proj) are in the quant ``ignore`` list and ship
bf16 ``.weight`` (passed through). The checkpoint's static FP8 KV-cache scales
``self_attn.{k,v}_scale`` are DROPPED: MAX computes KV-cache scales dynamically
at runtime and never consumes checkpoint static scales (same as deepseekV3/Kimi),
so a strict load would otherwise reject them as unexpected keys. Laguna structural
renames (router gate, expert correction bias, shared-expert pluralization) match
the bf16 donor.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

# Checkpoint -> MAX weight name mapping, applied in order with ``str.replace``
# semantics (``dict`` preserves insertion order). Order matters: do the NVFP4
# suffix renames before the structural ones so neither re-matches the other.
LAGUNA_SAFETENSOR_MAP: dict[str, str] = {
    "model.": "",
    # NVFP4 compressed-tensors suffixes -> MAX quantized-Linear suffixes.
    ".weight_packed": ".weight",
    ".weight_global_scale": ".weight_scale_2",
    ".input_global_scale": ".input_scale",
    # Laguna structural renames (same as the bf16 donor).
    ".mlp.gate.weight": ".mlp.gate.gate_score.weight",
    ".mlp.experts.e_score_correction_bias": ".mlp.gate.e_score_correction_bias",
    ".mlp.shared_expert.": ".mlp.shared_experts.",
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Converts Laguna-M.1-NVFP4 safetensor weights to MAX names.

    Renames the compressed-tensors NVFP4 suffixes to the per-expert quantized
    Linear names MAX's ``MoEQuantized`` expects, applies the Laguna structural
    renames, and reinterprets uint8 per-group scales as ``float8_e4m3fn``.
    """
    del huggingface_config, pipeline_config
    out: dict[str, WeightData] = {}
    for hf_name, value in state_dict.items():
        # FP8 KV-cache scales: the graph uses a bf16 KV cache, so these are
        # unused (strict load would reject them as unexpected). Drop them.
        if hf_name.endswith((".k_scale", ".v_scale")):
            continue
        max_name = hf_name
        for before, after in LAGUNA_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)
        data = value.data()
        # NVFP4 GLOBAL scales: compressed-tensors stores the *reciprocal* of
        # modelopt's convention. MAX's NVFP4 matmul uses weight_scale_2 and
        # input_scale as MULTIPLICATIVE dequant factors (modelopt:
        # weight_scale_2 = amax/(448*6), input_scale = act_amax/448 — small),
        # but compressed-tensors stores weight_global_scale = (448*6)/amax and
        # input_global_scale = 448/act_amax (large, e.g. 5504 / 1184). Passing
        # them through unchanged inflates every weight ~5504x → gibberish.
        # Reciprocate the two F32 scalar globals. The per-group ``weight_scale``
        # (e4m3) is identical in both conventions — do NOT touch it.
        if max_name.endswith((".weight_scale_2", ".input_scale")):
            arr = np.from_dlpack(data)
            # np.reciprocal on a 0-d array returns a numpy *scalar* (no
            # __dlpack__); force a 0-d ndarray so to_buffer() works downstream.
            recip = np.asarray(
                np.reciprocal(arr.astype(np.float32)), dtype=np.float32
            ).reshape(arr.shape)
            data = WeightData.from_numpy(recip, max_name)
        # Per-group NVFP4 scales are float8_e4m3fn; some exporters store them
        # as raw uint8. Reinterpret so the kernel reads the right dtype. (This
        # is the NVFP4 analogue of the MXFP4 path, where per-group scales are
        # instead reinterpreted as float8_e8m0fnu — see minimax_m2.)
        elif max_name.endswith(".weight_scale") and data.dtype == DType.uint8:
            data = dataclasses.replace(data, dtype=DType.float8_e4m3fn)
        out[max_name] = data
    return out
