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
"""Weight adapter for Nemotron-H checkpoints.

Maps HuggingFace ``NemotronHForCausalLM`` weight names to the MAX module names
and applies dtype/shape fixups:

* strip ``backbone.`` prefix; ``backbone.embeddings`` -> ``embed_tokens``;
  ``backbone.norm_f`` -> ``norm_f``; ``backbone.layers.N`` -> ``blocks.N``.
* conv1d weight ``[dim, 1, K]`` is kept 3-D (MAX expects depthwise [dim,1,K]).
* ``A_log`` / ``D`` / ``dt_bias`` cast to float32 (per-head scalars); the gated
  ``norm.weight`` cast to float32.
* MoE (Nemotron-3 hybrids): the router gate ``mixer.gate.weight`` ->
  ``mixer.gate.gate_score.weight`` (MAX ``MoEGate``); the
  ``e_score_correction_bias`` cast to float32; the routed-experts and
  shared-experts up/down projections map 1:1.
* FP8 (modelopt per-tensor static): F8_E4M3 weights are kept as-is; scale
  tensors (``weight_scale`` / ``input_scale``) cast to float32. Excluded
  modules (lm_head, the mamba in/out_proj at [11,16,23,31], all conv1d) stay
  bf16 — they have no scale tensors in the checkpoint. Attention q/k/v/o are
  bf16 (no scales) in the 4B, but per-tensor FP8 in the 8B Reasoning
  checkpoint; when FP8 they are dequantized to bf16 at load (widen E4M3->f32,
  apply the scalar ``weight_scale``) and their scale tensors (``weight_scale``
  / ``input_scale`` / KV-cache ``k_scale`` / ``v_scale``) are consumed/dropped.
"""

from __future__ import annotations

import re

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.graph.weights.weights import Shape
from max.pipelines.lib import PipelineConfig
from max.pipelines.weights._fp8 import fp8_e4m3fn_to_float32
from transformers import AutoConfig

# The mamba ``in_proj`` is ONE fused matmul in the nn.Module (matching the HF
# reference + vLLM), so its checkpoint tensors map 1:1: ``in_proj.weight`` stays
# F8_E4M3 via the generic FP8 path, and ``in_proj.weight_scale`` /
# ``in_proj.input_scale`` fall through to the generic scale->fp32 path. No
# row-slicing into separate projections is needed.

# Attention q/k/v/o projections. In the bf16 checkpoints (e.g. the 4B) these
# are bf16; in the FP8 8B Reasoning checkpoint they are per-tensor static FP8
# (E4M3 weight + scalar f32 ``weight_scale``) and are dequantized to bf16 at
# load (the attention ``Linear``s are bf16). q/k/v fuse into one
# ``qkv_proj.weight`` (concat order q, k, v along the out-dim) to match the
# single fused-QKV ``Linear`` in the nn.Module; ``o_proj`` stays standalone.
_ATTN_PROJ_RE = re.compile(r"^(blocks\.\d+\.mixer\.)([qkvo])_proj\.weight$")
# The attention projections' scale tensors: ``weight_scale`` (used to
# dequantize the FP8 weight, then dropped) plus ``input_scale`` and the
# KV-cache ``k_scale`` / ``v_scale`` (all unused once attention is bf16, so
# dropped). Only attention scales match here; mamba/MLP scales fall through to
# the generic FP8 scale->fp32 path.
_ATTN_SCALE_RE = re.compile(
    r"^(blocks\.\d+\.mixer\.)([qkvo])_proj\."
    r"(weight_scale|input_scale|k_scale|v_scale)$"
)

# Ordered prefix/name rewrites (applied in sequence; first match wins per
# group). The real FP8 checkpoint uses the ``backbone.`` prefix; the installed
# transformers ``NemotronHForCausalLM`` (used as the logit-verify reference)
# uses the ``model.`` prefix instead. Handle both.
_RENAMES: list[tuple[str, str]] = [
    ("backbone.embeddings.", "embed_tokens."),
    ("backbone.norm_f.", "norm_f."),
    ("backbone.layers.", "blocks."),
    ("backbone.", ""),
    ("model.embeddings.", "embed_tokens."),
    ("model.norm_f.", "norm_f."),
    ("model.layers.", "blocks."),
    ("model.", ""),
]

# Nearly all ``mixer.*`` names map 1:1 onto the MAX mixer Weights (the
# gated-norm weight's MAX Weight is declared ``norm.weight``, matching the
# checkpoint's ``mixer.norm.weight`` directly). The one exception is the MoE
# router gate: HF stores it at ``mixer.gate.weight`` while the MAX ``MoEGate``
# nests it under ``gate_score``. The MoE experts / shared_experts up/down
# projections and ``mixer.gate.e_score_correction_bias`` all map 1:1.
_MIXER_RENAMES: list[tuple[str, str]] = [
    (".mixer.gate.weight", ".mixer.gate.gate_score.weight"),
]

# fp32 params: per-head SSM scalars, the mamba gated-norm weight, and the MoE
# router correction bias. The block pre-norm (``blocks.{i}.norm.weight``) and
# final ``norm_f.weight`` stay bf16, so the gated-norm suffix is the specific
# ``.mixer.norm.weight``.
_FP32_SUFFIXES = (
    ".A_log",
    ".D",
    ".dt_bias",
    ".mixer.norm.weight",
    ".e_score_correction_bias",
)


def _concat_rows(parts: list[WeightData], new_name: str) -> WeightData:
    """Concatenate 2-D weights ``[out_i, in]`` along the out-dim (axis 0).

    All parts share the same ``in`` dim and dtype. bf16 is reinterpreted as
    uint16 for the numpy concat (numpy has no native bf16), since a row-concat
    of contiguous ``[out, in]`` matrices is a pure byte append; the result is
    reinterpreted back to the original dtype. No values change.
    """
    in_dim = int(parts[0].shape[1])
    dtype = parts[0].dtype
    total_out = sum(int(p.shape[0]) for p in parts)
    arrs = [
        np.from_dlpack(Buffer.from_dlpack(p.data).view(DType.uint16))
        if dtype == DType.bfloat16
        else np.from_dlpack(Buffer.from_dlpack(p.data))
        for p in parts
    ]
    cat = np.concatenate(arrs, axis=0)
    buf = Buffer.from_dlpack(cat).view(dtype=dtype, shape=(total_out, in_dim))
    return WeightData(
        data=buf,
        name=new_name,
        dtype=dtype,
        shape=Shape([total_out, in_dim]),
        quantization_encoding=parts[0].quantization_encoding,
    )


def _attn_proj_bf16(
    parts: dict[str, WeightData],
    scales: dict[str, WeightData],
    which: str,
    target_name: str,
) -> WeightData:
    """Return attention projection ``which`` as bf16, dequantizing if FP8.

    In the FP8 8B Reasoning checkpoint the attention q/k/v/o projections are
    per-tensor static FP8 (E4M3 weight + scalar f32 ``weight_scale``): widen
    E4M3 -> f32 via the shared host LUT (:func:`fp8_e4m3fn_to_float32`; the
    engine can't lower FP8 -> f32 on the host), apply the scalar scale, and cast
    to bf16 (the attention ``Linear``s are bf16). For checkpoints whose
    attention is already bf16 (e.g. the 4B) the weight passes through
    byte-identical and this is a no-op; the mamba/MLP FP8 ``Linear``s keep their
    native FP8 path untouched.
    """
    wd = parts[which]
    if wd.dtype != DType.float8_e4m3fn:
        return wd
    scale = scales.get(which)
    if scale is None:
        raise ValueError(
            f"missing weight_scale for FP8 attention weight '{target_name}'"
        )
    scale_f32 = np.from_dlpack(scale.astype(DType.float32).data).astype(
        np.float32
    )
    if scale_f32.size != 1:
        raise ValueError(
            f"attention FP8 weight '{target_name}' expects a per-tensor scalar "
            f"weight_scale, got shape {tuple(scale_f32.shape)}"
        )
    deq = np.ascontiguousarray(
        fp8_e4m3fn_to_float32(wd) * scale_f32.reshape(-1)[0]
    )
    return WeightData.from_numpy(deq, target_name).astype(DType.bfloat16)


def convert_nemotron_h_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Convert a Nemotron-H checkpoint to MAX module weight names.

    The mamba ``in_proj`` is ONE fused matmul in the nn.Module, so its
    checkpoint tensors map 1:1 (``in_proj.weight`` -> generic FP8 path;
    ``in_proj.weight_scale`` / ``in_proj.input_scale`` -> generic scale->fp32).
    The attention q/k/v weights are concatenated into one fused
    ``qkv_proj.weight`` (with ``o_proj`` standalone); on the FP8 8B Reasoning
    checkpoint the attention q/k/v/o projections are per-tensor FP8 and are
    dequantized to bf16 here (their scale tensors are consumed/dropped), while
    the mamba/MLP FP8 Linears keep their native FP8 path.
    """
    new_state_dict: dict[str, WeightData] = {}
    # Per-attention-layer q/k/v/o weights + their weight_scales, buffered for
    # FP8->bf16 dequant (8B) and q/k/v fusion into qkv_proj.
    attn_parts: dict[str, dict[str, WeightData]] = {}
    attn_scales: dict[str, dict[str, WeightData]] = {}
    for name, value in state_dict.items():
        max_name = name
        for before, after in _RENAMES:
            if before in max_name:
                max_name = max_name.replace(before, after)
        for before, after in _MIXER_RENAMES:
            max_name = max_name.replace(before, after)

        weight_data = value.data()

        # Buffer attention q/k/v/o weights for FP8->bf16 dequant + q/k/v fusion.
        am = _ATTN_PROJ_RE.match(max_name)
        if am is not None:
            attn_parts.setdefault(am.group(1), {})[am.group(2)] = weight_data
            continue

        # Buffer the attention weight_scale (to dequantize the FP8 weight) and
        # drop the other attention scales (input_scale + KV-cache k/v_scale),
        # all unused once attention is bf16. No-op for bf16-attention (4B)
        # checkpoints, which carry no attention scales; mamba/MLP scales fall
        # through to the generic FP8 scale->fp32 path below.
        sm = _ATTN_SCALE_RE.match(max_name)
        if sm is not None:
            if sm.group(3) == "weight_scale":
                attn_scales.setdefault(sm.group(1), {})[sm.group(2)] = (
                    weight_data
                )
            continue

        # Scale tensors -> float32 (FP8 kernels require f32 scales). The fused
        # mamba ``in_proj`` scales pass through here 1:1.
        if max_name.endswith(
            ("weight_scale", "input_scale", "weight_scale_inv")
        ):
            if max_name.endswith("weight_scale_inv"):
                max_name = max_name[: -len("weight_scale_inv")] + "weight_scale"
            weight_data = weight_data.astype(DType.float32)
            new_state_dict[max_name] = weight_data
            continue

        # FP8 weights stay F8_E4M3.
        if weight_data.dtype == DType.float8_e4m3fn:
            new_state_dict[max_name] = weight_data
            continue

        # Per-head scalars + gated norm weight -> float32.
        if max_name.endswith(_FP32_SUFFIXES):
            weight_data = weight_data.astype(DType.float32)

        # Router gate score: the FP8 (30B-A3B) checkpoint stores gate.weight as
        # f32, but the MAX MoEGate gate_score Linear is bf16 (routing math still
        # runs in f32 inside NemotronHMoEGate). Cast to the declared dtype; the
        # bf16 checkpoint's router is already bf16 so this is a no-op there.
        if (
            max_name.endswith(".mixer.gate.gate_score.weight")
            and weight_data.dtype == DType.float32
        ):
            weight_data = weight_data.astype(DType.bfloat16)

        # conv1d weight: keep [dim, 1, K]. HF stores it as [dim, 1, K] already
        # (a Conv1d depthwise weight); if it ever comes in 2-D, expand.
        if max_name.endswith(".conv1d.weight") and len(weight_data.shape) == 2:
            d0, d1 = weight_data.shape
            buf = Buffer.from_dlpack(weight_data.data).view(
                dtype=weight_data.dtype, shape=(int(d0), 1, int(d1))
            )
            weight_data = WeightData(
                data=buf,
                name=weight_data.name,
                dtype=weight_data.dtype,
                shape=Shape([d0, 1, d1]),
                quantization_encoding=weight_data.quantization_encoding,
            )

        new_state_dict[max_name] = weight_data

    # Emit the fused qkv_proj.weight (concat q,k,v) + o_proj.weight per
    # attention layer, dequantizing FP8 projections to bf16 (8B) or passing
    # bf16 through (4B). Attention scale tensors are consumed here or dropped
    # above.
    for prefix, parts in attn_parts.items():
        scales = attn_scales.get(prefix, {})
        fused_name = f"{prefix}qkv_proj.weight"
        new_state_dict[fused_name] = _concat_rows(
            [
                _attn_proj_bf16(parts, scales, "q", f"{prefix}q_proj.weight"),
                _attn_proj_bf16(parts, scales, "k", f"{prefix}k_proj.weight"),
                _attn_proj_bf16(parts, scales, "v", f"{prefix}v_proj.weight"),
            ],
            fused_name,
        )
        o_name = f"{prefix}o_proj.weight"
        new_state_dict[o_name] = _attn_proj_bf16(parts, scales, "o", o_name)

    return new_state_dict
