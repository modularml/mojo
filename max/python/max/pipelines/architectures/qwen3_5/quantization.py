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
"""Per-module quantization scheme for Qwen3.5 checkpoints.

modelopt exports Qwen3.8-27B as ``quant_algo: "MIXED_PRECISION"``: the MLP
projections and ``lm_head`` are NVFP4 with group size 16, while the attention
and Gated-DeltaNet projections are per-tensor FP8. One ``QuantConfig`` cannot
describe both, and reading the FP8 payload as packed FP4 produces plausible
tensors and garbage output without raising.

The representation here is two ``QuantConfig`` objects plus the module sets
they apply to, rather than a per-module field on ``QuantConfig`` itself:
``Linear`` already accepts a per-instance config, so per-module granularity
needs no change to a type two dozen other architectures share.

Everything this module cannot account for raises. A checkpoint that does not
match the shape described above is a checkpoint MAX has not been taught to
read, and the failure has to be at load time, not in the logits.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.graph.weights import WeightData
from max.nn.float8_scale_stacking import can_use_fused_mlp
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

# NVFP4's block scale covers 16 elements of the logical (unpacked) input dim.
# Anything else misreads every scale, so it is checked rather than assumed.
NVFP4_GROUP_SIZE = 16

# Suffixes that make up one quantizable unit. Quantization is all-or-nothing
# within a unit: modelopt quantizes a whole MLP or a whole attention block.
_MLP_PROJECTIONS = frozenset({"gate_proj", "up_proj", "down_proj"})
_FULL_ATTN_PROJECTIONS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
_LINEAR_ATTN_PROJECTIONS = frozenset({"in_proj_qkv", "in_proj_z", "out_proj"})

_LAYER_MODULE = re.compile(
    r"^layers\.(\d+)\.(mlp|self_attn|linear_attn)\.(\w+)$"
)


@dataclass(frozen=True)
class Qwen3_5QuantScheme:
    """Which Qwen3.5 modules are quantized, and how."""

    mlp: QuantConfig | None
    """NVFP4 config for the MLP projections and ``lm_head``."""

    attn: QuantConfig | None
    """Per-tensor static FP8 config for the attention and GDN projections."""

    mlp_layers: frozenset[int]
    """Layer indices whose ``gate_proj``/``up_proj``/``down_proj`` are NVFP4."""

    attn_layers: frozenset[int]
    """Layer indices whose attention or GDN projections are FP8."""

    quantize_lm_head: bool
    """Whether ``lm_head`` is NVFP4."""

    compute_dtype: DType
    """Dtype of everything unquantized: activations, norms, embeddings, the
    GDN conv and its ``in_proj_a``/``in_proj_b``, and the state pools.

    Distinct from ``Qwen3_5Config.dtype``, which is the *storage* dtype of the
    quantized bases (``uint8`` for packed NVFP4) and is wrong for all of the
    above."""

    def mlp_config(self, layer_idx: int) -> QuantConfig | None:
        """Returns the config for layer ``layer_idx``'s MLP projections."""
        return self.mlp if layer_idx in self.mlp_layers else None

    def attn_config(self, layer_idx: int) -> QuantConfig | None:
        """Returns the config for layer ``layer_idx``'s attention projections."""
        return self.attn if layer_idx in self.attn_layers else None


def storage_dtype(
    quant_config: QuantConfig | None, compute_dtype: DType
) -> DType:
    """Returns the on-disk dtype of a module's weight under ``quant_config``.

    NVFP4 arrives as ``uint8`` with two 4-bit codes per byte; per-tensor FP8
    arrives as ``float8_e4m3fn``; an unquantized module keeps the compute
    dtype.
    """
    if quant_config is None:
        return compute_dtype
    return DType.uint8 if quant_config.is_fp4 else DType.float8_e4m3fn


def _canonical(module: str) -> str:
    """Strips the checkpoint's wrapper prefixes off a module path.

    The weight adapter loads ``model.language_model.layers.0.mlp.gate_proj``
    as ``layers.0.mlp.gate_proj``; the quantization metadata is not adapted,
    so it is normalized to the same namespace here.
    """
    return module.removeprefix("model.language_model.").removeprefix("model.")


def _algo_of(entry: object, module: str) -> tuple[str, int | None]:
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"quantized_layers['{module}'] is {type(entry).__name__}, expected "
            "a mapping with a 'quant_algo' key"
        )
    algo = entry.get("quant_algo")
    if not isinstance(algo, str):
        raise ValueError(
            f"quantized_layers['{module}'] has no 'quant_algo'; got {entry!r}"
        )
    group_size = entry.get("group_size")
    if group_size is not None and not isinstance(group_size, int):
        raise ValueError(
            f"quantized_layers['{module}'] has a non-integer group_size "
            f"{group_size!r}"
        )
    return algo, group_size


def _check_unit_is_uniform(
    kind: str,
    layer_idx: int,
    found: set[str],
    expected: frozenset[str],
) -> None:
    """A partially quantized MLP or attention block cannot be represented."""
    if found and found != expected:
        raise ValueError(
            f"layers.{layer_idx}.{kind} is partially quantized: "
            f"{sorted(found)} are listed in quantized_layers but "
            f"{sorted(expected - found)} are not. MAX quantizes a "
            f"{kind} block all-or-nothing."
        )


def parse_quant_scheme(
    hf_quant_config: Mapping[str, Any] | None,
    state_dict: Mapping[str, WeightData],
    num_hidden_layers: int,
) -> Qwen3_5QuantScheme | None:
    """Reads modelopt's per-module ``quantized_layers`` map into a scheme.

    Args:
        hf_quant_config: The checkpoint's resolved quantization config, from
            the *top-level* Hugging Face config or a standalone
            ``hf_quant_config.json``. ``None`` for an unquantized checkpoint.
        state_dict: The adapted checkpoint weights, keyed by MAX module path.
        num_hidden_layers: The decoder's layer count.

    Returns:
        The scheme, or ``None`` when the checkpoint declares no quantization.

    Raises:
        ValueError: If the checkpoint's quantization metadata describes
            anything this architecture has not been taught to read.
    """
    if not hf_quant_config:
        return None

    quant_algo = hf_quant_config.get("quant_algo")
    if quant_algo not in ("NVFP4", "FP8", "MIXED_PRECISION"):
        raise ValueError(
            f"Qwen3.5 cannot read quant_algo {quant_algo!r}. Supported: "
            "'NVFP4', 'FP8', 'MIXED_PRECISION'."
        )

    quantized_layers = hf_quant_config.get("quantized_layers")
    if not quantized_layers:
        raise ValueError(
            f"the checkpoint declares quant_algo {quant_algo!r} but carries no "
            "'quantized_layers' map, so which modules are quantized -- and in "
            "which format -- is unknowable. Both the top-level "
            "quantization_config and hf_quant_config.json were checked."
        )

    nvfp4_layers: dict[int, set[str]] = {}
    fp8_full_attn: dict[int, set[str]] = {}
    fp8_linear_attn: dict[int, set[str]] = {}
    quantize_lm_head = False

    for raw_module, entry in quantized_layers.items():
        module = _canonical(str(raw_module))
        algo, group_size = _algo_of(entry, module)

        if algo == "NVFP4" and group_size not in (None, NVFP4_GROUP_SIZE):
            raise ValueError(
                f"'{module}' declares NVFP4 group_size {group_size}; MAX's "
                f"NVFP4 path is fixed at {NVFP4_GROUP_SIZE}."
            )

        if module == "lm_head":
            if algo != "NVFP4":
                raise ValueError(
                    f"lm_head is quantized as {algo!r}; only NVFP4 is wired."
                )
            quantize_lm_head = True
            continue

        match = _LAYER_MODULE.match(module)
        if match is None:
            raise ValueError(
                f"quantized_layers names '{raw_module}', which is not a "
                "Qwen3.5 decoder projection or lm_head. Refusing to guess "
                "its precision."
            )
        layer_idx, subtree, projection = (
            int(match.group(1)),
            match.group(2),
            match.group(3),
        )
        if layer_idx >= num_hidden_layers:
            raise ValueError(
                f"quantized_layers names layer {layer_idx}, but the config "
                f"declares {num_hidden_layers} layers."
            )

        if subtree == "mlp":
            if algo != "NVFP4" or projection not in _MLP_PROJECTIONS:
                raise ValueError(
                    f"'{module}' is quantized as {algo!r}; Qwen3.5 MLP "
                    "projections are wired for NVFP4 only."
                )
            nvfp4_layers.setdefault(layer_idx, set()).add(projection)
        else:
            expected = (
                _FULL_ATTN_PROJECTIONS
                if subtree == "self_attn"
                else _LINEAR_ATTN_PROJECTIONS
            )
            if algo != "FP8" or projection not in expected:
                raise ValueError(
                    f"'{module}' is quantized as {algo!r}; Qwen3.5 "
                    f"{subtree} projections are wired for per-tensor FP8 only."
                )
            bucket = (
                fp8_full_attn if subtree == "self_attn" else fp8_linear_attn
            )
            bucket.setdefault(layer_idx, set()).add(projection)

    for layer_idx, found in nvfp4_layers.items():
        _check_unit_is_uniform("mlp", layer_idx, found, _MLP_PROJECTIONS)
    for layer_idx, found in fp8_full_attn.items():
        _check_unit_is_uniform(
            "self_attn", layer_idx, found, _FULL_ATTN_PROJECTIONS
        )
    for layer_idx, found in fp8_linear_attn.items():
        _check_unit_is_uniform(
            "linear_attn", layer_idx, found, _LINEAR_ATTN_PROJECTIONS
        )

    compute_dtype = _compute_dtype(state_dict)
    mlp_layer_set = frozenset(nvfp4_layers)
    attn_layer_set = frozenset(fp8_full_attn) | frozenset(fp8_linear_attn)

    return Qwen3_5QuantScheme(
        mlp=(
            _nvfp4_config(mlp_layer_set, compute_dtype, state_dict)
            if mlp_layer_set or quantize_lm_head
            else None
        ),
        attn=(
            _fp8_config(attn_layer_set, compute_dtype)
            if attn_layer_set
            else None
        ),
        mlp_layers=mlp_layer_set,
        attn_layers=attn_layer_set,
        quantize_lm_head=quantize_lm_head,
        compute_dtype=compute_dtype,
    )


def _compute_dtype(state_dict: Mapping[str, WeightData]) -> DType:
    """The dtype of the unquantized weights, read off the embedding table."""
    embed = state_dict.get("embed_tokens.weight")
    if embed is None:
        raise ValueError(
            "cannot determine the compute dtype: the checkpoint has no "
            "'embed_tokens.weight'."
        )
    return embed.dtype


def _nvfp4_config(
    layers: frozenset[int],
    compute_dtype: DType,
    state_dict: Mapping[str, WeightData],
) -> QuantConfig:
    """modelopt NVFP4: E4M3 block scales over 16 elements, fp32 global scales.

    The activation scale is static because the checkpoint calibrated it -- the
    per-block activation scales are still computed at runtime, but against
    this fixed global scale rather than a per-token amax.
    """
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.STATIC,
            dtype=DType.float32,
            block_size=(1, NVFP4_GROUP_SIZE),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e4m3fn,
            block_size=(1, NVFP4_GROUP_SIZE),
        ),
        mlp_quantized_layers=set(layers),
        attn_quantized_layers=set(),
        embedding_output_dtype=compute_dtype,
        bias_dtype=None,
        format=QuantFormat.NVFP4,
        # gate_proj and up_proj share a weight_scale_2 and an input_scale in
        # this export, so the fused gate/up matmul is exact. The helper
        # re-derives that from the checkpoint rather than trusting it.
        can_use_fused_mlp=can_use_fused_mlp(state_dict),
    )


def _fp8_config(layers: frozenset[int], compute_dtype: DType) -> QuantConfig:
    """Per-tensor static W8A8 FP8: one fp32 scalar per weight and activation.

    ``FBGEMM_FP8`` is the format enum MAX already uses for per-tensor FP8
    (see ``_parse_tensorwise_fp8_config``); it selects the static-scaled
    matmul, and nothing downstream reads the name.
    """
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.TENSOR,
            origin=ScaleOrigin.STATIC,
            dtype=DType.float32,
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.TENSOR,
            dtype=DType.float32,
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(layers),
        embedding_output_dtype=compute_dtype,
        bias_dtype=None,
        format=QuantFormat.FBGEMM_FP8,
    )
