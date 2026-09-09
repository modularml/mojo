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
"""Heuristics for Kimi K2.x ModelOpt NVFP4 safetensor layouts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from max.dtype import DType
from max.nn.quant_config import QuantConfig


def _weight_entry_dtype(value: Any) -> DType | None:
    """Return ``value.dtype`` when present (e.g. :class:`~max.graph.weights.WeightData`)."""
    dt = getattr(value, "dtype", None)
    return dt if isinstance(dt, DType) else None


def infer_kimi_nvfp4_weight_flags(
    state_dict: Mapping[str, Any],
    *,
    first_k_dense_replace: int,
    quant_config: QuantConfig | None,
) -> tuple[DType | None, frozenset[int]]:
    """Infer NVFP4 vs BF16 regions from MAX-adapted Kimi checkpoint keys.

    Args:
        state_dict: Weight map after :func:`convert_kimik2_5_safetensor_state_dict`
            (``language_model.layers.*`` prefix).
        first_k_dense_replace: First layer index that uses MoE (HF ``text_config``).
        quant_config: Parsed quantization config, if any.

    Returns:
        ``(shared_experts_weight_dtype, dense_mlp_layers_without_quant)``.

        ``shared_experts_weight_dtype`` is ``None`` when shared experts match routed
        packed NVFP4 weights, or :class:`~max.dtype.DType.bfloat16` when shared experts
        omit ModelOpt scales (e.g. ``nvidia/Kimi-K2.6-NVFP4``). Apply via
        :func:`dataclasses.replace` on ``quant_config`` when not ``None``.

        ``dense_mlp_layers_without_quant`` lists dense-prefix layer indices whose
        ``gate_proj.weight`` is not packed NVFP4 (``dtype != uint8``) or is
        missing ``weight_scale`` while still ``uint8`` (e.g. layer ``0`` on
        ``nvidia/Kimi-K2.6-NVFP4``).
    """
    # Run layout checks for NVFP4/MXFP4 (``is_fp4``), and also when parsing
    # failed (``quant_config is None``) so dense BF16 tensors are still
    # detected from ``state_dict`` dtypes.
    if quant_config is not None and not quant_config.is_fp4:
        return None, frozenset()

    shared_weight_dtype: DType | None = None
    layer = first_k_dense_replace
    shared_w = (
        f"language_model.layers.{layer}.mlp.shared_experts.gate_proj.weight"
    )
    shared_scale = (
        f"language_model.layers.{layer}.mlp.shared_experts.gate_proj"
        ".weight_scale"
    )
    if shared_w in state_dict:
        wdt = _weight_entry_dtype(state_dict[shared_w])
        if wdt != DType.uint8 or shared_scale not in state_dict:
            shared_weight_dtype = DType.bfloat16

    dense_skip: set[int] = set()
    for i in range(max(first_k_dense_replace, 0)):
        gate_w = f"language_model.layers.{i}.mlp.gate_proj.weight"
        gate_scale = f"language_model.layers.{i}.mlp.gate_proj.weight_scale"
        if gate_w not in state_dict:
            continue
        wdt = _weight_entry_dtype(state_dict[gate_w])
        if wdt != DType.uint8 or gate_scale not in state_dict:
            dense_skip.add(i)

    return shared_weight_dtype, frozenset(dense_skip)
