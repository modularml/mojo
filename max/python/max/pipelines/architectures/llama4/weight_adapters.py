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
"""Weight adapters for Llama4 (text-only) safetensors checkpoints."""

from __future__ import annotations

import dataclasses
import re

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData, Weights

# Weights under these prefixes belong to the vision tower / multimodal
# projector and are skipped for the text-only model.
SKIP_PREFIXES = ("vision_model.", "multi_modal_projector.")

# Sub-module renames applied after stripping the model/language-model prefix.
# Order matters: the specific ``feed_forward.*`` rules run before the generic
# ``feed_forward.`` -> ``mlp.`` fallback (which catches dense-layer MLPs).
FEED_FORWARD_MAPPING = (
    ("feed_forward.router.", "mlp.gate.gate_score."),
    ("feed_forward.shared_expert.", "mlp.shared_experts."),
    ("feed_forward.experts.", "mlp.experts."),
    ("feed_forward.", "mlp."),
)

# Matches a split per-expert routed-expert weight/scale tensor name (after the
# ``language_model.``/``model.`` prefix has been stripped), e.g.
# ``layers.0.feed_forward.experts.3.gate_proj.weight_scale``.
_EXPERT_RE = re.compile(
    r"^layers\.(\d+)\.feed_forward\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale)$"
)


def _strip_prefix(name: str) -> str:
    """Removes the ``language_model.`` and/or ``model.`` checkpoint prefixes.

    Handles both the multimodal layout (``language_model.model.layers...`` and
    ``language_model.lm_head.weight``) and the text-only layout
    (``model.layers...``, ``lm_head.weight``). ``removeprefix`` is used rather
    than ``str.replace`` so the ``model.`` inside ``language_model.`` is never
    corrupted.
    """
    return name.removeprefix("language_model.").removeprefix("model.")


def _rename_feed_forward(max_name: str) -> str:
    for before, after in FEED_FORWARD_MAPPING:
        max_name = max_name.replace(before, after)
    return max_name


def _fp8_bytes(value: Weights) -> np.ndarray:
    """Views an FP8 weight's bytes as a uint8 array (e4m3 = 1 byte/elem).

    Stacking/concatenating/transposing FP8 tensors is pure byte movement, so it
    runs on the uint8 view; the caller restores the ``float8_e4m3fn`` dtype.
    """
    arr = np.from_dlpack(value.data().data)
    return arr if arr.dtype == np.uint8 else arr.view(np.uint8)


def _f32(value: Weights) -> np.ndarray:
    """Materializes a checkpoint scale as a float32 numpy array."""
    return np.from_dlpack(value.data().astype(DType.float32).data)


def _stack_fp8_experts(
    state_dict: dict[str, Weights],
) -> tuple[dict[str, WeightData], set[str]]:
    """Stacks split-per-expert FP8 routed experts into the fused MoE layout.

    The compressed-tensors FP8 Scout checkpoint stores each routed expert as a
    separate ``feed_forward.experts.{j}.{gate,up,down}_proj`` linear (FP8 e4m3
    weight ``[out, in]`` + per-output-channel ``weight_scale`` ``[out, 1]``).
    :class:`~max.nn.moe.stacked_moe.StackedMoE` instead consumes fused tensors:

      - ``mlp.experts.gate_up_proj``        FP8   ``[E, hidden, 2 * moe]``
      - ``mlp.experts.down_proj``           FP8   ``[E, moe, hidden]``
      - ``mlp.experts.gate_up_proj_scale``  f32   ``[E, 2 * moe, 1]``
      - ``mlp.experts.down_proj_scale``     f32   ``[E, hidden, 1]``

    The stacking is plain byte/array movement, done in numpy: FP8 weights are
    handled as their raw ``uint8`` bytes and reinterpreted as ``float8_e4m3fn``
    afterwards; scales are materialized as float32.

    Returns the fused ``WeightData`` plus the set of raw checkpoint keys that
    were consumed (so the caller skips them in the generic pass).
    """
    # layer -> proj -> expert_idx -> {"weight": uint8 array, "weight_scale": f32}
    grouped: dict[int, dict[str, dict[int, dict[str, np.ndarray]]]] = {}
    consumed: set[str] = set()
    for name, value in state_dict.items():
        if name.startswith(SKIP_PREFIXES):
            continue
        m = _EXPERT_RE.match(_strip_prefix(name))
        if m is None:
            continue
        layer, expert, proj, kind = (
            int(m.group(1)),
            int(m.group(2)),
            m.group(3),
            m.group(4),
        )
        grouped.setdefault(layer, {}).setdefault(proj, {}).setdefault(
            expert, {}
        )[kind] = _fp8_bytes(value) if kind == "weight" else _f32(value)
        consumed.add(name)

    fused: dict[str, WeightData] = {}
    for layer, projs in grouped.items():
        n_experts = len(projs["gate_proj"])
        order = list(range(n_experts))

        # gate_up weight: per expert cat([gate.T, up.T], axis=1) -> [hidden,2*moe]
        gate_up_w = np.ascontiguousarray(
            np.stack(
                [
                    np.concatenate(
                        [
                            projs["gate_proj"][e]["weight"].T,
                            projs["up_proj"][e]["weight"].T,
                        ],
                        axis=1,
                    )
                    for e in order
                ],
                axis=0,
            )
        )
        # gate_up scale: per expert cat([gate_scale, up_scale], axis=0)->[2*moe,1]
        gate_up_s = np.ascontiguousarray(
            np.stack(
                [
                    np.concatenate(
                        [
                            projs["gate_proj"][e]["weight_scale"],
                            projs["up_proj"][e]["weight_scale"],
                        ],
                        axis=0,
                    )
                    for e in order
                ],
                axis=0,
            )
        )
        # down weight: per expert down.T -> [moe, hidden]
        down_w = np.ascontiguousarray(
            np.stack([projs["down_proj"][e]["weight"].T for e in order], axis=0)
        )
        down_s = np.ascontiguousarray(
            np.stack(
                [projs["down_proj"][e]["weight_scale"] for e in order], axis=0
            )
        )

        prefix = f"layers.{layer}.mlp.experts."
        # The fused weights are stacked as raw uint8 bytes; reinterpret them as
        # e4m3 (1 byte/elem, so the uint8 shape is already the FP8 shape).
        fused[f"{prefix}gate_up_proj"] = dataclasses.replace(
            WeightData.from_numpy(gate_up_w, f"{prefix}gate_up_proj"),
            dtype=DType.float8_e4m3fn,
        )
        fused[f"{prefix}down_proj"] = dataclasses.replace(
            WeightData.from_numpy(down_w, f"{prefix}down_proj"),
            dtype=DType.float8_e4m3fn,
        )
        fused[f"{prefix}gate_up_proj_scale"] = WeightData.from_numpy(
            gate_up_s, f"{prefix}gate_up_proj_scale"
        )
        fused[f"{prefix}down_proj_scale"] = WeightData.from_numpy(
            down_s, f"{prefix}down_proj_scale"
        )
    return fused, consumed


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Maps Llama4 safetensors weight names to the MAX module names.

    Two checkpoint layouts are supported:

    - bf16 (e.g. ``unsloth/Llama-4-Scout-...``): routed experts are already
      fused (``feed_forward.experts.gate_up_proj``); only names are remapped.
    - compressed-tensors FP8-dynamic (e.g. ``RedHatAI/...-FP8-dynamic``): routed
      experts are split per-expert and are stacked into the fused FP8 layout;
      ``weight_scale`` tensors are cast to float32. Attention, router, lm_head
      and embeddings stay bf16 (they are not quantized in the checkpoint).
    """
    # FP8 checkpoints carry ``*.weight_scale`` tensors; bf16 ones do not.
    is_fp8 = any(name.endswith(".weight_scale") for name in state_dict)

    new_state_dict: dict[str, WeightData] = {}
    consumed: set[str] = set()
    if is_fp8:
        fused, consumed = _stack_fp8_experts(state_dict)
        new_state_dict.update(fused)

    for name, value in state_dict.items():
        if name.startswith(SKIP_PREFIXES) or name in consumed:
            continue
        max_name = _rename_feed_forward(_strip_prefix(name))
        weight_data = value.data()
        # FP8 kernels consume float32 scales; the checkpoint stores them in
        # bfloat16. Quantized weights themselves stay FP8.
        if is_fp8 and max_name.endswith(".weight_scale"):
            weight_data = weight_data.astype(DType.float32)
        new_state_dict[max_name] = weight_data
    return new_state_dict
