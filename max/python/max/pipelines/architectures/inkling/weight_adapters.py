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
"""Inkling safetensors adapter: prefix strip, skip the audio tower, few
reshapes."""

from __future__ import annotations

import logging
import re

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph import Shape
from max.graph.weights import WeightData, Weights
from transformers import AutoConfig

logger = logging.getLogger("max.pipelines")

# Inkling ships raw input_amax without a precomputed input_scale, so we
# derive the scale ourselves from the NVFP4 format maxima.
_FP8_E4M3_MAX = 448.0
_FP4_E2M1_MAX = 6.0
_TEXT = "model.llm."
_VISION = "model.visual."
_SKIP = {
    "model.mtp.": "MTP draft heads, which this architecture does not implement",
    "model.audio.": "the audio tower, which this architecture does not serve",
}

VISION_PREFIX = "vision."
"""Marks the tower's weights, which compile into a graph of their own."""


def nvfp4_input_scale(input_amax: float) -> float:
    """``input_amax / (448 * 6)``: MAX's multiplicative NVFP4 input scale."""
    return input_amax / (_FP8_E4M3_MAX * _FP4_E2M1_MAX)


def _rename(data: WeightData, name: str) -> WeightData:
    return WeightData(data.data, name, data.dtype, data.shape)


def _as_bytes(data: WeightData) -> np.ndarray:
    *lead, cols = (int(d) for d in data.shape)
    buf = Buffer.from_dlpack(data.data)
    return np.from_dlpack(
        buf.view(DType.uint8, [*lead, cols * data.dtype.size_in_bytes])
    )


def _from_bytes(
    rows: np.ndarray, data: WeightData, shape: list[int], name: str
) -> WeightData:
    buf = Buffer.from_dlpack(np.ascontiguousarray(rows)).view(data.dtype, shape)
    return WeightData(buf, name, buf.dtype, Shape(buf.shape))


def _split_w13(data: WeightData, gate: str, up: str) -> dict[str, WeightData]:
    """De-interleave ``[g0, u0, g1, u1, ...]`` rows into gate and up."""
    rows = _as_bytes(data)
    out_rows = int(data.shape[-2]) // 2
    shape = [out_rows, int(data.shape[-1])]
    if rows.ndim == 3:
        # sinks: [S, 2ff, H] -> expert-major [S*ff, H] per half
        return {
            gate: _from_bytes(
                rows[:, 0::2, :],
                data,
                [rows.shape[0] * out_rows, shape[1]],
                gate,
            ),
            up: _from_bytes(
                rows[:, 1::2, :], data, [rows.shape[0] * out_rows, shape[1]], up
            ),
        }
    return {
        gate: _from_bytes(rows[0::2], data, shape, gate),
        up: _from_bytes(rows[1::2], data, shape, up),
    }


def _sink_down(data: WeightData, name: str) -> WeightData:
    """``[S, H, ff]`` -> ``[H, S*ff]``."""
    sinks, hidden, ff = (int(d) for d in data.shape)
    return _from_bytes(
        _as_bytes(data).transpose(1, 0, 2), data, [hidden, sinks * ff], name
    )


def _as_float32(data: WeightData) -> np.ndarray:
    """Widens a float weight to float32 on the host.

    Workaround: WeightData.astype is a no-op under virtual devices, and numpy
    has no bfloat16, so bf16 bits are shifted into the high half of an fp32.
    """
    buf = data.to_buffer()
    if data.dtype == DType.bfloat16:
        bits = buf.view(DType.uint16).to_numpy().astype(np.uint32)
        return (bits << 16).view(np.float32)
    return buf.to_numpy().astype(np.float32)


def _input_scale(data: WeightData, name: str) -> WeightData:
    amax = _as_float32(data).reshape(-1)
    # The graph declares a single input scale shared by every expert in a
    # stacked tensor; a checkpoint with per-expert amax values must not be
    # silently collapsed to expert 0's scale.
    if amax.size > 1 and not np.all(amax == amax[0]):
        raise ValueError(
            f"{name}: expected one input_amax shared by the whole tensor,"
            f" got {amax.size} elements with {len(np.unique(amax))} distinct"
            " values; per-expert input scales are not supported"
        )
    return WeightData.from_numpy(
        np.asarray([nvfp4_input_scale(float(amax[0]))], np.float32),
        name,
    )


def _convert(name: str, data: WeightData) -> dict[str, WeightData]:
    if name.startswith(_VISION):
        # The tower holds its linears and norms in LayerLists, so the
        # checkpoint's layers.linear_N and layers.norm_N become linears.N
        # and norms.N.
        target = VISION_PREFIX + re.sub(
            r"^layers\.(linear|norm)_(\d+)\.",
            r"\g<1>s.\g<2>.",
            name[len(_VISION) :],
        )
        return {target: _rename(data, target)}
    if not name.startswith(_TEXT):
        raise ValueError(
            f"unrecognized Inkling checkpoint weight {name!r}: expected "
            f"{_TEXT!r}, {_VISION!r}, or one of {sorted(_SKIP)}"
        )

    rest = name[len(_TEXT) :]
    idx = rest.rfind(".mlp.")
    prefix = rest[: idx + 5] if idx >= 0 else ""

    if rest.endswith("mlp.w13_dn.weight"):
        return _split_w13(
            data, f"{prefix}gate_proj.weight", f"{prefix}up_proj.weight"
        )
    if rest.endswith("mlp.w2_md.weight"):
        target = f"{prefix}down_proj.weight"
        return {target: _rename(data, target)}
    if rest.endswith("shared_experts.shared_w13_weight"):
        return _split_w13(
            data,
            f"{prefix}shared_experts.gate_proj.weight",
            f"{prefix}shared_experts.up_proj.weight",
        )
    if rest.endswith("shared_experts.shared_w2_weight"):
        target = f"{prefix}shared_experts.down_proj.weight"
        return {target: _sink_down(data, target)}
    if rest.endswith(".input_amax"):
        target = rest[: -len(".input_amax")] + ".input_scale"
        return {target: _input_scale(data, target)}
    # The shared quant parser detects modelopt NVFP4 by the weight_scale_2 name.
    for suffix, house in (
        (".scale2", ".weight_scale_2"),
        (".scale", ".weight_scale"),
    ):
        if rest.endswith(suffix):
            target = rest[: -len(suffix)] + house
            return {target: _rename(data, target)}
    if rest.endswith("global_scale"):
        return {rest: WeightData.from_numpy(_as_float32(data), rest)}
    return {rest: _rename(data, rest)}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Maps an Inkling safetensors checkpoint onto the graph's weight names."""
    del huggingface_config, unused_kwargs
    converted: dict[str, WeightData] = {}
    skipped: dict[str, int] = {}
    for name, value in state_dict.items():
        reason: str | None
        if name.endswith(".original_shape"):
            reason = "NVFP4 original_shape metadata"
        else:
            reason = next(
                (r for p, r in _SKIP.items() if name.startswith(p)), None
            )
        if reason is not None:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        converted |= _convert(name, value.data())
    for reason, count in sorted(skipped.items()):
        logger.info(f"Inkling: skipped {count} checkpoint weights: {reason}")
    return converted
