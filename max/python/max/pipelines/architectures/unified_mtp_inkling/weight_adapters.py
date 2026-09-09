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
"""Safetensors adapter for unified Inkling MTP: ``target.*`` and ``draft.*``."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from max.graph.weights import WeightData, Weights
from max.pipelines.lib.config import PipelineConfig
from transformers import AutoConfig

from ..inkling.model_config import parse_inkling_mtp_config
from ..inkling.weight_adapters import _SKIP, _TEXT, _VISION, _convert, _rename

logger = logging.getLogger("max.pipelines")

_MTP = "model.mtp."


def _resolved_mtp_depths(
    huggingface_config: AutoConfig | None,
    pipeline_config: PipelineConfig | None,
) -> int | None:
    """Depths the draft will build, or ``None`` when that can't be resolved."""
    if huggingface_config is None or pipeline_config is None:
        return None
    mtp = parse_inkling_mtp_config(huggingface_config)
    spec = pipeline_config.speculative
    if mtp is None or spec is None or not spec.num_speculative_tokens:
        # A width of 0 is a config error; don't turn it into a weight-name
        # failure here.
        return None
    return mtp.num_depths_for(spec)


def _beyond_depth_cap(name: str, n_depths: int | None) -> bool:
    """Whether ``name`` is an MTP weight for a depth the draft won't build."""
    if n_depths is None:
        return False
    parts = name[len(_MTP) :].split(".", 2)
    if len(parts) != 3 or parts[0] != "layers" or not parts[1].isdigit():
        # Not a per-depth weight (``chain_norm``), so never capped.
        return False
    return int(parts[1]) >= n_depths


def convert_with_mtp_state_dict(
    state_dict: Mapping[str, Weights],
    huggingface_config: AutoConfig | None,
    pipeline_config: PipelineConfig | None = None,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Maps an Inkling checkpoint onto the fused target+draft module names.

    Shared embedding and LM-head weights are emitted only under ``target.*``.
    Weights for MTP depths past the requested speculative width are dropped
    without being read.
    """
    del unused_kwargs
    n_depths = _resolved_mtp_depths(huggingface_config, pipeline_config)
    converted: dict[str, WeightData] = {}
    skipped: dict[str, int] = {}
    for name, value in state_dict.items():
        reason: str | None = None
        if name.endswith(".original_shape"):
            reason = "NVFP4 original_shape metadata"
        elif not name.startswith(_MTP):
            # _SKIP also lists the MTP prefix, whose weights load here.
            reason = next(
                (r for p, r in _SKIP.items() if name.startswith(p)), None
            )
        elif _beyond_depth_cap(name, n_depths):
            reason = f"MTP depths past the {n_depths} the draft builds"
        if reason is not None:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        data = value.data()
        if name.startswith(_TEXT):
            for key, weight in _convert(name, data).items():
                converted[f"target.{key}"] = _rename(weight, f"target.{key}")
            continue
        if name.startswith(_VISION):
            converted |= _convert(name, data)
            continue
        if name.startswith(_MTP):
            converted |= _convert_mtp(name, data)
            continue
        raise ValueError(
            f"unrecognized Inkling checkpoint weight {name!r}: expected "
            f"{_TEXT!r}, {_VISION!r}, {_MTP!r}, or one of "
            f"{sorted(set(_SKIP) - {_MTP})}"
        )
    for reason, count in sorted(skipped.items()):
        logger.info(
            f"Inkling MTP: skipped {count} checkpoint weights: {reason}"
        )
    return converted


def _convert_mtp(name: str, data: WeightData) -> dict[str, WeightData]:
    rest = name[len(_MTP) :]
    if rest.startswith("chain_norm."):
        key = f"draft.{rest}"
        return {key: _rename(data, key)}
    parts = rest.split(".", 2)
    if len(parts) != 3 or parts[0] != "layers":
        raise ValueError(f"unrecognized Inkling MTP weight {name!r}")
    _, depth_idx, suffix = parts
    prefix = f"draft.layers.{depth_idx}."
    if suffix.startswith("transformer_block."):
        block = suffix[len("transformer_block.") :]
        fake = f"{_TEXT}layers.{depth_idx}.{block}"
        out = {}
        for key, weight in _convert(fake, data).items():
            inner = key[len(f"layers.{depth_idx}.") :]
            dest = f"{prefix}decoder_layer.{inner}"
            out[dest] = _rename(weight, dest)
        return out
    dest = f"{prefix}{suffix}"
    return {dest: _rename(data, dest)}
