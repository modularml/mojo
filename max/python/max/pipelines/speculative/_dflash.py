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
"""Reads the DFlash draft width from the draft HuggingFace config."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .config import SpeculativeConfig

logger = logging.getLogger("max.pipelines")


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass(frozen=True)
class DflashDraftHFConfig:
    """Parsed DFlash fields from a draft HuggingFace config."""

    mask_token_id: int
    target_layer_ids: list[int]
    block_size: int | None = None
    num_target_layers: int | None = None

    def draft_width(
        self, speculative: SpeculativeConfig, *, warn: bool = True
    ) -> int:
        """Returns the draft width, which is ``block_size - 1``.

        The drafter only works at its trained block size, so a width that
        disagrees is replaced with a warning. A checkpoint with no
        ``block_size`` needs an explicit width.
        """
        if self.block_size is None:
            if speculative.num_speculative_tokens is None:
                raise ValueError(
                    "The DFlash draft checkpoint declares no block_size; set"
                    " --num-speculative-tokens explicitly."
                )
            return speculative.num_speculative_tokens
        expected_spec = self.block_size - 1
        actual_spec = speculative.num_speculative_tokens
        if warn and actual_spec is not None and actual_spec != expected_spec:
            logger.warning(
                "DFlash draft was trained at block_size=%d, so"
                " num_speculative_tokens is being overridden from %d to"
                " %d. The DFlash draft's behavior is only defined at"
                " its trained block_size.",
                self.block_size,
                actual_spec,
                expected_spec,
            )
        return expected_spec


def parse_dflash_draft_hf_config(
    huggingface_config: Any,
) -> DflashDraftHFConfig:
    """Parses DFlash draft fields from a HuggingFace config object or dict."""
    dflash_cfg = _get(huggingface_config, "dflash_config", None)
    mask_token_id = _get(dflash_cfg, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError(
            "DFlash draft HF config is missing ``dflash_config.mask_token_id``."
        )
    target_layer_ids_raw = _get(dflash_cfg, "target_layer_ids", None)
    if target_layer_ids_raw is None:
        raise ValueError(
            "DFlash draft HF config is missing"
            " ``dflash_config.target_layer_ids``."
        )
    if not isinstance(target_layer_ids_raw, (list, tuple)):
        raise ValueError(
            "DFlash dflash_config.target_layer_ids must be a list of ints,"
            f" got {type(target_layer_ids_raw).__name__}."
        )
    target_layer_ids = [int(x) for x in target_layer_ids_raw]
    if not target_layer_ids:
        raise ValueError(
            "DFlash dflash_config.target_layer_ids must be non-empty."
        )

    raw_block_size = _get(
        dflash_cfg, "block_size", _get(huggingface_config, "block_size", None)
    )
    block_size = int(raw_block_size) if raw_block_size is not None else None
    raw_num_target_layers = _get(
        dflash_cfg,
        "num_target_layers",
        _get(huggingface_config, "num_target_layers", None),
    )
    num_target_layers = (
        int(raw_num_target_layers)
        if raw_num_target_layers is not None
        else None
    )

    return DflashDraftHFConfig(
        mask_token_id=int(mask_token_id),
        target_layer_ids=target_layer_ids,
        block_size=block_size,
        num_target_layers=num_target_layers,
    )


def dflash_draft_width(
    speculative: SpeculativeConfig,
    target_huggingface_config: Any,
    draft_huggingface_config: Any,
) -> int:
    """Returns the width the DFlash draft checkpoint was trained for."""
    del target_huggingface_config
    if draft_huggingface_config is None:
        raise ValueError("DFlash requires a draft model.")
    return parse_dflash_draft_hf_config(draft_huggingface_config).draft_width(
        speculative
    )
