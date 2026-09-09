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
"""Strict weight-name audit against the REAL draft snapshot (CPU-cheap).

Every tensor name and shape comes from the safetensors header of
``RedHatAI/gemma-4-31B-it-speculator.dspark`` without loading data (only the
tiny int64 ``d2t`` table is materialized). The gates:

1. The checkpoint's names minus the explicit skip list exactly cover the
   draft module's expected weights — no unconsumed keys, no phantom skips.
2. ``merge_unified_state_dict`` applies exactly that skip list.
3. The weight-time guards (lm_head / markov_w2 rows, d2t length + image,
   fc in-features) pass on the real tensors and fail loudly on tampered
   ones — including the 12B-style full-vocab lm_head alias shape this
   checkpoint must NOT be treated as having.
"""

from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from huggingface_hub import hf_hub_download
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.embedding import Embedding
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.dspark_draft.dspark_speculators_draft import (
    DSparkSpeculatorsDraft,
)
from max.pipelines.architectures.speculators_common import (
    SKIPPED_DRAFT_KEYS,
    SKIPPED_DRAFT_PREFIXES,
    DSparkSpeculatorsDraftConfig,
    merge_unified_state_dict,
    validate_draft_checkpoint_weights,
)
from safetensors import safe_open

DRAFT_REPO_ID = "RedHatAI/gemma-4-31B-it-speculator.dspark"
_CONFIG_PATH = (
    pathlib.Path(__file__).parent / "testdata" / "redhat_speculator_config.json"
)


class _ShapeOnly:
    """Header-shaped stand-in for tensors whose data the guards never read."""

    def __init__(self, shape: Sequence[int]) -> None:
        self.shape = tuple(shape)


@pytest.fixture(scope="module")
def draft_config() -> DSparkSpeculatorsDraftConfig:
    with open(_CONFIG_PATH) as f:
        raw = json.load(f)
    return DSparkSpeculatorsDraftConfig.from_huggingface_config(raw)


@pytest.fixture(scope="module")
def header() -> tuple[dict[str, tuple[int, ...]], np.ndarray]:
    """Tensor name -> shape from the real snapshot's header, plus d2t."""
    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        pytest.skip("HF Hub offline mode is enabled")
    path = hf_hub_download(DRAFT_REPO_ID, "model.safetensors")
    shapes: dict[str, tuple[int, ...]] = {}
    with safe_open(path, framework="np") as f:
        for name in f.keys():  # noqa: SIM118
            shapes[name] = tuple(f.get_slice(name).get_shape())
        d2t = f.get_tensor("d2t")
    return shapes, d2t


@pytest.fixture(scope="module")
def expected_draft_names(
    draft_config: DSparkSpeculatorsDraftConfig,
) -> set[str]:
    """Weight names of the draft module as the unified graph builds it
    (including the verbatim-loaded embed_tokens copy)."""
    device = DeviceRef.CPU()
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=draft_config.num_key_value_heads,
        head_dim=draft_config.head_dim,
        num_layers=draft_config.num_hidden_layers,
        devices=[device],
        page_size=128,
    )
    draft = DSparkSpeculatorsDraft(
        hidden_size=draft_config.hidden_size,
        num_hidden_layers=draft_config.num_hidden_layers,
        num_attention_heads=draft_config.num_attention_heads,
        num_key_value_heads=draft_config.num_key_value_heads,
        head_dim=draft_config.head_dim,
        intermediate_size=draft_config.intermediate_size,
        rms_norm_eps=draft_config.rms_norm_eps,
        rope_theta=draft_config.rope_theta,
        sliding_window=draft_config.sliding_window,
        layer_causal=[draft_config.causal] * draft_config.num_hidden_layers,
        vocab_size=draft_config.vocab_size,
        draft_vocab_size=draft_config.draft_vocab_size,
        markov_rank=draft_config.markov_rank,
        block_size=draft_config.block_size,
        sample_from_anchor=draft_config.sample_from_anchor,
        mask_token_id=draft_config.mask_token_id,
        num_context_features=draft_config.num_context_features,
        max_seq_len=8192,
        kv_params=kv_params,
        devices=[device],
        dtype=DType.bfloat16,
    )
    draft.embed_tokens = Embedding(
        vocab_size=draft_config.vocab_size,
        hidden_dim=draft_config.hidden_size,
        dtype=DType.bfloat16,
        device=device,
    )
    return set(draft.raw_state_dict().keys())


def _skipped(names: set[str]) -> set[str]:
    return {
        n
        for n in names
        if n in SKIPPED_DRAFT_KEYS or n.startswith(SKIPPED_DRAFT_PREFIXES)
    }


def test_checkpoint_names_cover_draft_module(
    header: tuple[dict[str, tuple[int, ...]], np.ndarray],
    expected_draft_names: set[str],
) -> None:
    shapes, _ = header
    names = set(shapes)
    # No phantom skips: every skip-list entry matches real tensors.
    for key in SKIPPED_DRAFT_KEYS:
        assert key in names, f"skip-list key {key!r} not in the checkpoint"
    for prefix in SKIPPED_DRAFT_PREFIXES:
        assert any(n.startswith(prefix) for n in names), (
            f"skip-list prefix {prefix!r} matches nothing in the checkpoint"
        )
    consumed = names - _skipped(names)
    unconsumed = consumed - expected_draft_names
    missing = expected_draft_names - consumed
    assert not unconsumed, (
        f"checkpoint keys with no draft-module consumer: {sorted(unconsumed)}"
    )
    assert not missing, (
        f"draft-module weights absent from the checkpoint: {sorted(missing)}"
    )


def test_merge_applies_exactly_the_skip_list(
    header: tuple[dict[str, tuple[int, ...]], np.ndarray],
) -> None:
    shapes, _ = header
    names = set(shapes)
    placeholder = object()
    unified = merge_unified_state_dict(
        {}, {name: placeholder for name in names}
    )
    assert set(unified) == {f"draft.{n}" for n in names - _skipped(names)}


def _guard_inputs(
    shapes: dict[str, tuple[int, ...]], d2t: np.ndarray
) -> dict[str, Any]:
    state: dict[str, Any] = {
        name: _ShapeOnly(shape) for name, shape in shapes.items()
    }
    state["d2t"] = d2t
    return state


def test_weight_guards_pass_on_real_checkpoint(
    header: tuple[dict[str, tuple[int, ...]], np.ndarray],
    draft_config: DSparkSpeculatorsDraftConfig,
) -> None:
    shapes, d2t = header
    # The pruned-vocab head: NOT the 12B [vocab, hidden] alias shape.
    assert shapes["lm_head.weight"] == (
        draft_config.draft_vocab_size,
        draft_config.hidden_size,
    )
    assert shapes["embed_tokens.weight"] == (
        draft_config.vocab_size,
        draft_config.hidden_size,
    )
    validate_draft_checkpoint_weights(_guard_inputs(shapes, d2t), draft_config)


@pytest.mark.parametrize(
    ("name", "shape", "match"),
    [
        # The 12B-style full-vocab alias shape must be rejected here.
        ("lm_head.weight", (262144, 5376), "lm_head"),
        ("markov_head.markov_w2.weight", (262144, 256), "markov_w2"),
        ("fc.weight", (5376, 5376), "fc.weight"),
        ("d2t", (262144,), "d2t"),
    ],
)
def test_weight_guards_reject_tampered_shapes(
    header: tuple[dict[str, tuple[int, ...]], np.ndarray],
    draft_config: DSparkSpeculatorsDraftConfig,
    name: str,
    shape: tuple[int, ...],
    match: str,
) -> None:
    shapes, d2t = header
    state = _guard_inputs(shapes, d2t)
    state[name] = _ShapeOnly(shape)
    with pytest.raises(ValueError, match=match):
        validate_draft_checkpoint_weights(state, draft_config)


def test_weight_guards_reject_out_of_range_d2t(
    header: tuple[dict[str, tuple[int, ...]], np.ndarray],
    draft_config: DSparkSpeculatorsDraftConfig,
) -> None:
    shapes, d2t = header
    bad_d2t = d2t.copy()
    bad_d2t[0] = draft_config.vocab_size  # maps id 0 out of [0, vocab)
    with pytest.raises(ValueError, match="outside the target vocabulary"):
        validate_draft_checkpoint_weights(
            _guard_inputs(shapes, bad_d2t), draft_config
        )


def test_weight_guards_reject_missing_d2t(
    header: tuple[dict[str, tuple[int, ...]], np.ndarray],
    draft_config: DSparkSpeculatorsDraftConfig,
) -> None:
    shapes, d2t = header
    state = _guard_inputs(shapes, d2t)
    del state["d2t"]
    with pytest.raises(ValueError, match="missing 'd2t'"):
        validate_draft_checkpoint_weights(state, draft_config)
