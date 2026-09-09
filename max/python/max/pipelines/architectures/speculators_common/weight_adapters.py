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
"""Weight handling shared by unified speculators-DSpark pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np

from .draft_config import DSparkSpeculatorsDraftConfig

# Draft checkpoint tensors with no inference-time consumer, skipped at load
# exactly like the vLLM reference: ``t2d`` is training-only and the
# confidence head gates dynamic proposal lengths, which MAX does not use.
SKIPPED_DRAFT_KEYS = ("t2d",)
SKIPPED_DRAFT_PREFIXES = ("confidence_head.",)


def merge_unified_state_dict(
    target_state_dict: dict[str, Any],
    draft_state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Merges target + draft weights for a unified model.

    Prefixes target weights with ``target.*`` and draft weights with
    ``draft.*`` verbatim (the speculators checkpoint carries no ``model.``
    prefix and its keys match the draft module 1:1, including the frozen
    ``embed_tokens`` / pruned-vocab ``lm_head`` copies and the ``d2t`` offset
    table). Training-only tensors are skipped.
    """
    unified: dict[str, Any] = {}
    for name, value in target_state_dict.items():
        unified[f"target.{name}"] = value
    for name, value in draft_state_dict.items():
        if name in SKIPPED_DRAFT_KEYS or name.startswith(
            SKIPPED_DRAFT_PREFIXES
        ):
            continue
        unified[f"draft.{name}"] = value
    return unified


def validate_draft_checkpoint_weights(
    draft_state_dict: dict[str, Any],
    draft_config: DSparkSpeculatorsDraftConfig,
) -> None:
    """Weight-time guards for a speculators DSpark draft checkpoint.

    Complements the config-time ``validate_dspark_fields`` checks with the
    invariants only the tensors themselves can prove: the pruned-vocab
    lm_head / markov_w2 row counts, the fc tap-concat width, and the ``d2t``
    offset table mapping into the target vocabulary. Raises with the
    offending tensor named, before any graph is built.
    """
    draft_vocab = draft_config.draft_vocab_size
    vocab = draft_config.vocab_size

    def _shape(name: str) -> tuple[int, ...]:
        value = draft_state_dict.get(name)
        if value is None:
            raise ValueError(f"DSpark draft checkpoint is missing '{name}'.")
        return tuple(int(d) for d in value.shape)

    lm_head_shape = _shape("lm_head.weight")
    if lm_head_shape[0] != draft_vocab:
        raise ValueError(
            f"DSpark draft lm_head.weight has {lm_head_shape[0]} rows;"
            f" expected draft_vocab_size={draft_vocab}."
        )
    markov_w2_shape = _shape("markov_head.markov_w2.weight")
    if markov_w2_shape[0] != draft_vocab:
        raise ValueError(
            f"DSpark draft markov_w2 has {markov_w2_shape[0]} rows;"
            f" expected draft_vocab_size={draft_vocab}."
        )
    fc_shape = _shape("fc.weight")
    expected_fc_in = (
        draft_config.num_context_features * draft_config.hidden_size
    )
    if len(fc_shape) != 2 or fc_shape[1] != expected_fc_in:
        raise ValueError(
            f"DSpark draft fc.weight has shape {fc_shape}; expected"
            f" in-features = num taps x hidden = {expected_fc_in}."
        )
    d2t_shape = _shape("d2t")
    if d2t_shape != (draft_vocab,):
        raise ValueError(
            f"DSpark draft d2t has shape {d2t_shape}; expected"
            f" [draft_vocab_size] = [{draft_vocab}]."
        )
    d2t = np.from_dlpack(draft_state_dict["d2t"]).astype(np.int64)
    mapped = np.arange(draft_vocab, dtype=np.int64) + d2t
    if int(mapped.min()) < 0 or int(mapped.max()) >= vocab:
        raise ValueError(
            "DSpark draft d2t maps ids outside the target vocabulary:"
            f" range [{int(mapped.min())}, {int(mapped.max())}] vs"
            f" [0, {vocab})."
        )
