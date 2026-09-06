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

"""Weight adapters for BertForMaskedLM models.

This module converts HuggingFace BertForMaskedLM weight names to MAX weight
names, and drops weights that MAX re-derives or does not use at runtime.

HuggingFace BertForMaskedLM naming convention:
- bert.embeddings.word_embeddings.weight
- bert.embeddings.LayerNorm.weight
- bert.embeddings.LayerNorm.bias
- bert.embeddings.position_ids        (buffer, not a learned weight)
- bert.encoder.layer.0.attention.self.query.weight
- bert.pooler.dense.weight            (unused by MAX)
- cls.predictions.transform.dense.weight  (LM head, weight-tied to embeddings)
- cls.predictions.bias

MAX naming convention:
- embeddings.word_embeddings.weight
- embeddings.layer_norm.weight
- embeddings.layer_norm.bias
- encoder.layer.0.attention.self.query.weight
(position_ids, pooler.*, and cls.* are omitted entirely)
"""

from __future__ import annotations

from collections.abc import Mapping

from max.graph.weights import WeightData, Weights

BERT_FOR_MASKED_LM_SAFETENSOR_MAP: dict[str, str | None] = {
    "bert.": "",
    ".LayerNorm.gamma": ".layer_norm.weight",
    ".LayerNorm.beta": ".layer_norm.bias",
    ".LayerNorm.": ".layer_norm.",
    ".position_ids": None,
    "pooler.": None,
    "cls.": None,
}


def convert_safetensor_state_dict(
    state_dict: Mapping[str, Weights],
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}
    value: Weights | None
    for weight_name, value in state_dict.items():
        max_name = weight_name
        for before, after in BERT_FOR_MASKED_LM_SAFETENSOR_MAP.items():
            if after is None:
                if before in max_name:
                    value = None
                    break
            else:
                max_name = max_name.replace(before, after)
        if value is not None:
            new_state_dict[max_name] = value.data()
    return new_state_dict
