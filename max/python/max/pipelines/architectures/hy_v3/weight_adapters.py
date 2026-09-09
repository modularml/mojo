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
"""Weight adapter for Hy3-preview.

Pure-rename adapter. MAX ``nn.moe.MoE`` stores experts as a ``LayerList``
of per-expert MLPs and builds the fused ``gate_up_proj`` tensor at
graph-build time via a property, so the on-disk per-expert layout
(``mlp.experts.{e}.{gate,up,down}_proj.weight``) passes through to MAX
unchanged once the leading ``model.`` namespace is stripped.

Renames:

* Strip leading ``model.``.
* ``mlp.router.gate.weight`` -> ``mlp.gate.gate_score.weight``
* ``mlp.expert_bias`` (FP32) -> ``mlp.gate.e_score_correction_bias``
* ``mlp.shared_mlp.`` -> ``mlp.shared_experts.``

Drop every ``model.layers.80.*`` key (the unused MTP layer; HF
reference also ignores it via ``_keys_to_ignore_on_load_unexpected``).
"""

from __future__ import annotations

from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

_TOP_LEVEL_RENAMES: tuple[tuple[str, str], ...] = (
    ("model.", ""),
    (".mlp.router.gate.weight", ".mlp.gate.gate_score.weight"),
    (".mlp.expert_bias", ".mlp.gate.e_score_correction_bias"),
    (".mlp.shared_mlp.", ".mlp.shared_experts."),
)


def _apply_renames(hf_name: str) -> str:
    out = hf_name
    for before, after in _TOP_LEVEL_RENAMES:
        out = out.replace(before, after)
    return out


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert Hy3 HF safetensor weights to the MAX module hierarchy."""
    num_hidden_layers = int(
        getattr(huggingface_config, "num_hidden_layers", 80)
    )
    mtp_prefix = f"model.layers.{num_hidden_layers}."

    tie_word_embeddings = bool(
        getattr(huggingface_config, "tie_word_embeddings", False)
    )

    out: dict[str, WeightData] = {}
    for hf_name, value in state_dict.items():
        if hf_name.startswith(mtp_prefix):
            continue
        max_name = _apply_renames(hf_name)
        if tie_word_embeddings and max_name == "lm_head.weight":
            continue
        out[max_name] = value.data()
    return out
