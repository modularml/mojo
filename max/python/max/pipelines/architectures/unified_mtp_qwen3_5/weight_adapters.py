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
"""Weight adapter for the fused Qwen3.5 MTP graph."""

from __future__ import annotations

from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

from ..qwen3_5.weight_adapters import convert_qwen3_5_state_dict

_MTP_PREFIX = "mtp."


def convert_qwen3_5_with_mtp_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Splits the checkpoint into ``target.*`` and ``draft.*``.

    The base adapter already drops ``mtp.*`` and does every rename, dtype
    promotion and conv1d reshape the target needs, so this only has to prefix
    its output and re-admit the MTP head under the draft's own names. Those 15
    tensors are BF16 in every published checkpoint, including the NVFP4 one,
    and their module paths already match :class:`Qwen3_5MTP` once the prefix
    is stripped.

    ``embed_tokens`` and ``lm_head`` appear only under ``target.*``: the draft
    aliases the target's modules, so ``state_dict()`` deduplicates them.
    """
    target = convert_qwen3_5_state_dict(
        state_dict,
        huggingface_config=huggingface_config,
        pipeline_config=pipeline_config,
    )
    converted: dict[str, WeightData] = {
        f"target.{name}": value for name, value in target.items()
    }
    for name, value in state_dict.items():
        if name.startswith(_MTP_PREFIX):
            converted[f"draft.{name[len(_MTP_PREFIX) :]}"] = value.data()
    return converted
