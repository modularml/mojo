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
"""Weight adapter for the fused Qwen3.5 DFlash2 graph."""

from __future__ import annotations

from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

from ..qwen3_5.weight_adapters import convert_qwen3_5_state_dict


def convert_qwen3_5_with_dflash2_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Converts the target half only.

    The base adapter does every rename, dtype promotion and conv1d reshape the
    target needs, and drops the ``mtp.*`` head this graph does not use. The
    drafter is a separate checkpoint, loaded in ``model.py``; ``model.py`` is
    also what applies the ``target.`` / ``draft.`` prefixes, since only it
    holds both halves.
    """
    return convert_qwen3_5_state_dict(
        state_dict,
        huggingface_config=huggingface_config,
        pipeline_config=pipeline_config,
    )
