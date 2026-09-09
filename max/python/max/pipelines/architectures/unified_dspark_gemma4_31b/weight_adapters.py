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
"""Weight adapters for the unified DSpark Gemma4 31B pipeline.

The speculators-format draft checkpoint handling (merge, validation, skip
lists) lives in the shared ``speculators_common`` package.
"""

from __future__ import annotations

from max.graph.weights import WeightData, Weights
from transformers import AutoConfig

from ..gemma4.weight_adapters import convert_safetensor_language_state_dict


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Converts the target Gemma4 checkpoint's language weights.

    The unified graph's target is text-only, so the vision tower is dropped.
    Draft weights are loaded separately in ``model.py`` from the
    ``draft_model`` checkpoint (its keys already match the draft module).
    """
    del huggingface_config
    return convert_safetensor_language_state_dict(state_dict)
