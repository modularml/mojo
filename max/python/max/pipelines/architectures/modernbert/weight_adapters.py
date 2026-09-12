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
"""Weight adapters for ModernBERT models."""

from __future__ import annotations

from collections.abc import Mapping

from max.graph.weights import WeightData, Weights


def convert_safetensor_state_dict(
    state_dict: Mapping[str, Weights],
) -> dict[str, WeightData]:
    """Convert HF ModernBERT checkpoint names to MAX graph names.

    Rules:
    - Strip leading ``model.`` from backbone weights.
    - Drop MLM-only weights under ``head.`` and ``decoder.``.
    """
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        if weight_name.startswith("head.") or weight_name.startswith(
            "decoder."
        ):
            continue

        max_name = (
            weight_name.removeprefix("model.")
            if weight_name.startswith("model.")
            else weight_name
        )
        new_state_dict[max_name] = value.data()

    return new_state_dict
