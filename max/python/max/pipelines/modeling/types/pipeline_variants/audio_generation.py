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
"""Audio generation input types for Modular's MAX API."""

from __future__ import annotations

__all__ = [
    "AudioGenerationInputs",
]

from dataclasses import dataclass
from typing import Generic

from max.pipelines.context import AudioGenerationContextType
from max.pipelines.modeling.types.pipeline import PipelineInputs
from max.pipelines.request import RequestID


@dataclass(frozen=True)
class AudioGenerationInputs(
    PipelineInputs, Generic[AudioGenerationContextType]
):
    """Input data structure for audio generation pipelines."""

    batch: dict[RequestID, AudioGenerationContextType]
    """A dictionary mapping RequestID to AudioGenerationContextType instances."""
