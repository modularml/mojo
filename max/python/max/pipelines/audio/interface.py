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
"""The contract between an audio architecture and the audio pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

from max.experimental.tensor import Tensor
from max.pipelines.modeling.base.tensor_struct import TensorStruct

if TYPE_CHECKING:
    from max.engine import InferenceSession
    from max.pipelines.lib.model_manifest import ModelManifest
    from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig

__all__ = [
    "AudioExecutor",
    "AudioExecutorOutputs",
]


@dataclass(frozen=True)
class AudioExecutorOutputs(TensorStruct):
    """One batch of generated waveforms."""

    waveform: Tensor
    """``(batch, channels, samples)`` in ``[-1, 1]``. An audio generator may
    produce fewer samples than the request asked for, so this is what says how
    long the audio actually is."""


_ContextT = TypeVar("_ContextT")
_InputsT = TypeVar("_InputsT", bound=TensorStruct)
_OutputsT = TypeVar("_OutputsT", bound=AudioExecutorOutputs, covariant=True)


class AudioExecutor(Protocol[_ContextT, _InputsT, _OutputsT]):
    """What :class:`AudioGenerationPipeline` requires of an architecture.

    Structurally this is
    :class:`~max.pipelines.lib.pipeline_executor.PipelineExecutor` --
    ``prepare_inputs`` then ``execute``, with the caller owning device
    placement in between -- narrowed so that ``execute`` returns waveforms,
    plus the rate those samples are meant to be played back at. The rate is
    a property of the model's vocoder rather than of the request, which is
    why it lives here and not on the context.

    Declared as a Protocol rather than as a subclass of ``PipelineExecutor``
    because that class lives in ``max.pipelines.lib``, which imports this
    package's pipeline from its registry: inheriting would close the cycle.
    Architectures subclass ``PipelineExecutor`` as usual and satisfy this by
    shape.
    """

    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
        runtime_config: PipelineRuntimeConfig,
    ) -> None:
        """Compiles the model's components and loads their weights."""
        ...

    @property
    def sample_rate(self) -> int:
        """Playback rate of the generated waveform, in hertz."""
        ...

    def prepare_inputs(self, contexts: list[_ContextT]) -> _InputsT:
        """Converts a batch of contexts into the executor's inputs."""
        ...

    def execute(self, inputs: _InputsT) -> _OutputsT:
        """Runs every stage of the model and returns the finished audio."""
        ...
