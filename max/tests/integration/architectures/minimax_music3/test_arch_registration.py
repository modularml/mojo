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
"""That the architecture is reachable, and asks for the audio task.

The registration is spread over three files -- ``arch.py``, the lazy entry in
``architectures/__init__.py`` and ``all_arches.bzl`` -- and a mismatch between
them fails only when someone tries to serve the model, at which point the
symptom is an unrelated-looking "unsupported architecture". This is the cheap
guard: no GPU, no weights, no network.
"""

from __future__ import annotations

from max.pipelines.architectures.minimax_music3 import minimax_music3_arch
from max.pipelines.context import AudioContext
from max.pipelines.lib import PIPELINE_REGISTRY
from max.pipelines.modeling.types import PipelineTask


def test_registered_under_the_checkpoints_class_name() -> None:
    """The name is what ``modular_model_index.json`` calls the pipeline, which
    is how the registry finds it."""
    assert minimax_music3_arch.name == "MiniMaxMusic3ModularPipeline"


def test_asks_for_the_audio_task() -> None:
    assert minimax_music3_arch.task == PipelineTask.AUDIO_GENERATION
    assert minimax_music3_arch.context_type is AudioContext


def test_is_reachable_from_the_registry() -> None:
    """Resolved through the lazy table rather than by direct import, which is
    the path the serving entrypoint actually takes: a missing entry in
    ``architectures/__init__.py`` fails here and nowhere else."""
    resolved = PIPELINE_REGISTRY.retrieve_architecture(
        "MiniMaxMusic3ModularPipeline"
    )

    assert resolved is not None
    assert resolved.task == PipelineTask.AUDIO_GENERATION


def test_supported_encodings_include_the_default() -> None:
    assert (
        minimax_music3_arch.default_encoding
        in minimax_music3_arch.supported_encodings
    )
