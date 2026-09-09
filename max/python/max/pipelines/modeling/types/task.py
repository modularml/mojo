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

"""Pipeline Tasks Module.

This module defines the set of supported pipeline tasks for the MAX API, encapsulated
in the `PipelineTask` enumeration. Pipeline tasks represent the high-level operations
that can be performed by a pipeline, such as text generation, embeddings generation,
pixel generation, and audio generation.

Each task type is associated with a specific input/output contract and is used to
route requests to the appropriate pipeline implementation. The `PipelineTask` enum
is used throughout the MAX API to ensure type safety and consistency when specifying
or querying the type of task a pipeline supports.

Typical usage includes:
    - Registering supported architectures and pipelines for a given task.
    - Determining the output type for a pipeline task.
    - Routing inference requests to the correct pipeline based on the task.

Available tasks:
    - TEXT_GENERATION: Generate text sequences from input prompts.
    - EMBEDDINGS_GENERATION: Generate vector embeddings for input data.
    - PIXEL_GENERATION: Generate/Edit images/videos from input data.
    - AUDIO_GENERATION: Generate audio waveforms from input data.

See the `PipelineTask` enum for further details on each task type.
"""

from enum import Enum


class InputModality(str, Enum):
    """Enum representing the types of input a model architecture accepts.

    Used by :class:`~max.pipelines.lib.registry.SupportedArchitecture` to
    explicitly declare what each architecture can consume.  Currently this
    is informational only -- it drives the generated models table in the
    docs and has no effect on architecture behavior at runtime.
    """

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"


class PipelineTask(str, Enum):
    """Enum representing the types of pipeline tasks supported."""

    TEXT_GENERATION = "text_generation"
    """Task for generating text."""
    EMBEDDINGS_GENERATION = "embeddings_generation"
    """Task for generating embeddings."""
    PIXEL_GENERATION = "pixel_generation"
    """Task for generating pixels."""
    AUDIO_GENERATION = "audio_generation"
    """Task for generating audio waveforms."""
    UNDEFINED = "undefined"
    """Undefined task, used as default when task should be auto-detected."""
