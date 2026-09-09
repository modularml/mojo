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
"""Modular engine provides methods to load and execute AI models."""

from max._core import __version__

from . import mlrt as mlrt
from .api import CompiledModel as CompiledModel
from .api import (
    CustomExtensionsType,
    DebugConfig,
    GPUProfilingMode,
    InferenceSession,
    LogLevel,
    Model,
    PrintStyle,
    TensorSpec,
    read,
)
