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

# The ``laguna`` model_type is registered as a local ``AutoConfig`` shim in
# ``..hf_config_shims`` (eagerly imported by the architectures package), so
# ``config.json`` loads without ``trust_remote_code`` and without executing the
# repo's remote ``configuration_laguna.py``.
from .arch import laguna_arch
from .model import LagunaModel
from .model_config import LagunaConfig
from .reasoning import LagunaReasoningParser
from .tool_parser import LagunaToolParser

__all__ = [
    "LagunaConfig",
    "LagunaModel",
    "LagunaReasoningParser",
    "LagunaToolParser",
    "laguna_arch",
]
