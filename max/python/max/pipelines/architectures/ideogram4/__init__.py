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
"""Ideogram 4 flow-matching text-to-image architecture."""

from .arch import Ideogram4ArchConfig, ideogram4_arch
from .ideogram4 import Ideogram4Transformer2DModel
from .model import Ideogram4TransformerModel
from .model_config import Ideogram4Config

__all__ = [
    "Ideogram4ArchConfig",
    "Ideogram4Config",
    "Ideogram4Transformer2DModel",
    "Ideogram4TransformerModel",
    "ideogram4_arch",
]
