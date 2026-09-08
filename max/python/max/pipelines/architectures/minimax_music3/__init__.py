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
"""Generates music from a text description and lyrics."""

from .arch import MiniMaxMusic3ArchConfig, minimax_music3_arch
from .model_config import MiniMaxMusic3Config
from .music3_executor import MiniMaxMusic3Executor, MiniMaxMusic3Inputs
from .tokenizer import MiniMaxMusic3Tokenizer

__all__ = [
    "MiniMaxMusic3ArchConfig",
    "MiniMaxMusic3Config",
    "MiniMaxMusic3Executor",
    "MiniMaxMusic3Inputs",
    "MiniMaxMusic3Tokenizer",
    "minimax_music3_arch",
]
