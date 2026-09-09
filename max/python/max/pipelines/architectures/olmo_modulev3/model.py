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

from __future__ import annotations

from typing import Literal

from ..llama3_modulev3.model import Llama3Model
from .model_config import OlmoConfig


class OlmoModel(Llama3Model):
    """Olmo pipeline model implementation."""

    config_class = OlmoConfig
    norm_method: Literal["rms_norm", "layer_norm"] = "layer_norm"
