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

"""Defines the :class:`ComponentModel` abstract base class for pipeline components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from max.driver import Device
from max.graph.weights import Weights

if TYPE_CHECKING:
    from max.pipelines.diffusion.config import DenoisingCacheConfig
    from max.pipelines.modeling.config_enums import SupportedEncoding


class ComponentModel(ABC):
    """Base interface for component models with weight-backed execution."""

    def __init__(
        self,
        config: Any,
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        *,
        cache_config: DenoisingCacheConfig | None = None,
    ) -> None:
        self.config = config
        self.encoding = encoding
        self.devices = devices
        self.weights = weights
        self.cache_config = cache_config

    @abstractmethod
    def load_model(self) -> Callable[..., Any] | None:
        """Load and return a runtime model instance."""
        ...
