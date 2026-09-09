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
"""Realization context for tracing a Module's forward pass."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from max.experimental.realization_context import GraphRealizationContext
from max.experimental.tensor import Tensor
from max.graph import BufferValue, Graph
from max.nn._identity import IdentityMap

if TYPE_CHECKING:
    from max.experimental.nn.module import Module


class ModuleTraceRealizationContext(GraphRealizationContext):
    """Graph realization context for tracing a :class:`Module`'s forward pass."""

    #: Settable function that materializes a weight from a name and tensor.
    create_external_constant: Callable[[str, str, Tensor, bool], Tensor] | None

    #: Weight-name prefix for each subgraphable module traced in this context.
    weight_prefixes: IdentityMap[Module[..., Any], str]

    def __init__(
        self,
        graph: Graph,
        signal_buffers: list[BufferValue] | None = None,
    ):
        """Initializes the module trace realization context.

        Args:
            graph: The graph to construct operations in.
            signal_buffers: GPU signal buffer graph values for
                multi-device collective ops.
        """
        super().__init__(graph, signal_buffers)
        self.create_external_constant = None
        self.weight_prefixes = IdentityMap()
