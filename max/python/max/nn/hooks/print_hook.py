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
"""Print hook for MAX Pipeline models."""

from __future__ import annotations

import functools
import logging
import os
from collections import deque
from collections.abc import Generator
from typing import Any

from max.experimental import functional as F
from max.experimental.nn import Module as ModuleV3
from max.experimental.tensor import Tensor
from max.graph import TensorValue
from max.nn._identity import IdentitySet
from max.nn.layer import Layer, add_layer_hook, clear_hooks

from .base_print_hook import BasePrintHook

logger = logging.getLogger("max.pipelines")


class PrintHook(BasePrintHook):
    """Hook that prints layer tensor inputs and outputs.

    Initialize the class before building the graph. This way, the hook can add
    print ops to the graph.

    Args:
        export_path: Intended as the directory to write recorded tensors
            into, but writing to disk isn't yet supported on the MAX execution
            path. When set, the path is ignored.
        filter: A list of layer names to restrict printing to. When ``None``,
            every named layer is printed.
    """

    def __init__(
        self, export_path: str | None = None, filter: list[str] | None = None
    ) -> None:
        super().__init__(export_path=export_path, filter=filter)
        add_layer_hook(self)
        # V3 modules have no global hook dispatch, so the V3 path wraps each
        # module's per-instance ``forward``. Track the originals so ``remove``
        # can restore them, and the set guards against double-wrapping.
        self._v3_wrapped = IdentitySet[ModuleV3[..., Any]]()
        self._v3_originals: list[tuple[ModuleV3[..., Any], Any]] = []
        if export_path is not None:
            logger.warning(
                "Export path is currently not supported. Values will be printed"
                " to stdout with COMPACT format."
            )

    def name_layers(self, model: Layer | ModuleV3[..., Any]) -> None:
        """Create names for all layers in the model based on nested attributes.

        Args:
            model: A V2 :obj:`~max.nn.layer.Layer` or V3 :obj:`Module` instance.

        Raises:
            TypeError: If ``model`` is neither a V2 ``Layer`` nor a V3
                ``Module``.
        """
        if isinstance(model, Layer):
            for layer, name in _walk_layers(model):
                self.add_layer(layer, name)
        elif isinstance(model, ModuleV3):
            self.name_layers_v3(model)
        else:
            raise TypeError(
                "PrintHook.name_layers expects a max.nn (V2) Layer or a "
                "max.experimental.nn (V3) Module, got "
                f"{type(model).__name__}."
            )

    @property
    def export_path(self) -> str | None:
        if self._export_path is None:
            return None
        return os.path.join(self._export_path, str(self._current_step))

    def name_layers_v3(self, model: ModuleV3[..., Any]) -> None:
        """Name all v3 Module layers and wrap their ``forward`` to fire the hook."""
        self.add_layer(model, "model")
        self._wrap_v3_forward(model)
        for rel_name, module in model.descendants:
            self.add_layer(module, f"model.{rel_name}")
            self._wrap_v3_forward(module)

    def _wrap_v3_forward(self, module: ModuleV3[..., Any]) -> None:
        """Shadow ``module.forward`` with a wrapper that invokes this hook."""
        if module in self._v3_wrapped:
            return
        self._v3_wrapped.add(module)

        # With no instance-level ``forward`` yet, this resolves to the bound
        # class method, which the wrapper defers to.
        original = module.forward

        @functools.wraps(original)
        def forward_with_hook(*args: Any, **kwargs: Any) -> Any:
            outputs = original(*args, **kwargs)
            self(module, args, kwargs, outputs)
            return outputs

        # ``object.__setattr__`` bypasses any custom ``__setattr__`` and works
        # even for frozen dataclasses; the attribute shadows the class method.
        object.__setattr__(module, "forward", forward_with_hook)
        self._v3_originals.append((module, original))

    def print_value(self, name: str, value: Any) -> bool:
        if isinstance(value, TensorValue):
            value.print(name)
            return True
        if not isinstance(value, Tensor):
            return False
        # Only non-real (symbolic) tensors carry a graph TensorValue that can
        # be printed. Real (eager) tensors are backed by concrete storage and
        # have no graph value to emit. Tensor.real is False exactly when the
        # tensor was created inside an active graph-tracing context and still
        # holds a RealizationState pointing at a graph value.
        if value.real:
            return False
        F.print(value, name)
        return True

    def remove(self) -> None:
        super().remove()
        clear_hooks()  # TODO: Add individual hook remover.
        # Restore V3 modules by dropping the instance-level ``forward`` so the
        # class method takes over again.
        for module, _ in self._v3_originals:
            module.__dict__.pop("forward", None)
        self._v3_originals.clear()
        self._v3_wrapped = IdentitySet[ModuleV3[..., Any]]()

    def __del__(self) -> None:
        self.summarize()


_SUPPORTED_TYPES = (Layer, list, tuple)


def _walk_layers(model: Layer) -> Generator[tuple[Layer, str], None, None]:
    """Walks through model and yields all layers with generated names."""
    seen = IdentitySet[Layer]()
    seen.add(model)
    queue: deque[tuple[Any, str]] = deque([(model, "model")])

    while queue:
        obj, name = queue.popleft()
        if isinstance(obj, Layer):
            yield obj, name
            for k, v in obj.__dict__.items():
                if v not in seen or isinstance(v, _SUPPORTED_TYPES):
                    queue.append((v, f"{name}.{k}"))
        elif isinstance(obj, list | tuple):
            for n, v in enumerate(obj):
                if v not in seen or isinstance(v, _SUPPORTED_TYPES):
                    queue.append((v, f"{name}.{n}"))
