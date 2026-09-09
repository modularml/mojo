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

from collections.abc import Sequence

from max.graph import TensorValue

from .layer import Layer, Module


class LayerList(Module):
    """Stores a list of layers.

    Can be used as a regular Python list.

    For a stack of layers where each layer consumes the previous layer's
    output, prefer :class:`~max.nn.Sequential`, which subclasses this class.

    Args:
        layers: The layers to store in the list.
    """

    def __init__(self, layers: Sequence[Layer]) -> None:
        super().__init__()
        self.layers = list(layers)

        # Only assign `Module` objects to Sequential.sublayers. We ensure that
        # the V2 functionality (getting sublayers) is correct by throwing
        # an error in the `sublayers` property if any layer is still V1.
        for n, layer in enumerate(layers):
            if isinstance(layer, Module):
                self._sublayers[str(n)] = layer

    @property
    def sublayers(self) -> dict[str, Module]:
        if len(self._sublayers) != len(self.layers):
            raise ValueError(
                "Not all layers in this Sequential object have "
                "been migrated to V2."
            )
        return super().sublayers

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, i: int) -> Layer:
        return self.layers[i]

    def __delitem__(self, i: int) -> None:
        del self.layers[i]

    def __setitem__(self, i: int, layer: Layer) -> None:
        self.layers[i] = layer

    def insert(self, i: int, layer: Layer) -> None:
        self.layers.insert(i, layer)

    def append(self, layer: Layer) -> None:
        self.layers.append(layer)

    def extend(self, layer: Layer) -> None:
        self.layers.append(layer)

    def __str__(self) -> str:
        return str(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __call__(self, *args, **kwargs) -> TensorValue:
        x = self.layers[0](*args, **kwargs)
        for layer in self.layers[1:]:
            x = layer(x)
        return x
