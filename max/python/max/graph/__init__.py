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
"""APIs to build inference graphs for MAX."""

from . import dtype_promotion, ops
from .dim import AlgebraicDim, Dim, DimLike, StaticDim, SymbolicDim
from .graph import (
    DevicePlacementPolicy,
    Graph,
    GraphBlock,
    GraphDebugConfig,
    KernelLibrary,
    Module,
)
from .graph import (
    default_custom_extensions as default_custom_extensions,
)
from .graph import (
    default_custom_extensions_scope as default_custom_extensions_scope,
)
from .type import (
    BufferType,
    DeviceKind,
    DeviceRef,
    Shape,
    ShapeLike,
    TensorType,
    Type,
    _ChainType,
    _OpaqueType,
)
from .value import (
    BufferValue,
    BufferValueLike,
    TensorValue,
    TensorValueLike,
    Value,
    _ChainValue,
    _OpaqueValue,
)
from .weight import ShardingStrategy, Weight
