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
"""Provides layout and layout tensor types, which abstract memory layout for
multidimensional data.

- The [`Layout`](/api/mojo/layout/layout/Layout) type represents a mapping
  between a set of logical coordinates and a linear index. It can be used, for
  example, to map logical tensor coordinates to a memory address, or to map GPU
  threads to tiles of data.

- The [`LayoutTensor`](/api/mojo/layout/layout_tensor/LayoutTensor) type is a
  high-performance tensor with explicit memory layout via a `Layout`.
"""
from .coord import (
    All,
    Coord,
    CoordLike,
    ComptimeInt,
    Idx,
    coord,
    coord_to_index_list,
)
from .int_tuple import (
    UNKNOWN_VALUE,
    IntTuple,
    create_unknown_int_tuple,
    to_index_list,
)
from .layout import Layout, LayoutList, composition, print_layout
from .layout_tensor import LayoutTensor, stack_allocation_like
from .tile_layout import (
    TensorLayout,
    Layout as MixedLayout,
    RowMajorLayout,
    ColMajorLayout,
    row_major,
    col_major,
)
from .runtime_layout import RuntimeLayout
from .runtime_tuple import RuntimeTuple
from .numpy import from_numpy, to_numpy
from .tile_tensor import (
    TileTensor,
    flatten_leading,
    stack_allocation,
    lt_to_tt,
    lt_to_tt_idx,
    LTToTTLayout,
)
from .tensor_engine import (
    TensorEngine,
    DefaultEngine,
)
