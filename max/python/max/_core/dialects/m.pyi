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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import enum
from collections.abc import Sequence
from typing import Protocol, overload

import max._core
import max._core.dialects.builtin

# C++ overloads on different int types look the same in Python, ignore these
# mypy: disable-error-code="overload-cannot-match"

class InOutSemantics(enum.Enum):
    none = 46

    in_ = 105

    out = 111

    mut = 109

class HasAlignedBytesInterface(Protocol):
    """
    This interface allows an attribute to describe the size and
    alignment of its underlying ArrayRef<uint8_t> data as an !M.aligned_bytes.
    """

    @property
    def aligned_bytes_type(self) -> AlignedBytesType: ...

class ArrayElementsAttr(max._core.Attribute):
    """
    The `#M.dense_array` attribute is an elements attribute backed by a
    primitive array. The attribute supports the full API expected by
    `ElementsAttr` and users thereof, including arbitrary shaped types and
    failable value iteration.

    Importantly, this attribute can only contain scalar integer and floating
    point types, does not bit-pack elements, and does not have special handling
    for splat elements.

    This attribute implements the HasAlignedBytesInterface by returning a
    !M.aligned_bytes type capturing the byte size of the underlying element data
    array and an alignment of the next power-of-two of the byte size of the
    element type.

    Example:

    ```mlir
    // An array of integers.
    #M.dense_array<-3, 0, 1, 42> : !M.array<4xi32>

    // A float tensor.
    #M.dense_array<3.4, 2.3, 5.2, 1.9> : tensor<2x2xf32>

    // A 0D vector.
    #M.dense_array<1> : vector<ui64>
    ```
    """

    @overload
    def __init__(
        self, data: Sequence[int], type: max._core.dialects.builtin.ShapedType
    ) -> None: ...
    @overload
    def __init__(
        self,
        data: PrimitiveArrayAttr,
        type: max._core.dialects.builtin.ShapedType,
    ) -> None: ...
    @property
    def data(self) -> PrimitiveArrayAttr: ...
    @property
    def type(self) -> max._core.dialects.builtin.ShapedType: ...

class DeviceInfoAttr(max._core.Attribute):
    """
    The `#M.device_info` attribute captures runtime-probed properties of a
    physical device: its label (e.g. `"gpu"`), the underlying API
    (e.g. `"cuda"`, `"hip"`), the architecture name
    (e.g. `"sm_100a"`, `"gfx942"`), and the model name
    (e.g. `"NVIDIA B200"`). This is used to drive per-kernel device
    registration and specialization.

    Example:
    ```mlir
      #M.device_info<"gpu", "cuda", "sm_100a", "NVIDIA B200">
    ```
    """

    def __init__(
        self,
        label: str,
        api: str,
        arch: str,
        model: str,
        tile_based_fusion: bool,
    ) -> None: ...
    @property
    def label(self) -> str: ...
    @property
    def api(self) -> str: ...
    @property
    def arch(self) -> str: ...
    @property
    def model(self) -> str: ...
    @property
    def tile_based_fusion(self) -> bool: ...

class DeviceRefAttr(max._core.Attribute):
    """
    The `#M.device_ref` attribute refers to a unique `#M.device_spec` within the
    overall model `#M.device_spec_collection`. It contain a label and id.

    Example:
    ```mlir
      #M.device_ref<"gpu", 0>
    ```
    """

    def __init__(self, label: str, id: int) -> None: ...
    @property
    def label(self) -> str: ...
    @property
    def id(self) -> int: ...

class PrimitiveArrayAttr(max._core.Attribute):
    """
    The `#M.primitives_array` attribute represents an array of primitive
    (boolean, integer, index, or floating point) data of equal size. The data is
    stored as a byte array with element types whose sizes are not multiples of
    bytes padded to the nearest byte. The underlying array is aligned to that
    byte size.

    Example:

    ```mlir
    // An array of integers.
    #M.primitives_array<si24: -2, 0, 2>

    // An array of floats.
    #M.primitives_array<bf16: 0.2, 1.2, 3.>

    // Boolean arrays use `i1` elements.
    #M.primitives_array<i1: true, false, true>
    ```
    """

    @overload
    def __init__(
        self, data: Sequence[int], element_type: max._core.Type
    ) -> None: ...
    @overload
    def __init__(
        self, data: Sequence[int], element_type: max._core.Type
    ) -> None: ...
    @property
    def data(self) -> Sequence[int]: ...
    @property
    def element_type(self) -> max._core.Type | None: ...

class AlignedBytesType(max._core.Type):
    """
    This type has no values and no runtime representation. It is intended only
    to be used as a type annotation on `dense_resource` attribute operands
    so as to convey a desired alignment. This is needed in two situations:
     - As a way to 'forward declare' the alignment for an attribute who's
       blob has not yet been parsed.
     - As a way to override the required alignment for an attribute without
       reallocating the underlying data.

    Example:
    ```mlir
    // An array of 4 uint8_ts with 16 byte alignment
    !M.aligned_bytes<4, align 16>
    ```
    """

    def __init__(self, size: int, align: int) -> None: ...
    @property
    def size(self) -> int: ...
    @property
    def align(self) -> int: ...

class DataLayout: ...
class IntArrayElementsAttr: ...
