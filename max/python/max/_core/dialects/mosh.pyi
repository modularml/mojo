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

from collections.abc import Sequence
from typing import overload

import max._core
import max._core.dialects.builtin
import max._core.dialects.kgen
import max._core.dialects.m
from max.mlir import Location

# C++ overloads on different int types look the same in Python, ignore these
# mypy: disable-error-code="overload-cannot-match"

class ParamFromValueOp(max._core.Operation):
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        param_decl: max._core.dialects.kgen.ParamDeclAttr,
        value: max._core.Value,
    ) -> None: ...
    @property
    def param_decl(self) -> max._core.dialects.kgen.ParamDeclAttr: ...
    @param_decl.setter
    def param_decl(
        self, arg: max._core.dialects.kgen.ParamDeclAttr, /
    ) -> None: ...
    @property
    def value(self) -> max._core.Value: ...

class ParamToValueOp(max._core.Operation):
    """
    The `mo.param.to_value` operation materializes the value of a parameter
    expression as an SSA value that may be used by other operations.
    Conceptually, it bridges the parameter value domain to the SSA value domain.

    Example:

    ```mlir
    %idx   = mo.param.to_value = <D0>
    %shape = mo.param.to_value: !mosh.ape = <[10, D1, 20]>
    ```

    Note that in the assembly format, we allow omitting output type annotation
    if it's `index`, for historical reasons.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.Type,
        value: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def value(self) -> max._core.dialects.builtin.TypedAttr: ...
    @value.setter
    def value(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...

class ShapeAttr(max._core.Attribute):
    """
    The `#mosh.ape` attribute contains a variable number of `TypedAttr`s, each
    of which has a kgen.scalar<si64> type. The type of this attribute is
    always `!mosh.ape`.

    Each dimension `TypedAttr` inside `values` is one of:
    1. `KGEN::ParamDeclRefAttr` for a dimension parameter, e.g., `D0`.
    2. `KGEN::ParamOperatorAttr` for a dimension parameter expression, e.g.,
    `add(D1, 2)`.
    3. `KGEN::SIMDAttr` for a concrete integer dimension, e.g., `42` (typed as
    `kgen.scalar<si64>`).

    Note that -1 can be used as a special dimension value that denotes a
    dimension to be inferred from other dimensions and total number of elements
    in the tensor. At most 1 dimension can be -1.

    Example:

    ```mlir
    kgen.param.declare N = <3>
    #mosh<ape[1, ?, N]> : !mosh.ape
    #mosh<ape[3, -1, 42]> : !mosh.ape
    ```
    """

    @overload
    def __init__(
        self,
        values: Sequence[max._core.dialects.builtin.TypedAttr],
        type: ShapeType,
    ) -> None: ...
    @overload
    def __init__(self, values: Sequence[int], type: ShapeType) -> None: ...
    @overload
    def __init__(
        self,
        int_dims: max._core.dialects.m.IntArrayElementsAttr,
        type: ShapeType,
    ) -> None: ...
    @overload
    def __init__(
        self,
        values: Sequence[max._core.dialects.builtin.TypedAttr],
        type: ShapeType,
    ) -> None: ...
    @property
    def values(self) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...
    @property
    def type(self) -> ShapeType: ...

class ShapeType(max._core.Type):
    def __init__(self) -> None: ...
