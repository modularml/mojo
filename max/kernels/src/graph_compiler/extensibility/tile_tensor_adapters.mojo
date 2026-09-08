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
"""Helpers for projecting dense tensor arguments to `TileTensor` views."""

from layout import TileTensor

from .tensor_arg_traits import (
    DenseTensor,
    MutableInput,
    Output,
    TileTensorable,
)


comptime _TILE_TENSOR_PROJECTION_DISABLED_MSG = (
    "to_tile_tensor requires a TileTensorable dense argument; the graph"
    " compiler disables the projection on targets whose storage cannot be"
    " viewed as a TileTensor at the execute boundary"
)


comptime TileProjection[
    mut: Bool, TensorType: DenseTensor
]: AnyType = TileTensor[
    mut=mut,
    TensorType.dtype,
    LayoutType=TensorType.LayoutType,
    origin=UntrackedOrigin[mut=mut],
    Engine=TensorType.Engine,
]


@always_inline
def to_tile_tensor[
    TensorType: DenseTensor, //, *, mut: Bool = False
](tensor: TensorType) -> TileProjection[mut, TensorType]:
    """Projects a dense tensor argument to a `TileTensor` view.

    Parameters:
        TensorType: The dense tensor-argument type (inferred).
        mut: Whether to project a mutable view (defaults to immutable).

    Args:
        tensor: The dense tensor argument to project.

    Returns:
        A `TileTensor` view over the argument's storage.
    """
    comptime assert conforms_to(
        TensorType, TileTensorable
    ), _TILE_TENSOR_PROJECTION_DISABLED_MSG
    comptime if mut:
        comptime assert TensorType.mut
    return rebind[TileProjection[mut, TensorType]](tensor.to_tile_tensor())


@always_inline
def to_tile_tensor[
    TensorType: Output & DenseTensor, //
](tensor: TensorType) -> TileProjection[True, TensorType]:
    """Projects an output dense tensor argument to a mutable `TileTensor` view.

    Output tensors are writable by construction, so the trait bound selects the
    mutable projection without call sites passing `mut=True`.

    Parameters:
        TensorType: The output dense tensor-argument type (inferred).

    Args:
        tensor: The output dense tensor argument to project.

    Returns:
        A mutable `TileTensor` view over the argument's storage.
    """
    comptime assert conforms_to(
        TensorType, TileTensorable
    ), _TILE_TENSOR_PROJECTION_DISABLED_MSG
    comptime assert TensorType.mut
    return rebind[TileProjection[True, TensorType]](tensor.to_tile_tensor())


@always_inline
def to_tile_tensor[
    TensorType: MutableInput & DenseTensor, //
](tensor: TensorType) -> TileProjection[True, TensorType]:
    """Projects a mutable-input dense tensor argument to a mutable `TileTensor`.

    Mutable inputs are read-write by construction, so the trait bound selects
    the mutable projection without call sites passing `mut=True`.

    Parameters:
        TensorType: The mutable-input dense tensor-argument type (inferred).

    Args:
        tensor: The mutable-input dense tensor argument to project.

    Returns:
        A mutable `TileTensor` view over the argument's storage.
    """
    comptime assert conforms_to(
        TensorType, TileTensorable
    ), _TILE_TENSOR_PROJECTION_DISABLED_MSG
    comptime assert TensorType.mut
    return rebind[TileProjection[True, TensorType]](tensor.to_tile_tensor())
