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
"""NumPy interoperability for `TileTensor`.

These are host-side helpers: they call into CPython, so build the tensor on the
host and pass the result to device code.
"""

from std.python import PythonObject
from std.python.numpy import copy_to_numpy_tensor, from_numpy_tensor

from .coord import DynamicCoord
from .tile_layout import RowMajorLayout, TensorLayout, row_major
from .tile_tensor import TileTensor
from .tensor_engine import TensorEngine


comptime _NumPyLayout[rank: Int] = RowMajorLayout[
    *DynamicCoord[.int, rank].element_types
]
"""The layout of a `TileTensor` borrowed from a NumPy array: row-major with
one runtime extent per axis."""


def from_numpy[
    mut: Bool, //, dtype: DType, rank: Int, origin: Origin[mut=mut]
](ref[origin] array: PythonObject) raises -> TileTensor[
    dtype, _NumPyLayout[rank], origin
]:
    """Borrows an N-D C-contiguous NumPy array as a `TileTensor`.

    The tensor aliases the NumPy array's buffer; no bytes are copied. Its
    origin is tied to `array`, so the compiler keeps `array` alive for as long
    as the tensor is used.

    This calls into CPython, so it is a host-side entry point.

    Example:

    ```mojo
    from layout import from_numpy
    from std.python import Python

    var np = Python.import_module("numpy")
    var array = np.arange(6, dtype="float64").reshape(2, 3)
    var tensor = from_numpy[.float64, 2](array)
    var value = tensor[1, 2]
    ```

    Parameters:
        mut: The mutability of the borrow, inferred from `array`.
        dtype: The expected element dtype of the array.
        rank: The expected number of dimensions.
        origin: The origin of the borrow, inferred from `array`.

    Args:
        array: A C-contiguous NumPy `ndarray` of `rank` dimensions whose dtype
            matches `dtype`.

    Returns:
        A row-major `TileTensor` viewing the array's buffer.

    Raises:
        If `array.ndim` is not `rank`, is not C-contiguous, has a dtype that
        does not match `dtype`, or is not writable when borrowed mutably.
    """
    var view = from_numpy_tensor[dtype, rank](array)
    return TileTensor(view.data, row_major(view.shape))


def _is_row_major_contiguous[
    dtype: DType,
    LayoutType: TensorLayout,
    origin: Origin,
    Engine: TensorEngine,
](tensor: TileTensor[dtype, LayoutType, origin, Engine=Engine]) -> Bool:
    """Reports whether the tensor covers its buffer in C order without gaps.

    Parameters:
        dtype: The element dtype of the tensor.
        LayoutType: The layout of the tensor.
        origin: The origin of the tensor.
        Engine: The engine of the tensor.

    Args:
        tensor: The tensor to inspect.

    Returns:
        True if the innermost stride is 1 and each outer stride is the product
        of the extents below it.
    """
    var expected = 1

    comptime for i in reversed(range(LayoutType.rank)):
        if Int(tensor.layout.stride[i]().value()) != expected:
            return False
        expected *= Int(tensor.layout.shape[i]().value())

    return True


def to_numpy[
    dtype: DType,
    LayoutType: TensorLayout,
    origin: Origin,
    Engine: TensorEngine,
](
    tensor: TileTensor[dtype, LayoutType, origin, Engine=Engine]
) raises -> PythonObject:
    """Copies a `TileTensor` into a new NumPy array of the same shape.

    Tensors whose layout is row-major and gap-free are copied with a single
    `memcpy`. Other layouts are gathered element by element.

    This calls into CPython, so it is a host-side entry point.

    Example:

    ```mojo
    from layout import TileTensor, row_major, to_numpy
    from layout.coord import Coord, Idx

    var storage = Array[Float32, 6](uninitialized=True)
    var tensor = TileTensor(Span(storage), row_major(Coord(Idx[2], Idx[3])))
    var array = to_numpy(tensor)  # a 2x3 NumPy float32 array
    ```

    Parameters:
        dtype: The element dtype of the tensor.
        LayoutType: The layout of the tensor.
        origin: The origin of the tensor.
        Engine: The engine of the tensor.

    Args:
        tensor: The tensor to copy.

    Constraints:
        `LayoutType` must be flat: nested layouts have no single extent per
        axis to hand to NumPy.

    Returns:
        A NumPy `ndarray` of dtype `dtype` with the tensor's shape.

    Raises:
        If NumPy is unavailable, or if the underlying NumPy calls fail.
    """
    comptime assert LayoutType.rank == LayoutType.flat_rank, (
        "to_numpy: nested layouts are not supported; flatten the layout before"
        " copying."
    )

    var shape = DynamicCoord[.int, LayoutType.rank]()

    comptime for i in range(LayoutType.rank):
        shape[i] = rebind[type_of(shape[i])](
            Int(Int(tensor.layout.shape[i]().value()))
        )

    var n = tensor.layout.size()

    if _is_row_major_contiguous(tensor):
        return copy_to_numpy_tensor(
            Span[Scalar[dtype], origin](unsafe_ptr=tensor.ptr, length=n), shape
        )

    # Gather into a contiguous buffer in C order, following the strides, then
    # hand that to the flat copy.
    var gathered = List[Scalar[dtype]](capacity=n)
    var read_coord = DynamicCoord[.int, LayoutType.rank]()

    for _ in range(n):
        gathered.append(Scalar[dtype](tensor.load[width=1](read_coord)))
        var carry = True

        comptime for i in reversed(range(LayoutType.rank)):
            if carry:
                var next = read_coord[i].value() + 1
                if Int(next) == Int(tensor.layout.shape[i]().value()):
                    next = 0
                else:
                    carry = False
                read_coord[i] = rebind[type_of(read_coord[i])](next)

    return copy_to_numpy_tensor(Span(gathered), shape)
