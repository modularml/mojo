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
"""NumPy interoperability helpers for Mojo collections.

These functions move flat numeric data between Mojo `Span` and NumPy
arrays when a Mojo program drives CPython (via `Python.import_module`), without
hand-written `ctypes` plumbing:

- `copy_to_numpy_array` builds a NumPy array from a Mojo `Span` by copying
  the data into a new, independent array.
- `from_numpy_array` borrows a NumPy array's buffer as a Mojo `Span`
  (zero-copy).

Only 1-D, C-contiguous arrays of the fixed-width numeric dtypes (`int8` through
`int64`, `uint8` through `uint64`, `float16`, `float32`, `float64`) are
supported. This targets the common case of handing computed numeric data to a
library such as `matplotlib`.
"""

from std.memory import unsafe_memcpy
from std.collections import Span, check_bounds
from std.utils.coord import Coord, CoordLike, DynamicCoord

from .python import Python
from .python_object import PythonObject

comptime _PY_C_CONTIGUOUS = "C_CONTIGUOUS"
comptime _PY_WRITEABLE = "WRITEABLE"


def _numpy_dtype_name[dtype: DType]() -> Optional[StaticString]:
    """Returns the NumPy dtype string for `dtype`, or `None` if unsupported."""
    if dtype == .int8:
        return StaticString("int8")
    elif dtype == .int16:
        return StaticString("int16")
    elif dtype == .int32:
        return StaticString("int32")
    elif dtype == .int64:
        return StaticString("int64")
    elif dtype == .uint8:
        return StaticString("uint8")
    elif dtype == .uint16:
        return StaticString("uint16")
    elif dtype == .uint32:
        return StaticString("uint32")
    elif dtype == .uint64:
        return StaticString("uint64")
    elif dtype == .float16:
        return StaticString("float16")
    elif dtype == .float32:
        return StaticString("float32")
    elif dtype == .float64:
        return StaticString("float64")
    else:
        return None


def _is_numpy_dtype[dtype: DType]() -> Bool:
    """Reports whether `dtype` maps to a supported NumPy dtype."""
    return _numpy_dtype_name[dtype]() is not None


# ===----------------------------------------------------------------------=== #
# copy_to_numpy_array
# ===----------------------------------------------------------------------=== #


def copy_to_numpy_array[
    dtype: DType, origin: Origin
](data: Span[Scalar[dtype], origin]) raises -> PythonObject:
    """Builds a 1-D NumPy array from a Mojo `Span` of scalars.

    The data is copied into a new, independent NumPy array, so the result
    remains valid after `data` is later mutated or freed. Unlike
    `from_numpy_array`, which returns a zero-copy view, this function does not
    alias `data`: mutating `data` after this call is not reflected in the
    returned array, and writes to the array are not reflected in `data`.

    Example:

    ```mojo
    from std.python.numpy import copy_to_numpy_array
    from std.math import sin

    var values = List[Float64](capacity=1024)
    for i in range(1024):
        var x = Float64(i) * 0.01
        values.append(sin(x) * sin(x))

    var arr = copy_to_numpy_array(values)  # an independent NumPy float64 array
    ```

    Parameters:
        dtype: The element dtype of the span (inferred).
        origin: The origin of the span (inferred).

    Args:
        data: The scalars to copy into a NumPy array.

    Constraints:
        `dtype` must be one of the fixed-width numeric dtypes supported by
        NumPy: `int8`-`int64`, `uint8`-`uint64`, `float16`, `float32`, or
        `float64`.

    Returns:
        A 1-D NumPy `ndarray` of dtype `dtype` and length `len(data)`.

    Raises:
        If NumPy is unavailable, or if the underlying NumPy calls fail.
    """
    comptime is_supported = _is_numpy_dtype[dtype]()
    comptime assert is_supported, String(
        "copy_to_numpy_array: unsupported dtype '",
        dtype,
        "'; expected a fixed-width numeric dtype (int8-int64, uint8-uint64,",
        " float16, float32, or float64). Note: `Int` is a machine-word integer",
        " and is not supported here — use a fixed-width scalar such as",
        " `Int64` (for example, `[Int64(i) for i in range(n)]`).",
    )

    var np = Python.import_module("numpy")
    var n = len(data)
    var dtype_str = String(_numpy_dtype_name[dtype]().value())

    if n == 0:
        return np.empty(0, dtype=dtype_str)

    var arr = np.empty(n, dtype=dtype_str)
    var dst = arr.ctypes.data.unsafe_get_as_pointer[dtype]()
    unsafe_memcpy(dest=dst, src=data.unsafe_ptr(), count=n)
    return arr


def copy_to_numpy_tensor[
    dtype: DType, origin: ImmOrigin, *shape_types: CoordLike
](
    data: Span[Scalar[dtype], origin], shape: Coord[*shape_types]
) raises -> PythonObject:
    """Builds an N-D NumPy array from a Mojo `Span` of scalars and a shape.

    The span supplies the elements in C order (last axis varying fastest) and
    `shape` supplies the extents, so `shape.product()` must equal `len(data)`.
    The data is copied into a new, independent NumPy array, so the result
    remains valid after `data` is later mutated or freed.

    Example:

    ```mojo
    from std.python.numpy import copy_to_numpy_tensor
    from std.utils.coord import Coord, Idx

    var values = List[Float64](capacity=6)
    for i in range(6):
        values.append(Float64(i))

    var arr = copy_to_numpy_tensor(values, Coord(Idx[2], Idx[3]))  # 2x3 array
    ```

    Dimensions may be compile-time (`Idx[N]`) or runtime (`Int`) in any mix.

    Parameters:
        dtype: The element dtype of the span (inferred).
        origin: The origin of the span (inferred).
        shape_types: The per-dimension types of `shape` (inferred).

    Args:
        data: The scalars to copy into a NumPy array, in C order.
        shape: The extents of the result, one element per axis.

    Constraints:
        `dtype` must be one of the fixed-width numeric dtypes supported by
        NumPy: `int8`-`int64`, `uint8`-`uint64`, `float16`, `float32`, or
        `float64`. `shape` must be flat (no nested `Coord`) and must not
        contain `All`.

    Returns:
        A NumPy `ndarray` of dtype `dtype` and shape `shape`.

    Raises:
        If `shape.product()` does not equal `len(data)`, if NumPy is
        unavailable, or if the underlying NumPy calls fail.
    """
    comptime is_supported = _is_numpy_dtype[dtype]()
    comptime assert is_supported, String(
        "copy_to_numpy_tensor: unsupported dtype '",
        dtype,
        "'; expected a fixed-width numeric dtype (int8-int64, uint8-uint64,",
        " float16, float32, or float64). Note: `Int` is a machine-word integer",
        " and is not supported here — use a fixed-width scalar such as",
        " `Int64` (for example, `[Int64(i) for i in range(n)]`).",
    )
    comptime assert shape.is_flat, (
        "copy_to_numpy_tensor: nested `Coord` shapes are not supported; pass"
        " one element per axis."
    )
    comptime assert not shape.contains_slices, (
        "copy_to_numpy_tensor: `All` is not a dimension; pass a concrete"
        " extent for every axis."
    )

    var np = Python.import_module("numpy")
    var dtype_str = _numpy_dtype_name[dtype]().value()

    var dims = List[PythonObject](capacity=shape.rank)
    comptime for i in range(shape.rank):
        dims.append(PythonObject(Int(shape[i].value())))

    var n = Int(shape.product())
    if n != len(data):
        raise Error(
            String(
                "copy_to_numpy_tensor: shape describes ",
                n,
                " elements but the span holds ",
                len(data),
                ".",
            )
        )

    var arr = np.empty(Python.list(dims), dtype=dtype_str)

    var dst = arr.ctypes.data.unsafe_get_as_pointer[dtype]()
    unsafe_memcpy(dest=dst, src=data.unsafe_ptr(), count=n)
    return arr


# ===----------------------------------------------------------------------=== #
# from_numpy_array
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct NumPyView[
    mut: Bool, //, dtype: DType, rank: Int, origin: Origin[mut=mut]
](ImplicitlyCopyable):
    """A borrowed view of a C-contiguous NumPy array's buffer and shape.

    `from_numpy_tensor` returns this rather than a bare `Span`, so the extents
    needed to index the buffer travel with it. `shape` is a `Coord`, so it can
    be handed straight back to `copy_to_numpy_tensor`.

    Parameters:
        mut: The mutability of the borrow.
        dtype: The element dtype.
        rank: The number of dimensions.
        origin: The origin of the borrow.
    """

    var data: Span[Scalar[Self.dtype], Self.origin]
    """The array's elements, in C order."""

    var shape: DynamicCoord[.int, Self.rank]
    """The extent of each axis. Subscript it with a compile-time index."""

    def __getitem__[
        *Ts: Intable
    ](self, *idx: *Ts) -> ref[Self.origin] Scalar[Self.dtype]:
        """Returns a reference to the element at `idx`.

        Parameters:
            Ts: The type of each index argument.

        Args:
            idx: One index per axis.

        Constraints:
            The number of indices must equal `rank`.

        Returns:
            A reference to the element, mutable if the view is.
        """
        comptime assert (
            Ts.length == Self.rank
        ), "NumPyView.__getitem__: expected one index per axis"

        var offset = 0

        comptime for i in range(Self.rank):
            var extent = Int(self.shape[i].value())
            var index = Int(idx[i])
            check_bounds(index, extent)
            offset = offset * extent + index

        return self.data[offset]


def _check_borrowable[
    dtype: DType, mut: Bool
](array: PythonObject, name: StaticString) raises:
    """Raises unless `array` can back a `Span[Scalar[dtype]]` borrow.

    Args:
        array: The NumPy array to check.
        name: The caller's name, used to prefix the error messages.

    Raises:
        If the dtype does not match, the array is not C-contiguous, or a
        mutable borrow is requested for a read-only array.
    """
    var actual_dtype = String(py=array.dtype)
    var expected_dtype = _numpy_dtype_name[dtype]().value()
    if actual_dtype != expected_dtype:
        raise Error(
            t"{name}: dtype mismatch: array is '{actual_dtype}' but"
            t" '{expected_dtype}' was requested"
        )

    if not Bool(py=array.flags[_PY_C_CONTIGUOUS]):
        raise Error(t"{name}: array must be C-contiguous")

    comptime if mut:
        if not Bool(py=array.flags[_PY_WRITEABLE]):
            raise Error(
                t"{name}: a mutable borrow requires a writable array (bind"
                t" `array` as a `read` argument to borrow a read-only array)"
            )


def from_numpy_array[
    mut: Bool,
    //,
    dtype: DType,
    origin: Origin[mut=mut],
](ref[origin] array: PythonObject) raises -> Span[Scalar[dtype], origin]:
    """Borrows a 1-D C-contiguous NumPy array as a Mojo `Span`.

    The returned span aliases the NumPy array's buffer; no bytes are copied. Its
    origin is tied to `array`, so the compiler keeps `array` alive for as long as
    the span is used; you must still not resize or reallocate `array` while the
    span is in use, or the span will dangle. Only pass arrays whose buffer is
    owned by NumPy (or another Python object).

    The borrow follows the mutability of the `array` reference: an immutable
    (`read`) reference yields a read-only span. A mutable reference
    yields a mutable span, so writes are visible to NumPy and vice versa.
    Creating a mutable span fails if the underlying NumPy array is not
    writable. Pass `array` as an immutable reference to avoid this error.

    Example:

    ```mojo
    from std.python import Python
    from std.python.numpy import from_numpy_array

    var np = Python.import_module("numpy")
    var array = np.arange(8, dtype="float64")
    var span = from_numpy_array[.float64](array)
    var total = Float64(0)
    for value in span:
        total += value
    ```

    Parameters:
        mut: The mutability of the borrow, inferred from `array`.
        dtype: The expected element dtype of the array.
        origin: The origin of the borrow, inferred from `array`.

    Args:
        array: A 1-D, C-contiguous NumPy `ndarray` whose dtype matches `dtype`.

    Constraints:
        `dtype` must be one of the fixed-width numeric dtypes supported by
        NumPy.

    Returns:
        A `Span` of length `array.size` viewing the array's buffer, with the same
        mutability and origin as the `array` binding.

    Raises:
        If `array` is not 1-D, is not C-contiguous, has a dtype that does not
        match `dtype`, or is not writable when borrowed mutably.
    """
    comptime is_supported = _is_numpy_dtype[dtype]()
    comptime assert is_supported, String(
        "from_numpy_array: unsupported dtype '",
        dtype,
        "'; expected a fixed-width numeric dtype (int8-int64, uint8-uint64,",
        " float16, float32, or float64).",
    )

    var ndim = Int(py=array.ndim)
    if ndim != 1:
        raise Error(
            String(t"from_numpy_array: expected a 1-D array, got ndim={ndim}")
        )

    _check_borrowable[dtype, mut](array, "from_numpy_array")

    var n = Int(py=array.size)
    # `unsafe_get_as_pointer` yields a mutable `MutAnyOrigin` pointer (the
    # buffer is Python-owned). Cast its mutability and origin to match the
    # inferred `origin` so the span's lifetime and mutability are checked
    # against the `array` binding.
    var ptr = array.ctypes.data.unsafe_get_as_pointer[dtype]()
    return Span[Scalar[dtype], origin](
        unsafe_ptr=ptr.unsafe_mut_cast[mut]().unsafe_origin_cast[origin](),
        length=n,
    )


def from_numpy_tensor[
    mut: Bool,
    //,
    dtype: DType,
    rank: Int,
    origin: Origin[mut=mut],
](ref[origin] array: PythonObject) raises -> NumPyView[dtype, rank, origin]:
    """Borrows an N-D C-contiguous NumPy array as a `NumPyView`.

    This is the N-D counterpart of `from_numpy_array`: the view aliases the
    array's whole buffer in C order (last axis varying fastest) and carries the
    extents needed to index it. No bytes are copied, and the same aliasing
    rules as `from_numpy_array` apply.

    Example:

    ```mojo
    from std.python import Python
    from std.python.numpy import from_numpy_tensor

    var np = Python.import_module("numpy")
    var array = np.arange(6, dtype="float64").reshape(2, 3)
    var view = from_numpy_tensor[.float64, 2](array)
    var value = view[1, 2]          # array[1, 2]
    var flat = view.data            # the buffer as a `Span`, in C order
    ```

    Parameters:
        mut: The mutability of the borrow, inferred from `array`.
        dtype: The expected element dtype of the array.
        rank: The expected number of dimensions.
        origin: The origin of the borrow, inferred from `array`.

    Args:
        array: A C-contiguous NumPy `ndarray` of `rank` dimensions whose dtype
            matches `dtype`.

    Constraints:
        `dtype` must be one of the fixed-width numeric dtypes supported by
        NumPy. `rank` must be positive.

    Returns:
        A `NumPyView` of the array's buffer and shape, with the same
        mutability and origin as the `array` binding.

    Raises:
        If `array.ndim` is not `rank`, is not C-contiguous, has a dtype that
        does not match `dtype`, or is not writable when borrowed mutably.
    """
    comptime is_supported = _is_numpy_dtype[dtype]()
    comptime assert is_supported, String(
        "from_numpy_tensor: unsupported dtype '",
        dtype,
        "'; expected a fixed-width numeric dtype (int8-int64, uint8-uint64,",
        " float16, float32, or float64).",
    )
    comptime assert rank > 0, String(
        "from_numpy_tensor: rank must be positive, got ", rank
    )

    var ndim = Int(py=array.ndim)
    if ndim != rank:
        raise Error(
            t"from_numpy_tensor: expected a {rank}-D array, got ndim={ndim}"
        )

    _check_borrowable[dtype, mut](array, "from_numpy_tensor")
    var np_shape = array.shape
    var shape = DynamicCoord[.int, rank]()

    comptime for i in range(rank):
        shape[i] = rebind[type_of(shape[i])](Int(py=np_shape[i]))

    var ptr = array.ctypes.data.unsafe_get_as_pointer[dtype]()
    var span = Span[Scalar[dtype], origin](
        unsafe_ptr=ptr.unsafe_mut_cast[mut]().unsafe_origin_cast[origin](),
        length=Int(py=array.size),
    )
    return NumPyView[dtype, rank, origin](span, shape)
