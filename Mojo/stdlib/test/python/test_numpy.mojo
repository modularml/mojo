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

from std.python import Python, PythonObject
from std.python.numpy import (
    copy_to_numpy_array,
    copy_to_numpy_tensor,
    from_numpy_array,
    from_numpy_tensor,
    NumPyView,
)
from std.utils.coord import Coord, Idx
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    TestSuite,
)


def test_copy_to_numpy_array_float64() raises:
    var values: List[Float64] = [1.0, 2.5, -3.0, 4.25]
    var arr = copy_to_numpy_array(values)

    assert_equal(Int(py=arr.size), 4)
    assert_equal(String(py=arr.dtype), "float64")
    for i in range(len(values)):
        assert_almost_equal(Float64(py=arr[i]), values[i])


def test_copy_to_numpy_array_is_independent() raises:
    # A copy must not observe later mutations of the Mojo data.
    var values: List[Float64] = [1.0, 2.0, 3.0]
    var arr = copy_to_numpy_array(values)
    values[0] = 99.0
    assert_almost_equal(Float64(py=arr[0]), 1.0)


def test_copy_to_numpy_array_from_span() raises:
    var values: List[Float32] = [1.5, 2.5, 3.5]
    var arr = copy_to_numpy_array(Span(values))
    assert_equal(String(py=arr.dtype), "float32")
    assert_almost_equal(Float64(py=arr[1]), 2.5)


def test_copy_to_numpy_array_empty() raises:
    var values = List[Float64]()
    var arr = copy_to_numpy_array(values)
    assert_equal(Int(py=arr.size), 0)
    assert_equal(String(py=arr.dtype), "float64")


def test_copy_to_numpy_array_float16() raises:
    # `float16` is the dtype `ctypes` cannot express directly, so exercise it
    # explicitly. Values chosen to be exactly representable in float16.
    var values: List[Float16] = [1.0, 2.5, -0.5]
    var arr = copy_to_numpy_array(values)
    assert_equal(Int(py=arr.size), 3)
    assert_equal(String(py=arr.dtype), "float16")
    assert_almost_equal(Float64(py=arr[1]), 2.5)
    # The copy is independent of later mutations.
    values[0] = 7.0
    assert_almost_equal(Float64(py=arr[0]), 1.0)


def test_from_numpy_array_float64() raises:
    var np = Python.import_module("numpy")
    var array = np.arange(5, dtype="float64")
    var span = from_numpy_array[.float64](array)

    assert_equal(len(span), 5)
    var total = Float64(0)
    for value in span:
        total += value
    assert_almost_equal(total, 10.0)
    _ = array


def test_from_numpy_array_aliases() raises:
    # A borrow must observe NumPy-side writes, and vice versa.
    var np = Python.import_module("numpy")
    var array = np.zeros(3, dtype="int64")
    var span = from_numpy_array[.int64](array)
    array[1] = 7
    assert_equal(Int(span[1]), 7)
    span[2] = 9
    assert_equal(Int(py=array[2]), 9)
    _ = array


def test_from_numpy_array_dtype_mismatch_raises() raises:
    var np = Python.import_module("numpy")
    var array = np.arange(4, dtype="float64")
    with assert_raises(contains="dtype mismatch"):
        _ = from_numpy_array[.int32](array)
    _ = array


def test_from_numpy_array_non_contiguous_raises() raises:
    var np = Python.import_module("numpy")
    # A strided slice is not C-contiguous.
    var array = np.arange(10, dtype="float64")[::2]
    with assert_raises(contains="C-contiguous"):
        _ = from_numpy_array[.float64](array)
    _ = array


def test_from_numpy_array_wrong_ndim_raises() raises:
    var np = Python.import_module("numpy")
    var array = np.ones(Python.tuple(2, 3), dtype="float64")
    with assert_raises(contains="1-D"):
        _ = from_numpy_array[.float64](array)
    _ = array


def test_roundtrip_int64_signed() raises:
    var values: List[Int64] = [
        -9223372036854775807,
        -1,
        0,
        5000000000,
        9223372036854775807,
    ]
    var arr = copy_to_numpy_array(values)
    var span = from_numpy_array[.int64](arr)
    assert_equal(len(span), len(values))
    for i in range(len(values)):
        assert_equal(span[i], values[i])
    _ = arr


def test_from_numpy_array_empty() raises:
    var np = Python.import_module("numpy")
    var array = np.empty(0, dtype="float64")
    var span = from_numpy_array[.float64](array)
    assert_equal(len(span), 0)
    _ = array


def test_from_numpy_array_read_only_raises() raises:
    var np = Python.import_module("numpy")
    var array = np.arange(4, dtype="float64")
    _ = array.setflags(write=False)
    with assert_raises(contains="writable"):
        _ = from_numpy_array[.float64](array)
    _ = array


def _assert_dtype_name[dtype: DType](expected: StaticString) raises:
    var values: List[Scalar[dtype]] = [Scalar[dtype](1), Scalar[dtype](2)]
    var arr = copy_to_numpy_array(values)
    assert_equal(String(py=arr.dtype), String(expected))


def test_copy_to_numpy_array_dtype_names() raises:
    _assert_dtype_name[.int8]("int8")
    _assert_dtype_name[.int16]("int16")
    _assert_dtype_name[.int32]("int32")
    _assert_dtype_name[.int64]("int64")
    _assert_dtype_name[.uint8]("uint8")
    _assert_dtype_name[.uint16]("uint16")
    _assert_dtype_name[.uint32]("uint32")
    _assert_dtype_name[.uint64]("uint64")
    _assert_dtype_name[.float16]("float16")
    _assert_dtype_name[.float32]("float32")
    _assert_dtype_name[.float64]("float64")


def _first_via_read_borrow[
    dtype: DType
](array: PythonObject) raises -> Scalar[dtype]:
    var span = from_numpy_array[dtype](array)
    return span[0]


def test_from_numpy_array_read_only_borrow() raises:
    var np = Python.import_module("numpy")
    var array = np.arange(4, dtype="float64")
    _ = array.setflags(write=False)
    var first = _first_via_read_borrow[.float64](array)
    assert_almost_equal(first, 0.0)
    _ = array


def _iota[dtype: DType](count: Int) -> List[Scalar[dtype]]:
    var values = List[Scalar[dtype]](capacity=count)
    for i in range(count):
        values.append(Scalar[dtype](i))
    return values^


def test_copy_to_numpy_tensor_shape_and_order() raises:
    var values = _iota[.float64](6)
    var arr = copy_to_numpy_tensor(Span(values), Coord(Idx[2], Idx[3]))

    assert_equal(Int(py=arr.ndim), 2)
    assert_equal(Int(py=arr.shape[0]), 2)
    assert_equal(Int(py=arr.shape[1]), 3)
    assert_equal(String(py=arr.dtype), "float64")
    # C order: the span's element `r * 3 + c` lands at `[r, c]`.
    for r in range(2):
        for c in range(3):
            assert_almost_equal(Float64(py=arr[r][c]), Float64(r * 3 + c))


def test_copy_to_numpy_tensor_dynamic_dim() raises:
    var values = _iota[.float32](6)
    var arr = copy_to_numpy_tensor(Span(values), Coord(Idx[3], Int(2)))
    assert_equal(Int(py=arr.shape[0]), 3)
    assert_equal(Int(py=arr.shape[1]), 2)
    assert_equal(String(py=arr.dtype), "float32")


def test_copy_to_numpy_tensor_rank3() raises:
    var values = _iota[.int64](8)
    var arr = copy_to_numpy_tensor(Span(values), Coord(Idx[2], Idx[2], Idx[2]))
    assert_equal(Int(py=arr.ndim), 3)
    assert_equal(Int(py=arr.size), 8)
    assert_equal(Int(py=arr[1][1][1]), 7)


def test_copy_to_numpy_tensor_is_independent() raises:
    var values = _iota[.float64](4)
    var arr = copy_to_numpy_tensor(Span(values), Coord(Idx[2], Idx[2]))
    values[0] = 99.0
    assert_almost_equal(Float64(py=arr[0][0]), 0.0)


def test_copy_to_numpy_tensor_size_mismatch_raises() raises:
    var values = _iota[.float64](6)
    with assert_raises(contains="shape describes 4 elements"):
        _ = copy_to_numpy_tensor(Span(values), Coord(Idx[2], Idx[2]))


def test_copy_to_numpy_tensor_zero_extent() raises:
    var values = List[Float64]()
    var arr = copy_to_numpy_tensor(Span(values), Coord(Idx[0], Idx[3]))
    assert_equal(Int(py=arr.size), 0)
    assert_equal(Int(py=arr.shape[0]), 0)
    assert_equal(Int(py=arr.shape[1]), 3)


def test_from_numpy_tensor_float64() raises:
    var np = Python.import_module("numpy")
    var array = np.arange(6, dtype="float64").reshape(2, 3)
    var view = from_numpy_tensor[.float64, 2](array)

    assert_equal(len(view.data), 6)
    assert_equal(view.shape[0], 2)
    assert_equal(view.shape[1], 3)
    for r in range(2):
        for c in range(3):
            assert_almost_equal(view[r, c], Float64(r * 3 + c))
    for i in range(6):
        assert_almost_equal(view.data[i], Float64(i))
    _ = array


def test_from_numpy_tensor_aliases() raises:
    var np = Python.import_module("numpy")
    var array = np.zeros(Python.tuple(2, 2), dtype="float64")
    var view = from_numpy_tensor[.float64, 2](array)
    view[1, 1] = 5.0
    assert_almost_equal(Float64(py=array[1][1]), 5.0)
    _ = array


def test_from_numpy_tensor_wrong_rank_raises() raises:
    var np = Python.import_module("numpy")
    var array = np.arange(6, dtype="float64").reshape(2, 3)
    with assert_raises(contains="expected a 3-D array"):
        _ = from_numpy_tensor[.float64, 3](array)
    _ = array


def test_from_numpy_tensor_non_contiguous_raises() raises:
    var np = Python.import_module("numpy")
    var array = np.arange(12, dtype="float64").reshape(3, 4)[:, ::2]
    with assert_raises(contains="C-contiguous"):
        _ = from_numpy_tensor[.float64, 2](array)
    _ = array


def test_from_numpy_tensor_dtype_mismatch_raises() raises:
    var np = Python.import_module("numpy")
    var array = np.arange(6, dtype="float64").reshape(2, 3)
    with assert_raises(contains="dtype mismatch"):
        _ = from_numpy_tensor[.int32, 2](array)
    _ = array


def test_numpy_tensor_roundtrip() raises:
    var values = _iota[.int32](12)
    var arr = copy_to_numpy_tensor(Span(values), Coord(Idx[3], Int(4)))
    var view = from_numpy_tensor[.int32, 2](arr)
    assert_equal(view.shape[0], 3)
    assert_equal(view.shape[1], 4)
    assert_equal(len(view.data), len(values))
    for i in range(len(values)):
        assert_equal(view.data[i], values[i])
    _ = arr


def _sum_view[
    dtype: DType, rank: Int, origin: Origin
](view: NumPyView[dtype, rank, origin]) -> Scalar[dtype]:
    var total = Scalar[dtype](0)
    for value in view.data:
        total += value
    return total


def test_numpy_view_passes_by_borrow() raises:
    # The view crosses a function boundary without a copy or a `ref` rebind.
    var np = Python.import_module("numpy")
    var array = np.arange(6, dtype="float64").reshape(2, 3)
    var view = from_numpy_tensor[.float64, 2](array)
    assert_almost_equal(_sum_view(view), 15.0)
    # The view is still usable after being borrowed.
    assert_almost_equal(view[1, 2], 5.0)
    _ = array


def test_numpy_view_rank3_indexing() raises:
    var np = Python.import_module("numpy")
    var array = np.arange(24, dtype="int64").reshape(2, 3, 4)
    var view = from_numpy_tensor[.int64, 3](array)
    assert_equal(len(view.data), 24)
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert_equal(view[i, j, k], Int64(i * 12 + j * 4 + k))
    _ = array


def test_numpy_view_shape_feeds_copy() raises:
    # A view's shape is a `Coord`, so it needs no conversion to go back.
    var np = Python.import_module("numpy")
    var array = np.arange(6, dtype="float64").reshape(2, 3)
    var view = from_numpy_tensor[.float64, 2](array)
    var copy = copy_to_numpy_tensor(view.data, view.shape)
    assert_equal(Int(py=copy.shape[0]), 2)
    assert_equal(Int(py=copy.shape[1]), 3)
    assert_almost_equal(Float64(py=copy[1][2]), 5.0)
    _ = array


def test_from_numpy_array_equivalent_dtype() raises:
    # `longlong` is a distinct NumPy dtype that compares equal to `int64`;
    # the borrow must accept it.
    var np = Python.import_module("numpy")
    var array = np.arange(4, dtype="longlong")
    var span = from_numpy_array[.int64](array)
    assert_equal(len(span), 4)
    assert_equal(span[3], 3)
    _ = array


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
