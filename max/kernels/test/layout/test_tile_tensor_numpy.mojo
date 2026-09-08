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

from layout import TileTensor, from_numpy, row_major, to_numpy
from layout.coord import Coord, Idx
from std.python import Python, PythonObject
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    TestSuite,
)


def test_from_numpy_shape_and_values() raises:
    var np = Python.import_module("numpy")
    var array = np.arange(6, dtype="float64").reshape(2, 3)
    var tensor = from_numpy[.float64, 2](array)

    assert_equal(tensor.layout.size(), 6)
    for r in range(2):
        for c in range(3):
            assert_almost_equal(tensor[r, c], Float64(r * 3 + c))
    _ = array


def test_from_numpy_rank3() raises:
    var np = Python.import_module("numpy")
    var array = np.arange(24, dtype="int64").reshape(2, 3, 4)
    var tensor = from_numpy[.int64, 3](array)

    assert_equal(tensor.layout.size(), 24)
    assert_equal(tensor[1, 2, 3], 23)
    _ = array


def test_from_numpy_aliases() raises:
    # The borrow is zero-copy, so writes are visible on both sides.
    var np = Python.import_module("numpy")
    var array = np.zeros(Python.tuple(2, 2), dtype="float64")
    var tensor = from_numpy[.float64, 2](array)
    tensor[1, 1] = 5.0
    assert_almost_equal(Float64(py=array[1][1]), 5.0)
    _ = array


def test_from_numpy_wrong_rank_raises() raises:
    var np = Python.import_module("numpy")
    var array = np.arange(6, dtype="float64").reshape(2, 3)
    with assert_raises(contains="expected a 3-D array"):
        _ = from_numpy[.float64, 3](array)
    _ = array


def test_from_numpy_non_contiguous_raises() raises:
    var np = Python.import_module("numpy")
    var array = np.arange(12, dtype="float64").reshape(3, 4)[:, ::2]
    with assert_raises(contains="C-contiguous"):
        _ = from_numpy[.float64, 2](array)
    _ = array


def test_to_numpy_contiguous() raises:
    var storage = Array[Float32, 6](
        fill_with=lambda (i: Int) -> Float32: Float32(i)
    )
    var tensor = TileTensor(Span(storage), row_major(Coord(Idx[2], Idx[3])))
    var array = to_numpy(tensor)

    assert_equal(Int(py=array.ndim), 2)
    assert_equal(Int(py=array.shape[0]), 2)
    assert_equal(Int(py=array.shape[1]), 3)
    assert_equal(String(py=array.dtype), "float32")
    for r in range(2):
        for c in range(3):
            assert_almost_equal(Float64(py=array[r][c]), Float64(r * 3 + c))


def test_to_numpy_is_independent() raises:
    var storage = Array[Float32, 4](
        fill_with=lambda (i: Int) -> Float32: Float32(i)
    )
    var tensor = TileTensor(Span(storage), row_major(Coord(Idx[2], Idx[2])))
    var array = to_numpy(tensor)
    storage[0] = 99.0
    assert_almost_equal(Float64(py=array[0][0]), 0.0)


def test_to_numpy_gathers_non_contiguous() raises:
    # A tile has the parent's strides, so it is not row-major contiguous and
    # must be gathered element by element.
    var storage = Array[Float32, 24](
        fill_with=lambda (i: Int) -> Float32: Float32(i)
    )
    var tensor = TileTensor(
        Span(storage), row_major(Coord(Idx[2], Idx[3], Idx[4]))
    )
    var tile = tensor.tile[1, 2, 2](0, 0, 0)
    var array = to_numpy(tile)

    assert_equal(Int(py=array.shape[0]), 1)
    assert_equal(Int(py=array.shape[1]), 2)
    assert_equal(Int(py=array.shape[2]), 2)
    # Offsets 0, 1, 4, 5 under strides (12, 4, 1).
    assert_almost_equal(Float64(py=array[0][0][0]), 0.0)
    assert_almost_equal(Float64(py=array[0][0][1]), 1.0)
    assert_almost_equal(Float64(py=array[0][1][0]), 4.0)
    assert_almost_equal(Float64(py=array[0][1][1]), 5.0)


def test_numpy_roundtrip() raises:
    var np = Python.import_module("numpy")
    var array = np.arange(12, dtype="int32").reshape(3, 4)
    var tensor = from_numpy[.int32, 2](array)
    var back = to_numpy(tensor)

    assert_equal(Int(py=back.shape[0]), 3)
    assert_equal(Int(py=back.shape[1]), 4)
    assert_equal(String(py=back.dtype), "int32")
    for r in range(3):
        for c in range(4):
            assert_equal(Int(py=back[r][c]), r * 4 + c)
    _ = array


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
