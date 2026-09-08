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

from std.math import iota


def test_matrix(ptr: MutPointer[Int32, MutAnyOrigin], rows: Int, cols: Int):
    # CHECK: [0, 1, 2, 3]
    print(ptr.load[width=4](0 * cols + 0))
    # CHECK: [4, 5, 6, 7]
    print(ptr.load[width=4](1 * cols + 0))
    # CHECK: [8, 9, 10, 11]
    print(ptr.load[width=4](2 * cols + 0))
    # CHECK: [12, 13, 14, 15]
    print(ptr.load[width=4](3 * cols + 0))

    var v = iota[.int32, 4]()
    ptr.store[width=4](3 * cols + 0, v)
    # CHECK: [0, 1, 2, 3]
    print(ptr.load[width=4](3 * cols + 0))


def test_matrix_static():
    print("== test_matrix_static")
    var data = Array[Int32, 16](fill_with=lambda (i: Int) -> Int32: Int32(i))
    test_matrix(data.unsafe_ptr().as_unsafe_any_origin(), 4, 4)


def test_matrix_dynamic():
    print("== test_matrix_dynamic")
    var data = Array[Int32, 16](fill_with=lambda (i: Int) -> Int32: Int32(i))
    test_matrix(data.unsafe_ptr().as_unsafe_any_origin(), 4, 4)


def test_matrix_dynamic_shape():
    print("== test_matrix_dynamic_shape")
    var data = Array[Int32, 16](fill_with=lambda (i: Int) -> Int32: Int32(i))
    test_matrix(data.unsafe_ptr().as_unsafe_any_origin(), 4, 4)


def main():
    test_matrix_static()
    test_matrix_dynamic()
    test_matrix_dynamic_shape()
