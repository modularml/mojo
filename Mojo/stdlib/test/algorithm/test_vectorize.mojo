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

from std.algorithm import vectorize
from std.memory import unsafe_memcmp
from std.testing import assert_equal
from std.testing import TestSuite
from std.math import iota
from std.sys.intrinsics import masked_load, masked_store


def test_vectorize() raises:
    # Create a mem of size 5
    var vector_stack: Array[Float32, 5] = [1.0, 2.0, 3.0, 4.0, 5.0]
    var vector = Span(vector_stack)

    @always_inline
    def add_two[width: Int](idx: Int) {var vector}:
        vector.unsafe_ptr().unsafe_store[width=width](
            idx, vector.unsafe_ptr().unsafe_load[width=width](idx) + 2
        )

    vectorize[2](len(vector), add_two)

    assert_equal(vector[0], 3.0)
    assert_equal(vector[1], 4.0)
    assert_equal(vector[2], 5.0)
    assert_equal(vector[3], 6.0)
    assert_equal(vector[4], 7.0)

    @always_inline
    def add[width: Int](idx: Int) {var vector}:
        vector.unsafe_ptr().unsafe_store[width=width](
            idx,
            vector.unsafe_ptr().unsafe_load[width=width](idx)
            + vector.unsafe_ptr().unsafe_load[width=width](idx),
        )

    vectorize[2](len(vector), add)

    assert_equal(vector[0], 6.0)
    assert_equal(vector[1], 8.0)
    assert_equal(vector[2], 10.0)
    assert_equal(vector[3], 12.0)
    assert_equal(vector[4], 14.0)


def test_vectorize_evl() raises:
    # Create a mem of size 5
    var vector_stack: Array[Float32, 5] = [1.0, 2.0, 3.0, 4.0, 5.0]
    var vector = Span(vector_stack)

    @always_inline
    def add_two[width: Int](idx: Int, evl: Int) {var vector}:
        if evl == width:
            vector.unsafe_ptr().unsafe_store[width=width](
                idx, vector.unsafe_ptr().unsafe_load[width=width](idx) + 2
            )
        else:
            for i in range(evl):
                vector[idx + i] = vector[idx + i] + 2

    vectorize[2](len(vector), add_two)

    assert_equal(vector[0], 3.0)
    assert_equal(vector[1], 4.0)
    assert_equal(vector[2], 5.0)
    assert_equal(vector[3], 6.0)
    assert_equal(vector[4], 7.0)

    @always_inline
    def add[width: Int](idx: Int, evl: Int) {var vector}:
        if evl == width:
            vector.unsafe_ptr().unsafe_store[width=width](
                idx,
                vector.unsafe_ptr().unsafe_load[width=width](idx)
                + vector.unsafe_ptr().unsafe_load[width=width](idx),
            )
        else:
            var ptr = vector.unsafe_ptr().unsafe_offset(idx)
            var incr = iota[.int32, width]()
            var mask = incr.lt(Int32(evl))
            var loaded = masked_load[width](ptr, mask, SIMD[.float32, width]())
            var result = loaded + loaded
            masked_store[width](result, ptr, mask)

    vectorize[2](len(vector), add)

    assert_equal(vector[0], 6.0)
    assert_equal(vector[1], 8.0)
    assert_equal(vector[2], 10.0)
    assert_equal(vector[3], 12.0)
    assert_equal(vector[4], 14.0)


def test_vectorize_unroll() raises:
    comptime buf_len = 23

    var vec_stack = Array[Float32, buf_len](fill={})
    var vec = Span(vec_stack)
    var buf_stack = Array[Float32, buf_len](fill={})
    var buf = Span(buf_stack)

    for i in range(buf_len):
        vec[i] = Float32(i)
        buf[i] = Float32(i)

    @always_inline
    def double_buf[simd_width: Int](idx: Int) {var buf}:
        buf.unsafe_ptr().unsafe_store[width=simd_width](
            idx,
            buf.unsafe_ptr().unsafe_load[width=simd_width](idx)
            + buf.unsafe_ptr().unsafe_load[width=simd_width](idx),
        )

    @always_inline
    def double_vec[simd_width: Int](idx: Int) {var vec}:
        vec.unsafe_ptr().unsafe_store[width=simd_width](
            idx,
            vec.unsafe_ptr().unsafe_load[width=simd_width](idx)
            + vec.unsafe_ptr().unsafe_load[width=simd_width](idx),
        )

    comptime simd_width = 4
    comptime unroll_factor = 2

    vectorize[simd_width, unroll_factor=unroll_factor](len(vec), double_vec)
    vectorize[simd_width](len(buf), double_buf)

    var err = unsafe_memcmp(vec.unsafe_ptr(), buf.unsafe_ptr(), len(buf))
    assert_equal(err, 0)


def test_vectorize_size_param() raises:
    var output = String()

    # remainder elements are correctly printed
    def printer[els: Int](n: Int) {mut output}:
        output.write(els, " ", n, "\n")

    vectorize[16, size=40](printer)
    assert_equal(output, "16 0\n16 16\n8 32\n")

    vectorize[16, size=8](printer)
    assert_equal(output, "16 0\n16 16\n8 32\n8 0\n")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
