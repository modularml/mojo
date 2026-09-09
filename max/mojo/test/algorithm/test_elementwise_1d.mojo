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

from std.math import erf, exp, tanh
from std.sys.info import simd_width_of

from max.algorithm import elementwise
from max.gpu.host import DeviceContext
from std.testing import assert_almost_equal
from std.testing import TestSuite

from std.utils.coord import Coord


def test_elementwise_1d() raises:
    var ctx = DeviceContext(api="cpu")
    comptime num_elements = 64
    var data = List(length=num_elements, fill=Float32(0))

    var vector = Span(data)

    for i in range(len(vector)):
        vector[i] = Float32(i)

    @always_inline
    @__copy_capture(vector)
    @__parameter
    def func[simd_width: Int, alignment: Int = 1](idx: Coord):
        var vector_ptr: Pointer[vector.T, vector.origin] = vector.unsafe_ptr()
        var elem = vector_ptr.unsafe_load[width=simd_width](idx[0].value())
        var val = exp(erf(tanh(elem + 1)))
        vector_ptr.unsafe_store[width=simd_width](idx[0].value(), val)

    elementwise[func, simd_width_of[DType.float32]()](Coord(num_elements), ctx)

    assert_almost_equal(vector[0], 2.051446)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
