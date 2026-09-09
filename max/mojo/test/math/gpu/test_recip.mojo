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

from std.math import recip

from max.gpu.host import DeviceContext
from std.testing import assert_almost_equal, TestSuite


def run_func[
    dtype: DType
](val: Scalar[dtype], ref_: Scalar[dtype], ctx: DeviceContext) raises:
    var out = ctx.enqueue_create_buffer[dtype](1)

    @__parameter
    def kernel(
        out_dev: Pointer[Scalar[dtype], MutAnyOrigin], lhs: Scalar[dtype]
    ):
        var result = recip(lhs)
        out_dev[unsafe_offset=0] = result

    ctx.enqueue_function[kernel](out, val, grid_dim=1, block_dim=1)
    with out.map_to_host() as out_host:
        assert_almost_equal(
            out_host[0],
            ref_,
            msg=String("while testing recip for the dtype ", dtype),
            atol=1e-8,
        )


def test_recip() raises:
    with DeviceContext() as ctx:
        run_func[.float64](8, 0.125, ctx)
        run_func[.float32](5, 0.2, ctx)
        run_func[.float16](-4, -0.25, ctx)
        run_func[.bfloat16](2, 0.5, ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
