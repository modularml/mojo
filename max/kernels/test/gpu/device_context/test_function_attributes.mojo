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

from max.gpu import thread_idx
from std.testing import assert_equal

from max.gpu.host import DeviceContext
from max.gpu.host.func_attribute import Attribute


def test_function_attributes() raises:
    def kernel(x: MutPointer[Int, MutAnyOrigin]):
        x[0] = thread_idx.x

    with DeviceContext() as ctx:
        var func = ctx.compile_function[kernel]()
        assert_equal(func.get_attribute(Attribute.LOCAL_SIZE_BYTES), 0)


def main() raises:
    test_function_attributes()
