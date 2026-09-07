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

from std.testing import assert_equal
from layout import Layout


def lookup_fn[idx: Int](value: Int) -> Int:
    comptime my_constants: List[Int] = [3, 6, 9]
    return comptime (my_constants[idx]) * value


def layout_size() -> Int:
    comptime layout = Layout.row_major(16, 8)
    var size = comptime (layout.size())
    return size


def main() raises:
    var x = lookup_fn[1](4)
    assert_equal(x, 24)

    var y = layout_size()
    assert_equal(y, 128)
