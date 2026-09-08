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

from std.builtin.globals import global_constant
from std.testing import assert_equal


def use_lookup(idx: Int) -> Int64:
    comptime numbers: Array[Int64, 10] = [
        1,
        3,
        14,
        34,
        63,
        101,
        148,
        204,
        269,
        343,
    ]
    ref lookup_table = global_constant[numbers]()
    if idx < len(lookup_table):
        return lookup_table[idx]
    else:
        return 0


def main() raises:
    var x = use_lookup(3)
    assert_equal(x, 34)
