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

from std.utils import StaticTuple
from debug_test_utils import keep_alive


def main():
    var tuple = StaticTuple[Int16, 4](1, 2, 3, 4)
    var simd = SIMD[.int16, 4](1, 2, 3, 4)
    keep_alive(tuple, simd)  # breakpoint
