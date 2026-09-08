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

from std.sys import align_of
from std.testing import *


@align(4)
struct TryToReduce:
    var x: Int  # Int has 8-byte natural alignment


def main() raises:
    # print(align_of[TryToReduce]())  # Prints 8
    assert_equal(align_of[TryToReduce](), 8, "align should be 8")
