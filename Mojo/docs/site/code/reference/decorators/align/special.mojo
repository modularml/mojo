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


@fieldwise_init
@align(1)
struct MinimalAlign:
    var x: Int


def main() raises:
    # print(align_of[MinimalAlign]())  # Prints 8
    assert_equal(8, align_of[MinimalAlign](), "align should be 8")
