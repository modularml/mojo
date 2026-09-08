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
@align(128)
struct Aligned[T: Copyable & Deinitable]:
    var value: Self.T


def main() raises:
    # print(align_of[Aligned[Int8]]())  # 128
    # print(align_of[Aligned[Int64]]())  # 128

    assert_equal(128, align_of[Aligned[Int8]](), "align should be 128")
    assert_equal(128, align_of[Aligned[Int64]](), "align should be 128")
