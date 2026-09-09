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

from std.sys import align_of, size_of
from std.testing import *


@fieldwise_init
@align(32)
struct AlignedTrivial(RegisterPassable):
    var value: Int


def main() raises:
    # print(align_of[AlignedTrivial]())  # 32
    # print(size_of[AlignedTrivial]())  # 32
    assert_equal(32, align_of[AlignedTrivial](), "align should be 32")
    assert_equal(32, size_of[AlignedTrivial](), "size_of should be 32")
