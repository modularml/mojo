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


# RUN: %parse-mojo-isolated %s | FileCheck %s


struct SomeStruct[x: Int, y: Int](Movable where False):
    @implicit
    def __init__(out self: SomeStruct[3, 4], other: SomeStruct[1, 1]):
        pass


def take_x_and_plus_one[x: Int](s: SomeStruct[x, x + 1]):
    pass


def test_some_struct(s: SomeStruct[1, 1]):
    # This should compile to a call
    # CHECK: lit.call @moco3113::@"take_x_and_plus_one{{.*}}#SomeStruct <:!Int {:scalar<index> 3}, :!Int {:scalar<index> 4}>>
    take_x_and_plus_one(s)
