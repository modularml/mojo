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

# RUN: %mojo -debug-level full %s 4 5 | FileCheck %s

from std.collections import OptionalReg
from std.sys import argv


trait Position(TrivialRegisterPassable):
    def foo(self) -> Self:
        ...


struct PositionImpl(Position, TrivialRegisterPassable):
    var x: Int
    var y: Int

    def __init__(out self, x: Int, y: Int):
        self.x = x
        self.y = y

    @no_inline
    def foo(self) -> Self:
        print(self.x, self.y)
        return self


def foo[position_t: Position](x: position_t) -> OptionalReg[position_t]:
    var xx = OptionalReg[position_t](x)
    _ = xx.value().foo()
    return xx


def main() raises:
    # CHECK: 4 5
    var pi = PositionImpl(atol(argv()[1]), atol(argv()[2]))
    _ = foo(pi)
