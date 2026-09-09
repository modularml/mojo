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

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s

# This program should be parsed without triggering a parser cycle.


trait BarAble:
    comptime bar: Foo


# CHECK-LABEL: lit.struct.decl @BarViaTrait
struct BarViaTrait(BarAble, Movable where False):
    def __init__(out self):
        pass

    comptime bar: Foo = 10


struct Bar(Movable where False):
    @implicit
    def __init__[B: BarAble](out self, b: B):
        pass


struct Foo(Movable where False):
    @implicit
    def __init__(out self, value: Int):
        pass

    # This will force the body resolution ConformanceOp in BarViaTrait,
    # leading to the construction call emission for `bar: Foo = 10`
    def __init__(out self, value: Bar = BarViaTrait()):
        pass
