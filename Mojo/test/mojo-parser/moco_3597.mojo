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


trait CoordLike(ImplicitlyCopyable, Movable):
    pass


struct Coord[*element_types: CoordLike]():
    @implicit
    @always_inline("nodebug")
    def __init__(out self, var tuple: Tuple[*Self.element_types]):
        pass


def callee(shape: Coord) raises:
    pass


def caller[MType: CoordLike, NType: CoordLike](m: MType, n: NType) raises:
    # CHECK: lit.call @{{.*}}@"callee[KGENParamList[{{.*}}::CoordLike & ::AnyType & ::Copyable & ::ImplicitlyCopyable & ::Movable]
    callee((m, n))
