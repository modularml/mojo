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
# RUN: %mojo %s 8 string | FileCheck %s

from std.sys import argv


def takeItParams[
    UU: Trait, T: def[U: Trait](impl: U) -> Int, //
](state: T, x: UU):
    var product2 = state.__call__[UU](x)
    print(product2)


trait Trait(Deinitable, ImplicitlyCopyable):
    def get(self) -> Int:
        ...


@fieldwise_init
struct Impl(Trait):
    var x: Int

    def get(self) -> Int:
        return self.x


@fieldwise_init
struct Impl2(Trait):
    var x: String

    def get(self) -> Int:
        return self.x.byte_length()


# COM: Ensure parametric closures are supported
def captureParams[X: Trait, Y: Trait](impl2: X, mut impl3: Y):
    def hasParams[U: Trait](impl: U) {imm} -> Int:
        return impl.get() + impl2.get() + impl3.get()

    takeItParams(hasParams, impl2)


@fieldwise_init
struct Parameter[*, base: ImplicitlyCopyable & Deinitable & Writable](Copyable):
    var impl: Self.base

    def useIt(self):
        print(self.impl)


def takeIt[f: def() -> None](impl: f):
    impl()


def captureIt(p: Parameter[...]):
    @no_inline
    def closure() {imm p}:
        p.useIt()

    takeIt(closure)


trait Coord(Deinitable, ImplicitlyCopyable):
    def prettyPrint(self):
        ...


@fieldwise_init
struct Cartesian(Coord):
    var x: Int
    var y: Int

    def prettyPrint(self):
        print("Cart:", self.x, ",", self.y)


@fieldwise_init
struct Sphere(Coord):
    var theta: Int
    var phi: Int

    def prettyPrint(self):
        print("sphere:", self.theta, ",", self.phi)


def hasParamVariadic[C: def[*TT: Coord](* args: * TT)](impl: C, x: Int):
    var state1 = Sphere(x, x)
    var state2 = Cartesian(x, x)
    impl(state1, state2)


def testCapturedParamWithVariadicParamsFromFn(t: Int):
    def closureImpl[*TT: Coord](*args: *TT) {var}:
        print(args.__len__())
        comptime for i in range(args.__len__()):
            args[i].prettyPrint()

    hasParamVariadic(closureImpl, t)


def main() raises:
    var num = atol(argv()[1])
    var str = argv()[2]
    var p1 = Parameter[base=String](str)
    var p2 = Parameter[base=Int](num)
    # CHECK: string
    captureIt(p1)
    # CHECK: 8
    captureIt(p2)

    # COM: Ensure parametric closures are supported
    var x = Impl(num)
    var y = Impl2(str)
    # CHECK: 22
    captureParams(x, y)

    # COM: Ensure closures with variadic type parameters work
    # CHECK: 2
    # CHECK: sphere: 8 , 8
    # CHECK: Cart: 8 , 8
    testCapturedParamWithVariadicParamsFromFn(num)
