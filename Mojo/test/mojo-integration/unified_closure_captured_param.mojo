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
# RUN: %mojo %s 1 2 3 4 | FileCheck %s

from std.sys import argv


trait Coord(Deinitable, ImplicitlyCopyable):
    def prettyPrint(self):
        ...


trait Euclidean:
    def distance(self, other: Self) -> Int:
        ...


@fieldwise_init
struct Cartesian(Coord, Euclidean):
    var x: Int
    var y: Int

    def prettyPrint(self):
        print("Cart:", self.x, ",", self.y)

    def distance(self, other: Cartesian) -> Int:
        return self.x - other.x


@fieldwise_init
struct Sphere(Coord):
    var theta: Int
    var phi: Int

    def prettyPrint(self):
        print("sphere:", self.theta, ",", self.phi)


@fieldwise_init
struct Polar(Coord):
    var r: Int
    var theta: Int

    def prettyPrint(self):
        print("polar:", self.r, ",", self.theta)


# ===----------------------------------------------------------------------=== #
# Captured Param From Struct
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct DefinesParam[T: Coord, R: Coord]:
    var state: Self.T

    def method[C: def(arg: Self.T) -> Self.R](self, impl: C) -> Self.R:
        return impl(self.state)


def testCapturedParamFromStruct[T: Coord, R: Coord](t: T, r: R):
    def closureImpl(arg: T) {var} -> R:
        t.prettyPrint()
        return r

    var definesParam = DefinesParam[T, R](t)
    _ = definesParam.method(closureImpl)


# ===----------------------------------------------------------------------=== #
# Captured Param From Function
# ===----------------------------------------------------------------------=== #


def func[T: Coord, R: Coord, C: def(arg: T) -> R](impl: C, state: T) -> R:
    return impl(state)


def testCapturedParamFromFn[T: Coord, R: Coord](t: T, r: R):
    def closureImpl(arg: T) {var} -> R:
        t.prettyPrint()
        return r

    _ = func[T, R](closureImpl, t)


# ===----------------------------------------------------------------------=== #
# Captured Param From Nested Closure Call
# ===----------------------------------------------------------------------=== #


def funcWithNestedCall[
    T: Coord, R: Coord, C: def(arg: T) -> R
](impl: C, state: T) -> R:
    def kernel() {imm} -> R:
        return impl(state)

    return kernel()


def testCapturedParamFromNestedCall[T: Coord, R: Coord](t: T, r: R):
    def closureImpl(arg: T) {var} -> R:
        t.prettyPrint()
        return r

    _ = funcWithNestedCall[T, R](closureImpl, t)


# ===----------------------------------------------------------------------=== #
# Captured Param With Existing Params
# ===----------------------------------------------------------------------=== #


def hasParam[R: Coord, C: def[TT: Coord](arg: TT) -> R](impl: C, x: Int) -> R:
    var state = Sphere(x, x)
    return impl(state)


def testCapturedParamWithOtherParamsFromFn[R: Coord](t: Int, r: R):
    def closureImpl[TT: Coord](arg: TT) {var} -> R:
        arg.prettyPrint()
        return r

    _ = hasParam[R, type_of(closureImpl)](closureImpl, t)


def topLevelConcrete[TT: Coord](arg: TT) -> Cartesian:
    arg.prettyPrint()
    return Cartesian(6, 7)


def testTopLevelConcreteWithOtherParams(t: Int):
    var result = hasParam[Cartesian](topLevelConcrete, t)
    result.prettyPrint()


# ===----------------------------------------------------------------------=== #
# Captured Param Default
# ===----------------------------------------------------------------------=== #
def funcWithDefault[R: Coord, C: def[N: Int = 3]() -> R](impl: C) -> R:
    return impl()


def testCapturedParamFromFnWithDefault[R: Coord](r: R):
    def closureImpl[N: Int = 3]() {var} -> R:
        print(N)
        return r

    _ = funcWithDefault[R, type_of(closureImpl)](closureImpl)


# ===----------------------------------------------------------------------=== #
# Captured Param From Nested Closure
# ===----------------------------------------------------------------------=== #


def testNestedClosureCapture[RR: Coord](r: RR, x: Int):
    def l1[R: Coord](arg0: R) {var} -> R:
        def l2[TT: Coord](arg: TT) {var} -> R:
            arg.prettyPrint()
            return arg0

        return hasParam[R, type_of(l2)](l2, x)

    _ = l1[RR](r)


# ===----------------------------------------------------------------------=== #
# Nested def closure: capture outer type param (A) with inner param (C)
# ===----------------------------------------------------------------------=== #


trait _NestedCapPrintable(Deinitable, ImplicitlyCopyable):
    def string(self) -> String:
        ...

    @staticmethod
    def makeIt(x: Int) -> Self:
        ...


@fieldwise_init
struct _NestedCapStringWrapper(_NestedCapPrintable):
    var x: String

    def string(self) -> String:
        return self.x

    @staticmethod
    def makeIt(x: Int) -> Self:
        return _NestedCapStringWrapper(String(x))


@fieldwise_init
struct _NestedCapHasParam[T: _NestedCapPrintable](
    Deinitable, ImplicitlyCopyable
):
    var x: Self.T

    def foo(self):
        print(self.x.string())


def _testNestedCapture[A: _NestedCapPrintable]():
    def closure[C: _NestedCapHasParam[A]]() {imm}:
        var m = materialize[C]()
        m.foo()

    _consume[
        A,
        A.makeIt(1),
        A.makeIt(2),
        type_of(closure),
    ](closure)


def _consume[
    A: _NestedCapPrintable,
    implA: A,
    implB: A,
    FuncType: def[C: _NestedCapHasParam[A]](),
](impl: FuncType):
    comptime AA = _NestedCapHasParam[A](implA)
    comptime BB = _NestedCapHasParam[A](implB)
    impl[AA]()
    impl[BB]()


# ===----------------------------------------------------------------------=== #
# Lazy Conformance
# ===----------------------------------------------------------------------=== #


def testLazyConformance[NOT_T: Coord](something: NOT_T):
    def closureImpl(arg1: NOT_T) {var} -> Sphere:
        something.prettyPrint()
        return Sphere(33, 34)

    var definesParam = DefinesParam[NOT_T, Sphere](something)
    _ = definesParam.method(closureImpl)


def manyCaptures[
    A: Coord,
    B: Coord,
    D: Coord,
    F: def[C: Coord](a: A, b: B, c: C) -> D,
](impl: F, arg1: A, arg2: B, r: Int):
    var polar = Polar(r, r)
    var result = impl(arg1, arg2, polar)
    result.prettyPrint()


def testLazyConformanceManyCaptures[BB: Coord](arg: BB, a0: Cartesian, r: Int):
    def closure[CC: Coord](a1: Cartesian, b1: BB, c1: CC) {var} -> BB:
        a1.prettyPrint()
        b1.prettyPrint()
        return arg

    manyCaptures[Cartesian, BB, BB, type_of(closure)](closure, a0, arg, r)


def superset[B: Coord, D: Coord, F: def(b: B) -> D](impl: F, arg: B):
    var result = impl(arg)
    result.prettyPrint()


def testLazyConformanceSuperset[BB: Coord & Euclidean](arg: BB):
    def closure(b1: BB) {var} -> BB:
        return arg

    superset[BB, BB, type_of(closure)](closure, arg)


def main() raises:
    var one = atol(argv()[1])
    var two = atol(argv()[2])
    var three = atol(argv()[3])
    var four = atol(argv()[4])
    var x = Cartesian(one, two)
    var y = Sphere(three, four)
    var polar = Polar(two, two)

    # CHECK: Cart: 1 , 2
    testCapturedParamFromStruct(x, y)
    # CHECK: Cart: 1 , 2
    testCapturedParamFromFn(x, y)
    # CHECK: Cart: 1 , 2
    testCapturedParamFromNestedCall(x, y)
    # CHECK: sphere: 4 , 4
    testCapturedParamWithOtherParamsFromFn(four, y)
    # CHECK: sphere: 3 , 3
    # CHECK: Cart: 6 , 7
    testTopLevelConcreteWithOtherParams(three)
    # CHECK: 3
    testCapturedParamFromFnWithDefault(x)
    # CHECK: sphere: 1 , 1
    testNestedClosureCapture(x, one)
    # CHECK: Cart: 1 , 2
    testLazyConformance(x)
    # CHECK: Cart: 1 , 2
    # CHECK: polar: 2 , 2
    # CHECK: polar: 2 , 2
    testLazyConformanceManyCaptures(polar, x, one)
    # CHECK: Cart: 1 , 2
    testLazyConformanceSuperset(x)
    # CHECK: 1
    # CHECK: 2
    _testNestedCapture[_NestedCapStringWrapper]()
