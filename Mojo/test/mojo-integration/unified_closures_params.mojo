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
# RUN: %mojo %s 1 4 | FileCheck %s

from std.sys import argv


trait Config(Deinitable):
    comptime R: AnyType

    @staticmethod
    def makeIt() -> Self:
        ...


@fieldwise_init
struct MyConfig(Config, ImplicitlyCopyable):
    comptime R: AnyType = Int
    var item: Self.R

    def thing(self) -> Self.R:
        return self.item

    @staticmethod
    def makeIt() -> Self:
        return MyConfig(3)


def chain[
    R: AnyType,
    Inner: Config,
    Outer: def(Inner),
    //,
](outer: Outer) where R == Inner.R:
    var c = Inner.makeIt()
    outer(c)


def testParamInf():
    def outer(i: MyConfig) {var}:
        print(i.thing())

    chain(outer)


def hasOrigin[F: def[T: MutOrigin](TypeWithOrigin[T]) -> None, //](f: F):
    f[MutAnyOrigin](TypeWithOrigin[MutAnyOrigin]())


@fieldwise_init
struct TypeWithOrigin[T: MutOrigin](ImplicitlyCopyable, Movable):
    var isMutable: Bool

    def __init__(out self):
        self.isMutable = Self.T.mut


def takeIt[f: ImplicitlyCopyable & def(z: Int) -> Int](impl: f, y: Int):
    print(impl(y))


@no_inline
def aThing[f: def(Int) capturing -> Int](y: Int):
    def aClosure(z: Int) {var} -> Int:
        return f(y)

    takeIt(aClosure, y)


@no_inline
def itCaptures[THREE: Int](one: Int, four: Int):
    @__parameter
    def aParam(z: Int) -> Int:
        return THREE + four + z

    aThing[aParam](one)

    # COM: Ensure nesting in legacy closure does not corrupt the symbol calculation
    comptime if THREE == 3:

        @__copy_capture(one, four)
        @__parameter
        def aParam2(zz: Int) -> Int:
            def thing(z: Int) {var zz} -> Int:
                return zz

            takeIt(thing, four)
            return one + one

        aThing[aParam2](one)


trait Coordinate:
    def prettyPrint(self):
        ...


@fieldwise_init
struct Cartesian(Coordinate, ImplicitlyCopyable):
    var x: Int
    var y: Int

    def prettyPrint(self):
        print("{", self.x, ",", self.y, "}")


@fieldwise_init
struct Polar[y: Int](Coordinate, ImplicitlyCopyable):
    var x: Int

    def prettyPrint(self):
        print("{", self.x, ",", self.y, "}")


def useDefinesCapturingParamClosure[
    X: Coordinate & ImplicitlyCopyable & Deinitable, C: def() -> X
](impl: C):
    var coordinate = impl()
    coordinate.prettyPrint()


def definesCapturingParamClosure[
    X: Coordinate & ImplicitlyCopyable & Deinitable
](something: X, one: Int) raises:
    def closureImpl() {var} -> X:
        return something

    # COM: check that concrete types can conform to traits with aliases
    def closureConcreteImpl() {var} -> Cartesian:
        return Cartesian(one, one)

    useDefinesCapturingParamClosure[X, type_of(closureImpl)](closureImpl)
    useDefinesCapturingParamClosure[Cartesian, type_of(closureConcreteImpl)](
        closureConcreteImpl
    )


def usesParamRefClosure[
    T: Coordinate & ImplicitlyCopyable & Deinitable,
    C: def[x: Int, Y: Coordinate](xx: T, unused: Y) -> Polar[x],
](impl: C, value: T):
    var result = impl[3, Cartesian](value, Cartesian(3, 3))
    result.prettyPrint()


def definesParamRefClosure[
    T: Coordinate & ImplicitlyCopyable & Deinitable
](value: T):
    def closureImpl[x: Int, Y: Coordinate](xx: T, unused: Y) {var} -> Polar[x]:
        _ = value
        return Polar[x](x)

    usesParamRefClosure[T, type_of(closureImpl)](closureImpl, value)


def takesThin[
    T: ImplicitlyCopyable & Writable, FuncType: def(T) -> None
](impl: FuncType, x: T):
    impl(x)


def callTakesThin[T: ImplicitlyCopyable & Writable](x: T):
    def takesItem(item: T):
        print(item)

    takesThin[T, type_of(takesItem)](takesItem, x)


@fieldwise_init
struct HasParamRank[N: Int](ImplicitlyCopyable):
    comptime rank = Self.N + Self.N

    def printMe(self):
        print(Self.rank)


def consumeHasParamRank[
    c: Int, r: Int, FuncType: def(a: HasParamRank[c]) -> HasParamRank[r]
](impl: FuncType):
    var p = impl(HasParamRank[c]())
    p.printMe()


def closureFromCapturedInt[a: Int, b: Int, c: Int](x: Int):
    def nested(a: HasParamRank[a + b]) {var} -> HasParamRank[b + c]:
        _ = x
        return HasParamRank[b + c]()

    consumeHasParamRank[a + b, b + c, type_of(nested)](nested)


def must_be_read_only_with_origin[
    Mut: Bool, //, o: Origin[mut=Mut], FuncType: def() -> None
](impl: FuncType, ptr: Pointer[Int, o, address_space=.GENERIC]):
    impl()


def demo_origin_closure[
    o: Origin[mut=True]
](ptr: Pointer[Int, o, address_space=.GENERIC]):
    var immut_ptr = ptr.as_imm()

    def read() {imm immut_ptr}:
        print("read only", immut_ptr[unsafe_offset=0])

    must_be_read_only_with_origin(read, immut_ptr)


def sinkClosureResult[
    T: ImplicitlyCopyable & Deinitable, F: def() -> T
](*, call: F) -> T:
    return call()


# COM: Forward a closure-typed parameter whose result type is only known equal
# COM: to the enclosing scope's `T` through the captured-parameter `where`
# COM: clause `eq(G.T, T)`.
def forwardClosureResult[
    T: ImplicitlyCopyable & Deinitable, //, G: def() -> T
](*, call: G) -> T:
    return sinkClosureResult(call=call)


# COM: Same forward, but the result is declared with the witness spelling `G.T`.
def forwardClosureResultWitness[
    T: ImplicitlyCopyable & Deinitable, //, G: def() -> T
](*, call: G) -> G.T:
    return sinkClosureResult(call=call)


def addsViaForwardedClosure(x: Int, y: Int) -> Int:
    def make() {var} -> Int:
        return x + y

    return forwardClosureResult(call=make)


def addsViaForwardedClosureWitness(x: Int, y: Int) -> Int:
    def make() {var} -> Int:
        return x + y

    return forwardClosureResultWitness(call=make)


@fieldwise_init
struct Boxed[T: ImplicitlyCopyable & Deinitable](
    Deinitable, ImplicitlyCopyable
):
    var value: Self.T


# COM: Sink whose signature carries the nesting: the leaf `T` binds directly,
# COM: so no lazy conformance (binding `T := Boxed[T]`) is required.
def sinkBoxedResult[
    T: ImplicitlyCopyable & Deinitable, F: def() -> Boxed[T]
](*, call: F) -> Boxed[T]:
    return call()


# COM: Forward a closure whose result is the *nested* type `Boxed[T]`. The
# COM: captured-parameter `where` clause binds only the leaf (`eq(G.T, T)`), so
# COM: the forwarded call comes back spelled `Boxed[G.T]`; the call-site rebind
# COM: externs the capture structurally back to `Boxed[T]`.
def forwardBoxedResult[
    T: ImplicitlyCopyable & Deinitable, //, G: def() -> Boxed[T]
](*, call: G) -> Boxed[T]:
    return sinkBoxedResult(call=call)


def addsViaForwardedBoxedClosure(x: Int, y: Int) -> Int:
    def make() {var} -> Boxed[Int]:
        return Boxed(x + y)

    return forwardBoxedResult(call=make).value


def main() raises:
    var one = atol(argv()[1])
    var four = atol(argv()[2])
    # CHECK: 8
    # CHECK: 1
    # CHECK: 2
    itCaptures[3](one, four)

    # Ensure origins are lowered
    # CHECK: True
    def closure[T: MutOrigin](_bar: TypeWithOrigin[T]) {imm}:
        print(_bar.isMutable)

    hasOrigin(closure)

    # COM: Test rebinds to traits with captures.
    # CHECK: { 1 , 4 }
    # CHECK: { 1 , 1 }
    definesCapturingParamClosure(Cartesian(one, four), one)

    # COM: Test param ref matching: both trait and impl use param types.
    # CHECK: { 3 , 3 }
    definesParamRefClosure(Cartesian(one, four))

    # COM: Test thin closure with captured type parameter.
    # CHECK: 1
    callTakesThin[Int](one)

    # COM: Test captured closure with param expressions in sugar-only space.
    # CHECK: 10
    closureFromCapturedInt[1, 2, 3](one)

    # COM: Test closure capture with pointer origins.
    # CHECK: read only 1
    var ptr = Pointer(to=one)
    demo_origin_closure(ptr)

    # CHECK: 3
    testParamInf()

    # COM: Forward a closure result through a captured-parameter `where` clause.
    # CHECK: 5
    print(addsViaForwardedClosure(one, four))

    # COM: Same forward, but the return type is spelled as the witness `G.T`.
    # CHECK: 5
    print(addsViaForwardedClosureWitness(one, four))

    # COM: Forward a closure whose result is the nested type `Boxed[T]`.
    # CHECK: 5
    print(addsViaForwardedBoxedClosure(one, four))
