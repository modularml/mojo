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
# RUN: %mojo %s 3 1 4 | FileCheck %s

from std.sys import argv


trait ATrait(Movable):
    # In order for a struct that depends on a capturing closure
    # to conform to a trait, all the methods of that trait must be
    # marked as capturing. This is temporary until we remove the capturing
    # effect. Note that the legacy closures are responsible for this restriction.
    # In particular, the following is not supported:
    # trait ATrait(Movable):
    #     def my_method(self) -> Int:
    #         ...

    # struct ParamStruct[func: def (x: Int) capturing -> Int](ATrait):
    #     def my_method(self) -> Int:
    #         return func(2)
    def my_method(self) capturing -> Int:
        ...


struct AStruct[func: def(x: Int) -> Int](ATrait):
    var myFunc: Self.func

    def __init__(out self, var x: Self.func):
        self.myFunc = x^

    def my_method(self) -> Int:
        return self.myFunc(3)


def takeIt[T: ATrait](impl: T):
    print(impl.my_method())


# COM: Test the capturing effect is propagated through to trait methods
trait DefinesClosure(def(z: Int) -> Int):
    pass


@fieldwise_init
struct DefinesClosureImpl(DefinesClosure):
    var x: Int

    def __call__(self, z: Int) -> Int:
        return z + self.x


def takeIt[f: def(z: Int) -> Int](impl: f, y: Int):
    print(impl(y))


# COM: Ensure closures work when a RegisterPassable struct forwards
# COM: a concrete type argument through a generic closure parameter.
@fieldwise_init
struct RegPassWrapper[U: RegisterPassable & Deinitable](
    RegisterPassable,
):
    var u: Self.U

    def apply_fn[FuncType: def(Self.U) -> Bool](self, func: FuncType) -> Bool:
        return func(self.u)


def testRegisterPassableUnifiedClosureAdaptor():
    def always_true(x: Int) {} -> Bool:
        return True

    var wrapper = RegPassWrapper(5)
    print(wrapper.apply_fn(always_true))


# COM: A parametric closure field whose signature captures a struct parameter
# COM: (`dtype`).
@fieldwise_init
struct ParametricClosureField[
    dtype: DType,
    F: ImplicitlyCopyable & Deinitable & (def[w: Int]() -> SIMD[dtype, w]),
](Deinitable, ImplicitlyCopyable):
    var f: Self.F

    def call[w: Int](self) -> SIMD[Self.dtype, w]:
        return self.f[w]()


def testParametricClosureField():
    def impl[w: Int]() {} -> SIMD[.int32, w]:
        return SIMD[.int32, w](7)

    var h = ParametricClosureField[.int32, type_of(impl)](impl)
    print(h.call[4]()[0])


# COM: Same capture, but the captured parameter (`dtype`) is infer-only
@fieldwise_init
struct InferredClosureField[
    dtype: DType,
    //,
    F: ImplicitlyCopyable & Deinitable & (def[w: Int]() -> SIMD[dtype, w]),
](Deinitable, ImplicitlyCopyable):
    var f: Self.F

    def call[w: Int](self) -> SIMD[Self.dtype, w]:
        return self.f[w]()


def testInferredClosureField():
    def impl[w: Int]() {} -> SIMD[.int32, w]:
        return SIMD[.int32, w](8)

    var h = InferredClosureField(impl)
    print(h.call[4]()[0])


# COM: A closure-typed struct parameter bound via a named parametric-alias
# COM: trait whose captured alias (`in_dtype`) differs from the enclosing struct
# COM: parameter it is bound to (`in_type`), with the captured auxiliary
# COM: parameter bound explicitly by keyword on `__call__`.
comptime _fn_trait[in_dtype: DType] = ImplicitlyCopyable & Deinitable & (
    def[w: Int]() -> SIMD[in_dtype, w]
)


@fieldwise_init
struct ExplicitAuxClosureField[
    in_type: DType,
    F: _fn_trait[in_type],
](Deinitable, ImplicitlyCopyable):
    var f: Self.F

    def call[w: Int](self) -> SIMD[Self.in_type, w]:
        return self.f.__call__[_in_type=Self.in_type, w=w]()


def testExplicitAuxClosureField():
    def impl[w: Int]() {} -> SIMD[.int32, w]:
        return SIMD[.int32, w](9)

    var h = ExplicitAuxClosureField[.int32, type_of(impl)](impl)
    print(h.call[4]()[0])


# COM: Same named parametric-alias trait as above, but with `in_type` infer-only
# COM: (the `//` marker).
@fieldwise_init
struct InferredAliasClosureField[
    in_type: DType,
    //,
    F: _fn_trait[in_type],
](Deinitable, ImplicitlyCopyable):
    var f: Self.F

    def call[w: Int](self) -> SIMD[Self.in_type, w]:
        return self.f.__call__[_in_type=Self.in_type, w=w]()


def testInferredAliasClosureField():
    def impl[w: Int]() {} -> SIMD[.int32, w]:
        return SIMD[.int32, w](10)

    var h = InferredAliasClosureField(impl)
    print(h.call[4]()[0])


# COM: A chain of parametric-alias traits.
comptime _chained_fn_trait0[
    FOO: DType, cw: Int
] = ImplicitlyCopyable & Deinitable & (def() -> SIMD[FOO, cw])

comptime _chained_fn_trait1[BAR: DType] = _chained_fn_trait0[BAR, 4]


@fieldwise_init
struct ChainedAliasClosureField[
    in_type: DType,
    //,
    F: _chained_fn_trait1[in_type],
](Deinitable, ImplicitlyCopyable):
    var f: Self.F

    def call(self) -> SIMD[Self.in_type, 4]:
        return self.f()


def testChainedAliasClosureField():
    def impl() -> SIMD[.int32, 4]:
        return SIMD[.int32, 4](11)

    var h = ChainedAliasClosureField(impl)
    print(h.call()[0])


# COM: A closure-typed parameter invoked from within a nested closure.
def nestedClosureParamCapture[
    dtype: DType,
    F: ImplicitlyCopyable
    & RegisterPassable
    & (def[w: Int]() -> SIMD[dtype, w]),
](f: F) -> Scalar[dtype]:
    @always_inline
    def body() {var f} -> Scalar[dtype]:
        @always_inline
        def inner[w: Int]() {var f} -> SIMD[dtype, w]:
            return f[w]()

        return inner[4]()[0]

    return body()


def testNestedClosureParamCapture():
    def impl[w: Int]() -> SIMD[.int32, w]:
        return SIMD[.int32, w](12)

    print(nestedClosureParamCapture[.int32, type_of(impl)](impl))


# COM: Nested `load_fn[simd_width, dtype]` shadows enclosing `dtype`, by-value-
# COM: captures a value whose type names that outer `dtype`, and is consumed via
# COM: `type_of` so the capture stays live. Storage publishes the capture under
# COM: the unmangled name; the trait thunk must uniquify its Pog-sourced decls.
@fieldwise_init
struct ShadowingLoadBuf[dtype: DType]:
    var data: SIMD[Self.dtype, 1]

    def raw_load[width: Int](self, i: Int) -> SIMD[Self.dtype, width]:
        return SIMD[Self.dtype, width](Int(self.data[0]) + i)


def takeShadowingLoader[
    dtype: DType,
    load_fn: def[simd_width: Int, dtype: DType](Int) -> SIMD[dtype, simd_width],
](loader: load_fn) -> SIMD[dtype, 1]:
    return loader[1, dtype](0)


def outerShadowingLoad[
    dtype: DType
](input: ShadowingLoadBuf[dtype],) -> SIMD[dtype, 1]:
    def load_fn[
        simd_width: Int, dtype: DType
    ](point: Int) {input,} -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            input.raw_load[width=simd_width](point)
        )

    return takeShadowingLoader[dtype, type_of(load_fn)](load_fn)


def testShadowingNestedClosureParam():
    var buf = ShadowingLoadBuf[.int32](Int32(13))
    print(outerShadowingLoad(buf)[0])


def main() raises:
    var y: Int = atol(argv()[1])
    var one = atol(argv()[2])
    var four = atol(argv()[3])

    def myclosure(x: Int) {var y} -> Int:
        return y + x

    var s = AStruct(myclosure)
    # CHECK: 6
    takeIt(s)
    # CHECK: 5
    var impl = DefinesClosureImpl(one)
    takeIt(impl, four)

    # COM: Ensure RegisterPassable struct can forward args through closure adaptor
    # CHECK: True
    testRegisterPassableUnifiedClosureAdaptor()

    # COM: Parametric closure field capturing a struct parameter.
    # CHECK: 7
    testParametricClosureField()

    # COM: Infer-only captured parameter bound from the closure argument.
    # CHECK: 8
    testInferredClosureField()

    # COM: Explicitly binding the captured auxiliary parameter by keyword.
    # CHECK: 9
    testExplicitAuxClosureField()

    # COM: Infer-only parameter bound through a named parametric-alias trait.
    # CHECK: 10
    testInferredAliasClosureField()

    # COM: Infer-only parameter bound through a chain of parametric-alias traits.
    # CHECK: 11
    testChainedAliasClosureField()

    # COM: Closure-typed function parameter invoked from within a nested closure,
    # COM: seeding a captured parameter from an ancestor function scope.
    # CHECK: 12
    testNestedClosureParamCapture()

    # COM: Nested closure param shadows enclosing capture param used in the
    # COM: captured value's type (pool.mojo load_fn motif).
    # CHECK: 13
    testShadowingNestedClosureParam()
