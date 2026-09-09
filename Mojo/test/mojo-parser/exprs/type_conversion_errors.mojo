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

# RUN: %parse-mojo-isolated %s -verify-diagnostics


struct RP_NotTrivial(RegisterPassable):
    pass


struct Foo(Movable where False):
    def __init__(out self):
        pass


# expected-note @+1 {{function declared here}}
def take_instance_param[a: Foo]():
    pass


# expected-note @+1 {{function declared here}}
def takes_instance_arg(a: Foo):
    pass


# COM: Issue #27654: Parser crash: Assertion failed: Types should match
# COM: https://github.com/modular/mojo/issues/1607 Improved error message for this common error
def test_type_instead_of_instance() -> Foo:
    # expected-error @+1 {{'take_instance_param' parameter 'a' has 'Foo' type, but value has type 'AnyStruct[Foo]'}}
    take_instance_param[Foo]
    # expected-error @+1 {{invalid call to 'takes_instance_arg': value passed to 'a' cannot be converted from type value 'Foo' to an instance of 'Foo'; did you mean to instantiate 'Foo'?}}
    takes_instance_arg(Foo)
    # expected-error @+1 {{cannot implicitly convert 'Foo' type as a value to an instance of 'Foo'; did you mean to instantiate 'Foo'?}}
    return Foo


# COM: https://github.com/modularml/modular/issues/29438
# COM: ensure we do not crash in the example below, but emit an error.
struct MadeFromPack[*Ts: AnyType](Movable where False):
    @implicit
    def __init__(out self, *args: * Self.Ts):
        pass


struct WrapsMadeFromPack[*Ts: AnyType](Movable where False):
    var data: MadeFromPack[*Self.Ts]

    @implicit
    def __init__(out self, *args: * Self.Ts):
        # expected-error @+1 {{cannot implicitly convert 'VariadicPack[False, Ts]' value to 'MadeFromPack[Ts]'}}
        self.data = args


struct Constructible(Movable where False):
    @implicit
    def __init__(out self, arg: Int):
        pass


def init_self_conversion():
    # expected-error @below {{cannot implicitly convert 'def __init__(arg: Int) thin -> Constructible' value to 'def() thin -> None'}}
    comptime f: def() thin -> None = Constructible.__init__


struct ConvertibleFromInt(ImplicitlyCopyable):
    @implicit
    def __init__(out self, arg: Int):
        pass


@fieldwise_init
# expected-note @below {{candidate declared here}}
# expected-note @below {{def __init__(out self, var a: ConvertibleFromInt, b: Int)    # note - generated function}}
struct AmbiguousCtor(Movable where False):
    var a: ConvertibleFromInt
    var b: Int

    # expected-note @below {{candidate declared here}}
    def __init__(out self, b: Int, a: ConvertibleFromInt):
        pass


struct AlsoConvertibleFromInt(Movable where False):
    @implicit
    def __init__(out self, arg: Int):
        pass


struct AmbiguousConversion(Movable where False):
    @implicit
    # expected-note @below {{candidate declared here}}
    def __init__(out self, x: ConvertibleFromInt):
        pass

    @implicit
    # expected-note @below {{candidate declared here}}
    def __init__(out self, x: AlsoConvertibleFromInt):
        pass


def ambiguous_ctor_call(x: Int):
    # expected-error @below {{ambiguous call}}
    AmbiguousCtor(x, x)

    # expected-error @below {{ambiguous call to '__init__', each candidate requires 1 implicit conversion}}
    AmbiguousConversion(x)


# MOCO-990: Conditional conformance trick fails on SIMD constructor from Bool
struct MySIMD[value: Int](Movable where False):
    def __init__(out self: MySIMD[0], val: MyBool):
        pass


struct MyBool(Movable where False):
    @implicit
    def __init__(out self, value: MySIMD[0]):
        pass


def test_bad_conversion(a: MySIMD[0]):
    # expected-error @+1 {{cannot implicitly convert 'MySIMD[Int(0)]' value to 'MySIMD[Int(1)]'}}
    var b: MySIMD[1] = a


# MOCO-1090: bad parameter inference error message.
def test_rp_trivial_inference(a: RP_NotTrivial, b: Foo):
    # expected-error @below {{invalid call to 'infer_rp_trivial': value passed to 'val' cannot be converted from 'RP_NotTrivial' to 'T', argument type 'RP_NotTrivial' does not conform to trait 'TrivialRegisterPassable'}}
    _ = infer_rp_trivial(a)

    # expected-error @below {{invalid call to 'infer_rp_trivial': value passed to 'val' cannot be converted from 'Foo' to 'T', argument type 'Foo' does not conform to trait 'TrivialRegisterPassable'}}
    _ = infer_rp_trivial(b)


# expected-note @below {{function declared here}}
def infer_rp_trivial[T: TrivialRegisterPassable](val: T):
    pass


struct ImplicitFromInt:
    @implicit
    def __init__(out self, value: Int):
        pass


def stripping_raises():
    def fn_raises() raises:
        pass

    def fn_raises_bool(arg: Int) raises Bool:
        pass

    def fn_raises_int(arg: Int) raises Int:
        pass

    # expected-error @+1 {{cannot implicitly convert 'def fn_raises() raises thin -> None' value to 'def() thin -> None'}}
    var fp: def() thin = fn_raises

    # expected-error @+2 {{cannot implicitly convert 'def fn_raises() raises thin -> None' value to 'def() raises Int thin -> None'}}
    # expected-note @+1 {{error type of the first type is 'Error' but the second type is 'Int'}}
    var fp2: def() thin raises Int = fn_raises

    # expected-error @+3 {{cannot implicitly convert 'def fn_raises_bool(arg: Int) raises Bool thin -> None' value to 'def(*args: *TypeList[Int]()) raises Int thin -> None' in 'var' initializer}}
    var fp3: def(
        *args: *TypeList.splat[1, Int]()
    ) raises Int thin = fn_raises_bool

    # Just because the RHS `raises` type is implicitly convertible to the LHS
    # `raises`, that does not imply that the function types should be implicitly
    # convertible. Maybe someday.
    # expected-error @+3 {{cannot implicitly convert 'def fn_raises_int(arg: Int) raises Int thin -> None' value to 'def(*args: *TypeList[Int]()) raises ImplicitFromInt thin -> None' in 'var' initializer}}
    var fp4: def(
        *args: *TypeList.splat[1, Int]()
    ) raises ImplicitFromInt thin = fn_raises_int

struct SimplePair(Copyable):
    var left: Int
    var right: Int

def test_field_erase(ref p: SimplePair, cond: Bool) -> ref[origin_of(p).subtree] Int:
    # This is ok. The origin union for left/right type erase.
    return p.left if cond else p.right

    var x: Int
    # expected-error @+1 {{cannot return reference with incompatible origin: 'origin_of(x)' vs 'origin_of(origin_of(p).subtree)'}}
    return x

def test_field_erase_2(ref p: List[SimplePair]) -> ref[origin_of(p).subtree] Int:
  return p[0].left
