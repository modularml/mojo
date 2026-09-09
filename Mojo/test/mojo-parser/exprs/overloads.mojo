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


# CHECK: [[SCALARF64:.*]] = #kgen.type<!lit.struct<#MLIRType <:non_struct_type scalar<f64>>>, scalar<f64>> : !AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable


struct MyInt(TrivialRegisterPassable):
    var value: Int

    @always_inline("nodebug")
    @implicit
    def __init__(out self, v: Int):
        self.value = v


def overloaded_arg(a: Int, b: MyInt):
    pass


def overloaded_arg(a: Int, b: Int):
    pass


# CHECK-LABEL: lit.fn @"test_kw_args_overload{{.*}}"(%x: !Int, %y: !Int)
def test_kw_args_overload(x: Int, y: Int):
    # CHECK: call {{.*}}@"overloaded_arg{{.*}}"(%x, %y)
    overloaded_arg(b=y, a=x)

    # CHECK: [[Y:%.*]] = lit.call {{.*}}@MyInt::@"__init__{{.*}}(%y)
    # CHECK-NEXT: call {{.*}}@"overloaded_arg{{.*}}"(%x, [[Y]])
    overloaded_arg(b=MyInt(y), a=x)


# COM: test parametric overload in the presence of keyword operands.
def take_kw_param_infer[
    A: TrivialRegisterPassable, B: TrivialRegisterPassable
](a: A, b: B):
    pass


def take_kw_param_infer[B: TrivialRegisterPassable](a: MyInt, b: B):
    pass


# CHECK-LABEL: lit.fn @"test_kw_args_param_infer
def test_kw_args_param_infer(
    x: Int, f: __mlir_type.`!kgen.scalar<f64>`, s: MyInt
):
    # CHECK: call {{.*}}@"take_kw_param_infer[::AnyType & ::Copyable & ::Deinitable & ::ImplicitlyCopyable & ::Movable & ::RegisterPassable & ::TrivialRegisterPassable,::AnyType & ::Copyable & ::Deinitable & ::ImplicitlyCopyable & ::Movable & ::RegisterPassable & ::TrivialRegisterPassable]{{.*}}"<:{{.*}}!Int, {{.*}}[[SCALARF64]]>(%x, %f)
    take_kw_param_infer(x, b=f)

    # CHECK: call {{.*}}@"take_kw_param_infer[::AnyType & ::Copyable & ::Deinitable & ::ImplicitlyCopyable & ::Movable & ::RegisterPassable & ::TrivialRegisterPassable,::AnyType & ::Copyable & ::Deinitable & ::ImplicitlyCopyable & ::Movable & ::RegisterPassable & ::TrivialRegisterPassable]{{.*}}"<:{{.*}}!Int, {{.*}}[[SCALARF64]]>(%x, %f)
    take_kw_param_infer[Int](b=f, a=x)

    # CHECK: call {{.*}}@"take_kw_param_infer[::AnyType & ::Copyable & ::Deinitable & ::ImplicitlyCopyable & ::Movable & ::RegisterPassable & ::TrivialRegisterPassable,::AnyType & ::Copyable & ::Deinitable & ::ImplicitlyCopyable & ::Movable & ::RegisterPassable & ::TrivialRegisterPassable]{{.*}}"<:{{.*}}!Int, {{.*}}[[SCALARF64]]>(%x, %f)
    take_kw_param_infer[Int, __mlir_type.`!kgen.scalar<f64>`](b=f, a=x)

    # CHECK: call {{.*}}@"take_kw_param_infer[::AnyType & ::Copyable & ::Deinitable & ::ImplicitlyCopyable & ::Movable & ::RegisterPassable & ::TrivialRegisterPassable]{{.*}}<:{{.*}}!Int>(%s, %x)
    take_kw_param_infer(s, b=x)

    # CHECK: call {{.*}}@"take_kw_param_infer[::AnyType & ::Copyable & ::Deinitable & ::ImplicitlyCopyable & ::Movable & ::RegisterPassable & ::TrivialRegisterPassable]{{.*}}<:{{.*}}!Int>(%s, %x)
    take_kw_param_infer(b=x, a=s)


# COM: Test overloading precedence in the presence of static methods.
struct StaticOverloadStruct(Movable where False):
    def __init__(out self):
        pass

    def foo(mut self):
        pass

    @staticmethod
    def foo():
        pass


# CHECK-LABEL: lit.fn @"test_static_overload()"
def test_static_overload():
    var a = StaticOverloadStruct()
    # CHECK-NEXT: %a = lit.var.decl
    # CHECK-NEXT: lit.call{{.*}}__init__{{.*}}(%a)
    # CHECK-NEXT: lit.call {{.*}}foo{{.*}}(%a)
    a.foo()


# COM: Issue https://github.com/modular/mojo/issues/1408
# COM: Test that the number of implicit conversions is more important than
# COM: convention mismatches.
struct MyElement(TrivialRegisterPassable):
    pass


struct ConvertibleFromInt(Movable where False):
    @implicit
    def __init__(out self, a: Int):
        pass


struct MyContainer[T: ImplicitlyCopyable & Deinitable](Movable where False):
    var v: Self.T

    def foo(self, limits: ConvertibleFromInt):
        pass

    def foo(self, index: Int) -> Self.T:
        return self.v


# CHECK-LABEL: lit.fn @"test_impl
def test_impl(a: MyContainer[MyElement], b: Int):
    # CHECK: lit.call {{.*}}@MyContainer::@"foo{{.*}}, "index": !Int
    _ = a.foo(b)


# Regression test for MOCO-3665: overload resolution must prefer `ref self`
# over `var self` for TrivialRegisterPassable types when called from a
# borrowed self context.
#
# In MLIR mangling, the self-argument convention is encoded as a suffix inside
# the quoted function name: '%' = ref/borrowed, '$' = owned/var.
@fieldwise_init
struct TrivSelfOverload(ImplicitlyCopyable, TrivialRegisterPassable):
    var x: Int

    # CHECK-LABEL: lit.fn @"which{{.*}}TrivSelfOverload%)"
    def which(ref self) -> Int: return 1

    # CHECK-LABEL: lit.fn @"which{{.*}}TrivSelfOverload$)"
    def which(var self) -> Int: return 2

    # CHECK-LABEL: lit.fn @"call_which{{.*}}TrivSelfOverload)"
    def call_which(self) -> Int:
        # Borrowed self must dispatch to the ref overload (name ends with '%)')
        # not the var overload ('$)'). Note: the ref overload's name also
        # contains '$' in its origin parameters (e.g. '[$0]'), so we match
        # 'TrivSelfOverload%)' specifically rather than checking for absence
        # of '$'.
        # CHECK: lit.call {{.*}}TrivSelfOverload%)
        return self.which()
