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

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct Thing(TrivialRegisterPassable):
    @implicit
    def __init__(out self, x: NMType):
        return

    def foo(self):
        pass

    @staticmethod
    def bar():
        pass


# COM: Test various things related to interfacing with generic types.
def anytype[T: TrivialRegisterPassable]():
    pass


def anytype_arg[T: TrivialRegisterPassable](x: T):
    pass


def anytype_result[T: TrivialRegisterPassable]() -> T:
    pass


@fieldwise_init
@__nonmaterializable(Thing)
struct NMType(ImplicitlyCopyable, RegisterPassable):
    pass


# CHECK-LABEL: lit.fn @"metatypes()"
def metatypes():
    # COM: Test that a local alias can retain type properties.
    # CHECK: lit.alias.decl [[T:\*"T.*]]: !mt_Thing = <!Thing>
    comptime T = Thing
    # CHECK-NEXT: [[TVAL:%.*]] = lit.call {{.*}}Thing::@"__init__(){{.*}}()
    # CHECK: call {{.*}}@Thing::@"foo({{.*}})"([[TVAL]])
    T().foo()
    # CHECK: call {{.*}}@Thing::@"bar()"
    T.bar()

    # COM: Test that binding to a generic type works.
    # CHECK: bound{{.*}}: !lit.generator<<>!kgen.func.literal<{{.*}}@"anytype[::AnyType & ::Copyable & ::Deinitable & ::ImplicitlyCopyable & ::Movable & ::RegisterPassable & ::TrivialRegisterPassable]()"<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable !Thing>>
    comptime bound = anytype[Thing]

    # COM: Test that result types are bound correctly.
    # CHECK: call {{.*}}@"anytype_result[::AnyType & ::Copyable & ::Deinitable & ::ImplicitlyCopyable & ::Movable & ::RegisterPassable & ::TrivialRegisterPassable]()"<:{{.*}} !Thing>
    var v: Thing = anytype_result[Thing]()

    # COM: Test that argument type inference works correctly.
    # CHECK: call {{.*}}@"anytype_arg[::AnyType & ::Copyable & ::Deinitable & ::ImplicitlyCopyable & ::Movable & ::RegisterPassable & ::TrivialRegisterPassable]($0)"<:{{.*}} !Thing>
    anytype_arg(v)

    # COM: Test inferring from a nonmaterializable type.
    comptime nm_alias = NMType()
    # CHECK: [[DNMVAL:%.*]] = lit.var.decl "anonymous*"
    # CHECK-NEXT: [[NMVAL:%.*]] = kgen.param.materialize: !NMType = <#alias_nm_alias>
    # CHECK-NEXT: lit.ref.store [[NMVAL]], [[DNMVAL]]
    # CHECK-NEXT: [[IMMUT:%.*]] = lit.ref.immut [[DNMVAL]]
    # CHECK-NEXT: [[MVAL:%.*]] = lit.call {{.*}}@Thing::@"__init__{{.*}}([[IMMUT]])
    # CHECK-NEXT: call {{.*}}@"anytype_arg[::AnyType & ::Copyable & ::Deinitable & ::ImplicitlyCopyable & ::Movable & ::RegisterPassable & ::TrivialRegisterPassable]($0)"<:{{.*}} !Thing>([[MVAL]])
    anytype_arg(nm_alias)


# Stef's crazy metatype stress test.
@fieldwise_init
struct StefStressTest[x: Int](Movable where False):
    @staticmethod
    def increment() -> type_of(StefStressTest[Self.x + 1]):
        while True:
            pass
        # return StefStressTest[x+1]  # Doesn't work yet.


def use_int(a: Int):
    pass


# CHECK-LABEL: lit.fn @"access_param_from_metatype()"
def access_param_from_metatype():
    # CHECK-NEXT: lit.alias.decl *"f1`": meta<!lit.struct<#StefStressTest <:!Int {:scalar<index> 1}>>>
    comptime f1 = StefStressTest[0].increment()
    # CHECK: [[TMP:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK: lit.call {{.*}}@"use_int{{.*}}([[TMP]])
    use_int(f1.x)

    # CHECK: lit.call {{.*}}@"increment()"{{.*}}<:!Int {:scalar<index> 0}>
    # CHECK: lit.call {{.*}}@"increment()"{{.*}}<:!Int {:scalar<index> 1}>
    # CHECK: [[TMP1:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 2}>
    # CHECK-NEXT: lit.call tail {{.*}}@"use_int(::SIMD[DType.int, 1])"([[TMP1]])
    use_int((StefStressTest[0].increment().increment()).x)


# COM: we should handle mt_Int : mt_mt_Int -> copyable : any_trait<Copyable>
# correctly, since Int is a Copyable
def meta_type_to_trait[T: type_of(Copyable), //, W: T](t: W):
    pass


def meta_type_to_trait_driver():
    # CHECK: lit.call @metatypes::@"meta_type_to_trait[{{.*}}]<:meta<!AnyType_Copyable_Movable> meta<!Int>, :meta<!Int> #alias_Int>(%1)
    meta_type_to_trait[Int](1)
