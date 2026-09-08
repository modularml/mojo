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

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -split-input-file | FileCheck %s

# LIT dialect asm aliases for trait composition.
# CHECK-DAG: !AnyType_Trait1 = !lit.trait<@std::@builtin::@stubs::@AnyType, @trait_composition::@Trait1>
# CHECK-DAG: !AnyType_Trait2 = !lit.trait<@std::@builtin::@stubs::@AnyType, @trait_composition::@Trait2>
# CHECK-DAG: !AnyType_Trait3 = !lit.trait<@std::@builtin::@stubs::@AnyType, @trait_composition::@Trait3>
# CHECK-DAG: !AnyType_Trait1_Trait2 = !lit.trait<@std::@builtin::@stubs::@AnyType, @trait_composition::@Trait1, @trait_composition::@Trait2>
# CHECK-DAG: !AnyType_Trait1_Trait2_Trait3 = !lit.trait<@std::@builtin::@stubs::@AnyType, @trait_composition::@Trait1, @trait_composition::@Trait2, @trait_composition::@Trait3>


trait Trait1:
    def f1(self):
        ...


trait Trait2:
    def f2(self):
        ...


trait Trait3:
    def f3(self):
        ...


comptime Traits12 = Trait1 & Trait2
comptime Traits123 = Trait1 & Trait2 & Trait3


@fieldwise_init
struct Struct123(Trait1, Trait2, Trait3):
    def f1(self):
        pass

    def f2(self):
        pass

    def f3(self):
        pass


# Use direct trait union as parent.
@fieldwise_init
struct Struct12Direct(Trait1 & Trait2):
    def f1(self):
        pass

    def f2(self):
        pass


# Use trait union alias.
@fieldwise_init
struct Struct12Alias(Traits12):
    def f1(self):
        pass

    def f2(self):
        pass


def useAny[T: AnyType](x: T):
    pass


# CHECK: lit.fn @"use1
# CHECK-SAME: <T: !AnyType_Trait1>
# CHECK-SAME: (%x: !lit.ref<:!AnyType_Trait1 T,
def use1[T: Trait1](x: T):
    # CHECK: lit.call tail[{{.*}}"self": !lit.ref<:!AnyType_Trait1 T,{{.*}} #kgen.get_witness<:!AnyType_Trait1 T, @trait_composition::@Trait1, "f1{{.*}}">][{{.*}}](%x)
    x.f1()


def use2[T: Trait2](x: T):
    x.f2()


# Use aliased trait composition.
# CHECK-LABEL: lit.fn @"use12
# CHECK-SAME: <T: !AnyType_Trait1_Trait2>
# CHECK-SAME: (%x: !lit.ref<:!AnyType_Trait1_Trait2 T,
def use12[T: Trait1 & Trait2](x: T):
    # CHECK: lit.call tail @trait_composition::@"use1
    # CHECK-SAME: <:!AnyType_Trait1 upcast(:!AnyType_Trait1_Trait2 T)>
    use1[T](x)
    # CHECK: lit.call tail @trait_composition::@"use2
    # CHECK-SAME: <:!AnyType_Trait2 upcast(:!AnyType_Trait1_Trait2 T)>
    use2[T](x)


# Use direct trait composition.
# CHECK-LABEL: lit.fn @"use23
def use23[T: Trait2 & Trait3](x: T):
    # CHECK: lit.call tail[
    # CHECK-SAME: "self": !lit.ref<:!AnyType_Trait2_Trait3 T,
    # CHECK-SAME: #kgen.get_witness<:!AnyType_Trait2_Trait3 T{{.*}}, @trait_composition::@Trait3, "f3{{.*}}">
    x.f3()


# CHECK-LABEL: lit.fn @"use123
def use123[T: Trait1 & Trait2 & Trait3](x: T):
    # CHECK: lit.call {{.*}}@"use23
    # CHECK-SAME: "x": !lit.ref<:!AnyType_Trait1_Trait2_Trait3 T,
    use23(x)


# CHECK-LABEL: lit.fn @"main_use()"
def main_use():
    var s123 = Struct123()

    # CHECK: lit.call {{.*}}@"useAny
    # CHECK-SAME: <:!AnyType !Struct123>
    useAny(s123)
    # CHECK: lit.call {{.*}}@"use1
    # CHECK-SAME: <:!AnyType_Trait1 !Struct123>
    use1(s123)
    # CHECK: lit.call {{.*}}@"use1
    # CHECK-SAME: <:!AnyType_Trait1 !Struct123>
    use1[Struct123](s123)
    # CHECK: lit.call {{.*}}@"use12
    # CHECK-SAME: <:!AnyType_Trait1_Trait2 !Struct123>
    use12(s123)
    # CHECK: lit.call {{.*}}@"use23
    # CHECK-SAME: <:!AnyType_Trait2_Trait3 !Struct123>
    use23(s123)
    # CHECK: lit.call {{.*}}@"use123
    # CHECK-SAME: <:!AnyType_Trait1_Trait2_Trait3 !Struct123>
    use123(s123)

    var s12direct = Struct12Direct()
    # CHECK: lit.call {{.*}}@"use12
    # CHECK-SAME: <:!AnyType_Trait1_Trait2 !Struct12Direct>
    use12(s12direct)
    # CHECK: lit.call {{.*}}@"use12
    # CHECK-SAME: <:!AnyType_Trait1_Trait2 !Struct12Direct>
    use12[Struct12Direct](s12direct)

    var s12alias = Struct12Alias()
    # CHECK: lit.call {{.*}}@"use12
    # CHECK-SAME: <:!AnyType_Trait1_Trait2 !Struct12Alias>
    use12(s12alias)
    # CHECK: lit.call {{.*}}@"use12
    # CHECK-SAME: <:!AnyType_Trait1_Trait2 !Struct12Alias>
    use12[Struct12Alias](s12alias)


# // -----

# Test conditional method that refines the self type to a different trait.


trait Trait1:
    def f1(self):
        ...


trait Trait2:
    def f2(self):
        ...


comptime Traits12 = Trait1 & Trait2


@fieldwise_init
struct Struct12(Traits12):
    def f1(self):
        pass

    def f2(self):
        pass


# conditional method
@fieldwise_init
struct Wrapper[T: AnyType](Movable where False):
    @__allow_legacy_custom_self_type
    def cond1[Trait: Trait1](self: Wrapper[Trait], other: Wrapper[Trait]):
        pass


# CHECK: lit.fn @"useCond1
def useCond1[
    ElementType: Trait1 & Trait2
](p1: Wrapper[ElementType], p2: Wrapper[ElementType]):
    # CHECK: lit.call {{.*}}@Wrapper::@"cond1
    # CHECK-SAME: <:!AnyType upcast(:!AnyType_Trait1_Trait2 ElementType), :!AnyType_Trait1 upcast(:!AnyType_Trait1_Trait2 ElementType)>
    p1.cond1(p2)


# // -----

# Check that constructor calls work with trait compositions.


trait Defaultable:
    def __init__(out self):
        ...


trait IntConstructable:
    def __init__(out self, x: Int):
        ...


# CHECK-LABEL: lit.fn @"useIntConstructable
def useIntConstructable[T: Defaultable & IntConstructable]() -> T:
    # CHECK: %[[INT33:.*]] = {{.*}} !Int = <{:scalar<index> 33}>
    # CHECK: lit.call tail[
    # CHECK-SAME: #kgen.get_witness<:!AnyType_Defaultable_IntConstructable T{{.*}}, @trait_composition::@IntConstructable, "__init__{{.*}}">
    # CHECK-SAME: %[[INT33]]
    return T(33)


struct MyStruct(Defaultable, IntConstructable, TrivialRegisterPassable):
    var x: Int

    def __init__(out self):
        self.x = 42

    def __init__(out self, x: Int):
        self.x = x


# // -----

# Check that we can call parametric trait methods on types that were declared
# with trait composition.


trait Writer:
    def write(self):
        ...


trait Writable:
    def write_to[T: Writer](self, x: T):
        ...


trait Defaultable:
    def __init__(out self):
        ...


struct YourStruct(Movable where False):
    var x: Int

    def __init__(out self):
        self.x = 42

    def foo[W: Writable](self, x: W):
        pass

    def do_it[W: Writable & Defaultable](self, x: W):
        self.foo(x)  # make sure this doesn't crash


# // -----

# Check that composition works with trait parameters.


trait Trait1:
    def f1(self):
        ...


trait Trait2:
    def f2(self):
        ...


trait Trait1C(Trait1):
    def f1C(self):
        ...


@fieldwise_init
struct Struct1C(Trait1C, Trait2):
    def f1(self):
        pass

    def f2(self):
        pass

    def f1C(self):
        pass


def trait_param[A: type_of(Trait1), T: A & Trait2](x: T):
    x.f1()
    x.f2()


def use_trait_param():
    var s1c = Struct1C()
    trait_param[Trait1C, Struct1C](s1c)


# // -----

# Check that a trait member binding `Self` to a parametric associated type
# declared on another trait (`P.HandleFor[Self]`) gets its `Self` upcast to
# the declaring trait's type when the member is inherited into a sub-trait,
# rather than substituted bare with the sub-trait's `Self` (MOCO-4154).


trait Plugin:
    comptime HandleFor[From: Handle]: Handle


trait Handle:
    def transfer[P: Plugin](self) -> P.HandleFor[Self]:
        ...


trait SubHandle(Handle):
    pass


def sub_handle_transfer_type[S: SubHandle]():
    # CHECK-LABEL: lit.fn @"sub_handle_transfer_type
    # CHECK:      lit.alias.decl *"f`{{[0-9]*}}"
    # CHECK-SAME:   "self": !lit.ref<:!AnyType_Handle_SubHandle S
    # CHECK-SAME:   :!AnyType_Handle upcast(:!AnyType_Handle_SubHandle S)
    comptime f = S.transfer


def main():
    pass
