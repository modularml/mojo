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

# RUN: %mojo %s | FileCheck %s


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
struct Struct123(ImplicitlyCopyable, Trait3, Traits12):
    def f1(self):
        print("f1")

    def f2(self):
        print("f2")

    def f3(self):
        print("f3")


def use1[T: Trait1](x: T):
    x.f1()


def use2[T: Trait2](x: T):
    x.f2()


def use12[T: Traits12](x: T):
    use1(x)
    use2(x)


def use23[T: Trait2 & Trait3](x: T):
    x.f2()
    x.f3()


def use123[T: Traits123](x: T):
    x.f1()
    use23(x)


# conditional method
@fieldwise_init
struct Wrapper[T: AnyType](ImplicitlyCopyable):
    @__allow_legacy_custom_self_type
    def cond1[Trait: Trait1](self: Wrapper[Trait], other: Wrapper[Trait]):
        print("cond")


def useCond1[
    ElementType: Traits12
](p1: Wrapper[ElementType], p2: Wrapper[ElementType]):
    p1.cond1(p2)


# constructor overloading
trait IntConstructable:
    def __init__(out self, x: Int):
        ...


def useIntConstructable[T: Defaultable & IntConstructable]() -> T:
    return T(33)


struct MyStruct(Defaultable, IntConstructable, TrivialRegisterPassable):
    var x: Int

    def __init__(out self):
        self.x = 42

    def __init__(out self, x: Int):
        self.x = x


def main():
    var s123 = Struct123()

    # CHECK: f1
    use1(s123)

    # CHECK: f1
    # CHECK: f2
    use12(s123)

    # CHECK: f2
    # CHECK: f3
    use23(s123)

    # CHECK: f1
    # CHECK: f2
    # CHECK: f3
    use123(s123)

    # CHECK: cond
    useCond1(Wrapper[Struct123](), Wrapper[Struct123]())

    # CHECK: 33
    print(useIntConstructable[MyStruct]().x)
