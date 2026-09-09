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


trait SubTraitT(TrivialRegisterPassable):
    def subget(self) -> Int:
        ...


trait SubTraitT2(TrivialRegisterPassable):
    def subget2(self) -> Int:
        ...


trait MainTraitT(TrivialRegisterPassable):
    comptime ret_type: SubTraitT
    comptime anything: AnyType

    def get(self) -> Self.ret_type:
        ...


trait MainTraitT2(TrivialRegisterPassable):
    comptime ret_type: SubTraitT2

    def get2(self) -> Self.ret_type:
        ...


@fieldwise_init
struct ImplT(SubTraitT, SubTraitT2, TrivialRegisterPassable):
    def subget(self) -> Int:
        return 0

    def subget2(self) -> Int:
        return 0

    def BAR(self) -> Int:
        return 1


@fieldwise_init
struct MainImplT(MainTraitT, MainTraitT2, TrivialRegisterPassable):
    # CHECK: lit.alias.decl *"ret_type{{.*}}": !mt_ImplT = <!ImplT>
    comptime ret_type = ImplT
    # CHECK: lit.alias.decl *"anything{{.*}}": meta<!Int> = <#alias_Int>
    comptime anything = Int

    def get(self) -> Self.ret_type:
        return ImplT()

    def get2(self) -> Self.ret_type:
        return ImplT()

    def doSomethingNonTraity(self) -> Int:
        # Verify the ImplT type is returned, not a type value of trait metatype.
        # CHECK: lit.call tail @{{.*}}::@MainImplT::@"get{{.*}}"(%self) : !lit.generator<("self": !MainImplT)
        # CHECK-SAME:  -> !kgen.param<:!mt_ImplT sugar_member_alias(!MainImplT, "ret_type", !ImplT)>>
        var impl = self.get()
        var a = impl.BAR()
        return a


def repro_issue[
    main_t: MainTraitT, main_t2: MainTraitT2
](t: main_t, t2: main_t2) -> Int:
    var a = t.get().subget()
    var b = t2.get2().subget2()
    var c = __mlir_op.`index.add`(a.__mlir_index__(), b.__mlir_index__())
    return Int(mlir_value=c)


@export
def callIt() abi("Mojo") -> Int:
    var t = MainImplT()
    var a = repro_issue(t, t)
    return a


# ===----------------------------------------------------------------------=== #
# Upcast tests
# ===----------------------------------------------------------------------=== #


# Just make sure this parses.
def declval[T: AnyType]() -> T:
    pass


trait MyThingTrait:
    def thing(self) -> __mlir_type.i1:
        ...


def propagate_type[T: MyThingTrait](range: T) -> type_of(declval[T]().thing()):
    pass
