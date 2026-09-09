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

# RUN: mkdir -p %t.packaged-trait
# RUN: mojo precompile %S/inputs/test_package -o %t.packaged-trait/test_package_trait.mojoc
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.packaged-trait %s --kgen-print-inline-type-values | FileCheck %s

from test_package_trait.module import (
    PackageTrait,
    UseTrait,
    UseTraitReg,
    trait_method,
    contains_thunk_ref,
)


# CHECK: lit.struct.decl @MyType({{.*}}PackageTrait
struct MyType(PackageTrait):
    def method(self):
        pass

    # CHECK: kgen.conformance {{.*}}::@PackageTrait
    # CHECK: kgen.witness "method{{.*}}" {{.*}} = {{.*}}::@MyType::@"method


# CHECK: lit.struct.decl @MyRegType({{.*}}PackageTrait
struct MyRegType(PackageTrait, RegisterPassable):
    def method(self):
        pass

    # CHECK: kgen.conformance {{.*}}::@PackageTrait
    # CHECK: kgen.witness "method{{.*}}" : !lit.generator<[1]("self": !lit.ref<!MyRegType, imm *[0,0]> imm_mem) -> !kgen.none> = {{.*}}::@MyRegType::@"method


def bind_trait[T: PackageTrait]():
    pass


# CHECK-LABEL: lit.fn @"test
def test():
    # CHECK-NEXT: <:!AnyType_PackageTrait !MyType>
    bind_trait[MyType]()
    # CHECK-NEXT: <:!AnyType_PackageTrait !MyRegType>
    bind_trait[MyRegType]()
    # CHECK-NEXT: <:!AnyType_UsedInPackageTrait !UseTrait>
    trait_method[UseTrait]()
    # CHECK-NEXT: <:!AnyType_UsedInPackageTrait !UseTraitReg>
    trait_method[UseTraitReg]()

    # COM: Anchor this decl reference to materialize it.
    contains_thunk_ref()


def use_trait[T: PackageTrait](x: UseTrait, y: T):
    y.method()


# This function calls a closure whose signature matches a closure uses by a
# function in the test_package_trait package. The signatures of these closures
# must match so that the closure trait is first loaded from the package.
def my_vectorize[
    func: def[x: Int, y: Int, z: Int](idx: Int) -> None
](closure: func):
    closure[0, 1, 1](0)


# CHECK-LABEL: lit.fn @"my_test
def my_test():
    def foo[width: Int, x: Int, y: Int](idx: Int) {var}:
        print(width + x + y)

    my_vectorize(foo)


# CHECK-LABEL: lit.package @test_package_trait

# CHECK: lit.trait.decl @PackageTrait
# CHECK: lit.trait.decl @UsedInPackageTrait

# CHECK-LABEL: lit.struct.decl @UseTrait
# CHECK: kgen.conformance {{.*}}::@UsedInPackageTrait
# CHECK: kgen.witness "method{{.*}}" {{.*}} = {{.*}}::@UseTrait::@"method

# CHECK-LABEL: lit.struct.decl @UseTraitReg
# CHECK: kgen.conformance {{.*}}::@UsedInPackageTrait
# CHECK: kgen.witness "method{{.*}}" : !lit.generator<[1]("self": !lit.ref<!UseTraitReg, imm *[0,0]> imm_mem) -> !kgen.none> = {{.*}}::@UseTraitReg::@"method
