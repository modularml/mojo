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

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s


trait R1:
    comptime N: Int

    def f1(self, x: Bool):
        ...

    def f1(self, x: Int):
        ...


trait R2:
    comptime T: AnyType

    def f2(self, x: Self.T):
        ...


trait R1Child(R1):
    pass


# CHECK-LABEL: lit.struct.decl @S1<X: !Int>
struct S1[X: Int](R1, R1Child, R2, TrivialRegisterPassable):
    comptime N: Int = Self.X
    comptime T: AnyType = Int

    # CHECK: lit.fn @"f1[[F1_BOOL_NAME:.+]]"({{.*}}, %x: !Bool)
    def f1(self, x: Bool):
        pass

    # CHECK: lit.fn @"f1[[F1_INT_NAME:.+]]"({{.*}}, %x: !Int)
    def f1(self, x: Int):
        pass

    # CHECK: lit.fn @"f2[[F2_NAME:.+]]"({{.*}}, %x: !Int)
    def f2(self, x: Int):
        pass

    # CHECK: lit.fn @"__deinit__[[DEL_NAME:.+]]"[
    # Synthesized function:
    # CHECK: lit.fn @"__init__[[MOVEINIT_NAME:.+]]"[

    # CHECK: kgen.conformance @{{.*}}AnyType
    # CHECK-NEXT: }

    # CHECK: kgen.conformance @{{.*}}Deinitable
    # CHECK-NEXT: kgen.witness "__deinit__{{.*}}" : {{.*}} = {{.*}}@S1::@"__deinit__{{.*}}"<:!Int X>
    # CHECK-NEXT: kgen.witness "__del__is_trivial" : !Bool = {:scalar<bool> true}

    # CHECK: kgen.conformance @{{.*}}Movable
    # CHECK-NEXT: kgen.witness "__init__{{.*}}(*, "move":{{.*}} = {{.*}}@S1::@"__init__{{.*}}"{{.*}}<:!Int X>
    # CHECK-NEXT: kgen.witness "__move_ctor_is_trivial" : !Bool = {:scalar<bool> true}

    # CHECK: kgen.conformance [[R1_REF:@[^ ]*R1]] {
    # CHECK-NEXT: kgen.witness "N" : !alias_Int1 = rebind(:!Int X)
    # CHECK-NEXT: kgen.witness "f1{{.*}}" : {{.*}} = {{.*}}@S1::@"f1{{.*}}"<:!Int X>
    # CHECK-NEXT: kgen.witness "f1{{.*}}" : {{.*}} = {{.*}}@S1::@"f1{{.*}}"<:!Int X>
    # CHECK-NEXT: immediateParents = #kgen<trait_symbols[]>

    # CHECK: kgen.conformance @{{.*}}R1Child
    # CHECK-NEXT: immediateParents = #kgen<trait_symbols[<[[R1_REF]]>]>

    # CHECK: kgen.conformance @{{.*}}R2
    # CHECK-NEXT: kgen.witness "T" : !AnyType = !Int
    # CHECK-NEXT: kgen.witness "f2{{.*}}" : {{.*}} = {{.*}}@S1::@"f2{{.*}}"<:!Int X>
