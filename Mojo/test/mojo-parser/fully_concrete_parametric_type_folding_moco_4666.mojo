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

# Regression test for MOCO-4666: `TypeList.contains` over a pack of fully
# concrete parametric types must fold for every element, not just the first.
# `Parametric[0]` and `Parametric[1]` are distinct but fully bound type values;
# both `isa` calls have to discharge `Self.Ts.contains[T]()`.


struct MyVariant[*Ts: AnyType](ImplicitlyCopyable):
    def isa[T: AnyType](self) where Self.Ts.contains[T]():
        pass


struct Parametric[x: __mlir_type.index]:
    pass


comptime A = Parametric[__mlir_attr.`0 : index`]
comptime B = Parametric[__mlir_attr.`1 : index`]

comptime GenericSigAB = MyVariant[A, B]
comptime GenericSigBA = MyVariant[B, A]


# CHECK-LABEL: lit.fn @"check_ab
def check_ab[func: GenericSigAB]():
    # CHECK: lit.call {{.*}}isa
    func.isa[A]()
    # CHECK: lit.call {{.*}}isa
    func.isa[B]()


# CHECK-LABEL: lit.fn @"check_ba
def check_ba[func: GenericSigBA]():
    # CHECK: lit.call {{.*}}isa
    func.isa[B]()
    # CHECK: lit.call {{.*}}isa
    func.isa[A]()
