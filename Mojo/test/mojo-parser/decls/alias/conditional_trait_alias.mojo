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


trait RequiresIntAlias:
    comptime Value: Int


struct ConditionallyConforms[value: Int](RequiresIntAlias, Movable where False) where value > 0:
    # Ensure the generator constraint has been discharged in the witness.
    comptime Value: Int where Self.value > 0 = Self.value
    # CHECK: kgen.conformance {{.*}}@RequiresIntAlias {
    # CHECK-NEXT: kgen.witness "Value" : !alias_Int1 = rebind(:!Int value)


struct ConditionallyConformsInferred[value: Int](RequiresIntAlias, Movable where False) where (
    value > 0
):
    # A 'where' clause must also work when the alias type is inferred from the
    # initializer (no explicit ': Int' type annotation).
    comptime Value where Self.value > 0 = Self.value
    # CHECK: kgen.conformance {{.*}}@RequiresIntAlias {
    # CHECK-NEXT: kgen.witness "Value" : !alias_Int1 = rebind(:!Int value)


trait RequiresParametricIntAlias:
    comptime Value[offset: Int]: Int


struct ConditionallyConformsParametric[value: Int](
    RequiresParametricIntAlias, Movable where False
) where (value > 0):
    # Ensure the generator constraint has been discharged in the witness.
    comptime Value[offset: Int]: Int where Self.value > 0 = (
        Self.value + offset
    )
    # CHECK: kgen.conformance {{.*}}@RequiresParametricIntAlias {
    # CHECK-NEXT: kgen.witness "Value" : !lit.generator<<"offset": !Int>!alias_Int1> = #kgen.gen<
    # CHECK-SAME: @"__add__(::SIMD[$0, $1],::SIMD[$0, $1])"{{.*}}, value, *(0,0)
    # CHECK-SAME: add(#lit.struct.extract<:!Int value, "_mlir_value">, #lit.struct.extract<:!Int *(0,0), "_mlir_value">)


# A `where` clause attached to the conformance list itself (conditional
# conformance), rather than a trailing struct-level `where`, must also discharge
# a matching `where` clause on the comptime member. Here `Foo` conforms to
# `RequiresIntAlias` only when `value > 0`, which implies the member's own
# `where Self.value > 0`, so the witness reduces to the unconstrained `!Int`.


struct ConfListConditional[value: Int = -1](RequiresIntAlias where value > 0, Movable where False):
    comptime Value: Int where Self.value > 0 = Self.value
    # CHECK: kgen.conformance {{.*}}@RequiresIntAlias {
    # CHECK-NEXT: kgen.witness "Value" : !alias_Int1 = rebind(:!Int value)


# Same, but parametric: the body constraint is discharged while the `offset`
# input parameter is retained on the witnessed generator.


struct ConfListConditionalParametric[value: Int = -1](
    RequiresParametricIntAlias where value > 0, Movable where False
):
    comptime Value[offset: Int]: Int where Self.value > 0 = Self.value + offset
    # CHECK: kgen.conformance {{.*}}@RequiresParametricIntAlias {
    # CHECK-NEXT: kgen.witness "Value" : !lit.generator<<"offset": !Int>!alias_Int1> = #kgen.gen<
    # CHECK-SAME: @"__add__(::SIMD[$0, $1],::SIMD[$0, $1])"{{.*}}, value, *(0,0)


# The conformance constraint may be strictly stronger than the member's `where`
# clause: conjunction elimination proves `(value > 0 and other > 0)` implies
# `value > 0`, so the member constraint is still discharged.


struct ConfListConditionalConjunction[value: Int = -1, other: Int = -1](
    RequiresIntAlias where value > 0 and other > 0, Movable where False
):
    comptime Value: Int where Self.value > 0 = Self.value
    # CHECK: kgen.conformance {{.*}}@RequiresIntAlias {
    # CHECK-NEXT: kgen.witness "Value" : !alias_Int1 = rebind(:!Int value)
