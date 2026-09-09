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


trait Base:
    pass


trait Extra:
    pass


trait Marker:
    pass


# CHECK-LABEL: lit.struct.decl @SingleLineTrailingWhere
# CHECK-SAME: <T: !AnyType_Base, {{{.*}}conforms_to(:!AnyType_Base T, :meta<!{{.*}}Extra> !{{.*}}Extra))
@fieldwise_init
struct SingleLineTrailingWhere[T: Base] (Movable where False) where conforms_to(T, Extra):
    pass


# CHECK-LABEL: lit.struct.decl @TrailingWhereWithParent
# CHECK-SAME: <T: !AnyType_Base, {{{.*}}conforms_to(:!AnyType_Base T, :meta<!{{.*}}Extra> !{{.*}}Extra))
# The `Movable where False` opt-out is erased from the canonical trait, leaving
# the injected `AnyType`/`Deinitable` plus the unconditional `Marker`.
# CHECK-SAME: (!AnyType_Deinitable_Marker)
@fieldwise_init
struct TrailingWhereWithParent[T: Base](Marker, Movable where False) where conforms_to(T, Extra):
    pass


# CHECK-LABEL: lit.struct.decl @MultilineParentTrailingWhere
# CHECK-SAME: <T: !AnyType_Base, {{{.*}}conforms_to(:!AnyType_Base T, :meta<!{{.*}}Extra> !{{.*}}Extra))
# Same trait set as @TrailingWhereWithParent above, so it shares that alias.
# CHECK-SAME: (!AnyType_Deinitable_Marker)
@fieldwise_init
struct MultilineParentTrailingWhere[T: Base](Marker, Movable where False) where conforms_to(
    T, Extra
):
    pass


# CHECK-LABEL: lit.struct.decl @MultipleTrailingWhereClauses
# CHECK-SAME: <T: !AnyType_Base, {{{.*}}conforms_to(:!AnyType_Base T, :meta<!{{.*}}Extra> !{{.*}}Extra))
# The second clause is provable from `T`'s declared bound, so it folds to `true`
# while still occupying its own constraint slot.
# CHECK-SAME: <true,
@fieldwise_init
struct MultipleTrailingWhereClauses[T: Base] (Movable where False) where conforms_to(
    T, Extra
) where conforms_to(T, Base):
    pass


struct ConditionallyDeletableWrapper[T: AnyType](
    Deinitable where conforms_to(T, Deinitable), Movable where False,
):
    var value: Self.T


# CHECK-LABEL: lit.struct.decl @TrailingWhereDeletableField
# CHECK-SAME: <T: !AnyType,
# CHECK-SAME: conforms_to(:!AnyType T,
# CHECK: lit.struct.field value : !lit.struct<#ConditionallyDeletableWrapper
# CHECK-SAME: <:!AnyType T>>
struct TrailingWhereDeletableField[T: AnyType](Movable where False)
    where conforms_to(T, Deinitable):
    var value: ConditionallyDeletableWrapper[Self.T]


# The auto-synthesized fieldwise `__init__` for a struct with a trailing
# `where` clause must be able to use that clause's assumption to prove a
# field's conditional conformance -- mirroring the dtor case above, but
# exercising StructEmitter::synthesizeFieldwiseInit's field-movability check
# instead of synthesizeEmptyDtor's.
struct ConditionallyMovableWrapper[T: Deinitable](
    Movable where conforms_to(T, Movable),
):
    var value: Self.T


# CHECK-LABEL: lit.struct.decl @TrailingWhereMovableField
# CHECK-SAME: <T: !AnyType_Deinitable
# CHECK: lit.struct.field value : !lit.struct<#ConditionallyMovableWrapper
# CHECK-SAME: <:!AnyType_Deinitable T>>
struct TrailingWhereMovableField[T: Deinitable](
    Movable where conforms_to(T, Movable)
):
    var value: ConditionallyMovableWrapper[Self.T]
