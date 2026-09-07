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

# RUN: %parse-mojo-isolated --verify-diagnostics %s | FileCheck %s


struct DT[a: Int](Movable where False):
    pass


# CHECK-LABEL: lit.trait.decl @B
trait B:
    comptime a: Int

    # This is a default value
    # CHECK:      lit.alias.decl *"c`2": !alias_Int1 = <sugar_member_alias(!kgen.param<:!B_AnyType *"_Self`">, "a", #kgen.get_witness<:!B_AnyType *"_Self`", @{{.*}}::@B, "a">)>
    # CHECK-SAME:   {defaultedAssociatedAlias}
    comptime c = Self.a

    # This is a dependent default type alias
    # CHECK:      lit.alias.decl *"T`3": meta<!lit.struct<#DT <:!Int #kgen.get_witness<:!B_AnyType *"_Self`", @{{.*}}::@B, "a">>>> = <@{{.*}}::@DT<:!Int #kgen.get_witness<:!B_AnyType *"_Self`", @{{.*}}::@B, "a">>>
    # CHECK-SAME:   {defaultedAssociatedAlias}
    comptime T = DT[Self.a]


# CHECK-LABEL: lit.struct.decl @Foo
struct Foo(B, Movable where False):
    comptime a: Int = 1

    # CHECK: lit.alias.decl *"c`2": !alias_Int1 = <sugar_member_alias(!Foo, "a", rebind(:!Int {:scalar<index> 1}))>
    #
    # Make sure that trait._Self is replaced properly.
    # CHECK: lit.alias.decl *"T`3": meta<!lit.struct<#DT <:!Int {:scalar<index> 1}>>> = <@default_associated_alias::@DT<:!Int {:scalar<index> 1}>>


# COM: A defaulted alias provided by a refining trait must be honored by a struct
# COM: that conforms only to the refinement.

# CHECK-LABEL: lit.trait.decl @C
trait C:
    # CHECK:      lit.alias.decl *"T`{{[0-9]+}}": !AnyType
    # CHECK-NOT:    {defaultedAssociatedAlias}
    comptime T: AnyType

# CHECK-LABEL: lit.trait.decl @D
trait D(C):
    # CHECK:      lit.alias.decl *"T`{{[0-9]+}}": !AnyType = <!Int>
    # CHECK-SAME:   {defaultedAssociatedAlias}
    comptime T: AnyType = Int

# CHECK-LABEL: lit.struct.decl @Bar
struct Bar(D, Movable where False):
    # COM: The defaulted `T = Int` from D must materialize on the struct exactly
    # COM: once with `{defaultedAssociatedAlias}` preserved, and both conformances
    # COM: (to C, the abstract parent, and to D, the refining provider) must emit
    # COM: a witness binding `T = Int`.
    # CHECK:      lit.alias.decl *"T`{{[0-9]+}}": !AnyType = <!Int>
    # CHECK-SAME:   {defaultedAssociatedAlias}
    # CHECK:      kgen.conformance {{.*}}@C {
    # CHECK-NEXT:   kgen.witness "T" : !AnyType = !Int
    # CHECK:      kgen.conformance {{.*}}@D {
    # CHECK-NEXT:   kgen.witness "T" : !AnyType = !Int
    pass


# COM: A defaulted alias whose default references `Self.X` must still resolve
# COM: that reference correctly when it reaches a conformer through a refining
# COM: trait rather than directly.

# CHECK-LABEL: lit.trait.decl @Parent
trait Parent:
    # CHECK:      lit.alias.decl *"A`{{[0-9]+}}": !alias_Int1 = <rebind(:!Int {:scalar<index> 1})>
    # CHECK-SAME:   {defaultedAssociatedAlias}
    comptime A: Int = 1
    # In Parent, `Self.X` lowers to references against Parent's `_Self`.
    # CHECK:      lit.alias.decl *"B`{{[0-9]+}}": !alias_Int1 = <sugar_member_alias(!kgen.param<:!Parent_AnyType *"_Self`">, "A", #kgen.get_witness<:!Parent_AnyType *"_Self`", @{{.*}}::@Parent, "A">)>
    # CHECK-SAME:   {defaultedAssociatedAlias}
    comptime B: Int = Self.A

# CHECK-LABEL: lit.trait.decl @Child
trait Child(Parent):
    pass



struct ConformsToChild(Child, Movable where False):
    # CHECK:      lit.alias.decl *"A`{{[0-9]+}}": !alias_Int1 = <rebind(:!Int {:scalar<index> 1})>
    # CHECK-SAME:   {defaultedAssociatedAlias}
    # CHECK:      lit.alias.decl *"B`{{[0-9]+}}": !alias_Int1 = <sugar_member_alias(!ConformsToChild, "A", rebind(:!Int {:scalar<index> 1}))>
    # CHECK-SAME:   {defaultedAssociatedAlias}
    # CHECK:      kgen.conformance {{.*}}@Parent {
    # CHECK-NEXT:   kgen.witness "A" : !alias_Int1 = rebind(:!Int {:scalar<index> 1})
    # CHECK-NEXT:   kgen.witness "B" : !alias_Int1 = sugar_member_alias(!ConformsToChild, "A", rebind(:!Int {:scalar<index> 1}))
    pass
