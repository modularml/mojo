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

# RUN: %parse-mojo-isolated -verify-diagnostics %s


trait SomeTrait:
    pass


struct StructWithField(Movable where False):
    var field: __mlir_type.index


# Issue #6879: Qualified lookup is looking up names wrong
def unqualified_name_lookup(a: StructWithField):
    # expected-error @+1 {{StructWithField' value has no attribute 'badPropertyError'}}
    a.badPropertyError

    # expected-error @+1 {{StructWithField' value has no attribute 'badPropertyError'}}
    StructWithField.badPropertyError

    # expected-error @+1 {{'SomeTrait' value has no attribute 'value'}}
    SomeTrait.value

    # expected-error @+1 {{cannot access instance field 'field' without an instance of 'StructWithField'}}
    StructWithField.field


struct DirectInstanceReference(Movable where False):
    comptime my_alias: Int = 8
    var value: Int

    def fxn(self):
        # expected-error @+1 {{cannot access instance field 'value' directly; did you mean 'self.'?}}
        var xx = value
        # expected-error @+1 {{cannot access comptime 'my_alias' directly; did you mean 'Self.'?}}
        _ = my_alias

    @staticmethod
    def stat():
        # expected-error @+1 {{cannot access method 'fxn' directly; did you mean 'Self.'?}}
        _ = fxn

    def direct_ref(self):
        # expected-error @+1 {{cannot access method 'fxn' directly; did you mean 'self.'?}}
        fxn(self)
        # expected-error @+1 {{cannot access method 'stat' directly; did you mean 'Self.'?}}
        stat()


def field_indexes(a: DirectInstanceReference):
    # expected-error @+1 {{'DirectInstanceReference' value has no attribute 'badField'}}
    a.badField = 42


trait DirectTraitMemberReference:
    comptime my_alias: Int

    def fxn(self):
        # expected-error @+1 {{cannot access comptime 'my_alias' directly; did you mean 'Self.'?}}
        _ = my_alias

    @staticmethod
    def stat():
        # expected-error @+1 {{cannot access method 'fxn' directly; did you mean 'Self.'?}}
        _ = fxn

    def direct_ref(self):
        # expected-error @+1 {{cannot access method 'fxn' directly; did you mean 'Self.'?}}
        fxn(self)
        # expected-error @+1 {{cannot access method 'stat' directly; did you mean 'Self.'?}}
        stat()


struct StructWithParam[a: Int](Movable where False):
    pass


@fieldwise_init
struct UnqualifiedStructParameterAccess[
    my_param: Int,  # expected-note {{parameter 'my_param' declared here}}
    other_param: Int,
    struct_param: StructWithParam[my_param],  # this should be okay
](Movable where False):
    # expected-error @+1 {{unqualified access to struct parameter 'my_param'; use 'Self.my_param' instead}}
    comptime my_alias = my_param

    def bar(self) -> Int:
        def nested_fn() capturing:
            # expected-error @+1 {{unqualified access to struct parameter 'my_param'; use 'Self.my_param' instead}}
            comptime my_different_alias = my_param

        def shadowing_nested_fn[my_param: Int]() capturing:
            # There should be no warning here because the comptime is shadowed.
            comptime my_different_alias = my_param

        nested_fn()
        shadowing_nested_fn[4]()

        # expected-error @+1 {{unqualified access to struct parameter 'my_param'; use 'Self.my_param' instead}}
        comptime my_other_alias = my_param
        # expected-error @+1 {{unqualified access to struct parameter 'my_param'; use 'Self.my_param' instead}}
        return my_param
