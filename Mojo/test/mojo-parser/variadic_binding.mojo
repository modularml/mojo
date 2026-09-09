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


@fieldwise_init
struct SomeCopyable(Copyable):
    pass


@fieldwise_init
struct SomeVA[*elt_types: AnyType](Movable where False):
    def __getitem_param__[idx: Int](ref self) -> ref[self] Self.elt_types[idx]:
        pass


def only_copyable[T: Copyable](t: T):
    pass


# CHECK-LABEL: lit.fn @"f0
# CHECK-SAME: "elt_type.values`": param_list<!AnyType_Copyable_Movable>
# CHECK-SAME: (%t: !lit.ref<!lit.struct<#SomeVA <:param_list<!AnyType> upcast(:param_list<!AnyType_Copyable_Movable> *"elt_type.values`")
def f0[*elt_type: Copyable](t: SomeVA[*elt_type]):
    # Should be able to call only_copyable without an downcast.

    # CHECK lit.call @variadic_binding::@"only_copyable
    only_copyable(t[0])
    pass


# CHECK-LABEL: lit.fn @"f1
# CHECK-SAME: "elt_type.values`1": param_list<meta<!Int>>
# CHECK-SAME:(%t: !lit.ref<!lit.struct<#SomeVA <:param_list<!AnyType> upcast(:param_list<meta<!Int>> *"elt_type.values`1")
def f1[*elt_type: type_of(Int)](t: SomeVA[*elt_type]):
    pass


# CHECK-LABEL: lit.fn @"foo
def foo():
    # CHECK: lit.call {{.*}}@"f0{{.*}}:param_list<!AnyType_Copyable_Movable> [!SomeCopyable, !SomeCopyable]>
    f0(SomeVA[SomeCopyable, SomeCopyable]())

    # CHECK: lit.call {{.*}}@"f1{{.*}}:param_list<meta<!Int>> [!Int, !Int]>
    f1(SomeVA[Int, Int]())
