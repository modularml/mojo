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

# Basic IR checks for type refinement.
#
# Keep this test parser-only and focused on refinement rebinds.
#
# RUN: %parse-mojo-isolated %s | FileCheck %s


trait Base:
    pass


trait Extra:
    pass


trait SomeTrait:
    pass


trait StaticExtra:
    @staticmethod
    def static_value() -> Int:
        ...


trait StaticOriginal:
    @staticmethod
    def original_static_value() -> Int:
        ...


trait StaticRefined:
    @staticmethod
    def refined_static_value() -> Int:
        ...


trait GuardedParam:
    pass


trait OriginalParam:
    pass


trait RefinedParam:
    pass


def accepts_guarded_param[T: GuardedParam]():
    pass


def accepts_original_param[T: OriginalParam]():
    pass


def accepts_refined_param[T: RefinedParam]():
    pass


@fieldwise_init
struct GuardedParamBox[T: GuardedParam](Movable where False):
    pass


def use_base[T: Base](imm x: T):
    pass


def use_extra[T: Extra](imm x: T):
    pass


struct MiniPack[
    element_trait: type_of(AnyType), //, *element_types: element_trait
](Movable where False):
    def get_element[
        index: Int
    ](self) -> ref[self] Self.element_types[index]:
        while True:
            pass

    # CHECK-LABEL: lit.fn @"refine_param_list_get_after_assert
    # CHECK: kgen.param.assert <sugar_preserved({{.*}}conforms_to(:!kgen.param<{{.*}}element_trait> #kgen.param_list.get<{{.*}}element_types.values{{.*}}, to_builtin(:scalar<index> #lit.struct.extract<:!Int i, "_mlir_value">)>{{.*}}>
    # CHECK: [[ITEM:%.*]] = lit.call tail @type_refinement_ir::@MiniPack::@"get_element
    # CHECK: [[REBIND:%.*]] = kgen.rebind [[ITEM]] : {{.*}}#kgen.param_list.get<{{.*}}element_types.values{{.*}}, to_builtin(:scalar<index> #lit.struct.extract<:!Int i, "_mlir_value">)>{{.*}} to {{.*}}downcast(:!kgen.param<{{.*}}> #kgen.param_list.get<{{.*}}element_types.values{{.*}}, to_builtin(:scalar<index> #lit.struct.extract<:!Int i, "_mlir_value">)>){{.*}}
    # CHECK: lit.call{{.*}}@"use_extra{{.*}}([[REBIND]])
    def refine_param_list_get_after_assert[i: Int](self):
        comptime element_type = Self.element_types[i]
        comptime assert conforms_to(element_type, Extra)
        use_extra(self.get_element[i]())


struct BaseMiniPack[
    element_trait: type_of(Base), //, *element_types: element_trait
](Movable where False):
    def get_element[
        index: Int
    ](self) -> ref[self] Self.element_types[index]:
        while True:
            pass

    # CHECK-LABEL: lit.fn @"refine_param_list_get_preserves_parametric_bound
    # CHECK: kgen.param.assert <sugar_preserved({{.*}}conforms_to(:!kgen.param<{{.*}}element_trait> #kgen.param_list.get<{{.*}}element_types.values{{.*}}, to_builtin(:scalar<index> #lit.struct.extract<:!Int i, "_mlir_value">)>{{.*}}>
    # CHECK: [[EXTRA_ITEM:%.*]] = lit.call tail @type_refinement_ir::@BaseMiniPack::@"get_element
    # CHECK: [[EXTRA_REBIND:%.*]] = kgen.rebind [[EXTRA_ITEM]] : {{.*}}#kgen.param_list.get<{{.*}}element_types.values{{.*}}, to_builtin(:scalar<index> #lit.struct.extract<:!Int i, "_mlir_value">)>{{.*}} to {{.*}}Base_Extra downcast(:!kgen.param<{{.*}}> #kgen.param_list.get<{{.*}}element_types.values{{.*}}, to_builtin(:scalar<index> #lit.struct.extract<:!Int i, "_mlir_value">)>){{.*}}
    # CHECK: lit.call{{.*}}@"use_extra{{.*}}([[EXTRA_REBIND]])
    # CHECK: [[BASE_ITEM:%.*]] = lit.call tail @type_refinement_ir::@BaseMiniPack::@"get_element
    # CHECK: [[BASE_REBIND:%.*]] = kgen.rebind [[BASE_ITEM]] : {{.*}}#kgen.param_list.get<{{.*}}element_types.values{{.*}}, to_builtin(:scalar<index> #lit.struct.extract<:!Int i, "_mlir_value">)>{{.*}} to {{.*}}Base_Extra downcast(:!kgen.param<{{.*}}> #kgen.param_list.get<{{.*}}element_types.values{{.*}}, to_builtin(:scalar<index> #lit.struct.extract<:!Int i, "_mlir_value">)>){{.*}}
    # CHECK: lit.call{{.*}}@"use_base{{.*}}([[BASE_REBIND]])
    def refine_param_list_get_preserves_parametric_bound[i: Int](self):
        comptime element_type = Self.element_types[i]
        comptime assert conforms_to(element_type, Extra)
        use_extra(self.get_element[i]())
        use_base(self.get_element[i]())


# CHECK-LABEL: lit.fn @"refine_type_base_static_member
# CHECK: kgen.param.if <{{.*}}conforms_to(:!AnyType T, :meta<!{{.*}}StaticExtra> !{{.*}}StaticExtra){{.*}}> {
# CHECK: lit.call{{.*}}#kgen.get_witness<:!AnyType T, @type_refinement_ir::@StaticExtra, "static_value{{.*}}">
def refine_type_base_static_member[T: AnyType]():
    comptime if conforms_to(T, StaticExtra):
        _ = T.static_value()


# CHECK-LABEL: lit.fn @"refine_type_base_static_member_preserves_original_bound
# CHECK: lit.call{{.*}}#kgen.get_witness<:!AnyType_StaticOriginal T, @type_refinement_ir::@StaticRefined, "refined_static_value{{.*}}">
# CHECK: lit.call{{.*}}#kgen.get_witness<:!AnyType_StaticOriginal T, @type_refinement_ir::@StaticOriginal, "original_static_value{{.*}}">
def refine_type_base_static_member_preserves_original_bound[
    T: StaticOriginal
]():
    comptime if conforms_to(T, StaticRefined):
        _ = T.refined_static_value()
        _ = T.original_static_value()


# CHECK-LABEL: lit.fn @"refine_type_base_alias_static_member_preserves_original_bound
# CHECK: lit.call{{.*}}#kgen.get_witness<:!AnyType_StaticOriginal T, @type_refinement_ir::@StaticRefined, "refined_static_value{{.*}}">
# CHECK: lit.call{{.*}}#kgen.get_witness<:!AnyType_StaticOriginal T, @type_refinement_ir::@StaticOriginal, "original_static_value{{.*}}">
def refine_type_base_alias_static_member_preserves_original_bound[
    T: StaticOriginal
]():
    comptime Alias = T
    comptime if conforms_to(T, StaticRefined):
        _ = Alias.refined_static_value()
        _ = Alias.original_static_value()


# CHECK-LABEL: lit.fn @"refine_type_of_static_member_preserves_original_bound
# CHECK: lit.call{{.*}}#kgen.get_witness<:!AnyType_StaticOriginal T, @type_refinement_ir::@StaticRefined, "refined_static_value{{.*}}">
# CHECK: lit.call{{.*}}#kgen.get_witness<:!AnyType_StaticOriginal T, @type_refinement_ir::@StaticOriginal, "original_static_value{{.*}}">
def refine_type_of_static_member_preserves_original_bound[
    T: StaticOriginal
](imm x: T):
    comptime if conforms_to(T, StaticRefined):
        _ = type_of(x).refined_static_value()
        _ = type_of(x).original_static_value()


# CHECK-LABEL: lit.fn @"refine_type_value_function_binding
# CHECK: lit.call{{.*}}@"accepts_guarded_param{{.*}}<:!AnyType_GuardedParam downcast(:!AnyType T)>
def refine_type_value_function_binding[T: AnyType]():
    comptime if conforms_to(T, GuardedParam):
        accepts_guarded_param[T]()


# CHECK-LABEL: lit.fn @"refine_type_value_struct_binding
# CHECK: #GuardedParamBox <:!AnyType_GuardedParam downcast(:!AnyType T)>
def refine_type_value_struct_binding[T: AnyType]():
    comptime if conforms_to(T, GuardedParam):
        _ = GuardedParamBox[T]()


# CHECK-LABEL: lit.fn @"refine_type_value_binding_preserves_original_bound
# CHECK: lit.call{{.*}}@"accepts_refined_param{{.*}}<:!AnyType_RefinedParam upcast(:!AnyType_OriginalParam_RefinedParam downcast(:!AnyType_OriginalParam T))>
# CHECK: lit.call{{.*}}@"accepts_original_param{{.*}}<:!AnyType_OriginalParam T>
def refine_type_value_binding_preserves_original_bound[T: OriginalParam]():
    comptime if conforms_to(T, RefinedParam):
        accepts_refined_param[T]()
        accepts_original_param[T]()


# CHECK-LABEL: lit.fn @"refine_type_value_alias_binding_preserves_original_bound
# CHECK: lit.call{{.*}}@"accepts_refined_param{{.*}}<:!AnyType_RefinedParam upcast(:!AnyType_OriginalParam_RefinedParam downcast(:!AnyType_OriginalParam T))>
# CHECK: lit.call{{.*}}@"accepts_original_param{{.*}}<:!AnyType_OriginalParam #alias_Alias
def refine_type_value_alias_binding_preserves_original_bound[
    T: OriginalParam
]():
    comptime Alias = T
    comptime if conforms_to(T, RefinedParam):
        accepts_refined_param[Alias]()
        accepts_original_param[Alias]()


struct OriginalParamPack[*Ts: OriginalParam](Movable where False):
    # CHECK-LABEL: lit.fn @"refine_type_value_variadic_binding_preserves_original_bound
    # CHECK: lit.call{{.*}}@"accepts_refined_param{{.*}}<:!AnyType_RefinedParam upcast(:!AnyType_OriginalParam_RefinedParam downcast(:!AnyType_OriginalParam #kgen.param_list.get<{{.*}}Ts.values{{.*}}))>
    # CHECK: lit.call{{.*}}@"accepts_original_param
    def refine_type_value_variadic_binding_preserves_original_bound[
        i: Int
    ](self):
        comptime assert conforms_to(Self.Ts[i], RefinedParam)
        accepts_refined_param[Self.Ts[i]]()
        accepts_original_param[Self.Ts[i]]()


trait HasOriginalParamElement:
    comptime Element: OriginalParam


# CHECK-LABEL: lit.fn @"refine_type_value_associated_binding_preserves_original_bound
# CHECK: lit.call{{.*}}@"accepts_refined_param{{.*}}<:!AnyType_RefinedParam upcast(:!AnyType_OriginalParam_RefinedParam downcast(:!AnyType_OriginalParam #kgen.get_witness<{{.*}}Element{{.*}}))>
# CHECK: lit.call{{.*}}@"accepts_original_param
def refine_type_value_associated_binding_preserves_original_bound[C: AnyType]():
    comptime if conforms_to(C, HasOriginalParamElement):
        comptime assert conforms_to(C.Element, RefinedParam)
        accepts_refined_param[C.Element]()
        accepts_original_param[C.Element]()


# CHECK-LABEL: lit.fn @"refine_from_where
# CHECK-SAME: ([[ARG:%.*]]: !lit.ref<:!AnyType_Base T, imm *"x`"> imm_mem)
# CHECK: [[WHERE_REF:%.*]] = kgen.rebind [[ARG]] : {{.*}} to {{.*}}Base_Extra downcast(:!AnyType_Base T){{.*}}
# CHECK: lit.call{{.*}}@{{.*}}@"use_base
# CHECK: lit.call{{.*}}@{{.*}}@"use_extra
def refine_from_where[T: Base](imm x: T) where conforms_to(T, Extra):
    use_base(x)
    use_extra(x)


# CHECK-LABEL: lit.fn @"refine_in_comptime_if
# CHECK-SAME: ([[ARG:%.*]]: !lit.ref<:!AnyType_Base T, imm *"x`"> imm_mem)
# CHECK: kgen.param.if <{{.*}}conforms_to(:!AnyType_Base T, :meta<!{{.*}}Extra> !{{.*}}Extra){{.*}}> {
# CHECK: [[IF_REF:%.*]] = kgen.rebind [[ARG]] : {{.*}} to {{.*}}Base_Extra downcast(:!AnyType_Base T){{.*}}
# CHECK: lit.call{{.*}}@{{.*}}@"use_base
# CHECK: lit.call{{.*}}@{{.*}}@"use_extra
def refine_in_comptime_if[T: Base](imm x: T):
    comptime if conforms_to(T, Extra):
        use_base(x)
        use_extra(x)


# CHECK-LABEL: lit.fn @"refine_after_comptime_assert
# CHECK-SAME: ([[ARG:%.*]]: !lit.ref<:!AnyType_Base T, imm *"x`"> imm_mem)
# CHECK: kgen.param.assert <{{.*}}conforms_to(:!AnyType_Base T, :meta<!{{.*}}Extra> !{{.*}}Extra){{.*}}>
# CHECK: [[ASSERT_REF:%.*]] = kgen.rebind [[ARG]] : {{.*}} to {{.*}}Base_Extra downcast(:!AnyType_Base T){{.*}}
# CHECK: lit.call{{.*}}@{{.*}}@"use_base
# CHECK: lit.call{{.*}}@{{.*}}@"use_extra
def refine_after_comptime_assert[T: Base](imm x: T):
    comptime assert conforms_to(T, Extra)
    use_base(x)
    use_extra(x)


# CHECK-LABEL: lit.fn @"refinement_does_not_leak_after_comptime_if
# CHECK-SAME: ([[ARG:%.*]]: !lit.ref<:!AnyType_Base T, imm *"x`"> imm_mem)
# CHECK: kgen.param.if <{{.*}}conforms_to(:!AnyType_Base T, :meta<!{{.*}}Extra> !{{.*}}Extra){{.*}}> {
# CHECK: [[IF_REF:%.*]] = kgen.rebind [[ARG]] : {{.*}} to {{.*}}Base_Extra downcast(:!AnyType_Base T){{.*}}
# CHECK: lit.call{{.*}}@{{.*}}@"use_extra
# CHECK-NOT: kgen.rebind [[ARG]]
# CHECK: lit.call{{.*}}@{{.*}}@"use_base
def refinement_does_not_leak_after_comptime_if[T: Base](imm x: T):
    comptime if conforms_to(T, Extra):
        use_extra(x)
    use_base(x)


# CHECK-LABEL: lit.fn @"no_refinement_before_comptime_assert
# CHECK-SAME: ([[ARG:%.*]]: !lit.ref<:!AnyType_Base T, imm *"x`"> imm_mem)
# CHECK-NOT: kgen.rebind [[ARG]]
# CHECK: lit.call{{.*}}@{{.*}}@"use_base
# CHECK: kgen.param.assert <{{.*}}conforms_to(:!AnyType_Base T, :meta<!{{.*}}Extra> !{{.*}}Extra){{.*}}>
# CHECK: [[ASSERT_REF:%.*]] = kgen.rebind [[ARG]] : {{.*}} to {{.*}}Base_Extra downcast(:!AnyType_Base T){{.*}}
# CHECK: lit.call{{.*}}@{{.*}}@"use_extra
def no_refinement_before_comptime_assert[T: Base](imm x: T):
    use_base(x)
    comptime assert conforms_to(T, Extra)
    use_extra(x)


# CHECK-LABEL: lit.fn @"refine_in_comptime_if_no_call
# CHECK-SAME: ([[ARG:%.*]]: !lit.ref<:!AnyType_SomeTrait T, mut *"x`"> owned_in_mem)
# CHECK: kgen.param.if <{{.*}}conforms_to(:!AnyType_SomeTrait T, :meta<!{{.*}}Deinitable> !{{.*}}Deinitable){{.*}}> {
# CHECK: [[IF_REBIND:%.*]] = kgen.rebind [[ARG]] : {{.*}} to {{.*}}Deinitable{{.*}}downcast(:!AnyType_SomeTrait T){{.*}}
# CHECK: lit.ownership.use [[IF_REBIND]]
def refine_in_comptime_if_no_call[T: SomeTrait](var x: T):
    comptime if conforms_to(T, Deinitable):
        _ = x


# CHECK-LABEL: lit.fn @"refine_after_comptime_assert_no_call
# CHECK-SAME: ([[ARG:%.*]]: !lit.ref<:!AnyType_SomeTrait T, mut *"x`"> owned_in_mem)
# CHECK: kgen.param.assert <{{.*}}conforms_to(:!AnyType_SomeTrait T, :meta<!{{.*}}Deinitable> !{{.*}}Deinitable){{.*}}>
# CHECK: [[ASSERT_REBIND:%.*]] = kgen.rebind [[ARG]] : {{.*}} to {{.*}}Deinitable{{.*}}downcast(:!AnyType_SomeTrait T){{.*}}
# CHECK: lit.ownership.use [[ASSERT_REBIND]]
def refine_after_comptime_assert_no_call[T: SomeTrait](var x: T):
    comptime assert conforms_to(T, Deinitable)
    _ = x


# A `where conforms_to(Self.T, Refined)` clause must widen the effective
# trait bound of `Self.T` when matching it against a stricter declared
# parameter bound.
struct ParamBox[T: AnyType](Movable where False):
    pass


def accepts_refined_box[T: RefinedParam](box: ParamBox[T]):
    pass


struct WhereParamMatcherContainer[T: OriginalParam](Movable where False):
    def refine_where_during_nested_param_binding(
        self, box: ParamBox[Self.T]
    ) where conforms_to(Self.T, RefinedParam):
        # Inferred binding: `accepts_refined_value`'s T = Self.T, which is
        # declared `OriginalParam` but widened to `RefinedParam` by the
        # where clause.
        accepts_refined_box(box)
