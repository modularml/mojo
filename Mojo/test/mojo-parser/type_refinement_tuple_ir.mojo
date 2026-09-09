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

# Parser-only IR checks for tuple-unpack type refinement.
#
# RUN: %parse-mojo-isolated %s | FileCheck %s


trait Base:
    pass


trait Extra:
    pass


# CHECK-LABEL: lit.fn @"tuple_unpack_param_no_call
# CHECK: [[OUTER_A:%.*]] = lit.var.decl "a" var : !lit.ref<:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T, mut
# CHECK: [[OUTER_B:%.*]] = lit.var.decl "b" var : !lit.ref<:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T, mut
# CHECK: kgen.rebind [[OUTER_A]] : {{.*}}to {{.*}}Extra{{.*}}downcast(:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T){{.*}}
# CHECK: kgen.rebind [[OUTER_B]] : {{.*}}to {{.*}}Extra{{.*}}downcast(:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T){{.*}}
# CHECK: lit.ownership.use %{{.*}}
# CHECK: lit.ownership.use %{{.*}}
def tuple_unpack_param_no_call[T: Base & Copyable & ImplicitlyCopyable](
    imm pair: Tuple[T, T]
) where conforms_to(T, Extra):
    var a, b = pair
    _ = a
    _ = b


# CHECK-LABEL: lit.fn @"tuple_unpack_shadow_no_call
# CHECK: [[OUTER_A:%.*]] = lit.var.decl "a" var : !lit.ref<:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T, mut
# CHECK: [[OUTER_B:%.*]] = lit.var.decl "b" var : !lit.ref<:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T, mut
# CHECK: kgen.rebind [[OUTER_A]] : {{.*}}to {{.*}}Extra{{.*}}downcast(:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T){{.*}}
# CHECK: kgen.rebind [[OUTER_B]] : {{.*}}to {{.*}}Extra{{.*}}downcast(:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T){{.*}}
# CHECK: kgen.param.if <true> {
# CHECK: [[INNER_A:%.*]] = lit.var.decl "a" var : !lit.ref<:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T, mut
# CHECK: [[INNER_B:%.*]] = lit.var.decl "b" var : !lit.ref<:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T, mut
# CHECK: kgen.rebind [[INNER_A]] : {{.*}}to {{.*}}Extra{{.*}}downcast(:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T){{.*}}
# CHECK: kgen.rebind [[INNER_B]] : {{.*}}to {{.*}}Extra{{.*}}downcast(:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T){{.*}}
def tuple_unpack_shadow_no_call[T: Base & Copyable & ImplicitlyCopyable](
    imm first: Tuple[T, T], imm second: Tuple[T, T]
) where conforms_to(T, Extra):
    var a, b = first
    _ = a
    _ = b
    comptime if True:
        var a, b = second
        _ = a
        _ = b


# CHECK-LABEL: lit.fn @"tuple_unpack_ref_no_call
# CHECK: [[A_SLOT:%.*]] = lit.var.decl "a" ref : !lit.ref<!lit.ref<:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T, imm
# CHECK: [[B_SLOT:%.*]] = lit.var.decl "b" ref : !lit.ref<!lit.ref<:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T, imm
# CHECK: [[A_REF:%.*]] = lit.ref.load [[A_SLOT]]
# CHECK: [[A_REBIND:%.*]] = kgen.rebind [[A_REF]] : {{.*}}to {{.*}}Extra{{.*}}downcast(:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T){{.*}}
# CHECK: [[B_REF:%.*]] = lit.ref.load [[B_SLOT]]
# CHECK: [[B_REBIND:%.*]] = kgen.rebind [[B_REF]] : {{.*}}to {{.*}}Extra{{.*}}downcast(:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T){{.*}}
# CHECK: lit.ownership.use [[A_REBIND]]
# CHECK: lit.ownership.use [[B_REBIND]]
def tuple_unpack_ref_no_call[T: Base & Copyable & ImplicitlyCopyable](
    imm pair: Tuple[T, T]
) where conforms_to(T, Extra):
    ref (a, b) = pair
    _ = a
    _ = b


# CHECK-LABEL: lit.fn @"for_ref_capture_single
# CHECK: [[ITEM_SLOT:%.*]] = lit.var.decl "item" ref : !lit.ref<!lit.ref<:{{.*}}Extra{{.*}}downcast(:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T), imm
# CHECK: [[ITEM_REF:%.*]] = lit.ref.load [[ITEM_SLOT]]
# CHECK: lit.ownership.use [[ITEM_REF]]
def for_ref_capture_single[T: Base & ImplicitlyCopyable](
    items: List[T]
) where conforms_to(T, Extra):
    for ref item in items:
        _ = item


# CHECK-LABEL: lit.fn @"for_ref_capture_tuple
# CHECK: [[LEFT_SLOT:%.*]] = lit.var.decl "left" ref : !lit.ref<!lit.ref<:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T, imm
# CHECK: [[RIGHT_SLOT:%.*]] = lit.var.decl "right" ref : !lit.ref<!lit.ref<:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T, imm
# CHECK: [[LEFT_REF:%.*]] = lit.ref.load [[LEFT_SLOT]]
# CHECK: [[LEFT_REBIND:%.*]] = kgen.rebind [[LEFT_REF]] : {{.*}}to {{.*}}Extra{{.*}}downcast(:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T){{.*}}
# CHECK: [[RIGHT_REF:%.*]] = lit.ref.load [[RIGHT_SLOT]]
# CHECK: [[RIGHT_REBIND:%.*]] = kgen.rebind [[RIGHT_REF]] : {{.*}}to {{.*}}Extra{{.*}}downcast(:!AnyType_Copyable_ImplicitlyCopyable_Movable_Base T){{.*}}
# CHECK: lit.ownership.use [[LEFT_REBIND]]
# CHECK: lit.ownership.use [[RIGHT_REBIND]]
def for_ref_capture_tuple[T: Base & ImplicitlyCopyable](
    pairs: List[Tuple[T, T]]
) where conforms_to(T, Extra):
    for ref (left, right) in pairs:
        _ = left
        _ = right
