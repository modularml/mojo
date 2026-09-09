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

# Inferring the origin parameter of a callee's argument can require an implicit
# conversion whose constructor binds its operand by `ref`.  When the operand is
# an rvalue, it has to be materialized into memory first so that the inferred
# origin has something to point at.


struct RefWrapper[origin: ImmOrigin](Copyable, Movable):
    var n: Int

    @implicit
    def __init__[
        value_origin: ImmOrigin, //
    ](ref[value_origin] value: Int, out self: RefWrapper[value_origin]):
        self.n = value


def take_wrapper(w: RefWrapper) -> Int:
    return w.n


def take_wrapper_param[w: RefWrapper]() -> Int:
    return w.n

# CHECK-LABEL: lit.fn @"infer_origin_of_materialized_rvalue()"
def infer_origin_of_materialized_rvalue():
    var a = 1
    var b = 2

    # The `a + b` rvalue is spilled to a temporary, and both the implicit
    # `RefWrapper` conversion and `take_wrapper` itself are parameterized on
    # that temporary's origin.
    # CHECK: %[[SUM:.*]] = lit.call tail @std::@builtin::@stubs::@SIMD::@"__add__
    # CHECK: %[[TMP:.*]] = lit.var.decl "anonymous*" synth : !lit.ref<!Int, mut *"[[ORIGIN:[^"]*]]">
    # CHECK: lit.ref.store %{{.*}}, %[[TMP]]
    # CHECK: lit.call @{{.*}}@RefWrapper::@"__init__{{.*}}"[mut {{.*}}]<:origin<false> (mutcast mut *"[[ORIGIN]]")
    # CHECK: lit.call @{{.*}}@"take_wrapper{{.*}}"[muttoimm {{.*}}]<:origin<false> (mutcast mut *"[[ORIGIN]]")
    _ = take_wrapper(a + b)

    comptime c = 1
    comptime d = 2

    # Binding the conversion as a *parameter* has no runtime temporary to point
    # at, so the origin is inferred as the comptime origin, and the `RefWrapper`
    # is built by the interpreter from the folded `c + d` instead of being
    # spilled to a stack slot.
    # CHECK: lit.call tail @{{.*}}@"take_wrapper_param{{[^"]*}}"<:origin<false> #lit.comptime.origin
    # CHECK-SAME: apply_result_slot
    # CHECK-SAME: @RefWrapper::@"__init__{{[^"]*}}"<:origin<false> #lit.comptime.origin
    # CHECK-SAME: store_to_mem({:scalar<index> 3})
    _ = take_wrapper_param[c + d]()


def take_span(s: Span[...]):
    pass


# CHECK-LABEL: lit.fn @"infer_origin_of_materialized_literal()"
def infer_origin_of_materialized_literal():
    # Same thing one level deeper: the list literal first materializes an
    # `Array` temporary, which the implicit `Span` conversion then borrows.
    # CHECK: %[[ARRAY:.*]] = lit.var.decl "__call_result_tmp__" synth : !lit.ref<!lit.struct<#Array {{.*}}, mut *"[[ORIGIN:[^"]*]]">
    # CHECK: lit.call @std::@builtin::@stubs::@Span::@"__init__{{.*}}"<:!Bool {:scalar<bool> false}, :origin<false> (mutcast mut *"[[ORIGIN]]")
    # CHECK: lit.call @{{.*}}@"take_span{{.*}}"<:!Bool {:scalar<bool> false}, :origin<false> (mutcast mut *"[[ORIGIN]]")
    take_span([1, 2, 3])
