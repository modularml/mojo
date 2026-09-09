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

# Binding a function to a function type carrying trailing `where` clauses. The
# obligations are part of the type, so the direction decides whether the
# conversion is free: a function may promise less than the type it is bound to,
# never more. See `fn_type_where_conversion_errors.mojo` for the refusals.

# RUN: %parse-mojo-isolated %s | FileCheck %s


trait Marker:
    pass


comptime MarkedKernel = def[T: AnyType](Int) thin -> None where conforms_to(
    T, Marker
)


def marked_impl[T: AnyType](x: Int) where conforms_to(T, Marker):
    pass


def plain_impl[T: AnyType](x: Int):
    pass


def takes_marked[F: MarkedKernel]():
    pass


##===----------------------------------------------------------------------===##
# The obligations match
##===----------------------------------------------------------------------===##


# The obligation is stated in two places, so the two `ConstraintAttr`s differ in
# source location while denoting the same requirement. The rebound source keeps
# its body-constraint block, i.e. nothing was shed to make the binding work.
# CHECK-LABEL: lit.fn @"bind_marked()"
# CHECK: lit.call {{.*}}@"takes_marked[def[::AnyType]({{.*}}) thin -> None where conforms_to(
# CHECK-SAME: rebind(:!lit.generator<<"T": !AnyType, {<sugar_preserved(
# CHECK-SAME: @"marked_impl[
def bind_marked():
    takes_marked[marked_impl]()


##===----------------------------------------------------------------------===##
# The function promises less than the type
##===----------------------------------------------------------------------===##


# An unconstrained function demands nothing of its callers, so it satisfies a
# constrained type outright, and gaining an obligation stays a zero-cost rebind:
# the rebind names `plain_impl` directly. Treating this direction as unproven
# instead routes it through the generator-conversion path, which reaches the same
# verdict only after synthesizing a transparent thunk in between -- so the
# generator type below sitting immediately before `plain_impl` is the assertion
# that no thunk was introduced.
# CHECK-LABEL: lit.fn @"bind_plain()"
# CHECK: lit.call {{.*}}@"takes_marked[def[::AnyType]({{.*}}) thin -> None where conforms_to(
# CHECK-SAME: )]()"<:!alias_MarkedKernel{{[0-9]*}} rebind(:!lit.generator<<"T": !AnyType>("x": !Int) -> !kgen.none> {{.*}}@"plain_impl[
def bind_plain():
    takes_marked[plain_impl]()


##===----------------------------------------------------------------------===##
# The caller supplies the evidence
##===----------------------------------------------------------------------===##


comptime PlainKernel = def(Int) thin -> None


def takes_plain[F: PlainKernel]():
    pass


# Binding `T` at the reference site makes the obligation concrete, and the
# caller's own `where` clause is the evidence for it, so it can be shed to reach
# the unconstrained type: the rebind target carries no body-constraint block.
# The obligation can only be discharged because it is quantified over a
# parameter the caller binds -- one quantified over the function type's own
# parameter is out of any caller's reach.
# CHECK-LABEL: lit.fn @"shed_with_caller_evidence[
# CHECK: lit.call {{.*}}@"takes_plain[def({{.*}}) thin -> None]
# CHECK-SAME: rebind(:!lit.generator<("x": !Int) -> !kgen.none>
def shed_with_caller_evidence[T: AnyType]() where conforms_to(T, Marker):
    takes_plain[marked_impl[T]]()


##===----------------------------------------------------------------------===##
# The same rule applies to non-function generator types
##===----------------------------------------------------------------------===##

# A `where` clause on a `comptime` generator alias is a body constraint too, so
# `canZeroCostConvert` treats these exactly like function types: gaining an
# obligation is free, shedding an unprovable one is refused.

comptime plain_gen[U: AnyType] = U


comptime marked_gen[U: AnyType] where conforms_to(U, Marker) = U


comptime MarkedGenType = type_of(marked_gen)


def takes_marked_gen[F: MarkedGenType]():
    pass


# The unconstrained alias satisfies the constrained generator type outright, and
# gaining the obligation stays a zero-cost rebind naming `plain_gen` directly --
# no thunk or discharge is synthesized in between.
# CHECK-LABEL: lit.fn @"gain_nonfn_constraint()"
# CHECK: lit.call {{.*}}@"takes_marked_gen[__generator_type[::AnyType] ::AnyType where conforms_to(
# CHECK-SAME: rebind(:!lit.generator<<"U": !AnyType>!AnyType> {{.*}}plain_gen
def gain_nonfn_constraint():
    takes_marked_gen[plain_gen]()
