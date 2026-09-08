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


# A callee whose type is named by a `comptime` alias carries a `#kgen.sugar`
# wrapper. `lit.bind_params` requires an operand that is structurally a function
# generator, so the callee is rebound to the desugared type before its
# parameters are bound. Each case below checks the sugar kind it covers on the
# rebind operand.

# Use an explicit Origin parameter so the type is a valid runtime fn pointer
# argument (bare `def(ref arg: Int)` invents a non-singleton Bool mutability
# parameter and is rejected as an argument type).
comptime FnT = def[a: MutOrigin](ref [a] arg: Int) thin abi("C") -> Int


# CHECK-LABEL: lit.fn @"__call__{{.*}}PackWrapper
# CHECK: %[[FP:[0-9]+]] = kgen.rebind {{.*}}sugar_member_alias
# CHECK-NEXT: %[[BP:[0-9]+]] = lit.bind_params %[[FP]]
# CHECK-NEXT: lit.call_indirect tail %[[BP]]
struct PackWrapper[*Args: AnyType, Ret: AnyType]:
    comptime T = def(*args: *Self.Args) thin abi("C") -> Self.Ret
    var ptr: Self.T

    def __call__(self, *args: *Self.Args) -> Self.Ret:
        return self.ptr(*args)


# The sugar is what needs rebinding, not the argument pack: a fixed-arity
# callee with a singleton Origin parameter takes the same path.

# CHECK-LABEL: lit.fn @"__call__{{.*}}RefWrapper
# CHECK: %[[FP2:[0-9]+]] = kgen.rebind {{.*}}sugar_member_alias
# CHECK-NEXT: %[[BP2:[0-9]+]] = lit.bind_params %[[FP2]]
# CHECK-NEXT: lit.call_indirect tail %[[BP2]]
struct RefWrapper[Arg: AnyType, Ret: AnyType]:
    comptime T = def[a: MutOrigin](ref [a] arg: Self.Arg) thin abi("C") -> Self.Ret
    var ptr: Self.T

    def __call__(self, mut arg: Self.Arg) -> Self.Ret:
        return self.ptr(arg)


# A plain alias reference is a different sugar kind than a `T.x` member alias,
# and it reaches the same site without a struct or a field: this callee is an
# ordinary argument.

# CHECK-LABEL: lit.fn @"call_through_alias
# CHECK: %[[FP3:[0-9]+]] = kgen.rebind {{.*}} : !alias_FnT
# CHECK-NEXT: %[[BP3:[0-9]+]] = lit.bind_params %[[FP3]]
# CHECK-NEXT: lit.call_indirect tail %[[BP3]]
def call_through_alias(p: FnT, mut x: Int) -> Int:
    return p(x)


# A plain alias stored in a struct as a member still carries its plain alias
# sugar kind, distinct from a member alias.

# CHECK-LABEL: lit.fn @"__call__{{.*}}PlainAliasWrapper
# CHECK: %[[FP4:[0-9]+]] = kgen.rebind {{.*}} : !alias_FnT
# CHECK-NEXT: %[[BP4:[0-9]+]] = lit.bind_params %[[FP4]]
# CHECK-NEXT: lit.call_indirect tail %[[BP4]]
struct PlainAliasWrapper:
    var ptr: FnT

    def __call__(self, mut x: Int) -> Int:
        return self.ptr(x)
