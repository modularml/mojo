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

# A `thin` function type may carry trailing `where` clauses constraining its own
# parameters. `w` is not in scope for the alias itself, so this only parses
# because the clause binds to the function type.
#
# The constraints occupy the body-constraint slot of the generator's pog list,
# which an unconstrained generator leaves off entirely.
# CHECK: #alias_Kernel = {{.*}}!lit.generator<<"w": !Int, {
comptime Kernel = def[w: Int](Int) thin -> None where w > 0


# Being part of the type, the constraints show up in the mangled name too.
# CHECK-LABEL: lit.fn @"takes_kernel[
# CHECK-SAME: thin -> None where (
def takes_kernel[F: Kernel]():
    F[4](0)


# Multiple clauses accumulate, and a clause may carry a failure message.
comptime Bounded = def[w: Int]() thin -> None where w > 0 where (
    w < 64, "width must fit a warp"
)


# CHECK-LABEL: lit.fn @"takes_bounded[
# CHECK-SAME: thin -> None where {{.*}}, (lt {{.*}}, 64)
def takes_bounded[F: Bounded]():
    F[8]()


def impl[w: Int](x: Int):
    pass


# An unconstrained function satisfies a constrained function type: it demands
# nothing of its callers that the type has to promise.
# CHECK-LABEL: lit.fn @"bind_unconstrained
def bind_unconstrained():
    takes_kernel[impl]()


# Parenthesizing the result type reattaches the `where` clause to the
# declaration, so the constraint lands on `parenthesized` -- in the `{...}`
# block following its argument list -- rather than on the function type it
# returns, which stays unconstrained.
# CHECK-LABEL: lit.fn @"parenthesized[
# CHECK-SAME: (){(
# CHECK: kgen.create_closure[!lit.generator<() -> !kgen.none>
def parenthesized[n: Int]() -> (def() thin -> None) where n > 0:
    def inner():
        pass

    return inner
