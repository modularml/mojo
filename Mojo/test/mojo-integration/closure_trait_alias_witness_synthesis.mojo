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

# Regression test for MOCO-4281 (bug B): a `comptime` alias for a closure
# function-type signature, used as a generic parameter bound, failed to
# synthesize a witness table for a *capturing* closure passed to that
# parameter.
#
# `IREmitter::canMetaTypeUpCastTo`/`emitTypeValueUpCastToTrait` each have two
# branches for matching a value against a trait-bound target: one reached when
# the target's *metatype* is `AnyTraitType`, and one reached when the target
# is itself directly `AnyTraitType`. Only the first had a closure-rebindability
# check (which synthesizes the wrapper struct's witness table as needed). A
# bound reached through a `comptime` alias resolves to the second, bare-
# `AnyTraitType` shape, which fell straight through to a plain conformance
# check with no witness synthesis -- so the wrapper struct for a capturing
# closure never got a witness table entry, and elaboration later failed to
# find one.
#
# Writing the bound inline (`[F: def(Int) -> Int]`), or passing a
# non-capturing closure, does not trigger the failure -- the inline case
# resolves through the first (already-working) branch, and a non-capturing
# closure never needs witness-table dispatch at all.

# RUN: kgen -elaborate -O0 %s -S


comptime IntTransform = def(Int) -> Int


def apply[F: IntTransform](f: F, x: Int) -> Int:
    return f(x)


def main():
    var scale = 2

    def scale_by(x: Int) {var scale} -> Int:
        return x * scale

    _ = apply(scale_by, 21)
