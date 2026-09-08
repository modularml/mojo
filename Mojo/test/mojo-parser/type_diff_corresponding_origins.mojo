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

# Regression test for the diagnostic type-differ (`~MojoInflightDiag`).
#
# When a struct fails to satisfy a trait method, the differ must look past the
# implicit `self` origin (which positionally corresponds between the two
# signatures) and report the real argument-type mismatch: here `.argument` is
# `Float64` in the trait requirement but `Int` in the struct's method.
#
# The names are intentionally long (>30 chars) to trigger the differ's
# long-type drill-down heuristic. This exercises the differ directly and does
# not depend on the closure-trait machinery.

# expected-note @below {{trait 'LongishTraitNameForTheDiff' declared here}}
trait LongishTraitNameForTheDiff:
    # expected-note @below {{no 'method_with_a_name' candidates have type 'def(self: StructWithLongishNameForDiff, argument: Float64) thin -> None'}}
    def method_with_a_name(self, argument: Float64):
        ...


# expected-error @below {{'StructWithLongishNameForDiff' does not implement all requirements for 'LongishTraitNameForTheDiff'}}
struct StructWithLongishNameForDiff(LongishTraitNameForTheDiff, Movable where False):
    # expected-note @below {{candidate declared here with type 'def(self: StructWithLongishNameForDiff, argument: Int) thin -> None'}}
    # expected-note @below {{.argument.dtype of the first value is 'DType.float64' but the second value is 'DType.int'}}
    def method_with_a_name(self, argument: Int):
        pass
