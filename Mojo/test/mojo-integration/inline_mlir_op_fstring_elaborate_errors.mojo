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

# RUN: not kgen -elaborate -O0 %s -S 2>&1 | FileCheck %s

# A deferred f-string op whose template is only well-formed once parameters
# bind: the parser cannot reject it, so the elaborator's `lowerFStringMLIROp`
# re-parse is what surfaces the malformed assembly after binding.


# CHECK: failed to parse f-string MLIR op
def bogus[
    T: __mlir_type.`!kgen.dtype`
](x: __mlir_type[`!kgen.scalar<`, T, `>`]) -> __mlir_type[
    `!kgen.scalar<`, T, `>`
]:
    return __mlir_op[`pop.add totally bogus %{x} : %{type_of(x)}`]


@export
def top(
    a: __mlir_type.`!kgen.scalar<si32>`,
) abi("Mojo") -> __mlir_type.`!kgen.scalar<si32>`:
    return bogus[__mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`](a)
