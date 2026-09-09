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
# RUN: %parse-mojo-isolated %s -split-input-file | FileCheck %s

# COM: `==` between `Int` parameters emits a lane-wise `eq` over their extracted
# COM: `_mlir_value` fields, where `==` between type values emits
# COM: `#kgen.param.identical`. Both are identity assertions for int-like
# COM: operands, so both feed the equality saturation behind
# COM: `canDischargeConstraint` -- see `KGEN::getIdentityProposition`.


def needs_equal[N: Int, K: Int]() where N == K:
    pass


# COM: Neither assumption states `N == K` on its own, so discharging the callee's
# COM: constraint takes transitivity across both.
# CHECK-LABEL: lit.fn @"transitive_split
def transitive_split[N: Int, M: Int, K: Int]() where N == M where M == K:
    # CHECK: lit.call tail @{{.*}}needs_equal
    needs_equal[N, K]()


# // -----


def needs_equal[N: Int, K: Int]() where N == K:
    pass


# COM: The same two facts as one conjunction, which the saturation flattens
# COM: before relating them.
# CHECK-LABEL: lit.fn @"transitive_conj
def transitive_conj[N: Int, M: Int, K: Int]() where N == M and M == K:
    # CHECK: lit.call tail @{{.*}}needs_equal
    needs_equal[N, K]()
