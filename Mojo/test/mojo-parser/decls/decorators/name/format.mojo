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

# Test that @__name can accept a t-string argument directly, producing a
# parametric linkage name expression.


# CHECK: lit.fn @"parametric_tstring[::SIMD[DType.int, 1],::String]()"
# CHECK-SAME: linkageName = #kgen.linkage_name<#kgen.param.expr<data_to_str
# CHECK-SAME: __make_tstring{{.*}}:string "my_name_{}_{}"
@__name(t"my_name_{A}_{B}")
@no_inline
def parametric_tstring[A: Int, B: String]():
    pass


# Test a t-string with no interpolations. Even with a static template the
# linkage name is represented as a param.expr because it goes through
# StringSlice construction.

# CHECK: lit.fn @"static_tstring()"
# CHECK-SAME: linkageName = #kgen.linkage_name<#kgen.param.expr<data_to_str
@__name(t"static_name")
def static_tstring():
    pass
