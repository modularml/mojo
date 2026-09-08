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

# RUN: %parse-mojo-isolated -debug-level full -mlir-print-debuginfo %s | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #

# Check single file debug info generation.


@fieldwise_init
struct MemPair(Movable where False):
    var x: Int
    var y: Int


# CHECK-LABEL: lit.fn @"power
def power(lhs: Int, rhs: Int) -> MemPair:
    return MemPair(lhs, rhs)
    # CHECK: lit.end_fn
    # CHECK-NEXT: } loc(#[[LOC_FUNC:.*]])


# CHECK: ![[SP_TYPE:.*]] = !debuginfo.subroutine<(!Int, !Int, !lit.ref<!MemPair, mut *"__result__`">) -> (!kgen.none): DW_CC_normal>
# CHECK: #power_name = #debuginfo.source_name<(fn)"power"(#SIMD_name, #SIMD_name) from <(module)"debuginfo">>
# CHECK: #[[SP:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #{{.*}}, sourceName = #power_name, linkageName = "power{{.*}}", file = #{{.*}}, line = [[LN:[0-9]+]], scopeLine = [[LN]], subprogramFlags = "Definition|Optimized"> : ![[SP_TYPE]]
# CHECK: #[[LOC_FUNC]] = loc(fused<#[[SP]]>
