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

# RUN: %parse-mojo-isolated -debug-level full -mlir-print-debuginfo %s --kgen-print-inline-type-values -split-input-file | FileCheck %s


##===----------------------------------------------------------------------===##
# def/def
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"testAlwaysInline
# CHECK-SAME: always_inline
@always_inline
def testAlwaysInline():
    # CHECK: lit.return {{.*}} loc(#[[LOC_INLINE:.+]])
    pass


# CHECK-LABEL: lit.fn @"testAlwaysInlineNoDebug
# CHECK-SAME: always_inline_no_debug
@always_inline("nodebug")
def testAlwaysInlineNoDebug():
    # CHECK: lit.return {{.*}} loc(#[[LOC_INLINE_NODEBUG:.+]])
    pass


# CHECK-DAG: #[[LOC_INLINE_NODEBUG]] = loc("{{.+}}":{{[0-9]+}}:{{[0-9]+}})
# CHECK-DAG: #[[LOC_INLINE]] = loc(fused<

# // -----


# CHECK-DAG: lit.fn @"fn_where_clause{{.*}}, #[[LOC_WHERE_FN:loc[0-9]+]]>}{{.*}} attributes
def fn_where_clause[x: Int]() where x:
    pass


# COM: Make sure this is a FileLineColLoc and not a FusedLoc.
# CHECK-DAG: #[[LOC_WHERE_FN]] = loc("{{.*}}":{{[0-9]+}}:{{[0-9]+}})
