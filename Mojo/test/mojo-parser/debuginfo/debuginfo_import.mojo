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

# RUN: %parse-mojo-isolated -I %S/inputs -debug-level full -mlir-print-debuginfo %s | FileCheck %s

from debuginfo_module import imported_fn

# Check that we properly generate functions that get resolved within other functions.
# This is mostly checking that the scope of the nested function is not another function.

# CHECK-DAG: #[[CALLED_STRUCT_BOUND:.*]] = #debuginfo.source_name<(struct)"CalledStruct{{.*}}param{{.*}}"
# CHECK-DAG: #[[CALLED_STRUCT:.*]] = #debuginfo.source_name<(struct)"CalledStruct"[<"index">] from <(module)"debuginfo_import">>
# CHECK-DAG: #test_name = #debuginfo.source_name<(fn)"test"(#[[CALLED_STRUCT_BOUND]]) from #[[CALLED_STRUCT]]>
# CHECK-DAG: #debuginfo.subprogram<compileUnit = #{{.*}}, scope = {{.*}}, sourceName = #test_name, linkageName = "test({{.*}}::CalledStruct[{{.*}}])"


struct CalledStruct[param: __mlir_type.index](Movable where False):
    def test(self):
        imported_fn()


def callerFn[rows: __mlir_type.index](arg0: CalledStruct[rows]):
    return arg0.test()
