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

from debuginfo_module import VeryUniqueStruct

# CHECK-DAG: #[[FILE:file[0-9]+]] = #debuginfo.file<"[[FILENAME:.*debuginfo_module.mojo]]" in
# CHECK-DAG: !VeryUniqueStruct

# CHECK-DAG: lit.struct.decl @VeryUniqueStruct
# CHECK-DAG: lit.struct.field very_unique_field : index loc(#[[LOC:loc[0-9]+]])
# CHECK-DAG: lit.fn @"very_unique_func{{.*}}"(%C-3PO: index loc(#[[LINE_LOC:.*]])

# CHECK-DAG: #[[LOC]] = loc(fused<#[[FILE]]>[#loc{{[0-9]+}}])


def caller():
    var y = VeryUniqueStruct.very_unique_func(__mlir_attr.`0 : index`)
