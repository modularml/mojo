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

# RUN: %parse-mojo-isolated -debug-level full -verify-diagnostics -mlir-print-debuginfo -I=%S/inputs %s | FileCheck %s

# Test import of a module, and we properly allow import of an imported decl.

from imported_module import *


# CHECK-LABEL: lit.fn @"foo
def foo():
    imported_fn()


# CHECK-LABEL: lit.file_module @imported_module

# CHECK-LABEL: lit.fn @"imported_fn
# CHECK: } loc(#[[LOC_IMPORTED_FN:.+]])

# CHECK: #[[FILE_IMPORTED_MODULE:.+]] = #debuginfo.file<"{{.*}}/imported_module.mojo"
# CHECK: #[[SP_IMPORTED_FN:.+]] = #debuginfo.subprogram<{{.*}}scope = #[[FILE_IMPORTED_MODULE]]{{.*}}linkageName = "imported_fn()"{{.*}}file = #[[FILE_IMPORTED_MODULE]]
# CHECK: #[[LOC_IMPORTED_FN]] = loc(fused<#[[SP_IMPORTED_FN]]
