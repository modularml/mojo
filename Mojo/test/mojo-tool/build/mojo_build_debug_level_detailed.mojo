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

# Test that debug level options produce the expected DWARF debug sections.
# This test validates the specific debug sections generated for each level.

# RUN: mkdir -p %t

# Test -g0: No debug metadata at all
# RUN: %mojo-build %s -g0 -o %t/g0.ll --emit llvm
# RUN: FileCheck %s --check-prefix=NO-DEBUG --input-file=%t/g0.ll

# Test -g1: Line tables only (minimal debug info)
# RUN: %mojo-build %s -g1 -o %t/g1.ll --emit llvm
# RUN: FileCheck %s --check-prefix=LINE-TABLES --input-file=%t/g1.ll

# Test -g2/-g: Full debug info including variable locations
# RUN: %mojo-build %s -g2 -o %t/g2.ll --emit llvm
# RUN: FileCheck %s --check-prefix=FULL-DEBUG --input-file=%t/g2.ll

# RUN: %mojo-build %s -g -o %t/g.ll --emit llvm
# RUN: FileCheck %s --check-prefix=FULL-DEBUG --input-file=%t/g.ll

# NO-DEBUG-NOT: !dbg
# NO-DEBUG-NOT: !DICompileUnit
# NO-DEBUG-NOT: !DISubprogram
# NO-DEBUG-NOT: !DIFile

# LINE-TABLES-DAG: !DICompileUnit
# LINE-TABLES-DAG: !DISubprogram
# LINE-TABLES-DAG: !DIFile
# LINE-TABLES-DAG: !dbg

# FULL-DEBUG-DAG: !DICompileUnit
# FULL-DEBUG-DAG: !DISubprogram
# FULL-DEBUG-DAG: !DIFile
# FULL-DEBUG-DAG: !dbg

# LLVM Debug Metadata Explanation:
# - !dbg: Debug location metadata attached to LLVM instructions that maps
#   them back to source file locations
# - !DICompileUnit: Describes the compilation unit (source file) being compiled
# - !DISubprogram: Describes functions with their signatures and source locations
# - !DIFile: Describes source files referenced in the debug information
# - !DILocalVariable: Describes local variables (present with full debug info)
# - !DILocation: Describes specific source locations (file:line:column)

def test_function(x: Int, y: Int) -> Int:
    var local_var = x + y
    var another_var = local_var * 2
    return another_var

def main():
    var result = test_function(5, 10)
    print("Result:", result)