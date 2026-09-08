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

# Test that debug level options control debug metadata in LLVM IR output.

# RUN: mkdir -p %t

# Build with -g0 and verify no debug metadata
# RUN: %mojo-build %s -g0 -o %t/no_debug.ll --emit llvm
# RUN: FileCheck %s --check-prefix=NO-DEBUG --input-file=%t/no_debug.ll

# Build with -g1, -g2, -g and verify debug metadata is present
# RUN: %mojo-build %s -g1 -o %t/g1_debug.ll --emit llvm  
# RUN: FileCheck %s --check-prefix=WITH-DEBUG --input-file=%t/g1_debug.ll

# RUN: %mojo-build %s -g2 -o %t/g2_debug.ll --emit llvm  
# RUN: FileCheck %s --check-prefix=WITH-DEBUG --input-file=%t/g2_debug.ll

# RUN: %mojo-build %s -g -o %t/with_debug.ll --emit llvm  
# RUN: FileCheck %s --check-prefix=WITH-DEBUG --input-file=%t/with_debug.ll

# Also test --debug-level options
# RUN: %mojo-build %s --debug-level=none -o %t/none.ll --emit llvm
# RUN: FileCheck %s --check-prefix=NO-DEBUG --input-file=%t/none.ll

# RUN: %mojo-build %s --debug-level=line-tables -o %t/line.ll --emit llvm
# RUN: FileCheck %s --check-prefix=WITH-DEBUG --input-file=%t/line.ll

# RUN: %mojo-build %s --debug-level=full -o %t/full.ll --emit llvm
# RUN: FileCheck %s --check-prefix=WITH-DEBUG --input-file=%t/full.ll

# NO-DEBUG-NOT: !dbg
# NO-DEBUG-NOT: !DICompileUnit
# NO-DEBUG-NOT: !DISubprogram
# NO-DEBUG-NOT: !DIFile

# WITH-DEBUG-DAG: !dbg
# WITH-DEBUG-DAG: !DICompileUnit
# WITH-DEBUG-DAG: !DISubprogram
# WITH-DEBUG-DAG: !DIFile

def add_numbers(x: Int, y: Int) -> Int:
    return x + y

def main():
    var result = add_numbers(5, 10)
    print(result)