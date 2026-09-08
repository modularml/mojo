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
# RUN: %mojo -debug-level full %s | FileCheck %s

from std.compile import compile_info
from std.sys import size_of


def get_type(dtype: DType) -> DType:
    return dtype


def compiled_fn[dtype: DType](M: SIMD[get_type(dtype), 4]) -> Int:
    comptime b = size_of[get_type(dtype)]()
    return b + Int(M[0])


def main():
    comptime myCompiledFn = compiled_fn[.uint32]
    print(compile_info[myCompiledFn, emission_kind="llvm"]())


# CHECK: define {{.*}} @"compile_assembly_debuginfo::compiled_fn{{.*}} !dbg ![[SP:[0-9]+]]
# CHECK-NOT @"compile_assembly_debuginfo::compiled_fn
# The metadata nodes now print dependencies-first (llvm/llvm-project#216838), so
# the arg/function/subroutine types appear before the subprogram that uses them.
# The function arg type should have been concretized into the actual dtype.
# CHECK: ![[ARG_TYPE:[0-9]+]] = !DICompositeType(tag: DW_TAG_array_type, name: "!kgen.simd<4, ui32>"
# CHECK: ![[FUNCTION_TYPE:[0-9]+]] = !{!{{[0-9]+}}, ![[ARG_TYPE]]}
# CHECK: ![[SUBROUTINE:[0-9]+]] = !DISubroutineType({{.*}}types: ![[FUNCTION_TYPE]]
# CHECK: ![[SP]] = distinct !DISubprogram({{.*}}type: ![[SUBROUTINE]]
