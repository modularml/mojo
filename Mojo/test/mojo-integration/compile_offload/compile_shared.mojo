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
# REQUIRES: system-linux
# RUN: %mojo %s -o %t
# RUN: llvm-objdump -t %t | FileCheck %s

from std.compile import compile_info
from std.sys import argv, size_of


def get_type(dtype: DType) -> DType:
    return dtype


def compiled_fn[dtype: DType](M: SIMD[get_type(dtype), 4]) -> Int:
    comptime b = size_of[get_type(dtype)]()
    return b + Int(M[0])


def main() raises:
    comptime myCompiledFn = compiled_fn[.uint32]
    # compile myCompiledFn into a shared object binary
    var myShared = compile_info[myCompiledFn, emission_kind="object"]()

    var idx = 0
    var args = argv()
    for arg in argv():
        idx = idx + 1
        if arg == "-o":
            break

    # write the shared object binary to a file for checking
    with open(args[idx], "w") as f:
        f.write(myShared)


# CHECK: dynamic
# CHECK: compile_shared::compiled_fn[::DType]
