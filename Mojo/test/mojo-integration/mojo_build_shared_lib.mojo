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

# RUN: mkdir %t
# COM: For sake of keeping the test commands simple, this hard-codes the use of
# COM:   `.dylib`, even on platforms like Linux where .so would otherwise be used.
# RUN: %mojo-build %s -o %t/example.dylib --emit shared-lib

# COM: Check that this file compiled to a dynamic library:
# RUN: test -f %t/example.dylib

# COM: Check that `file` recognizes it as a dynamic library
# RUN: file %t/example.dylib | FileCheck %s

# COM: Full output:
# COM:   - macOS: "example.dylib: Mach-O 64-bit dynamically linked shared library arm64"
# COM:   - Linux: "ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), dynamically linked, with debug_info, not stripped"
# CHECK: dynamically linked


@export
def foo() abi("C"):
    pass
