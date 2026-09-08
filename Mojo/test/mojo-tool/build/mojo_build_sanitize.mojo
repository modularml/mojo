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


# Note: Don't run with pre-existing sanitizers to ensure sanitizers work in a
#       clean environment.
# UNSUPPORTED: asan,msan,tsan
# TODO: Support windows when we build with sanitizers.
# TODO: Mac requires using a non-apple clang, as our sanitizers are different.
# UNSUPPORTED: system-darwin

# RUN: not %mojo-build %s --sanitize unknown -o %t 2>&1 | FileCheck %s --check-prefix=ERROR

# ERROR: error: invalid sanitizer 'unknown', expected one of: `address` or `thread`

# Check that we have the expected sanitizer symbols in our built executables.

# RUN: %mojo-build %s --sanitize=address -o %t
# RUN: llvm-objdump %t -t | FileCheck %s --check-prefix=ASAN

# RUN: %mojo-build %s --sanitize thread -o %t
# RUN: llvm-objdump %t -t | FileCheck %s --check-prefix=TSAN

# ASAN: __asan_init
# TSAN: __tsan_init


def main():
    print("sanitizer")
    return
