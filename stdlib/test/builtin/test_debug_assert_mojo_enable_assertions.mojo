# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
#
# This file only tests that conversion of negative IntLiteral to UInt fails.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_not
<<<<<<<< HEAD:stdlib/test/builtin/test_debug_assert_mojo_enable_assertions.mojo
# RUN: not --crash %bare-mojo -D MOJO_ENABLE_ASSERTIONS -debug-level full %s 2>&1 | FileCheck %s
========
# RUN: not mojo %s 2>&1 | FileCheck %s
>>>>>>>> origin/nightly:stdlib/test/builtin/test_uint_error.mojo


fn main():
<<<<<<<< HEAD:stdlib/test/builtin/test_debug_assert_mojo_enable_assertions.mojo
    print("== test_fail")
    # CHECK: Assert Error: this fails
    debug_assert(False, "this fails")
========
    # CHECK: integer value -1 is negative, but is being converted to an unsigned type
    print(UInt(-1))
>>>>>>>> origin/nightly:stdlib/test/builtin/test_uint_error.mojo
    # CHECK-NOT: is never reached
    print("is never reached")
