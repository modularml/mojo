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
# This file only tests the debug_assert function
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_not
# RUN: not --crash %mojo -debug-level full %s 2>&1 | FileCheck %s


# CHECK-LABEL: test_fail
fn main():
    print("== test_fail")
    # CHECK: Assert Error: fail
    debug_assert(False, "fail")
    # CHECK-NOT: is never reached
    print("is never reached")
