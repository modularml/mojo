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
# RUN: %mojo -D ASSERT_WARNING -debug-level full %s | FileCheck %s -check-prefix=CHECK-WARN


# CHECK-WARN: test_ok
fn main():
    print("== test_ok")
    # CHECK-WARN: Assert Warning: failed, but we don't terminate
    debug_assert(False, "failed, but we don't terminate")
    # CHECK-WARN: is reached
    print("is reached")
