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
<<<<<<<< HEAD:stdlib/test/sys/test_dlhandle.mojo
# RUN: not --crash mojo %s 2>&1

from sys import DLHandle


def check_invalid_dlhandle():
    _ = DLHandle("/an/invalid/library")


def main():
    check_invalid_dlhandle()
========
# RUN: not --crash %bare-mojo %s 2>&1 | FileCheck %s -check-prefix=CHECK-FAIL


# CHECK-FAIL-LABEL: test_fail
fn main():
    print("== test_fail")
    # CHECK-FAIL: formatted failure message: 2, 4
    debug_assert[assert_mode="safe"](
        False, "formatted failure message: ", 2, ", ", Scalar[DType.uint8](4)
    )
    # CHECK-FAIL-NOT: is never reached
    print("is never reached")
>>>>>>>> origin/nightly:stdlib/test/builtin/test_debug_assert_default_error.mojo
