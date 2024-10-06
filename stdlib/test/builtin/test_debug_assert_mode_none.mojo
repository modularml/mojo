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
# RUN: %bare-mojo -D ASSERT=none %s 2>&1 | FileCheck %s -check-prefix=CHECK-OK


def main():
    test_debug_assert_mode_none_true()
    test_debug_assert_mode_none_false()


# CHECK-OK-LABEL: test_debug_assert_mode_none_true
def test_debug_assert_mode_none_true():
    print("== test_debug_assert_mode_none_true")

    debug_assert[assert_mode="safe"](True, "ok")
    debug_assert[assert_mode="safe", cpu_only=True](True, "ok")
    # CHECK-OK: is reached
    print("is reached")


# CHECK-OK-LABEL: test_debug_assert_mode_none_false
def test_debug_assert_mode_none_false():
    print("== test_debug_assert_mode_none_false")
    debug_assert(False, "ok")
    debug_assert[assert_mode="safe"](False, "ok")
    debug_assert[assert_mode="safe", cpu_only=True](False, "ok")
    # CHECK-OK: is reached
    print("is reached")
