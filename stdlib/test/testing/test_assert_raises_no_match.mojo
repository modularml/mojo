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
# REQUIRES: has_not
# RUN: not %mojo -debug-level full %s 2>&1 | FileCheck %s -check-prefix=CHECK-FAIL

from testing import assert_raises


# CHECK-FAIL-LABEL: test_assert_raises_no_match
fn test_assert_raises_no_match() raises:
    print("== test_assert_raises_no_match")
    # CHECK-FAIL-NOT: is never reached
    # CHECK: AssertionError
    with assert_raises(contains="Some"):
        raise "OtherError"
    # CHECK-FAIL-NOT: is never reached
    print("is never reached")


def main():
    test_assert_raises_no_match()
