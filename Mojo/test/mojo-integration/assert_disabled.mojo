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

# Verify that assert is a no-op when assertions are disabled.
# %mojo-no-debug-no-assert does not pass -D ASSERT=all, so assertions with
# the default assert_mode="none" are not enabled.
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s


def main():
    # These would trap if assertions were enabled, but they should be no-ops.
    assert False
    assert False, "this should be a no-op"

    # CHECK: assertions disabled, all passed
    print("assertions disabled, all passed")
