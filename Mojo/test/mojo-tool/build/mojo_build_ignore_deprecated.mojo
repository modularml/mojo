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

# Test that `mojo build --ignore-deprecated=<name>` (the GNU `=` form, as used
# by BUILD.bazel `copts`) suppresses only the named deprecation warning, while
# an unrelated `@deprecated` declaration still warns.

# RUN: %mojo-build-no-werror --ignore-deprecated=ignoredFn %s 2>&1 | FileCheck %s


# CHECK-NOT: warning: ignoredFn is deprecated
@deprecated("ignoredFn is deprecated")
def ignoredFn():
    pass


# CHECK: warning: notIgnoredFn is deprecated
@deprecated("notIgnoredFn is deprecated")
def notIgnoredFn():
    pass


def main():
    ignoredFn()
    notIgnoredFn()
    return
