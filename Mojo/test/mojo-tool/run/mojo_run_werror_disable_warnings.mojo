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

# Test that -Werror takes precedence over --disable-warnings for `mojo run`
# When both flags are passed, warnings should be promoted to errors first,
# then the error should NOT be suppressed (only warnings are suppressed by --disable-warnings)

# RUN: not mojo run -Werror --disable-warnings %s 2>&1 | FileCheck %s --check-prefix=BOTH-FLAGS
# RUN: not mojo run -Werror %s 2>&1 | FileCheck %s --check-prefix=ONLY-WERROR

# BOTH-FLAGS: error: assignment to 'foo' was never used
# BOTH-FLAGS-NOT: warning: assignment to 'foo' was never used

# ONLY-WERROR: error: assignment to 'foo' was never used
# ONLY-WERROR-NOT: warning: assignment to 'foo' was never used


def main() raises:
    var foo = 1
