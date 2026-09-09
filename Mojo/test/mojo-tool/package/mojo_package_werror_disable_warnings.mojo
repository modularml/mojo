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

# Test that -Werror takes precedence over --disable-warnings for `mojo precompile`

# RUN: not mojo precompile -Werror --disable-warnings %S/test_package_werror 2>&1 | FileCheck %s --check-prefix=BOTH-FLAGS
# RUN: not mojo precompile -Werror %S/test_package_werror 2>&1 | FileCheck %s --check-prefix=ONLY-WERROR

# BOTH-FLAGS: error: assignment to 'foo' was never used
# BOTH-FLAGS-NOT: warning: assignment to 'foo' was never used

# ONLY-WERROR: error: assignment to 'foo' was never used
# ONLY-WERROR-NOT: warning: assignment to 'foo' was never used
