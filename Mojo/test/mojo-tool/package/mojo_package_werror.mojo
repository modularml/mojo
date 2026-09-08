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

# RUN: mojo precompile -Wno-error %S/test_package_werror 2>&1 | FileCheck %s --check-prefix=WNO-ERROR
# RUN: not mojo precompile -Werror %S/test_package_werror 2>&1 | FileCheck %s --check-prefix=WERROR
# RUN: mojo precompile -Werror -Wno-error %S/test_package_werror 2>&1 | FileCheck %s --check-prefix=WERROR-THEN-WNO
# RUN: not mojo precompile -Wno-error -Werror %S/test_package_werror 2>&1 | FileCheck %s --check-prefix=WNO-THEN-WERROR

# WNO-ERROR: warning: assignment to 'foo' was never used
# WNO-ERROR-NOT: error: assignment to 'foo' was never used

# WERROR: error: assignment to 'foo' was never used
# WERROR-NOT: warning: assignment to 'foo' was never used

# -Werror followed by -Wno-error: warnings remain warnings (last wins)
# WERROR-THEN-WNO: warning: assignment to 'foo' was never used
# WERROR-THEN-WNO-NOT: error: assignment to 'foo' was never used

# -Wno-error followed by -Werror: warnings become errors (last wins)
# WNO-THEN-WERROR: error: assignment to 'foo' was never used
# WNO-THEN-WERROR-NOT: warning: assignment to 'foo' was never used
