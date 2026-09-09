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

# RUN: mojo precompile --experimental-fixit %S/test_package_fixit | FileCheck %s --check-prefix=AUTO-FIXIT
# RUN: mojo precompile --experimental-fixit %S/test_package_fixit | FileCheck %s --check-prefix=NO-FIXIT

# AUTO-FIXIT: Fixits applied.
# NO-FIXIT: No fixits to apply.

# After applying the fixits, the build should succeed.
# RUN: mojo precompile %S/test_package_fixit -o test_package_fixit.mojoc

# The `grep` is used to remove the `CHECK` lines from the output so FileCheck
# doesn't match on its own directives.
# RUN: cat %S/test_package_fixit/__init__.mojo | grep -v "# CHECK" | FileCheck %S/test_package_fixit/__init__.mojo
# RUN: cat %S/test_package_fixit/old_impl.mojo | grep -v "# CHECK" | FileCheck %S/test_package_fixit/old_impl.mojo
