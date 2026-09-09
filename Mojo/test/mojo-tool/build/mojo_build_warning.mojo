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

# RUN: %mojo-build-no-werror %s 2>&1 | FileCheck %s --check-prefix=WITH-WARNINGS
# RUN: %mojo-build-no-werror %s --disable-warnings 2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT-WARNINGS


# This warning is issued via the Diags class.
#
# WITH-WARNINGS: warning: Use bar instead
# WITHOUT-WARNINGS-NOT: warning: Use bar instead
@deprecated("Use bar instead")
def deprecatedFn():
    pass


# This warning is issued via the MLIR diagnostic engine
#
# WITH-WARNINGS: warning: assignment to 'x' was never used
# WITHOUT-WARNINGS-NOT: warning: assignment to 'x' was never used
def unusedLocal():
    var x = 42


def main():
    deprecatedFn()
    unusedLocal()
    return
