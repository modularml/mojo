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

# RUN: mkdir -p %t.dir
# RUN: mojo precompile %S/inputs/link_dep/dotted.dep -o %t.dir/dotted.dep.mojoc
# RUN: mojo precompile -I %t.dir %S/inputs/link_dep/user_pkg -o %t.dir/user_pkg.mojoc
# RUN: mojo run -I %t.dir %s | FileCheck %s

# A precompiled package records its precompiled imports as link dependencies
# by their package *names*. A dependency on a dotted-named package must
# round-trip as a single name, not be re-split as a module path when the
# dependency is resolved from bytecode.

from user_pkg import user_value


def main() raises:
    # CHECK: 42
    print(user_value())
