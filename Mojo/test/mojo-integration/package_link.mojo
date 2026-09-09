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
#
# This test checks that this Mojo module can import and call a function in
# package 2, which in turn calls a function in package 1. (Each function is
# decorated so as not to be inlined, ensuring that its body remains in its
# respective package.)
#
# A test failure could mean that symbols in transitive package dependencies are
# not being added to the Mojo JIT properly, so Mojo users may experience
# inscrutable "failed to materialize symbols" errors when attempting to use Mojo
# packages, even when doing so in a way that ought to be valid.
#
# ===----------------------------------------------------------------------=== #

# RUN: rm -rf %t.package-link && mkdir -p %t.package-link
# RUN: mojo precompile %S/inputs/test_package -o %t.package-link/test_package.mojoc
# RUN: mojo precompile -I %t.package-link %S/inputs/test_package_2 -o %t.package-link/test_package_2.mojoc
# RUN: %mojo -I %t.package-link %s | FileCheck %s

from test_package_2.module import dont_inline_me_either


def main():
    # CHECK: Don't you dare!
    dont_inline_me_either()
