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
# This test checks that this Mojo module can import and call a function
# `dont_inline_me` in package 1, in 2 ways:
#
# 1. By calling it directly.
# 2. By calling it via another function in package 2, `dont_inline_me_either`,
#    which in turn calls package 1's `dont_inline_me`.
#
# A test failure could mean that symbols such as `dont_inline_me` are being
# registered with the Mojo JIT twice (once when used by this importing module,
# and again when being used by package 2), using the wrong linkage type. This
# would result in Mojo users encountering "duplicate definition of symbol"
# errors when attempting to use Mojo packages, even when doing so in a way that
# ought to be valid.
#
# ===----------------------------------------------------------------------=== #

# RUN: rm -rf %t.package-link-2 && mkdir -p %t.package-link-2
# RUN: mojo precompile %S/inputs/test_package -o %t.package-link-2/test_package.mojoc
# RUN: mojo precompile -I %t.package-link-2 %S/inputs/test_package_2 -o %t.package-link-2/test_package_2.mojoc
# RUN: %mojo -I %t.package-link-2 %s | FileCheck %s

from test_package.module import dont_inline_me
from test_package_2.module import dont_inline_me_either


def main():
    # CHECK: Don't you dare!
    dont_inline_me()
    # CHECK: Don't you dare!
    dont_inline_me_either()
