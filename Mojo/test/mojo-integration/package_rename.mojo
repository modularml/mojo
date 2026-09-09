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
# Tests that the `-o` output file name of the `.mojoc` determines the name by
# which the package can be imported -- the original directory name doesn't
# matter.
#
# ===----------------------------------------------------------------------=== #

# RUN: rm -rf %t.package-rename && mkdir -p %t.package-rename
# RUN: mojo precompile %S/inputs/test_package -o %t.package-rename/renamed-package.mojoc
# RUN: %mojo -I %t.package-rename %s | FileCheck %s

from `renamed-package`.module import identity


def main():
    # CHECK: hi
    print(identity("hi"))
