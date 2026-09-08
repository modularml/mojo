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
# RUN: mojo precompile %S/../package/test_package -o %t.dir/test_binary_package.mojoc
# RUN: not mojo run -I %t.dir %s 2>&1 | FileCheck %s

# COM: This will import a package with the wrong case, so we expect an error.
# CHECK: error: unable to locate module 'TEST_BINARY_PACKAGE'
from TEST_BINARY_PACKAGE.inner1.myfile import print10


def main() raises:
    # CHECK: 10
    print10()
