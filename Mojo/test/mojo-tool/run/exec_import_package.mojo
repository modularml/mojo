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
# RUN: mojo run -I %t.dir %s | FileCheck %s

from test_binary_package.inner1.myfile import print10
from test_binary_package.inner1 import myfile_copy


def main() raises:
    # CHECK: 10
    # CHECK: 10
    print10()
    myfile_copy.print10()
