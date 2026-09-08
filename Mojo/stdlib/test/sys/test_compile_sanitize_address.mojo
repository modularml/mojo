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
# RUN: %mojo-build --sanitize address %s -o %t 2>&1
# RUN: %t | FileCheck %s --check-prefix=CHECK-ON
# RUN: %mojo-build  %s -o %t 2>&1
# RUN: %t | FileCheck %s --check-prefix=CHECK-OFF

from std.sys.compile import SanitizeAddress


def main() raises:
    print(SanitizeAddress)
    # CHECK-ON: True
    # CHECK-OFF: False
