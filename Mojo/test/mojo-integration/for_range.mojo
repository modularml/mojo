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

# RUN: %mojo -debug-level full %s | FileCheck %s


def main():
    # CHECK: 0
    # CHECK: 1
    # CHECK: 2
    for x in range(0, 3, 1):
        print(x)

    # CHECK: 9
    # CHECK: 6
    # CHECK: 3
    for y in range(9, 0, -3):
        print(y)

    # CHECK: 42
    for z in range(0, 0, -3):
        print(z)
    else:
        print(42)
