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


def is_positive(a: Int) -> Bool:
    return a > 0


def test_andor[a: Int, b: Int]():
    comptime if is_positive(a) and is_positive(b):
        print(2)
    elif is_positive(a) or is_positive(b):
        print(1)
    else:
        print(0)


def main():
    # CHECK: 2
    test_andor[1, 1]()
    # CHECK: 1
    test_andor[1, -1]()
    # CHECK: 1
    test_andor[-1, 1]()
    # CHECK: 0
    test_andor[-1, -1]()
