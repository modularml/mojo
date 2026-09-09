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
# RUN: %mojo %s | FileCheck %s


def unsafe_factorial(next: Int, thusFar: Int) -> Int:
    if next > 1:
        return unsafe_factorial(next - 1, thusFar * next)
    if next <= 1:
        return thusFar
    return 1


def another_unsafe_factorial(next: Int, thusFar: Int) -> Int:
    """This checks to make sure we can properly handle recursive chains."""
    if next > 1:
        return yet_another_unsafe_factorial(next - 1, thusFar * next)
    if next <= 1:
        return thusFar
    return 1


def yet_another_unsafe_factorial(next: Int, thusFar: Int) -> Int:
    return another_unsafe_factorial(next, thusFar)


def main():
    var x = unsafe_factorial(3, 1)
    var y = another_unsafe_factorial(3, 1)
    # CHECK: 6
    print(x)
    # CHECK: 6
    print(y)
