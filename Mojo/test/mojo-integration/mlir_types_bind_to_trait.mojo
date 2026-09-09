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
# RUN: %mojo -debug-level full -O0 %s 2 3 | FileCheck %s

comptime Testdef = def(x: Int) thin raises -> Tuple[Bool, Int]


@no_inline
def printIt[T: ImplicitlyCopyable & Deinitable]():
    if T.__del__is_trivial:
        print("deinit trivial")
    if T.__move_ctor_is_trivial:
        print("move trivial")
    if T.__copy_ctor_is_trivial:
        print("copy trivial")


def main() raises:
    # CHECK: deinit trivial
    # CHECK: move trivial
    # CHECK: copy trivial
    printIt[Testdef]()
