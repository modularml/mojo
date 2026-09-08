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


def print_or(value: Int, condition: Bool):
    @always_inline
    @__parameter
    def do_print(value: Int):
        if condition:
            print(value)
            return
        print("refuse\n")

    do_print(value)


def main():
    # CHECK: 5
    print_or(5, True)
    # CHECK: refuse
    print_or(9, False)
