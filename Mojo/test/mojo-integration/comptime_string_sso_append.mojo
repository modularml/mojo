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

# Tests that an in-place `String` append (`+=`) whose result stays within the
# inline (SSO) buffer produces the same value at compile time (`comptime`) and
# at runtime.


def append_small() -> String:
    var s: String = "ab"  # stays inline (SSO)
    s += "cd"  # still inline
    return s


def main():
    # Runtime.
    var rt = append_small()
    # CHECK: abcd
    print(rt)

    # Compile time.
    comptime ct = append_small()
    # CHECK: abcd
    print(ct)
