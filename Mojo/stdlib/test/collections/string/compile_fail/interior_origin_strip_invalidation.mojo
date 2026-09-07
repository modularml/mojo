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

# RUN: not %mojo %s 2>&1 | FileCheck %s

# Derived views such as `strip()` also carry an interior origin, so they are
# invalidated when the string is mutated.


def main():
    var s = String("  hello world  ")
    var trimmed = s.strip()
    s += "some more text to force a reallocation"
    # CHECK: use of invalidated interior reference 's["bytes"]'
    print(trimmed)
