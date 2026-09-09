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

from std.collections import Set

comptime keys: List[Int] = [1, 2, 3, 7, 5]
comptime num_set = Set(keys)


def main() raises:
    # CHECK: 5
    comptime l = len(num_set)
    print(l)
    # CHECK: True
    comptime contains = num_set.__contains__(7)
    print(contains)
