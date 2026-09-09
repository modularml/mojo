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


def f0(not_used: String, values: List[List[String]], i: Int) -> List[String]:
    return values[i].copy()


def main():
    # CHECK: [hello]
    # CHECK-NEXT: [world]
    comptime not_used = String("not_used")
    comptime res0 = f0(not_used, [["hello"], ["world"]], 0)
    print(materialize[String(res0)]())
    comptime res1 = f0(not_used, [["hello"], ["world"]], 1)
    print(materialize[String(res1)]())
