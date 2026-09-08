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


# CHECK: 170141183460469231731687303715884105728
# CHECK: 340282366920938463463374607431768211455
# CHECK: -170141183460469231731687303715884105728
# CHECK: -1{{$}}
def main():
    var i: UInt128 = 1 << 127
    print(i)
    var j: UInt128 = (1 << 128) - 1
    print(j)
    var k: Int128 = 1 << 127
    print(k)
    var l: Int128 = (1 << 128) - 1
    print(l)
