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

# RUN: %parse-mojo-isolated %s | FileCheck %s


struct PCall(Movable where False):
    def __init__(out self):
        pass

    def __call__[x: Int](ref self, y: Int) -> Int:
        return x + y


# CHECK: lit.fn @"main()"
def main():
    var pc = PCall()
    # CHECK: lit.call {{.*}}::@PCall::@"__call__[::SIMD[DType.int, 1],{{.*}}]({{.*}}::PCall%,::SIMD[DType.int, 1])"
    _ = pc[1](2)
