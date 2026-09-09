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

# TileTensor's in-place binary ops require same-rank operands to have
# identical shapes; there is no broadcasting between same-rank tensors.

# RUN: not %mojo %s 2>&1 | FileCheck %s

from layout import TileTensor, row_major


def main():
    var a_data = Array[Float32, 4](fill=0)
    var b_data = Array[Float32, 6](fill=0)
    var a = TileTensor(a_data, row_major[2, 2]())
    var b = TileTensor(b_data, row_major[2, 3]())

    # CHECK: requires shape to be the same for tensors of the same rank
    a += b
