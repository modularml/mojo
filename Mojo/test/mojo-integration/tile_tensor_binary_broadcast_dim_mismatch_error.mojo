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

# Rank-1 -> rank-2 broadcasting in TileTensor's in-place binary ops requires
# the rank-1 operand's dimension to match the tensor's first dimension.

# RUN: not %mojo %s 2>&1 | FileCheck %s

from layout import TileTensor, row_major


def main():
    var a_data = Array[Float32, 8](fill=0)
    var b_data = Array[Float32, 3](fill=0)
    var a = TileTensor(a_data, row_major[2, 4]())
    var b = TileTensor(b_data, row_major[3]())

    # CHECK: 1d tensor operand must have a dim that matches the tensors
    a += b
