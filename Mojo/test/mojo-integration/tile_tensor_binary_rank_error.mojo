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

# Broadcasting in TileTensor's in-place binary ops only supports a lower-rank
# rhs against a rank-2 destination; a rank-3 destination is rejected.

# RUN: not %mojo %s 2>&1 | FileCheck %s

from layout import TileTensor, row_major


def main():
    var a_data = Array[Float32, 24](fill=0)
    var b_data = Array[Float32, 2](fill=0)
    var a = TileTensor(a_data, row_major[2, 3, 4]())
    var b = TileTensor(b_data, row_major[2]())

    # CHECK: Only supports rank-2 tensor, or same rank
    a += b
