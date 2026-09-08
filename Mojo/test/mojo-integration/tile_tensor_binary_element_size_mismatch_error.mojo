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

# TileTensor's in-place binary ops accept an rhs with a different engine,
# but the two engines must agree on logical element size. A
# vectorized rhs (element_width=4) against a scalar destination is rejected
# at compile time.

# RUN: not %mojo %s 2>&1 | FileCheck %s

from layout import TileTensor, row_major


def main():
    var a_data = Array[Float32, 8](fill=0)
    var b_data = Array[Float32, 8](fill=0)
    var a = TileTensor(a_data, row_major[2, 4]())
    var b = TileTensor(b_data, row_major[2, 4]())

    # CHECK: in-place binary ops require operands with the same element size
    a += b.vectorize[1, 4]()
