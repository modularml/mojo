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

# RUN: not %mojo-no-debug %s 2>&1 | FileCheck %s

from layout import TileTensor, row_major


def main():
    var src_data = Array[Int32, 4](fill=0)
    var dst_data = Array[Int32, 6](fill=0)

    var src = TileTensor(src_data, row_major[2, 2]())
    var dst = TileTensor(dst_data, row_major[2, 3]())

    # CHECK: TensorEngine.copy_from requires matching total element count
    dst.copy_from(src)
