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

from std.compile import compile_info
from max.gpu.host import get_gpu_target
from layout import Layout, LayoutTensor, UNKNOWN_VALUE


# CHECK-LABEL: test_no_alloca_fill
def test_no_alloca_fill():
    print("== test_no_alloca_fill")

    def layout_tensor_kernel(
        output: LayoutTensor[
            .float32,
            Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
            MutAnyOrigin,
        ],
        i: Int,
        j: Int,
    ):
        var reg_tile = (
            LayoutTensor[.float32, Layout.row_major(4, 4), MutAnyOrigin]
            .stack_allocation()
            .fill(0)
        )

        output.tile[4, 4](i, j).copy_from(reg_tile)

    # CHECK-NOT: alloca float, i64 16, align 4
    print(
        compile_info[
            layout_tensor_kernel,
            emission_kind="llvm",
            target=get_gpu_target(),
        ]()
    )


def main():
    test_no_alloca_fill()
