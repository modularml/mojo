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

from max.gpu.host import DeviceContext
from linalg.utils_gpu import select_config


# CHECK-LABEL: test_matmul_selection
def test_matmul_selection() raises:
    print("=== test_matmul_selection")
    with DeviceContext() as ctx:
        var config0 = select_config[.bfloat16, .bfloat16, .bfloat16](
            1024, 1024, 1024, ctx
        )
        # CHECK: kernel_bfloat16_bfloat16_128x128_4_NN
        # CHECK: (128, 128, 32)
        # CHECK: (64, 64, 32)
        # CHECK: 4
        # CHECK: 1
        print(config0)
        print(config0.block_tile_shape)
        print(config0.warp_tile_shape)
        print(config0.num_pipeline_stages)
        print(config0.num_k_partitions)

        var config1 = select_config[
            .bfloat16, .bfloat16, .bfloat16, transpose_b=True
        ](528, 3072, 3072, ctx)
        # CHECK: kernel_bfloat16_bfloat16_256x64_4_NT
        # CHECK: (64, 256, 32)
        # CHECK: (64, 64, 32)
        # CHECK: 4
        # CHECK: 1
        print(config1)
        print(config1.block_tile_shape)
        print(config1.warp_tile_shape)
        print(config1.num_pipeline_stages)
        print(config1.num_k_partitions)

        var config2 = select_config[
            .bfloat16, .bfloat16, .bfloat16, transpose_b=True
        ](482, 4096, 14336, ctx)
        # CHECK: kernel_bfloat16_bfloat16_256x128_3_k3_NT
        # CHECK: (128, 256, 64)
        # CHECK: (64, 64, 64)
        # CHECK: 3
        # CHECK: 3
        # CHECK: 3948544
        print(config2)
        print(config2.block_tile_shape)
        print(config2.warp_tile_shape)
        print(config2.num_pipeline_stages)
        print(config2.num_k_partitions)
        print(config2.work_space_size(482, 4096))


def main() raises:
    test_matmul_selection()
