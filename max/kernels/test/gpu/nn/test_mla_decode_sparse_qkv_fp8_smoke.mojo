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

"""Compile-only smoke test for MLA_SM100_Decode_Sparse_QKV_FP8.

Importing the struct forces Mojo to compile the module and catches ICEs or
type errors in the struct's comptime constants, SMEM layout, and helper
signatures. Numerical correctness is covered by
test_mla_decode_sparse_qkv_fp8.mojo.
"""

from std.sys import has_nvidia_gpu_accelerator
from max.gpu.host import DeviceContext

from nn.attention.gpu.nvidia.sm100.mla_decode_sparse_qkv_fp8 import (
    MLA_SM100_Decode_Sparse_QKV_FP8,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    MLA_SM100_Decode_Config,
)


def main() raises:
    comptime if not has_nvidia_gpu_accelerator():
        return

    _ = DeviceContext()
    print("MLA_SM100_Decode_Sparse_QKV_FP8 struct compiled.")
