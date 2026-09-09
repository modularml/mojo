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

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code


def outer[y: Int]():
    @__parameter
    def param[x: Int](y: SIMD[.float32, y], /):
        pass

    print(
        _compile_code[
            param[y],
            target=get_gpu_target["sm_90a"](),
        ]()
    )


def main():
    # CHECK: .debug_
    outer[2]()
