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


def test_fp8_constructor(ctx: DeviceContext) raises:
    def kernel(ptr: MutPointer[Float8_e5m2fnuz, MutAnyOrigin]):
        ptr[] = Float8_e5m2fnuz(42.0)

    # CHECK: v_mov_b32_e32 {{.*}}, 0x55
    # CHECK: store i8 85, ptr %{{.*}}, align 1
    _ = ctx.compile_function[
        kernel,
        dump_llvm=True,
        dump_asm=True,
    ]()


def main() raises:
    with DeviceContext() as ctx:
        test_fp8_constructor(ctx)
