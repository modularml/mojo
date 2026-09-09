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
# RUN: mkdir -p %t
# RUN: %mojo-build --emit asm \
# RUN:   --target-triple x86_64-unknown-linux-gnu \
# RUN:   --target-cpu sapphirerapids \
# RUN:   --target-accelerator nvidia:sm_90a \
# RUN:   -o %t/out.s %s
# RUN: cat %t/*.ptx > %t/all_kernels.ptx
# RUN: FileCheck %s --check-prefix=CHECK-PTX --input-file=%t/all_kernels.ptx
# RUN: cat %t/all_kernels.ptx %t/out.s > %t/combined.txt
# RUN: FileCheck %s --check-prefix=CHECK-BOTH --input-file=%t/combined.txt

from std.sys.info import simd_width_of

from max.algorithm import elementwise
from max.gpu.host import DeviceContext

from std.utils.coord import Coord


def test_trace_description_elementwise(ctx: DeviceContext) raises:
    var len = 1024

    def func[simd_width: Int, alignment: Int = 1](idx: Coord) {var}:
        pass

    # Verify the PTX entry point has the expected sanitized+hashed name.
    # CHECK-PTX: my_function_with_some_name_r1_w{{[0-9]+}}_b{{[0-9]+}}_gs_False_{{[0-9a-f]+}}
    # Verify the host (x86 ASM) kernel lookup name matches the PTX entry point
    # CHECK-BOTH: [[NAME:my_function_with_some_name_r1_w[0-9]+_b[0-9]+_gs_False_[0-9a-f]+]]
    # CHECK-BOTH: [[NAME]]
    elementwise[
        simd_width=simd_width_of[DType.float32](),
        _trace_description="my_function_with_some_name",
        target="gpu",
    ](func, Coord(len), ctx)


def main() raises:
    with DeviceContext() as ctx:
        test_trace_description_elementwise(ctx)
