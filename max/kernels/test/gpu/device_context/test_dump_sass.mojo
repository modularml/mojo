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

from std.pathlib import Path
from std.sys._assembly import inlined_assembly

from max.gpu.host import DeviceContext

comptime ptxas_path = Path("/usr/local/cuda/bin/ptxas")
comptime nvdisasm_path = Path("/usr/local/cuda/bin/nvdisasm")


def test__dump_sass() raises:
    def kernel_inlined_assembly():
        inlined_assembly["nanosleep.u32 $0;", NoneType, constraints="r"](
            UInt32(100)
        )

    with DeviceContext() as ctx:
        _ = ctx.compile_function[kernel_inlined_assembly, _dump_sass=True]()


def main() raises:
    if ptxas_path.exists() and nvdisasm_path.exists():
        test__dump_sass()
