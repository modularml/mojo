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


from max.gpu.host.compile import _compile_code, get_gpu_target
from std.testing import assert_true

comptime _TargetType = __mlir_type.`!kgen.target`


def kernel(
    src: ImmPointer[Float32, ImmutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
):
    var v = src.load[width=8, alignment=32]()
    dst.store[width=8, alignment=32](v)


def test_kernel_load_32B_width[target: _TargetType]() raises:
    var asm = _compile_code[kernel, target=target]().asm
    assert_true(("v4.b64" in asm) or ("v8.b32" in asm))


def test_kernel_load_16B_width[target: _TargetType]() raises:
    var asm = _compile_code[kernel, target=target]().asm
    assert_true(("v2.b64" in asm) or ("v4.b32" in asm))


def main() raises:
    test_kernel_load_16B_width[get_gpu_target["sm_80"]()]()
    test_kernel_load_16B_width[get_gpu_target["sm_90a"]()]()
    test_kernel_load_32B_width[get_gpu_target["sm_100a"]()]()
