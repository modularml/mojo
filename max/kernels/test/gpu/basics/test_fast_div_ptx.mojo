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
from layout import IntTuple, Layout, LayoutTensor
from std.python import Python, PythonObject
from std.testing import assert_true

from std.utils.fast_div import FastDiv


def contains_fastdiv_div_sequence(asm: String, width: Int = 32) raises -> Bool:
    var re = Python.import_module("re")
    var w = String(width)
    var fastdiv_pattern = String(
        r"ld\.global\.b" + w + r"\s+[^;]+;\s*"
        r"mov\.b" + w + r"\s+[^;]+;\s*"
        r"mul\.hi\.u" + w + r"\s+[^;]+;\s*"
        r"sub\.s" + w + r"\s+[^;]+;\s*"
        r"shr\.u" + w + r"\s+[^;]+;\s*"
        r"add\.s" + w + r"\s+[^;]+;\s*"
        r"shr\.u" + w + r"\s+[^;]+;\s*"
        r"st\.global\.b" + w + r"\s+[^;]+;"
    )
    var result = re.search(fastdiv_pattern, asm)
    return result is not PythonObject(None)


def contains_power_of_2_sequence(asm: String, width: Int = 32) raises -> Bool:
    var re = Python.import_module("re")
    var w = String(width)
    var shift_pattern = String(
        r"ld\.global\.b" + w + r"\s+[^;]+;\s*"
        r"shr\.u" + w + r"\s+[^;]+;\s*"
        r"st\.global\.b" + w + r"\s+[^;]+;"
    )
    var shift_result = re.search(shift_pattern, asm)
    return shift_result is not PythonObject(None)


def fast_div_kernel[
    dtype: DType,
    layout: Layout,
    divisor: Int,
](input: LayoutTensor[dtype, layout, MutAnyOrigin],):
    comptime fast_div = FastDiv[dtype](divisor)
    var x = input[0]
    var result = rebind[Scalar[fast_div.uint_type]](x) / fast_div
    input[0] = result.cast[dtype]()


def main() raises:
    comptime layout = Layout(IntTuple(1))

    # Test uint32 FastDiv.
    comptime kernel_u32_pow2 = fast_div_kernel[.uint32, layout, 4]
    comptime kernel_u32_div = fast_div_kernel[.uint32, layout, 3]

    var asm = _compile_code[
        kernel_u32_pow2,
        target=get_gpu_target["sm_90"](),
    ]().asm
    assert_true(contains_power_of_2_sequence(asm))

    asm = _compile_code[
        kernel_u32_div,
        target=get_gpu_target["sm_90"](),
    ]().asm
    assert_true(contains_fastdiv_div_sequence(asm))

    # Test uint64 FastDiv.
    comptime kernel_u64_pow2 = fast_div_kernel[.uint64, layout, 4]
    comptime kernel_u64_div = fast_div_kernel[.uint64, layout, 3]

    asm = _compile_code[
        kernel_u64_pow2,
        target=get_gpu_target["sm_90"](),
    ]().asm
    assert_true(contains_power_of_2_sequence(asm, width=64))

    asm = _compile_code[
        kernel_u64_div,
        target=get_gpu_target["sm_90"](),
    ]().asm
    assert_true(contains_fastdiv_div_sequence(asm, width=64))
