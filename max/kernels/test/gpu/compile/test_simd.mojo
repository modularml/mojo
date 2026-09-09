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


from std.sys.info import _accelerator_arch

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from max.gpu.host.info import GPUInfo, _is_sm10x_gpu
from std.testing import assert_true


def test_operation[
    dtype: DType,
    target_arch: StaticString,
    op_fn: def[width: SIMDLength](
        x: SIMD[dtype, width], y: type_of(x)
    ) thin raises -> type_of(x),
    op_name: StaticString,
]() raises:
    var scalar: String
    var pairwise: String
    var suffix: String

    # sm_80 does not support trivial add/sub/mul bfloat16 operations, but
    # these can be implemented using the FMA instruction. Verify that the
    # backend is using FMA and not falling back to widening the inputs to
    # float32.
    # sm_90 and later has wider support for bfloat16 operations.
    # sm_100 has support for f32x2 add/sub/mul/fma.
    var prefix: String

    comptime if target_arch == "sm_80" and dtype == .bfloat16:
        prefix = "fma.rn"
    else:
        prefix = String(op_name)

    comptime if dtype == .float16:
        suffix = ".f16"
    elif dtype == .float32:
        suffix = ".f32"
    else:
        suffix = ".bf16"

    scalar = prefix + suffix
    pairwise = scalar + "x2 "

    comptime target = get_gpu_target[target_arch]()
    assert_true(scalar in _compile_code[op_fn[width=1], target=target]())
    assert_true(pairwise in _compile_code[op_fn[width=2], target=target]())
    assert_true(pairwise in _compile_code[op_fn[width=8], target=target]())


def test_add[dtype: DType, target_arch: StaticString]() raises:
    def add[
        width: SIMDLength
    ](x: SIMD[dtype, width], y: type_of(x)) -> type_of(x):
        return x + y

    test_operation[dtype, target_arch, add, "add"]()


def test_sub[dtype: DType, target_arch: StaticString]() raises:
    def sub[
        width: SIMDLength
    ](x: SIMD[dtype, width], y: type_of(x)) -> type_of(x):
        return x - y

    test_operation[dtype, target_arch, sub, "sub"]()


def test_mul[dtype: DType, target_arch: StaticString]() raises:
    def mul[
        width: SIMDLength
    ](x: SIMD[dtype, width], y: type_of(x)) -> type_of(x):
        return x * y

    test_operation[dtype, target_arch, mul, "mul"]()


def test_half_float_instruction_selection() raises:
    def test_operations[dtype: DType, target_arch: StaticString]() raises:
        test_add[dtype, target_arch]()
        test_sub[dtype, target_arch]()
        test_mul[dtype, target_arch]()

    def test_types[dtype: DType]() raises:
        test_operations[dtype, "sm_80"]()
        test_operations[dtype, "sm_90"]()

    test_types[.bfloat16]()
    test_types[.float16]()


def test_fma[dtype: DType]() raises:
    def fma[
        width: SIMDLength
    ](x: SIMD[dtype, width], y: type_of(x), z: type_of(x)) -> type_of(x):
        return x * y + z

    def fma_manual[
        width: SIMDLength
    ](x: SIMD[dtype, width], y: type_of(x), z: type_of(x)) -> type_of(x):
        return x.fma(y, z)

    comptime if dtype == .bfloat16:
        assert_true("fma.rn.bf16 " in _compile_code[fma[width=1]]())
        assert_true("fma.rn.bf16x2 " in _compile_code[fma[width=2]]())
        assert_true("fma.rn.bf16x2 " in _compile_code[fma[width=8]]())
    elif dtype == .float32:
        assert_true("fma.rn.f32 " in _compile_code[fma_manual[width=1]]())
        assert_true("fma.rn.f32x2 " in _compile_code[fma_manual[width=2]]())
        assert_true("fma.rn.f32x2 " in _compile_code[fma_manual[width=8]]())
    else:
        assert_true("fma.rn.f16 " in _compile_code[fma[width=1]]())
        assert_true("fma.rn.f16x2 " in _compile_code[fma[width=2]]())
        assert_true("fma.rn.f16x2 " in _compile_code[fma[width=8]]())


def test_cast() raises:
    def cast[
        src_type: DType, dst_type: DType, width: SIMDLength
    ](src: SIMD[src_type, width]) -> SIMD[dst_type, width]:
        return src.cast[dst_type]()

    assert_true(
        "cvt.rn.f16x2.f32"
        in _compile_code[
            cast[src_type=DType.float32, dst_type=DType.float16, width=4]
        ]()
    )
    assert_true(
        "cvt.rn.bf16x2.f32"
        in _compile_code[
            cast[src_type=DType.float32, dst_type=DType.bfloat16, width=4]
        ]()
    )
    assert_true(
        "cvt.f32.bf16"
        in _compile_code[
            cast[src_type=DType.bfloat16, dst_type=DType.float32, width=1]
        ]()
    )
    assert_true(
        "cvt.f32.bf16"
        in _compile_code[
            cast[src_type=DType.bfloat16, dst_type=DType.float32, width=4]
        ]()
    )


def main() raises:
    test_half_float_instruction_selection()

    test_fma[.bfloat16]()
    test_fma[.float16]()

    test_cast()

    comptime device = GPUInfo.from_name[_accelerator_arch()]()

    comptime if _is_sm10x_gpu(device):
        test_add[.float32, "sm_100"]()
        test_mul[.float32, "sm_100"]()
        test_fma[.float32]()
