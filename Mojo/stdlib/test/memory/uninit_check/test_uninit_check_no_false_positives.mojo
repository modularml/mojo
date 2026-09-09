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
# Tests that the poison check does NOT produce false positives for legitimate
# values, including NaN/Inf bit patterns, near-poison integers, and masked-off
# lanes.

from std.memory import Pointer, alloc
from std.sys.intrinsics import masked_load
from std.testing import assert_true


def test_normal_float32() raises:
    """Loading a properly initialized Float32 should not trigger abort."""
    var value = Float32(42.0)
    var val = Pointer(to=value).unsafe_load()
    assert_true(val == 42.0)


def test_qnan_not_flagged():
    """Canonical qNaN (0x7FC00000) is not the poison pattern; loading must
    not trigger abort. The poison pattern is intentionally non-NaN so the
    uninit-read check coexists with the nan-check pass."""
    var value = UInt32(0x7FC00000)
    var ptr = Pointer(to=value).unsafe_bitcast[Float32]()
    _ = ptr.unsafe_load()


def test_snan_not_flagged():
    """A signaling NaN (0x7F800001) is not the poison pattern."""
    var value = UInt32(0x7F800001)
    var ptr = Pointer(to=value).unsafe_bitcast[Float32]()
    _ = ptr.unsafe_load()


def test_inf_not_flagged():
    """Positive infinity (0x7F800000) is not the poison pattern."""
    var value = UInt32(0x7F800000)
    var ptr = Pointer(to=value).unsafe_bitcast[Float32]()
    _ = ptr.unsafe_load()


def test_integer_not_checked() raises:
    """Integer types should not be checked for poison patterns."""
    var value = UInt32(0xFFFFFFFF)
    var val = Pointer(to=value).unsafe_load()
    assert_true(val == 0xFFFFFFFF)


def test_near_poison_float():
    """A float value close to but not equal to poison (FLT_MAX - 1 ulp =
    0x7F7FFFFE) should not trigger abort."""
    var value = UInt32(0x7F7FFFFE)
    var ptr = Pointer(to=value).unsafe_bitcast[Float32]()
    _ = ptr.unsafe_load()


def test_masked_load_poison_in_masked_off_lane() raises:
    """Poison in a masked-off lane (passthrough) should not trigger abort."""
    # masked_load requires a contiguous buffer, so alloc is needed here.
    var allocation = alloc[Float32]({count = 4}).into_managed()
    var ptr = allocation.unsafe_ptr()
    ptr.unsafe_store(0, Float32(1.0))
    ptr.unsafe_store(1, Float32(2.0))
    ptr.unsafe_store(2, Float32(3.0))
    ptr.unsafe_store(3, Float32(4.0))

    # Poison lanes 2 and 3 with the debug allocator poison pattern
    # (FLT_MAX bits = 0x7F7FFFFF). Masked-off lanes must not trigger.
    ptr.unsafe_offset(2).unsafe_bitcast[UInt32]().unsafe_store(
        UInt32(0x7F7FFFFF)
    )
    ptr.unsafe_offset(3).unsafe_bitcast[UInt32]().unsafe_store(
        UInt32(0x7F7FFFFF)
    )

    # mask=False for lanes 2,3 means those lanes use passthrough, not memory.
    var mask = SIMD[.bool, 4](True, True, False, False)
    var passthrough = SIMD[.float32, 4](0)
    var val = masked_load(ptr, mask, passthrough)

    assert_true(val[0] == 1.0)
    assert_true(val[1] == 2.0)


def test_normal_float64() raises:
    """Loading a properly initialized Float64 should not trigger abort."""
    var value = Float64(3.14159)
    var val = Pointer(to=value).unsafe_load()
    assert_true(val == 3.14159)


def test_normal_float16():
    """Loading a properly initialized Float16 should not trigger abort."""
    var value = Float16(1.5)
    _ = Pointer(to=value).unsafe_load()


def test_normal_bfloat16():
    """Loading a properly initialized BFloat16 should not trigger abort."""
    var value = BFloat16(1.5)
    _ = Pointer(to=value).unsafe_load()


def test_fp8_e4m3fn_poison_pattern_not_flagged():
    """The poison check excludes `float8_e4m3fn` because its `max_finite`
    sentinel (0x7E = 448.0) collides with legitimate saturate-to-max values
    in narrow-fp8 quantization. Constructing a value with the would-be-
    poison bit pattern must not abort."""
    _ = Float8_e4m3fn(from_bits=UInt8(0x7E))


def test_fp8_e5m2_poison_pattern_not_flagged():
    """The poison check excludes `float8_e5m2` for the same reason as
    `float8_e4m3fn`. Constructing a value with the would-be-poison bit
    pattern (0x7B = 57344.0) must not abort."""
    _ = Float8_e5m2(from_bits=UInt8(0x7B))


def main() raises:
    test_normal_float32()
    test_qnan_not_flagged()
    test_snan_not_flagged()
    test_inf_not_flagged()
    test_integer_not_checked()
    test_near_poison_float()
    test_masked_load_poison_in_masked_off_lane()
    test_normal_float64()
    test_normal_float16()
    test_normal_bfloat16()
    test_fp8_e4m3fn_poison_pattern_not_flagged()
    test_fp8_e5m2_poison_pattern_not_flagged()
