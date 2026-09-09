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
"""Test the max.graph dtype operations."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from max.dtype import DType, finfo

int_dtype = st.sampled_from(
    [
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
    ]
)

float_dtype = st.sampled_from(
    [
        DType.float4_e2m1fn,
        DType.float6_e2m3fn,
        DType.float6_e3m2fn,
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
        DType.float8_e5m2,
        DType.float8_e5m2fnuz,
        DType.bfloat16,
        DType.float16,
        DType.float32,
        DType.float64,
    ]
)


def test_roundtrip() -> None:
    for dtype in DType:
        assert isinstance(dtype, DType)
        assert DType(dtype._mlir) == dtype


@given(dtype=int_dtype | float_dtype)
def test_numpy_roundtrip(dtype: DType) -> None:
    # There is no f4 / f6 / float8 / bf16 in numpy, so we cannot roundtrip
    # f4 / f6 / float8 / bf16
    if dtype in [
        DType.float4_e2m1fn,
        DType.float6_e2m3fn,
        DType.float6_e3m2fn,
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
        DType.float8_e5m2,
        DType.float8_e5m2fnuz,
        DType.bfloat16,
    ]:
        return
    np_dtype = dtype.to_numpy()
    assert dtype == DType.from_numpy(np_dtype)


@given(int_dtype=int_dtype, float_dtype=float_dtype)
def test_is_integral(int_dtype: DType, float_dtype: DType) -> None:
    assert int_dtype.is_integral()
    assert not float_dtype.is_integral()


@given(int_dtype=int_dtype, float_dtype=float_dtype)
def test_is_float(int_dtype: DType, float_dtype: DType) -> None:
    assert not int_dtype.is_float()
    assert float_dtype.is_float()


def test_dtype_alignment() -> None:
    assert DType.bool.align == 1
    assert DType.int8.align == 1
    assert DType.int16.align == 2
    assert DType.int32.align == 4
    assert DType.int64.align == 8
    assert DType.uint8.align == 1
    assert DType.uint16.align == 2
    assert DType.uint32.align == 4
    assert DType.uint64.align == 8
    assert DType.float16.align == 2
    assert DType.float32.align == 4
    assert DType.float64.align == 8
    assert DType.bfloat16.align == 2
    assert DType.float4_e2m1fn.align == 1
    assert DType.float6_e2m3fn.align == 1
    assert DType.float6_e3m2fn.align == 1
    assert DType.float8_e4m3fn.align == 1
    assert DType.float8_e4m3fnuz.align == 1
    assert DType.float8_e5m2.align == 1
    assert DType.float8_e5m2fnuz.align == 1


# ── finfo tests ──────────────────────────────────────────────────────────


class TestFinfo:
    """Tests for the finfo class (torch.finfo-compatible)."""

    @pytest.mark.parametrize(
        "dtype", [DType.float16, DType.float32, DType.float64]
    )
    def test_standard_floats_match_numpy(self, dtype: DType) -> None:
        info = finfo(dtype)
        np_info = np.finfo(dtype.to_numpy())
        assert info.bits == np_info.bits
        assert info.eps == float(np_info.eps)
        assert info.max == float(np_info.max)
        assert info.min == float(np_info.min)
        assert info.tiny == float(np_info.tiny)

    def test_bfloat16(self) -> None:
        info = finfo(DType.bfloat16)
        assert info.bits == 16
        assert info.eps == 2**-7
        assert info.max == (2 - 2**-7) * 2**127
        assert info.min == -(2 - 2**-7) * 2**127
        assert info.tiny == 2**-126

    def test_float8_e4m3fn(self) -> None:
        info = finfo(DType.float8_e4m3fn)
        assert info.bits == 8
        assert info.eps == 0.125
        assert info.max == 448.0
        assert info.min == -448.0
        assert info.tiny == 2**-6

    def test_float8_e4m3fnuz(self) -> None:
        info = finfo(DType.float8_e4m3fnuz)
        assert info.bits == 8
        assert info.eps == 0.125
        assert info.max == 240.0
        assert info.min == -240.0
        assert info.tiny == 2**-7

    def test_float8_e5m2(self) -> None:
        info = finfo(DType.float8_e5m2)
        assert info.bits == 8
        assert info.eps == 0.25
        assert info.max == 57344.0
        assert info.min == -57344.0
        assert info.tiny == 2**-14

    def test_float8_e5m2fnuz(self) -> None:
        info = finfo(DType.float8_e5m2fnuz)
        assert info.bits == 8
        assert info.eps == 0.25
        assert info.max == 57344.0
        assert info.min == -57344.0
        assert info.tiny == 2**-15

    def test_float8_e8m0fnu(self) -> None:
        info = finfo(DType.float8_e8m0fnu)
        assert info.bits == 8
        assert info.eps == 1.0
        assert info.max == float(2**127)
        assert info.min == float(2**-127)
        assert info.tiny == float(2**-127)

    def test_float4_e2m1fn(self) -> None:
        info = finfo(DType.float4_e2m1fn)
        assert info.bits == 4
        assert info.eps == 0.5
        assert info.max == 6.0
        assert info.min == -6.0
        assert info.tiny == 1.0

    def test_float6_e2m3fn(self) -> None:
        info = finfo(DType.float6_e2m3fn)
        assert info.bits == 6
        assert info.eps == 0.125
        assert info.max == 7.5
        assert info.min == -7.5
        assert info.tiny == 1.0

    def test_float6_e3m2fn(self) -> None:
        info = finfo(DType.float6_e3m2fn)
        assert info.bits == 6
        assert info.eps == 0.25
        assert info.max == 28.0
        assert info.min == -28.0
        assert info.tiny == 0.25

    @pytest.mark.parametrize("dtype", [DType.int32, DType.bool, DType.uint8])
    def test_non_float_raises_type_error(self, dtype: DType) -> None:
        with pytest.raises(
            TypeError, match="finfo only supports floating-point"
        ):
            finfo(dtype)

    def test_smallest_normal_aliases_tiny(self) -> None:
        for dt in [DType.float32, DType.bfloat16, DType.float8_e4m3fn]:
            info = finfo(dt)
            assert info.smallest_normal == info.tiny

    def test_dtype_attribute_returns_dtype_enum(self) -> None:
        info = finfo(DType.float32)
        assert info.dtype is DType.float32
        assert isinstance(info.dtype, DType)

    def test_monkey_patched_static_access(self) -> None:
        info = DType.finfo(DType.bfloat16)  # type: ignore[attr-defined]
        assert info.bits == 16

    def test_standalone_import(self) -> None:
        # Verify the import path works (already imported at top of file).
        assert finfo is not None
        info = finfo(DType.float32)
        assert info.bits == 32

    def test_repr(self) -> None:
        info = finfo(DType.float32)
        r = repr(info)
        assert "finfo(" in r
        assert "float32" in r
