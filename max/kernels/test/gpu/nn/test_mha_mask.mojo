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

from std.sys import has_amd_gpu_accelerator, has_nvidia_gpu_accelerator
from std.sys.info import CompilationTarget

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from nn.attention.mha_mask import (
    AndMask,
    CausalMask,
    MASK_VALUE,
    MaskName,
    MHAMask,
    NullMask,
    SlidingWindowCausalMask,
    SlidingWindowNonCausalMask,
    TileMaskStatus,
)
from nn.attention.mha_utils import dispatch_mask
from std.testing import assert_equal, assert_true

from std.utils.index import Index, IndexList


def test_causal_mask() raises:
    # Kernels instantiate `mask()` at the QK accumulator type, which is always
    # float32. MASK_VALUE does not fit a narrower dtype.
    comptime type = DType.float32

    print("test_causal_mask")
    var mask = CausalMask()

    # Check mask value.
    # TODO(KERN-782): should be -inf but softmax saturates with NaNs.
    comptime mask_val = Scalar[type](MASK_VALUE)
    var masked_vec = mask.mask(Index(0, 0, 4, 3), SIMD[type, 4](0, 1, 2, 3))
    assert_equal(masked_vec, SIMD[type, 4](0, 1, mask_val, mask_val))

    masked_vec = mask.mask(Index(0, 0, 4, 0), SIMD[type, 4](0, 1, 2, 3))
    assert_equal(masked_vec, SIMD[type, 4](0, 1, 2, 3))

    masked_vec = mask.mask(Index(0, 0, 1, 6), SIMD[type, 4](0, 1, 2, 3))
    assert_equal(masked_vec, SIMD[type, 4](mask_val))

    # Check tile status.
    assert_true(
        mask.status(UInt32(0), Index(4, 4), Index(4, 4))
        == TileMaskStatus.PARTIAL_MASK
    )
    assert_true(
        mask.status(UInt32(0), Index(0, 2), Index(2, 2))
        == TileMaskStatus.FULL_MASK
    )
    assert_true(
        mask.status(UInt32(0), Index(2, 0), Index(2, 2))
        == TileMaskStatus.NO_MASK
    )
    assert_true(
        mask.status(UInt32(0), Index(2, 1), Index(2, 2))
        == TileMaskStatus.NO_MASK
    )
    assert_true(
        mask.status(UInt32(0), Index(1, 5), Index(2, 2))
        == TileMaskStatus.FULL_MASK
    )
    assert_true(
        mask.status(UInt32(0), Index(64, 0), Index(64, 128))
        == TileMaskStatus.PARTIAL_MASK
    )
    assert_true(
        mask.status(UInt32(0), Index(64, 128), Index(64, 128))
        == TileMaskStatus.FULL_MASK
    )
    assert_true(
        mask.status(UInt32(0), Index(64, 256), Index(64, 128))
        == TileMaskStatus.FULL_MASK
    )
    assert_true(
        mask.status(UInt32(0), Index(64, 384), Index(64, 128))
        == TileMaskStatus.FULL_MASK
    )


def test_causal_mask_asm() raises:
    """Verify mask comparison is not in 64 bits."""

    print("== test_causal_mask_asm")

    def kernel(
        q_idx: UInt32, k_idx: UInt32, x: MutPointer[Float32, MutAnyOrigin]
    ):
        var mask = CausalMask()
        var vec = mask.mask(
            IndexList[4, element_type=.uint32](0, 0, Int(q_idx), Int(k_idx)),
            SIMD[.float32, 4](0),
        )
        if (
            mask.status(
                UInt32(0),
                Index[dtype=DType.uint32](q_idx, k_idx),
                Index[dtype=DType.uint32](4, 5),
            )
            == TileMaskStatus.PARTIAL_MASK
        ):
            x[0] = vec[3]

        x[0] = vec[2]

    var asm = _compile_code[kernel, target=get_gpu_target()]().asm
    print(asm)

    comptime if has_nvidia_gpu_accelerator():
        assert_true("setp.lt.u64" not in asm)
        assert_true("setp.lt.s64" not in asm)
    elif has_amd_gpu_accelerator():
        assert_true("s_cselect_b64" in asm)
        assert_true("v_cndmask_b32_e64" in asm)
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name(),
        ]()


def test_and_mask() raises:
    comptime type = DType.float32
    comptime mask_val = Scalar[type](MASK_VALUE)

    print("test_and_mask")
    # A position is masked only where both operands mask it, so and-ing a
    # causal mask with a null mask leaves every position visible.
    var mask = AndMask[CausalMask(), NullMask()]()

    var masked_vec = mask.mask(Index(0, 0, 4, 3), SIMD[type, 4](0, 1, 2, 3))
    assert_equal(masked_vec, SIMD[type, 4](0, 1, 2, 3))

    masked_vec = mask.mask(Index(0, 0, 4, 0), SIMD[type, 4](0, 1, 2, 3))
    assert_equal(masked_vec, SIMD[type, 4](0, 1, 2, 3))

    masked_vec = mask.mask(Index(0, 0, 1, 6), SIMD[type, 4](0, 1, 2, 3))
    assert_equal(masked_vec, SIMD[type, 4](0, 1, 2, 3))

    # Check tile status.
    assert_true(
        mask.status(UInt32(0), Index(4, 4), Index(4, 4))
        == TileMaskStatus.NO_MASK
    )
    assert_true(
        mask.status(UInt32(0), Index(0, 2), Index(2, 2))
        == TileMaskStatus.NO_MASK
    )
    assert_true(
        mask.status(UInt32(0), Index(2, 0), Index(2, 2))
        == TileMaskStatus.NO_MASK
    )

    var mask2 = AndMask[CausalMask(), CausalMask()]()

    # And-ing a causal mask with itself stays causal.
    assert_equal(
        mask2.mask(Index(0, 0, 4, 3), SIMD[type, 4](0, 1, 2, 3)),
        SIMD[type, 4](0, 1, mask_val, mask_val),
    )

    assert_true(
        mask2.status(UInt32(0), Index(4, 4), Index(4, 4))
        == TileMaskStatus.PARTIAL_MASK
    )
    assert_true(
        mask2.status(UInt32(0), Index(64, 384), Index(64, 128))
        == TileMaskStatus.FULL_MASK,
        msg=String(
            t"lhs ="
            t" {mask2.status(UInt32(0), Index(0, 0), Index(0, 0))} rhs ="
            t" {TileMaskStatus.FULL_MASK}"
        ),
    )


def test_sliding_window_causal_mask() raises:
    print("test_sliding_window_causal_mask")

    comptime mask = SlidingWindowCausalMask[3]()

    @always_inline
    def check_status(
        offset: IndexList[2, ...],
        size: type_of(offset),
        expected: TileMaskStatus,
    ) raises:
        var status = mask.status(UInt32(0), offset, size)
        assert_equal(
            status,
            expected,
            msg=String(t"  {offset}, {size} > {status} (expected: {expected})"),
        )

        # K > 0 1 2 3 4 5 6 7 8
        # Q v x-----------------x
        # 0 | 1 0 0 0 0 0 0 0 0
        # 1 | 1 1 0 0 0 0 0 0 0
        # 2 | 1 1 1 0 0 0 0 0 0
        # 3 | 0 1 1 1 0 0 0 0 0
        # 4 | 0 0 1 1 1 0 0 0 0
        # 5 | 0 0 0 1 1 1 0 0 0
        # 6 | 0 0 0 0 1 1 1 0 0
        # 7 | 0 0 0 0 0 1 1 1 0
        # 8 | 0 0 0 0 0 0 1 1 1

    check_status(Index(0, 0), Index(4, 4), TileMaskStatus.PARTIAL_MASK)
    check_status(Index(4, 0), Index(4, 4), TileMaskStatus.PARTIAL_MASK)

    check_status(Index(2, 1), Index(2, 2), TileMaskStatus.NO_MASK)
    check_status(Index(3, 1), Index(1, 3), TileMaskStatus.NO_MASK)
    check_status(Index(3, 3), Index(3, 1), TileMaskStatus.NO_MASK)

    check_status(Index(0, 4), Index(4, 4), TileMaskStatus.FULL_MASK)
    check_status(Index(4, 0), Index(4, 2), TileMaskStatus.FULL_MASK)
    check_status(Index(1, 4), Index(3, 2), TileMaskStatus.FULL_MASK)


def test_sliding_window_causal_mask_asm() raises:
    """Verify mask comparison is not in 64 bits."""

    print("== test_sliding_window_causal_mask_asm")

    def kernel(
        q_idx: UInt32, k_idx: UInt32, x: MutPointer[Float32, MutAnyOrigin]
    ):
        var mask = SlidingWindowCausalMask[8]()
        var vec = mask.mask(
            IndexList[4, element_type=.uint32](0, 0, Int(q_idx), Int(k_idx)),
            SIMD[.float32, 4](0),
        )
        if (
            mask.status(
                UInt32(0),
                Index[dtype=DType.uint32](q_idx, k_idx),
                Index[dtype=DType.uint32](64, 32),
            )
            == TileMaskStatus.PARTIAL_MASK
        ):
            x[0] = vec[3]

        x[0] = vec[2]

    var asm = _compile_code[kernel, target=get_gpu_target()]().asm
    print(asm)

    comptime if has_nvidia_gpu_accelerator():
        assert_true("setp.lt.u64" not in asm)
        assert_true("setp.lt.s64" not in asm)
    elif has_amd_gpu_accelerator():
        # there is nothing special about these instructions
        assert_true("s_cselect_b64" in asm)
        assert_true("v_cndmask_b32_e64" in asm)
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name(),
        ]()


def test_sliding_window_noncausal_mask() raises:
    print("test_sliding_window_noncausal_mask")

    comptime type = DType.float32
    comptime mask_val = Scalar[type](MASK_VALUE)

    comptime window = 4
    comptime mask = SlidingWindowNonCausalMask[window]()

    # Predicate: visible iff `k + window > q` (window=4). Future keys (k > q)
    # are always visible since there is no causal upper bound.

    # q=6, lanes k in {0,1,2,3}: only k=3 visible.
    var masked_vec = mask.mask(Index(0, 0, 6, 0), SIMD[type, 4](0, 1, 2, 3))
    assert_equal(masked_vec, SIMD[type, 4](mask_val, mask_val, mask_val, 3))

    # q=6, lanes k in {3,4,5,6}: all visible (k=6 is the diagonal).
    masked_vec = mask.mask(Index(0, 0, 6, 3), SIMD[type, 4](0, 1, 2, 3))
    assert_equal(masked_vec, SIMD[type, 4](0, 1, 2, 3))

    # q=6, lanes k in {5,6,7,8}: all visible, including FUTURE keys 7 and 8.
    masked_vec = mask.mask(Index(0, 0, 6, 5), SIMD[type, 4](0, 1, 2, 3))
    assert_equal(masked_vec, SIMD[type, 4](0, 1, 2, 3))

    # q=2 (small query): window lower bound is below 0, so all keys visible.
    masked_vec = mask.mask(Index(0, 0, 2, 0), SIMD[type, 4](0, 1, 2, 3))
    assert_equal(masked_vec, SIMD[type, 4](0, 1, 2, 3))

    @always_inline
    def check_status(
        offset: IndexList[2, ...],
        size: type_of(offset),
        expected: TileMaskStatus,
    ) raises:
        var status = mask.status(UInt32(0), offset, size)
        assert_equal(
            status,
            expected,
            msg=String(t"  {offset}, {size} > {status} (expected: {expected})"),
        )

    # FULL_MASK: tile entirely below the window band.
    check_status(Index(8, 0), Index(2, 2), TileMaskStatus.FULL_MASK)
    check_status(Index(6, 0), Index(2, 2), TileMaskStatus.FULL_MASK)

    # NO_MASK: tile fully visible, including future-key tiles (k > q).
    check_status(Index(0, 0), Index(4, 4), TileMaskStatus.NO_MASK)
    check_status(Index(2, 4), Index(2, 2), TileMaskStatus.NO_MASK)
    check_status(Index(6, 6), Index(2, 2), TileMaskStatus.NO_MASK)

    # PARTIAL_MASK: tile straddling the window lower edge.
    check_status(Index(4, 0), Index(4, 4), TileMaskStatus.PARTIAL_MASK)
    check_status(Index(6, 2), Index(2, 2), TileMaskStatus.PARTIAL_MASK)


def test_sliding_window_noncausal_mask_dispatch() raises:
    """The `"sliding_window_noncausal"` string (emitted by Python's
    `AttentionMaskVariant`) resolves through `dispatch_mask` to
    `SlidingWindowNonCausalMask`.
    """
    print("test_sliding_window_noncausal_mask_dispatch")

    var dispatched_name = String("")

    def capture[mask_t: MHAMask](mask: mask_t) raises {mut}:
        dispatched_name = mask_t.get_type_name()

    dispatch_mask[MaskName.SLIDING_WINDOW_NONCAUSAL.name, local_window_size=4](
        capture
    )
    assert_equal(dispatched_name, "SlidingWindowNonCausalMask")


def main() raises:
    test_causal_mask()
    test_causal_mask_asm()
    test_and_mask()
    test_sliding_window_causal_mask()
    test_sliding_window_causal_mask_asm()
    test_sliding_window_noncausal_mask()
    test_sliding_window_noncausal_mask_dispatch()
